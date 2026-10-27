"""
tools/fetch_tiles.py
====================

Builds an offline MBTiles pack for a launch site, so the map works with no
internet at all.

    python tools/fetch_tiles.py --bbox 74.80 32.68 74.92 32.78 \
        --zoom 12 18 --source "ISRO Bhuvan (boundaries only)" \
        --out assets/maps/jammu.mbtiles

BEFORE YOU RUN THIS
-------------------
Downloading many tiles from someone else's server is a cost to them, and
most free providers' terms say what you may and may not do. This tool
therefore refuses any source that is not explicitly marked
``"bulk_download": true`` in ``map_sources.json``, and says which provider
and why. The OpenStreetMap standard tile layer is refused on purpose: the
OSMF Tile Usage Policy forbids bulk downloading. For OSM data offline,
render your own tiles from an extract, or use a provider whose terms allow
packs.

It is also rate limited (2 requests/second by default, or the source's own
``rate_limit`` if that is lower), makes strictly one request at a time, is
resumable (tiles already in the file are skipped, so an interrupted run
continues), backs off when the server answers HTTP 429 or 503, sends a
User-Agent naming this application and the ``tile_contact`` from
map_settings.json, and refuses jobs over 20,000 tiles unless ``--force``. A
source's own ``max_tiles`` is a hard cap that ``--force`` does not lift.

See the README for building a pack with QGIS or Mobile Atlas Creator
instead, which is often the better route.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Iterator, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from map_window import (SETTINGS_FILE, TILE_CONTACT_FIELD,  # noqa: E402
                        TILE_PX, USER_AGENT, MapSource, default_user_agent,
                        load_sources, lonlat_to_tile, tile_bounds_3857)
from tile_fetcher import _is_image  # noqa: E402

#: Above this, a run is almost certainly a mistake -- a whole state at z16
#: rather than a launch site -- so it needs --force to proceed.
MAX_TILES_DEFAULT = 20000

#: Requests per second. Deliberately gentle; raise it only for a provider
#: whose terms say you may.
DEFAULT_RATE = 2.0

#: HTTP statuses that mean "slow down", not "this tile is missing".
BACKOFF_STATUSES = (429, 503)
#: First wait after a 429/503 that carries no Retry-After, seconds; doubled
#: on each consecutive one, up to BACKOFF_MAX_S.
BACKOFF_START_S = 30.0
BACKOFF_MAX_S = 600.0
#: Consecutive slow-down answers after which the run stops (resumable).
BACKOFF_GIVE_UP = 5


def tile_range(west: float, south: float, east: float, north: float,
               zoom: int) -> Tuple[int, int, int, int]:
    """Inclusive (x0, y0, x1, y1) tile range covering a bounding box."""
    x0, y0 = lonlat_to_tile(west, north, zoom)      # north -> smaller y
    x1, y1 = lonlat_to_tile(east, south, zoom)
    span = 1 << zoom
    x0, x1 = max(0, min(x0, x1)), min(span - 1, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(span - 1, max(y0, y1))
    return x0, y0, x1, y1


def count_tiles(west: float, south: float, east: float, north: float,
                zmin: int, zmax: int) -> int:
    total = 0
    for zoom in range(zmin, zmax + 1):
        x0, y0, x1, y1 = tile_range(west, south, east, north, zoom)
        total += (x1 - x0 + 1) * (y1 - y0 + 1)
    return total


def iter_tiles(west: float, south: float, east: float, north: float,
               zmin: int, zmax: int) -> Iterator[Tuple[int, int, int]]:
    for zoom in range(zmin, zmax + 1):
        x0, y0, x1, y1 = tile_range(west, south, east, north, zoom)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                yield zoom, x, y


def open_mbtiles(path: str, name: str, description: str,
                 bounds: Tuple[float, float, float, float],
                 zmin: int, zmax: int, fmt: str = "png",
                 attribution: str = "") -> sqlite3.Connection:
    """Create or reopen an MBTiles file, with the metadata the spec requires."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT, value TEXT)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tiles (zoom_level INTEGER, "
        "tile_column INTEGER, tile_row INTEGER, tile_data BLOB)")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS tile_index ON tiles "
        "(zoom_level, tile_column, tile_row)")
    meta = {
        "name": name,
        "type": "baselayer",
        "version": "1.0.0",
        "description": description,
        "format": fmt,
        "bounds": "%.6f,%.6f,%.6f,%.6f" % bounds,
        "minzoom": str(zmin),
        "maxzoom": str(zmax),
    }
    if attribution:
        meta["attribution"] = attribution
    for key, value in meta.items():
        conn.execute("DELETE FROM metadata WHERE name=?", (key,))
        conn.execute("INSERT INTO metadata (name, value) VALUES (?, ?)",
                     (key, value))
    conn.commit()
    return conn


def have_tile(conn: sqlite3.Connection, z: int, x: int, y: int) -> bool:
    """True when this tile is already stored (that is what makes it resumable)."""
    tms_y = (1 << z) - 1 - y
    row = conn.execute(
        "SELECT 1 FROM tiles WHERE zoom_level=? AND tile_column=? AND "
        "tile_row=? LIMIT 1", (z, x, tms_y)).fetchone()
    return row is not None


def store_tile(conn: sqlite3.Connection, z: int, x: int, y: int,
               data: bytes) -> None:
    """Store one tile, flipping slippy y to the TMS row MBTiles uses."""
    tms_y = (1 << z) - 1 - y
    conn.execute(
        "INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, "
        "tile_data) VALUES (?, ?, ?, ?)", (z, x, tms_y, sqlite3.Binary(data)))


def tile_url(source: MapSource, z: int, x: int, y: int) -> str:
    """The provider's URL for one tile. Mirrors the map window's builder."""
    if source.type == "xyz":
        url = (source.url.replace("{z}", str(z)).replace("{x}", str(x))
               .replace("{y}", str(y)))
        if source.key_env and source.api_key:
            sep = "&" if "?" in url else "?"
            url = "%s%s%s=%s" % (url, sep, source.key_param, source.api_key)
        return url
    if source.type == "wms":
        west, south, east, north = tile_bounds_3857(x, y, z)
        params = [
            ("service", "WMS"), ("version", "1.1.1"), ("request", "GetMap"),
            ("layers", source.layers), ("styles", ""), ("srs", "EPSG:3857"),
            ("bbox", "%.6f,%.6f,%.6f,%.6f" % (west, south, east, north)),
            ("width", str(TILE_PX)), ("height", str(TILE_PX)),
            ("format", "image/png"), ("transparent", "true"),
        ]
        if source.key_env and source.api_key:
            params.append((source.key_param, source.api_key))
        sep = "&" if "?" in source.url else "?"
        from urllib.parse import quote
        return source.url + sep + "&".join(
            "%s=%s" % (k, quote(str(v), safe=",:")) for k, v in params)
    raise ValueError("cannot download from source type %r" % source.type)


def tile_format(source: MapSource) -> str:
    """The MBTiles ``format`` for this source's tiles: "jpg" or "png"."""
    path = source.url.split("?", 1)[0].lower()
    if source.type == "xyz" and path.endswith((".jpg", ".jpeg")):
        return "jpg"
    return "png"


def read_tile_contact(root: str = ROOT) -> str:
    """The ``tile_contact`` field of map_settings.json, or ""."""
    try:
        with open(os.path.join(root, SETTINGS_FILE), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get(TILE_CONTACT_FIELD, "") or "").strip()


def fetch(url: str, timeout: float = 30.0,
          user_agent: str = USER_AGENT) -> Tuple[Optional[bytes], int, float]:
    """One tile: ``(data or None, HTTP status, Retry-After seconds or 0)``.

    Data comes back only when its bytes really are an image: a WMS error
    arrives as XML with HTTP 200, and a captive portal as HTML. Status 0
    means there was no HTTP answer at all (DNS, refused, timeout).
    """
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            ctype = response.headers.get("Content-Type", "")
            data = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        try:
            retry_after = float(exc.headers.get("Retry-After", "") or 0.0)
        except (TypeError, ValueError):
            retry_after = 0.0          # an HTTP-date; fall back to backoff
        return None, exc.code, retry_after
    except (urllib.error.URLError, OSError, ValueError):
        return None, 0, 0.0
    if not data or not _is_image(ctype, data):
        return None, status, 0.0
    return data, status, 0.0


def download(conn: sqlite3.Connection, source: MapSource,
             bbox: Tuple[float, float, float, float], zmin: int, zmax: int,
             rate: float, user_agent: str, total: int,
             fetcher: Callable = fetch, sleep: Callable = time.sleep,
             clock: Callable = time.time) -> Tuple[int, int, int, str]:
    """Fetch every missing tile, strictly one at a time.

    Returns ``(stored, skipped, failed, stopped)``; *stopped* is "" or why
    the run ended early. A 429 or 503 is not a missing tile: the same tile is
    retried after the server's Retry-After, or an exponential backoff, and
    the run stops after BACKOFF_GIVE_UP of them in a row. Everything stored
    is kept, so re-running the same command resumes.
    """
    west, south, east, north = bbox
    interval = 1.0 / rate if rate > 0 else 0.0
    stored = skipped = failed = 0
    stopped = ""
    next_at = clock()
    try:
        for index, (z, x, y) in enumerate(
                iter_tiles(west, south, east, north, zmin, zmax), 1):
            if have_tile(conn, z, x, y):
                skipped += 1
                continue
            url = tile_url(source, z, x, y)
            backoffs = 0
            while True:
                now = clock()
                if now < next_at:
                    sleep(next_at - now)
                next_at = clock() + interval
                data, status, retry_after = fetcher(url, 30.0, user_agent)
                if status not in BACKOFF_STATUSES:
                    break
                backoffs += 1
                if backoffs >= BACKOFF_GIVE_UP:
                    stopped = ("HTTP %d %d times in a row at tile %d/%d/%d"
                               % (status, backoffs, z, x, y))
                    break
                wait = retry_after or min(
                    BACKOFF_MAX_S, BACKOFF_START_S * 2 ** (backoffs - 1))
                print("  HTTP %d: the server asks us to slow down; waiting "
                      "%.0f s" % (status, wait))
                sleep(wait)
                next_at = clock() + interval
            if stopped:
                break
            if data:
                store_tile(conn, z, x, y, data)
                stored += 1
            else:
                failed += 1
            if stored and stored % 50 == 0:
                conn.commit()
                print("  %d/%d  stored %d  skipped %d  failed %d"
                      % (index, total, stored, skipped, failed))
    except KeyboardInterrupt:
        stopped = "interrupted"
    finally:
        conn.commit()
    return stored, skipped, failed, stopped


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an offline MBTiles pack for the ground station map.",
        epilog="Only sources marked bulk_download=true in map_sources.json "
               "may be downloaded in bulk. See the README.")
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                        help="bounding box in degrees")
    parser.add_argument("--zoom", nargs=2, type=int, required=True,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--source", required=True,
                        help="source name from map_sources.json")
    parser.add_argument("--out", required=True, help="output .mbtiles path")
    parser.add_argument("--name", default="",
                        help="pack name in the metadata (default: the source)")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE,
                        help="requests per second (default %.1f; never above "
                             "the source's own rate_limit)" % DEFAULT_RATE)
    parser.add_argument("--max-tiles", type=int, default=MAX_TILES_DEFAULT)
    parser.add_argument("--force", action="store_true",
                        help="proceed past --max-tiles (not past a source's "
                             "own hard cap)")
    parser.add_argument("--dry-run", action="store_true",
                        help="count tiles and check permissions, download nothing")
    args = parser.parse_args(argv)

    west, south, east, north = args.bbox
    if west > east or south > north:
        print("error: --bbox is WEST SOUTH EAST NORTH; west must be < east "
              "and south < north.", file=sys.stderr)
        return 2
    zmin, zmax = sorted(args.zoom)

    sources, note = load_sources(ROOT)
    if note:
        print("note: %s" % note)
    source = next((s for s in sources if s.name == args.source), None)
    if source is None:
        print("error: no source named %r. Available: %s"
              % (args.source, ", ".join(s.name for s in sources)),
              file=sys.stderr)
        return 2
    if source.type == "mbtiles":
        print("error: %r is already an offline pack; there is nothing to "
              "download." % source.name, file=sys.stderr)
        return 2
    if not source.bulk_download:
        print("REFUSED: %r is not marked bulk_download in map_sources.json.\n"
              "\n"
              "Downloading a tile pack is a sustained load on someone else's\n"
              "server, and most providers' terms govern it. In particular the\n"
              "OpenStreetMap Foundation's Tile Usage Policy forbids bulk\n"
              "downloading from tile.openstreetmap.org.\n"
              "\n"
              "To proceed you must (a) read that provider's terms yourself and\n"
              "(b) set \"bulk_download\": true for this source in\n"
              "map_sources.json. Alternatively build the pack in QGIS or\n"
              "Mobile Atlas Creator -- see the README." % source.name,
              file=sys.stderr)
        return 3
    if source.needs_key:
        print("error: %r needs an API key; set the %s environment variable."
              % (source.name, source.key_env), file=sys.stderr)
        return 2

    total = count_tiles(west, south, east, north, zmin, zmax)
    rate = args.rate
    if source.rate_limit > 0 and (rate <= 0 or rate > source.rate_limit):
        rate = source.rate_limit
    contact = read_tile_contact()
    user_agent = default_user_agent(contact)
    print("source     : %s" % source.name)
    print("bbox       : %.5f %.5f %.5f %.5f" % (west, south, east, north))
    print("zoom       : %d..%d" % (zmin, zmax))
    print("tiles      : %d" % total)
    print("rate limit : %.2f req/s, one request at a time  (about %.1f "
          "minutes at full download)" % (rate, total / max(rate, 0.01) / 60.0))
    print("user agent : %s" % user_agent)
    if not contact:
        print("note: %r in %s is empty. Fill it in so the provider can reach "
              "you; the project page is sent meanwhile."
              % (TILE_CONTACT_FIELD, SETTINGS_FILE))
    if source.max_tiles and total > source.max_tiles:
        print("\nREFUSED: %d tiles is over the hard cap of %d per job for %r. "
              "--force does not lift this cap: split the job into smaller "
              "boxes or zoom ranges." % (total, source.max_tiles, source.name),
              file=sys.stderr)
        return 3
    if total > args.max_tiles and not args.force:
        print("\nREFUSED: %d tiles is over the %d limit. Narrow the bbox or "
              "the zoom range, or pass --force if you are certain."
              % (total, args.max_tiles), file=sys.stderr)
        return 3
    if args.dry_run:
        print("\ndry run: nothing downloaded.")
        return 0

    conn = open_mbtiles(
        args.out, name=args.name or source.name,
        description="Offline pack for the CU Jammu Astro ground station. "
                    "Source: %s. %s" % (source.name, source.attribution),
        bounds=(west, south, east, north), zmin=zmin, zmax=zmax,
        fmt=tile_format(source), attribution=source.attribution)
    try:
        stored, skipped, failed, stopped = download(
            conn, source, (west, south, east, north), zmin, zmax, rate,
            user_agent, total)
    finally:
        conn.close()

    print("\nwrote %s" % args.out)
    print("stored %d, already present %d, failed %d"
          % (stored, skipped, failed))
    if stopped == "interrupted":
        print("interrupted; what was fetched is saved and the run is "
              "resumable -- re-run the same command.")
    elif stopped:
        print("stopped early: %s. What was fetched is saved; re-run the same "
              "command later to resume." % stopped)
    if failed:
        print("note: failed tiles are simply absent; re-run to retry them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

It is also rate limited (2 requests/second by default), resumable (tiles
already in the file are skipped, so an interrupted run continues), sends a
User-Agent naming this application, and refuses jobs over 20,000 tiles
unless ``--force``.

See the README for building a pack with QGIS or Mobile Atlas Creator
instead, which is often the better route.
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from typing import Iterator, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from map_window import (MapSource, TILE_PX, USER_AGENT, load_sources,  # noqa: E402
                        lonlat_to_tile, tile_bounds_3857)

#: Above this, a run is almost certainly a mistake -- a whole state at z16
#: rather than a launch site -- so it needs --force to proceed.
MAX_TILES_DEFAULT = 20000

#: Requests per second. Deliberately gentle; raise it only for a provider
#: whose terms say you may.
DEFAULT_RATE = 2.0


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
                 zmin: int, zmax: int, fmt: str = "png") -> sqlite3.Connection:
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


def fetch(url: str, timeout: float = 30.0) -> Optional[bytes]:
    """One tile, or None. A WMS error arrives as XML with HTTP 200."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            ctype = response.headers.get("Content-Type", "")
            data = response.read()
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not data or (ctype and not ctype.startswith("image")):
        return None
    return data


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
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE,
                        help="requests per second (default %.1f)" % DEFAULT_RATE)
    parser.add_argument("--max-tiles", type=int, default=MAX_TILES_DEFAULT)
    parser.add_argument("--force", action="store_true",
                        help="proceed past the tile-count limit")
    parser.add_argument("--dry-run", action="store_true",
                        help="count tiles and check permissions, download nothing")
    args = parser.parse_args(argv)

    west, south, east, north = args.bbox
    if west > east or south > north:
        print("error: --bbox is WEST SOUTH EAST NORTH; west must be < east "
              "and south < north.", file=sys.stderr)
        return 2
    zmin, zmax = sorted(args.zoom)

    sources, note = load_sources(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
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
    print("source     : %s" % source.name)
    print("bbox       : %.5f %.5f %.5f %.5f" % (west, south, east, north))
    print("zoom       : %d..%d" % (zmin, zmax))
    print("tiles      : %d" % total)
    print("rate limit : %.2f req/s  (about %.1f minutes at full download)"
          % (args.rate, total / max(args.rate, 0.01) / 60.0))
    if total > args.max_tiles and not args.force:
        print("\nREFUSED: %d tiles is over the %d limit. Narrow the bbox or "
              "the zoom range, or pass --force if you are certain."
              % (total, args.max_tiles), file=sys.stderr)
        return 3
    if args.dry_run:
        print("\ndry run: nothing downloaded.")
        return 0

    conn = open_mbtiles(
        args.out, name=source.name,
        description="Offline pack for the CU Jammu Astro ground station. "
                    "Source: %s. %s" % (source.name, source.attribution),
        bounds=(west, south, east, north), zmin=zmin, zmax=zmax)

    interval = 1.0 / args.rate if args.rate > 0 else 0.0
    stored = skipped = failed = 0
    next_at = time.time()
    try:
        for index, (z, x, y) in enumerate(
                iter_tiles(west, south, east, north, zmin, zmax), 1):
            if have_tile(conn, z, x, y):
                skipped += 1
                continue
            now = time.time()
            if now < next_at:
                time.sleep(next_at - now)
            next_at = time.time() + interval
            data = fetch(tile_url(source, z, x, y))
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
        print("\ninterrupted; what was fetched is saved and the run is "
              "resumable -- re-run the same command.")
    finally:
        conn.commit()
        conn.close()

    print("\nwrote %s" % args.out)
    print("stored %d, already present %d, failed %d"
          % (stored, skipped, failed))
    if failed:
        print("note: failed tiles are simply absent; re-run to retry them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

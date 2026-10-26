"""
tools/build_india_pack.py
=========================

Builds the bundled India overview pack from Natural Earth.

    python tools/build_india_pack.py --out assets/maps/india_overview.osmpack.json

Run once, by a maintainer, with a network connection. The result is
committed, so a fresh clone has something sensible on the map when zoomed
out without downloading anything.

SOURCE AND LICENCE
------------------
Natural Earth (naturalearthdata.com), via the nvkelso/natural-earth-vector
GeoJSON mirror. Natural Earth is in the **public domain**: "no permission
needed", no attribution required, though it is customary and this pack
credits it anyway.

WHAT IS KEPT
------------
India's national outline, state and union-territory boundary lines, and
populated places above a size threshold. Coordinates are simplified and
rounded, which is why a world-scale dataset ends up a few hundred
kilobytes: this is a context layer for a zoomed-out view, not a survey.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from map_basemap import PACK_FORMAT  # noqa: E402

BASE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "master/geojson/")
USER_AGENT = "CU-Jammu-Astro-GroundStation/1.0 (bundled overview pack build)"

#: India's bounding box, with a margin, used to clip the state lines file.
INDIA_BOX = (67.0, 6.0, 98.5, 36.5)

#: Points closer together than this (degrees) are dropped. At the zoom
#: levels this layer is for, finer detail is sub-pixel.
SIMPLIFY_DEG = 0.02

#: Minimum population for a place to be included.
MIN_POPULATION = 300_000


def fetch(name: str) -> Dict[str, Any]:
    url = BASE + name
    print("  fetching %s ..." % name, flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def simplify(points: Sequence[Sequence[float]],
             tolerance: float = SIMPLIFY_DEG) -> List[List[float]]:
    """Drop points closer than *tolerance* to the last kept one.

    A plain distance filter rather than Douglas-Peucker: at overview zooms
    the difference is invisible and this is one pass with no recursion.
    """
    out: List[List[float]] = []
    for point in points:
        try:
            lon, lat = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (math.isfinite(lon) and math.isfinite(lat)):
            continue
        if out:
            if (abs(lon - out[-1][0]) < tolerance
                    and abs(lat - out[-1][1]) < tolerance):
                continue
        out.append([round(lon, 4), round(lat, 4)])
    return out


def in_box(lon: float, lat: float, box: Tuple[float, ...] = INDIA_BOX) -> bool:
    return box[0] <= lon <= box[2] and box[1] <= lat <= box[3]


def rings_of(geometry: Dict[str, Any]) -> List[List[List[float]]]:
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "Polygon":
        return [coords[0]] if coords else []
    if kind == "MultiPolygon":
        return [poly[0] for poly in coords if poly]
    if kind == "LineString":
        return [coords]
    if kind == "MultiLineString":
        return list(coords)
    return []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--out",
                        default=os.path.join("assets", "maps",
                                             "india_overview.osmpack.json"))
    args = parser.parse_args(argv)

    areas: List[Dict[str, Any]] = []
    lines: List[Dict[str, Any]] = []
    places: List[Dict[str, Any]] = []

    try:
        countries = fetch("ne_110m_admin_0_countries.geojson")
        states = fetch("ne_10m_admin_1_states_provinces_lines.geojson")
        cities = fetch("ne_110m_populated_places_simple.geojson")
    except Exception as exc:                               # noqa: BLE001
        print("fetch failed: %s: %s" % (type(exc).__name__, exc),
              file=sys.stderr)
        return 1

    for feature in countries.get("features", ()):
        props = feature.get("properties") or {}
        name = (props.get("NAME") or props.get("name") or "")
        if name != "India":
            continue
        for ring in rings_of(feature.get("geometry") or {}):
            simple = simplify(ring)
            if len(simple) >= 3:
                # "bare" renders as a neutral landmass, which is what an
                # overview wants: somewhere to put the boundaries on.
                areas.append({"k": "bare", "c": simple})

    for feature in states.get("features", ()):
        props = feature.get("properties") or {}
        if (props.get("adm0_name") or props.get("ADM0_NAME") or "") != "India":
            continue
        for ring in rings_of(feature.get("geometry") or {}):
            simple = simplify(ring, SIMPLIFY_DEG / 2)
            if len(simple) >= 2:
                lines.append({"k": "boundary", "c": simple})

    for feature in cities.get("features", ()):
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        if not in_box(lon, lat):
            continue
        population = props.get("pop_max") or 0
        if population < MIN_POPULATION:
            continue
        name = props.get("name") or ""
        if not name:
            continue
        places.append({"n": name, "lon": round(lon, 4), "lat": round(lat, 4),
                       "r": 1 if population > 5_000_000 else 2})

    pack = {
        "format": PACK_FORMAT,
        "name": "India overview (Natural Earth)",
        "attribution": "Natural Earth — public domain",
        "licence": ("Public domain. Natural Earth imposes no restrictions: "
                    "https://www.naturalearthdata.com/about/terms-of-use/"),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bounds": {"west": INDIA_BOX[0], "south": INDIA_BOX[1],
                   "east": INDIA_BOX[2], "north": INDIA_BOX[3]},
        "lines": lines,
        "areas": areas,
        "places": places,
    }

    directory = os.path.dirname(os.path.abspath(args.out))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(pack, handle, separators=(",", ":"))

    print("\nwrote %s  (%.0f KB)"
          % (args.out, os.path.getsize(args.out) / 1024.0))
    print("  outline rings   : %d" % len(areas))
    print("  boundary lines  : %d" % len(lines))
    print("  places          : %d" % len(places))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

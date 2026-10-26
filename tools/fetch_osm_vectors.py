"""
tools/fetch_osm_vectors.py
==========================

Downloads an OpenStreetMap vector basemap for a launch site, once, at home,
so the map has real content with no network at the site and no tile server.

    python tools/fetch_osm_vectors.py \
        --bbox 74.8305 32.7041 74.8835 32.7491 \
        --out assets/maps/jammu.osmpack.json

Unlike a tile pack this is a single Overpass query returning geometry, which
the application renders itself. A 5 x 5 km site is typically a few hundred
kilobytes and takes one request.

BEING A GOOD CITIZEN OF OVERPASS
--------------------------------
Overpass is donated infrastructure with published usage limits. This tool
therefore makes exactly ONE request per run, refuses areas over
:data:`MAX_SIDE_KM` on a side, identifies itself with a User-Agent naming
this application and a contact, and prints the server's own rate-limit
status when it declines. If it refuses, wait rather than retrying in a loop.

The data is © OpenStreetMap contributors and licensed ODbL. The pack records
that and the map displays it whenever the pack is shown; keep the attribution
if you redistribute the pack.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from map_basemap import PACK_FORMAT  # noqa: E402

#: Overpass mirrors, tried in order. One request total, not one each.
ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

USER_AGENT = ("CU-Jammu-Astro-GroundStation/1.0 "
              "(offline launch-site basemap; "
              "+https://github.com/24beeca18ece-tech/Ground_Station)")

#: Refused above this. A launch site is a few kilometres; anything larger is
#: a region download, which is not what Overpass is for -- use a Geofabrik
#: extract and a converter instead.
MAX_SIDE_KM = 10.0

#: Overpass server-side timeout, seconds. Also sent in the query itself.
QUERY_TIMEOUT_S = 180

#: Coordinates are rounded to this many decimals in the saved pack. Six is
#: about 0.1 m, far finer than the geometry's own accuracy, and halves the
#: file size against full float repr.
COORD_DECIMALS = 6

#: Which OSM tags become which pack feature class.
AREA_TAGS = {
    "natural=water": "water", "waterway=riverbank": "water",
    "landuse=reservoir": "water", "natural=wetland": "water",
    "landuse=forest": "forest", "natural=wood": "forest",
    "natural=scrub": "scrub", "natural=heath": "scrub",
    "landuse=grass": "grass", "leisure=park": "grass",
    "landuse=meadow": "grass", "natural=grassland": "grass",
    "landuse=farmland": "farmland", "landuse=orchard": "farmland",
    "natural=bare_rock": "bare", "natural=sand": "bare",
    "natural=scree": "bare",
    "landuse=residential": "residential",
    "building": "building",
}


def bbox_side_km(west: float, south: float, east: float,
                 north: float) -> Tuple[float, float]:
    """(width, height) of the bounding box in kilometres."""
    mid_lat = math.radians((south + north) / 2.0)
    width = abs(east - west) * 111.320 * math.cos(mid_lat)
    height = abs(north - south) * 110.574
    return width, height


def build_query(west: float, south: float, east: float, north: float) -> str:
    """One Overpass QL query for everything the pack needs."""
    box = "%.6f,%.6f,%.6f,%.6f" % (south, west, north, east)
    return """[out:json][timeout:%d];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track|path|footway|cycleway)$"](%s);
  way["railway"~"^(rail|light_rail|narrow_gauge)$"](%s);
  way["natural"~"^(water|wetland|wood|scrub|heath|grassland|bare_rock|sand|scree)$"](%s);
  way["waterway"="riverbank"](%s);
  way["landuse"~"^(forest|reservoir|grass|meadow|farmland|orchard|residential)$"](%s);
  way["leisure"="park"](%s);
  way["building"](%s);
  node["place"~"^(city|town|village|hamlet|suburb|locality)$"](%s);
);
out body geom;
""" % (QUERY_TIMEOUT_S, box, box, box, box, box, box, box, box)


def classify(tags: Dict[str, str]) -> Tuple[str, str]:
    """(kind, class) for one OSM element: ('line'|'area'|'', class)."""
    if "highway" in tags:
        highway = tags["highway"]
        if highway in ("footway", "cycleway"):
            highway = "path"
        return "line", highway
    if "railway" in tags:
        return "line", "rail"
    for key in ("natural", "landuse", "leisure", "waterway"):
        if key in tags:
            name = AREA_TAGS.get("%s=%s" % (key, tags[key]))
            if name:
                return "area", name
    if "building" in tags:
        return "area", "building"
    return "", ""


PLACE_RANK = {"city": 1, "town": 2, "suburb": 3, "village": 4,
              "hamlet": 5, "locality": 6}


def convert(payload: Dict[str, Any], bounds: Dict[str, float],
            name: str) -> Dict[str, Any]:
    """Overpass JSON to the pack format the map renders."""
    lines: List[Dict[str, Any]] = []
    areas: List[Dict[str, Any]] = []
    places: List[Dict[str, Any]] = []

    def coords_of(element: Dict[str, Any]) -> List[List[float]]:
        return [[round(p["lon"], COORD_DECIMALS), round(p["lat"], COORD_DECIMALS)]
                for p in element.get("geometry", ()) if "lon" in p]

    for element in payload.get("elements", ()):
        tags = element.get("tags") or {}
        if element.get("type") == "node":
            place = tags.get("place")
            if place and tags.get("name"):
                places.append({"n": tags["name"],
                               "lon": round(element["lon"], COORD_DECIMALS),
                               "lat": round(element["lat"], COORD_DECIMALS),
                               "r": PLACE_RANK.get(place, 6)})
            continue
        if element.get("type") != "way":
            continue
        kind, feature_class = classify(tags)
        if not kind:
            continue
        points = coords_of(element)
        if kind == "area" and len(points) >= 3:
            areas.append({"k": feature_class, "c": points})
        elif kind == "line" and len(points) >= 2:
            lines.append({"k": feature_class, "c": points})

    return {
        "format": PACK_FORMAT,
        "name": name,
        "attribution": "© OpenStreetMap contributors, ODbL",
        "licence": "ODbL 1.0 — https://www.openstreetmap.org/copyright",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bounds": bounds,
        "lines": lines,
        "areas": areas,
        "places": places,
    }


def query_overpass(query: str) -> Optional[Dict[str, Any]]:
    """One request. Mirrors are a fallback for an outage, not a retry loop."""
    data = urllib.parse.urlencode({"data": query}).encode("ascii")
    last_error = ""
    for endpoint in ENDPOINTS:
        request = urllib.request.Request(
            endpoint, data=data,
            headers={"User-Agent": USER_AGENT,
                     "Content-Type": "application/x-www-form-urlencoded"})
        try:
            print("  querying %s ..." % endpoint, flush=True)
            with urllib.request.urlopen(request, timeout=QUERY_TIMEOUT_S + 30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:300]
            except Exception:                              # noqa: BLE001
                pass
            last_error = "HTTP %s: %s" % (exc.code, body.strip())
            if exc.code in (429, 504):
                print("    the server is rate limiting or overloaded. Wait a "
                      "few minutes; do not retry in a loop.", file=sys.stderr)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = "%s: %s" % (type(exc).__name__, exc)
        print("    failed: %s" % last_error, file=sys.stderr)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch an OpenStreetMap vector basemap for a launch site.",
        epilog="Data © OpenStreetMap contributors, ODbL. One request per run; "
               "areas over %g km a side are refused." % MAX_SIDE_KM)
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--out", required=True,
                        help="output path, e.g. assets/maps/site.osmpack.json")
    parser.add_argument("--name", default="", help="name shown in the dropdown")
    parser.add_argument("--dry-run", action="store_true",
                        help="check size and print the query, fetch nothing")
    args = parser.parse_args(argv)

    west, south, east, north = args.bbox
    if west >= east or south >= north:
        print("error: --bbox is WEST SOUTH EAST NORTH, west < east, "
              "south < north.", file=sys.stderr)
        return 2

    width_km, height_km = bbox_side_km(west, south, east, north)
    print("area       : %.2f x %.2f km" % (width_km, height_km))
    if width_km > MAX_SIDE_KM or height_km > MAX_SIDE_KM:
        print("\nREFUSED: %.1f x %.1f km is over the %g km limit.\n"
              "\n"
              "Overpass is donated infrastructure for extracting small areas,\n"
              "not for regional downloads. For something this large, take a\n"
              "Geofabrik extract and convert it to the same pack format, or\n"
              "narrow the bounding box to the launch site plus a margin."
              % (width_km, height_km, MAX_SIDE_KM), file=sys.stderr)
        return 3

    query = build_query(west, south, east, north)
    if args.dry_run:
        print("\n--- query (not sent) ---\n%s" % query)
        return 0

    payload = query_overpass(query)
    if payload is None:
        print("\nno data: every endpoint failed. If this is a rate limit, "
              "wait before trying again.", file=sys.stderr)
        return 1

    name = args.name or os.path.splitext(os.path.basename(args.out))[0]
    pack = convert(payload, {"west": west, "south": south,
                             "east": east, "north": north}, name)
    directory = os.path.dirname(os.path.abspath(args.out))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(pack, handle, separators=(",", ":"))

    size_kb = os.path.getsize(args.out) / 1024.0
    print("\nwrote %s  (%.0f KB)" % (args.out, size_kb))
    print("  roads/rail : %d" % len(pack["lines"]))
    print("  areas      : %d" % len(pack["areas"]))
    print("  places     : %d" % len(pack["places"]))
    print("  attribution: %s" % pack["attribution"])
    if not pack["lines"] and not pack["areas"]:
        print("\nnote: the query returned no mapped features for this box. "
              "Check the coordinates (west south east north, in degrees).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

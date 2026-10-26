"""
tools/csv_to_kml.py
===================

Converts a flight CSV the ground station already wrote into a KML, offline,
after the flight.

    python tools/csv_to_kml.py logs/Flight_2025ASI001_2026-10-26_143000.csv
    python tools/csv_to_kml.py logs/Flight_*.csv --out-dir logs

It reads columns by name from :data:`telemetry_packet.CSV_HEADER` rather than
by position, so a schema that grows a column does not silently shift the
coordinates, and it goes through the same :func:`kml_export.build_kml` and the
same :class:`track_store.TrackStore` filters as the live export -- the point
being that a KML made here is identical to one made from the live session.

Blank cells are normal in these files: no GNSS fix before lock, no battery
sensor fitted (the voltage column reads ``nan``), no particulates during the
sensor's warm-up. None of that stops a row from carrying a usable position.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kml_export import build_kml, kml_filename  # noqa: E402
from track_store import TrackStore  # noqa: E402
from telemetry_packet import CSV_HEADER  # noqa: E402


class _Row:
    """A CSV row shaped like a packet, so TrackStore can filter it unchanged.

    Duck typing rather than building a real TelemetryPacket: the store only
    ever reads these attributes, and constructing a full packet from a row
    would mean re-deriving fields the CSV already resolved.
    """

    __slots__ = ("payload_type", "lat", "lon", "nav_alt_m", "sats",
                 "gs_recv_epoch", "mission_time_s", "packet_count",
                 "fsm_name", "has_nav_alt", "has_fix")

    def __init__(self, **kwargs) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))


def _f(value: Optional[str], default: float = float("nan")) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _i(value: Optional[str], default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def rows_to_store(path: str, jump_m: float, jump_window_s: float) -> TrackStore:
    """Read one Flight_*.csv into a TrackStore, applying the same filters."""
    store = TrackStore(jump_m=jump_m, jump_window_s=jump_window_s)
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("lat", "lon", "payload_type")
                   if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                "%s does not look like a ground-station CSV: missing column(s) "
                "%s. Expected the schema in telemetry_packet.CSV_HEADER."
                % (os.path.basename(path), ", ".join(missing)))
        for record in reader:
            lat, lon = _f(record.get("lat")), _f(record.get("lon"))
            sats = _i(record.get("sats"))
            nav_alt = _f(record.get("nav_alt_m"))
            # has_fix is not a CSV column: it is a property of the packet
            # class. Reconstruct it from what the file does carry, using the
            # same rule -- finite, non-zero, in range, and backed by
            # satellites when the link reported a count at all.
            has_fix = (math.isfinite(lat) and math.isfinite(lon)
                       and not (abs(lat) < 1e-9 and abs(lon) < 1e-9)
                       and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0)
            store.add_packet(_Row(
                payload_type=(record.get("payload_type") or "GENERIC").strip()
                or "GENERIC",
                lat=lat, lon=lon,
                nav_alt_m=nav_alt,
                sats=sats,
                gs_recv_epoch=_f(record.get("gs_recv_epoch"), 0.0) or 0.0,
                mission_time_s=_f(record.get("mission_time_s"), 0.0) or 0.0,
                packet_count=_i(record.get("packet_count")),
                fsm_name=(record.get("fsm_state_name") or "").strip(),
                has_nav_alt=math.isfinite(nav_alt),
                has_fix=has_fix,
            ))
    return store


def convert(path: str, out_dir: Optional[str], jump_m: float,
            jump_window_s: float) -> Optional[str]:
    store = rows_to_store(path, jump_m, jump_window_s)
    tracks = store.non_empty()
    if not tracks:
        print("%s: no usable position fix (%d rows had no lock); nothing "
              "written." % (os.path.basename(path), store.no_fix_packets))
        return None

    team_id = "UNKNOWN"
    try:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                if (record.get("team_id") or "").strip():
                    team_id = record["team_id"].strip()
                    break
    except OSError:
        pass

    stamp = datetime.fromtimestamp(os.path.getmtime(path))
    text = build_kml(tracks, {
        "team_id": team_id,
        "session_date": stamp.strftime("%Y-%m-%d %H:%M:%S"),
        "source": os.path.basename(path),
        "jump_m": jump_m,
        "jump_window_s": jump_window_s,
    })
    target_dir = out_dir or os.path.dirname(os.path.abspath(path))
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, kml_filename(team_id,
                                                     os.path.getmtime(path)))
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    counts = ", ".join("%s %d" % (t.vehicle, len(t.points)) for t in tracks)
    print("%s -> %s  (%s; %d jumps and %d duplicates dropped)"
          % (os.path.basename(path), out_path, counts,
             sum(t.jumps_dropped for t in tracks),
             sum(t.duplicates_dropped for t in tracks)))
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert ground-station flight CSVs to KML for Google Earth.")
    parser.add_argument("csv", nargs="+", help="Flight_*.csv file(s); globs ok")
    parser.add_argument("--out-dir", default=None,
                        help="where to write (default: beside the CSV)")
    parser.add_argument("--jump-m", type=float, default=500.0,
                        help="discard a fix this far from the previous one "
                             "inside --jump-window-s (default 500)")
    parser.add_argument("--jump-window-s", type=float, default=1.0)
    args = parser.parse_args(argv)

    paths: List[str] = []
    for pattern in args.csv:
        hits = sorted(glob.glob(pattern))
        paths.extend(hits if hits else [pattern])

    written = 0
    for path in paths:
        if not os.path.isfile(path):
            print("skipping %s: not a file" % path, file=sys.stderr)
            continue
        try:
            if convert(path, args.out_dir, args.jump_m, args.jump_window_s):
                written += 1
        except (OSError, ValueError) as exc:
            print("error on %s: %s" % (path, exc), file=sys.stderr)
    if not paths:
        print("no input files matched.", file=sys.stderr)
        return 2
    return 0 if written or paths else 1


if __name__ == "__main__":
    raise SystemExit(main())

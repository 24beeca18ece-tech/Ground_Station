"""
tools/test_map_and_kml.py
=========================

Tests for the live map and the KML export.

Run from the repository root::

    python -m pytest tools/test_map_and_kml.py -v

Qt runs offscreen, so these need no display; the fixture sets
``QT_QPA_PLATFORM`` before PyQt5 is imported by the map module.
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import map_window as mw  # noqa: E402
from kml_export import build_kml, kml_filename  # noqa: E402
from track_store import (Track, TrackPoint, TrackStore, bearing_deg,  # noqa: E402
                         decimate, haversine_m, is_usable_fix)

KML_NS = {"k": "http://www.opengis.net/kml/2.2",
          "gx": "http://www.google.com/kml/ext/2.2"}

# A 1x1 transparent PNG, used to fill a synthetic MBTiles file.
PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6360000002000100ffff0300000600"
    "0557bfabd40000000049454e44ae426082")


class FakePacket:
    """Shaped like a TelemetryPacket, for the fields the store reads."""

    def __init__(self, lat=32.72, lon=74.85, alt=300.0, count=1,
                 payload="CANSAT", has_fix=True, has_nav_alt=True,
                 epoch=None, sats=9, fsm="ASCENT"):
        self.lat = lat
        self.lon = lon
        self.nav_alt_m = alt
        self.packet_count = count
        self.payload_type = payload
        self.has_fix = has_fix
        self.has_nav_alt = has_nav_alt
        self.gs_recv_epoch = 1_700_000_000.0 if epoch is None else epoch
        self.mission_time_s = float(count) * 0.05
        self.sats = sats
        self.fsm_name = fsm


# ---------------------------------------------------------------------------
# Web Mercator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lat,lon", [
    (32.7266, 74.8570), (0.0, 0.0), (-33.8600, 151.2100),
    (60.1699, 24.9384), (-54.8019, -68.3030), (85.0, 179.9), (-85.0, -179.9),
])
@pytest.mark.parametrize("zoom", [1, 5, 13, 19])
def test_lonlat_world_round_trip(lat, lon, zoom):
    x, y = mw.lonlat_to_world(lon, lat, zoom)
    lon2, lat2 = mw.world_to_lonlat(x, y, zoom)
    assert lon2 == pytest.approx(lon, abs=1e-6)
    assert lat2 == pytest.approx(lat, abs=1e-6)


def test_world_pixel_is_inside_the_world_at_every_zoom():
    for zoom in range(1, 20):
        size = mw.TILE_PX * (2 ** zoom)
        for lat, lon in ((85.0, 179.99), (-85.0, -179.99), (0.0, 0.0)):
            x, y = mw.lonlat_to_world(lon, lat, zoom)
            assert 0.0 <= x <= size
            assert 0.0 <= y <= size


def test_known_tile_for_jammu():
    """Cross-checked against an independent calculation of the same tile."""
    assert mw.lonlat_to_tile(74.857, 32.7266, 13) == (5799, 3307)


def test_tile_bounds_are_3857_metres_and_tile_sized():
    west, south, east, north = mw.tile_bounds_3857(5799, 3307, 13)
    assert west == pytest.approx(8331024.59, abs=0.05)
    assert north == pytest.approx(3859764.18, abs=0.05)
    span = mw.EQUATOR_M / (2 ** 13)
    assert east - west == pytest.approx(span)
    assert north - south == pytest.approx(span)


def test_zoom_zero_tile_covers_the_whole_mercator_extent():
    west, south, east, north = mw.tile_bounds_3857(0, 0, 0)
    assert west == pytest.approx(-mw.ORIGIN_SHIFT)
    assert east == pytest.approx(mw.ORIGIN_SHIFT)
    assert north == pytest.approx(mw.ORIGIN_SHIFT)
    assert south == pytest.approx(-mw.ORIGIN_SHIFT)


def test_metres_per_pixel_halves_each_zoom():
    a = mw.metres_per_pixel(32.7266, 13)
    b = mw.metres_per_pixel(32.7266, 14)
    assert a / b == pytest.approx(2.0)
    assert a == pytest.approx(16.08, abs=0.05)


def test_clamp_lat_bounds_the_poles():
    assert mw.clamp_lat(90.0) == pytest.approx(85.05112878)
    assert mw.clamp_lat(-90.0) == pytest.approx(-85.05112878)


# ---------------------------------------------------------------------------
# MBTiles
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_mbtiles(tmp_path):
    """A one-tile MBTiles at z3, stored in TMS row order like a real one."""
    path = str(tmp_path / "tiny.mbtiles")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE metadata (name TEXT, value TEXT)")
    conn.execute("CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER,"
                 " tile_row INTEGER, tile_data BLOB)")
    conn.executemany("INSERT INTO metadata VALUES (?, ?)",
                     [("name", "tiny"), ("format", "png"), ("minzoom", "3"),
                      ("maxzoom", "3")])
    # Slippy y = 2 for z3 means TMS row 2**3 - 1 - 2 = 5.
    conn.execute("INSERT INTO tiles VALUES (3, 5, 5, ?)",
                 (sqlite3.Binary(PNG_1PX),))
    conn.commit()
    conn.close()
    return path


def test_mbtiles_read_applies_the_tms_y_flip(tiny_mbtiles):
    """Slippy y=2 must find TMS row 5; y=5 must find nothing."""
    assert mw.read_mbtiles_tile(tiny_mbtiles, 3, 5, 2) == PNG_1PX
    assert mw.read_mbtiles_tile(tiny_mbtiles, 3, 5, 5) is None


def test_mbtiles_metadata_reads_back(tiny_mbtiles):
    meta = mw.read_mbtiles_metadata(tiny_mbtiles)
    assert meta["name"] == "tiny"
    assert meta["format"] == "png"


def test_missing_mbtiles_file_returns_none_not_an_exception(tmp_path):
    missing = str(tmp_path / "nope.mbtiles")
    assert mw.read_mbtiles_tile(missing, 3, 5, 2) is None
    assert mw.read_mbtiles_metadata(missing) == {}


def test_corrupt_mbtiles_file_does_not_raise(tmp_path):
    bad = tmp_path / "bad.mbtiles"
    bad.write_bytes(b"this is definitely not a sqlite database")
    assert mw.read_mbtiles_tile(str(bad), 3, 5, 2) is None
    assert mw.read_mbtiles_metadata(str(bad)) == {}


def test_mbtiles_without_a_tiles_table_does_not_raise(tmp_path):
    path = tmp_path / "empty.mbtiles"
    sqlite3.connect(str(path)).close()
    assert mw.read_mbtiles_tile(str(path), 3, 5, 2) is None


def test_find_mbtiles_handles_a_missing_directory(tmp_path):
    assert mw.find_mbtiles(str(tmp_path / "absent")) == []


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def test_missing_sources_file_falls_back_and_says_so(tmp_path):
    sources, note = mw.load_sources(str(tmp_path))
    assert [s.name for s in sources] == [s["name"] for s in mw.BUILTIN_SOURCES]
    assert "not found" in note


def test_bad_sources_file_falls_back_and_says_so(tmp_path):
    (tmp_path / mw.SOURCES_FILE).write_text("{ this is not json", encoding="utf-8")
    sources, note = mw.load_sources(str(tmp_path))
    assert sources and "could not be read" in note


def test_sources_file_is_honoured(tmp_path):
    (tmp_path / mw.SOURCES_FILE).write_text(
        '{"sources": [{"name": "X", "type": "xyz", '
        '"url": "https://e/{z}/{x}/{y}.png", "attribution": "A"}]}',
        encoding="utf-8")
    sources, note = mw.load_sources(str(tmp_path))
    assert [s.name for s in sources] == ["X"]
    assert note == ""


def test_entries_with_an_unknown_type_are_skipped(tmp_path):
    (tmp_path / mw.SOURCES_FILE).write_text(
        '{"sources": [{"name": "bad", "type": "magic"}, '
        '{"name": "ok", "type": "xyz", "url": "u"}]}', encoding="utf-8")
    sources, _ = mw.load_sources(str(tmp_path))
    assert [s.name for s in sources] == ["ok"]


def test_api_key_comes_from_the_environment_never_the_file(monkeypatch):
    source = mw.MapSource(name="k", type="xyz", url="u", key_env="TEST_MAP_KEY")
    monkeypatch.delenv("TEST_MAP_KEY", raising=False)
    assert source.needs_key is True and source.api_key == ""
    monkeypatch.setenv("TEST_MAP_KEY", "secret")
    assert source.needs_key is False and source.api_key == "secret"


def test_shipped_example_sources_file_parses_and_has_no_google():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import json
    with open(os.path.join(root, "map_sources.example.json"),
              encoding="utf-8") as handle:
        data = json.load(handle)
    # The file names Google only to say it is NOT used as a tile source.
    # What must hold is that no Google host appears in any URL.
    urls = [s.get("url", "") for s in data["sources"]]
    urls += [s.get("url", "") for s in data.get("_disabled_examples", [])]
    assert not any("google" in u.lower() for u in urls)
    names = [s["name"] for s in data["sources"]]
    assert "Offline (MBTiles)" in names


# ---------------------------------------------------------------------------
# Fix filtering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("packet,reason", [
    (FakePacket(has_fix=False), "has_fix False"),
    (FakePacket(lat=0.0, lon=0.0), "exactly 0,0"),
    (FakePacket(lat=float("nan")), "NaN latitude"),
    (FakePacket(lon=float("inf")), "infinite longitude"),
    (FakePacket(lat=95.0), "latitude out of range"),
])
def test_unusable_fixes_are_rejected(packet, reason):
    assert is_usable_fix(packet) is False, reason
    store = TrackStore()
    assert store.add_packet(packet) is False
    assert store.total_points == 0


def test_a_real_fix_is_accepted():
    assert is_usable_fix(FakePacket()) is True


def test_cansat_and_rocket_go_to_separate_tracks():
    store = TrackStore()
    store.add_packet(FakePacket(payload="CANSAT", count=1))
    store.add_packet(FakePacket(payload="ROCKET", count=1, lat=32.80))
    assert sorted(store.tracks) == ["CANSAT", "ROCKET"]
    assert len(store.get("CANSAT")) == 1
    assert len(store.get("ROCKET")) == 1


def test_duplicate_packet_counts_are_dropped():
    store = TrackStore()
    store.add_packet(FakePacket(count=7))
    store.add_packet(FakePacket(count=7, lat=32.73))
    track = store.get("CANSAT")
    assert len(track) == 1 and track.duplicates_dropped == 1


def test_a_gps_jump_is_dropped_and_counted():
    store = TrackStore()
    store.add_packet(FakePacket(count=1, lat=32.7200, epoch=1000.0))
    # ~5 km away, 0.5 s later: 36000 km/h, a receiver glitch.
    store.add_packet(FakePacket(count=2, lat=32.7650, epoch=1000.5))
    store.add_packet(FakePacket(count=3, lat=32.7201, epoch=1001.0))
    track = store.get("CANSAT")
    assert track.jumps_dropped == 1
    assert len(track) == 2


def test_a_long_gap_is_not_treated_as_a_jump():
    """Real signal loss moves a long way legitimately."""
    store = TrackStore()
    store.add_packet(FakePacket(count=1, lat=32.72, epoch=1000.0))
    store.add_packet(FakePacket(count=2, lat=32.80, epoch=1060.0))
    assert store.get("CANSAT").jumps_dropped == 0
    assert len(store.get("CANSAT")) == 2


def test_clear_forgets_every_track():
    store = TrackStore()
    store.add_packet(FakePacket())
    store.clear()
    assert store.total_points == 0 and store.non_empty() == []


def test_apogee_and_bounds():
    store = TrackStore()
    for i, alt in enumerate((100.0, 900.0, 400.0), start=1):
        store.add_packet(FakePacket(count=i, alt=alt, lat=32.72 + i * 0.001))
    track = store.get("CANSAT")
    assert track.apogee.alt_m == pytest.approx(900.0)
    assert track.first.alt_m == pytest.approx(100.0)
    assert track.last.alt_m == pytest.approx(400.0)
    min_lat, min_lon, max_lat, max_lon = track.bounds()
    assert min_lat < max_lat and min_lon == pytest.approx(max_lon)


def test_distance_and_bearing_are_sane():
    # One degree of latitude is about 111 km; due north is 0 degrees.
    assert haversine_m(32.0, 74.0, 33.0, 74.0) == pytest.approx(111195, rel=0.01)
    assert bearing_deg(32.0, 74.0, 33.0, 74.0) == pytest.approx(0.0, abs=0.01)
    assert bearing_deg(32.0, 74.0, 32.0, 75.0) == pytest.approx(90.0, abs=0.5)


def test_decimate_keeps_the_endpoints():
    points = [TrackPoint(32.0 + i * 1e-4, 74.0, 0.0, float(i), 0.0, i, 9, "X")
              for i in range(1000)]
    thinned = decimate(points, 50)
    assert len(thinned) == 50
    assert thinned[0] is points[0]
    assert thinned[-1] is points[-1]


def test_decimate_leaves_short_tracks_alone():
    points = [TrackPoint(32.0, 74.0, 0.0, 0.0, 0.0, 1, 9, "X")]
    assert decimate(points, 50) == points


# ---------------------------------------------------------------------------
# KML
# ---------------------------------------------------------------------------

def make_track(vehicle="CANSAT", n=5, with_alt=True, start_epoch=1_700_000_000.0):
    points = [
        TrackPoint(lat=32.72 + i * 1e-4, lon=74.85 + i * 1e-4,
                   alt_m=(300.0 + i * 50.0) if with_alt else float("nan"),
                   epoch=start_epoch + i, mission_time_s=i * 1.0,
                   packet_count=i + 1, sats=11, fsm_state="ASCENT",
                   has_alt=with_alt)
        for i in range(n)]
    return Track.from_points(vehicle, points)


def test_kml_parses_as_xml():
    root = ET.fromstring(build_kml([make_track()], {"team_id": "T"}))
    assert root.tag.endswith("kml")


def test_kml_coordinate_order_is_lon_lat_alt():
    # Three different numbers, so a transposition cannot pass by luck.
    # (A Track stores columns; .points builds values on demand, so the
    # coordinates have to be given at construction, not patched after.)
    point = TrackPoint(lat=11.5, lon=77.25, alt_m=123.0, epoch=1_700_000_000.0,
                       mission_time_s=0.0, packet_count=1, sats=11,
                       fsm_state="ASCENT", has_alt=True)
    track = Track.from_points("CANSAT", [point])
    root = ET.fromstring(build_kml([track], {"team_id": "T"}))
    line = root.find(".//k:LineString/k:coordinates", KML_NS).text.strip()
    lon, lat, alt = (float(v) for v in line.split(","))
    assert (lon, lat, alt) == pytest.approx((77.25, 11.5, 123.0))


def test_gx_track_pairs_every_when_with_a_coord():
    root = ET.fromstring(build_kml([make_track(n=9)], {"team_id": "T"}))
    gx_track = root.find(".//gx:Track", KML_NS)
    assert len(gx_track.findall("k:when", KML_NS)) == 9
    assert len(gx_track.findall("gx:coord", KML_NS)) == 9


def test_gx_track_coords_are_space_separated():
    """gx:coord uses spaces; <coordinates> uses commas. Mixing them is silent."""
    root = ET.fromstring(build_kml([make_track(n=2)], {"team_id": "T"}))
    coord = root.find(".//gx:coord", KML_NS).text
    assert "," not in coord and len(coord.split()) == 3


def test_extended_data_arrays_match_the_point_count():
    root = ET.fromstring(build_kml([make_track(n=6)], {"team_id": "T"}))
    for name in ("fsm_state", "sats", "packet_count"):
        array = root.find(".//gx:SimpleArrayData[@name='%s']" % name, KML_NS)
        assert len(array.findall("gx:value", KML_NS)) == 6


def test_xml_special_characters_are_escaped():
    track = make_track(vehicle='Can&Sat <"x">')
    text = build_kml([track], {"team_id": 'T&<">'})
    ET.fromstring(text)                      # must still parse
    assert "&amp;" in text and "&lt;" in text and "&gt;" in text
    assert 'Can&Sat <"x">' not in text       # never raw


def test_timestamps_are_utc_iso8601_with_z():
    root = ET.fromstring(build_kml([make_track(n=2)], {"team_id": "T"}))
    when = root.find(".//gx:Track/k:when", KML_NS).text
    assert when.endswith("Z") and "T" in when
    assert len(when) == 20


def test_no_nav_alt_gives_clamp_to_ground_and_says_so():
    root = ET.fromstring(build_kml([make_track(with_alt=False)],
                                   {"team_id": "T"}))
    modes = {el.text for el in root.findall(".//k:altitudeMode", KML_NS)}
    assert modes == {"clampToGround"}
    description = root.find(".//k:Document/k:description", KML_NS).text
    assert "clamped to the ground" in description


def test_nav_alt_present_gives_absolute():
    root = ET.fromstring(build_kml([make_track(with_alt=True)],
                                   {"team_id": "T"}))
    modes = {el.text for el in root.findall(".//k:altitudeMode", KML_NS)}
    assert modes == {"absolute"}


def test_jump_count_is_reported_in_the_description():
    track = make_track()
    track.jumps_dropped = 3
    track.duplicates_dropped = 2
    root = ET.fromstring(build_kml([track], {"team_id": "T"}))
    description = root.find(".//k:Document/k:description", KML_NS).text
    assert "3 GPS jumps" in description and "2 duplicate" in description


def test_jump_filter_drops_a_synthetic_5km_glitch_and_the_kml_reports_it():
    store = TrackStore()
    for i in range(5):
        store.add_packet(FakePacket(count=i + 1, lat=32.72 + i * 1e-4,
                                    epoch=1000.0 + i))
    store.add_packet(FakePacket(count=99, lat=32.7650, epoch=1004.5))  # 5 km
    track = store.get("CANSAT")
    assert track.jumps_dropped == 1 and len(track) == 5
    root = ET.fromstring(build_kml([track], {"team_id": "T"}))
    assert "1 GPS jumps" in root.find(".//k:Document/k:description",
                                      KML_NS).text


def test_two_vehicles_make_two_folders_with_different_styles():
    text = build_kml([make_track("CANSAT"), make_track("ROCKET")],
                     {"team_id": "T"})
    root = ET.fromstring(text)
    names = [f.find("k:name", KML_NS).text
             for f in root.findall(".//k:Folder", KML_NS)]
    assert names == ["CANSAT", "ROCKET"]
    style_refs = {el.text for el in root.findall(".//k:styleUrl", KML_NS)}
    assert "#track-cansat" in style_refs and "#track-rocket" in style_refs


def test_launch_apogee_and_last_placemarks_exist():
    track = make_track(n=7)
    track.points[3].alt_m = 5000.0            # unmistakable apogee
    root = ET.fromstring(build_kml([track], {"team_id": "T"}))
    names = [p.find("k:name", KML_NS).text
             for p in root.findall(".//k:Placemark", KML_NS)]
    assert any(n.startswith("Launch") for n in names)
    assert any(n.startswith("Apogee") for n in names)
    assert any(n.startswith("Last known") for n in names)


def test_empty_track_list_still_produces_a_valid_warning_document():
    root = ET.fromstring(build_kml([], {"team_id": "T"}))
    assert root.findall(".//k:Folder", KML_NS) == []
    assert "no usable position fix" in root.find(
        ".//k:Document/k:description", KML_NS).text


def test_kml_filename_matches_the_csv_pattern():
    name = kml_filename("2025ASI001", 1_700_000_000.0)
    assert name.startswith("Flight_2025ASI001_") and name.endswith(".kml")


# ---------------------------------------------------------------------------
# csv_to_kml
# ---------------------------------------------------------------------------

def test_csv_to_kml_on_a_ground_station_csv(tmp_path):
    """Round trip through the CSV the ground station itself writes."""
    import csv as csv_mod
    from telemetry_packet import CSV_HEADER, build_frame, parse_frame
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import csv_to_kml

    path = tmp_path / "Flight_2025ASI001_2026-10-26_120000.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv_mod.writer(handle)
        writer.writerow(CSV_HEADER)
        for i in range(1, 11):
            body = ("2025ASI001,CANSAT,%.2f,%d,235.94,985.23,27.40,nan,"
                    "09:05:0%d,%.6f,%.6f,%.1f,11,0.12,-0.05,9.79,0.40,"
                    "-1.20,0.06,1,4.10,6.20,7.00,7.40,,"
                    % (i * 0.05, i, i % 10, 32.72 + i * 1e-4,
                       74.85 + i * 1e-4, 300.0 + i * 10))
            row = parse_frame(build_frame(body)).to_csv_row()
            row[0] = i                       # seq, stamped by the logger
            writer.writerow(row)

    out = csv_to_kml.convert(str(path), str(tmp_path), 500.0, 1.0)
    assert out and os.path.exists(out)
    root = ET.fromstring(open(out, encoding="utf-8").read())
    assert len(root.find(".//gx:Track", KML_NS).findall("gx:coord", KML_NS)) == 10
    assert "Flight_2025ASI001" in os.path.basename(out)


def test_csv_to_kml_tolerates_blank_cells_and_no_fix_rows(tmp_path):
    import csv as csv_mod
    from telemetry_packet import CSV_HEADER
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import csv_to_kml

    path = tmp_path / "blank.csv"
    index = {name: i for i, name in enumerate(CSV_HEADER)}
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv_mod.writer(handle)
        writer.writerow(CSV_HEADER)
        for lat, lon in (("", ""), ("0", "0"), ("32.72", "74.85")):
            row = [""] * len(CSV_HEADER)
            row[index["payload_type"]] = "CANSAT"
            row[index["lat"]] = lat
            row[index["lon"]] = lon
            row[index["voltage_v"]] = "nan"
            row[index["packet_count"]] = "1"
            writer.writerow(row)
    store = csv_to_kml.rows_to_store(str(path), 500.0, 1.0)
    assert store.total_points == 1           # only the real coordinate
    assert store.no_fix_packets == 2


def test_csv_to_kml_rejects_a_file_that_is_not_a_flight_csv(tmp_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import csv_to_kml
    path = tmp_path / "other.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not look like"):
        csv_to_kml.rows_to_store(str(path), 500.0, 1.0)


# ---------------------------------------------------------------------------
# fetch_tiles
# ---------------------------------------------------------------------------

def test_fetch_tiles_counts_a_launch_site_pack():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fetch_tiles
    # The 5 x 5 km example documented in the README.
    total = fetch_tiles.count_tiles(74.8305, 32.7041, 74.8835, 32.7491, 12, 18)
    assert 1500 < total < 3000


def test_fetch_tiles_refuses_a_source_without_bulk_permission(capsys):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fetch_tiles
    code = fetch_tiles.main(["--bbox", "74.8", "32.7", "74.9", "32.8",
                             "--zoom", "12", "14", "--source", "OpenStreetMap",
                             "--out", "unused.mbtiles", "--dry-run"])
    assert code == 3
    assert "REFUSED" in capsys.readouterr().err


def test_fetch_tiles_refuses_an_oversized_job(capsys, tmp_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fetch_tiles
    (tmp_path / "map_sources.json").write_text(
        '{"sources": [{"name": "Y", "type": "xyz", "url": "https://e/{z}/{x}/{y}",'
        ' "bulk_download": true}]}', encoding="utf-8")
    original = fetch_tiles.load_sources
    fetch_tiles.load_sources = lambda _base: original(str(tmp_path))
    try:
        code = fetch_tiles.main(["--bbox", "70", "20", "80", "30", "--zoom",
                                 "12", "16", "--source", "Y", "--out",
                                 str(tmp_path / "o.mbtiles"), "--dry-run"])
    finally:
        fetch_tiles.load_sources = original
    assert code == 3
    assert "over the" in capsys.readouterr().err


def test_fetch_tiles_writes_a_valid_mbtiles_with_the_y_flip(tmp_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fetch_tiles
    path = str(tmp_path / "out.mbtiles")
    conn = fetch_tiles.open_mbtiles(path, "t", "d", (74.8, 32.7, 74.9, 32.8),
                                    3, 3)
    assert fetch_tiles.have_tile(conn, 3, 5, 2) is False
    fetch_tiles.store_tile(conn, 3, 5, 2, PNG_1PX)
    conn.commit()
    assert fetch_tiles.have_tile(conn, 3, 5, 2) is True   # resumable
    conn.close()
    # The map's reader must find exactly what the writer stored.
    assert mw.read_mbtiles_tile(path, 3, 5, 2) == PNG_1PX
    meta = mw.read_mbtiles_metadata(path)
    assert meta["format"] == "png" and meta["minzoom"] == "3"
    assert "bounds" in meta and "name" in meta


def test_fetch_tiles_rejects_a_backwards_bbox(capsys):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fetch_tiles
    code = fetch_tiles.main(["--bbox", "80", "40", "70", "30", "--zoom",
                             "12", "13", "--source", "OpenStreetMap",
                             "--out", "x.mbtiles", "--dry-run"])
    assert code == 2


# ---------------------------------------------------------------------------
# The window itself
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_map_window_opens_renders_and_closes_with_no_tiles(qapp, tmp_path):
    """Fully offline, no tile files at all: it must still paint a frame."""
    from PyQt5.QtCore import QThreadPool
    from PyQt5.QtGui import QImage

    store = TrackStore()
    for i in range(1, 60):
        store.add_packet(FakePacket(count=i, lat=32.72 + i * 2e-4,
                                    lon=74.85 + i * 2e-4, alt=300.0 + i * 20,
                                    epoch=time.time()))
        store.add_packet(FakePacket(count=i, payload="ROCKET",
                                    lat=32.73 + i * 2e-4, lon=74.86 + i * 2e-4,
                                    alt=500.0 + i * 40, epoch=time.time()))

    before = QThreadPool.globalInstance().activeThreadCount()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    window.show()
    qapp.processEvents()
    # Size first: fit_all() frames the tracks for the canvas it is given, so
    # fitting before the resize would put the whole path off the image.
    window.canvas.resize(900, 600)
    window.canvas.ground_station = (32.7266, 74.8570)
    window.canvas.fit_all()

    image = QImage(900, 600, QImage.Format_ARGB32)
    image.fill(0)
    window.canvas.render(image)

    assert window.canvas.paint_errors == 0, window.canvas._last_paint_error
    # The tracks, vehicles, scale bar and attribution are all drawn over the
    # grey "no tile" squares, so a correct frame has several colours.
    colours = {image.pixel(x, y) for x in range(0, 900, 7)
               for y in range(0, 600, 7)}
    assert len(colours) > 3, "only %d colour(s): nothing was drawn" % len(colours)

    window.shutdown()
    window.close()
    qapp.processEvents()
    assert QThreadPool.globalInstance().activeThreadCount() <= before


def test_map_window_survives_an_empty_store(qapp, tmp_path):
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.show()
    qapp.processEvents()
    from PyQt5.QtGui import QImage
    image = QImage(400, 300, QImage.Format_ARGB32)
    window.canvas.resize(400, 300)
    window.canvas.render(image)
    assert window.canvas.paint_errors == 0
    window.shutdown()
    window.close()


def test_zoom_around_the_cursor_keeps_that_point_fixed(qapp, tmp_path):
    from PyQt5.QtCore import QPoint
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    canvas = window.canvas
    canvas.resize(800, 600)
    anchor = QPoint(200, 150)
    before = canvas._pixel_to_lonlat(anchor)
    canvas.set_zoom(canvas.zoom + 2, anchor)
    after = canvas._pixel_to_lonlat(anchor)
    assert after[0] == pytest.approx(before[0], abs=1e-4)
    assert after[1] == pytest.approx(before[1], abs=1e-4)
    window.shutdown()
    window.close()


def test_zoom_is_clamped_to_the_supported_range(qapp, tmp_path):
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.canvas.set_zoom(99)
    assert window.canvas.zoom == mw.MAX_ZOOM
    window.canvas.set_zoom(-5)
    assert window.canvas.zoom == mw.MIN_ZOOM
    window.shutdown()
    window.close()


def test_ground_station_settings_round_trip(qapp, tmp_path):
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.gs_lat.setValue(32.726600)
    window.gs_lon.setValue(74.857000)
    window.gs_check.setChecked(True)
    assert os.path.exists(os.path.join(str(tmp_path), mw.SETTINGS_FILE))
    window.shutdown()
    window.close()

    again = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    assert again.gs_lat.value() == pytest.approx(32.7266)
    assert again.canvas.ground_station is not None
    again.shutdown()
    again.close()


def test_no_ground_station_file_shows_nothing_and_does_not_raise(qapp, tmp_path):
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    assert window.canvas.ground_station is None
    window.shutdown()
    window.close()


def test_wms_tile_url_is_3857_and_tile_sized(qapp, tmp_path):
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    source = mw.MapSource(name="w", type="wms", url="https://host/wms",
                          layers="a:b")
    window.canvas.provider.set_source(source, str(tmp_path))
    url = window.canvas.provider._tile_url(("w", 13, 5799, 3307))
    assert "EPSG%3A3857" in url or "EPSG:3857" in url
    assert "width=256" in url and "height=256" in url
    assert "8331024" in url
    window.shutdown()
    window.close()


def test_xyz_tile_url_substitutes_z_x_y(qapp, tmp_path):
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    source = mw.MapSource(name="x", type="xyz",
                          url="https://host/{z}/{x}/{y}.png")
    window.canvas.provider.set_source(source, str(tmp_path))
    assert window.canvas.provider._tile_url(("x", 4, 5, 6)) == \
        "https://host/4/5/6.png"
    window.shutdown()
    window.close()


def test_offline_source_finds_a_pack(qapp, tmp_path, tiny_mbtiles):
    import shutil
    maps = tmp_path / "assets" / "maps"
    maps.mkdir(parents=True)
    shutil.copy(tiny_mbtiles, str(maps / "pack.mbtiles"))
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    source = mw.MapSource(name="o", type="mbtiles", path="assets/maps")
    window.canvas.provider.set_source(source, str(tmp_path))
    assert window.canvas.provider.offline_ready is True
    window.shutdown()
    window.close()

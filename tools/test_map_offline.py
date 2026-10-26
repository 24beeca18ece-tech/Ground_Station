"""
tools/test_map_offline.py
=========================

Tests for the map working with no network and no tile files: the grid base
layer, vector packs, image overlays and the pack dropdown.

    python -m pytest tools/test_map_offline.py -v
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import map_basemap as mb  # noqa: E402
import map_window as mw  # noqa: E402
from track_store import Track, TrackStore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture()
def maps_dir(tmp_path):
    directory = tmp_path / "assets" / "maps"
    directory.mkdir(parents=True)
    return directory


def tiny_pack() -> dict:
    """A hand-made pack: one road, one lake, one place."""
    return {
        "format": mb.PACK_FORMAT,
        "name": "tiny",
        "attribution": "© OpenStreetMap contributors, ODbL",
        "bounds": {"west": 74.80, "south": 32.70, "east": 74.90, "north": 32.78},
        "lines": [{"k": "primary",
                   "c": [[74.82, 32.72], [74.85, 32.74], [74.88, 32.76]]}],
        "areas": [{"k": "water",
                   "c": [[74.83, 32.73], [74.84, 32.73], [74.84, 32.74],
                         [74.83, 32.74]]}],
        "places": [{"n": "Jammu", "lon": 74.857, "lat": 32.7266, "r": 2}],
    }


# ---------------------------------------------------------------------------
# Graticule
# ---------------------------------------------------------------------------

def test_graticule_step_gets_finer_as_the_view_narrows():
    wide = mb.choose_graticule_step(10.0, 800)
    narrow = mb.choose_graticule_step(0.01, 800)
    assert narrow < wide


def test_graticule_lines_are_never_closer_than_the_minimum():
    for span in (0.001, 0.05, 1.0, 40.0, 180.0):
        step = mb.choose_graticule_step(span, 900)
        assert step / span * 900 >= mb.MIN_GRATICULE_PX * 0.999


def test_degrees_and_minutes_formatting():
    assert mb.format_lat(32.0) == "32°N"
    assert mb.format_lat(-32.5) == "32°30.0'S"
    assert mb.format_lon(74.857) .startswith("74°51")
    assert mb.format_lon(-0.25) == "0°15.0'W"


def test_graticule_handles_a_degenerate_view_without_raising(qapp):
    from PyQt5.QtGui import QImage, QPainter
    from PyQt5.QtCore import QPointF
    image = QImage(100, 100, QImage.Format_ARGB32)
    painter = QPainter(image)
    mb.draw_graticule(painter, 100, 100, (float("nan"), 1.0), (2.0, 0.0),
                      lambda lon, lat: QPointF(0, 0))
    painter.end()


# ---------------------------------------------------------------------------
# Range rings
# ---------------------------------------------------------------------------

def test_range_rings_draw_and_skip_the_unreadable_ones(qapp):
    from PyQt5.QtGui import QImage, QPainter
    from PyQt5.QtCore import QPointF
    image = QImage(400, 400, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    # 1 m per pixel: the 250 m ring is 250 px, the 2 km ring is off screen
    # but must not raise.
    mb.draw_range_rings(painter, 32.7, 74.8,
                        lambda lon, lat: QPointF(200, 200), 1.0)
    painter.end()
    # Sample the whole image: a 250 px ring centred 200 px from the edge
    # never crosses the middle row.
    assert any(image.pixel(x, y) != 0
               for x in range(0, 400, 3) for y in range(0, 400, 3))


def test_range_rings_do_nothing_with_a_nonsense_scale(qapp):
    from PyQt5.QtGui import QImage, QPainter
    from PyQt5.QtCore import QPointF
    image = QImage(50, 50, QImage.Format_ARGB32)
    painter = QPainter(image)
    for scale in (0.0, -1.0, float("nan")):
        mb.draw_range_rings(painter, 32.7, 74.8,
                            lambda lon, lat: QPointF(25, 25), scale)
    painter.end()


# ---------------------------------------------------------------------------
# Pack discovery and selection
# ---------------------------------------------------------------------------

def test_grid_is_always_available_even_with_no_maps_directory(tmp_path):
    packs = mw.find_packs(str(tmp_path))
    assert len(packs) == 1 and packs[0].kind == "grid"


def test_every_pack_kind_is_discovered(maps_dir, tmp_path):
    (maps_dir / "site.osmpack.json").write_text(json.dumps(tiny_pack()),
                                                encoding="utf-8")
    (maps_dir / "tiles.mbtiles").write_bytes(b"not really, discovery only")
    (maps_dir / "photo.json").write_text(
        json.dumps({"west": 74.8, "south": 32.7, "east": 74.9, "north": 32.8}),
        encoding="utf-8")
    (maps_dir / "photo.png").write_bytes(b"")
    kinds = {p.kind for p in mw.find_packs(str(tmp_path))}
    assert kinds == {"grid", "vector", "mbtiles", "image"}


def test_a_json_without_a_matching_image_is_not_offered(maps_dir, tmp_path):
    (maps_dir / "orphan.json").write_text("{}", encoding="utf-8")
    assert [p.kind for p in mw.find_packs(str(tmp_path))] == ["grid"]


def test_the_default_pack_makes_no_network_request(qapp, tmp_path):
    """Opening the map at the launch site must not phone anywhere."""
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.show()
    qapp.processEvents()
    assert window.canvas.pack.kind == "grid"
    assert window.canvas.provider._inflight == 0
    assert window.canvas.provider._pending == {}
    window.shutdown()
    window.close()


def test_a_vector_pack_is_preferred_over_the_bare_grid(qapp, maps_dir, tmp_path):
    (maps_dir / "site.osmpack.json").write_text(json.dumps(tiny_pack()),
                                                encoding="utf-8")
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.show()
    qapp.processEvents()
    assert window.canvas.pack.kind == "vector"
    assert window.canvas.provider._inflight == 0
    window.shutdown()
    window.close()


# ---------------------------------------------------------------------------
# Vector packs
# ---------------------------------------------------------------------------

def test_a_tiny_pack_loads_with_its_features(qapp, maps_dir):
    path = str(maps_dir / "t.osmpack.json")
    open(path, "w", encoding="utf-8").write(json.dumps(tiny_pack()))
    pack = mb.load_vector_pack(path)
    assert pack is not None
    assert len(pack.lines) == 1 and len(pack.areas) == 1
    assert pack.places[0][0] == "Jammu"
    assert "ODbL" in pack.attribution


@pytest.mark.parametrize("content", ["", "{", "[]", "null", "not json at all"])
def test_a_broken_pack_returns_none_rather_than_raising(qapp, maps_dir, content):
    path = str(maps_dir / "bad.osmpack.json")
    open(path, "w", encoding="utf-8").write(content)
    assert mb.load_vector_pack(path) is None


def test_a_missing_pack_file_returns_none(tmp_path):
    assert mb.load_vector_pack(str(tmp_path / "nope.json")) is None


def test_degenerate_features_are_skipped(qapp, maps_dir):
    data = tiny_pack()
    data["lines"].append({"k": "primary", "c": [[74.8, 32.7]]})   # 1 point
    data["areas"].append({"k": "water", "c": [[74.8, 32.7], [74.81, 32.7]]})
    data["places"].append({"n": "bad"})                            # no coords
    path = str(maps_dir / "d.osmpack.json")
    open(path, "w", encoding="utf-8").write(json.dumps(data))
    pack = mb.load_vector_pack(path)
    assert len(pack.lines) == 1 and len(pack.areas) == 1
    assert len(pack.places) == 1


def test_a_broken_pack_falls_back_to_the_grid_and_logs(qapp, maps_dir, tmp_path):
    (maps_dir / "bad.osmpack.json").write_text("{{{", encoding="utf-8")
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    messages = []
    window.log_message.connect(messages.append)
    window.show()
    qapp.processEvents()
    index = next(i for i, p in enumerate(window.packs) if p.kind == "vector")
    # The broken pack is already selected at construction, before the test
    # could connect; move away and back so the signal fires with a listener.
    window.pack_combo.setCurrentIndex(0)
    qapp.processEvents()
    messages.clear()
    window.pack_combo.setCurrentIndex(index)
    qapp.processEvents()
    assert window.canvas.pack.kind == "grid"
    assert any("grid" in m for m in messages), messages
    window.shutdown()
    window.close()


def test_a_vector_pack_renders_without_error(qapp, maps_dir, tmp_path):
    from PyQt5.QtGui import QImage
    (maps_dir / "site.osmpack.json").write_text(json.dumps(tiny_pack()),
                                                encoding="utf-8")
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.resize(700, 500)
    window.show()
    qapp.processEvents()
    window.canvas.zoom = 13
    window.canvas.center_lat, window.canvas.center_lon = 32.74, 74.85
    image = QImage(window.canvas.width(), window.canvas.height(),
                   QImage.Format_ARGB32)
    window.canvas.render(image)
    assert window.canvas.paint_errors == 0, window.canvas._last_paint_error
    colours = {image.pixel(x, y)
               for x in range(0, window.canvas.width(), 5)
               for y in range(0, window.canvas.height(), 5)}
    assert len(colours) > 3
    window.shutdown()
    window.close()


def test_vector_projection_is_cached_per_zoom(qapp, maps_dir, tmp_path):
    (maps_dir / "site.osmpack.json").write_text(json.dumps(tiny_pack()),
                                                encoding="utf-8")
    pack = mb.load_vector_pack(str(maps_dir / "site.osmpack.json"))
    first = pack._project(13, mw.lonlat_to_world_array)
    again = pack._project(13, mw.lonlat_to_world_array)
    assert again is first
    other = pack._project(14, mw.lonlat_to_world_array)
    assert other is not first


def test_the_bundled_india_pack_is_present_and_loads():
    path = os.path.join(ROOT, "assets", "maps",
                        "india_overview.osmpack.json")
    assert os.path.isfile(path), "the bundled overview pack is missing"
    assert os.path.getsize(path) < 600 * 1024, "the bundled pack got large"
    pack = mb.load_vector_pack(path)
    assert pack is not None
    assert pack.feature_count > 100
    assert "public domain" in pack.attribution.lower()


# ---------------------------------------------------------------------------
# Image overlays
# ---------------------------------------------------------------------------

def write_image(path: str, size: int = 64) -> None:
    from PyQt5.QtGui import QImage, QColor
    image = QImage(size, size, QImage.Format_RGB32)
    image.fill(QColor("#884422"))
    image.save(path)


def test_an_image_overlay_loads_with_its_bounds(qapp, maps_dir):
    write_image(str(maps_dir / "site.png"))
    (maps_dir / "site.json").write_text(json.dumps(
        {"west": 74.80, "south": 32.70, "east": 74.90, "north": 32.78}),
        encoding="utf-8")
    overlay = mb.load_image_overlay(str(maps_dir / "site.json"))
    assert overlay is not None and overlay.is_valid
    assert overlay.west == pytest.approx(74.80)
    assert overlay.path.endswith(".png")


def test_an_overlay_with_no_image_beside_it_is_rejected(qapp, maps_dir):
    (maps_dir / "lonely.json").write_text(json.dumps(
        {"west": 74.8, "south": 32.7, "east": 74.9, "north": 32.8}),
        encoding="utf-8")
    assert mb.load_image_overlay(str(maps_dir / "lonely.json")) is None


@pytest.mark.parametrize("bounds", [
    {"west": 74.9, "south": 32.7, "east": 74.8, "north": 32.8},   # reversed
    {"west": 74.8, "south": 32.8, "east": 74.9, "north": 32.7},
    {"west": "x", "south": 32.7, "east": 74.9, "north": 32.8},    # not a number
    {},                                                            # no bounds
])
def test_bad_overlay_bounds_are_rejected(qapp, maps_dir, bounds):
    write_image(str(maps_dir / "b.png"))
    (maps_dir / "b.json").write_text(json.dumps(bounds), encoding="utf-8")
    assert mb.load_image_overlay(str(maps_dir / "b.json")) is None


def test_an_overlay_is_decoded_off_the_gui_thread(qapp, maps_dir, tmp_path):
    import time
    write_image(str(maps_dir / "site.png"), size=256)
    (maps_dir / "site.json").write_text(json.dumps(
        {"west": 74.80, "south": 32.70, "east": 74.90, "north": 32.78}),
        encoding="utf-8")
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.show()
    qapp.processEvents()
    index = next(i for i, p in enumerate(window.packs) if p.kind == "image")
    window.pack_combo.setCurrentIndex(index)
    assert window.canvas.overlay is not None
    deadline = time.time() + 5
    while time.time() < deadline and window.canvas.overlay.image is None:
        qapp.processEvents()
        time.sleep(0.02)
    assert window.canvas.overlay.image is not None
    assert window.canvas.overlay.error == ""
    window.shutdown()
    window.close()


def test_a_huge_overlay_is_downsampled_on_load(qapp, maps_dir):
    from PyQt5.QtGui import QImage, QColor
    big = QImage(5000, 300, QImage.Format_RGB32)
    big.fill(QColor("#224466"))
    big.save(str(maps_dir / "big.png"))
    job = mw._ImageLoadJob(str(maps_dir / "big.png"))
    got = {}
    job.signals.done.connect(lambda p, i, e: got.update(image=i, error=e))
    job.run()
    assert got["error"] == ""
    assert max(got["image"].width(), got["image"].height()) <= mw._ImageLoadJob.MAX_EDGE


def test_an_undecodable_image_reports_an_error_and_does_not_raise(qapp, maps_dir):
    (maps_dir / "junk.png").write_bytes(b"this is not a PNG")
    job = mw._ImageLoadJob(str(maps_dir / "junk.png"))
    got = {}
    job.signals.done.connect(lambda p, i, e: got.update(image=i, error=e))
    job.run()
    assert got["image"] is None and got["error"]


def test_overlay_opacity_slider_only_shows_for_an_image(qapp, maps_dir, tmp_path):
    write_image(str(maps_dir / "s.png"))
    (maps_dir / "s.json").write_text(json.dumps(
        {"west": 74.8, "south": 32.7, "east": 74.9, "north": 32.8}),
        encoding="utf-8")
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.show()
    qapp.processEvents()
    window.pack_combo.setCurrentIndex(0)                  # grid
    qapp.processEvents()
    assert window.opacity.isVisible() is False
    index = next(i for i, p in enumerate(window.packs) if p.kind == "image")
    window.pack_combo.setCurrentIndex(index)
    qapp.processEvents()
    assert window.canvas.overlay is not None
    window.opacity.setValue(40)
    assert window.canvas.overlay_opacity == pytest.approx(0.40)
    window.shutdown()
    window.close()


# ---------------------------------------------------------------------------
# Remembering the choice
# ---------------------------------------------------------------------------

def test_the_chosen_pack_is_remembered(qapp, maps_dir, tmp_path):
    (maps_dir / "site.osmpack.json").write_text(json.dumps(tiny_pack()),
                                                encoding="utf-8")
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.show()
    qapp.processEvents()
    window.pack_combo.setCurrentIndex(0)                  # choose the grid
    qapp.processEvents()
    window.shutdown()
    window.close()

    again = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    again.show()
    qapp.processEvents()
    assert again.canvas.pack.kind == "grid", "the choice was not remembered"
    again.shutdown()
    again.close()


def test_a_remembered_pack_that_vanished_falls_back(qapp, maps_dir, tmp_path):
    settings = tmp_path / mw.SETTINGS_FILE
    settings.write_text(json.dumps({"map_pack": "deleted (vector)"}),
                        encoding="utf-8")
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.show()
    qapp.processEvents()
    assert window.canvas.pack.kind == "grid"
    window.shutdown()
    window.close()


# ---------------------------------------------------------------------------
# fetch_osm_vectors
# ---------------------------------------------------------------------------

def test_overpass_tool_refuses_too_large_an_area(capsys):
    import fetch_osm_vectors as fov
    code = fov.main(["--bbox", "74.0", "32.0", "75.0", "33.0",
                     "--out", "unused.json", "--dry-run"])
    assert code == 3
    assert "REFUSED" in capsys.readouterr().err


def test_overpass_tool_accepts_a_launch_site(capsys):
    import fetch_osm_vectors as fov
    code = fov.main(["--bbox", "74.8305", "32.7041", "74.8835", "32.7491",
                     "--out", "unused.json", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "highway" in out and "out body geom" in out


def test_overpass_tool_rejects_a_backwards_bbox():
    import fetch_osm_vectors as fov
    assert fov.main(["--bbox", "75", "33", "74", "32", "--out", "x",
                     "--dry-run"]) == 2


def test_overpass_conversion_classifies_and_rounds():
    import fetch_osm_vectors as fov
    payload = {"elements": [
        {"type": "way", "tags": {"highway": "primary"},
         "geometry": [{"lon": 74.812345678, "lat": 32.712345678},
                      {"lon": 74.82, "lat": 32.72}]},
        {"type": "way", "tags": {"natural": "water"},
         "geometry": [{"lon": 74.83, "lat": 32.73}, {"lon": 74.84, "lat": 32.73},
                      {"lon": 74.84, "lat": 32.74}]},
        {"type": "way", "tags": {"highway": "footway"},
         "geometry": [{"lon": 74.85, "lat": 32.75}, {"lon": 74.86, "lat": 32.76}]},
        {"type": "node", "tags": {"place": "town", "name": "Somewhere"},
         "lon": 74.857, "lat": 32.7266},
        {"type": "way", "tags": {"amenity": "cafe"},
         "geometry": [{"lon": 74.8, "lat": 32.7}]},        # not a basemap class
    ]}
    pack = fov.convert(payload, {"west": 74.8, "south": 32.7,
                                 "east": 74.9, "north": 32.8}, "t")
    assert [line["k"] for line in pack["lines"]] == ["primary", "path"]
    assert [area["k"] for area in pack["areas"]] == ["water"]
    assert pack["places"][0]["n"] == "Somewhere"
    assert pack["lines"][0]["c"][0] == [74.812346, 32.712346]
    assert "ODbL" in pack["attribution"]


def test_a_generated_pack_is_loadable_by_the_map(tmp_path):
    import fetch_osm_vectors as fov
    payload = {"elements": [
        {"type": "way", "tags": {"highway": "residential"},
         "geometry": [{"lon": 74.81, "lat": 32.71}, {"lon": 74.82, "lat": 32.72}]},
    ]}
    pack = fov.convert(payload, {"west": 74.8, "south": 32.7,
                                 "east": 74.9, "north": 32.8}, "roundtrip")
    path = tmp_path / "rt.osmpack.json"
    path.write_text(json.dumps(pack), encoding="utf-8")
    loaded = mb.load_vector_pack(str(path))
    assert loaded is not None and len(loaded.lines) == 1

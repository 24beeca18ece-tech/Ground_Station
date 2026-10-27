"""
tools/test_map_zoom.py
======================

The map never goes blank on a tile pack: zoom is limited to what the data
supports (pack min zoom - 2 .. max zoom + 3), a missing tile falls back
through every parent level before it is grey, outside the pack the grid is
drawn with the pack's outline and an "outside map pack" status, and GO TO
PAD returns to the launch pad. None of it touches the network.

    python -m pytest tools/test_map_zoom.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import map_window as mw  # noqa: E402
from track_store import TrackStore  # noqa: E402
from test_sat_pack import (PAD_LAT, PAD_LON, close_enough, grab,  # noqa: E402,F401
                           jpeg_tile, make_pack, no_network, pack_dir,
                           pad_tile, pixel, qapp, settled, wait_for)
from test_live_tracking import PAD_BOX_10KM  # noqa: E402


class Wheel:
    """The two things MapCanvas.wheelEvent reads, and accept()."""

    def __init__(self, steps, pos):
        from PyQt5.QtCore import QPoint
        self._delta = QPoint(0, 120 * steps)
        self._pos = pos

    def angleDelta(self):                       # noqa: N802 - Qt naming
        return self._delta

    def pos(self):
        return self._pos

    def accept(self):
        pass


def share_close_to(image, rgb, tol):
    """Fraction of a 10 x 6 grid of sample points close to *rgb*. Markers,
    rings and grid lines cover a few points; a blank or grey map covers all."""
    points = [(int(image.width() * (i + 0.5) / 10),
               int(image.height() * (j + 0.5) / 6))
              for i in range(10) for j in range(6)]
    return sum(close_enough(pixel(image, x, y), rgb, tol=tol)
               for x, y in points) / len(points)


def pad_window(qapp, tmp_path, tiles=None, zmin=8, zmax=14):
    maps = pack_dir(tmp_path)
    if tiles is None:
        x, y = pad_tile(13)
        tiles = {(13, x, y): jpeg_tile("#2e6b30")}
    make_pack(os.path.join(maps, "launchpad_s2.mbtiles"), tiles, zmin=zmin,
              zmax=zmax, bounds=PAD_BOX_10KM)
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    return window, settled(qapp, window)


# ---------------------------------------------------------------------------
# Zoom limits
# ---------------------------------------------------------------------------

def test_zoom_range_follows_the_pack(qapp, tmp_path, no_network):
    window, canvas = pad_window(qapp, tmp_path)
    assert canvas.zoom_range == (6, 17)          # 8 - 2 .. 14 + 3
    canvas.set_zoom(25)
    assert canvas.zoom == 17
    canvas.set_zoom(0)
    assert canvas.zoom == 6
    window.shutdown()
    window.close()


def test_the_wheel_stops_at_the_limits_and_says_so(qapp, tmp_path, no_network):
    from PyQt5.QtCore import QPoint
    window, canvas = pad_window(qapp, tmp_path)
    centre = QPoint(canvas.width() // 2, canvas.height() // 2)
    for _ in range(20):
        canvas.wheelEvent(Wheel(+1, centre))
    assert canvas.zoom == 17
    window._update_status()
    assert "max zoom for this map" in window.status.text()
    assert "overzoomed x8 from z14" in window.status.text()
    for _ in range(30):
        canvas.wheelEvent(Wheel(-1, centre))
    assert canvas.zoom == 6
    window._update_status()
    assert "min zoom for this map" in window.status.text()
    canvas.set_zoom(12)
    window._update_status()
    assert "zoom for this map" not in window.status.text()
    window.shutdown()
    window.close()


def test_fit_all_and_go_to_pad_respect_the_limits(qapp, tmp_path, no_network):
    window, canvas = pad_window(qapp, tmp_path, zmax=11)       # range 6..14
    canvas.ground_station = (PAD_LAT, PAD_LON)                 # fit -> z15
    canvas.fit_all()
    assert canvas.zoom == 14
    canvas.go_to_pad()                                         # wants z13
    assert canvas.zoom == 13
    window.shutdown()
    window.close()


def test_changing_pack_brings_the_zoom_into_range(qapp, tmp_path, no_network):
    window, canvas = pad_window(qapp, tmp_path)
    window.pack_combo.setCurrentIndex(0)                       # grid
    assert canvas.zoom_range == (mw.MIN_ZOOM, mw.MAX_ZOOM)
    canvas.set_zoom(19)
    assert canvas.zoom == 19
    index = next(i for i, p in enumerate(window.packs) if p.kind == "mbtiles")
    window.pack_combo.setCurrentIndex(index)
    assert canvas.zoom == 17
    window.shutdown()
    window.close()


def test_maximum_zoom_never_leaves_a_blank_map(qapp, tmp_path, no_network):
    """The reported bug: z18 at the pad drew nothing but "no tile"."""
    x, y = pad_tile(14)
    tiles = {(14, x + dx, y + dy): jpeg_tile("#2e6b30")
             for dx in range(-3, 4) for dy in range(-3, 4)}
    window, canvas = pad_window(qapp, tmp_path, tiles=tiles)
    canvas.center_lat, canvas.center_lon = PAD_LAT, PAD_LON
    canvas.set_zoom(19)
    assert canvas.zoom == 17
    for _ in range(6):
        grab(canvas)
        wait_for(qapp, lambda: False, frames=20)
    image = grab(canvas)
    assert canvas.provider.tiles_drawn > 0 and canvas.paint_errors == 0
    assert share_close_to(image, (0x2e, 0x6b, 0x30), tol=40) >= 0.85
    window.shutdown()
    window.close()


# ---------------------------------------------------------------------------
# Parent fallback through every level
# ---------------------------------------------------------------------------

def _fallback(qapp, provider, z, x, y, frames=400):
    """overzoomed() walks one parent level per read; drive it to an answer."""
    import time
    for _ in range(frames):
        image = provider.overzoomed(z, x, y)
        if image is not None:
            return image
        if not provider._pending:
            # Nothing being read: ask once more, then the answer is final.
            image = provider.overzoomed(z, x, y)
            if image is not None or not provider._pending:
                return image
        qapp.processEvents()
        time.sleep(0.003)
    return None


@pytest.mark.parametrize("z", range(9, 18))
def test_parent_fallback_reaches_the_only_tile_from_every_level(qapp, tmp_path,
                                                                no_network, z):
    """Only z8 exists. Every deeper level, up to the max zoom of 17, finds it
    -- 1 to 9 levels up -- rather than drawing grey."""
    from PyQt5.QtGui import QImage
    x8, y8 = pad_tile(8)
    path = make_pack(tmp_path / "only8.mbtiles",
                     {(8, x8, y8): jpeg_tile("#000000", ("#ff0000", "#00ff00",
                                                         "#0000ff", "#ffff00"))},
                     zmin=8, zmax=14)
    provider = mw.TileProvider()
    provider.set_source(mw.MapSource(name="p", type="mbtiles", path=path,
                                     min_zoom=8, max_zoom=14))
    try:
        x, y = pad_tile(z)
        image = _fallback(qapp, provider, z, x, y)
        assert isinstance(image, QImage) and not image.isNull()
        # It is the right part of the z8 tile: the pad's quadrant of it.
        px, py = mw.lonlat_to_world(PAD_LON, PAD_LAT, 8)
        quad = (int(px % 256) >= 128) + 2 * (int(py % 256) >= 128)
        rgb = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))[quad]
        assert close_enough(pixel(image, 128, 128), rgb, tol=60)
    finally:
        provider.shutdown()


def test_a_tile_is_grey_only_when_no_level_has_it(qapp, tmp_path, no_network):
    x8, y8 = pad_tile(8)
    path = make_pack(tmp_path / "only8.mbtiles",
                     {(8, x8, y8): jpeg_tile("#336699")}, zmin=8, zmax=14)
    provider = mw.TileProvider()
    provider.set_source(mw.MapSource(name="p", type="mbtiles", path=path,
                                     min_zoom=8, max_zoom=14))
    try:
        # Far from the one tile: nothing at any level, so grey (None).
        x, y = mw.lonlat_to_tile(PAD_LON + 3.0, PAD_LAT + 3.0, 12)
        assert _fallback(qapp, provider, 12, x, y) is None
        # And it asked every level on the way, once each.
        for level in range(8, 12):
            key = ("p", level, x >> (12 - level), y >> (12 - level))
            assert key in provider._failed
    finally:
        provider.shutdown()


# ---------------------------------------------------------------------------
# Outside the pack
# ---------------------------------------------------------------------------

def test_outside_the_pack_draws_the_grid_and_says_so(qapp, tmp_path, no_network):
    window, canvas = pad_window(qapp, tmp_path)
    canvas.zoom = 13
    canvas.center_lat, canvas.center_lon = PAD_LAT + 0.5, PAD_LON + 0.5
    asked = []
    real_get = canvas.provider.get
    canvas.provider.get = lambda z, x, y: asked.append((z, x, y)) or real_get(z, x, y)
    image = grab(canvas)
    assert canvas.view_outside_pack()
    assert asked == []                   # no tile asked for outside the pack
    window._update_status()
    assert "outside map pack" in window.status.text()
    # Grid background, not grey "no tile" squares.
    from PyQt5.QtGui import QColor
    from map_basemap import GRID_BG
    bg = QColor(GRID_BG)
    # The same background the "Grid only" pack draws.
    assert share_close_to(image, (bg.red(), bg.green(), bg.blue()), tol=6) >= 0.85
    assert canvas.paint_errors == 0
    window.shutdown()
    window.close()


def test_coverage_outline_is_visible_zoomed_out(qapp, tmp_path, no_network):
    window, canvas = pad_window(qapp, tmp_path)
    canvas.zoom = 10
    canvas.center_lat, canvas.center_lon = PAD_LAT, PAD_LON
    image = grab(canvas)
    assert canvas.coverage_drawn
    west, south, east, north = PAD_BOX_10KM
    corner = canvas._lonlat_to_pixel(west, north)
    edge = canvas._lonlat_to_pixel(west, (north + south) / 2)
    from PyQt5.QtGui import QColor
    pad = QColor(mw.COL_PAD)
    found = any(close_enough(pixel(image, int(edge.x()) + dx, int(edge.y()) + dy),
                             (pad.red(), pad.green(), pad.blue()), tol=70)
                for dx in (-1, 0, 1) for dy in range(-6, 7))
    assert found, "no outline on the pack's west edge"
    assert 0 < corner.x() < canvas.width() and 0 < corner.y() < canvas.height()
    window.shutdown()
    window.close()


def test_go_to_pad_button(qapp, tmp_path, no_network):
    window, canvas = pad_window(qapp, tmp_path)
    window.follow_combo.setCurrentText("CANSAT")
    canvas.zoom = 7
    canvas.center_lat, canvas.center_lon = 10.0, 10.0
    window.pad_btn.click()
    assert (canvas.center_lat, canvas.center_lon) == (PAD_LAT, PAD_LON)
    assert canvas.zoom == mw.PAD_ZOOM == 13
    assert canvas.follow == "" and window.follow_combo.currentText() == "Off"
    assert not canvas.view_outside_pack()
    grab(canvas)
    assert canvas.pad_drawn and canvas.paint_errors == 0
    window.shutdown()
    window.close()

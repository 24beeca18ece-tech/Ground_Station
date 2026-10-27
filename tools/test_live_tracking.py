"""
tools/test_live_tracking.py
===========================

Live tracking on the launch-pad satellite pack: scripted fixes inside the
pack, the track's screen positions at zoom 13 against an independent Web
Mercator calculation, the pad marker under the track, and the status before
any fix and for a vehicle outside the pack.

    python -m pytest tools/test_live_tracking.py -v
"""

from __future__ import annotations

import math
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


#: The real pack's bounds: 10 km x 10 km around the pad.
PAD_BOX_10KM = (84.2542, 26.6751, 84.3549, 26.7650)


class Fix:
    """What TrackStore.add_packet reads from a telemetry packet."""

    def __init__(self, lat, lon, count, epoch, vehicle="CANSAT", fix=True):
        self.lat, self.lon = lat, lon
        self.has_fix = fix
        self.nav_alt_m, self.has_nav_alt = 300.0 + count, True
        self.packet_count, self.payload_type = count, vehicle
        self.gs_recv_epoch, self.mission_time_s = epoch, count * 0.3
        self.sats, self.fsm_name = 9, "DESCENT"


def mercator_px(lat, lon, zoom):
    """World pixel of lat/lon at *zoom*, written out independently of
    map_window: x = (lon + 180)/360 * 256 * 2^z,
    y = (1 - ln(tan(lat) + sec(lat))/pi)/2 * 256 * 2^z."""
    size = 256.0 * 2 ** zoom
    phi = math.radians(lat)
    return ((lon + 180.0) / 360.0 * size,
            (1.0 - math.log(math.tan(phi) + 1.0 / math.cos(phi)) / math.pi)
            / 2.0 * size)


def scripted_flight(n=12, start=(PAD_LAT, PAD_LON)):
    """A flight leaving the pad north-east, ~150 m (about 9 px at z13) per
    fix: never thinned away, and clear of the marker drawn on the newest."""
    import time
    now = time.time() - n * 0.5
    return [Fix(start[0] + 0.001 * i, start[1] + 0.001 * i, i + 1,
                now + 0.5 * i) for i in range(n)]


def window_with_pack(qapp, tmp_path, store, colour="#1f4d1f"):
    maps = pack_dir(tmp_path)
    x, y = pad_tile(13)
    tiles = {(13, x + dx, y + dy): jpeg_tile(colour)
             for dx in range(-3, 4) for dy in range(-3, 4)}
    make_pack(os.path.join(maps, "launchpad_s2.mbtiles"), tiles,
              bounds=PAD_BOX_10KM)
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    canvas = settled(qapp, window)
    canvas.zoom = 13
    canvas.center_lat, canvas.center_lon = PAD_LAT + 0.003, PAD_LON + 0.0025
    for _ in range(5):
        grab(canvas)
        wait_for(qapp, lambda: False, frames=20)
    return window, canvas


def test_track_points_land_where_the_mercator_maths_says(qapp, tmp_path,
                                                         no_network):
    store = TrackStore()
    fixes = scripted_flight()
    for fix in fixes:
        assert store.add_packet(fix)
    window, canvas = window_with_pack(qapp, tmp_path, store)
    polygon = canvas._polyline_for(store.get("CANSAT"))
    assert polygon is not None and polygon.count() == len(fixes)
    cx, cy = mercator_px(canvas.center_lat, canvas.center_lon, 13)
    for i, fix in enumerate(fixes):
        wx, wy = mercator_px(fix.lat, fix.lon, 13)
        expected = (wx - cx + canvas.width() / 2.0,
                    wy - cy + canvas.height() / 2.0)
        point = polygon.at(i)
        assert point.x() == pytest.approx(expected[0], abs=0.01)
        assert point.y() == pytest.approx(expected[1], abs=0.01)
    assert no_network == []
    window.shutdown()
    window.close()


def test_track_is_drawn_over_the_imagery_and_the_pad(qapp, tmp_path,
                                                     no_network):
    from PyQt5.QtGui import QColor
    store = TrackStore()
    fixes = scripted_flight()
    for fix in fixes:
        store.add_packet(fix)
    window, canvas = window_with_pack(qapp, tmp_path, store)
    image = grab(canvas)
    cyan = QColor(mw.VEHICLE_COLOURS["CANSAT"])
    rgb = (cyan.red(), cyan.green(), cyan.blue())
    assert canvas.provider.tiles_drawn > 0           # imagery underneath
    # Mid-segment points (between fixes, away from the newest fix's marker)
    # show the track colour over the imagery.
    for a, b in zip(fixes[:-2], fixes[1:-1]):
        mid = canvas._lonlat_to_pixel((a.lon + b.lon) / 2, (a.lat + b.lat) / 2)
        assert close_enough(pixel(image, int(round(mid.x())),
                                  int(round(mid.y()))), rgb, tol=60)
    # The first fix is on the pad: the track is drawn over the pad marker,
    # not hidden under it.
    pad = canvas._lonlat_to_pixel(PAD_LON, PAD_LAT)
    toward = canvas._lonlat_to_pixel(fixes[1].lon, fixes[1].lat)
    probe = (pad.x() + (toward.x() - pad.x()) * 0.15,
             pad.y() + (toward.y() - pad.y()) * 0.15)
    assert canvas.pad_drawn
    assert close_enough(pixel(image, int(round(probe[0])), int(round(probe[1]))),
                        rgb, tol=60)
    assert no_network == []
    window.shutdown()
    window.close()


def test_rocket_track_has_its_own_colour(qapp, tmp_path):
    from PyQt5.QtGui import QColor
    store = TrackStore()
    import time
    now = time.time() - 6
    for i in range(10):
        store.add_packet(Fix(PAD_LAT + 0.0006 * i, PAD_LON - 0.0008 * i, i + 1,
                             now + 0.5 * i, vehicle="ROCKET"))
    window, canvas = window_with_pack(qapp, tmp_path, store)
    canvas.center_lon = PAD_LON - 0.003
    image = grab(canvas)
    red = QColor(mw.VEHICLE_COLOURS["ROCKET"])
    track = store.get("ROCKET")
    mid = canvas._lonlat_to_pixel((track.lon[3] + track.lon[4]) / 2,
                                  (track.lat[3] + track.lat[4]) / 2)
    assert mw.VEHICLE_COLOURS["ROCKET"] != mw.VEHICLE_COLOURS["CANSAT"]
    assert close_enough(pixel(image, int(round(mid.x())), int(round(mid.y()))),
                        (red.red(), red.green(), red.blue()), tol=60)
    window.shutdown()
    window.close()


def test_status_before_any_fix(qapp, tmp_path):
    store = TrackStore()
    assert not store.add_packet(Fix(0.0, 0.0, 1, 1.7e9, fix=False))
    window, canvas = window_with_pack(qapp, tmp_path, store)
    window._update_status()
    assert "no GPS fix yet" in window.status.text()
    assert canvas.paint_errors == 0
    window.shutdown()
    window.close()


def test_vehicle_outside_the_pack(qapp, tmp_path):
    store = TrackStore()
    far = (PAD_LAT + 0.12, PAD_LON + 0.12)               # ~15 km NE
    for fix in scripted_flight(start=far):
        store.add_packet(fix)
    window, canvas = window_with_pack(qapp, tmp_path, store)
    canvas.center_lat, canvas.center_lon = far
    image = grab(canvas)
    window._update_status()
    assert "CANSAT outside map pack" in window.status.text()
    # Still drawn there, on the grid.
    track = store.get("CANSAT")
    mid = canvas._lonlat_to_pixel((track.lon[2] + track.lon[3]) / 2,
                                  (track.lat[2] + track.lat[3]) / 2)
    from PyQt5.QtGui import QColor
    cyan = QColor(mw.VEHICLE_COLOURS["CANSAT"])
    assert close_enough(pixel(image, int(round(mid.x())), int(round(mid.y()))),
                        (cyan.red(), cyan.green(), cyan.blue()), tol=60)
    assert canvas.paint_errors == 0
    window.shutdown()
    window.close()

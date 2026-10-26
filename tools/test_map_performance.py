"""
tools/test_map_performance.py
=============================

Regression tests for the things that made a long session hang.

These assert *shape*, not absolute speed: a ratio between a short track and
a very long one, a bound on what is retained, a cap on requests in flight.
Absolute timings vary with the machine; the ratios do not, and it is the
ratios that were wrong.

    python -m pytest tools/test_map_performance.py -v
"""

from __future__ import annotations

import math
import os
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import map_window as mw  # noqa: E402
from track_store import (MAX_POINTS_PER_VEHICLE, STATIONARY_AFTER_S,  # noqa: E402
                         STATIONARY_EVERY_S, Track, TrackStore)


class FakePacket:
    def __init__(self, lat=32.72, lon=74.85, alt=300.0, count=1,
                 payload="CANSAT", epoch=1_700_000_000.0, sats=9):
        self.lat, self.lon, self.nav_alt_m = lat, lon, alt
        self.packet_count, self.payload_type = count, payload
        self.has_fix = self.has_nav_alt = True
        self.gs_recv_epoch = epoch
        self.mission_time_s = count * 0.05
        self.sats, self.fsm_name = sats, "ASCENT"


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def fill(track: Track, n: int) -> None:
    start = len(track)
    for i in range(start, start + n):
        track.append(32.7266 + 2e-5 * i + 2e-6 * math.sin(i / 37.0),
                     74.8570 + 3e-5 * i + 2e-6 * math.cos(i / 29.0),
                     300.0 + i * 0.01, 1.7e9 + i * 0.05, i * 0.05, i, 11,
                     "ASCENT", True)


# ---------------------------------------------------------------------------
# Paint cost must not grow with the flight
# ---------------------------------------------------------------------------

def _paint_ms(canvas, reps: int = 6) -> float:
    from PyQt5.QtGui import QImage
    image = QImage(canvas.width(), canvas.height(), QImage.Format_ARGB32)
    canvas.render(image)                       # warm, and fill the caches
    start = time.perf_counter()
    for _ in range(reps):
        canvas.render(image)
    return (time.perf_counter() - start) * 1000.0 / reps


@pytest.mark.slow
def test_paint_time_is_flat_from_1k_to_200k_points(qapp, tmp_path):
    """The headline regression: paint must not scale with track length.

    Before the projection cache this was 2.3x and rising, with the whole
    track converted to pixels in a Python loop on every frame.
    """
    store = TrackStore()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    window.resize(1000, 700)
    window.show()
    qapp.processEvents()
    track = store.tracks.setdefault("CANSAT", Track("CANSAT"))

    fill(track, 1000)
    window.canvas.fit_all()
    qapp.processEvents()
    small = _paint_ms(window.canvas)

    fill(track, 199_000)
    assert len(track) == 200_000
    window.canvas.fit_all()
    qapp.processEvents()
    large = _paint_ms(window.canvas)

    window.shutdown()
    window.close()
    assert large < max(small, 0.4) * 3.0, (
        "paint grew from %.2f ms at 1k points to %.2f ms at 200k (%.1fx)"
        % (small, large, large / max(small, 1e-9)))


@pytest.mark.slow
def test_drawn_point_count_is_bounded_by_the_window_not_the_track(qapp, tmp_path):
    store = TrackStore()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    window.resize(900, 600)
    window.show()
    qapp.processEvents()
    track = store.tracks.setdefault("CANSAT", Track("CANSAT"))

    fill(track, 2000)
    window.canvas.fit_all()
    few = window.canvas._polyline_for(track).size()

    fill(track, 98_000)
    window.canvas.fit_all()
    many = window.canvas._polyline_for(track).size()

    window.shutdown()
    window.close()
    assert many <= mw.MAX_DRAWN_POINTS
    # 50x the data must not mean anything like 50x the drawn points.
    assert many < few * 3, "%d -> %d drawn points" % (few, many)


def test_projection_is_reused_and_only_new_points_are_projected(qapp, tmp_path):
    store = TrackStore()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    window.show()
    qapp.processEvents()
    track = store.tracks.setdefault("CANSAT", Track("CANSAT"))
    fill(track, 5000)

    window.canvas._polyline_for(track)
    projection = window.canvas._projections["CANSAT"]
    assert projection.wx.shape[0] == 5000
    first_array = projection.wx

    # No change: the projection object must not be rebuilt.
    window.canvas._polyline_for(track)
    assert window.canvas._projections["CANSAT"].wx is first_array

    fill(track, 10)
    window.canvas._polyline_for(track)
    assert window.canvas._projections["CANSAT"].wx.shape[0] == 5010

    window.shutdown()
    window.close()


def test_changing_zoom_reprojects(qapp, tmp_path):
    store = TrackStore()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    window.show()
    track = store.tracks.setdefault("CANSAT", Track("CANSAT"))
    fill(track, 500)
    window.canvas._polyline_for(track)
    before = window.canvas._projections["CANSAT"].wx[0]
    window.canvas.set_zoom(window.canvas.zoom + 3)
    window.canvas._polyline_for(track)
    after = window.canvas._projections["CANSAT"].wx[0]
    assert after == pytest.approx(before * 8.0, rel=1e-9)
    window.shutdown()
    window.close()


def test_vectorised_projection_matches_the_scalar_one():
    lons = np.array([74.857, 0.0, -179.9, 151.21])
    lats = np.array([32.7266, 0.0, 85.0, -33.86])
    for zoom in (1, 13, 19):
        xs, ys = mw.lonlat_to_world_array(lons, lats, zoom)
        for i in range(len(lons)):
            x, y = mw.lonlat_to_world(float(lons[i]), float(lats[i]), zoom)
            assert xs[i] == pytest.approx(x, rel=1e-12)
            assert ys[i] == pytest.approx(y, rel=1e-12)


def test_an_unchanged_view_does_not_repaint(qapp, tmp_path):
    """The 5 Hz timer must not repaint an identical frame."""
    store = TrackStore()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    window.show()
    qapp.processEvents()
    canvas = window.canvas
    painted = []
    original = canvas.update
    canvas.update = lambda *a, **k: (painted.append(1), original(*a, **k))[1]
    # Settle first: a tile arriving or an explicit invalidation legitimately
    # asks for one more frame. What must not happen is a repaint every tick
    # once the view has stopped changing.
    for _ in range(10):
        canvas._tick()
        qapp.processEvents()
    painted.clear()
    for _ in range(8):
        canvas._tick()
    assert painted == [], "repainted %d times with nothing changed" % len(painted)
    window.shutdown()
    window.close()


# ---------------------------------------------------------------------------
# What the store keeps
# ---------------------------------------------------------------------------

def test_points_are_stored_compactly():
    """A list of dataclasses measured 363 bytes a fix; columns are ~44."""
    track = Track("CANSAT")
    fill(track, 50_000)
    columns = sum(getattr(track, name).nbytes for name in
                  ("lat", "lon", "alt", "epoch", "mission", "count", "sats",
                   "fsm"))
    per_point = columns / len(track)
    assert per_point < 80, "%.0f bytes per stored fix" % per_point


def test_stationary_vehicle_is_thinned_but_never_silently():
    """A vehicle on the pad must not fill the store with the same spot."""
    store = TrackStore()
    epoch = 1_700_000_000.0
    stored = 0
    # 20 Hz for five minutes, not moving at all.
    for i in range(1, 6001):
        if store.add_packet(FakePacket(count=i, epoch=epoch + i * 0.05)):
            stored += 1
    track = store.get("CANSAT")
    assert len(track) == stored
    # After the first STATIONARY_AFTER_S at full rate, one per
    # STATIONARY_EVERY_S: far fewer than 6000.
    assert stored < 6000 * 0.25, "stored %d of 6000 stationary fixes" % stored
    assert track.stationary_dropped > 0


def test_movement_resumes_full_rate_immediately():
    store = TrackStore()
    epoch = 1_700_000_000.0
    for i in range(1, 3001):                       # sit still
        store.add_packet(FakePacket(count=i, epoch=epoch + i * 0.05))
    before = len(store.get("CANSAT"))
    # Now it launches: every fix must be kept.
    for j in range(3001, 3101):
        store.add_packet(FakePacket(count=j, lat=32.72 + (j - 3000) * 1e-4,
                                    epoch=epoch + j * 0.05))
    assert len(store.get("CANSAT")) == before + 100


def test_hard_cap_thins_instead_of_growing():
    track = Track("CANSAT")
    fill(track, MAX_POINTS_PER_VEHICLE + 5000)
    assert len(track) <= MAX_POINTS_PER_VEHICLE
    assert track.thinned > 0
    # The newest data must survive at full resolution.
    assert int(track.count[len(track) - 1]) == MAX_POINTS_PER_VEHICLE + 4999


def test_cap_keeps_the_endpoints():
    track = Track("CANSAT")
    fill(track, MAX_POINTS_PER_VEHICLE + 1000)
    assert int(track.count[0]) == 0
    assert track.first is not None and track.last is not None


# ---------------------------------------------------------------------------
# Tile requests with no network
# ---------------------------------------------------------------------------

def dead_source() -> mw.MapSource:
    """TEST-NET-1 (RFC 5737) is reserved and must not be routed."""
    return mw.MapSource(name="dead", type="xyz",
                        url="http://192.0.2.1/{z}/{x}/{y}.png",
                        attribution="unreachable", min_zoom=1, max_zoom=19)


def test_one_paint_never_starts_more_requests_than_the_cap(qapp, tmp_path):
    store = TrackStore()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    window.resize(1200, 900)
    window.show()
    qapp.processEvents()
    window.canvas.set_source(dead_source(), str(tmp_path))

    from PyQt5.QtGui import QImage
    image = QImage(window.canvas.width(), window.canvas.height(),
                   QImage.Format_ARGB32)
    window.canvas.render(image)
    provider = window.canvas.provider
    assert provider._inflight <= mw.MAX_INFLIGHT, provider._inflight

    # Repainting must not multiply them either.
    for _ in range(5):
        window.canvas.render(image)
    assert provider._inflight <= mw.MAX_INFLIGHT, provider._inflight
    window.shutdown()
    window.close()


def test_a_failed_tile_is_not_re_requested_for_the_negative_cache_window(
        qapp, tmp_path):
    store = TrackStore()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    window.show()
    provider = window.canvas.provider
    provider.set_source(dead_source(), str(tmp_path))
    key = ("dead", 13, 100, 100)

    provider._store(key, None)                    # as a failure would
    assert key in provider._failed
    sent = []
    provider._request = lambda k: sent.append(k)
    assert provider.get(13, 100, 100) is None
    assert sent == [], "re-requested a tile inside the negative-cache window"

    # Past the window it is allowed again: the network may have come back.
    provider._failed[key] = time.monotonic() - 1.0
    assert provider.get(13, 100, 100) is None
    assert sent == [key]
    window.shutdown()
    window.close()


def test_a_source_that_keeps_failing_is_declared_offline(qapp, tmp_path):
    store = TrackStore()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    window.show()
    provider = window.canvas.provider
    provider.set_source(dead_source(), str(tmp_path))
    assert provider.is_offline is False
    for i in range(mw.OFFLINE_AFTER_FAILURES):
        provider._store(("dead", 13, i, i), None)
    assert provider.is_offline is True

    sent = []
    provider._request = lambda k: sent.append(k)
    assert provider.get(13, 999, 999) is None
    assert sent == [], "kept requesting from a source already known offline"

    provider.retry_source()
    assert provider.is_offline is False
    window.shutdown()
    window.close()


def test_offscreen_requests_are_aborted_on_pan(qapp, tmp_path):
    store = TrackStore()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    window.show()
    provider = window.canvas.provider
    provider.set_source(dead_source(), str(tmp_path))
    provider._pending = {("dead", 13, 1, 1): None, ("dead", 13, 2, 2): None}
    provider._queue = [("dead", 13, 3, 3)]
    provider.abort_pending(keep={("dead", 13, 1, 1)})
    assert set(provider._pending) == {("dead", 13, 1, 1)}
    assert provider._queue == []
    window.shutdown()
    window.close()


def test_the_negative_cache_does_not_grow_without_bound(qapp, tmp_path):
    store = TrackStore()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    provider = window.canvas.provider
    provider.set_source(dead_source(), str(tmp_path))
    for i in range(5000):
        provider._failed[("dead", 13, i, i)] = time.monotonic() - 1.0
    provider._mark_failed(("dead", 13, 99999, 99999))
    assert len(provider._failed) < 4096
    window.shutdown()
    window.close()


def test_mbtiles_source_is_never_declared_offline(qapp, tmp_path):
    """A missing pack is not a network failure and must not say OFFLINE."""
    store = TrackStore()
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    provider = window.canvas.provider
    provider.set_source(mw.MapSource(name="o", type="mbtiles",
                                     path="assets/maps"), str(tmp_path))
    for i in range(mw.OFFLINE_AFTER_FAILURES * 2):
        provider._store(("o", 13, i, i), None)
    assert provider.is_offline is False
    window.shutdown()
    window.close()


# ---------------------------------------------------------------------------
# Chart series storage
# ---------------------------------------------------------------------------

def test_chart_series_are_numpy_and_redraw_hands_out_views(qapp):
    """A redraw must not allocate: pyqtgraph gets a view, not a new list.

    Slicing a Python list of 40,000 floats allocates a fresh list every
    frame; sixteen series at 15 Hz was measured as the dominant remaining
    source of heap churn after the logger fix.
    """
    import numpy as _np
    from dashboard_ui import MAX_PLOT_POINTS, StripChart
    chart = StripChart("t", "u", [("a", "#ff0000"), ("b", "#00ff00")])
    for i in range(500):
        chart.add_point(i * 0.05, {"a": float(i), "b": float(i) * 2})
    assert isinstance(chart._x, _np.ndarray)
    assert isinstance(chart._y["a"], _np.ndarray)
    view = chart._x[10:chart._n]
    assert view.base is not None, "slicing copied instead of viewing"


def test_chart_series_respects_its_cap(qapp):
    from dashboard_ui import MAX_PLOT_POINTS, _COMPACT_SLACK, StripChart
    chart = StripChart("t", "u", [("a", "#ff0000")])
    for i in range(MAX_PLOT_POINTS + _COMPACT_SLACK * 3):
        chart.add_point(i * 0.05, {"a": float(i)})
    assert chart._n <= MAX_PLOT_POINTS + _COMPACT_SLACK
    # The newest sample must survive compaction: it is the live edge.
    assert chart._x[chart._n - 1] == pytest.approx(
        (MAX_PLOT_POINTS + _COMPACT_SLACK * 3 - 1) * 0.05)


def test_chart_x_stays_non_decreasing_through_compaction(qapp):
    import numpy as _np
    from dashboard_ui import MAX_PLOT_POINTS, StripChart
    chart = StripChart("t", "u", [("a", "#ff0000")])
    for i in range(MAX_PLOT_POINTS + 9000):
        chart.add_point(i * 0.05, {"a": float(i)})
    window = chart._x[:chart._n]
    assert _np.all(_np.diff(window) >= 0), "compaction scrambled the x series"


def test_chart_clear_resets_without_reallocating(qapp):
    from dashboard_ui import StripChart
    chart = StripChart("t", "u", [("a", "#ff0000")])
    for i in range(1000):
        chart.add_point(i * 0.05, {"a": float(i)})
    before = chart._x
    chart.clear_data()
    assert chart._n == 0
    assert chart._x is before, "clear reallocated the buffer"


def test_the_dashboard_constructs(qapp, tmp_path):
    """Nothing else in the suite builds a Dashboard, and a missing import
    in it was therefore invisible until the application was run."""
    from dashboard_ui import Dashboard
    window = Dashboard(log_dir=str(tmp_path))
    assert window.chart_alt._n == 0
    window.close()

"""
tools/test_long_session.py
==========================

A 20-minute session at 20 Hz, headless and compressed: 24,000 simulated
CanSat packets through the real radio entry point, merger and dashboard,
with CSV logging on, the map status line ticking, and the dashboard
rendering once per simulated second.

What made the PC lag on a long run was GUI-thread work that grew with the
session: the ground-track plot redrew every fix it had ever received, up to
40,000, on every render; and the map's status line built an object for every
stored fix twice a second, even with the map closed. This guards both, and
memory:

* render time at the end of the session is within 2x of the start
* the map status tick does not grow with the track
* RSS has stopped growing: under 3 MB over the last 3 minutes
* (expected to fail, see its reason) RSS grows less than 20 MB after a
  3-minute warm-up

    python -m pytest tools/test_long_session.py -v -s
"""

from __future__ import annotations

import os
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RATE_HZ = 20
MINUTES = 20
WARMUP_MIN = 3


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def run_session(qapp, tmp_path, minutes=MINUTES, prepare=None):
    """Drive the dashboard; return per-minute RSS and timing samples."""
    import psutil
    from dashboard_ui import Dashboard
    from packet_sim import MissionSim
    from telemetry_packet import build_frame, parse_frame

    proc = psutil.Process()
    sim = MissionSim(origin=(26.720028, 84.304583))
    win = Dashboard(log_dir=str(tmp_path / "logs"))
    win.resize(1600, 950)
    win.show()
    # The radio thread is not used: packets are handed to the same slot the
    # serial worker's signal calls. The callout would speak every 3 s.
    win.callout_timer.stop()
    win.render_timer.stop()
    win.attitude_timer.stop()
    win.telemetry_active = True
    if prepare is not None:
        prepare(win)
    win.toggle_logging()
    t0 = 1.79e9
    samples = []
    per_second = RATE_HZ
    total = minutes * 60 * per_second
    for i in range(total):
        frame = build_frame(sim.next_payload())
        packet = parse_frame(frame, gs_recv_epoch=t0 + i / RATE_HZ)
        win.on_radio_packet(packet, "RX1" if i % 2 else "RX2")
        if i % per_second == per_second - 1:
            # One simulated second: drain, render, status, attitude, events.
            # Release everything held (the merge window has "passed").
            released = win.merger.drain(now=time.monotonic() + 10.0)
            released += win.merger.flush()
            for p, _src in released:
                win.on_packet(p)
            start = time.perf_counter()
            win._render()
            render_ms = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            win.map_window._update_status()
            status_ms = (time.perf_counter() - start) * 1000.0
            win.attitude.redraw()
            qapp.processEvents()
            second = (i + 1) // per_second
            if second % 60 == 0:
                samples.append(dict(minute=second // 60,
                                    rss_mb=proc.memory_info().rss / 2 ** 20,
                                    render_ms=render_ms, status_ms=status_ms,
                                    packets=win.session_packets,
                                    track=sum(len(t) for t in
                                              win.track_store.tracks.values())))
    win.csv_logger.stop()
    win.csv_logger.wait(10000)
    win.map_window.shutdown()
    win.close()
    return samples


@pytest.fixture(scope="module")
def session(qapp, tmp_path_factory):
    """One 20-minute session, shared by the tests below."""
    samples = run_session(qapp, tmp_path_factory.mktemp("session"))
    for s in samples:
        print("  min %2d  RSS %6.1f MB  render %6.1f ms  map status %5.2f ms  "
              "track %5d" % (s["minute"], s["rss_mb"], s["render_ms"],
                             s["status_ms"], s["track"]))
    return samples


def _med(key, rows):
    values = sorted(r[key] for r in rows)
    return values[len(values) // 2]


@pytest.mark.slow
def test_a_20_minute_session_does_not_grow(session):
    end = session[-1]
    # Every packet reached the dashboard. (The track itself is shorter: the
    # store thins repeated fixes while the vehicle sits on the pad.)
    assert end["packets"] == MINUTES * 60 * RATE_HZ
    assert end["track"] > MINUTES * 60 * RATE_HZ * 0.9
    # Render and status cost do not grow with the session (medians, so one
    # slow frame cannot decide it). Before the fix: render 5 -> 40-70 ms,
    # map status 1 -> 61 ms over the same 20 minutes.
    early = [s for s in session if WARMUP_MIN <= s["minute"] < WARMUP_MIN + 4]
    late = session[-4:]
    assert _med("render_ms", late) < 2.0 * _med("render_ms", early) + 5.0
    assert _med("status_ms", late) < 2.0 * _med("status_ms", early) + 1.0
    # Memory has stopped growing: the last 3 minutes add almost nothing.
    by_min = {s["minute"]: s["rss_mb"] for s in session}
    tail = by_min[MINUTES] - by_min[MINUTES - 3]
    print("  RSS over the last 3 minutes: %+.1f MB" % tail)
    assert tail < 3.0


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason=(
    "Bounded buffers fill during minutes 3-17 of a 20 Hz session -- the live "
    "CSV table's 20,000 rows (~15 MB) and the chart and track buffers "
    "(~9 MB) -- so growth after a 3-minute warm-up measured +23.5 MB, over "
    "the 20 MB target, then flat. Meeting it needs a smaller live table "
    "(MAX_ROWS in csv_table_window.py): a behaviour decision, not made here."))
def test_rss_growth_after_warm_up_is_under_20_mb(session):
    warm = next(s for s in session if s["minute"] == WARMUP_MIN)
    growth = session[-1]["rss_mb"] - warm["rss_mb"]
    print("  RSS growth after %d min warm-up: %+.1f MB" % (WARMUP_MIN, growth))
    assert growth < 20.0

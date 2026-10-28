#!/usr/bin/env python3
"""
tools/soak_app.py
=================

The ground station as tools/soak_guard.py runs it: the real dashboard, on the
real display, connected to packet_sim, with telemetry and CSV logging on --
what an operator does. Not meant to be run by hand (the guard starts it, and
kills it if memory runs short).

Options switch on the windows an operator opens, and switch off one component
at a time for bisecting a leak. The switches are applied here, to this
process only; the application's own code is unchanged.

    --map             open the map window (the launch pad pack loads)
    --diag            open the diagnostics window
    --disable NAME    csv | charts | groundtrack | attitude | map | voice |
                      link | merger
                      (charts = the strip charts; groundtrack = the
                      dashboard's GNSS ground-track plot, a separate widget)
    --tracemalloc     record allocations; write the top growth at the end
    --profile S D     cProfile the GUI thread from S seconds for D seconds
    --stats FILE      JSON lines, every 5 s: packets, GUI lateness, objects

GUI lateness: a 50 ms timer measures how late it fires. That lateness is the
lag an operator feels -- the GUI thread is what repaints and drains the
merge buffer.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

COMPONENTS = ("csv", "charts", "groundtrack", "attitude", "map", "voice",
              "link", "merger")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--map", action="store_true")
    parser.add_argument("--diag", action="store_true")
    parser.add_argument("--disable", choices=COMPONENTS, default=None)
    parser.add_argument("--tracemalloc", action="store_true")
    parser.add_argument("--profile", nargs=2, type=float, default=None,
                        metavar=("START_S", "SECONDS"))
    parser.add_argument("--stats", required=True)
    parser.add_argument("--log-dir", required=True)
    return parser.parse_args(argv)


def disable(win, name):
    """Switch one component off in this process only."""
    import dashboard_ui as dui
    if name == "csv":
        win._soak_no_logging = True               # START LOGGING is not pressed
    elif name == "charts":
        dui.StripChart.add_point = lambda self, *a, **k: None
        dui.StripChart.redraw = lambda self, *a, **k: None
    elif name == "groundtrack":
        dui.GpsTrackPlot.add_fix = lambda self, *a, **k: None
        dui.GpsTrackPlot.redraw = lambda self, *a, **k: None
    elif name == "attitude":
        win.attitude_timer.stop()
        win.attitude.on_packet = lambda packet: None
    elif name == "map":
        win.track_store.add_packet = lambda packet: False
    elif name == "voice":
        win.callout_timer.stop()
        win.callout.stop()
    elif name == "link":
        win.status_timer.stop()
    elif name == "merger":
        from packet_merger import PacketMerger

        class Passthrough(PacketMerger):
            """Emit each packet as it arrives: no buffering, no ordering."""

            def add(self, packet, source, now=None):
                self._through = getattr(self, "_through", [])
                self._through.append((packet, source))
                self.released += 1
                self.per_source[source] = self.per_source.get(source, 0) + 1

            def drain(self, now=None):
                out, self._through = getattr(self, "_through", []), []
                return out

        win.merger = Passthrough(on_warning=win._on_merge_warning,
                                 on_gap=win._on_merge_gap)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.tracemalloc:
        import tracemalloc
        tracemalloc.start(12)

    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv[:1])
    from dashboard_ui import Dashboard, apply_dark_palette
    apply_dark_palette(app)

    win = Dashboard(log_dir=args.log_dir)
    win._soak_no_logging = False
    if args.disable:
        disable(win, args.disable)
    win.showMaximized()

    stats = open(args.stats, "a", encoding="utf-8")
    state = {"late": [], "last": time.monotonic(), "t0": time.monotonic(),
             "snap0": None}

    def heartbeat():
        now = time.monotonic()
        state["late"].append(max(0.0, (now - state["last"]) - 0.050) * 1000.0)
        state["last"] = now

    beat = QTimer()
    beat.setTimerType(Qt.PreciseTimer)
    beat.timeout.connect(heartbeat)
    beat.start(50)

    def sample():
        late = sorted(state["late"]) or [0.0]
        state["late"] = []
        workers = getattr(win, "serial_workers", {})
        frames = sum(getattr(w, "total_frames", 0) for w in workers.values())
        record = {
            "t": round(time.monotonic() - state["t0"], 1),
            "packets": win.session_packets,
            "frames_read": frames,
            "late_p50_ms": round(late[len(late) // 2], 1),
            "late_p95_ms": round(late[int(len(late) * 0.95) - 1], 1),
            "late_max_ms": round(late[-1], 1),
            "gc_objects": len(gc.get_objects()),
            "event_log_blocks": win.event_log.blockCount(),
            "track_points": sum(len(t) for t in win.track_store.tracks.values()),
        }
        stats.write(json.dumps(record) + "\n")
        stats.flush()

    sampler = QTimer()
    sampler.timeout.connect(sample)
    sampler.start(5000)

    def start():
        win.radio_widgets["RX1"]["port"].setEditText(
            "socket://127.0.0.1:%d" % args.port)
        win.toggle_connection("RX1")
        QTimer.singleShot(800, begin)

    def begin():
        if not win.telemetry_active:
            win.telemetry_btn.click()
        if not win._soak_no_logging:
            win.log_btn.click()
        if args.map:
            win.map_btn.click()
        if args.diag:
            win.diag_btn.click()
        if args.tracemalloc:
            import tracemalloc
            # Warm-up first: the snapshot to compare against is taken after
            # 60 s, so startup allocations are not mistaken for growth.
            QTimer.singleShot(60000, lambda: state.update(
                snap0=tracemalloc.take_snapshot()))

    QTimer.singleShot(1500, start)

    if args.profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()          # this thread only: the GUI

        def stop_profile():
            profiler.disable()
            with open(args.stats + ".profile.txt", "w", encoding="utf-8") as out:
                out.write("GUI thread, %.0f s from t=%.0f s; packets so far "
                          "%d\n\n" % (args.profile[1], args.profile[0],
                                      win.session_packets))
                for key in ("tottime", "cumulative"):
                    out.write("=== top by %s ===\n" % key)
                    stats_view = pstats.Stats(profiler, stream=out)
                    stats_view.sort_stats(key).print_stats(25)
            profiler.dump_stats(args.stats + ".prof")

        QTimer.singleShot(int(args.profile[0] * 1000), profiler.enable)
        QTimer.singleShot(int((args.profile[0] + args.profile[1]) * 1000),
                          stop_profile)

    def write_tracemalloc():
        import tracemalloc
        if state["snap0"] is None:
            return
        snap = tracemalloc.take_snapshot()
        top = snap.compare_to(state["snap0"], "traceback")
        with open(args.stats + ".tracemalloc.txt", "w", encoding="utf-8") as out:
            out.write("tracemalloc: growth since the 60 s mark, top 25\n\n")
            for stat in top[:25]:
                out.write("%+.1f KiB  %+d blocks\n" % (stat.size_diff / 1024,
                                                      stat.count_diff))
                for line in stat.traceback.format()[-8:]:
                    out.write("    %s\n" % line)
                out.write("\n")

    # The guard ends the run with a terminate; on Windows that is not
    # catchable, so the tracemalloc report is written on a timer instead.
    if args.tracemalloc:
        QTimer.singleShot(int(float(os.environ.get("SOAK_REPORT_AT_S", "280"))
                              * 1000), write_tracemalloc)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

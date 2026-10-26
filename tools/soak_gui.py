"""
tools/soak_gui.py
=================

Runs the real dashboard under a sustained packet load and measures whether
anything grows without bound.

NOT part of the pytest run: it takes minutes to hours and needs a packet
source. Run it deliberately, before a flight or after touching the map.

    # 20 real minutes, map closed -- the baseline
    python tools/soak_gui.py --minutes 20 --map closed

    # same, with the map open on the offline source
    python tools/soak_gui.py --minutes 20 --map offline

    # map open on a source that cannot be reached: no network at the site
    python tools/soak_gui.py --minutes 20 --map dead-network

    # a long flight compressed: inject at 40x instead of waiting
    python tools/soak_gui.py --minutes 5 --speedup 40 --map offline

WHAT IT WATCHES
---------------
Every --interval seconds: resident memory, the ten most numerous Python
object types, stored track points, pending tile requests, live threads, GUI
heartbeat latency (p50/p95/max over the interval), map paint time, and CPU.

A heartbeat timer fires at a fixed 50 ms. How late it is IS the user-visible
stall: the GUI thread is the one that repaints the charts and drains the
merge buffer, so if it is late, the application is unresponsive.

ACCEPTANCE (checked and printed at the end)
-------------------------------------------
* RSS grows by less than 50 MB after the first two minutes (startup and
  cache fill are not a leak)
* heartbeat p95 under 400 ms, max under 1 s
* thread count and pending-request count flat
"""

from __future__ import annotations

import argparse
import collections
import gc
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import psutil  # noqa: E402
from PyQt5.QtCore import QThreadPool, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from dashboard_ui import Dashboard, apply_daylight_palette  # noqa: E402
from telemetry_packet import build_frame, parse_frame  # noqa: E402

#: A source pointed at an address that cannot answer, to stand in for "the
#: launch site has no network". TEST-NET-1 (RFC 5737) is reserved and must
#: not be routed, so this fails the way a dead link fails.
DEAD_HOST = "http://192.0.2.1/{z}/{x}/{y}.png"

HEARTBEAT_MS = 50


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


class Soak:
    def __init__(self, args) -> None:
        self.args = args
        self.proc = psutil.Process()
        self.samples: List[Dict] = []
        self.beats: List[float] = []
        self.paint_ms: List[float] = []
        self.last_beat = time.perf_counter()
        self.started = time.time()
        self.injected = 0
        self.failures: List[str] = []

        self.log_dir = os.path.join(
            os.environ.get("TEMP", "/tmp"), "gcs_soak_%d" % os.getpid())
        shutil.rmtree(self.log_dir, ignore_errors=True)

        self.app = QApplication.instance() or QApplication(sys.argv[:1])
        apply_daylight_palette(self.app)
        self.win = Dashboard(log_dir=self.log_dir)
        self.win.showMaximized()

        self.sim: Optional[subprocess.Popen] = None
        if not args.speedup:
            self._start_simulator()
        else:
            self._start_injector()

        self._open_map()

        self.hb = QTimer()
        self.hb.timeout.connect(self._beat)
        self.hb.start(HEARTBEAT_MS)

        self.sampler = QTimer()
        self.sampler.timeout.connect(self._sample)
        self.sampler.start(int(args.interval * 1000))

        QTimer.singleShot(int(args.minutes * 60 * 1000), self._finish)

    # -- sources -----------------------------------------------------------

    def _start_simulator(self) -> None:
        exe = sys.executable
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sim = subprocess.Popen(
            [exe, os.path.join(root, "packet_sim.py"), "--port",
             str(self.args.port), "--payload-type", "cansat",
             "--rate", str(self.args.rate)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        self.win.port_combo.setEditText("socket://127.0.0.1:%d" % self.args.port)
        QTimer.singleShot(300, lambda: self.win.toggle_connection("RX1"))
        QTimer.singleShot(700, lambda: self.win.telemetry_btn.click())
        QTimer.singleShot(900, lambda: self.win.log_btn.click())

    def _start_injector(self) -> None:
        """Feed on_packet directly, --speedup times faster than real time.

        Bypasses the serial layer on purpose: this mode exists to reach a
        200,000-point track in minutes, which is about what the display and
        the store do with a long flight, not about the radio.
        """
        QTimer.singleShot(400, lambda: self.win.telemetry_btn.click())
        QTimer.singleShot(600, lambda: self.win.log_btn.click())
        self.inject_timer = QTimer()
        self.inject_timer.timeout.connect(self._inject)
        self.inject_timer.start(20)              # 50 batches/second

    def _inject(self) -> None:
        import math
        per_batch = max(1, int(self.args.rate * self.args.speedup / 50.0))
        for _ in range(per_batch):
            self.injected += 1
            i = self.injected
            # A real flight path: climbing, drifting downrange, with the
            # kind of small jitter a receiver actually produces.
            lat = 32.7266 + 2.0e-5 * i + 2e-6 * math.sin(i / 37.0)
            lon = 74.8570 + 3.0e-5 * i + 2e-6 * math.cos(i / 29.0)
            alt = 300.0 + 700.0 * math.sin(i / 4000.0) ** 2
            body = ("2025ASI001,CANSAT,%.2f,%d,%.2f,985.23,27.40,7.95,"
                    "09:05:07,%.6f,%.6f,%.1f,11,0.12,-0.05,9.79,0.40,"
                    "-1.20,0.06,3,4.10,6.20,7.00,7.40,120,1"
                    % (i * 0.05, i, alt, lat, lon, alt))
            try:
                self.win.on_packet(parse_frame(build_frame(body)))
            except Exception as exc:                      # noqa: BLE001
                self.failures.append("inject: %r" % exc)
                return

    def _open_map(self) -> None:
        mode = self.args.map
        if mode == "closed":
            return
        import map_window as mw
        QTimer.singleShot(1500, self.win.map_btn.click)

        def pick():
            canvas = self.win.map_window.canvas
            if mode == "dead-network":
                source = mw.MapSource(
                    name="Dead network", type="xyz", url=DEAD_HOST,
                    attribution="unreachable on purpose",
                    min_zoom=1, max_zoom=19)
                canvas.set_source(source, os.getcwd())
            canvas.fit_all()
        QTimer.singleShot(2000, pick)

    # -- measurement -------------------------------------------------------

    def _beat(self) -> None:
        now = time.perf_counter()
        self.beats.append(now - self.last_beat)
        self.last_beat = now

    def _paint_time_ms(self) -> float:
        """Time one full repaint of the map, off the normal timer path."""
        if self.args.map == "closed" or not self.win.map_window.isVisible():
            return 0.0
        from PyQt5.QtGui import QImage
        canvas = self.win.map_window.canvas
        width = max(64, canvas.width())
        height = max(64, canvas.height())
        image = QImage(width, height, QImage.Format_ARGB32)
        start = time.perf_counter()
        canvas.render(image)
        return (time.perf_counter() - start) * 1000.0

    def _pending_requests(self) -> int:
        if self.args.map == "closed":
            return 0
        provider = self.win.map_window.canvas.provider
        return len(getattr(provider, "_pending", ()))

    def _top_types(self, n: int = 10) -> List[str]:
        counts: collections.Counter = collections.Counter()
        for obj in gc.get_objects():
            counts[type(obj).__name__] += 1
        return ["%s=%d" % (name, count) for name, count in counts.most_common(n)]

    def _sample(self) -> None:
        beats, self.beats = self.beats, []
        paint = self._paint_time_ms()
        self.paint_ms.append(paint)
        rss = self.proc.memory_info().rss / (1024 * 1024)
        store = self.win.track_store
        sample = {
            "t": time.time() - self.started,
            "rss_mb": rss,
            "points": store.total_points,
            "pending": self._pending_requests(),
            "threads": self.proc.num_threads(),
            "pool": QThreadPool.globalInstance().activeThreadCount(),
            "hb_p50": percentile(beats, 0.50) * 1000,
            "hb_p95": percentile(beats, 0.95) * 1000,
            "hb_max": (max(beats) * 1000) if beats else 0.0,
            "paint_ms": paint,
            "cpu": self.proc.cpu_percent(interval=None),
            "packets": self.win.session_packets,
        }
        self.samples.append(sample)
        print("  t=%6.0fs rss=%7.1fMB pts=%7d pend=%3d thr=%2d pool=%d "
              "hb p50/p95/max=%5.0f/%5.0f/%6.0f ms paint=%6.1f ms cpu=%5.1f%% "
              "pkts=%d"
              % (sample["t"], rss, sample["points"], sample["pending"],
                 sample["threads"], sample["pool"], sample["hb_p50"],
                 sample["hb_p95"], sample["hb_max"], paint, sample["cpu"],
                 sample["packets"]), flush=True)
        if self.args.types and len(self.samples) % 4 == 0:
            print("        types: %s" % ", ".join(self._top_types()), flush=True)

    # -- end ---------------------------------------------------------------

    def _finish(self) -> None:
        try:
            if self.win.logging_enabled:
                self.win.log_btn.click()
            if self.win.telemetry_active:
                self.win.telemetry_btn.click()
            for rid in ("RX1", "RX2"):
                if self.win.radio_state[rid]["connected"]:
                    self.win.toggle_connection(rid)
        except Exception:
            pass
        if self.sim is not None:
            try:
                self.sim.kill()
            except Exception:
                pass
        self.report()
        self.win.close()
        QTimer.singleShot(200, self.app.quit)

    def report(self) -> None:
        print()
        print("=" * 100)
        print("SOAK RESULT  map=%s  %.0f min  rate=%s  speedup=%s"
              % (self.args.map, self.args.minutes, self.args.rate,
                 self.args.speedup or "none"))
        print("=" * 100)
        if len(self.samples) < 3:
            print("  too few samples to judge")
            return

        # The first two minutes are start-up and cache fill, not a leak.
        settle = [s for s in self.samples if s["t"] >= 120.0] or self.samples[-2:]
        rss0, rss1 = settle[0]["rss_mb"], settle[-1]["rss_mb"]
        growth = rss1 - rss0
        p95 = max(s["hb_p95"] for s in settle)
        worst = max(s["hb_max"] for s in settle)
        threads = [s["threads"] for s in settle]
        pending = [s["pending"] for s in settle]
        paints = [s["paint_ms"] for s in settle if s["paint_ms"]]

        print("  points stored     : %d -> %d"
              % (self.samples[0]["points"], self.samples[-1]["points"]))
        print("  RSS after settle  : %.1f -> %.1f MB   (growth %+.1f MB)"
              % (rss0, rss1, growth))
        print("  heartbeat p95     : %.0f ms (worst interval)" % p95)
        print("  heartbeat max     : %.0f ms" % worst)
        print("  threads           : %d -> %d  (min %d max %d)"
              % (threads[0], threads[-1], min(threads), max(threads)))
        print("  pending requests  : min %d max %d" % (min(pending), max(pending)))
        if paints:
            print("  map paint         : first %.1f ms, last %.1f ms, max %.1f ms"
                  % (paints[0], paints[-1], max(paints)))
        print()

        checks = [
            ("RSS growth after 2 min < 50 MB", growth < 50.0,
             "%+.1f MB" % growth),
            ("heartbeat p95 < 400 ms", p95 < 400.0, "%.0f ms" % p95),
            ("heartbeat max < 1000 ms", worst < 1000.0, "%.0f ms" % worst),
            ("thread count flat (within 2)", max(threads) - min(threads) <= 2,
             "%d..%d" % (min(threads), max(threads))),
            ("pending requests capped (<= 8)", max(pending) <= 8,
             "max %d" % max(pending)),
            ("no injection failures", not self.failures,
             "; ".join(self.failures[:2])),
        ]
        if paints and len(paints) > 2:
            checks.append(("paint time not growing (last < 3x first)",
                           paints[-1] < max(paints[0], 0.5) * 3.0,
                           "%.1f -> %.1f ms" % (paints[0], paints[-1])))
        ok = True
        for label, passed, detail in checks:
            ok = ok and passed
            print("  %-44s %s   %s" % (label, "PASS" if passed else "FAIL", detail))
        print()
        print("SOAK PASS" if ok else "SOAK FAIL")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=15.0,
                        help="seconds between samples (default 15)")
    parser.add_argument("--map", choices=("closed", "offline", "dead-network"),
                        default="offline")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="packets per second")
    parser.add_argument("--speedup", type=float, default=0.0,
                        help="inject packets this many times faster than real "
                             "time instead of running packet_sim")
    parser.add_argument("--port", type=int, default=5790)
    parser.add_argument("--types", action="store_true",
                        help="also print the top object types")
    args = parser.parse_args(argv)

    soak = Soak(args)
    return soak.app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

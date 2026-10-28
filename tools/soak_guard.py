#!/usr/bin/env python3
"""
tools/soak_guard.py
===================

Runs the ground station and packet_sim as child processes for a timed soak,
samples both every 5 s, and kills everything before the PC runs out of
memory. One soak at a time: a lock file refuses a second.

    python tools/soak_guard.py --label baseline --minutes 5
    python tools/soak_guard.py --label map --minutes 5 -- --map
    python tools/soak_guard.py --label no-charts --minutes 5 -- --disable charts

Everything after ``--`` goes to tools/soak_app.py.

LIMITS (the run is killed, and says so, when either is crossed)
    GCS resident memory above GCS_RSS_LIMIT_MB (1.5 GB)
    free system RAM below MIN_FREE_MB (1.5 GB)

OUTPUT
    tools/soak_log.csv: one row per sample -- label, seconds, RSS and CPU of
    the GCS and of the simulator, free RAM, system CPU, and the GCS's own
    figures (packets, GUI lateness, Python objects) from tools/soak_app.py.
    A summary is printed at the end: RSS at each minute, growth per minute
    after the first, CPU, and GUI lateness.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

GCS_RSS_LIMIT_MB = 1536.0
MIN_FREE_MB = 1536.0
SAMPLE_S = 5.0
LOCK = os.path.join(HERE, ".soak.lock")
LOG = os.path.join(HERE, "soak_log.csv")
FIELDS = ["label", "t_s", "gcs_rss_mb", "gcs_cpu_pct", "sim_rss_mb",
          "sim_cpu_pct", "free_mb", "system_cpu_pct", "gcs_threads",
          "packets", "late_p95_ms", "late_max_ms", "gc_objects",
          "track_points", "event_log_blocks", "killed"]


def machine() -> str:
    vm = psutil.virtual_memory()
    return ("RAM total %.2f GB, free %.2f GB (%.0f%% used); CPUs %d logical, "
            "%s physical" % (vm.total / 2 ** 30, vm.available / 2 ** 30,
                             vm.percent, psutil.cpu_count(),
                             psutil.cpu_count(logical=False)))


def take_lock() -> None:
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip() or 0)
        except ValueError:
            pid = 0
        if pid and psutil.pid_exists(pid):
            sys.exit("another soak is running (pid %d); one at a time" % pid)
    with open(LOCK, "w") as handle:
        handle.write(str(os.getpid()))


def kill_tree(proc: Optional[psutil.Popen]) -> None:
    if proc is None:
        return
    try:
        children = proc.children(recursive=True)
    except psutil.Error:
        children = []
    for p in children + [proc]:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(children + [proc], timeout=5)


def tree(proc: psutil.Popen) -> List[psutil.Process]:
    """The process and its children. A venv's python.exe on Windows is a
    small launcher that starts the real interpreter as a child, so the
    launcher alone reads ~4 MB and 0 % CPU -- the child is the program."""
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.Error:
        return [proc]


def measure(proc: psutil.Popen) -> Dict[str, float]:
    """RSS (MB), CPU (% of one core) and threads, summed over the tree.
    CPU needs a previous call per process; the first reading of a new child
    is 0 and is caught up by the next sample."""
    rss = cpu = threads = 0.0
    for p in tree(proc):
        try:
            rss += p.memory_info().rss / 2 ** 20
            if p.pid not in _seen:
                _seen[p.pid] = p
                p.cpu_percent(None)
            cpu += _seen[p.pid].cpu_percent(None)
            threads += p.num_threads()
        except psutil.Error:
            pass
    return {"rss": rss, "cpu": cpu, "threads": threads}


_seen: Dict[int, psutil.Process] = {}


def last_stats(path: str) -> Dict:
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().strip().splitlines()
        return json.loads(lines[-1]) if lines else {}
    except (OSError, ValueError):
        return {}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    app_args: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, app_args = argv[:i], argv[i + 1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--port", type=int, default=5790)
    args = parser.parse_args(argv)

    print(machine())
    if psutil.virtual_memory().available / 2 ** 20 < MIN_FREE_MB + 256:
        sys.exit("refusing to start: free RAM is already near the %.0f MB floor"
                 % MIN_FREE_MB)
    take_lock()
    work = tempfile.mkdtemp(prefix="soak_")
    stats_path = os.path.join(work, "stats.jsonl")
    sim = gcs = None
    killed = ""
    rows: List[Dict] = []
    try:
        sim = psutil.Popen([PY, os.path.join(ROOT, "packet_sim.py"), "--port",
                            str(args.port), "--rate", str(args.rate),
                            "--origin", "26.720028", "84.304583"],
                           stdout=open(os.devnull, "w"),
                           stderr=open(os.devnull, "w"), cwd=ROOT)
        time.sleep(1.5)
        env = dict(os.environ)
        env["SOAK_REPORT_AT_S"] = str(max(70.0, args.minutes * 60 - 20))
        gcs = psutil.Popen([PY, os.path.join(HERE, "soak_app.py"), "--port",
                            str(args.port), "--stats", stats_path, "--log-dir",
                            os.path.join(work, "logs")] + app_args,
                           stdout=open(os.path.join(work, "gcs.out"), "w"),
                           stderr=open(os.path.join(work, "gcs.err"), "w"),
                           cwd=ROOT, env=env)
        for p in (gcs, sim):
            measure(p)
        psutil.cpu_percent(None)
        new_file = not os.path.exists(LOG)
        log = open(LOG, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(log, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        start = time.monotonic()
        end = start + args.minutes * 60.0
        while time.monotonic() < end:
            time.sleep(SAMPLE_S)
            if gcs.poll() is not None:
                killed = "GCS exited by itself (code %s)" % gcs.returncode
                break
            g, m = measure(gcs), measure(sim)
            gcs_rss, gcs_cpu, threads = g["rss"], g["cpu"], int(g["threads"])
            sim_rss, sim_cpu = m["rss"], m["cpu"]
            free = psutil.virtual_memory().available / 2 ** 20
            s = last_stats(stats_path)
            row = dict(label=args.label,
                       t_s=round(time.monotonic() - start, 1),
                       gcs_rss_mb=round(gcs_rss, 1), gcs_cpu_pct=round(gcs_cpu, 1),
                       sim_rss_mb=round(sim_rss, 1), sim_cpu_pct=round(sim_cpu, 1),
                       free_mb=round(free), system_cpu_pct=psutil.cpu_percent(None),
                       gcs_threads=threads, packets=s.get("packets", ""),
                       late_p95_ms=s.get("late_p95_ms", ""),
                       late_max_ms=s.get("late_max_ms", ""),
                       gc_objects=s.get("gc_objects", ""),
                       track_points=s.get("track_points", ""),
                       event_log_blocks=s.get("event_log_blocks", ""), killed="")
            if gcs_rss > GCS_RSS_LIMIT_MB:
                killed = "GCS RSS %.0f MB > %.0f MB" % (gcs_rss, GCS_RSS_LIMIT_MB)
            elif free < MIN_FREE_MB:
                killed = "free RAM %.0f MB < %.0f MB" % (free, MIN_FREE_MB)
            row["killed"] = killed
            writer.writerow(row)
            log.flush()
            rows.append(row)
            if killed:
                break
        log.close()
    finally:
        kill_tree(gcs)
        kill_tree(sim)
        try:
            os.remove(LOCK)
        except OSError:
            pass

    summarise(args.label, rows, killed, work)
    return 1 if killed else 0


def summarise(label: str, rows: List[Dict], killed: str, work: str) -> None:
    print("\n== %s ==" % label)
    if killed:
        print("STOPPED: %s" % killed)
    if not rows:
        print("no samples")
        return
    marks = []
    for minute in range(0, 6):
        target = max(SAMPLE_S, minute * 60.0)
        near = min(rows, key=lambda r: abs(r["t_s"] - target))
        if abs(near["t_s"] - target) <= SAMPLE_S:
            marks.append((minute, near))
    print("RSS (MB) at minute: " + "  ".join("%d:%6.1f" % (m, r["gcs_rss_mb"])
                                            for m, r in marks))
    after = [r for r in rows if r["t_s"] >= 60.0]
    if len(after) >= 2:
        span = (after[-1]["t_s"] - after[0]["t_s"]) / 60.0
        rate = (after[-1]["gcs_rss_mb"] - after[0]["gcs_rss_mb"]) / span
        print("growth after the first minute: %+.2f MB/min" % rate)
    cpu = [r["gcs_cpu_pct"] for r in rows[1:]] or [0]
    print("GCS CPU (%% of one core): mean %.1f  max %.1f | simulator mean %.1f"
          % (sum(cpu) / len(cpu), max(cpu),
             sum(r["sim_cpu_pct"] for r in rows) / len(rows)))
    late = [r["late_p95_ms"] for r in rows if r["late_p95_ms"] != ""]
    if late:
        print("GUI lateness p95 (ms): first %.0f  last %.0f  max %.0f; "
              "packets %s" % (late[0], late[-1], max(late), rows[-1]["packets"]))
    print("free RAM: start %d MB, end %d MB; threads %s -> %s"
          % (rows[0]["free_mb"], rows[-1]["free_mb"], rows[0]["gcs_threads"],
             rows[-1]["gcs_threads"]))
    print("work dir: %s" % work)


if __name__ == "__main__":
    sys.exit(main())

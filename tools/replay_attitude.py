#!/usr/bin/env python3
"""
tools/replay_attitude.py
========================

Replays known rotations through the attitude estimator exactly as this station
receives them, and measures what the operator would see -- old behaviour
against new.

The link: one frame every 300 ms, routed by packet_count % 3 (1 -> RX1,
2 -> RX2, 0 -> LoRa, not ingested), so frames 1, 2, 4, 5, 7, ... arrive and
the gaps alternate 300 ms and 600 ms. Each arrival gets 50 ms of latency plus
0-30 ms of jitter; the display is sampled at 30 Hz of wall time.

Three displays are compared:

  old         the pre-change estimator (kept verbatim in
              tools/test_attitude_smooth.py), drawn when a packet lands
  new raw     the new estimator, SMOOTH off: the newest packet at once
  new smooth  the new estimator through AttitudePlayback (the default)

Reported per case:

  angle err %   integrated rotation against a 1 kHz ground-truth integration
                of the same underlying rate (sum of step angles, so whole
                turns count), as a percentage of the true total
  final err     orientation error at the last packet, degrees
  speed         displayed angular speed per 30 Hz frame once the playback
                delay has filled: min, max, and max/min (the stutter index)
  max jump      largest displayed angle change between consecutive frames,
                and its ratio to the expected step (true rate / 30)

    python tools/replay_attitude.py            # synthetic cases + newest log
    python tools/replay_attitude.py --csv logs/Flight_X.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
import sys
from typing import Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import attitude_widget as aw  # noqa: E402

FRAME_S = 0.3
RENDER_HZ = 30.0
LATENCY_S = 0.05
JITTER_S = 0.03
GRAVITY = 9.80665
TRUTH_HZ = 1000.0

RateFn = Callable[[float], np.ndarray]


def truth_attitudes(rate_fn: RateFn, times: List[float]) -> List[np.ndarray]:
    """Ground truth at *times*: exact 1 kHz integration of *rate_fn*."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    t = 0.0
    out = []
    step = 1.0 / TRUTH_HZ
    for target in times:
        while t < target - 1e-12:
            h = min(step, target - t)
            omega = np.radians(rate_fn(t + 0.5 * h))
            q = aw.quat_normalise(aw.quat_multiply(q, aw.quat_from_rotvec(omega * h)))
            t += h
        out.append(q.copy())
    return out


def truth_total_angle(rate_fn: RateFn, t_end: float, start: float = 0.0) -> float:
    """Total rotation angle (deg) from *start* to *t_end*, whole turns counted."""
    n = int(round((t_end - start) * TRUTH_HZ))
    total = 0.0
    for i in range(n):
        t = start + (i + 0.5) / TRUTH_HZ
        total += float(np.linalg.norm(rate_fn(t))) / TRUTH_HZ
    return total


def simo_packets(rate_fn: RateFn, duration_s: float, seed: int = 1,
                 accel_fn: Optional[Callable[[float], np.ndarray]] = None):
    """``(mission_t, arrival_wall, gyro, accel)`` for the frames that arrive."""
    rng = random.Random(seed)
    packets = []
    k = 1
    while k * FRAME_S <= duration_s + 1e-9:
        t = k * FRAME_S
        if k % 3 != 0:                         # every third frame goes to LoRa
            gyro = rate_fn(t)
            accel = accel_fn(t) if accel_fn else np.array([0.0, 0.0, GRAVITY])
            packets.append((t, t + LATENCY_S + rng.uniform(0.0, JITTER_S),
                            gyro, accel))
        k += 1
    return packets


def _legacy_class():
    from test_attitude_smooth import LegacyAttitudeEstimator
    return LegacyAttitudeEstimator


def replay(rate_fn: RateFn, duration_s: float, mode: str,
           packets=None) -> Dict[str, object]:
    """Run one display mode; return frames and estimator snapshots."""
    if packets is None:
        packets = simo_packets(rate_fn, duration_s)
    if mode == "old":
        estimator = _legacy_class()()
    else:
        estimator = aw.AttitudeEstimator()
    playback = aw.AttitudePlayback()

    frames = []                                # (wall, q shown)
    steps = []                                 # estimator q after each packet
    index = 0
    end_wall = packets[-1][1] + aw.PLAYBACK_DELAY_S + 1.0
    n_frames = int(end_wall * RENDER_HZ) + 1
    for f in range(n_frames):
        wall = f / RENDER_HZ
        while index < len(packets) and packets[index][1] <= wall:
            t, _arrival, gyro, accel = packets[index]
            estimator.update(tuple(gyro), tuple(accel), t)
            steps.append((t, estimator.q.copy()))
            index += 1
        if not steps:
            continue
        if mode == "new smooth":
            shown = playback.sample(estimator, wall)
        else:
            shown = estimator.q.copy()
        frames.append((wall, shown))
    return {"frames": frames, "steps": steps, "packets": packets,
            "estimator": estimator}


def metrics(run: Dict[str, object], rate_fn: RateFn,
            true_speed: Optional[float]) -> Dict[str, float]:
    steps = run["steps"]
    frames = run["frames"]
    packets = run["packets"]

    # Integrated angle, whole turns counted: each step is < 180 deg here.
    est_total = sum(aw.quat_angle_deg(a[1], b[1]) for a, b in zip(steps, steps[1:]))
    first_t, last_t = steps[0][0], steps[-1][0]
    true_total = truth_total_angle(rate_fn, last_t, start=first_t)
    # Final orientation error, both started from identity at the first packet.
    truth = truth_attitudes(lambda t: rate_fn(t + first_t), [last_t - first_t])[0]
    final_err = aw.quat_angle_deg(steps[-1][1], truth)

    # Displayed motion, once the playback delay has filled and while packets
    # are still arriving.
    fill = packets[0][1] + aw.PLAYBACK_DELAY_S + 2 * FRAME_S
    last_arrival = packets[-1][1]
    speeds, jumps = [], []
    for (w0, q0), (w1, q1) in zip(frames, frames[1:]):
        if w0 < fill or w1 > last_arrival:
            continue
        step = aw.quat_angle_deg(q0, q1)
        jumps.append(step)
        speeds.append(step * RENDER_HZ)
    out = {
        "angle_err_pct": (100.0 * (est_total - true_total) / true_total
                          if true_total > 1e-9 else float("nan")),
        "est_total": est_total, "true_total": true_total,
        "final_err_deg": final_err,
        "speed_min": min(speeds), "speed_max": max(speeds),
        "stutter": (max(speeds) / min(speeds)) if min(speeds) > 1e-9
        else float("inf"),
        "max_jump": max(jumps),
    }
    if true_speed:
        expected = true_speed / RENDER_HZ
        out["jump_ratio"] = out["max_jump"] / expected
    return out


def constant_rate(dps: float, axis) -> RateFn:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return lambda t: dps * axis


def sinusoid(amplitude_dps: float, hz: float) -> RateFn:
    return lambda t: np.array([0.0, 0.0, amplitude_dps * math.sin(2 * math.pi * hz * t)])


CASES = [
    ("30 deg/s about z", constant_rate(30.0, (0, 0, 1)), 30.0),
    ("90 deg/s about z", constant_rate(90.0, (0, 0, 1)), 90.0),
    ("90 deg/s tilted axis", constant_rate(90.0, (0.3, 0.4, 0.866)), 90.0),
    ("1 Hz sine, 60 deg/s amplitude", sinusoid(60.0, 1.0), None),
]
MODES = ("old", "new raw", "new smooth")
DURATION_S = 20.0


def acceptance(m: Dict[str, float], true_speed: float) -> List[str]:
    """Failures against the brief's acceptance; empty means pass."""
    fails = []
    if not (0.9 * true_speed <= m["speed_min"] and m["speed_max"] <= 1.1 * true_speed):
        fails.append("speed outside +-10%")
    if abs(m["angle_err_pct"]) >= 2.0:
        fails.append("angle error >= 2%")
    if m["jump_ratio"] > 1.5:
        fails.append("frame jump > 1.5x step")
    return fails


def _row(mode: str, m: Dict[str, float], true_speed: Optional[float]) -> str:
    text = ("  %-11s angle err %+7.2f%%  final err %6.2f deg  "
            "speed %6.1f..%6.1f deg/s  stutter %s  max jump %6.2f deg"
            % (mode, m["angle_err_pct"], m["final_err_deg"], m["speed_min"],
               m["speed_max"],
               ("%6.2f" % m["stutter"]) if math.isfinite(m["stutter"]) else "   inf",
               m["max_jump"]))
    if true_speed:
        text += " (%.1fx step)" % m["jump_ratio"]
        if mode == "new smooth":
            fails = acceptance(m, true_speed)
            text += "   " + ("PASS" if not fails else "FAIL: " + ", ".join(fails))
    return text


def run_synthetic() -> bool:
    ok = True
    for name, rate_fn, true_speed in CASES:
        print()
        print("%s  (%.0f s, %s)" % (name, DURATION_S,
              "true speed %.0f deg/s, step %.2f deg/frame"
              % (true_speed, true_speed / RENDER_HZ) if true_speed
              else "rate changes sign every 0.5 s"))
        for mode in MODES:
            m = metrics(replay(rate_fn, DURATION_S, mode), rate_fn, true_speed)
            print(_row(mode, m, true_speed))
            if true_speed and mode == "new smooth" and acceptance(m, true_speed):
                ok = False
        if not true_speed:
            print("  -> about 2.2 samples/s of a 1 Hz rate: under-sampled. No "
                  "integrator or display can recover the true motion from this; "
                  "the numbers show the sampling limit, not a defect.")
    return ok


def load_csv_rate(path: str):
    """Gyro and accel from a flight CSV as functions of time from its start."""
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append([float(row[k]) for k in (
                    "mission_time_s", "gyro_x", "gyro_y", "gyro_z",
                    "acc_x", "acc_y", "acc_z")])
            except (KeyError, TypeError, ValueError):
                continue
    data = np.array(sorted(rows))
    # Keep strictly increasing times (two radios can log the same packet).
    keep = np.concatenate(([True], np.diff(data[:, 0]) > 1e-9))
    data = data[keep]
    t = data[:, 0] - data[0, 0]

    def gyro(x):
        return np.array([np.interp(x, t, data[:, i]) for i in (1, 2, 3)])

    def accel(x):
        return np.array([np.interp(x, t, data[:, i]) for i in (4, 5, 6)])

    return gyro, accel, float(t[-1]), len(t)


def run_csv(path: str) -> None:
    gyro, accel, span, n = load_csv_rate(path)
    print()
    print("Logged session %s" % os.path.basename(path))
    print("  %d samples over %.1f s, linearly interpolated as the underlying "
          "rate; resampled at 300 ms with every third frame removed."
          % (n, span))
    packets = simo_packets(gyro, span - FRAME_S, accel_fn=accel)
    for mode in MODES:
        run = replay(gyro, span, mode, packets=packets)
        m = metrics(run, gyro, None)
        print("  %-11s angle err %+7.2f%%  final err %6.2f deg  "
              "max frame jump %6.2f deg"
              % (mode, m["angle_err_pct"], m["final_err_deg"], m["max_jump"]))
    print("  (a logged session is not constant-rate: no speed/stutter "
          "acceptance applies; it shows the error on real motion)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", help="flight CSV (default: newest logs/Flight_*.csv)")
    args = parser.parse_args()
    ok = run_synthetic()
    path = args.csv
    if not path:
        logs = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "logs",
                                             "Flight_*.csv")),
                      key=os.path.getmtime)
        path = logs[-1] if logs else None
    if path:
        run_csv(path)
    else:
        print("\nNo logs/Flight_*.csv found: logged-session replay skipped.")
    print()
    print("ACCEPTANCE (constant-rate cases, new smooth): %s"
          % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

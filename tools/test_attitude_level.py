"""
tools/test_attitude_level.py
============================

Roll and pitch come back to level after a tilt, even when the gyro samples
miss the motion: the accelerometer levelling, its gates, the learned gravity
reference, RESET ATTITUDE, and a replay of the real tilt test.

Synthetic flights use this station's packet pattern: a frame every 300 ms,
every third one lost to LoRa.

    python -m pytest tools/test_attitude_level.py -v
"""

from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import attitude_widget as aw  # noqa: E402

G = aw.GRAVITY_MS2
TILT_CSV = os.path.join(ROOT, "logs", "Flight_TEST001_2026-10-28_014839.csv")


def simo_times(duration_s):
    return [k * 0.3 for k in range(1, int(duration_s / 0.3) + 1) if k % 3 != 0]


def tilt_profile(t_up, t_down, angle=45.0, still=20.0, hold=3.0):
    """Roll (deg) and roll rate (deg/s) of a board tilted about x and back.
    Smooth (cosine) moves of *t_up* / *t_down* seconds."""
    t1 = still
    t2 = t1 + t_up
    t3 = t2 + hold
    t4 = t3 + t_down

    def at(t):
        if t < t1:
            return 0.0, 0.0
        if t < t2:
            u = (t - t1) / t_up
            return (angle * (1 - math.cos(math.pi * u)) / 2,
                    angle * math.pi * math.sin(math.pi * u) / (2 * t_up))
        if t < t3:
            return angle, 0.0
        if t < t4:
            u = (t - t3) / t_down
            return (angle * (1 + math.cos(math.pi * u)) / 2,
                    -angle * math.pi * math.sin(math.pi * u) / (2 * t_down))
        return 0.0, 0.0
    return at, t4


def body_accel(roll_deg, pitch_deg=0.0, g=G):
    """Specific force a still body reads at this roll and pitch."""
    q = aw.quat_from_euler_deg(roll_deg, pitch_deg, 0.0)
    return tuple(aw.quat_to_matrix(q).T @ np.array([0.0, 0.0, g]))


def fly(estimator, profile, end, bias=(0.0, 0.0, 0.0), g=G, settle=15.0):
    for t in simo_times(end + settle):
        roll, rate = profile(t)
        gyro = (rate + bias[0], bias[1], bias[2])
        estimator.update(gyro, body_accel(roll, g=g), t, 1.7e9 + t)
    return estimator.euler_deg


# ---------------------------------------------------------------------------
# Tilt and return, 2 of 3 packets
# ---------------------------------------------------------------------------

#: Fast: 0.4 s moves, the return placed wholly between the packets at 23.1 s
#: and 23.7 s (23.4 s goes to LoRa) -- as on the real board, where the
#: return to flat fell between two packets and the gyro never saw it.
FAST = dict(t_up=0.4, t_down=0.4, hold=2.8)
SLOW = dict(t_up=4.0, t_down=4.0, hold=3.0)


@pytest.mark.parametrize("label,shape", [("fast", FAST), ("slow", SLOW)])
@pytest.mark.parametrize("scale", [1.0, 0.5])      # 0.5: the real board
def test_tilt_45_and_back_returns_to_level(label, shape, scale):
    profile, end = tilt_profile(**shape)
    roll, pitch, _yaw = fly(aw.AttitudeEstimator(), profile, end, g=G * scale)
    assert abs(roll) < 2.0 and abs(pitch) < 2.0, (label, roll, pitch)


def test_pure_gyro_does_not_return_to_level_after_a_fast_tilt():
    """The problem being fixed: the sparse samples miss most of a fast move."""
    profile, end = tilt_profile(**FAST)
    roll, _pitch, _yaw = fly(aw.AttitudeEstimator(use_accel_reference=False),
                             profile, end)
    assert abs(roll) > 10.0, roll


def test_levelling_never_touches_yaw():
    est = aw.AttitudeEstimator(auto_bias=False)
    est.q = aw.quat_from_euler_deg(20.0, -10.0, 37.0)   # tilted, yaw 37
    est.update((0, 0, 0), body_accel(0.0), 0.0, 1.7e9)
    for t in simo_times(30.0):
        est.update((0, 0, 0), body_accel(0.0), t, 1.7e9 + t)
    roll, pitch, yaw = est.euler_deg
    assert abs(roll) < 0.5 and abs(pitch) < 0.5
    assert yaw == pytest.approx(37.0, abs=1e-6)


def test_levelling_time_constant_is_a_few_seconds_at_any_packet_rate():
    def roll_after(step, seconds=4.0):
        est = aw.AttitudeEstimator(auto_bias=False)
        est.q = aw.quat_from_euler_deg(30.0, 0.0, 0.0)
        t = 0.0
        while t <= seconds + 1e-9:
            est.update((0, 0, 0), body_accel(0.0), t, 1.7e9 + t)
            t += step
        return est.euler_deg[0], est.timeline_s
    # 30 deg decays as exp(-elapsed / tau) at 20 Hz and at this station's
    # rate alike: the gain is per unit time, not per packet.
    for step in (0.05, 0.45):
        roll, elapsed = roll_after(step)
        assert roll == pytest.approx(30.0 * math.exp(-elapsed / aw.LEVEL_TAU_S),
                                     abs=0.05), step
    assert 1.0 <= aw.LEVEL_TAU_S <= 5.0                   # "a few seconds"


# ---------------------------------------------------------------------------
# Bias
# ---------------------------------------------------------------------------

def test_constant_gyro_bias_on_a_still_board(capsys):
    # The real board measured +0.30 / -0.40 / -0.88 deg/s; this is about 2.5x
    # that on x and y, more on z. Until the bias is learned it leans roll and
    # pitch by about bias x LEVEL_TAU_S seconds: 1.0 deg here, 2.0 deg at a
    # 2 deg/s offset.
    bias = (1.0, -0.75, 3.0)
    est = aw.AttitudeEstimator()
    worst = 0.0
    for t in simo_times(60.0):
        est.update(bias, body_accel(0.0), t, 1.7e9 + t)
        roll, pitch, _ = est.euler_deg
        if t > 1.0:
            worst = max(worst, abs(roll), abs(pitch))
    roll, pitch, yaw = est.euler_deg
    assert worst < 2.0, worst
    assert est.bias_settled
    assert est.bias == pytest.approx(bias, abs=0.4)
    with capsys.disabled():
        print("\n    bias test: roll %+.2f pitch %+.2f, worst %.2f deg; yaw "
              "drift %+.1f deg over 60 s (bias %+.2f deg/s on z, no "
              "magnetometer)" % (roll, pitch, worst, yaw, bias[2]))
    # Yaw is not corrected by anything; it drifts until the bias is learned.
    assert abs(yaw) > 1.0


def test_yaw_drift_is_reported_in_the_panel(qapp):
    widget = aw.AttitudeWidget(settings_path=None, use_gl=False)
    widget.redraw(force=True)
    tip = widget.rpy_label.toolTip()
    assert "Yaw drifts, no magnetometer" in tip
    assert "levelling" in tip.lower()
    assert "no magnetometer" in widget.ref_btn.toolTip().lower()
    widget.close()


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _level_still(est, seconds=10.0, start=0.0):
    for t in simo_times(seconds):
        est.update((0, 0, 0), body_accel(0.0), start + t, 1.7e9 + start + t)
    return start + seconds


@pytest.mark.parametrize("label,accel", [
    ("2 g spike pointing 40 deg off", body_accel(40.0, g=2 * G)),
    ("free fall", body_accel(40.0, g=0.05 * G)),
    ("boost 7 g", body_accel(40.0, g=7 * G)),
])
def test_correction_does_not_pull_outside_the_gravity_window(label, accel):
    est = aw.AttitudeEstimator()
    t = _level_still(est)
    before = est.euler_deg
    for k in range(1, 8):                            # ~3 s of it
        est.update((0, 0, 0), accel, t + 0.45 * k, 1.7e9 + t + 0.45 * k)
    after = est.euler_deg
    assert abs(after[0] - before[0]) < 0.5, (label, before, after)
    assert abs(after[1] - before[1]) < 0.5, (label, before, after)
    assert not est.accel_valid


def test_correction_does_not_pull_during_a_hard_manoeuvre():
    """1 g, but turning at 40 deg/s: the gyro alone."""
    est = aw.AttitudeEstimator()
    t = _level_still(est)
    for k in range(1, 8):
        est.update((0.0, 0.0, 40.0), body_accel(30.0), t + 0.45 * k,
                   1.7e9 + t + 0.45 * k)
        assert not est.accel_valid
    assert abs(est.euler_deg[0]) < 0.5


def test_gravity_reference_is_learned_from_the_board():
    est = aw.AttitudeEstimator()
    _level_still(est, seconds=5.0)
    assert est.gravity_settled and est.gravity_ref == pytest.approx(G, rel=1e-3)
    half = aw.AttitudeEstimator()
    for t in simo_times(5.0):
        half.update((0.3, -0.4, -0.9), body_accel(3.0, g=0.5 * G), t, 1.7e9 + t)
    assert half.gravity_settled
    assert half.gravity_ref == pytest.approx(0.5 * G, rel=1e-3)
    # Not learned from free fall or from a moving board.
    moving = aw.AttitudeEstimator()
    for t in simo_times(5.0):
        moving.update((30.0, 0, 0), body_accel(10.0 * t), t, 1.7e9 + t)
    assert not moving.gravity_settled


# ---------------------------------------------------------------------------
# RESET ATTITUDE
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_reset_zeros_yaw_and_levels_from_the_accelerometer(qapp):
    widget = aw.AttitudeWidget(settings_path=None, use_gl=False)
    est = widget.estimator
    est.q = aw.quat_from_euler_deg(25.0, 5.0, 80.0)     # wrong, with yaw
    est.update((0, 0, 0), body_accel(10.0), 0.0, 1.7e9)  # board at roll 10
    widget.reset_btn.click()
    roll, pitch, yaw = est.euler_deg
    assert roll == pytest.approx(10.0, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6)
    # The panel shows it at once.
    assert widget.rpy_label.text().split()[:2] == ["R", "+10.0"]
    widget.close()


def test_reset_with_no_reading_is_exactly_level(qapp):
    widget = aw.AttitudeWidget(settings_path=None, use_gl=False)
    widget.reset_btn.click()
    assert np.array_equal(widget.estimator.q, [1.0, 0.0, 0.0, 0.0])
    widget.close()


# ---------------------------------------------------------------------------
# The real tilt test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TILT_CSV),
                    reason="the tilt-test log is not on this machine (logs/ "
                           "is not committed)")
def test_real_tilt_test_returns_to_level(capsys):
    rows = sorted(csv.DictReader(open(TILT_CSV, newline="", encoding="utf-8")),
                  key=lambda r: int(r["packet_count"]))
    t0 = float(rows[0]["gs_recv_epoch"])

    def replay(levelling):
        est = aw.AttitudeEstimator(use_accel_reference=levelling)
        for r in rows:
            # As the app feeds it: mission time, receive time as fallback.
            est.update(tuple(float(r[k]) for k in ("gyro_x", "gyro_y", "gyro_z")),
                       tuple(float(r[k]) for k in ("acc_x", "acc_y", "acc_z")),
                       float(r["mission_time_s"]),
                       float(r["gs_recv_epoch"]) - t0)
        return est.euler_deg

    level = np.mean([aw.accel_roll_pitch_deg(
        [float(r[k]) for k in ("acc_x", "acc_y", "acc_z")]) for r in rows[-25:]],
        axis=0)
    roll, pitch, yaw = replay(True)
    gyro_roll, gyro_pitch, _ = replay(False)
    with capsys.disabled():
        print("\n    real tilt test end error: roll %+.1f pitch %+.1f (gyro "
              "only: %+.1f %+.1f), yaw drift %+.1f deg"
              % (roll - level[0], pitch - level[1], gyro_roll - level[0],
                 gyro_pitch - level[1], yaw))
    assert abs(roll - level[0]) < 3.0
    assert abs(pitch - level[1]) < 3.0
    assert abs(gyro_roll - level[0]) > 10.0          # the reported problem

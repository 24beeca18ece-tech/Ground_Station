"""
tools/test_attitude_smooth.py
=============================

The attitude estimator and its smooth playback, under this station's real
packet pattern: one frame every 300 ms with every third going to LoRa, so
gaps alternate 300 ms and 600 ms.

Also holds :class:`LegacyAttitudeEstimator`, the estimator as it was before,
verbatim, which tools/replay_attitude.py measures the new one against.

    python -m pytest tools/test_attitude_smooth.py -v
"""

from __future__ import annotations

import json
import math
import os
import sys
import tracemalloc
from typing import Optional, Tuple

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import attitude_widget as aw  # noqa: E402
from attitude_widget import (  # noqa: E402
    BIAS_ACCEL_HI, BIAS_ACCEL_LO, BIAS_MAX_RATE_DPS, GRAVITY_MS2,
    quat_multiply, quat_normalise, quat_to_euler_deg, quat_to_matrix,
)

# The Mahony constants the legacy estimator below used. attitude_widget no
# longer has them (the attitude-level branch replaced that correction), so
# they are kept here at their old values for the verbatim copy.
MAHONY_KP = 1.6
ACCEL_GATE_LO = 0.75
ACCEL_GATE_HI = 1.25

# ---------------------------------------------------------------------------
# The estimator as it was before the attitude-smooth branch, kept VERBATIM
# (only renamed) as the reference the replay tool and the tests compare
# against. Do not "fix" it: its faults are the point.
# ---------------------------------------------------------------------------

LEGACY_MAX_DT_S = 0.25
LEGACY_BIAS_ALPHA = 0.02
LEGACY_BIAS_SETTLE_SAMPLES = 60


class LegacyAttitudeEstimator:
    """Quaternion attitude estimate driven by the telemetry gyro rates.

    Two modes:

    * **Pure gyro (default).**  Orientation is the integral of GYRO_X/Y/Z and
      nothing else.  Zero on all three axes holds the orientation perfectly
      still, and the rotation rate corresponds one to one with the transmitted
      body rates.  Yaw, roll and pitch all drift over time -- that is inherent
      to integrating a rate sensor, and the RESET button exists for it.
    * **Accelerometer reference (opt-in).**  Adds a Mahony proportional
      correction that pulls roll and pitch towards the measured gravity vector,
      bounding their drift.  This term comes from the accelerometer, so while it
      is enabled the model can move even with the gyro at zero.

    Pure computation, no Qt.  Kept separate from the widget so it can be unit
    tested and so a future firmware-side estimator could replace it without
    touching the rendering code.
    """

    __slots__ = ("q", "_last_t", "accel_valid", "samples", "corrected",
                 "use_accel_reference", "auto_bias", "bias", "bias_samples",
                 "at_rest")

    def __init__(self, use_accel_reference: bool = False,
                 auto_bias: bool = True) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_t: Optional[float] = None
        self.accel_valid = False   # was the last sample inside the gravity gate
        self.samples = 0
        self.corrected = 0
        #: When False the estimate is a pure integral of the telemetry gyro.
        self.use_accel_reference = bool(use_accel_reference)
        #: Subtract an auto-learned turn-on bias from the raw gyro.
        self.auto_bias = bool(auto_bias)
        #: Running gyro bias estimate, deg/s, one per axis.
        self.bias = np.zeros(3)
        self.bias_samples = 0
        self.at_rest = False

    def reset(self) -> None:
        """Re-zero the estimate (clears accumulated drift).

        Deliberately leaves :attr:`use_accel_reference` alone -- resetting the
        orientation should not silently change the filter mode.
        """
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_t = None
        self.accel_valid = False
        self.samples = 0
        self.corrected = 0
        # Re-arm bias learning: RESET is what an operator presses between bench
        # runs, which is exactly when a fresh zero should be taken.
        self.bias = np.zeros(3)
        self.bias_samples = 0
        self.at_rest = False

    @property
    def bias_settled(self) -> bool:
        """True once enough stationary samples have been averaged to trust it."""
        return self.bias_samples >= LEGACY_BIAS_SETTLE_SAMPLES

    def update(self, gyro_dps: Tuple[float, float, float],
               accel_ms2: Tuple[float, float, float],
               timestamp_s: float) -> None:
        """Advance the estimate using one telemetry sample.

        *timestamp_s* is the packet mission time; dt is derived from successive
        packets so the estimate stays correct regardless of GUI frame rate.
        """
        gx, gy, gz = (v if math.isfinite(v) else 0.0 for v in gyro_dps)

        if self._last_t is None or not math.isfinite(timestamp_s):
            self._last_t = timestamp_s if math.isfinite(timestamp_s) else None
            return
        dt = timestamp_s - self._last_t
        self._last_t = timestamp_s
        if dt <= 0.0:
            # Non-monotonic or duplicated timestamp: nothing sensible to do.
            return
        dt = min(dt, LEGACY_MAX_DT_S)

        self.samples += 1

        # Body rate straight from the telemetry, and nothing else.  In the
        # default (pure gyro) mode this is the *only* term that reaches the
        # integrator, so zero on all three axes means the orientation is held
        # exactly still and the rotation rate tracks GYRO_X/Y/Z one to one.
        raw = np.array([gx, gy, gz])

        ax, ay, az = (v if math.isfinite(v) else 0.0 for v in accel_ms2)
        accel = np.array([ax, ay, az])
        norm = float(np.linalg.norm(accel))
        self.accel_valid = False

        # --- turn-on bias estimation -----------------------------------------
        # Learn only while the vehicle is demonstrably at rest: gravity-only
        # specific force AND a body rate small enough that it cannot be real
        # motion. In flight neither holds, so the estimate freezes at whatever
        # was learned on the pad -- which is the value that matters.
        g_ratio = norm / GRAVITY_MS2 if norm > 1e-6 else 0.0
        self.at_rest = (
            BIAS_ACCEL_LO <= g_ratio <= BIAS_ACCEL_HI
            and float(np.max(np.abs(raw - self.bias))) < BIAS_MAX_RATE_DPS
        )
        if self.auto_bias and self.at_rest:
            self.bias += LEGACY_BIAS_ALPHA * (raw - self.bias)
            self.bias_samples += 1

        corrected_dps = raw - self.bias if self.auto_bias else raw
        omega = np.radians(corrected_dps)

        # --- optional gravity reference (Mahony proportional correction) -----
        # Off by default.  This term is derived from the accelerometer, not the
        # gyroscope, so while it is enabled the model can rotate even when the
        # gyro reads zero -- that is the point of it (it pulls roll/pitch back
        # towards the measured gravity vector) but it breaks the 1:1
        # correspondence with GYRO_X/Y/Z, so the operator opts in.
        if self.use_accel_reference and norm > 1e-6:
            if ACCEL_GATE_LO <= g_ratio <= ACCEL_GATE_HI:
                # Measured gravity direction in body frame.
                measured = accel / norm
                # Gravity direction the current estimate predicts, in body
                # frame: world +Z rotated by the inverse of q, which for a
                # rotation matrix is its transpose.
                predicted = quat_to_matrix(self.q).T @ np.array([0.0, 0.0, 1.0])
                # Rotation error as a vector; small-angle => cross product.
                error = np.cross(measured, predicted)
                omega = omega + MAHONY_KP * error
                self.accel_valid = True
                self.corrected += 1

        # --- quaternion integration ----------------------------------------
        # q_dot = 0.5 * q (x) [0, omega]
        q_dot = 0.5 * quat_multiply(self.q, np.array([0.0, *omega]))
        self.q = quat_normalise(self.q + q_dot * dt)

    # -- outputs -----------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        return quat_to_matrix(self.q)

    @property
    def euler_deg(self) -> Tuple[float, float, float]:
        return quat_to_euler_deg(self.q)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

G = (0.0, 0.0, GRAVITY_MS2)


def yaw_deg(q):
    return quat_to_euler_deg(q)[2]


def simo_times(n_frames, start=0.3):
    """Mission times of the frames that reach this station."""
    return [k * 0.3 for k in range(1, n_frames + 1) if k % 3 != 0]


def feed(est, rate_dps, times, accel=G):
    for t in times:
        est.update(tuple(rate_dps), accel, t)


class FakePacket:
    def __init__(self, t, gyro=(0.0, 0.0, 90.0), recv=None):
        self.mission_time_s = t
        self.gs_recv_epoch = 1.7e9 + t if recv is None else recv
        self.gyro_x, self.gyro_y, self.gyro_z = gyro
        self.acc_x, self.acc_y, self.acc_z = G


# ---------------------------------------------------------------------------
# Exponential map and integration
# ---------------------------------------------------------------------------

def test_exponential_map_90dps_for_600ms_is_exactly_54_degrees():
    est = aw.AttitudeEstimator(auto_bias=False)
    est.update((0.0, 0.0, 90.0), G, 0.0)
    est.update((0.0, 0.0, 90.0), G, 0.6)
    assert yaw_deg(est.q) == pytest.approx(54.0, abs=1e-9)


def test_quat_from_rotvec_half_turn_does_not_blow_up():
    q = aw.quat_from_rotvec(np.array([0.0, 0.0, math.pi]))
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)
    assert aw.quat_angle_deg(np.array([1.0, 0, 0, 0]), q) == pytest.approx(180.0)
    # A single 180-degree estimator step: 300 deg/s over 0.6 s.
    est = aw.AttitudeEstimator(auto_bias=False)
    est.update((0.0, 0.0, 300.0), G, 0.0)
    est.update((0.0, 0.0, 300.0), G, 0.6)
    assert np.all(np.isfinite(est.q))
    assert aw.quat_angle_deg(np.array([1.0, 0, 0, 0]), est.q) == pytest.approx(180.0)


def test_quaternion_stays_unit_length_over_100000_updates():
    est = aw.AttitudeEstimator(auto_bias=False)
    rate = (37.0, -91.0, 143.0)
    t = 0.0
    for i in range(100_000):
        t += 0.3 if i % 2 else 0.6
        est.update(rate, G, t)
    assert abs(float(np.linalg.norm(est.q)) - 1.0) < 1e-12


@pytest.mark.parametrize("rate", [(0.0, 0.0, 90.0), (30.0, 40.0, 86.6)])
def test_simo_gap_pattern_matches_ground_truth_within_2_percent(rate):
    from replay_attitude import truth_attitudes
    est = aw.AttitudeEstimator(auto_bias=False)
    times = simo_times(60)                    # 300, 600, 300, 600 ms gaps
    feed(est, rate, times)
    span = times[-1] - times[0]
    true_q = truth_attitudes(lambda t: np.array(rate), [span])[0]
    total = float(np.linalg.norm(rate)) * span
    error = aw.quat_angle_deg(est.q, true_q)
    assert error / total < 0.02


def test_simo_gap_pattern_old_estimator_under_integrates():
    """Documents cause (a): the 0.25 s clamp loses 17-58 % per step."""
    old = LegacyAttitudeEstimator(auto_bias=False)
    old.update((0.0, 0.0, 90.0), G, 0.3)
    old.update((0.0, 0.0, 90.0), G, 0.6)     # 300 ms gap
    # 22.5 = 90 * 0.25 from the clamp; the first-order step loses 0.3 more.
    assert yaw_deg(old.q) == pytest.approx(22.22, abs=0.01)     # not 27
    old.update((0.0, 0.0, 90.0), G, 1.2)     # 600 ms gap
    assert yaw_deg(old.q) == pytest.approx(44.4, abs=0.1)       # not 81


def test_trapezoid_uses_the_previous_rate_as_well():
    est = aw.AttitudeEstimator(auto_bias=False)
    est.update((0.0, 0.0, 0.0), G, 0.0)
    est.update((0.0, 0.0, 100.0), G, 0.6)
    # Mean of 0 and 100 deg/s over 0.6 s.
    assert yaw_deg(est.q) == pytest.approx(30.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------

def test_two_second_dropout_does_not_spin_the_model():
    est = aw.AttitudeEstimator(auto_bias=False)
    feed(est, (0.0, 0.0, 90.0), [0.0, 0.3, 0.9])   # last normal gap 600 ms
    before = yaw_deg(est.q)
    epoch = est.epoch
    est.update((0.0, 0.0, 90.0), G, 2.9)            # 2 s with no packets
    # Only the last normal interval (0.6 s) is integrated: 54, not 180.
    assert yaw_deg(est.q) - before == pytest.approx(54.0, abs=1e-6)
    assert est.dropouts == 1
    assert est.epoch == epoch + 1


def test_gaps_up_to_max_dt_are_integrated_in_full():
    for gap in (0.3, 0.6, 0.9, 1.2):
        est = aw.AttitudeEstimator(auto_bias=False)
        est.update((0.0, 0.0, 50.0), G, 10.0)
        est.update((0.0, 0.0, 50.0), G, 10.0 + gap)
        assert yaw_deg(est.q) == pytest.approx(50.0 * gap, abs=1e-6)


def test_stalled_mission_time_falls_back_to_receive_time():
    est = aw.AttitudeEstimator(auto_bias=False)
    est.update((0.0, 0.0, 90.0), G, 5.0, recv_s=100.0)
    est.update((0.0, 0.0, 90.0), G, 5.0, recv_s=100.3)       # did not advance
    assert yaw_deg(est.q) == pytest.approx(27.0, abs=1e-6)
    est.update((0.0, 0.0, 90.0), G, float("nan"), recv_s=100.9)  # not finite
    assert yaw_deg(est.q) == pytest.approx(81.0, abs=1e-6)
    # The playback time axis never runs backwards.
    times = [k[0] for k in est.keyframes]
    assert times == sorted(times)


def test_bias_learning_is_per_unit_time():
    """Cause (d): settling takes 5 s of rest whatever the packet rate."""
    bias = (0.0, 0.0, 2.0)
    for gaps in ((0.05,), (0.3, 0.6)):
        est = aw.AttitudeEstimator()
        t = 0.0
        est.update(bias, G, t)
        i = 0
        while not est.bias_settled:
            t += gaps[i % len(gaps)]
            i += 1
            est.update(bias, G, t)
        assert 5.0 <= est.rest_time_s < 5.0 + max(gaps)
        # alpha = 1 - exp(-dt/7.5): after ~5 s, 1 - exp(-5/7.5) ~ 49 % learned.
        assert est.bias[2] == pytest.approx(2.0 * (1 - math.exp(-t / 7.5)),
                                            rel=0.02)


def test_old_bias_learning_was_per_packet():
    old = LegacyAttitudeEstimator()
    t = 0.0
    while not old.bias_settled:
        t += 0.45
        old.update((0.0, 0.0, 2.0), G, t)
    # 60 packets at ~2.2/s: 27 s to "settle", against 3 s at 20 Hz.
    assert t > 25.0


def test_mahony_correction_does_not_depend_on_packet_rate():
    tilted = (0.0, GRAVITY_MS2 * math.sin(0.3), GRAVITY_MS2 * math.cos(0.3))

    def roll_after(step):
        est = aw.AttitudeEstimator(use_accel_reference=True, auto_bias=False)
        t = 0.0
        while t < 2.0 - 1e-9:
            est.update((0.0, 0.0, 0.0), tilted, t)
            t += step
        est.update((0.0, 0.0, 0.0), tilted, 2.0)
        return quat_to_euler_deg(est.q)[0]

    fast, slow = roll_after(0.05), roll_after(0.5)
    assert abs(fast) > 1.0                       # it does correct
    assert slow == pytest.approx(fast, abs=1.0)  # by the same amount


def test_reset_clears_history():
    est = aw.AttitudeEstimator()
    feed(est, (0.0, 0.0, 90.0), simo_times(9))
    assert len(est.keyframes) > 1
    epoch = est.epoch
    est.reset()
    assert len(est.keyframes) == 0
    assert est.epoch > epoch
    assert np.array_equal(est.q, [1.0, 0.0, 0.0, 0.0])


def test_keyframe_history_is_bounded():
    est = aw.AttitudeEstimator()
    feed(est, (0.0, 0.0, 90.0), simo_times(3000))
    assert len(est.keyframes) == aw.KEYFRAME_HISTORY


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def test_slerp_returns_the_exact_keyframes_at_the_boundaries():
    a = aw.quat_from_rotvec(np.array([0.1, 0.2, 0.3]))
    b = aw.quat_from_rotvec(np.array([-0.4, 0.5, 1.1]))
    assert np.array_equal(aw.quat_slerp(a, b, 0.0), a)
    assert np.array_equal(aw.quat_slerp(a, b, 1.0), b)
    est = aw.AttitudeEstimator(auto_bias=False)
    feed(est, (10.0, 20.0, 90.0), simo_times(9))
    for t, q, _ in est.keyframes:
        assert np.array_equal(aw.interpolate_keyframes(est.keyframes, t), q)


def test_interpolation_follows_a_step_of_more_than_half_a_turn():
    est = aw.AttitudeEstimator(auto_bias=False)
    est.update((0.0, 0.0, 400.0), G, 0.0)
    est.update((0.0, 0.0, 400.0), G, 0.6)        # 240 degrees in one step
    mid = aw.interpolate_keyframes(est.keyframes, 0.3)
    assert yaw_deg(mid) == pytest.approx(120.0, abs=1e-6)   # not -60


def test_playback_holds_and_never_extrapolates_when_packets_stop():
    est = aw.AttitudeEstimator(auto_bias=False)
    player = aw.AttitudePlayback()
    last_q = None
    for t in simo_times(12):
        est.update((0.0, 0.0, 90.0), G, t)
        last_q = est.q.copy()
        player.sample(est, t + 0.05)
    last_t = est.keyframes[-1][0]
    for extra in np.arange(0.0, 10.0, 1 / 30):
        shown = player.sample(est, last_t + 0.05 + extra)
        assert player.clock <= last_t
    assert np.array_equal(shown, last_q)


def test_playback_runs_delay_behind_and_corrects_drift_gently():
    est = aw.AttitudeEstimator(auto_bias=False)
    player = aw.AttitudePlayback()
    times = simo_times(300)
    i = 0
    wall = 0.3
    rates = set()
    previous_clock = None
    while wall < times[-1]:
        while i < len(times) and times[i] + 0.05 <= wall:
            est.update((0.0, 0.0, 90.0), G, times[i])
            i += 1
        if i:
            player.sample(est, wall)
            rates.add(round(player.rate, 6))
            if previous_clock is not None and player.jumps == 0:
                # No jump between frames: at most 5 % faster than wall time.
                assert player.clock - previous_clock <= 1.05 / 30 + 1e-9
            previous_clock = player.clock
        wall += 1 / 30
    lag = (times[i - 1] + (wall - 1 / 30 - times[i - 1] - 0.05)) - player.clock
    assert lag == pytest.approx(aw.PLAYBACK_DELAY_S, abs=0.1)
    assert rates <= {0.95, 1.0, 1.05}


def test_playback_corrects_an_offset_by_rate_not_jump():
    est = aw.AttitudeEstimator(auto_bias=False)
    player = aw.AttitudePlayback()
    times = simo_times(200)
    feed(est, (0.0, 0.0, 30.0), times[:3])
    player.sample(est, times[2] + 0.05)
    player.clock -= 0.5                       # knocked 0.5 s behind
    jumps = player.jumps
    wall = times[2] + 0.05
    i = 3
    seen_fast = False
    while i < len(times):
        wall += 1 / 30
        while i < len(times) and times[i] + 0.05 <= wall:
            est.update((0.0, 0.0, 30.0), G, times[i])
            i += 1
        before = player.clock
        player.sample(est, wall)
        assert player.clock - before <= 1.05 / 30 + 1e-9
        seen_fast |= player.rate > 1.0
    assert seen_fast
    assert player.jumps == jumps
    target = est.keyframes[-1][0] + (wall - player._arrival_wall) - aw.PLAYBACK_DELAY_S
    assert abs(target - player.clock) < aw.PLAYBACK_DRIFT_S


def test_playback_jumps_after_a_dropout():
    est = aw.AttitudeEstimator(auto_bias=False)
    player = aw.AttitudePlayback()
    for t in simo_times(9):
        est.update((0.0, 0.0, 90.0), G, t)
        player.sample(est, t + 0.05)
    est.update((0.0, 0.0, 90.0), G, 10.0)    # dropout
    player.sample(est, 10.05)
    assert player.jumps == 1
    assert player.clock == pytest.approx(10.0 - aw.PLAYBACK_DELAY_S, abs=1e-6)


# ---------------------------------------------------------------------------
# Replay acceptance (synthetic constant rates)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", [0, 1, 2])
def test_replay_constant_rate_meets_acceptance(case):
    from replay_attitude import CASES, acceptance, metrics, replay
    name, rate_fn, speed = CASES[case]
    m = metrics(replay(rate_fn, 12.0, "new smooth"), rate_fn, speed)
    assert acceptance(m, speed) == [], (name, m)


def test_replay_old_behaviour_stutters():
    from replay_attitude import CASES, metrics, replay
    name, rate_fn, speed = CASES[1]
    m = metrics(replay(rate_fn, 12.0, "old"), rate_fn, speed)
    assert m["angle_err_pct"] < -17.0          # clamp under-integration
    assert m["speed_min"] == 0.0               # stands still between packets


# ---------------------------------------------------------------------------
# The widget
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _drive(widget, clock, seconds, start=0.3, tail=1.0):
    """Packets in the SIMO pattern, frames at 30 Hz, for *seconds*."""
    times = [t for t in simo_times(int(seconds / 0.3) + 3, start) if t <= seconds]
    i = 0
    frames = 0
    clock.t = start
    while clock.t < seconds + tail:
        while i < len(times) and times[i] + 0.05 <= clock.t:
            widget.on_packet(FakePacket(times[i]))
            i += 1
        widget.redraw()
        frames += 1
        clock.t += 1 / 30
    return frames


@pytest.mark.parametrize("use_gl", [True, False])
@pytest.mark.parametrize("smooth", [True, False])
def test_widget_receives_packets_and_renders(qapp, tmp_path, use_gl, smooth):
    widget = aw.AttitudeWidget(settings_path=str(tmp_path / "none.json"),
                               use_gl=use_gl)
    if not use_gl:
        assert widget.fallback is not None and not widget.gl_available
    widget.smooth_check.setChecked(smooth)
    clock = FakeClock()
    widget.clock = clock
    _drive(widget, clock, 6.0, tail=0.0)    # stop while packets still flow
    roll, pitch, yaw = quat_to_euler_deg(widget.shown_q)
    # The readout is the displayed attitude, which with SMOOTH lags the
    # newest packet by about the playback delay (90 deg/s * 0.8 s = 72 deg).
    assert widget.rpy_label.text() == "R %+7.1f  P %+7.1f  Y %+7.1f" % (roll, pitch, yaw)
    lag = aw.quat_angle_deg(widget.shown_q, widget.estimator.q)
    if smooth:
        assert 40.0 < lag < 100.0, lag
    else:
        assert lag == 0.0
    widget.close()


def test_widget_draws_continuously_with_smooth_and_holds_without_packets(qapp, tmp_path):
    widget = aw.AttitudeWidget(settings_path=None, use_gl=False)
    clock = FakeClock()
    widget.clock = clock
    before = widget.frames_drawn
    frames = _drive(widget, clock, 6.0)
    drawn = widget.frames_drawn - before
    # Continuous: nearly every 30 Hz tick draws, not one per packet (~13).
    assert drawn > 0.7 * frames
    # Packets stopped: once the playback reaches the last keyframe it holds,
    # and nothing more is drawn.
    for _ in range(60):
        clock.t += 1 / 30
        widget.redraw()
    held = widget.frames_drawn
    for _ in range(60):
        clock.t += 1 / 30
        widget.redraw()
    assert widget.frames_drawn == held
    assert np.array_equal(widget.shown_q, widget.estimator.q)
    widget.close()


def test_widget_memory_is_bounded_at_30hz(qapp):
    widget = aw.AttitudeWidget(settings_path=None, use_gl=False)
    clock = FakeClock()
    widget.clock = clock
    _drive(widget, clock, 20.0)
    tracemalloc.start()
    _drive(widget, clock, 20.0, start=20.3)
    base, _ = tracemalloc.get_traced_memory()
    _drive(widget, clock, 80.0, start=40.3)
    grown, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(widget.estimator.keyframes) == aw.KEYFRAME_HISTORY
    assert grown - base < 64 * 1024, grown - base
    widget.close()


def test_reset_and_clear_playback_clear_history_and_clock(qapp):
    widget = aw.AttitudeWidget(settings_path=None, use_gl=False)
    clock = FakeClock()
    widget.clock = clock
    _drive(widget, clock, 3.0)
    assert widget.playback.clock is not None
    widget.reset_btn.click()
    assert len(widget.estimator.keyframes) == 0
    assert widget.playback.clock is None
    _drive(widget, clock, 3.0, start=10.0)
    q = widget.estimator.q.copy()
    widget.clear_playback()
    assert len(widget.estimator.keyframes) == 0
    assert widget.playback.clock is None
    assert np.array_equal(widget.estimator.q, q)      # orientation kept
    widget.redraw()
    assert np.array_equal(widget.shown_q, q)          # held, not jumped back
    widget.close()


def test_smooth_choice_is_remembered_in_an_existing_settings_file(qapp, tmp_path):
    path = tmp_path / "map_settings.json"
    path.write_text(json.dumps({"map_pack": "x", "osm_contact": ""}))
    widget = aw.AttitudeWidget(settings_path=str(path), use_gl=False)
    assert widget.smooth_check.isChecked()            # default on
    widget.smooth_check.setChecked(False)
    data = json.loads(path.read_text())
    assert data == {"map_pack": "x", "osm_contact": "", "attitude_smooth": False}
    widget.close()
    again = aw.AttitudeWidget(settings_path=str(path), use_gl=False)
    assert not again.smooth_check.isChecked()
    again.close()


def test_smooth_choice_without_a_settings_file_is_session_only(qapp, tmp_path):
    path = tmp_path / "absent.json"
    widget = aw.AttitudeWidget(settings_path=str(path), use_gl=False)
    widget.smooth_check.setChecked(False)
    assert not widget.smooth
    assert not path.exists()
    widget.close()


def test_smooth_tooltip_states_the_delay(qapp):
    widget = aw.AttitudeWidget(settings_path=None, use_gl=False)
    tip = widget.smooth_check.toolTip()
    assert ("%.1f s" % aw.PLAYBACK_DELAY_S) in tip
    assert "immediately" in tip
    widget.close()

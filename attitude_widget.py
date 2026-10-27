"""
attitude_widget.py
==================

Live 3D attitude display driven by the GYRO_X/Y/Z telemetry stream.

WHAT IT DOES
------------
Integrates body angular rates into an orientation estimate and draws a model of
the vehicle rotating in real time against a fixed world grid, so the operator
can see the vehicle's attitude rather than reading three rate numbers.

ESTIMATOR: quaternion integration + Mahony complementary filter
---------------------------------------------------------------
Orientation is held as a unit quaternion, not Euler angles.  This matters for
this vehicle specifically: the rocket tumbles at high rate through the recovery
events and the CanSat spins under an active reaction wheel, and Euler
integration hits gimbal lock at 90 degrees of pitch and produces garbage
exactly when the flight gets interesting.

Pure gyro integration drifts without bound, so the accelerometer is used as a
gravity reference through a Mahony-style complementary filter: the error
between the measured gravity direction and the direction the current estimate
predicts is fed back as a small rate correction.

**The gate matters more than the filter.**  An accelerometer only measures
gravity when the vehicle is not accelerating.  Under a 7 g motor burn the
accelerometer points along the thrust axis, and blindly trusting it would
violently drag the estimate to a wrong attitude at the worst possible moment.
So the correction is applied only while the measured specific force magnitude
is close to 1 g (see :data:`ACCEL_GATE_LO` / :data:`ACCEL_GATE_HI`).  During
boost and during parachute snatch loads the filter coasts on the gyro alone.

Known limitation: heading (rotation about the gravity vector) has no absolute
reference, because there is no magnetometer in the packet.  Yaw therefore
drifts slowly and the "reset" button exists to re-zero it between test runs.
Roll and pitch are bounded by the gravity reference and do not drift.

THREADING
---------
Consistent with the rest of the application: :meth:`AttitudeWidget.on_packet`
is a plain slot called from the GUI thread and does only cheap arithmetic.
The scene is redrawn by a timer at :data:`ATTITUDE_RENDER_HZ`, so a burst of
packets after a dropout costs the same number of redraws as a steady stream.

SMOOTH PLAYBACK
---------------
Packets arrive about twice a second (two of every three 300 ms frames; see
:data:`MAX_DT_S`), and drawing each one as it lands moves the model in steps.
With SMOOTH on, :class:`AttitudePlayback` shows the model
:data:`PLAYBACK_DELAY_S` behind the newest packet and interpolates between
keyframes on a wall-clock driven playback clock, so it moves continuously at
the true rate. It holds the last attitude when packets stop; it never
extrapolates.

GRACEFUL DEGRADATION
--------------------
``pyqtgraph.opengl`` needs a working OpenGL context.  Field laptops on RDP,
in VMs or with broken GPU drivers do not always have one.  If the import or the
context creation fails, the widget falls back to a 2D artificial-horizon style
view drawn with QPainter, which needs nothing but Qt.  The dashboard must not
fail to start because a 3D toy could not initialise.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import time
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# 3D rendering is optional -- see GRACEFUL DEGRADATION above.
try:
    import pyqtgraph.opengl as gl
    from pyqtgraph import Vector
    from pyqtgraph.opengl.shaders import (
        FragmentShader,
        ShaderProgram,
        VertexShader,
    )
    _GL_IMPORT_OK = True
    _GL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the host machine
    gl = None
    _GL_IMPORT_OK = False
    _GL_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def _build_two_sided_shader():
    """A camera-relative, two-sided shader with a high ambient floor.

    pyqtgraph's stock ``shaded`` shader is unusable for a rotating model on a
    dark background.  Its fragment stage is::

        float p = dot(v_normal, normalize(vec3(1.0, -1.0, -1.0)));
        p = p < 0. ? 0. : p * 0.8;
        vec3 rgb = v_color.rgb * (0.2 + p);

    That is a *single fixed light direction* with no two-sided term, and every
    face pointing away from it is multiplied by 0.2.  As the vehicle rotates,
    whichever faces swing away from that one light drop to 20 % brightness --
    which against the panel background is indistinguishable from it.
    Small parts (nose cone, thin fins) vanish first, which is precisely the
    "visible at rest, gone once it starts moving" symptom.

    This replacement lights from the camera instead (``abs(normal.z)`` in eye
    space), so a face is bright whenever it faces the viewer regardless of the
    model's orientation, and never falls below ``_AMBIENT``.  Using ``abs()``
    also makes it two-sided, so an inward-wound triangle renders identically to
    an outward-wound one -- geometry winding mistakes can no longer make a part
    invisible.
    """
    # Daylight theme: the floor stays well clear of zero (which is what made
    # the stock shader lose parts mid-rotation) but is lower than the dark
    # theme's 0.55, because on a LIGHT background a deeper shadow increases
    # contrast rather than swallowing the part. shade never exceeds 1.0, so
    # colours are only ever darkened -- a dark model cannot wash out.
    ambient, diffuse = 0.45, 0.55

    # pyqtgraph switched its GLSL from the legacy fixed-pipeline style to
    # explicit uniforms/attributes.  Match whichever this install uses, or the
    # program will not link.
    modern = "u_mvp" in getattr(gl.shaders.getShaderProgram("shaded").shaders[0],
                                "code", "")

    if modern:
        vertex = """
            uniform mat4 u_mvp;
            uniform mat3 u_normal;
            attribute vec4 a_position;
            attribute vec3 a_normal;
            attribute vec4 a_color;
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                v_normal = normalize(u_normal * a_normal);
                v_color = a_color;
                gl_Position = u_mvp * a_position;
            }
        """
        fragment = """
            #ifdef GL_ES
            precision mediump float;
            #endif
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                float d = abs(normalize(v_normal).z);
                float shade = %.3f + %.3f * d;
                gl_FragColor = vec4(v_color.rgb * shade, v_color.a);
            }
        """ % (ambient, diffuse)
    else:  # pragma: no cover - older pyqtgraph
        vertex = """
            varying vec3 normal;
            void main() {
                normal = normalize(gl_NormalMatrix * gl_Normal);
                gl_FrontColor = gl_Color;
                gl_BackColor = gl_Color;
                gl_Position = ftransform();
            }
        """
        fragment = """
            varying vec3 normal;
            void main() {
                float d = abs(normalize(normal).z);
                float shade = %.3f + %.3f * d;
                gl_FragColor = vec4(gl_Color.rgb * shade, gl_Color.a);
            }
        """ % (ambient, diffuse)

    return ShaderProgram("gcsTwoSided",
                         [VertexShader(vertex), FragmentShader(fragment)])


#: Resolved lazily so an import-time GLSL problem cannot stop the app starting.
_MODEL_SHADER = None


def _model_shader():
    """Return the custom shader, falling back to the stock one on any error."""
    global _MODEL_SHADER
    if _MODEL_SHADER is None:
        try:
            _MODEL_SHADER = _build_two_sided_shader()
        except Exception:
            _MODEL_SHADER = "shaded"
    return _MODEL_SHADER


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Render tick. The model is drawn from the playback clock, not from packet
#: arrival, so this is the frame rate of the motion itself.
ATTITUDE_RENDER_HZ = 30

#: Default camera framing, chosen at the MAXIMISED window size, which is how
#: this dashboard is run.
#:
#: The distance has to be much larger than the geometry alone suggests, because
#: pyqtgraph applies its field of view HORIZONTALLY and derives the vertical
#: from the viewport aspect (``t = r * h / w`` in GLViewWidget.projectionMatrix).
#: Maximised, the GL viewport is about 413x168 -- wide and short -- so the
#: visible vertical half-extent is only ``d * tan(30deg) * (168/413)``, i.e.
#: ``d * 0.235``. The model spans roughly +/-2.5 about the origin, so anything
#: under ~11 clips the fins and the ground plane no matter how it looks in a
#: taller test window.
#:
#: Rendered at 7.0 / 8.0 / 8.5 / 9.0 / 9.5 (all clipped at the bottom) and then
#: 11.0 / 12.5 / 14.0 / 15.5: 12.5 is the first that clears the nose cone, the
#: fins and all four grid edges with visible margin, without shrinking the
#: model the way 14+ does. RESET restores exactly these values, so the initial
#: view and a reset view are always identical.
CAMERA_DISTANCE = 12.5
CAMERA_ELEVATION = 16
CAMERA_AZIMUTH = 45

#: Height of the GYRO / RESET buttons. Both are pinned to it: left to their own
#: size hints they came out 26 px and 36 px, and the taller one overran the
#: bottom of the panel and was clipped.
BUTTON_H = 26

#: Mahony proportional feedback gain, 1/s.  Higher pulls harder towards the
#: accelerometer reference (less drift, more vibration sensitivity).
MAHONY_KP = 1.6

#: Accelerometer trust gate, in g.  Outside this band the specific force is not
#: gravity and the correction is skipped entirely.
ACCEL_GATE_LO = 0.75
ACCEL_GATE_HI = 1.25

GRAVITY_MS2 = 9.80665

# --- packet timing ------------------------------------------------------------
# The flight computer sends a frame every 300 ms and routes them by
# packet_count % 3: 1 to RX1, 2 to RX2, 0 to LoRa, which this station does not
# ingest. So two of every three frames arrive and the gaps alternate 300 ms and
# 600 ms; one lost frame makes a 900 ms gap, two lost in a row 1.2 s. All of
# those are ordinary SIMO operation and must be integrated in full -- the old
# 0.25 s clamp cut every one of them short, by 17 % on a 300 ms gap and 58 % on
# a 600 ms gap, so the apparent spin rate alternated with the gap pattern.

#: Longest step integrated as it is: covers the 300/600 ms SIMO pattern and
#: up to two lost frames in a row (1.2 s). A gap between this and
#: DROPOUT_GAP_S is integrated as MAX_DT_S.
MAX_DT_S = 1.2
#: A gap longer than this is a link dropout, not a SIMO gap. The body rate over
#: it is unknown, so only the last normal interval is integrated -- one stale
#: rate held for seconds would spin the model.
DROPOUT_GAP_S = 1.5
#: Frame period of the flight computer, used as the "last normal interval"
#: before one has been seen.
NOMINAL_FRAME_S = 0.3

# --- smooth playback ----------------------------------------------------------
#: The model is shown this far behind the newest packet, so there is always a
#: newer keyframe to interpolate towards. Must exceed the longest regular gap
#: (600 ms) with margin; one lost frame (900 ms) holds briefly instead.
PLAYBACK_DELAY_S = 0.8
#: Playback-clock error tolerated before it is corrected.
PLAYBACK_DRIFT_S = 0.3
#: Correction is a rate change of at most this fraction -- never a jump.
PLAYBACK_MAX_RATE_ADJUST = 0.05
#: Keyframes retained for playback. ~20 s at two frames per 0.9 s; bounded so
#: a long session costs the same memory as a short one.
KEYFRAME_HISTORY = 48

# --- gyro bias estimation ---------------------------------------------------
# A real MEMS gyro reads a constant non-zero rate when it is perfectly still --
# its turn-on bias, typically a few deg/s and different every power cycle. Pure
# integration turns that into unbounded drift: 2 deg/s rotates the model a full
# turn in three minutes while the vehicle sits on the bench. The simulator emits
# zero-mean noise and has no bias, which is why this only shows on real hardware.
#
# The fix is what every flight computer does: while the vehicle is demonstrably
# stationary, average the gyro and subtract that average from then on.

#: Time constant of the running bias estimate, seconds. Applied per unit time
#: (alpha = 1 - exp(-dt / tau)), so it converges equally fast at 20 Hz or at
#: the ~2.2 packets/s this station receives. Slow enough that a genuine slow
#: rotation is not absorbed as bias.
BIAS_TAU_S = 7.5
#: Body rate below which the vehicle may be considered at rest, deg/s.
BIAS_MAX_RATE_DPS = 6.0
#: Specific-force window for "at rest", in g.
BIAS_ACCEL_LO, BIAS_ACCEL_HI = 0.85, 1.15
#: Accumulated time at rest before the estimate is reported as settled.
BIAS_SETTLE_S = 5.0

# Palette, matched to dashboard_ui.py.
COL_PANEL = "#e9edf3"   # daylight: light plot ground
COL_TEXT = "#0d1520"
COL_TEXT_DIM = "#41506a"
COL_ACCENT = "#0a4fa8"
COL_GRID = "#5b6a80"
COL_BODY = "#46566e"   # was pale grey for a dark panel
COL_NOSE = "#c0182b"
COL_FIN = "#0a4fa8"

# Axis colours: X red, Y green, Z blue (standard aerospace/graphics convention).
#: World axis colours, X/Y/Z. These are the same red/green/blue as the
#: accelerometer and gyro chart traces (dashboard_ui.COL_XYZ), so an axis in the
#: 3D view and its trace on a chart are visibly the same channel. Darkened from
#: the dark theme's values, which were too light to read on a light ground.
AXIS_COLORS = ((0.812, 0.122, 0.180, 1.0),   # X - #cf1f2e
               (0.106, 0.592, 0.196, 1.0),   # Y - #1b9732
               (0.051, 0.286, 0.639, 1.0))   # Z - #0d49a3


# ---------------------------------------------------------------------------
# Quaternion helpers
#
# Convention: q = [w, x, y, z], unit norm, rotating BODY -> WORLD.
# ---------------------------------------------------------------------------

def quat_normalise(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12 or not math.isfinite(n):
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation matrix that maps body vectors into world."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def quat_to_euler_deg(q: np.ndarray) -> Tuple[float, float, float]:
    """Return ``(roll, pitch, yaw)`` in degrees, aerospace Z-Y-X order.

    Used for the numeric readout only -- the estimator never round-trips
    through Euler angles, so gimbal lock here is a display artefact at worst.
    """
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def quat_from_rotvec(phi: np.ndarray) -> np.ndarray:
    """Exponential map: the unit quaternion of rotation vector *phi* (rad).

    Exact for any angle, including half a turn or more in one step, which a
    first-order ``q + q_dot * dt`` is not.
    """
    theta = float(np.linalg.norm(phi))
    if theta < 1e-12:
        # sin(theta/2)/theta -> 1/2; renormalised anyway.
        return quat_normalise(np.array([1.0, *(0.5 * phi)]))
    half = 0.5 * theta
    return np.array([math.cos(half), *(math.sin(half) / theta * phi)])


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest rotation angle between two orientations, degrees."""
    dot = abs(float(np.dot(a, b)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def quat_slerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    """Spherical linear interpolation, shortest path, *u* in [0, 1].

    Returns copies of *a* and *b* themselves at the end points, exactly.
    """
    if u <= 0.0:
        return a.copy()
    if u >= 1.0:
        return b.copy()
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        return quat_normalise(a + u * (b - a))
    theta = math.acos(min(1.0, dot))
    return ((math.sin((1.0 - u) * theta) * a + math.sin(u * theta) * b)
            / math.sin(theta))


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class AttitudeEstimator:
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

    Each step integrates the trapezoid mean of the previous and the current
    bias-corrected rate over the interval, through the exact exponential map,
    and appends a keyframe to a short bounded history that
    :class:`AttitudePlayback` interpolates for a smooth display.

    Pure computation, no Qt.  Kept separate from the widget so it can be unit
    tested and so a future firmware-side estimator could replace it without
    touching the rendering code.
    """

    __slots__ = ("q", "_last_t", "_last_recv", "_prev_raw", "_last_normal_dt",
                 "accel_valid", "samples", "corrected", "dropouts",
                 "use_accel_reference", "auto_bias", "bias", "bias_samples",
                 "rest_time_s", "at_rest", "timeline_s", "keyframes", "epoch")

    def __init__(self, use_accel_reference: bool = False,
                 auto_bias: bool = True) -> None:
        #: When False the estimate is a pure integral of the telemetry gyro.
        self.use_accel_reference = bool(use_accel_reference)
        #: Subtract an auto-learned turn-on bias from the raw gyro.
        self.auto_bias = bool(auto_bias)
        #: ``(time_s, q, rotvec)`` per update, oldest first. *rotvec* is the
        #: body-frame rotation since the previous keyframe (None after a reset
        #: or a dropout), so interpolation can follow the integrated path even
        #: through a step of more than half a turn.
        self.keyframes: Deque[Tuple[float, np.ndarray, Optional[np.ndarray]]] = \
            deque(maxlen=KEYFRAME_HISTORY)
        #: Bumped on every discontinuity (reset, cleared history, dropout), so
        #: a playback clock knows to re-anchor rather than glide.
        self.epoch = 0
        self.reset()

    def reset(self) -> None:
        """Re-zero the estimate (clears accumulated drift) and the history.

        Deliberately leaves :attr:`use_accel_reference` alone -- resetting the
        orientation should not silently change the filter mode.
        """
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_t: Optional[float] = None
        self._last_recv: Optional[float] = None
        self._prev_raw: Optional[np.ndarray] = None
        self._last_normal_dt = NOMINAL_FRAME_S
        self.accel_valid = False   # was the last sample inside the gravity gate
        self.samples = 0
        self.corrected = 0
        self.dropouts = 0
        # Re-arm bias learning: RESET is what an operator presses between bench
        # runs, which is exactly when a fresh zero should be taken.
        #: Running gyro bias estimate, deg/s, one per axis.
        self.bias = np.zeros(3)
        self.bias_samples = 0
        #: Seconds of rest that have gone into :attr:`bias`.
        self.rest_time_s = 0.0
        self.at_rest = False
        #: Playback time axis: the mission time while it advances, carried on
        #: by receive time when it does not, so it never runs backwards.
        self.timeline_s: Optional[float] = None
        self.clear_history()

    def clear_history(self) -> None:
        """Drop the keyframes but keep the orientation (new logging session)."""
        self.keyframes.clear()
        self.epoch += 1

    @property
    def bias_settled(self) -> bool:
        """True once enough time at rest has been averaged to trust it."""
        return self.rest_time_s >= BIAS_SETTLE_S

    def _elapsed(self, timestamp_s: float,
                 recv_s: Optional[float]) -> Optional[float]:
        """Time since the previous sample, or None if it cannot be known.

        Mission time first. If it is not finite or does not advance -- a
        flight-computer reboot restarts it -- the ground receive time stands in.
        """
        elapsed = None
        if (self._last_t is not None and math.isfinite(timestamp_s)
                and timestamp_s > self._last_t):
            elapsed = timestamp_s - self._last_t
        elif (recv_s is not None and self._last_recv is not None
              and math.isfinite(recv_s) and recv_s > self._last_recv):
            elapsed = recv_s - self._last_recv
        if math.isfinite(timestamp_s):
            self._last_t = timestamp_s
        if recv_s is not None and math.isfinite(recv_s):
            self._last_recv = recv_s
        return elapsed

    def update(self, gyro_dps: Tuple[float, float, float],
               accel_ms2: Tuple[float, float, float],
               timestamp_s: float, recv_s: Optional[float] = None) -> None:
        """Advance the estimate using one telemetry sample.

        *timestamp_s* is the packet mission time; dt is derived from successive
        packets so the estimate stays correct regardless of GUI frame rate.
        *recv_s* is the ground receive time, used only when the mission time
        gives no interval.
        """
        gx, gy, gz = (v if math.isfinite(v) else 0.0 for v in gyro_dps)
        # Body rate straight from the telemetry, and nothing else.  In the
        # default (pure gyro) mode this is the *only* term that reaches the
        # integrator, so zero on all three axes means the orientation is held
        # exactly still and the rotation rate tracks GYRO_X/Y/Z one to one.
        raw = np.array([gx, gy, gz])

        first = self.timeline_s is None
        elapsed = self._elapsed(timestamp_s, recv_s)
        if first:
            self.timeline_s = timestamp_s if math.isfinite(timestamp_s) else 0.0
            self._prev_raw = raw
            self.keyframes.append((self.timeline_s, self.q.copy(), None))
            return
        if elapsed is None:
            # Duplicated or non-monotonic, and no receive time to fall back
            # on: nothing sensible to integrate.
            return

        dropout = elapsed > DROPOUT_GAP_S
        if dropout:
            # The rate over the gap is unknown. Integrate one normal interval
            # on the newest rate only, and tell the playback to re-anchor.
            dt = self._last_normal_dt
            prev_raw = raw
            self.dropouts += 1
            self.epoch += 1
        else:
            dt = min(elapsed, MAX_DT_S)
            self._last_normal_dt = dt
            prev_raw = self._prev_raw if self._prev_raw is not None else raw
        self._prev_raw = raw
        self.timeline_s += elapsed
        self.samples += 1

        ax, ay, az = (v if math.isfinite(v) else 0.0 for v in accel_ms2)
        accel = np.array([ax, ay, az])
        norm = float(np.linalg.norm(accel))
        self.accel_valid = False

        # --- turn-on bias estimation -----------------------------------------
        # Learn only while the vehicle is demonstrably at rest: gravity-only
        # specific force AND a body rate small enough that it cannot be real
        # motion. In flight neither holds, so the estimate freezes at whatever
        # was learned on the pad -- which is the value that matters. Weighted
        # by time rather than by sample, so the packet rate does not matter.
        g_ratio = norm / GRAVITY_MS2 if norm > 1e-6 else 0.0
        self.at_rest = (
            BIAS_ACCEL_LO <= g_ratio <= BIAS_ACCEL_HI
            and float(np.max(np.abs(raw - self.bias))) < BIAS_MAX_RATE_DPS
        )
        if self.auto_bias and self.at_rest:
            alpha = 1.0 - math.exp(-dt / BIAS_TAU_S)
            self.bias += alpha * (raw - self.bias)
            self.bias_samples += 1
            self.rest_time_s += dt

        bias = self.bias if self.auto_bias else np.zeros(3)
        # Trapezoid rule: the mean of the rate at the start and at the end of
        # the interval, both corrected with the current bias estimate.
        omega = np.radians(0.5 * ((prev_raw - bias) + (raw - bias)))

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
                # Per unit time: over dt the correction removes the fraction
                # 1 - exp(-Kp * dt) of the error whatever the packet rate. A
                # plain Kp * error * dt overshoots once Kp * dt > 1, which a
                # 900 ms gap reaches.
                gain = (1.0 - math.exp(-MAHONY_KP * dt)) / dt
                omega = omega + gain * error
                self.accel_valid = True
                self.corrected += 1

        # --- quaternion integration: the exact rotation increment ------------
        phi = omega * dt
        self.q = quat_normalise(quat_multiply(self.q, quat_from_rotvec(phi)))
        self.keyframes.append((self.timeline_s, self.q.copy(),
                               None if dropout else phi))

    # -- outputs -----------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        return quat_to_matrix(self.q)

    @property
    def euler_deg(self) -> Tuple[float, float, float]:
        return quat_to_euler_deg(self.q)


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

class AttitudePlayback:
    """Turns about two keyframes a second into continuous motion.

    A playback clock runs :data:`PLAYBACK_DELAY_S` behind the newest keyframe
    and advances with the wall clock; the displayed attitude is interpolated
    between the keyframes either side of it. Its target is the newest
    keyframe's time, plus the wall time since that keyframe arrived, minus the
    delay -- a line advancing at wall rate between packets rather than a
    staircase that steps 300 or 600 ms on each one.

    * Drift beyond :data:`PLAYBACK_DRIFT_S` is corrected by running the clock
      up to :data:`PLAYBACK_MAX_RATE_ADJUST` fast or slow, never by a jump.
    * It jumps only when the estimator reports a discontinuity: a reset, a
      cleared history or a link dropout.
    * It never passes the newest keyframe, so when packets stop the model
      holds the last attitude. Nothing is extrapolated.

    Pure computation: the caller supplies the wall clock.
    """

    def __init__(self, delay_s: float = PLAYBACK_DELAY_S) -> None:
        self.delay_s = delay_s
        #: Re-anchoring jumps so far (reset, dropout), for diagnostics.
        self.jumps = 0
        self.reset()

    def reset(self) -> None:
        self.clock: Optional[float] = None
        self.rate = 1.0
        self._last_wall: Optional[float] = None
        self._arrival_wall = 0.0
        self._arrival_time: Optional[float] = None
        self._epoch: Optional[int] = None

    def sample(self, estimator: AttitudeEstimator, wall_s: float) -> np.ndarray:
        """The attitude to display at wall time *wall_s*."""
        keyframes = estimator.keyframes
        if not keyframes:
            self.reset()
            return estimator.q.copy()
        newest_t = keyframes[-1][0]
        if self._arrival_time != newest_t:
            # A keyframe that is new since the last frame arrived about now.
            self._arrival_time = newest_t
            self._arrival_wall = wall_s
        target = newest_t + (wall_s - self._arrival_wall) - self.delay_s

        if self.clock is None or self._epoch != estimator.epoch:
            if self.clock is not None:
                self.jumps += 1
            self.clock = target
            self.rate = 1.0
            self._epoch = estimator.epoch
        else:
            self.clock += max(0.0, wall_s - self._last_wall) * self.rate
            error = target - self.clock
            if abs(error) > PLAYBACK_DRIFT_S:
                self.rate = 1.0 + math.copysign(PLAYBACK_MAX_RATE_ADJUST, error)
            elif abs(error) < 0.25 * PLAYBACK_DRIFT_S:
                self.rate = 1.0
        self._last_wall = wall_s
        # Hold, never extrapolate.
        self.clock = min(self.clock, newest_t)
        return interpolate_keyframes(keyframes, self.clock)


def interpolate_keyframes(keyframes, t: float) -> np.ndarray:
    """Attitude at time *t* from ``(time, q, rotvec)`` keyframes.

    Before the first keyframe it is the first, after the last the last. In
    between it follows that interval's integrated rotation, scaled by the
    fraction of the interval -- which is exactly the slerp between the two
    keyframes whenever the step is under half a turn, and still turns the
    right way when a fast spin makes it more (plain slerp would take the
    short way round, backwards).
    """
    first_t, first_q, _ = keyframes[0]
    if t <= first_t:
        return first_q.copy()
    previous = keyframes[0]
    for frame in itertools.islice(keyframes, 1, None):
        t1, q1, phi = frame
        if t <= t1:
            t0, q0, _ = previous
            span = t1 - t0
            u = (t - t0) / span if span > 0.0 else 1.0
            if u >= 1.0:
                return q1.copy()
            if phi is None:
                return quat_slerp(q0, q1, u)
            return quat_normalise(quat_multiply(q0, quat_from_rotvec(u * phi)))
        previous = frame
    return keyframes[-1][1].copy()


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def _lathe(profile: Sequence[Tuple[float, float]], segments: int = 28,
           cap_start: bool = True, cap_end: bool = True):
    """Surface of revolution about +Z from a ``(radius, z)`` profile.

    One primitive covers the cylindrical body, the ogive nose and the CanSat
    can, so there is a single piece of winding logic to get right rather than
    three.  Returns ``(vertices, faces)``.
    """
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    rings: List[int] = []

    for radius, z in profile:
        rings.append(len(verts))
        if radius <= 1e-9:
            verts.append([0.0, 0.0, z])          # degenerate ring = a point
        else:
            for i in range(segments):
                a = 2.0 * math.pi * i / segments
                verts.append([radius * math.cos(a), radius * math.sin(a), z])

    for k in range(len(profile) - 1):
        r0, _ = profile[k]
        r1, _ = profile[k + 1]
        base0, base1 = rings[k], rings[k + 1]
        for i in range(segments):
            j = (i + 1) % segments
            if r0 <= 1e-9:                        # point -> ring (tip fan)
                faces.append([base0, base1 + i, base1 + j])
            elif r1 <= 1e-9:                      # ring -> point
                faces.append([base0 + i, base0 + j, base1])
            else:
                faces.append([base0 + i, base0 + j, base1 + i])
                faces.append([base0 + j, base1 + j, base1 + i])

    # End caps, so the solid is closed and never shows a hollow interior.
    if cap_start and profile[0][0] > 1e-9:
        centre = len(verts)
        verts.append([0.0, 0.0, profile[0][1]])
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([centre, rings[0] + j, rings[0] + i])
    if cap_end and profile[-1][0] > 1e-9:
        centre = len(verts)
        verts.append([0.0, 0.0, profile[-1][1]])
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([centre, rings[-1] + i, rings[-1] + j])

    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def _ogive_profile(radius: float, z_base: float, z_tip: float,
                   steps: int = 14) -> List[Tuple[float, float]]:
    """Tangent-ogive nose profile, base first, tip last.

    A tangent ogive is the curved (not straight-sided) nose the reference
    airframe uses; the radius follows a circular arc of radius ``rho`` that
    meets the body tube tangentially, so there is no crease at the joint.
    """
    length = z_tip - z_base
    rho = (radius * radius + length * length) / (2.0 * radius)
    points: List[Tuple[float, float]] = []
    for i in range(steps + 1):
        # x measured from the tip back towards the base.
        x = length * (1.0 - i / steps)
        y = math.sqrt(max(rho * rho - (length - x) ** 2, 0.0)) + radius - rho
        points.append((max(y, 0.0), z_tip - x))
    return points


def _fin(inner_r: float, outer_r: float,
         root_fwd_z: float, root_aft_z: float,
         tip_fwd_z: float, tip_aft_z: float,
         angle_rad: float, thickness: float):
    """One swept trapezoidal fin as a closed slab, rotated about +Z.

    Built as a solid rather than a zero-thickness plate: a thin plate seen
    edge-on covers almost no pixels, which is the other half of why the fins
    read as "missing" while the model turns.
    """
    half = thickness / 2.0
    plate = [
        (inner_r, root_fwd_z),   # 0 root leading edge
        (outer_r, tip_fwd_z),    # 1 tip  leading edge (swept aft)
        (outer_r, tip_aft_z),    # 2 tip  trailing edge
        (inner_r, root_aft_z),   # 3 root trailing edge
    ]
    verts: List[List[float]] = []
    for sign in (-1.0, 1.0):
        for radius, z in plate:
            verts.append([radius, sign * half, z])

    faces = [
        [0, 1, 2], [0, 2, 3],      # -y face
        [4, 6, 5], [4, 7, 6],      # +y face
        [0, 4, 5], [0, 5, 1],      # leading edge
        [1, 5, 6], [1, 6, 2],      # tip edge
        [2, 6, 7], [2, 7, 3],      # trailing edge
        [3, 7, 4], [3, 4, 0],      # root edge
    ]

    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    rotated = [[x * ca - y * sa, x * sa + y * ca, z] for x, y, z in verts]
    return np.array(rotated, dtype=float), np.array(faces, dtype=int)


def _merge(parts):
    """Concatenate ``(verts, faces)`` pairs into one mesh."""
    all_v, all_f, offset = [], [], 0
    for verts, faces in parts:
        all_v.append(verts)
        all_f.append(faces + offset)
        offset += len(verts)
    return np.vstack(all_v), np.vstack(all_f)


# ---------------------------------------------------------------------------
# Vehicle geometry
#
# Proportions follow the reference airframe (CU Jammu Astro Rocketry-069):
# orange body tube, black ogive nose cone about a quarter of the overall
# length, and four swept black fins at the base.  Overall length is 4.0 units
# centred on the origin, so nose : body = 1 : 3 exactly.
# ---------------------------------------------------------------------------

ROCKET_RADIUS = 0.30
ROCKET_BODY_Z0 = -2.00          # tail
ROCKET_BODY_Z1 = 1.00           # body/nose joint  -> body length 3.0
ROCKET_NOSE_Z1 = 2.00           # tip              -> nose length 1.0
ROCKET_FIN_COUNT = 4

#: World ground-plane height, just below the fin trailing edge.
GROUND_Z = -2.15

# Model palette.  The reference airframe's nose and fins are black, but pure
# black on the panel is hard to separate, so they are rendered as a graphite
# that still reads as "black hardware" against the orange tube.
_C_BODY_ORANGE = (0.91, 0.38, 0.10, 1.0)
_C_GRAPHITE = (0.24, 0.26, 0.31, 1.0)      # deeper, to read on light
_C_CANSAT_BODY = (0.36, 0.43, 0.54, 1.0)   # darkened for a light background
_C_CANSAT_CAP = (0.91, 0.22, 0.31, 1.0)
_C_CANSAT_BASE = (0.18, 0.22, 0.29, 1.0)


def build_rocket_parts():
    """Return ``{part_name: (vertices, faces)}`` for the rocket.

    Parts are kept separate so each becomes its own persistent ``GLMeshItem``
    with its own colour.  Local placement is baked into the vertices, so every
    part shares one orientation matrix and they can never drift out of sync.
    """
    body = _lathe([
        (ROCKET_RADIUS, ROCKET_BODY_Z0),
        (ROCKET_RADIUS, ROCKET_BODY_Z1),
    ])
    nose = _lathe(_ogive_profile(ROCKET_RADIUS, ROCKET_BODY_Z1, ROCKET_NOSE_Z1),
                  cap_start=True, cap_end=False)

    # Swept trapezoidal fins flaring outward at the base, 90 degrees apart.
    fin_parts = []
    for k in range(ROCKET_FIN_COUNT):
        fin_parts.append(_fin(
            inner_r=ROCKET_RADIUS * 0.96,
            outer_r=ROCKET_RADIUS + 0.58,
            root_fwd_z=ROCKET_BODY_Z0 + 1.05,
            root_aft_z=ROCKET_BODY_Z0,
            tip_fwd_z=ROCKET_BODY_Z0 + 0.46,
            tip_aft_z=ROCKET_BODY_Z0 - 0.06,
            angle_rad=2.0 * math.pi * k / ROCKET_FIN_COUNT,
            thickness=0.055,
        ))

    return {"body": body, "nose": nose, "fins": _merge(fin_parts)}


#: Tessellated CAD geometry for the CanSat, produced from voronoi.step.
#: See tools/step_to_mesh.py for the conversion; the .npz holds float32
#: vertices and uint32 faces, already centred, scaled and stood on the ground
#: plane, so loading it at runtime costs one np.load and no geometry work.
CANSAT_MESH_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "cansat_mesh.npz")


def _load_cansat_mesh():
    """Return ``(vertices, faces)`` from the converted CAD mesh, or None.

    Returns None rather than raising if the file is missing or unreadable: a
    checkout without the converted asset still starts, and the caller falls
    back to the simple can with a visible warning rather than an empty view.
    """
    try:
        with np.load(CANSAT_MESH_FILE) as data:
            verts = np.asarray(data["vertices"], dtype=np.float32)
            faces = np.asarray(data["faces"], dtype=np.uint32)
    except Exception:
        return None
    if verts.ndim != 2 or verts.shape[1] != 3 or len(faces) == 0:
        return None
    return verts, faces


def _build_cansat_placeholder():
    """The original plain can, kept only as a fallback."""
    return {
        "body": _lathe([(0.42, -0.75), (0.42, 0.62)], segments=26),
        "cap": _lathe([(0.42, 0.62), (0.42, 0.75)], segments=26),
        "base": _lathe([(0.30, -0.84), (0.30, -0.75)], segments=26),
    }


#: True when the real CAD geometry was used, False when the placeholder was.
#: Read by the widget so the fallback is stated rather than passed off as the
#: real airframe.
CANSAT_MESH_IS_CAD = False


def build_cansat_parts():
    """Return ``{part_name: (vertices, faces)}`` for the CanSat.

    Prefers the tessellated CAD geometry. The CAD mesh is a single shell, so it
    is returned as one part -- the placeholder's body/cap/base split existed
    only to colour a lathe, and there is no equivalent division in the real
    geometry to map those names onto.
    """
    global CANSAT_MESH_IS_CAD
    mesh = _load_cansat_mesh()
    if mesh is None:
        CANSAT_MESH_IS_CAD = False
        return _build_cansat_placeholder()
    CANSAT_MESH_IS_CAD = True
    return {"shell": mesh}


#: Which colour each part is drawn in.
ROCKET_PART_COLORS = {
    "body": _C_BODY_ORANGE,
    "nose": _C_GRAPHITE,
    "fins": _C_GRAPHITE,
}
CANSAT_PART_COLORS = {
    # Real CAD geometry: one shell.
    "shell": _C_CANSAT_BODY,
    # Placeholder fallback parts.
    "body": _C_CANSAT_BODY,
    "cap": _C_CANSAT_CAP,
    "base": _C_CANSAT_BASE,
}


# ---------------------------------------------------------------------------
# 2D fallback view (no OpenGL required)
# ---------------------------------------------------------------------------

class _AttitudeFallbackView(QWidget):
    """Artificial-horizon style 2D attitude view drawn with QPainter.

    Used when an OpenGL context is unavailable.  It shows roll and pitch
    against a pitch ladder plus a plan-view heading needle -- less pretty than
    the 3D model, but it conveys the same orientation information and it works
    everywhere Qt works.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self.reason = ""

    def set_attitude(self, roll: float, pitch: float, yaw: float) -> None:
        self._roll, self._pitch, self._yaw = roll, pitch, yaw
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor(COL_PANEL))

        size = min(rect.width(), rect.height()) - 16
        if size <= 40:
            painter.end()
            return
        cx = rect.width() / 2.0
        cy = rect.height() / 2.0
        radius = size / 2.0

        painter.save()
        painter.setClipRect(QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius))
        painter.translate(cx, cy)
        painter.rotate(-self._roll)
        # Pitch translates the horizon vertically; 1 degree ~ radius/50 px.
        offset = self._pitch * radius / 50.0
        painter.translate(0.0, offset)

        big = radius * 3.0
        painter.fillRect(QRectF(-big, -big, 2 * big, big), QColor("#3f7fbc"))   # sky
        painter.fillRect(QRectF(-big, 0.0, 2 * big, big), QColor("#8a6234"))    # ground
        painter.setPen(QPen(QColor(COL_TEXT), 2))
        painter.drawLine(int(-big), 0, int(big), 0)

        painter.setPen(QPen(QColor(COL_TEXT_DIM), 1))
        for deg in (-30, -20, -10, 10, 20, 30):
            y = -deg * radius / 50.0
            half = radius * (0.28 if deg % 20 else 0.44)
            painter.drawLine(int(-half), int(y), int(half), int(y))
        painter.restore()

        # Fixed aircraft reference marker.
        painter.setPen(QPen(QColor(COL_ACCENT), 2.5))
        painter.drawLine(int(cx - radius * 0.4), int(cy), int(cx - radius * 0.1), int(cy))
        painter.drawLine(int(cx + radius * 0.1), int(cy), int(cx + radius * 0.4), int(cy))
        painter.drawEllipse(QPointF(cx, cy), 3.0, 3.0)

        painter.setPen(QPen(QColor(COL_GRID), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Heading needle, top-right corner.
        painter.save()
        painter.translate(rect.width() - 34, 34)
        painter.setPen(QPen(QColor(COL_GRID), 1))
        painter.drawEllipse(QPointF(0, 0), 20, 20)
        painter.rotate(self._yaw)
        painter.setBrush(QBrush(QColor(COL_NOSE)))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF([QPointF(0, -18), QPointF(-5, 6), QPointF(5, 6)]))
        painter.restore()

        painter.setPen(QColor(COL_TEXT_DIM))
        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(6, rect.height() - 6, "2D fallback (no OpenGL)")
        painter.end()


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

class AttitudeWidget(QWidget):
    """Vehicle attitude panel: 3D model, axis overlay, readouts and reset."""

    def __init__(self, parent: Optional[QWidget] = None,
                 settings_path: Optional[str] = None,
                 use_gl: bool = True) -> None:
        super().__init__(parent)
        self.estimator = AttitudeEstimator()
        self.playback = AttitudePlayback()
        #: Wall clock for the playback. Replaceable so tests can drive time.
        self.clock = time.monotonic
        #: The settings file the SMOOTH choice is kept in, if it exists.
        self.settings_path = settings_path
        self.smooth = self._load_smooth_setting()
        #: The quaternion currently on screen; a frame is drawn only when the
        #: one to display differs from it.
        self.shown_q: Optional[np.ndarray] = None
        self.frames_drawn = 0
        self._dirty = False
        self._payload_type = ""
        self._mesh_kind = ""
        self._use_gl = bool(use_gl)
        self.gl_available = False

        self.view = None            # GLViewWidget, when available
        self.fallback = None        # _AttitudeFallbackView otherwise
        #: {"ROCKET": {part: GLMeshItem}, "CANSAT": {...}} -- built once, never
        #: removed. See _build_models().
        self._parts = {"ROCKET": {}, "CANSAT": {}}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(5)

        layout.addWidget(self._build_view(), 1)
        # Stretch 0 and a fixed vertical policy on the controls, so shrinking
        # the window takes height from the 3D view and never from the buttons.
        layout.addLayout(self._build_readouts(), 0)

    # -- construction ------------------------------------------------------

    def _build_view(self) -> QWidget:
        """Create the 3D view, or the 2D fallback if OpenGL is unavailable."""
        if _GL_IMPORT_OK and self._use_gl:
            try:
                view = gl.GLViewWidget()
                view.setCameraPosition(distance=CAMERA_DISTANCE,
                                       elevation=CAMERA_ELEVATION,
                                       azimuth=CAMERA_AZIMUTH)
                view.setMinimumHeight(80)
                view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                view.setBackgroundColor(QColor(COL_PANEL))

                # World reference: a fixed ground grid.  Because the grid never
                # moves and the model does, the grid *is* the horizon reference
                # -- the viewer can always see which way is down.
                grid = gl.GLGridItem()
                grid.setSize(5, 5)
                grid.setSpacing(0.5, 0.5)
                # Sits just under the fin trailing edge so the vehicle stands
                # on the ground plane instead of sinking through it.
                grid.translate(0, 0, GROUND_Z)
                grid.setColor(QColor(70, 86, 108, 215))
                view.addItem(grid)

                # World axes at the origin: X red, Y green, Z blue (up).
                for index, color in enumerate(AXIS_COLORS):
                    end = [0.0, 0.0, 0.0]
                    end[index] = 2.0 if index == 2 else 1.6
                    line = gl.GLLinePlotItem(
                        pos=np.array([[0.0, 0.0, GROUND_Z],
                                      [end[0], end[1], end[2] + GROUND_Z]]),
                        color=color, width=2.0, antialias=True,
                    )
                    view.addItem(line)

                self.view = view
                self.gl_available = True
                self._build_models()
                self._set_mesh("ROCKET")
                return view
            except Exception as exc:  # pragma: no cover - driver dependent
                reason = str(exc)
        elif not self._use_gl:
            reason = "OpenGL view disabled"
        else:
            reason = _GL_IMPORT_ERROR or "pyqtgraph.opengl unavailable"

        self.fallback = _AttitudeFallbackView()
        self.fallback.reason = reason
        self.gl_available = False
        return self.fallback

    def _build_readouts(self) -> QVBoxLayout:
        stack = QVBoxLayout()
        stack.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(6)

        self.rpy_label = QLabel("R --  P --  Y --")
        rpy_font = QFont("Consolas")
        rpy_font.setPointSize(10)
        rpy_font.setBold(True)
        self.rpy_label.setFont(rpy_font)
        self.rpy_label.setStyleSheet("color: %s;" % COL_TEXT)
        # Must be allowed to shrink: its natural width would otherwise set a
        # floor on the whole bottom row of the dashboard.
        self.rpy_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.rpy_label.setAlignment(Qt.AlignCenter)
        stack.addWidget(self.rpy_label)

        # Real toggle, not a status lamp: OFF means the orientation is a pure
        # integral of GYRO_X/Y/Z, ON blends in the accelerometer gravity
        # reference.  Defaults to OFF so the model tracks the transmitted body
        # rates exactly and holds still when they are zero.
        self.ref_btn = QPushButton("GYRO")
        ref_font = QFont()
        ref_font.setPointSize(8)
        ref_font.setBold(True)
        self.ref_btn.setFont(ref_font)
        self.ref_btn.setCheckable(True)
        self.ref_btn.setChecked(False)
        # Fixed width so the toggle never squeezes the RESET button off the row.
        self.ref_btn.setFixedWidth(74)
        self.ref_btn.setFixedHeight(BUTTON_H)
        self.ref_btn.setToolTip(
            "Attitude source toggle.\n\n"
            "GYRO (off, default): orientation is the integral of GYRO_X/Y/Z only.\n"
            "  Rotation corresponds 1:1 with the transmitted body rates and the\n"
            "  model holds perfectly still when they read zero. All axes drift\n"
            "  slowly; use RESET between runs.\n\n"
            "ACC REF (on): adds an accelerometer gravity reference that bounds\n"
            "  roll and pitch drift. Because that correction comes from the\n"
            "  accelerometer, the model can move slightly even at zero gyro.\n"
            "  Shows G-LOCK while the accelerometer reads close to 1 g, and\n"
            "  ACC REF while it is coasting on the gyro through boost or\n"
            "  deployment shock, when the accelerometer is not a valid\n"
            "  gravity reference."
        )
        self.ref_btn.toggled.connect(self._on_ref_toggled)
        self._style_ref(False)
        row.addWidget(self.ref_btn)

        self.reset_btn = QPushButton("RESET")
        self.reset_btn.setFont(ref_font)
        self.reset_btn.setFixedWidth(72)
        self.reset_btn.setFixedHeight(BUTTON_H)
        self.reset_btn.setToolTip(
            "Re-zero the orientation estimate.\n"
            "Yaw has no absolute reference without a magnetometer, so it drifts;\n"
            "reset it between test runs."
        )
        self.reset_btn.clicked.connect(self.reset_orientation)
        row.addWidget(self.reset_btn)

        self.smooth_check = QCheckBox("SMOOTH")
        self.smooth_check.setFont(ref_font)
        self.smooth_check.setFixedHeight(BUTTON_H)
        self.smooth_check.setChecked(self.smooth)
        self.smooth_check.setToolTip(
            "Smooth playback (default on).\n\n"
            "Packets reach this station about twice a second (300 / 600 ms\n"
            "gaps), so the model is shown %.1f s behind the newest packet and\n"
            "moves continuously between them. That adds a delay of %.1f s.\n\n"
            "Unchecked: the newest packet's attitude is shown immediately,\n"
            "with no delay, stepping once per packet (the old behaviour)."
            % (PLAYBACK_DELAY_S, PLAYBACK_DELAY_S)
        )
        self.smooth_check.toggled.connect(self._on_smooth_toggled)
        row.addWidget(self.smooth_check)
        row.addStretch(1)
        stack.addLayout(row)
        return stack

    def _build_models(self) -> None:
        """Create every mesh item **once** and add it to the view for good.

        Both vehicles' parts are built up front and added to the GLViewWidget
        here.  Nothing is ever removed, rebuilt or re-added afterwards --
        switching vehicle only toggles ``setVisible`` and the per-frame update
        only calls ``setTransform``.  That removes the whole class of "an item
        got dropped from the scene during an update" failure.
        """
        self._parts = {"ROCKET": {}, "CANSAT": {}}

        for kind, builder, palette in (
            ("ROCKET", build_rocket_parts, ROCKET_PART_COLORS),
            ("CANSAT", build_cansat_parts, CANSAT_PART_COLORS),
        ):
            for name, (verts, faces) in builder().items():
                # drawEdges stays off: on a lathe surface it renders every
                # triangle diagonal and the model reads as a hairy barrel.
                # Shading plus the part colours carry the form instead.
                item = gl.GLMeshItem(
                    vertexes=verts, faces=faces,
                    color=palette[name],
                    smooth=False, drawEdges=False,
                    shader=_model_shader(), glOptions="opaque",
                )
                item.setVisible(False)
                self.view.addItem(item)
                self._parts[kind][name] = item

    def set_vehicle(self, payload_type: str) -> None:
        """Choose which vehicle model is displayed.

        Called by the dashboard, which resolves auto-detection against the
        operator's manual override.  The widget deliberately does not decide
        this for itself from the packet stream.
        """
        self._payload_type = payload_type or ""
        self._set_mesh(self._payload_type)

    def _set_mesh(self, payload_type: str) -> None:
        """Show the model for this vehicle by toggling visibility only."""
        kind = "CANSAT" if payload_type == "CANSAT" else "ROCKET"
        self._mesh_kind = kind
        if not self.gl_available:
            return
        for group, items in self._parts.items():
            visible = (group == kind)
            for item in items.values():
                item.setVisible(visible)
        # Re-apply the current orientation so a freshly shown part is not left
        # at identity for a frame.
        self.redraw(force=True)

    @property
    def _active_items(self):
        """The mesh items currently on screen."""
        if not self.gl_available:
            return []
        return list(self._parts.get(self._mesh_kind, {}).values())

    # -- telemetry slot (hot path: keep cheap) -----------------------------

    def on_packet(self, packet) -> None:
        """Feed one validated packet into the estimator.

        Called from the GUI thread via the same queued signal the rest of the
        dashboard uses.  Does arithmetic only -- no repaint here.
        """
        try:
            # Mission time drives the integration; the receive time is the
            # estimator's fallback when mission time is missing or stalls.
            self.estimator.update(
                (packet.gyro_x, packet.gyro_y, packet.gyro_z),
                (packet.acc_x, packet.acc_y, packet.acc_z),
                packet.mission_time_s,
                packet.gs_recv_epoch,
            )
            self._dirty = True
        except Exception:
            # An attitude display fault must never interrupt ingestion.
            pass

    def reset_camera(self) -> None:
        """Restore the default camera framing.

        Separate from the orientation reset so either can be used alone, but
        the RESET button does both: a viewer who has zoomed or orbited until
        the model is off-screen wants one action that gives everything back.
        """
        if self.view is None:
            return
        try:
            self.view.setCameraPosition(
                distance=CAMERA_DISTANCE,
                elevation=CAMERA_ELEVATION,
                azimuth=CAMERA_AZIMUTH,
            )
            # pyqtgraph pans by moving the camera's centre; setCameraPosition
            # alone leaves that offset in place, so a panned view would come
            # back still off-centre.
            self.view.opts["center"] = Vector(0.0, 0.0, 0.0)
            self.view.update()
        except Exception:
            # A camera reset must never be able to take the widget down.
            pass

    def reset_orientation(self) -> None:
        """Re-zero the estimate *and* the camera; clears accumulated yaw drift.

        The orientation behaviour is unchanged -- the camera restore is added
        alongside it, not in place of it.
        """
        self.estimator.reset()
        self.playback.reset()
        self.reset_camera()
        self._dirty = True
        self.redraw(force=True)

    def clear(self) -> None:
        """Session reset hook, called by the dashboard's Clear button."""
        self.reset_orientation()

    def clear_playback(self) -> None:
        """Drop the keyframe history and the playback clock, keeping the
        orientation. Called when a new logging session starts."""
        self.estimator.clear_history()
        self.playback.reset()
        self._dirty = True

    # -- rendering (timer driven) ------------------------------------------

    def displayed_q(self) -> np.ndarray:
        """The attitude to show now: interpolated playback, or the newest."""
        if self.smooth:
            return self.playback.sample(self.estimator, self.clock())
        return self.estimator.q.copy()

    def redraw(self, force: bool = False) -> None:
        """Draw the displayed attitude.  Called by a QTimer at
        ATTITUDE_RENDER_HZ; draws only when the attitude to show has changed
        (or a control changed, or *force*), so a held model costs nothing.

        The 3D view, the 2D fallback and the R/P/Y readout all take the same
        quaternion from :meth:`displayed_q`.
        """
        try:
            q = self.displayed_q()
        except Exception:
            q = self.estimator.q.copy()
        changed = self.shown_q is None or not np.array_equal(q, self.shown_q)
        if not (changed or self._dirty or force):
            return
        self._dirty = False
        self.shown_q = q
        self.frames_drawn += 1
        try:
            roll, pitch, yaw = quat_to_euler_deg(q)
            self.rpy_label.setText(
                "R %+7.1f  P %+7.1f  Y %+7.1f" % (roll, pitch, yaw)
            )
            self._update_bias_tooltip()
            self._style_ref(self.estimator.accel_valid)

            if self.gl_available:
                transform = np.eye(4)
                transform[:3, :3] = quat_to_matrix(q)
                matrix = _as_qmatrix(transform)
                # Every part gets the *same* matrix, so the assembly can never
                # come apart or rotate at different rates.  setTransform only:
                # the geometry itself is never touched after construction.
                for item in self._active_items:
                    item.setTransform(matrix)
            elif self.fallback is not None:
                self.fallback.set_attitude(roll, pitch, yaw)
        except Exception:
            pass

    def _update_bias_tooltip(self) -> None:
        """Expose the learned gyro bias; it must not be invisible magic."""
        est = self.estimator
        bx, by, bz = (float(v) for v in est.bias)
        if not est.auto_bias:
            state = "auto-zero OFF"
        elif est.bias_settled:
            state = "settled (%.1f s at rest)" % est.rest_time_s
        elif est.bias_samples:
            state = "learning (%.1f / %.1f s at rest)" % (est.rest_time_s,
                                                          BIAS_SETTLE_S)
        else:
            state = "not learned yet - hold the vehicle still"
        self.rpy_label.setToolTip(
            "Roll / pitch / yaw, degrees.\n\n"
            "Gyro turn-on bias auto-zero: %s\n"
            "  estimated bias  X %+.3f  Y %+.3f  Z %+.3f deg/s\n"
            "  vehicle at rest: %s\n\n"
            "A real MEMS gyro reads a small non-zero rate when perfectly still.\n"
            "That bias is learned while the vehicle is stationary and subtracted,\n"
            "so the model holds position instead of drifting. RESET re-arms it."
            % (state, bx, by, bz, "yes" if est.at_rest else "no")
        )

    def _on_ref_toggled(self, checked: bool) -> None:
        """Switch between pure gyro integration and the complementary filter."""
        self.estimator.use_accel_reference = bool(checked)
        self._style_ref(self.estimator.accel_valid)
        self._dirty = True

    def _on_smooth_toggled(self, checked: bool) -> None:
        self.smooth = bool(checked)
        self.playback.reset()
        self._save_smooth_setting()
        self._dirty = True
        self.redraw(force=True)

    # -- SMOOTH persistence ------------------------------------------------
    # Kept in the map window's settings file when there is one, as one more
    # key. No file: the choice lasts for the session only; this never
    # creates the file.

    _SMOOTH_KEY = "attitude_smooth"

    def _load_smooth_setting(self) -> bool:
        if not self.settings_path:
            return True
        try:
            with open(self.settings_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return True
        if isinstance(data, dict) and isinstance(data.get(self._SMOOTH_KEY), bool):
            return data[self._SMOOTH_KEY]
        return True

    def _save_smooth_setting(self) -> None:
        if not self.settings_path or not os.path.isfile(self.settings_path):
            return
        try:
            with open(self.settings_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return
            data[self._SMOOTH_KEY] = self.smooth
            with open(self.settings_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except Exception:
            # Remembering a display preference must never break the panel.
            pass

    def _style_ref(self, locked: bool) -> None:
        """Colour the toggle for its three meaningful states."""
        if not self.ref_btn.isChecked():
            # Pure gyro: orientation is 1:1 with the telemetry body rates.
            self.ref_btn.setText("GYRO")
            style = "color:#ffffff; background:#41506a;"
        elif locked:
            # Filter on and the accelerometer is inside the 1 g gate.
            self.ref_btn.setText("G-LOCK")
            style = "color:#ffffff; background:#0d7a3d;"
        else:
            # Filter on but coasting: accelerometer outside the gate.
            self.ref_btn.setText("ACC REF")
            style = "color:#ffffff; background:#a05000;"
        self.ref_btn.setStyleSheet(
            style + " border-radius:3px; padding:2px 4px; font-weight:700;"
        )


def _as_qmatrix(transform: np.ndarray):
    """Convert a 4x4 numpy array into the QMatrix4x4 pyqtgraph expects."""
    from PyQt5.QtGui import QMatrix4x4
    return QMatrix4x4(*[float(v) for v in transform.flatten()])

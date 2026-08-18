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
is a plain slot called from the GUI thread, does only cheap arithmetic, and
sets a dirty flag.  The 3D scene is redrawn by a timer at
:data:`ATTITUDE_RENDER_HZ`, so a 20 Hz packet rate or a burst after a dropout
costs the same number of redraws.

GRACEFUL DEGRADATION
--------------------
``pyqtgraph.opengl`` needs a working OpenGL context.  Field laptops on RDP,
in VMs or with broken GPU drivers do not always have one.  If the import or the
context creation fails, the widget falls back to a 2D artificial-horizon style
view drawn with QPainter, which needs nothing but Qt.  The dashboard must not
fail to start because a 3D toy could not initialise.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
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
    _GL_IMPORT_OK = True
    _GL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the host machine
    gl = None
    _GL_IMPORT_OK = False
    _GL_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

ATTITUDE_RENDER_HZ = 20

#: Mahony proportional feedback gain, 1/s.  Higher pulls harder towards the
#: accelerometer reference (less drift, more vibration sensitivity).
MAHONY_KP = 1.6

#: Accelerometer trust gate, in g.  Outside this band the specific force is not
#: gravity and the correction is skipped entirely.
ACCEL_GATE_LO = 0.75
ACCEL_GATE_HI = 1.25

GRAVITY_MS2 = 9.80665

#: Clamp on the integration step.  After a link dropout the next packet can
#: carry a timestamp seconds later; integrating that in one step would spin the
#: model wildly, so the gap is capped.
MAX_DT_S = 0.25

# Palette, matched to dashboard_ui.py.
COL_PANEL = "#161d27"
COL_TEXT = "#dbe3ee"
COL_TEXT_DIM = "#8b9aad"
COL_ACCENT = "#4aa8ff"
COL_GRID = "#2b3746"
COL_BODY = "#c8d2e0"
COL_NOSE = "#e8384f"
COL_FIN = "#4aa8ff"

# Axis colours: X red, Y green, Z blue (standard aerospace/graphics convention).
AXIS_COLORS = ((0.92, 0.30, 0.32, 1.0),
               (0.34, 0.80, 0.42, 1.0),
               (0.32, 0.62, 0.95, 1.0))


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


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class AttitudeEstimator:
    """Quaternion attitude estimate from gyro rates with a gravity reference.

    Pure computation, no Qt.  Kept separate from the widget so it can be unit
    tested and so a future firmware-side estimator could replace it without
    touching the rendering code.
    """

    __slots__ = ("q", "_last_t", "accel_valid", "samples", "corrected")

    def __init__(self) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_t: Optional[float] = None
        self.accel_valid = False   # was the last sample inside the gravity gate
        self.samples = 0
        self.corrected = 0

    def reset(self) -> None:
        """Re-zero the estimate (clears accumulated yaw drift)."""
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_t = None
        self.accel_valid = False
        self.samples = 0
        self.corrected = 0

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
        dt = min(dt, MAX_DT_S)

        self.samples += 1

        # --- gravity reference (Mahony proportional correction) -------------
        omega = np.array([math.radians(gx), math.radians(gy), math.radians(gz)])

        ax, ay, az = (v if math.isfinite(v) else 0.0 for v in accel_ms2)
        accel = np.array([ax, ay, az])
        norm = float(np.linalg.norm(accel))
        self.accel_valid = False

        if norm > 1e-6:
            g_ratio = norm / GRAVITY_MS2
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
# Mesh construction
# ---------------------------------------------------------------------------

def _cylinder(radius: float, z0: float, z1: float, segments: int = 24):
    """Closed cylinder along +Z.  Returns ``(vertices, faces)``."""
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        x, y = radius * math.cos(a), radius * math.sin(a)
        verts.append([x, y, z0])
        verts.append([x, y, z1])
    for i in range(segments):
        b0, b1 = 2 * i, 2 * ((i + 1) % segments)
        faces.append([b0, b1, b0 + 1])
        faces.append([b1, b1 + 1, b0 + 1])
    # End caps.
    base_c = len(verts)
    verts.append([0.0, 0.0, z0])
    verts.append([0.0, 0.0, z1])
    for i in range(segments):
        b0, b1 = 2 * i, 2 * ((i + 1) % segments)
        faces.append([base_c, b1, b0])
        faces.append([base_c + 1, b0 + 1, b1 + 1])
    return np.array(verts), np.array(faces)


def _cone(radius: float, z0: float, z1: float, segments: int = 24):
    """Cone with its base at *z0* and apex at *z1*."""
    verts: List[List[float]] = [[0.0, 0.0, z1]]          # apex
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        verts.append([radius * math.cos(a), radius * math.sin(a), z0])
    faces: List[List[int]] = []
    for i in range(segments):
        faces.append([0, 1 + i, 1 + (i + 1) % segments])
    base_c = len(verts)
    verts.append([0.0, 0.0, z0])
    for i in range(segments):
        faces.append([base_c, 1 + (i + 1) % segments, 1 + i])
    return np.array(verts), np.array(faces)


def _fin(inner_r: float, span: float, z_root0: float, z_root1: float,
         z_tip: float, angle_rad: float, thickness: float = 0.02):
    """One flat trapezoidal fin, rotated *angle_rad* about +Z."""
    outer_r = inner_r + span
    plate = [
        [inner_r, 0.0, z_root0], [outer_r, 0.0, z_tip],
        [outer_r, 0.0, z_root1], [inner_r, 0.0, z_root1],
    ]
    verts: List[List[float]] = []
    for sign in (-1.0, 1.0):
        for px, _, pz in plate:
            verts.append([px, sign * thickness, pz])
    faces = [
        [0, 1, 2], [0, 2, 3],          # one face
        [4, 6, 5], [4, 7, 6],          # the other
        [0, 4, 5], [0, 5, 1],          # edges
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ]
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    rotated = [[x * ca - y * sa, x * sa + y * ca, z] for x, y, z in verts]
    return np.array(rotated), np.array(faces)


def _merge(parts):
    """Concatenate ``(verts, faces, rgba)`` parts into one coloured mesh.

    Per-face colours rather than one flat colour, because the whole point of
    the model is showing which way the vehicle points: a uniformly grey body is
    indistinguishable from itself rotated 180 degrees.  Colouring the nose lets
    the viewer read the attitude instantly.
    """
    all_v, all_f, all_c, offset = [], [], [], 0
    for verts, faces, rgba in parts:
        all_v.append(verts)
        all_f.append(faces + offset)
        all_c.append(np.tile(np.array(rgba, dtype=float), (len(faces), 1)))
        offset += len(verts)
    return np.vstack(all_v), np.vstack(all_f), np.vstack(all_c)


# Model palette.
_C_BODY = (0.72, 0.76, 0.83, 1.0)   # airframe grey
_C_NOSE = (0.91, 0.22, 0.31, 1.0)   # nose / +Z end, red
_C_FIN = (0.29, 0.66, 1.0, 1.0)     # fins, blue
_C_BASE = (0.24, 0.29, 0.36, 1.0)   # tail / -Z end, dark


def build_rocket_mesh():
    """Rocket silhouette: body tube, red nose cone and three fins, axis +Z."""
    return _merge([
        (*_cylinder(0.22, -1.0, 0.55), _C_BODY),
        (*_cone(0.22, 0.55, 1.25), _C_NOSE),
        (*_fin(0.20, 0.42, -1.0, -0.45, -0.85, 0.0), _C_FIN),
        (*_fin(0.20, 0.42, -1.0, -0.45, -0.85, 2.0 * math.pi / 3.0), _C_FIN),
        (*_fin(0.20, 0.42, -1.0, -0.45, -0.85, 4.0 * math.pi / 3.0), _C_FIN),
    ])


def build_cansat_mesh():
    """CanSat: soda-can body with a red top cap so its +Z end is identifiable."""
    return _merge([
        (*_cylinder(0.42, -0.75, 0.62, segments=26), _C_BODY),
        (*_cylinder(0.42, 0.62, 0.75, segments=26), _C_NOSE),   # top cap
        (*_cylinder(0.30, -0.84, -0.75, segments=26), _C_BASE),  # base plate
    ])


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
        painter.fillRect(QRectF(-big, -big, 2 * big, big), QColor("#1e4b73"))   # sky
        painter.fillRect(QRectF(-big, 0.0, 2 * big, big), QColor("#4a3520"))    # ground
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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.estimator = AttitudeEstimator()
        self._dirty = False
        self._payload_type = ""
        self._mesh_kind = ""
        self.gl_available = False

        self.view = None            # GLViewWidget, when available
        self.fallback = None        # _AttitudeFallbackView otherwise
        self._model = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        layout.addWidget(self._build_view(), 1)
        layout.addLayout(self._build_readouts())

    # -- construction ------------------------------------------------------

    def _build_view(self) -> QWidget:
        """Create the 3D view, or the 2D fallback if OpenGL is unavailable."""
        if _GL_IMPORT_OK:
            try:
                view = gl.GLViewWidget()
                view.setCameraPosition(distance=3.5, elevation=16, azimuth=45)
                view.setMinimumHeight(190)
                view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                view.setBackgroundColor(QColor(COL_PANEL))

                # World reference: a fixed ground grid.  Because the grid never
                # moves and the model does, the grid *is* the horizon reference
                # -- the viewer can always see which way is down.
                grid = gl.GLGridItem()
                grid.setSize(6, 6)
                grid.setSpacing(0.5, 0.5)
                grid.translate(0, 0, -1.6)
                grid.setColor(QColor(60, 76, 96, 190))
                view.addItem(grid)

                # World axes at the origin: X red, Y green, Z blue (up).
                for index, color in enumerate(AXIS_COLORS):
                    end = [0.0, 0.0, 0.0]
                    end[index] = 1.9 if index == 2 else 1.5
                    line = gl.GLLinePlotItem(
                        pos=np.array([[0.0, 0.0, -1.6],
                                      [end[0], end[1], end[2] - 1.6]]),
                        color=color, width=2.0, antialias=True,
                    )
                    view.addItem(line)

                self.view = view
                self.gl_available = True
                self._set_mesh("ROCKET")
                return view
            except Exception as exc:  # pragma: no cover - driver dependent
                reason = str(exc)
        else:
            reason = _GL_IMPORT_ERROR or "pyqtgraph.opengl unavailable"

        self.fallback = _AttitudeFallbackView()
        self.fallback.reason = reason
        self.gl_available = False
        return self.fallback

    def _build_readouts(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self.rpy_label = QLabel("R --  P --  Y --")
        rpy_font = QFont("Consolas")
        rpy_font.setPointSize(10)
        rpy_font.setBold(True)
        self.rpy_label.setFont(rpy_font)
        self.rpy_label.setStyleSheet("color: %s;" % COL_TEXT)
        row.addWidget(self.rpy_label, 1)

        self.ref_label = QLabel("GYRO")
        ref_font = QFont()
        ref_font.setPointSize(8)
        ref_font.setBold(True)
        self.ref_label.setFont(ref_font)
        self.ref_label.setAlignment(Qt.AlignCenter)
        # Fixed width so the lamp never squeezes the RESET button off the row.
        self.ref_label.setFixedWidth(74)
        self.ref_label.setToolTip(
            "GRAVITY LOCK: the accelerometer reads close to 1 g, so it is being\n"
            "used to correct gyro drift in roll and pitch.\n"
            "GYRO ONLY: the vehicle is accelerating (boost, deployment shock), so\n"
            "the accelerometer is not a valid gravity reference and the estimate\n"
            "is coasting on the gyroscope."
        )
        self._style_ref(False)
        row.addWidget(self.ref_label)

        self.reset_btn = QPushButton("RESET")
        self.reset_btn.setFixedWidth(72)
        self.reset_btn.setToolTip(
            "Re-zero the orientation estimate.\n"
            "Yaw has no absolute reference without a magnetometer, so it drifts;\n"
            "reset it between test runs."
        )
        self.reset_btn.clicked.connect(self.reset_orientation)
        row.addWidget(self.reset_btn)
        return row

    def _set_mesh(self, payload_type: str) -> None:
        """Swap the displayed model between the rocket and the CanSat shape."""
        kind = "CANSAT" if payload_type == "CANSAT" else "ROCKET"
        if kind == self._mesh_kind or not self.gl_available:
            self._mesh_kind = kind
            return
        self._mesh_kind = kind

        if self._model is not None:
            try:
                self.view.removeItem(self._model)
            except Exception:
                pass
            self._model = None

        verts, faces, face_colors = (build_cansat_mesh() if kind == "CANSAT"
                                     else build_rocket_mesh())
        # drawEdges is deliberately off: on a 26-segment cylinder it renders
        # every triangle diagonal and the model reads as a hairy barrel rather
        # than a vehicle.  Shading plus the coloured nose carries the form.
        mesh = gl.GLMeshItem(
            vertexes=verts, faces=faces, faceColors=face_colors,
            smooth=False, drawEdges=False,
            shader="shaded", glOptions="opaque",
        )
        self.view.addItem(mesh)
        self._model = mesh

    # -- telemetry slot (hot path: keep cheap) -----------------------------

    def on_packet(self, packet) -> None:
        """Feed one validated packet into the estimator.

        Called from the GUI thread via the same queued signal the rest of the
        dashboard uses.  Does arithmetic only -- no repaint here.
        """
        try:
            payload = getattr(packet, "payload_type", "") or ""
            if payload != self._payload_type:
                self._payload_type = payload
                self._set_mesh(payload)

            timestamp = packet.mission_time_s
            if not math.isfinite(timestamp):
                timestamp = packet.gs_recv_epoch

            self.estimator.update(
                (packet.gyro_x, packet.gyro_y, packet.gyro_z),
                (packet.acc_x, packet.acc_y, packet.acc_z),
                timestamp,
            )
            self._dirty = True
        except Exception:
            # An attitude display fault must never interrupt ingestion.
            pass

    def reset_orientation(self) -> None:
        """Re-zero the estimate; clears accumulated yaw drift."""
        self.estimator.reset()
        self._dirty = True
        self.redraw(force=True)

    def clear(self) -> None:
        """Session reset hook, called by the dashboard's Clear button."""
        self.reset_orientation()

    # -- rendering (timer driven) ------------------------------------------

    def redraw(self, force: bool = False) -> None:
        """Apply the current estimate to the 3D model.  Called by a QTimer."""
        if not (self._dirty or force):
            return
        self._dirty = False
        try:
            roll, pitch, yaw = self.estimator.euler_deg
            self.rpy_label.setText(
                "R %+7.1f  P %+7.1f  Y %+7.1f" % (roll, pitch, yaw)
            )
            self._style_ref(self.estimator.accel_valid)

            if self.gl_available and self._model is not None:
                transform = np.eye(4)
                transform[:3, :3] = self.estimator.matrix
                # GLMeshItem wants a row-major 4x4 in flat form.
                self._model.resetTransform()
                self._model.applyTransform(_as_qmatrix(transform), local=False)
            elif self.fallback is not None:
                self.fallback.set_attitude(roll, pitch, yaw)
        except Exception:
            pass

    def _style_ref(self, locked: bool) -> None:
        if locked:
            self.ref_label.setText("G-LOCK")
            self.ref_label.setStyleSheet(
                "color:#0b1219; background:#35c46b; border-radius:3px; padding:2px 4px;"
            )
        else:
            self.ref_label.setText("GYRO")
            self.ref_label.setStyleSheet(
                "color:#0b1219; background:#e9c135; border-radius:3px; padding:2px 4px;"
            )


def _as_qmatrix(transform: np.ndarray):
    """Convert a 4x4 numpy array into the QMatrix4x4 pyqtgraph expects."""
    from PyQt5.QtGui import QMatrix4x4
    return QMatrix4x4(*[float(v) for v in transform.flatten()])

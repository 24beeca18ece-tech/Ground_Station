"""
diagnostics_window.py
=====================

Two raw-diagnostic views for hardware bring-up and range testing, kept separate
from the styled summary panels on the main dashboard:

* :class:`DiagnosticsWindow` -- every field of the most recent packet as a plain
  ``label : value`` table, plus live link statistics.  A debug console view, not
  a flight display.
* :class:`RawPacketStrip` -- a single line showing the most recent raw frame
  exactly as it came off the wire, corrupt frames included.

WHY A SEPARATE WINDOW RATHER THAN A TAB
---------------------------------------
The table is 26 value rows plus 5 section headers.  The Event log / Settings tab
strip at the bottom right of the dashboard is roughly 200 px tall, so putting the
table there would need scrolling -- which is exactly what this dashboard has
already been reorganised to avoid.  A separate non-modal window can be sized to
show every row at once, can be moved to a second monitor during bring-up, and
can be closed without perturbing the main layout at all.

SCOPE
-----
Rocket-specific fields only.  The CanSat payload channels (SPS30 particulates,
reaction-wheel RPM, recovery stage) are deliberately **not** surfaced here --
they are handled by a separate ground station application.  They remain
untouched everywhere else in this app.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from telemetry_packet import FSM_COLORS, FSM_UNKNOWN_COLOR

# ---------------------------------------------------------------------------
# Palette (matched to dashboard_ui.py)
# ---------------------------------------------------------------------------

COL_BG = "#0a0e13"
COL_PANEL = "#12171f"
COL_BORDER = "#2b3746"
COL_TEXT = "#dbe3ee"
COL_DIM = "#8b9aad"
COL_HEADER = "#4aa8ff"
COL_NUM = "#7fd6ff"
COL_OK = "#35c46b"
COL_WARN = "#e9c135"
COL_ALERT = "#e8384f"

MONO = "Consolas, 'DejaVu Sans Mono', monospace"

#: Longest raw frame rendered in the strip before hard truncation.  Elision
#: handles the display width; this only bounds the work done per packet.
RAW_STRIP_MAX_CHARS = 400


def _sanitise_raw(raw: str) -> str:
    """Make an arbitrary received frame safe for a one-line label.

    A corrupt frame can contain newlines, NULs and other control bytes.  Those
    would either break the single-line guarantee or render as boxes, so they are
    replaced with a visible placeholder.  The frame is otherwise untouched --
    this view exists to show exactly what arrived.
    """
    if not raw:
        return ""
    text = raw[:RAW_STRIP_MAX_CHARS]
    return "".join(ch if 32 <= ord(ch) < 127 else "·" for ch in text)


# ---------------------------------------------------------------------------
# Raw packet strip
# ---------------------------------------------------------------------------

class RawPacketStrip(QFrame):
    """One line showing the most recent raw frame, overwritten in place.

    Deliberately not a log: there is exactly one line, replaced on every frame.
    Its job is answering "is anything arriving on the wire at all?" during
    bring-up, before parsing correctness matters.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setFixedHeight(24)
        self.setStyleSheet(
            "QFrame { background-color: %s; border-top: 1px solid %s; }"
            % (COL_BG, COL_BORDER)
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(8)

        font = QFont("Consolas")
        font.setPointSize(9)

        self.tag = QLabel("NO RX")
        self.tag.setFont(font)
        self.tag.setFixedWidth(74)
        self.tag.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.tag)

        self.text = QLabel("waiting for data…")
        self.text.setFont(font)
        # Ignored so a long frame cannot force the whole window wider.
        self.text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.text.setStyleSheet("color: %s; background: transparent;" % COL_DIM)
        layout.addWidget(self.text, 1)

        self._raw = ""
        self._placeholder = "waiting for data…"
        self._set_tag("NO RX", COL_DIM, "#1b232e")

    # -- API ---------------------------------------------------------------

    def show_packet(self, raw: str) -> None:
        """Display a frame that passed checksum validation."""
        self._raw = _sanitise_raw(raw)
        self._set_tag("RX", "#0b1219", COL_OK)
        self.text.setStyleSheet("color: %s; background: transparent;" % COL_TEXT)
        self._apply_elision()

    def show_corrupt(self, raw: str) -> None:
        """Display a frame that failed validation, tagged so it is unmistakable."""
        self._raw = _sanitise_raw(raw)
        self._set_tag("CORRUPT", "#ffffff", COL_ALERT)
        self.text.setStyleSheet("color: %s; background: transparent;" % COL_WARN)
        self._apply_elision()

    def show_rejected(self, raw: str) -> None:
        """Display a frame that arrived intact but carries impossible values.

        Tagged differently from CORRUPT on purpose: the bytes are fine, so the
        link is healthy and the fault is upstream in a sensor.
        """
        self._raw = _sanitise_raw(raw)
        self._set_tag("REJECTED", "#0b1219", COL_WARN)
        self.text.setStyleSheet("color: %s; background: transparent;" % COL_WARN)
        self._apply_elision()

    def clear(self) -> None:
        self._raw = ""
        self._set_tag("NO RX", COL_DIM, "#1b232e")
        self.text.setStyleSheet("color: %s; background: transparent;" % COL_DIM)
        self.text.setText(self._placeholder)

    # -- internals ---------------------------------------------------------

    def _set_tag(self, text: str, fg: str, bg: str) -> None:
        self.tag.setText(text)
        self.tag.setStyleSheet(
            "color: %s; background: %s; border-radius: 3px; font-weight: 700;"
            % (fg, bg)
        )

    def _apply_elision(self) -> None:
        """Fit the frame to the current width, ellipsising the tail."""
        if not self._raw:
            self.text.setText(self._placeholder)
            return
        metrics = QFontMetrics(self.text.font())
        width = max(self.text.width() - 4, 40)
        self.text.setText(metrics.elidedText(self._raw, Qt.ElideRight, width))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._apply_elision()


# ---------------------------------------------------------------------------
# Diagnostics table
# ---------------------------------------------------------------------------

#: (key, caption) rows grouped under section headers.  Order is the wire order
#: of the packet, then the link statistics, so the table reads top to bottom the
#: same way the frame does.
SECTIONS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("TELEMETRY", [
        ("mission_time", "MISSION TIME"),
        ("timestamp", "TIMESTAMP"),
        ("packet_count", "PACKET COUNT"),
        ("altitude", "ALTITUDE"),
        ("pressure", "PRESSURE"),
        ("temperature", "TEMPERATURE"),
        ("voltage", "VOLTAGE"),
    ]),
    ("NAVIGATION", [
        ("nav_time", "NAV TIME"),
        ("lat", "LATITUDE"),
        ("lon", "LONGITUDE"),
        ("nav_alt", "NAV ALTITUDE"),
        ("sats", "SATELLITES"),
    ]),
    ("INERTIAL", [
        ("acc_x", "ACCEL X"),
        ("acc_y", "ACCEL Y"),
        ("acc_z", "ACCEL Z"),
        ("gyro_x", "GYRO X"),
        ("gyro_y", "GYRO Y"),
        ("gyro_z", "GYRO Z"),
    ]),
    ("FLIGHT STATE / RECOVERY", [
        ("fsm", "FSM STATE"),
        ("solenoid", "SOLENOID FIRED"),
        ("nichrome", "NICHROME FIRED"),
    ]),
    ("LINK DIAGNOSTICS", [
        ("rate", "PACKET RATE"),
        ("age", "PACKET AGE"),
        ("valid_total", "VALID/TOTAL"),
        ("corrupt", "CORRUPT (CHECKSUM)"),
        ("rejected", "REJECTED (BOUNDS)"),
        ("conn", "CONNECTION STATE"),
    ]),
]


class DiagnosticsWindow(QDialog):
    """Non-modal raw telemetry table.

    Updates are applied directly in :meth:`update_packet` with no throttling --
    they are plain ``setText`` calls, not a chart redraw.  The dashboard skips
    the call entirely while the window is hidden, so a closed diagnostics view
    costs nothing on the packet hot path.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Qt.Window gives it a real title bar and taskbar presence, and keeps it
        # non-modal so the dashboard stays fully interactive behind it.
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Telemetry Diagnostics")
        self.setStyleSheet("QDialog { background-color: %s; }" % COL_PANEL)
        self.resize(430, 780)

        self._value_labels: Dict[str, QLabel] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(4)

        subtitle = QLabel("Raw field dump of the most recent packet")
        sub_font = QFont()
        sub_font.setPointSize(8)
        subtitle.setFont(sub_font)
        subtitle.setStyleSheet("color: %s;" % COL_DIM)
        outer.addWidget(subtitle)

        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 0)

        name_font = QFont("Consolas")
        name_font.setPointSize(9)
        value_font = QFont("Consolas")
        value_font.setPointSize(9)
        value_font.setBold(True)
        header_font = QFont("Consolas")
        header_font.setPointSize(8)
        header_font.setBold(True)

        row = 0
        for section, fields in SECTIONS:
            header = QLabel(section)
            header.setFont(header_font)
            header.setStyleSheet(
                "color: %s; background: #172231; padding: 3px 6px;"
                " border-radius: 3px; letter-spacing: 1px;" % COL_HEADER
            )
            grid.addWidget(header, row, 0, 1, 2)
            row += 1

            for key, caption in fields:
                name = QLabel(caption)
                name.setFont(name_font)
                name.setStyleSheet("color: %s;" % COL_DIM)
                name.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                grid.addWidget(name, row, 0)

                value = QLabel("--")
                value.setFont(value_font)
                value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                value.setStyleSheet("color: %s;" % COL_DIM)
                value.setTextInteractionFlags(Qt.TextSelectableByMouse)
                grid.addWidget(value, row, 1)

                self._value_labels[key] = value
                row += 1

            spacer = QLabel("")
            spacer.setFixedHeight(6)
            grid.addWidget(spacer, row, 0, 1, 2)
            row += 1

        outer.addLayout(grid)
        outer.addStretch(1)

    # -- helpers -----------------------------------------------------------

    def _set(self, key: str, text: str, color: str = COL_NUM) -> None:
        label = self._value_labels.get(key)
        if label is not None:
            label.setText(text)
            label.setStyleSheet("color: %s;" % color)

    @staticmethod
    def _num(value: float, digits: int, unit: str = "") -> Tuple[str, str]:
        """Format a number, flagging non-finite values as missing data."""
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return "NO DATA", COL_WARN
        text = ("%%.%df" % digits) % value
        return (text + (" " + unit if unit else "")), COL_NUM

    # -- updates -----------------------------------------------------------

    def update_packet(self, packet, voltage_warn: float = 7.0) -> None:
        """Refresh every telemetry row from one validated packet."""
        try:
            self.setWindowTitle(
                "Telemetry Diagnostics — %s / %s"
                % (packet.team_id, packet.payload_type)
            )

            self._set("mission_time", packet.mission_time_hms, COL_TEXT)
            self._set("timestamp", packet.timestamp_raw or "--", COL_NUM)
            self._set("packet_count", "%d" % packet.packet_count, COL_NUM)

            self._set("altitude", *self._num(packet.altitude_m, 2, "m"))
            self._set("pressure", *self._num(packet.pressure_hpa, 2, "hPa"))
            self._set("temperature", *self._num(packet.temp_c, 2, "°C"))

            voltage, color = self._num(packet.voltage_v, 2, "V")
            if math.isfinite(packet.voltage_v):
                if packet.voltage_v < voltage_warn * 0.9:
                    color = COL_ALERT
                elif packet.voltage_v < voltage_warn:
                    color = COL_WARN
                else:
                    color = COL_OK
            self._set("voltage", voltage, color)

            self._set("nav_time", packet.nav_time or "--",
                      COL_NUM if packet.nav_time else COL_WARN)

            # A GPS field of 0.0 with no fix must never read like a real
            # coordinate -- say NO FIX instead, in a warning colour.
            if packet.has_fix:
                self._set("lat", "%.6f°" % packet.lat, COL_NUM)
                self._set("lon", "%.6f°" % packet.lon, COL_NUM)
            else:
                self._set("lat", "NO FIX", COL_WARN)
                self._set("lon", "NO FIX", COL_WARN)

            self._set("nav_alt", *self._num(packet.nav_alt_m, 2, "m"))

            sats = packet.sats
            sat_color = COL_OK if sats >= 6 else (COL_WARN if sats >= 4 else COL_ALERT)
            self._set("sats", "%d" % sats, sat_color)

            for key, value in (("acc_x", packet.acc_x), ("acc_y", packet.acc_y),
                               ("acc_z", packet.acc_z)):
                self._set(key, *self._num(value, 3, "m/s²"))
            for key, value in (("gyro_x", packet.gyro_x), ("gyro_y", packet.gyro_y),
                               ("gyro_z", packet.gyro_z)):
                self._set(key, *self._num(value, 3, "°/s"))

            self._set("fsm", "%s (%d)" % (packet.fsm_name, packet.fsm_state),
                      FSM_COLORS.get(packet.fsm_state, FSM_UNKNOWN_COLOR))

            # Rocket-only recovery flags.  A CanSat or legacy packet does not
            # carry them, and showing "SAFE" there would be a lie.
            if packet.is_rocket:
                for key, fired in (("solenoid", packet.solenoid_fired),
                                   ("nichrome", packet.nichrome_fired)):
                    self._set(key, "FIRED" if fired else "SAFE",
                              COL_ALERT if fired else COL_OK)
            else:
                self._set("solenoid", "n/a", COL_DIM)
                self._set("nichrome", "n/a", COL_DIM)
        except Exception:
            # A diagnostics view must never disturb the ingestion path.
            pass

    def update_link(self, rate: float, age: Optional[float], valid: int,
                    total: int, corrupt: int, connected: bool,
                    stale_after: float = 2.0, rejected: int = 0) -> None:
        """Refresh the link statistics rows."""
        try:
            self._set("rate", "%.1f pkt/s" % rate,
                      COL_NUM if rate > 0 else COL_WARN)

            if age is None:
                self._set("age", "--", COL_DIM)
            else:
                self._set("age", "%.1f s" % age,
                          COL_ALERT if age > stale_after else COL_OK)

            self._set("valid_total", "%d / %d" % (valid, total), COL_NUM)
            self._set("corrupt", "%d" % corrupt,
                      COL_ALERT if corrupt else COL_OK)
            # Distinct from CORRUPT: the link delivered these frames perfectly,
            # a sensor produced impossible numbers inside them.
            self._set("rejected", "%d" % rejected,
                      COL_WARN if rejected else COL_OK)
            self._set("conn", "CONNECTED" if connected else "DISCONNECTED",
                      COL_OK if connected else COL_ALERT)
        except Exception:
            pass

    def clear(self) -> None:
        """Blank every row, e.g. after a sensor-data reset."""
        for label in self._value_labels.values():
            label.setText("--")
            label.setStyleSheet("color: %s;" % COL_DIM)
        self.setWindowTitle("Telemetry Diagnostics")

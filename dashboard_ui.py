"""
dashboard_ui.py
===============

The ground control station main window: every widget, every plot, every layout.

THREADING MODEL
---------------
Everything in this module lives on the **GUI thread**.  It never opens a serial
port and never writes a file directly.  It owns the two worker threads and talks
to them only through their documented interfaces:

    SerialWorker.packet_received  --(queued Qt signal)-->  Dashboard.on_packet
    SerialWorker.bad_frame        --(queued Qt signal)-->  Dashboard.on_bad_frame
    Dashboard                     --(thread-safe queue)-->  CsvLoggerThread

RENDER DECOUPLING — the reason the UI never freezes
---------------------------------------------------
``on_packet`` is deliberately cheap: it appends numbers to Python lists and
returns.  It does **not** repaint anything.  All repainting happens on a
``QTimer`` at :data:`RENDER_HZ`, so whether the radio delivers 5 packets/s or
dumps a 500-packet burst after a dropout, the number of repaints per second is
constant.  This is what lets the dashboard tolerate the "up to 20 Hz plus
bursts" requirement without the event loop falling behind.
"""

from __future__ import annotations

import bisect
import math
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QStackedWidget,
    QTabWidget,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from attitude_widget import ATTITUDE_RENDER_HZ, AttitudeWidget
from csv_logger import CsvLoggerThread
from diagnostics_window import DiagnosticsWindow, RawPacketStrip
from serial_worker import SerialWorker, list_serial_ports
from session_summary_widget import SessionSummaryWidget
from telemetry_packet import (
    FSM_COLORS,
    FSM_STATES,
    PAYLOAD_CANSAT,
    PAYLOAD_GENERIC,
    PAYLOAD_ROCKET,
    RECOVERY_STAGE_COLORS,
    RECOVERY_STAGES,
    TelemetryPacket,
)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

RENDER_HZ = 15                 # plot / readout repaint rate
STATUS_HZ = 5                  # link-status (staleness, rate) refresh rate
STALE_AFTER_S = 2.0            # telemetry considered stale after this gap
RATE_WINDOW_S = 3.0            # averaging window for packets/sec
MAX_PLOT_POINTS = 40000        # hard cap per series (~33 min at 20 Hz)
DEFAULT_WINDOW_S = 60          # visible strip-chart width in seconds
DEFAULT_VOLTAGE_WARN = 7.0     # battery warning threshold, volts
#: A mission time this much lower than the previous sample is treated as a
#: time-base restart (flight-computer reboot, restarted transmitter, or a
#: different vehicle on the link) rather than jitter.
TIME_RESET_TOLERANCE_S = 0.5
#: Gap inserted in the plot clock across a detected restart.
PLOT_TIME_GAP_S = 0.01
EVENT_LOG_LINES = 400

BAUD_RATES = ["9600", "19200", "38400", "57600", "115200", "230400", "921600"]

#: Convenience entry so the dashboard can talk to ``packet_sim.py`` over TCP
#: without any virtual-COM-port driver installed (pyserial URL handler).
SIM_PORT_URL = "socket://127.0.0.1:5555"

# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------

COL_BG = "#0e1218"
COL_PANEL = "#161d27"
COL_PANEL_HI = "#1d2633"
COL_BORDER = "#2b3746"
COL_TEXT = "#dbe3ee"
COL_TEXT_DIM = "#8b9aad"
COL_ACCENT = "#4aa8ff"
COL_OK = "#35c46b"
COL_WARN = "#e9c135"
COL_ALERT = "#e8384f"

# Per-quantity trace colours, reused between the plots and the readout tiles.
COL_ALT = "#4aa8ff"
COL_PRESS = "#ffb74d"
COL_TEMP = "#ff7a7a"
COL_VOLT = "#7ee787"
COL_XYZ = ("#ff6b6b", "#4ade80", "#60a5fa")  # X, Y, Z

# Vehicle-specific payload colours.
COL_PM1 = "#5ad2f4"    # SPS30 PM1.0
COL_PM25 = "#f4b73f"   # SPS30 PM2.5
COL_PM10 = "#f4685a"   # SPS30 PM10
COL_WHEEL = "#c79bff"  # reaction wheel RPM

DARK_QSS = f"""
QWidget {{
    background-color: {COL_BG};
    color: {COL_TEXT};
    font-family: 'Segoe UI', 'DejaVu Sans', sans-serif;
    font-size: 11pt;
}}
QMainWindow, QScrollArea, QSplitter {{ background-color: {COL_BG}; }}
QGroupBox {{
    background-color: {COL_PANEL};
    border: 1px solid {COL_BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {COL_ACCENT};
    letter-spacing: 1px;
}}
QLabel {{ background: transparent; }}
QPushButton {{
    background-color: {COL_PANEL_HI};
    border: 1px solid {COL_BORDER};
    border-radius: 5px;
    padding: 7px 16px;
    font-weight: 600;
}}
QPushButton:hover  {{ background-color: #26313f; border-color: {COL_ACCENT}; }}
QPushButton:pressed{{ background-color: #10161e; }}
QPushButton:disabled {{ color: #5a6675; border-color: #222b36; }}
QPushButton#connectBtn[connected="true"] {{
    background-color: {COL_ALERT}; border-color: {COL_ALERT}; color: #ffffff;
}}
QPushButton#connectBtn[connected="false"] {{
    background-color: #1f6f3f; border-color: #2c8f52; color: #ffffff;
}}
QPushButton#logBtn[logging="true"] {{
    background-color: #8a5a00; border-color: #b87700; color: #ffffff;
}}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {COL_PANEL_HI};
    border: 1px solid {COL_BORDER};
    border-radius: 5px;
    padding: 5px 8px;
    selection-background-color: {COL_ACCENT};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {COL_ACCENT}; }}
QComboBox QAbstractItemView {{
    background-color: {COL_PANEL_HI};
    border: 1px solid {COL_BORDER};
    selection-background-color: {COL_ACCENT};
    selection-color: #06121f;
}}
QPlainTextEdit {{
    background-color: #0a0e13;
    border: 1px solid {COL_BORDER};
    border-radius: 5px;
    font-family: 'Consolas', 'DejaVu Sans Mono', monospace;
    font-size: 9pt;
    color: {COL_TEXT_DIM};
}}
QCheckBox {{ spacing: 8px; }}
QScrollBar:vertical {{
    background: {COL_BG}; width: 12px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {COL_BORDER}; border-radius: 6px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #3d4c5e; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QSplitter::handle {{ background: {COL_BORDER}; width: 3px; }}
QTabWidget::pane {{
    background: {COL_PANEL};
    border: 1px solid {COL_BORDER};
    border-radius: 6px;
    top: -1px;
}}
QTabBar::tab {{
    background: {COL_BG};
    color: {COL_TEXT_DIM};
    border: 1px solid {COL_BORDER};
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 5px 14px;
    margin-right: 2px;
    font-weight: 600;
}}
QTabBar::tab:selected {{ background: {COL_PANEL}; color: {COL_ACCENT}; }}
QTabBar::tab:hover {{ color: {COL_TEXT}; }}
"""


# ---------------------------------------------------------------------------
# Small reusable widgets
# ---------------------------------------------------------------------------

class ReadoutTile(QFrame):
    """A single labelled numeric readout ("ALTITUDE  1234.5 m").

    ``set_level`` recolours the value for normal / warning / alert conditions —
    used for battery voltage, satellite count and link staleness.
    """

    LEVEL_COLORS = {"normal": COL_TEXT, "ok": COL_OK, "warn": COL_WARN, "alert": COL_ALERT}

    def __init__(self, caption: str, unit: str = "", value_color: str = COL_TEXT,
                 value_pt: int = 17, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(
            "QFrame { background-color: %s; border: 1px solid %s; border-radius: 5px; }"
            % (COL_PANEL_HI, COL_BORDER)
        )
        self._base_color = value_color
        self._unit = unit

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 5)
        layout.setSpacing(0)

        self.caption = QLabel(caption.upper())
        caption_font = QFont()
        caption_font.setPointSize(8)
        caption_font.setBold(True)
        self.caption.setFont(caption_font)
        self.caption.setStyleSheet("color: %s;" % COL_TEXT_DIM)

        self.value = QLabel("--")
        value_font = QFont("Consolas")
        value_font.setPointSize(value_pt)
        value_font.setBold(True)
        self.value.setFont(value_font)
        self.value.setStyleSheet("color: %s;" % value_color)

        # Ignored horizontal policy: a QLabel otherwise reports its full
        # text width as a hard minimum, and eight of them side by side pin the
        # sidebar to a width no 1366 px laptop can satisfy.
        for label in (self.caption, self.value):
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.caption)
        layout.addWidget(self.value)

    def set_value(self, text: str) -> None:
        self.value.setText("%s %s" % (text, self._unit) if self._unit else text)

    def set_level(self, level: str) -> None:
        color = self.LEVEL_COLORS.get(level, self._base_color)
        if level == "normal":
            color = self._base_color
        self.value.setStyleSheet("color: %s;" % color)


class StatusLight(QFrame):
    """A labelled indicator lamp for a latching event (pyro fired / safe).

    Deliberately shows three visual states rather than two: a lamp that is dark
    because no telemetry has arrived must not look the same as a lamp that is
    dark because the charge has not fired.
    """

    def __init__(self, caption: str, fired_text: str = "FIRED",
                 safe_text: str = "SAFE", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(
            "QFrame { background-color: %s; border: 1px solid %s; border-radius: 5px; }"
            % (COL_PANEL_HI, COL_BORDER)
        )
        self._fired_text = fired_text
        self._safe_text = safe_text

        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 6, 9, 7)
        layout.setSpacing(3)

        caption_label = QLabel(caption.upper())
        caption_font = QFont()
        caption_font.setPointSize(8)
        caption_font.setBold(True)
        caption_label.setFont(caption_font)
        caption_label.setStyleSheet("color: %s;" % COL_TEXT_DIM)
        layout.addWidget(caption_label)

        self.lamp = QLabel("NO DATA")
        lamp_font = QFont()
        lamp_font.setPointSize(11)
        lamp_font.setBold(True)
        self.lamp.setFont(lamp_font)
        self.lamp.setAlignment(Qt.AlignCenter)
        self.lamp.setMinimumHeight(30)
        for label in (caption_label, self.lamp):
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.lamp)

        self.set_state(None)

    def set_state(self, fired: Optional[bool]) -> None:
        if fired is None:
            text, bg, fg = "NO DATA", "#242c38", COL_TEXT_DIM
        elif fired:
            text, bg, fg = self._fired_text, COL_ALERT, "#ffffff"
        else:
            text, bg, fg = self._safe_text, "#1f6f3f", "#ffffff"
        self.lamp.setText(text)
        self.lamp.setStyleSheet(
            "background-color: %s; color: %s; border-radius: 4px;" % (bg, fg)
        )


class StripChart(pg.PlotWidget):
    """Scrolling time-series plot holding its own data buffers.

    Data is appended by :meth:`add_point` (cheap, called per packet) and drawn by
    :meth:`redraw` (called by the render timer).  The two are separate so packet
    arrival rate and repaint rate are independent.

    X axis is *mission time* in seconds taken from the packet TIMESTAMP field,
    falling back to ground-station elapsed time when the flight computer sends a
    timestamp that cannot be interpreted.
    """

    def __init__(self, title: str, y_label: str,
                 series: Sequence[Tuple[str, str]],
                 parent: Optional[QWidget] = None,
                 legend: bool = True) -> None:
        super().__init__(parent)
        self.setTitle(title, color=COL_TEXT, size="10pt")
        self.setLabel("left", y_label, color=COL_TEXT_DIM)
        self.setLabel("bottom", "mission time", units="s", color=COL_TEXT_DIM)
        self.showGrid(x=True, y=True, alpha=0.22)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=True, y=True)
        self.setMinimumHeight(118)
        # A plot inside the scrollable left column must not take focus: Qt
        # responds by calling ensureWidgetVisible on the QScrollArea, which
        # silently scrolls the payload readouts out of view.
        self.setFocusPolicy(Qt.NoFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.getPlotItem().setContentsMargins(4, 4, 10, 4)

        self._x: List[float] = []
        #: Count of samples whose x had to be clamped to keep the series
        #: non-decreasing. Non-zero means a caller passed an unordered clock.
        self.bad_x = 0
        self._series_names: List[str] = [name for name, _ in series]
        self._y: Dict[str, List[float]] = {name: [] for name, _ in series}
        self._curves: Dict[str, pg.PlotDataItem] = {}

        if len(series) > 1 and legend:
            # Anchored top-left (clear of the Y tick labels): the right-hand
            # corner is where the newest samples arrive and would be covered.
            self.addLegend(
                offset=(58, 10), labelTextColor=COL_TEXT_DIM,
                verSpacing=-3, brush=pg.mkBrush(20, 26, 35, 190),
            )

        for name, color in series:
            curve = self.plot(
                [], [],
                pen=pg.mkPen(color, width=2),
                name=name if (len(series) > 1 and legend) else None,
                autoDownsample=True,
                clipToView=True,
            )
            self._curves[name] = curve

    # -- data ---------------------------------------------------------------

    def add_point(self, x: float, values: Dict[str, float]) -> None:
        """Append one sample.  Non-finite values are stored as NaN (gap in line).

        The x series is kept non-decreasing. Callers are expected to pass the
        dashboard's shared monotonic plot clock, but a chart must not be able to
        render garbage if one ever does not: a single backward x makes the
        polyline double back across the plot and breaks both the bisect window
        search here and pyqtgraph's clipToView, which is exactly the "scattered
        and disordered" failure this guard exists to prevent.
        """
        if not math.isfinite(x):
            self.bad_x += 1
            x = self._x[-1] if self._x else 0.0
        elif self._x and x < self._x[-1]:
            self.bad_x += 1
            x = self._x[-1]
        self._x.append(x)
        for name in self._series_names:
            value = values.get(name, float("nan"))
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                value = float("nan")
            self._y[name].append(float(value))

        if len(self._x) > MAX_PLOT_POINTS:
            drop = len(self._x) - MAX_PLOT_POINTS
            del self._x[:drop]
            for name in self._series_names:
                del self._y[name][:drop]

    def clear_data(self) -> None:
        self._x.clear()
        self.bad_x = 0
        for name in self._series_names:
            self._y[name].clear()
            self._curves[name].setData([], [])

    # -- rendering ----------------------------------------------------------

    def redraw(self, window_s: float, autoscale_y: bool = True) -> None:
        """Redraw the trailing *window_s* seconds of data."""
        if not self._x:
            return
        latest = self._x[-1]
        # Before the buffer holds a full window, show everything from the first
        # sample rather than padding the view with empty pre-launch time.
        cutoff = max(latest - float(window_s), self._x[0])
        # bisect is valid because add_point guarantees _x is non-decreasing.
        # Without that guarantee this search returns a meaningless index and
        # pyqtgraph's clipToView (which also assumes sorted x) mis-slices the
        # series, scattering the trace.
        start = bisect.bisect_left(self._x, cutoff)
        if start >= len(self._x):
            start = max(0, len(self._x) - 1)

        xs = self._x[start:]
        for name in self._series_names:
            self._curves[name].setData(xs, self._y[name][start:])

        self.setXRange(cutoff, max(latest, cutoff + 1e-3), padding=0.01)
        if autoscale_y:
            self.enableAutoRange(axis="y")
        else:
            self.disableAutoRange(axis="y")


class GpsTrackPlot(pg.PlotWidget):
    """Lat/Lon ground track: a faint path plus a bright marker at the last fix."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Ground track", color=COL_TEXT, size="10pt")
        self.setLabel("left", "latitude", units="°", color=COL_TEXT_DIM)
        self.setLabel("bottom", "longitude", units="°", color=COL_TEXT_DIM)
        self.showGrid(x=True, y=True, alpha=0.22)
        self.setMenuEnabled(False)
        self.setMinimumHeight(118)
        self.setFocusPolicy(Qt.NoFocus)   # see StripChart: keeps the column put

        self._lats: List[float] = []
        self._lons: List[float] = []

        self._track = self.plot(
            [], [], pen=pg.mkPen("#3f7fbf", width=1.5),
            symbol="o", symbolSize=3.5,
            symbolBrush=pg.mkBrush("#2f6fa8"), symbolPen=None,
        )
        self._current = self.plot(
            [], [], pen=None, symbol="+", symbolSize=16,
            symbolPen=pg.mkPen(COL_ALERT, width=2.5),
        )
        self._dirty = False

    def add_fix(self, lat: float, lon: float) -> None:
        self._lats.append(lat)
        self._lons.append(lon)
        if len(self._lats) > MAX_PLOT_POINTS:
            del self._lats[:len(self._lats) - MAX_PLOT_POINTS]
            del self._lons[:len(self._lons) - MAX_PLOT_POINTS]
        self._dirty = True

    def clear_data(self) -> None:
        self._lats.clear()
        self._lons.clear()
        self._track.setData([], [])
        self._current.setData([], [])
        self._dirty = False

    def redraw(self) -> None:
        if not self._dirty or not self._lats:
            return
        self._dirty = False
        self._track.setData(self._lons, self._lats)
        self._current.setData([self._lons[-1]], [self._lats[-1]])


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class Dashboard(QMainWindow):
    """The GCS main window.  Owns the serial thread and the CSV logger thread."""

    def __init__(self, log_dir: str = "logs") -> None:
        super().__init__()
        self.setWindowTitle("Ground Control Station — CanSat / Rocketry Telemetry")
        self.resize(1680, 980)
        self.setMinimumSize(1180, 720)

        # --- live state (GUI thread only) ----------------------------------
        self.latest: Optional[TelemetryPacket] = None
        self.first_mission_time: Optional[float] = None
        # Shared monotonic plot clock -- see _plot_time().
        self._last_mission_time: Optional[float] = None
        self._last_plot_time: float = 0.0
        self._plot_time_offset: float = 0.0
        self._time_resets: int = 0
        self.session_start = time.time()
        self.last_packet_epoch: Optional[float] = None
        self.recv_times: deque = deque(maxlen=1000)   # for packets/sec
        self.session_packets = 0
        self.total_frames = 0
        self.valid_packets = 0
        self.corrupt_packets = 0
        self.resyncs = 0
        self.logging_enabled = False
        self.is_connected = False
        self._readouts_dirty = False
        self._was_stale = False
        self._last_fsm: Optional[int] = None
        #: PAYLOAD_TYPE seen in the stream; drives the auto-detected panel.
        self._detected_payload: str = ""

        # --- worker threads --------------------------------------------------
        self.serial_worker = SerialWorker(self)
        self.csv_logger = CsvLoggerThread(log_dir=log_dir, parent=self)

        self._build_ui()

        # Non-modal diagnostics window, created hidden. Parented to the main
        # window so it closes with it.
        self.diagnostics = DiagnosticsWindow(self)

        self._wire_signals()

        # Logger runs for the whole session: the errors log is always active,
        # while CSV rows are only queued when the user enables logging.
        self.csv_logger.start()
        self.serial_worker.start()

        # --- timers ----------------------------------------------------------
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._render)
        self.render_timer.start(int(1000 / RENDER_HZ))

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_link_status)
        self.status_timer.start(int(1000 / STATUS_HZ))

        # The attitude model runs on its own timer: it wants a higher, smoother
        # frame rate than the strip charts, and it is cheap (one 4x4 transform).
        self.attitude_timer = QTimer(self)
        self.attitude_timer.timeout.connect(self.attitude.redraw)
        self.attitude_timer.start(int(1000 / ATTITUDE_RENDER_HZ))

        # Resolve the initial vehicle choice so the panel titles and the 3D
        # model agree before any telemetry arrives.
        self._apply_payload_panel()

        # Once the first layout pass has run, park the left column at the top.
        QTimer.singleShot(0, lambda: self._left_scroll.verticalScrollBar().setValue(0))

        self.refresh_ports()
        self.append_event("Ground station ready. Select a port and press CONNECT.")
        self.append_event(
            "No radio? Run  python packet_sim.py  and connect to %s" % SIM_PORT_URL
        )

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self) -> None:
        pg.setConfigOptions(antialias=True, background=COL_PANEL, foreground=COL_TEXT_DIM)
        self.setStyleSheet(DARK_QSS)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        root_layout.addWidget(self._build_connection_bar())

        # Right-hand side is itself split vertically: strip charts on top,
        # attitude + session summary below.  A splitter rather than a tab widget
        # so a reviewer can see the flight profile and the summary at the same
        # time, and still drag either to full height when they want detail.
        right = QSplitter(Qt.Vertical)
        right.addWidget(self._build_plot_grid())
        right.addWidget(self._build_analysis_row())
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 1)
        right.setSizes([620, 300])

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_column())
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([616, 1064])
        root_layout.addWidget(splitter, 1)

        # Single-line raw frame readout, pinned to the bottom of the window and
        # above the status bar. Shows what actually arrived on the wire, valid
        # or not, which is the first thing you want during bring-up.
        self.raw_strip = RawPacketStrip()
        root_layout.addWidget(self.raw_strip)

        self.statusBar().setStyleSheet(
            "QStatusBar { background: %s; color: %s; border-top: 1px solid %s; }"
            % (COL_PANEL, COL_TEXT_DIM, COL_BORDER)
        )
        self.statusBar().showMessage("Disconnected")

    # -- top bar ---------------------------------------------------------

    def _build_connection_bar(self) -> QWidget:
        box = QGroupBox("CONNECTION")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)  # lets you type a URL such as socket://…
        self.port_combo.setMinimumWidth(190)
        self.port_combo.setToolTip(
            "Serial device (COM7, /dev/ttyUSB0) or a pyserial URL such as\n"
            "socket://127.0.0.1:5555 for the synthetic packet generator."
        )
        layout.addWidget(self.port_combo)

        self.refresh_btn = QPushButton("RESCAN")
        self.refresh_btn.setFixedWidth(92)
        self.refresh_btn.setToolTip("Rescan available serial ports")
        layout.addWidget(self.refresh_btn)

        layout.addSpacing(6)
        layout.addWidget(QLabel("Baud:"))
        self.baud_combo = QComboBox()
        self.baud_combo.setEditable(True)
        self.baud_combo.addItems(BAUD_RATES)
        self.baud_combo.setCurrentText("9600")
        self.baud_combo.setFixedWidth(110)
        layout.addWidget(self.baud_combo)

        layout.addSpacing(10)
        self.connect_btn = QPushButton("CONNECT")
        self.connect_btn.setObjectName("connectBtn")
        self.connect_btn.setProperty("connected", "false")
        self.connect_btn.setFixedWidth(140)
        layout.addWidget(self.connect_btn)

        self.log_btn = QPushButton("START LOGGING")
        self.log_btn.setObjectName("logBtn")
        self.log_btn.setProperty("logging", "false")
        self.log_btn.setFixedWidth(170)
        layout.addWidget(self.log_btn)

        self.diag_btn = QPushButton("DIAGNOSTICS")
        self.diag_btn.setFixedWidth(148)
        self.diag_btn.setToolTip(
            "Open the raw telemetry diagnostics table in a separate window. "
            "Shows every field of the most recent packet plus link statistics."
        )
        layout.addWidget(self.diag_btn)

        self.log_path_label = QLabel("CSV: not started")
        self.log_path_label.setStyleSheet("color: %s;" % COL_TEXT_DIM)
        self.log_path_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.log_path_label.setMinimumWidth(0)
        layout.addWidget(self.log_path_label, 1)

        self.conn_state_label = QLabel("● DISCONNECTED")
        self.conn_state_label.setStyleSheet("color: %s; font-weight: 700;" % COL_ALERT)
        layout.addWidget(self.conn_state_label)

        return box

    # -- left column -----------------------------------------------------

    def _build_left_column(self) -> QWidget:
        """Left sidebar, laid out as TWO sub-columns so nothing needs scrolling.

        A single vertical stack of every panel came to roughly 1460 px, against
        about 680 px of usable height on a 1366x768 laptop — a 2:1 overshoot
        that no amount of padding trimming can close.  Splitting the sidebar
        into two side-by-side sub-columns halves the required height, and the
        two lowest-priority panels (display settings, event log) move out to the
        bottom-right tab strip entirely:

            sub-column A            sub-column B
            ------------            ------------
            Flight state            GPS / NavIC + ground track
            Live telemetry          Link status
            Payload

        Both sub-columns fit inside one screenful at 768 px, which is what the
        operator needs: flight state, telemetry, GPS, ground track and link
        status all visible at once, with no scrolling.
        """
        panel = QWidget()
        panel.setMinimumWidth(548)
        row = QHBoxLayout(panel)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)

        col_a = QWidget()
        left = QVBoxLayout(col_a)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)
        left.addWidget(self._build_fsm_banner())
        left.addWidget(self._build_readouts())
        left.addWidget(self._build_payload_panel())
        left.addStretch(1)

        col_b = QWidget()
        right = QVBoxLayout(col_b)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(6)
        right.addWidget(self._build_gps_panel())
        right.addWidget(self._build_link_panel())
        right.addStretch(1)

        col_a.setMinimumWidth(286)
        col_b.setMinimumWidth(254)
        row.addWidget(col_a, 6)
        row.addWidget(col_b, 5)

        # Safety net only: at heights below ~730 px the sidebar scrolls rather
        # than clipping.  At 768 and above no scrollbar appears at all.
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(560)
        self._left_scroll = scroll
        return scroll

    def _build_fsm_banner(self) -> QWidget:
        box = QGroupBox("FLIGHT STATE")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 6, 8, 8)

        self.fsm_label = QLabel("NO DATA")
        fsm_font = QFont()
        fsm_font.setPointSize(21)
        fsm_font.setBold(True)
        self.fsm_label.setFont(fsm_font)
        self.fsm_label.setAlignment(Qt.AlignCenter)
        self.fsm_label.setMinimumHeight(52)
        self._style_fsm(None)
        layout.addWidget(self.fsm_label)
        return box

    def _build_readouts(self) -> QWidget:
        box = QGroupBox("LIVE TELEMETRY")
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)

        self.tile_team = ReadoutTile("Team ID", value_pt=15)
        self.tile_count = ReadoutTile("Packet count", value_pt=15)
        self.tile_mission = ReadoutTile("Mission time", value_pt=15)
        self.tile_alt = ReadoutTile("Altitude", "m", COL_ALT)
        self.tile_press = ReadoutTile("Pressure", "hPa", COL_PRESS)
        self.tile_temp = ReadoutTile("Temperature", "°C", COL_TEMP)
        self.tile_volt = ReadoutTile("Battery", "V", COL_VOLT)
        self.tile_sats = ReadoutTile("Satellites", "", COL_TEXT)

        for index, tile in enumerate(
            (
                self.tile_team, self.tile_count, self.tile_mission,
                self.tile_alt, self.tile_press, self.tile_temp,
                self.tile_volt, self.tile_sats,
            )
        ):
            grid.addWidget(tile, index // 2, index % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return box

    # -- vehicle-specific payload panels ---------------------------------

    def _build_payload_panel(self) -> QWidget:
        """Stacked CanSat / Rocket / generic panels, switched by PAYLOAD_TYPE.

        A ``QStackedWidget`` rather than show/hide so the layout height stays
        constant when the vehicle changes -- nothing below it jumps around.
        """
        self.payload_box = QGroupBox("PAYLOAD — waiting for telemetry")
        outer = QVBoxLayout(self.payload_box)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self.payload_stack = QStackedWidget()
        self.payload_stack.addWidget(self._build_generic_page())   # index 0
        self.payload_stack.addWidget(self._build_cansat_page())    # index 1
        self.payload_stack.addWidget(self._build_rocket_page())    # index 2
        # Without an explicit floor the stack is the first thing the scrollable
        # column squeezes when the content is taller than the viewport, and the
        # panel collapses to a few pixels.  Each page declares its own minimum
        # height and _apply_payload_panel() applies the active one.
        self.payload_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        outer.addWidget(self.payload_stack)

        # Sensor reset lives here rather than only in the Settings tab: between
        # bench runs the operator wants the payload traces cleared without
        # hunting for it, and it applies to both vehicle types.
        self.reset_sensors_btn = QPushButton("RESET SENSOR DATA")
        self.reset_sensors_btn.setToolTip(
            "Clear the payload sensor history for this session:\n"
            "particulate and reaction-wheel traces, all strip charts, the ground\n"
            "track, the attitude estimate and the link counters.\n"
            "The CSV log is NOT touched and keeps recording."
        )
        self.reset_sensors_btn.clicked.connect(self.reset_sensor_data)
        outer.addWidget(self.reset_sensors_btn)
        return self.payload_box

    def _build_generic_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(
            "No vehicle-specific sensors in this stream.\n"
            "Legacy v1 packets carry the common telemetry only."
        )
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: %s;" % COL_TEXT_DIM)
        label.setMinimumHeight(64)
        layout.addWidget(label)
        page.setMinimumHeight(80)
        return page

    def _build_cansat_page(self) -> QWidget:
        """CanSat: SPS30 air quality, reaction wheel, recovery stage."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- SPS30 particulate trio ----------------------------------------
        air_label = QLabel("AIR QUALITY — SPS30 (µg/m³)")
        air_font = QFont()
        air_font.setPointSize(8)
        air_font.setBold(True)
        air_label.setFont(air_font)
        air_label.setStyleSheet("color: %s; letter-spacing: 1px;" % COL_ACCENT)
        air_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(air_label)

        air_row = QHBoxLayout()
        air_row.setSpacing(6)
        self.tile_pm1 = ReadoutTile("PM1.0", "", COL_PM1, value_pt=13)
        self.tile_pm25 = ReadoutTile("PM2.5", "", COL_PM25, value_pt=13)
        self.tile_pm10 = ReadoutTile("PM10", "", COL_PM10, value_pt=13)
        for tile in (self.tile_pm1, self.tile_pm25, self.tile_pm10):
            # Equal stretch and a small floor: three tiles must share the
            # column width evenly without the last one being clipped.
            tile.setMinimumWidth(56)
            air_row.addWidget(tile, 1)
        layout.addLayout(air_row)

        # Particulate time series.  The SPS30 is the scored science payload, so
        # its history matters as much as its instantaneous value -- the whole
        # point of flying it is the concentration profile through the descent.
        self.chart_pm = StripChart(
            "Particulates", "µg/m³",
            [("PM1.0", COL_PM1), ("PM2.5", COL_PM25), ("PM10", COL_PM10)],
            legend=False,   # the colour-matched tiles above are the legend
        )
        self.chart_pm.setMinimumHeight(86)
        self.chart_pm.setMaximumHeight(104)
        self.chart_pm.setTitle(None)
        layout.addWidget(self.chart_pm)

        # --- reaction wheel -------------------------------------------------
        wheel_row = QHBoxLayout()
        wheel_row.setSpacing(6)
        self.tile_wheel = ReadoutTile("Reaction wheel", "RPM", COL_WHEEL, value_pt=15)
        self.tile_wheel.setToolTip(
            "Active stabilisation reaction wheel speed.\n"
            "Signed: sign indicates spin direction. Saturates at ±1124 RPM."
        )
        wheel_row.addWidget(self.tile_wheel, 1)

        self.tile_recovery = ReadoutTile("Recovery", "", COL_TEXT, value_pt=15)
        self.tile_recovery.setToolTip(
            "Recovery sequencer stage: STOWED -> DROGUE -> PARAFOIL."
        )
        wheel_row.addWidget(self.tile_recovery, 1)
        layout.addLayout(wheel_row)

        # --- reaction wheel history ----------------------------------------
        self.chart_wheel = StripChart(
            "Reaction wheel RPM", "RPM", [("rpm", COL_WHEEL)]
        )
        # Compact: this page is pinned, so every pixel it takes is a pixel the
        # scrollable panels below lose. The RPM value is the primary readout;
        # the trace is here for trend, not for precise reading.
        self.chart_wheel.setMinimumHeight(84)
        self.chart_wheel.setMaximumHeight(96)
        self.chart_wheel.setTitle(None)
        layout.addWidget(self.chart_wheel)
        page.setMinimumHeight(336)
        return page

    def _build_rocket_page(self) -> QWidget:
        """Rocket: dual-stage recovery status lights."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        caption = QLabel("DUAL-STAGE RECOVERY")
        caption_font = QFont()
        caption_font.setPointSize(8)
        caption_font.setBold(True)
        caption.setFont(caption_font)
        caption.setStyleSheet("color: %s; letter-spacing: 1px;" % COL_ACCENT)
        caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(caption)

        lights = QHBoxLayout()
        lights.setSpacing(6)
        self.light_solenoid = StatusLight("Solenoid")
        self.light_solenoid.setToolTip(
            "6 V solenoid latch, released at apogee to deploy the drogue."
        )
        self.light_nichrome = StatusLight("Nichrome")
        self.light_nichrome.setToolTip(
            "Nichrome line cutter, fired at 400 m AGL to deploy the main."
        )
        lights.addWidget(self.light_solenoid)
        lights.addWidget(self.light_nichrome)
        layout.addLayout(lights)

        self.recovery_summary = QLabel("Recovery sequence not started")
        self.recovery_summary.setAlignment(Qt.AlignCenter)
        self.recovery_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.recovery_summary.setStyleSheet(
            "color: %s; background:#242c38; border-radius:4px; padding:5px;"
            % COL_TEXT_DIM
        )
        layout.addWidget(self.recovery_summary)
        # The stack reserves the tallest page's height (CanSat), so send the
        # slack to the bottom rather than letting it open gaps between rows.
        layout.addStretch(1)
        page.setMinimumHeight(152)
        return page

    def _build_gps_panel(self) -> QWidget:
        box = QGroupBox("GPS / NavIC")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        grid = QGridLayout()
        grid.setSpacing(6)
        self.tile_lat = ReadoutTile("Latitude", "", COL_TEXT, value_pt=13)
        self.tile_lon = ReadoutTile("Longitude", "", COL_TEXT, value_pt=13)
        self.tile_nav_alt = ReadoutTile("Nav altitude", "m", COL_TEXT, value_pt=13)
        self.tile_nav_time = ReadoutTile("Nav time", "", COL_TEXT, value_pt=13)
        grid.addWidget(self.tile_lat, 0, 0)
        grid.addWidget(self.tile_lon, 0, 1)
        grid.addWidget(self.tile_nav_alt, 1, 0)
        grid.addWidget(self.tile_nav_time, 1, 1)
        layout.addLayout(grid)

        # Kept close to square: a ground track stretched vertically
        # misrepresents the shape of the trajectory at a glance.
        self.gps_plot = GpsTrackPlot()
        self.gps_plot.setMinimumHeight(180)
        self.gps_plot.setMaximumHeight(250)
        layout.addWidget(self.gps_plot)
        return box

    def _build_link_panel(self) -> QWidget:
        box = QGroupBox("LINK STATUS")
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)

        self.tile_rate = ReadoutTile("Packet rate", "pkt/s", COL_ACCENT, value_pt=15)
        self.tile_age = ReadoutTile("Packet age", "s", COL_TEXT, value_pt=15)
        self.tile_total = ReadoutTile("Valid/Total", "", COL_TEXT, value_pt=15)
        self.tile_corrupt = ReadoutTile("Corrupt", "", COL_TEXT, value_pt=15)

        grid.addWidget(self.tile_rate, 0, 0)
        grid.addWidget(self.tile_age, 0, 1)
        grid.addWidget(self.tile_total, 1, 0)
        grid.addWidget(self.tile_corrupt, 1, 1)

        self.stale_banner = QLabel("")
        self.stale_banner.setAlignment(Qt.AlignCenter)
        stale_font = QFont()
        stale_font.setPointSize(12)
        stale_font.setBold(True)
        self.stale_banner.setFont(stale_font)
        self.stale_banner.setMinimumHeight(28)
        grid.addWidget(self.stale_banner, 2, 0, 1, 2)

        self.tile_total.set_value("0/0")
        self.tile_corrupt.set_value("0")
        self.tile_rate.set_value("0.0")
        self.tile_age.set_value("--")
        return box

    def _build_settings_panel(self) -> QWidget:
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)

        grid.addWidget(QLabel("Chart window (s)"), 0, 0)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(5, 1800)
        self.window_spin.setValue(DEFAULT_WINDOW_S)
        self.window_spin.setSingleStep(5)
        grid.addWidget(self.window_spin, 0, 1)

        grid.addWidget(QLabel("Batt. warn (V)"), 1, 0)
        self.volt_spin = QDoubleSpinBox()
        self.volt_spin.setRange(0.0, 60.0)
        self.volt_spin.setDecimals(2)
        self.volt_spin.setSingleStep(0.1)
        self.volt_spin.setValue(DEFAULT_VOLTAGE_WARN)
        grid.addWidget(self.volt_spin, 1, 1)

        grid.addWidget(QLabel("Vehicle panel"), 2, 0)
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.addItems(["Auto-detect", "CanSat", "Rocket"])
        self.vehicle_combo.setToolTip(
            "Which payload panel to show.\n"
            "Auto-detect follows the PAYLOAD_TYPE field in the incoming stream; "
            "the manual settings pin the panel regardless of what arrives."
        )
        self.vehicle_combo.currentIndexChanged.connect(self._apply_payload_panel)
        grid.addWidget(self.vehicle_combo, 2, 1)

        self.autoscroll_check = QCheckBox("Auto-scale Y axes")
        self.autoscroll_check.setChecked(True)
        grid.addWidget(self.autoscroll_check, 3, 0, 1, 2)

        self.clear_btn = QPushButton("Clear plots && counters")
        grid.addWidget(self.clear_btn, 4, 0, 1, 2)

        grid.setColumnStretch(1, 1)
        return box

    def _build_event_log(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(EVENT_LOG_LINES)
        self.event_log.setMinimumHeight(96)
        layout.addWidget(self.event_log)
        return box

    # -- plot grid -------------------------------------------------------

    def _build_plot_grid(self) -> QWidget:
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)

        self.chart_alt = StripChart("Altitude", "m", [("alt", COL_ALT)])
        self.chart_press = StripChart("Pressure", "hPa", [("press", COL_PRESS)])
        self.chart_temp = StripChart("Temperature", "°C", [("temp", COL_TEMP)])
        self.chart_volt = StripChart("Battery voltage", "V", [("volt", COL_VOLT)])
        self.chart_acc = StripChart(
            "Accelerometer", "m/s²",
            [("X", COL_XYZ[0]), ("Y", COL_XYZ[1]), ("Z", COL_XYZ[2])],
        )
        self.chart_gyro = StripChart(
            "Gyroscope", "°/s",
            [("X", COL_XYZ[0]), ("Y", COL_XYZ[1]), ("Z", COL_XYZ[2])],
        )

        self.charts = [
            self.chart_alt, self.chart_press, self.chart_temp,
            self.chart_volt, self.chart_acc, self.chart_gyro,
        ]

        grid.addWidget(self.chart_alt, 0, 0)
        grid.addWidget(self.chart_press, 0, 1)
        grid.addWidget(self.chart_temp, 1, 0)
        grid.addWidget(self.chart_volt, 1, 1)
        grid.addWidget(self.chart_acc, 2, 0)
        grid.addWidget(self.chart_gyro, 2, 1)
        for row in range(3):
            grid.setRowStretch(row, 1)
        for col in range(2):
            grid.setColumnStretch(col, 1)
        return container

    # -- attitude + session summary row ----------------------------------

    def _build_analysis_row(self) -> QWidget:
        """Bottom-right row: live attitude model beside the session donut."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        row = QSplitter(Qt.Horizontal)
        row.setChildrenCollapsible(False)
        outer.addWidget(row)

        attitude_box = QGroupBox("VEHICLE ATTITUDE")
        self.attitude_box = attitude_box
        attitude_layout = QVBoxLayout(attitude_box)
        attitude_layout.setContentsMargins(8, 8, 8, 8)
        self.attitude = AttitudeWidget()
        attitude_layout.addWidget(self.attitude)
        attitude_box.setMinimumWidth(248)
        row.addWidget(attitude_box)

        summary_box = QGroupBox("SESSION SUMMARY")
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.setContentsMargins(8, 8, 8, 8)
        self.summary = SessionSummaryWidget()
        summary_layout.addWidget(self.summary)
        summary_box.setMinimumWidth(248)
        row.addWidget(summary_box)

        # Display settings and the event log live here rather than in the left
        # sidebar: they are the two panels an operator consults occasionally
        # rather than watching continuously, so they are the right things to
        # cost a tab click in exchange for GPS and link status never scrolling.
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_event_log(), "Event log")
        tabs.addTab(self._build_settings_panel(), "Settings")
        tabs.setMinimumWidth(240)
        row.addWidget(tabs)
        row.setSizes([340, 340, 300])

        container.setMinimumHeight(186)
        return container

    # ==================================================================
    # Signal wiring
    # ==================================================================

    def _wire_signals(self) -> None:
        # Worker -> GUI.  These are cross-thread, so Qt delivers them queued on
        # the GUI event loop; the slots must stay short.
        self.serial_worker.packet_received.connect(self.on_packet)
        self.serial_worker.bad_frame.connect(self.on_bad_frame)
        self.serial_worker.stats_updated.connect(self.on_stats)
        self.serial_worker.connection_changed.connect(self.on_connection_changed)
        self.serial_worker.log_message.connect(self.append_event)

        self.csv_logger.file_opened.connect(self.on_log_file_opened)
        self.csv_logger.error_occurred.connect(self.append_event)

        # Widgets -> GUI slots.
        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.log_btn.clicked.connect(self.toggle_logging)
        self.clear_btn.clicked.connect(self.clear_session)
        self.diag_btn.clicked.connect(self.toggle_diagnostics)

    # ==================================================================
    # Slots — connection control
    # ==================================================================

    def refresh_ports(self) -> None:
        """Rescan serial ports, preserving whatever the user had selected."""
        current = self.port_combo.currentText().strip()
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        for device, description in list_serial_ports():
            self.port_combo.addItem(description, device)
        # Always offer the simulator URL so testing needs no driver install.
        self.port_combo.addItem("%s  (packet_sim.py)" % SIM_PORT_URL, SIM_PORT_URL)
        self.port_combo.blockSignals(False)

        if current:
            index = self.port_combo.findData(current)
            if index < 0:
                index = self.port_combo.findText(current, Qt.MatchStartsWith)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)
            else:
                self.port_combo.setEditText(current)

    def selected_port(self) -> str:
        """Resolve the combo selection to a device name or pyserial URL.

        The combo is editable, so the text may either be one of the discovered
        entries ("COM7 — USB Serial Device") whose ``itemData`` holds the real
        device name, or something the operator typed by hand ("socket://…").
        """
        text = self.port_combo.currentText().strip()
        if not text:
            return ""
        index = self.port_combo.findText(text)
        if index >= 0:
            data = self.port_combo.itemData(index)
            if data:
                return str(data)
        return text

    def toggle_connection(self) -> None:
        if self.is_connected:
            self.serial_worker.request_disconnect()
            self.append_event("Disconnect requested.")
            self.csv_logger.log_note("Disconnect requested by operator")
            # Reflect intent immediately; the worker confirms via the signal.
            self._set_connect_button(False)
            return

        port = self.selected_port()
        if not port:
            QMessageBox.warning(self, "No port", "Select or type a serial port first.")
            return
        try:
            baud = int(self.baud_combo.currentText().strip())
        except ValueError:
            QMessageBox.warning(self, "Bad baud rate", "Baud rate must be a number.")
            return

        self.append_event("Connecting to %s @ %d baud…" % (port, baud))
        self.csv_logger.log_note("Connect requested: %s @ %d" % (port, baud))
        self.serial_worker.request_connect(port, baud)
        self._set_connect_button(True)

    def _set_connect_button(self, connected: bool) -> None:
        self.connect_btn.setText("DISCONNECT" if connected else "CONNECT")
        self.connect_btn.setProperty("connected", "true" if connected else "false")
        # Re-polish so the dynamic-property stylesheet rule takes effect.
        self.connect_btn.style().unpolish(self.connect_btn)
        self.connect_btn.style().polish(self.connect_btn)

    def on_connection_changed(self, connected: bool, message: str) -> None:
        self.is_connected = connected
        self._set_connect_button(connected)
        color = COL_OK if connected else COL_ALERT
        self.conn_state_label.setText("● CONNECTED" if connected else "● DISCONNECTED")
        self.conn_state_label.setStyleSheet("color: %s; font-weight: 700;" % color)
        self.statusBar().showMessage(message)
        self.append_event(message)

    # ==================================================================
    # Slots — telemetry (hot path: keep these cheap!)
    # ==================================================================

    def on_packet(self, packet: TelemetryPacket) -> None:
        """Store one validated packet.  No widget is touched here — see _render."""
        try:
            self.latest = packet
            self.session_packets += 1

            # Raw wire view and the diagnostics table. Both are plain setText
            # work; the table is skipped entirely while its window is hidden so
            # a closed diagnostics view costs nothing here.
            self.raw_strip.show_packet(packet.raw_frame)
            if self.diagnostics.isVisible():
                self.diagnostics.update_packet(
                    packet, voltage_warn=float(self.volt_spin.value())
                )
            self.last_packet_epoch = packet.gs_recv_epoch
            self.recv_times.append(packet.gs_recv_epoch)

            # One shared x-axis value for every chart on this packet.
            x = self._plot_time(packet)

            self.chart_alt.add_point(x, {"alt": packet.altitude_m})
            self.chart_press.add_point(x, {"press": packet.pressure_hpa})
            self.chart_temp.add_point(x, {"temp": packet.temp_c})
            self.chart_volt.add_point(x, {"volt": packet.voltage_v})
            self.chart_acc.add_point(
                x, {"X": packet.acc_x, "Y": packet.acc_y, "Z": packet.acc_z}
            )
            self.chart_gyro.add_point(
                x, {"X": packet.gyro_x, "Y": packet.gyro_y, "Z": packet.gyro_z}
            )

            # Vehicle-specific series and panel switching.
            if packet.payload_type != self._detected_payload:
                self._detected_payload = packet.payload_type
                self._apply_payload_panel()
                self.append_event("Payload type detected: %s" % packet.payload_type)

            if packet.is_cansat:
                self.chart_wheel.add_point(
                    x, {"rpm": float(packet.reaction_wheel_rpm)}
                )
                self.chart_pm.add_point(x, {
                    "PM1.0": packet.pm1_0,
                    "PM2.5": packet.pm2_5,
                    "PM10": packet.pm10,
                })

            # Attitude estimator and session summary get every packet; both are
            # cheap accumulate-only slots that repaint on their own timers.
            self.attitude.on_packet(packet)
            self.summary.on_packet(packet)

            if packet.has_fix:
                self.gps_plot.add_fix(packet.lat, packet.lon)

            if packet.fsm_state != self._last_fsm:
                previous = FSM_STATES.get(self._last_fsm, "—")
                self._last_fsm = packet.fsm_state
                self.append_event(
                    "FSM %s → %s at T+%s"
                    % (previous, packet.fsm_name, packet.mission_time_hms)
                )

            if self.logging_enabled:
                self.csv_logger.log_packet(packet)

            self._readouts_dirty = True
        except Exception as exc:  # a display bug must never stop ingestion
            self.append_event("Packet handling error: %r" % exc)

    def _plot_time(self, packet: TelemetryPacket) -> float:
        """Return the shared x-axis value for this packet, in seconds.

        Every strip chart plots against this one number, so the charts can
        never disagree about the time base and a bug here cannot affect one
        chart differently from another.

        The value is derived from the raw ``MISSION_TIME_S`` float on the
        packet.  It is never parsed back out of ``MISSION_TIME_HMS`` or any
        other formatted string -- those are display-only, and re-deriving a
        coordinate from ``HH:MM:SS`` is precisely how an x-axis ends up
        wrapping every 60 seconds.

        The result is guaranteed non-decreasing.  If the source time base jumps
        backwards -- a flight-computer reboot, a restarted transmitter, or a
        different vehicle appearing on the link -- the plot clock carries on
        forward from where it was instead of drawing back across the chart.
        """
        mission = packet.mission_time_s
        if not math.isfinite(mission):
            # No usable mission time: fall back on ground-station elapsed time,
            # which is monotonic by construction.
            mission = packet.gs_recv_epoch - self.session_start

        if self._last_mission_time is not None:
            if mission < self._last_mission_time - TIME_RESET_TOLERANCE_S:
                # Rebase so the next x continues just after the last one.
                self._plot_time_offset = (
                    self._last_plot_time + PLOT_TIME_GAP_S - mission
                )
                self._time_resets += 1
                self.append_event(
                    "Mission time jumped backwards (%.2fs → %.2fs); plot clock "
                    "continues forward. Flight computer reset?"
                    % (self._last_mission_time, mission)
                )
                self.csv_logger.log_note(
                    "mission time reset: %.3f -> %.3f"
                    % (self._last_mission_time, mission)
                )

        self._last_mission_time = mission
        x = mission + self._plot_time_offset
        self._last_plot_time = x

        if self.first_mission_time is None:
            self.first_mission_time = mission
        return x

    def on_bad_frame(self, raw: str, reason: str) -> None:
        """A frame failed checksum or parsing — always archived, never fatal."""
        # Still show the bytes: during bring-up, seeing corrupt traffic is far
        # more informative than seeing nothing.
        self.raw_strip.show_corrupt(raw)
        # Errors are logged unconditionally so a corrupted-link investigation
        # still has data even if CSV logging was never switched on.
        self.csv_logger.log_error(raw, reason)

    def on_stats(self, total_frames: int, valid: int, corrupt: int,
                 resyncs: int) -> None:
        self.total_frames = total_frames
        self.valid_packets = valid
        self.corrupt_packets = corrupt
        self.resyncs = resyncs
        self.summary.set_link_stats(valid, corrupt, resyncs)

    def on_log_file_opened(self, path: str) -> None:
        self.log_path_label.setText("CSV: %s" % path)
        self.append_event("Logging to %s" % path)

    # ==================================================================
    # Rendering (timer driven)
    # ==================================================================

    def _render(self) -> None:
        """Repaint readouts and charts at a fixed rate, independent of RX rate."""
        try:
            if self._readouts_dirty:
                self._readouts_dirty = False
                self._update_readouts()

                window_s = float(self.window_spin.value())
                autoscale = self.autoscroll_check.isChecked()
                for chart in self.charts:
                    chart.redraw(window_s, autoscale)
                # The reaction-wheel chart lives in the CanSat payload page, so
                # it is redrawn explicitly rather than via self.charts (which is
                # the main grid only) and only when that page is visible.
                if self.payload_stack.currentIndex() == 1:
                    self.chart_wheel.redraw(window_s, autoscale)
                    self.chart_pm.redraw(window_s, autoscale)
                self.gps_plot.redraw()
        except Exception as exc:  # pragma: no cover - defensive
            self.append_event("Render error: %r" % exc)

    def _update_readouts(self) -> None:
        packet = self.latest
        if packet is None:
            return

        self.tile_team.set_value(packet.team_id)
        self.tile_count.set_value(str(packet.packet_count))
        self.tile_mission.set_value(packet.mission_time_hms)
        self.tile_alt.set_value(self._fmt(packet.altitude_m, 1))
        self.tile_press.set_value(self._fmt(packet.pressure_hpa, 2))
        self.tile_temp.set_value(self._fmt(packet.temp_c, 1))
        self.tile_volt.set_value(self._fmt(packet.voltage_v, 2))
        self.tile_sats.set_value(str(packet.sats))

        # Battery warning: configurable threshold, plus a hard alert 10% below it.
        threshold = float(self.volt_spin.value())
        voltage = packet.voltage_v
        if not math.isfinite(voltage):
            self.tile_volt.set_level("alert")
        elif voltage < threshold * 0.9:
            self.tile_volt.set_level("alert")
        elif voltage < threshold:
            self.tile_volt.set_level("warn")
        else:
            self.tile_volt.set_level("ok")

        self.tile_sats.set_level(
            "ok" if packet.sats >= 6 else ("warn" if packet.sats >= 4 else "alert")
        )

        self.tile_lat.set_value(self._fmt(packet.lat, 6))
        self.tile_lon.set_value(self._fmt(packet.lon, 6))
        self.tile_nav_alt.set_value(self._fmt(packet.nav_alt_m, 1))
        self.tile_nav_time.set_value(packet.nav_time or "--")
        self.tile_lat.set_level("normal" if packet.has_fix else "warn")
        self.tile_lon.set_level("normal" if packet.has_fix else "warn")

        self._update_payload_readouts(packet)
        self._style_fsm(packet.fsm_state)

    def toggle_diagnostics(self) -> None:
        """Show or hide the raw diagnostics table window."""
        if self.diagnostics.isVisible():
            self.diagnostics.hide()
            self.append_event("Diagnostics window closed.")
            return

        self.diagnostics.show()
        self.diagnostics.raise_()
        self.diagnostics.activateWindow()
        # Populate immediately rather than waiting for the next packet, so the
        # window is never briefly blank on a stale or disconnected link.
        if self.latest is not None:
            self.diagnostics.update_packet(
                self.latest, voltage_warn=float(self.volt_spin.value())
            )
        self._push_link_diagnostics()
        self.append_event("Diagnostics window opened.")

    def _push_link_diagnostics(self) -> None:
        """Send the current link statistics to the diagnostics table."""
        if not self.diagnostics.isVisible():
            return
        now = time.time()
        age = (now - self.last_packet_epoch
               if self.last_packet_epoch is not None else None)
        rate = len(self.recv_times) / RATE_WINDOW_S if self.recv_times else 0.0
        self.diagnostics.update_link(
            rate=rate, age=age, valid=self.valid_packets,
            total=self.total_frames, corrupt=self.corrupt_packets,
            connected=self.is_connected, stale_after=STALE_AFTER_S,
        )

    def _apply_payload_panel(self) -> None:
        """Show the panel for the selected (or auto-detected) vehicle type."""
        choice = self.vehicle_combo.currentText()
        if choice == "CanSat":
            payload = PAYLOAD_CANSAT
            source = "manual"
        elif choice == "Rocket":
            payload = PAYLOAD_ROCKET
            source = "manual"
        else:
            payload = self._detected_payload or ""
            source = "auto"

        if payload == PAYLOAD_CANSAT:
            self.payload_stack.setCurrentIndex(1)
            title = "PAYLOAD — CANSAT"
        elif payload == PAYLOAD_ROCKET:
            self.payload_stack.setCurrentIndex(2)
            title = "PAYLOAD — ROCKET"
        elif payload == PAYLOAD_GENERIC:
            self.payload_stack.setCurrentIndex(0)
            title = "PAYLOAD — GENERIC (v1)"
        else:
            self.payload_stack.setCurrentIndex(0)
            title = "PAYLOAD — waiting for telemetry"
        if source == "manual":
            title += "  [PINNED]"
        self.payload_box.setTitle(title)

        # The 3D model follows the same resolved choice, so a pinned vehicle is
        # no longer silently overridden by whatever the stream announces.
        if hasattr(self, "attitude"):
            self.attitude.set_vehicle(payload or PAYLOAD_ROCKET)
            if source == "manual":
                suffix = "%s  [PINNED]" % (payload or PAYLOAD_ROCKET)
            elif payload:
                suffix = "%s  (auto-detected)" % payload
            else:
                suffix = "awaiting telemetry"
            self.attitude_box.setTitle("VEHICLE ATTITUDE — %s" % suffix)

        # Re-floor the stack to the visible page so a short page (rocket) does
        # not leave dead space and a tall one (CanSat) is not squeezed away.
        current = self.payload_stack.currentWidget()
        if current is not None:
            height = current.minimumHeight()
            self.payload_stack.setMinimumHeight(height)
            self.payload_stack.setMaximumHeight(height)

    def _update_payload_readouts(self, packet: TelemetryPacket) -> None:
        """Refresh the vehicle-specific tiles from the latest packet."""
        if packet.is_cansat:
            self.tile_pm1.set_value(self._fmt(packet.pm1_0, 1))
            self.tile_pm25.set_value(self._fmt(packet.pm2_5, 1))
            self.tile_pm10.set_value(self._fmt(packet.pm10, 1))

            # WHO 24-hour guideline for PM2.5 is 15 µg/m³; well above that is
            # exactly the plume the SPS30 is flown to measure, so a high value
            # is a successful measurement, not a fault. Colour it as
            # "elevated"/"high" rather than warn/alert.
            pm25 = packet.pm2_5
            if not math.isfinite(pm25):
                self.tile_pm25.set_level("normal")
            elif pm25 >= 55.0:
                self.tile_pm25.set_level("alert")
            elif pm25 >= 15.0:
                self.tile_pm25.set_level("warn")
            else:
                self.tile_pm25.set_level("ok")

            rpm = packet.reaction_wheel_rpm
            self.tile_wheel.set_value("%+d" % rpm)
            # Near saturation the wheel can no longer authority-control the
            # spin, which is worth flagging to the operator.
            self.tile_wheel.set_level("alert" if abs(rpm) >= 1050 else "normal")

            stage = packet.recovery_stage
            self.tile_recovery.set_value(
                RECOVERY_STAGES.get(stage, "UNKNOWN(%s)" % stage)
            )
            color = RECOVERY_STAGE_COLORS.get(stage, COL_TEXT)
            self.tile_recovery.value.setStyleSheet("color: %s;" % color)

        elif packet.is_rocket:
            self.light_solenoid.set_state(packet.solenoid_fired)
            self.light_nichrome.set_state(packet.nichrome_fired)

            if packet.nichrome_fired:
                text = "Main deployed (400 m AGL)"
                style = "color:#0b1219; background:%s;" % COL_OK
            elif packet.solenoid_fired:
                text = "Drogue deployed (apogee)"
                style = "color:#0b1219; background:%s;" % COL_WARN
            else:
                text = "Recovery armed — no events fired"
                style = "color:%s; background:#242c38;" % COL_TEXT_DIM
            self.recovery_summary.setText(text)
            self.recovery_summary.setStyleSheet(
                style + " border-radius:4px; padding:5px;"
            )

    def _style_fsm(self, state: Optional[int]) -> None:
        if state is None:
            text, color, fg = "NO DATA", "#333c48", COL_TEXT_DIM
        else:
            text = FSM_STATES.get(state, "UNKNOWN (%s)" % state)
            color = FSM_COLORS.get(state, "#8a2be2")
            fg = "#080d14"
        self.fsm_label.setText("%s%s" % (text, "" if state is None else "  [%d]" % state))
        self.fsm_label.setStyleSheet(
            "background-color: %s; color: %s; border-radius: 6px;"
            "letter-spacing: 2px; padding: 6px;" % (color, fg)
        )

    def _update_link_status(self) -> None:
        """Packet rate, staleness and counters — runs even when no data arrives."""
        now = time.time()

        # Packets per second over a trailing window.
        while self.recv_times and (now - self.recv_times[0]) > RATE_WINDOW_S:
            self.recv_times.popleft()
        rate = len(self.recv_times) / RATE_WINDOW_S if self.recv_times else 0.0
        self.tile_rate.set_value("%.1f" % rate)

        self.tile_total.set_value("%d/%d" % (self.valid_packets, self.total_frames))
        self.tile_corrupt.set_value(str(self.corrupt_packets))
        self.tile_corrupt.set_level("alert" if self.corrupt_packets else "normal")

        self._push_link_diagnostics()

        if self.last_packet_epoch is None:
            self.tile_age.set_value("--")
            self.tile_age.set_level("normal")
            self._set_stale_banner(None)
            return

        age = now - self.last_packet_epoch
        self.tile_age.set_value("%.1f" % age)

        stale = age > STALE_AFTER_S
        self.tile_age.set_level("alert" if stale else "ok")
        self._set_stale_banner(stale)

        if stale != self._was_stale:
            self._was_stale = stale
            if stale:
                self.append_event("TELEMETRY STALE — no valid packet for >%.0f s"
                                  % STALE_AFTER_S)
                self.csv_logger.log_note("telemetry stale")
            else:
                self.append_event("Telemetry recovered.")
                self.csv_logger.log_note("telemetry recovered")

    def _set_stale_banner(self, stale: Optional[bool]) -> None:
        if stale is None:
            self.stale_banner.setText("AWAITING TELEMETRY")
            self.stale_banner.setStyleSheet(
                "background:#242c38; color:%s; border-radius:4px;" % COL_TEXT_DIM
            )
        elif stale:
            self.stale_banner.setText("!! TELEMETRY STALE !!")
            self.stale_banner.setStyleSheet(
                "background:%s; color:#ffffff; border-radius:4px;" % COL_ALERT
            )
        else:
            self.stale_banner.setText("LINK NOMINAL")
            self.stale_banner.setStyleSheet(
                "background:#12351f; color:%s; border-radius:4px;" % COL_OK
            )

    @staticmethod
    def _fmt(value: float, digits: int) -> str:
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return "--"
        return ("%%.%df" % digits) % value

    # ==================================================================
    # Logging control / session management
    # ==================================================================

    def toggle_logging(self) -> None:
        self.logging_enabled = not self.logging_enabled
        self.log_btn.setText("STOP LOGGING" if self.logging_enabled else "START LOGGING")
        self.log_btn.setProperty("logging", "true" if self.logging_enabled else "false")
        self.log_btn.style().unpolish(self.log_btn)
        self.log_btn.style().polish(self.log_btn)

        if self.logging_enabled:
            path = self.csv_logger.csv_path
            self.append_event(
                "CSV logging ON%s"
                % ("" if path is None else " → %s" % path)
            )
            if path is None:
                self.log_path_label.setText("CSV: waiting for first packet (TEAM_ID)…")
            self.csv_logger.log_note("CSV logging started")
        else:
            self.append_event("CSV logging OFF (file flushed and left open).")
            self.csv_logger.log_note("CSV logging stopped")

    def reset_sensor_data(self) -> None:
        """Clear the session's sensor history, readouts and counters.

        Deliberately does not stop or truncate the CSV: resetting the display
        between bench runs must never cost recorded telemetry.
        """
        self.clear_session()
        self.latest = None
        self._last_fsm = None

        # Blank the payload readouts so stale values are not mistaken for live
        # ones while waiting for the next packet.
        for tile in (self.tile_pm1, self.tile_pm25, self.tile_pm10,
                     self.tile_wheel, self.tile_recovery):
            tile.set_value("--")
            tile.set_level("normal")
        self.light_solenoid.set_state(None)
        self.light_nichrome.set_state(None)
        self.recovery_summary.setText("Recovery sequence not started")
        self.recovery_summary.setStyleSheet(
            "color:%s; background:#242c38; border-radius:4px; padding:5px;"
            % COL_TEXT_DIM
        )
        self._style_fsm(None)
        self.diagnostics.clear()
        self.raw_strip.clear()
        self.append_event("Sensor data reset (CSV logging unaffected).")

    def clear_session(self) -> None:
        """Wipe the plots and the on-screen counters (does not touch the CSV)."""
        for chart in self.charts:
            chart.clear_data()
        self.chart_wheel.clear_data()
        self.chart_pm.clear_data()
        self.gps_plot.clear_data()
        self.attitude.clear()
        self.summary.clear()
        self.recv_times.clear()
        self.session_packets = 0
        self.first_mission_time = None
        self._last_mission_time = None
        self._last_plot_time = 0.0
        self._plot_time_offset = 0.0
        self._time_resets = 0
        self.session_start = time.time()
        self.serial_worker.reset_counters()
        self.total_frames = self.valid_packets = self.corrupt_packets = 0
        self.resyncs = 0
        self.tile_total.set_value("0/0")
        self.tile_corrupt.set_value("0")
        self.append_event("Plots and counters cleared.")

    def append_event(self, text: str) -> None:
        """Append a timestamped line to the on-screen event log and status bar."""
        self.event_log.appendPlainText("[%s] %s" % (time.strftime("%H:%M:%S"), text))
        self.statusBar().showMessage(text)

    # ==================================================================
    # Clean shutdown
    # ==================================================================

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop timers, stop both worker threads, flush and close every file.

        Order matters: timers first (so nothing repaints against half-torn-down
        state), then the serial thread (so no new packets are produced), then the
        logger (so it can drain whatever is still queued before closing files).
        """
        try:
            self.render_timer.stop()
            self.status_timer.stop()
            self.attitude_timer.stop()
            self.diagnostics.close()

            self.serial_worker.stop()
            if not self.serial_worker.wait(3000):
                self.serial_worker.terminate()
                self.serial_worker.wait(500)

            self.csv_logger.log_note("session ended")
            self.csv_logger.stop()
            if not self.csv_logger.wait(5000):
                self.csv_logger.terminate()
                self.csv_logger.wait(500)
        except Exception:
            # Shutdown must never raise — the user is trying to close the window.
            pass
        finally:
            event.accept()


def apply_dark_palette(app: QApplication) -> None:
    """Apply the Fusion style + dark stylesheet at application level."""
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)

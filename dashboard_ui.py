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

from csv_logger import CsvLoggerThread
from serial_worker import SerialWorker, list_serial_ports
from telemetry_packet import FSM_COLORS, FSM_STATES, TelemetryPacket

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
        layout.setContentsMargins(9, 6, 9, 7)
        layout.setSpacing(1)

        self.caption = QLabel(caption.upper())
        caption_font = QFont()
        caption_font.setPointSize(8)
        caption_font.setBold(True)
        self.caption.setFont(caption_font)
        self.caption.setStyleSheet("color: %s; letter-spacing: 1px;" % COL_TEXT_DIM)

        self.value = QLabel("--")
        value_font = QFont("Consolas")
        value_font.setPointSize(value_pt)
        value_font.setBold(True)
        self.value.setFont(value_font)
        self.value.setStyleSheet("color: %s;" % value_color)

        layout.addWidget(self.caption)
        layout.addWidget(self.value)

    def set_value(self, text: str) -> None:
        self.value.setText("%s %s" % (text, self._unit) if self._unit else text)

    def set_level(self, level: str) -> None:
        color = self.LEVEL_COLORS.get(level, self._base_color)
        if level == "normal":
            color = self._base_color
        self.value.setStyleSheet("color: %s;" % color)


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
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle(title, color=COL_TEXT, size="10pt")
        self.setLabel("left", y_label, color=COL_TEXT_DIM)
        self.setLabel("bottom", "mission time", units="s", color=COL_TEXT_DIM)
        self.showGrid(x=True, y=True, alpha=0.22)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=True, y=True)
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.getPlotItem().setContentsMargins(4, 4, 10, 4)

        self._x: List[float] = []
        self._series_names: List[str] = [name for name, _ in series]
        self._y: Dict[str, List[float]] = {name: [] for name, _ in series}
        self._curves: Dict[str, pg.PlotDataItem] = {}

        if len(series) > 1:
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
                name=name if len(series) > 1 else None,
                autoDownsample=True,
                clipToView=True,
            )
            self._curves[name] = curve

    # -- data ---------------------------------------------------------------

    def add_point(self, x: float, values: Dict[str, float]) -> None:
        """Append one sample.  Non-finite values are stored as NaN (gap in line)."""
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
        # bisect works because mission time is monotonic for a healthy link; a
        # non-monotonic timestamp only costs a slightly wrong window, never a crash.
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
        self.setMinimumHeight(165)

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
        self.session_start = time.time()
        self.last_packet_epoch: Optional[float] = None
        self.recv_times: deque = deque(maxlen=1000)   # for packets/sec
        self.session_packets = 0
        self.total_frames = 0
        self.valid_packets = 0
        self.corrupt_packets = 0
        self.logging_enabled = False
        self.is_connected = False
        self._readouts_dirty = False
        self._was_stale = False
        self._last_fsm: Optional[int] = None

        # --- worker threads --------------------------------------------------
        self.serial_worker = SerialWorker(self)
        self.csv_logger = CsvLoggerThread(log_dir=log_dir, parent=self)

        self._build_ui()
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

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_column())
        splitter.addWidget(self._build_plot_grid())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 1250])
        root_layout.addWidget(splitter, 1)

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
        self.port_combo.setMinimumWidth(300)
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

        self.log_path_label = QLabel("CSV: not started")
        self.log_path_label.setStyleSheet("color: %s;" % COL_TEXT_DIM)
        layout.addWidget(self.log_path_label, 1)

        self.conn_state_label = QLabel("● DISCONNECTED")
        self.conn_state_label.setStyleSheet("color: %s; font-weight: 700;" % COL_ALERT)
        layout.addWidget(self.conn_state_label)

        return box

    # -- left column -----------------------------------------------------

    def _build_left_column(self) -> QWidget:
        """Left column: a pinned header plus a scrollable remainder.

        The flight-state banner and the primary readouts are deliberately kept
        *outside* the scroll area — on a short laptop screen the operator must
        never be able to scroll the mission-critical state out of view.
        """
        panel = QWidget()
        panel.setMinimumWidth(400)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        outer.addWidget(self._build_fsm_banner())
        outer.addWidget(self._build_readouts())

        scroll_content = QWidget()
        inner = QVBoxLayout(scroll_content)
        inner.setContentsMargins(0, 0, 4, 0)
        inner.setSpacing(8)
        inner.addWidget(self._build_gps_panel(), 1)
        inner.addWidget(self._build_link_panel())
        inner.addWidget(self._build_settings_panel())
        inner.addWidget(self._build_event_log())

        scroll = QScrollArea()
        scroll.setWidget(scroll_content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)
        return panel

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
        self.fsm_label.setMinimumHeight(66)
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

        self.gps_plot = GpsTrackPlot()
        layout.addWidget(self.gps_plot, 1)
        return box

    def _build_link_panel(self) -> QWidget:
        box = QGroupBox("LINK STATUS")
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)

        self.tile_rate = ReadoutTile("Packet rate", "pkt/s", COL_ACCENT, value_pt=15)
        self.tile_age = ReadoutTile("Last packet age", "s", COL_TEXT, value_pt=15)
        self.tile_total = ReadoutTile("Valid / total pkts", "", COL_TEXT, value_pt=15)
        self.tile_corrupt = ReadoutTile("Corrupt packets", "", COL_TEXT, value_pt=15)

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
        box = QGroupBox("DISPLAY SETTINGS")
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)

        grid.addWidget(QLabel("Chart window (s):"), 0, 0)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(5, 1800)
        self.window_spin.setValue(DEFAULT_WINDOW_S)
        self.window_spin.setSingleStep(5)
        grid.addWidget(self.window_spin, 0, 1)

        grid.addWidget(QLabel("Battery warning (V):"), 1, 0)
        self.volt_spin = QDoubleSpinBox()
        self.volt_spin.setRange(0.0, 60.0)
        self.volt_spin.setDecimals(2)
        self.volt_spin.setSingleStep(0.1)
        self.volt_spin.setValue(DEFAULT_VOLTAGE_WARN)
        grid.addWidget(self.volt_spin, 1, 1)

        self.autoscroll_check = QCheckBox("Auto-scale Y axes")
        self.autoscroll_check.setChecked(True)
        grid.addWidget(self.autoscroll_check, 2, 0, 1, 2)

        self.clear_btn = QPushButton("Clear plots && counters")
        grid.addWidget(self.clear_btn, 3, 0, 1, 2)

        grid.setColumnStretch(1, 1)
        return box

    def _build_event_log(self) -> QWidget:
        box = QGroupBox("EVENT LOG")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(EVENT_LOG_LINES)
        self.event_log.setFixedHeight(112)
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
            self.last_packet_epoch = packet.gs_recv_epoch
            self.recv_times.append(packet.gs_recv_epoch)

            # Plot X axis: mission time, normalised so the session starts near 0.
            mission = packet.mission_time_s
            if not math.isfinite(mission):
                mission = packet.gs_recv_epoch - self.session_start
            if self.first_mission_time is None:
                self.first_mission_time = mission
            x = mission

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

    def on_bad_frame(self, raw: str, reason: str) -> None:
        """A frame failed checksum or parsing — always archived, never fatal."""
        # Errors are logged unconditionally so a corrupted-link investigation
        # still has data even if CSV logging was never switched on.
        self.csv_logger.log_error(raw, reason)

    def on_stats(self, total_frames: int, valid: int, corrupt: int) -> None:
        self.total_frames = total_frames
        self.valid_packets = valid
        self.corrupt_packets = corrupt

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

        self._style_fsm(packet.fsm_state)

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

    def clear_session(self) -> None:
        """Wipe the plots and the on-screen counters (does not touch the CSV)."""
        for chart in self.charts:
            chart.clear_data()
        self.gps_plot.clear_data()
        self.recv_times.clear()
        self.session_packets = 0
        self.first_mission_time = None
        self.session_start = time.time()
        self.serial_worker.reset_counters()
        self.total_frames = self.valid_packets = self.corrupt_packets = 0
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

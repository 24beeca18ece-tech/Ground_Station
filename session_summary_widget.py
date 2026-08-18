"""
session_summary_widget.py
=========================

Live "at a glance" summary of the current session, as a donut chart with a
legend.  Two views are available from the dropdown:

* **Time in FSM state** (default) -- what proportion of the mission has been
  spent in each flight state, colour-matched to the flight-state banner.  This
  is the most immediately meaningful one-glance summary of a flight: a reviewer
  can see the whole mission profile without reading a single number.
* **Packet integrity** -- valid vs. checksum-failed vs. frame resync events, as
  a proportion of everything the receiver pulled off the link this session.
  This is the link-quality summary to put on screen during a demo.

WHY QPainter AND NOT matplotlib OR pyqtgraph
--------------------------------------------
Three options were considered:

* *pyqtgraph* -- already a dependency and used for every other plot, but it has
  no pie/donut primitive.  Building one means assembling ``QGraphicsPathItem``
  wedges by hand inside a ``PlotItem`` and then fighting its axis/aspect
  machinery, which is more code than drawing the arcs directly.
* *matplotlib* with ``FigureCanvasQTAgg`` -- has a one-line ``pie()``, but it
  is a ~30 MB dependency added for a single widget, it needs its own styling
  pass to match the dark theme, and each redraw rasterises a whole figure.
* *QPainter* on a plain ``QWidget`` -- a donut is a handful of ``drawPie``
  calls.  Zero new dependencies, exact theme control, and a redraw costs
  microseconds.

QPainter won: it is the lightest of the three and it is the only one that adds
nothing to ``requirements.txt``.  A pie chart is arcs and text; it does not
need a plotting library.

THREADING / UPDATE RATE
-----------------------
:meth:`SessionSummaryWidget.on_packet` is a cheap GUI-thread slot that only
accumulates counters.  Repainting is driven by a 1 Hz timer owned by this
widget -- there is no value in redrawing a proportion chart at 20 Hz, and doing
so would burn cycles the strip charts need.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from telemetry_packet import FSM_COLORS, FSM_STATES, FSM_UNKNOWN_COLOR

# Palette, matched to dashboard_ui.py.
COL_PANEL = "#161d27"
COL_TEXT = "#dbe3ee"
COL_TEXT_DIM = "#8b9aad"
COL_OK = "#35c46b"
COL_WARN = "#e9c135"
COL_ALERT = "#e8384f"

#: Redraw interval.  Proportions change slowly; 1 Hz is plenty.
SUMMARY_REDRAW_MS = 1000

#: Largest gap that may be attributed to a single state, in seconds.  After a
#: link dropout the next packet can be many seconds later; charging all of that
#: silence to whichever state happened to be current would badly distort the
#: chart, so the gap is capped.
MAX_STATE_DT_S = 2.0

VIEW_FSM = "Time in FSM state"
VIEW_INTEGRITY = "Packet integrity"

# Integrity slice colours.
COL_VALID = COL_OK
COL_CORRUPT = COL_ALERT
COL_RESYNC = COL_WARN


class DonutChart(QWidget):
    """Donut chart plus legend, drawn with QPainter.

    Slices are supplied as ``(label, value, "#rrggbb")`` tuples.  Zero-valued
    slices are dropped rather than drawn as invisible slivers that still take a
    legend row.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._slices: List[Tuple[str, float, str]] = []
        self._centre_value = ""
        self._centre_caption = ""
        self._unit = ""
        self._empty_text = "NO DATA YET"

    def set_data(self, slices, centre_value: str = "", centre_caption: str = "",
                 unit: str = "", empty_text: str = "NO DATA YET") -> None:
        self._slices = [(n, float(v), c) for n, v, c in slices if v and v > 0]
        self._centre_value = centre_value
        self._centre_caption = centre_caption
        self._unit = unit
        self._empty_text = empty_text
        self.update()

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor(COL_PANEL))

        total = sum(value for _, value, _ in self._slices)
        if total <= 0.0:
            painter.setPen(QColor(COL_TEXT_DIM))
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignCenter, self._empty_text)
            painter.end()
            return

        # Reserve the left square for the donut, the rest for the legend.
        margin = 8
        diameter = max(60, min(rect.height() - 2 * margin,
                               int(rect.width() * 0.46)))
        pie_rect = QRectF(margin, (rect.height() - diameter) / 2.0,
                          diameter, diameter)

        # Slices, largest first so the eye lands on the dominant one.
        ordered = sorted(self._slices, key=lambda s: s[1], reverse=True)

        painter.setPen(Qt.NoPen)
        start = 90 * 16               # start at 12 o'clock
        for _, value, color in ordered:
            span = -int(round(value / total * 360.0 * 16.0))  # clockwise
            painter.setBrush(QColor(color))
            painter.drawPie(pie_rect, start, span)
            start += span

        # Punch the donut hole using the panel colour.
        hole = diameter * 0.56
        hole_rect = QRectF(
            pie_rect.center().x() - hole / 2.0,
            pie_rect.center().y() - hole / 2.0,
            hole, hole,
        )
        painter.setBrush(QColor(COL_PANEL))
        painter.drawEllipse(hole_rect)

        # Centre readout.
        if self._centre_value:
            painter.setPen(QColor(COL_TEXT))
            value_font = QFont("Consolas")
            value_font.setPointSize(max(8, int(diameter * 0.11)))
            value_font.setBold(True)
            painter.setFont(value_font)
            painter.drawText(
                QRectF(hole_rect.x(), hole_rect.y() + hole * 0.16,
                       hole, hole * 0.45),
                Qt.AlignCenter, self._centre_value,
            )
        if self._centre_caption:
            painter.setPen(QColor(COL_TEXT_DIM))
            cap_font = QFont()
            cap_font.setPointSize(7)
            painter.setFont(cap_font)
            painter.drawText(
                QRectF(hole_rect.x(), hole_rect.y() + hole * 0.55,
                       hole, hole * 0.3),
                Qt.AlignCenter, self._centre_caption,
            )

        # Legend.
        legend_x = margin + diameter + 12
        legend_w = rect.width() - legend_x - margin
        if legend_w < 60:
            painter.end()
            return

        row_h = min(19, max(13, int((rect.height() - 2 * margin) / max(len(ordered), 1))))
        legend_font = QFont()
        legend_font.setPointSize(8)
        painter.setFont(legend_font)

        y = (rect.height() - row_h * len(ordered)) / 2.0
        for name, value, color in ordered:
            pct = value / total * 100.0

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(
                QRectF(legend_x, y + row_h * 0.24, 9, 9), 2, 2
            )

            painter.setPen(QColor(COL_TEXT))
            text_rect = QRectF(legend_x + 15, y, legend_w - 15 - 74, row_h)
            metrics = painter.fontMetrics()
            painter.drawText(
                text_rect, Qt.AlignVCenter | Qt.AlignLeft,
                metrics.elidedText(name, Qt.ElideRight, int(text_rect.width())),
            )

            painter.setPen(QColor(COL_TEXT_DIM))
            painter.drawText(
                QRectF(legend_x + legend_w - 74, y, 74, row_h),
                Qt.AlignVCenter | Qt.AlignRight,
                "%5.1f%%  %s" % (pct, self._format_value(value)),
            )
            y += row_h

        painter.end()

    def _format_value(self, value: float) -> str:
        if self._unit == "s":
            return "%.0fs" % value
        return "%d" % int(round(value))


class SessionSummaryWidget(QWidget):
    """Panel wrapping :class:`DonutChart` with a view selector."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # --- accumulated session state -------------------------------------
        self._state_seconds: Dict[int, float] = {}
        self._last_state: Optional[int] = None
        self._last_time: Optional[float] = None
        self.valid_packets = 0
        self.corrupt_packets = 0
        self.resyncs = 0
        self._dirty = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        header = QHBoxLayout()
        header.setSpacing(6)
        caption = QLabel("View:")
        caption.setStyleSheet("color: %s;" % COL_TEXT_DIM)
        header.addWidget(caption)

        self.view_combo = QComboBox()
        self.view_combo.addItems([VIEW_FSM, VIEW_INTEGRITY])
        self.view_combo.setToolTip(
            "Time in FSM state: proportion of mission time spent in each flight "
            "state.\nPacket integrity: valid vs. corrupt vs. resync events on the "
            "link this session."
        )
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        header.addWidget(self.view_combo, 1)
        layout.addLayout(header)

        self.chart = DonutChart()
        layout.addWidget(self.chart, 1)

        # Own redraw timer: proportions do not need the 15 Hz chart cadence.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.redraw)
        self._timer.start(SUMMARY_REDRAW_MS)
        self.redraw()

    # -- data intake (cheap GUI-thread slots) ------------------------------

    def on_packet(self, packet) -> None:
        """Accumulate time-in-state from one validated packet.

        Time is attributed to the state carried by the *previous* packet, since
        that is the state the vehicle was in over the interval between them.
        """
        try:
            now = packet.mission_time_s
            if not math.isfinite(now):
                now = packet.gs_recv_epoch

            if self._last_state is not None and self._last_time is not None:
                dt = now - self._last_time
                if dt > 0.0:
                    # Cap so a dropout is not charged entirely to one state.
                    dt = min(dt, MAX_STATE_DT_S)
                    self._state_seconds[self._last_state] = (
                        self._state_seconds.get(self._last_state, 0.0) + dt
                    )
                    self._dirty = True

            self._last_state = packet.fsm_state
            self._last_time = now
        except Exception:
            # A summary panel fault must never interrupt ingestion.
            pass

    def set_link_stats(self, valid: int, corrupt: int, resyncs: int) -> None:
        """Update the integrity counters from the serial worker's statistics."""
        if (valid, corrupt, resyncs) != (self.valid_packets,
                                         self.corrupt_packets, self.resyncs):
            self.valid_packets = valid
            self.corrupt_packets = corrupt
            self.resyncs = resyncs
            self._dirty = True

    def clear(self) -> None:
        """Session reset hook, called by the dashboard's Clear button."""
        self._state_seconds.clear()
        self._last_state = None
        self._last_time = None
        self.valid_packets = 0
        self.corrupt_packets = 0
        self.resyncs = 0
        self._dirty = True
        self.redraw()

    # -- rendering ---------------------------------------------------------

    def _on_view_changed(self, _index: int) -> None:
        self._dirty = True
        self.redraw()

    def redraw(self) -> None:
        """Rebuild the slice list and repaint.  Driven by the 1 Hz timer."""
        if not self._dirty:
            return
        self._dirty = False
        try:
            if self.view_combo.currentText() == VIEW_INTEGRITY:
                self._render_integrity()
            else:
                self._render_fsm()
        except Exception:
            pass

    def _render_fsm(self) -> None:
        slices = []
        for state, seconds in self._state_seconds.items():
            name = FSM_STATES.get(state, "UNKNOWN(%s)" % state)
            color = FSM_COLORS.get(state, FSM_UNKNOWN_COLOR)
            slices.append((name, seconds, color))

        total = sum(s for _, s, _ in slices)
        self.chart.set_data(
            slices,
            centre_value=("%.0fs" % total) if total else "",
            centre_caption="mission time" if total else "",
            unit="s",
            empty_text="AWAITING TELEMETRY",
        )

    def _render_integrity(self) -> None:
        slices = [
            ("Valid packets", self.valid_packets, COL_VALID),
            ("Corrupt (checksum)", self.corrupt_packets, COL_CORRUPT),
            ("Resync events", self.resyncs, COL_RESYNC),
        ]
        total = self.valid_packets + self.corrupt_packets + self.resyncs
        good = (self.valid_packets / total * 100.0) if total else 0.0
        self.chart.set_data(
            slices,
            centre_value=("%.1f%%" % good) if total else "",
            centre_caption="good" if total else "",
            unit="",
            empty_text="AWAITING TELEMETRY",
        )

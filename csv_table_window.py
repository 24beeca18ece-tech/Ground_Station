"""
csv_table_window.py
===================

Live tabular view of the current logging session's CSV, one row per packet and
one column per CSV field.

WHERE THE DATA COMES FROM
-------------------------
Rows arrive on ``CsvLoggerThread.row_written``, which the logger emits with the
*same list it just handed to ``csv.writer``*.  The file on disk is never read
back: re-parsing it on every update would put a second reader in contention
with the thread actively appending to it, and would show stale content anyway
because the writer only flushes periodically.  What this table shows is exactly
what was written, at the moment it was written.

PERFORMANCE
-----------
A ``QAbstractTableModel`` over a bounded deque rather than a ``QTableWidget``.
A widget-per-cell table at 35 columns and tens of thousands of rows would
allocate hundreds of thousands of item objects; a model hands Qt only the cells
currently on screen.  Incoming rows are buffered and flushed into the model on a
timer (:data:`FLUSH_MS`) so a 20 Hz packet stream costs a handful of model
updates per second, not twenty.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Deque, List, Optional

from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from telemetry_packet import CSV_HEADER

# Palette, matched to the rest of the application.
COL_BG = "#0a0e13"
COL_PANEL = "#12171f"
COL_BORDER = "#2b3746"
COL_TEXT = "#dbe3ee"
COL_DIM = "#8b9aad"
COL_HEADER = "#4aa8ff"
COL_NUM = "#7fd6ff"
COL_OK = "#35c46b"
COL_ALERT = "#e8384f"

#: Rows retained in memory.  ~17 minutes at 20 Hz; the CSV on disk keeps
#: everything, this is only the scrollback.
MAX_ROWS = 20000

#: How often buffered rows are pushed into the model.
FLUSH_MS = 200

#: Distance from the bottom, in pixels, still counted as "at the bottom" for
#: the follow-tail behaviour.
STICKY_BOTTOM_PX = 32


class CsvTableModel(QAbstractTableModel):
    """Bounded, filterable table over the rows written to the CSV."""

    def __init__(self, headers: List[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._headers = list(headers)
        self._rows: Deque[List[Any]] = deque(maxlen=MAX_ROWS)
        self._view: List[List[Any]] = []      # rows passing the current filter
        self._filter = ""
        try:
            self._state_col = self._headers.index("fsm_state_name")
            self._valid_col = self._headers.index("checksum_valid")
        except ValueError:                    # pragma: no cover - schema change
            self._state_col = self._valid_col = -1

    # -- Qt model interface ------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:      # noqa: N802
        return 0 if parent.isValid() else len(self._view)

    def columnCount(self, parent=QModelIndex()) -> int:   # noqa: N802
        return 0 if parent.isValid() else len(self._headers)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._headers[section]
            return section + 1                # 1-based row numbers
        if role == Qt.ForegroundRole and orientation == Qt.Horizontal:
            return QColor(COL_HEADER)
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self._view[index.row()]
        col = index.column()
        if role == Qt.DisplayRole:
            return "" if col >= len(row) else str(row[col])
        if role == Qt.ForegroundRole:
            # A failed checksum should never appear here (only validated packets
            # are logged), so flag it loudly if one ever does.
            if col == self._valid_col and col < len(row) and str(row[col]) != "1":
                return QColor(COL_ALERT)
            if col == self._state_col:
                return QColor(COL_OK)
            return QColor(COL_NUM if col > 2 else COL_TEXT)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    # -- data intake -------------------------------------------------------

    def append_rows(self, rows: List[List[Any]]) -> None:
        """Add rows, keeping the filtered view consistent."""
        if not rows:
            return
        matching = [r for r in rows if self._matches(r)]
        # The deque may evict from the head; if it does, the filtered view has
        # to be rebuilt rather than merely extended.
        evicting = len(self._rows) + len(rows) > MAX_ROWS
        self._rows.extend(rows)
        if evicting:
            self.rebuild()
            return
        if matching:
            first = len(self._view)
            self.beginInsertRows(QModelIndex(), first, first + len(matching) - 1)
            self._view.extend(matching)
            self.endInsertRows()

    def set_filter(self, text: str) -> None:
        self._filter = (text or "").strip().lower()
        self.rebuild()

    def rebuild(self) -> None:
        self.beginResetModel()
        self._view = [r for r in self._rows if self._matches(r)]
        self.endResetModel()

    def clear(self) -> None:
        self.beginResetModel()
        self._rows.clear()
        self._view = []
        self.endResetModel()

    def _matches(self, row: List[Any]) -> bool:
        """Case-insensitive substring match against any cell in the row."""
        if not self._filter:
            return True
        for cell in row:
            if self._filter in str(cell).lower():
                return True
        return False

    @property
    def total_rows(self) -> int:
        return len(self._rows)


class CsvTableWindow(QDialog):
    """Non-modal window showing the live CSV as a scrollable table."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("CSV Log — no session")
        self.resize(1150, 620)
        self.setStyleSheet(
            "QDialog { background-color: %s; }"
            "QLineEdit { background: %s; color: %s; border: 1px solid %s;"
            "            border-radius: 4px; padding: 5px 8px; }"
            "QCheckBox { color: %s; }"
            % (COL_PANEL, COL_BG, COL_TEXT, COL_BORDER, COL_DIM)
        )

        self._pending: List[List[Any]] = []
        self._csv_path: Optional[str] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(7)

        # --- header row: file, filter, follow toggle ------------------------
        top = QHBoxLayout()
        top.setSpacing(8)

        self.path_label = QLabel("No logging session started")
        self.path_label.setStyleSheet("color: %s;" % COL_DIM)
        top.addWidget(self.path_label, 1)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter rows (e.g. DESCENT, 2025ASI001)…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setFixedWidth(300)
        self.filter_edit.setToolTip(
            "Case-insensitive substring match against every cell in a row.\n"
            "Type a flight state (DESCENT), a team ID, or any value fragment."
        )
        self.filter_edit.textChanged.connect(self._on_filter)
        top.addWidget(self.filter_edit)

        self.follow_check = QCheckBox("Follow")
        self.follow_check.setChecked(True)
        self.follow_check.setToolTip(
            "Keep the newest row in view.\n"
            "Automatically suspends while you scroll up to read history, and\n"
            "resumes when you scroll back to the bottom."
        )
        top.addWidget(self.follow_check)

        self.clear_btn = QPushButton("Clear view")
        self.clear_btn.setToolTip(
            "Clear this table only. The CSV file on disk is not touched."
        )
        self.clear_btn.clicked.connect(self.clear)
        top.addWidget(self.clear_btn)
        outer.addLayout(top)

        # --- table -----------------------------------------------------------
        self.model = CsvTableModel(CSV_HEADER, self)
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.setAlternatingRowColors(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.setStyleSheet(
            "QTableView { background: %s; alternate-background-color: #10161e;"
            "             color: %s; gridline-color: %s;"
            "             selection-background-color: #24507d; }"
            "QHeaderView::section { background: #172231; color: %s;"
            "             border: 0; border-right: 1px solid %s; padding: 4px 6px;"
            "             font-weight: 600; }"
            "QTableCornerButton::section { background: #172231; border: 0; }"
            % (COL_BG, COL_TEXT, COL_BORDER, COL_HEADER, COL_BORDER)
        )
        mono = QFont("Consolas")
        mono.setPointSize(9)
        self.view.setFont(mono)
        self.view.verticalHeader().setDefaultSectionSize(19)
        self.view.verticalHeader().setStyleSheet(
            "QHeaderView::section { background: #101720; color: %s; border: 0;"
            " padding-right: 6px; }" % COL_DIM
        )
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(self.view, 1)

        self.status = QLabel("0 rows")
        self.status.setStyleSheet("color: %s;" % COL_DIM)
        outer.addWidget(self.status)

        # Buffered flush: a 20 Hz stream becomes 5 model updates a second.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._flush)
        self._timer.start(FLUSH_MS)

        self.view.resizeColumnsToContents()

    # -- intake ------------------------------------------------------------

    def add_row(self, row) -> None:
        """Queue one CSV record. Cheap: called once per logged packet."""
        self._pending.append(list(row))

    def set_csv_path(self, path: str) -> None:
        """Point the window at a new session file and clear the old rows."""
        self._csv_path = path
        name = os.path.basename(path)
        self.setWindowTitle("CSV Log — %s" % name)
        self.path_label.setText(path)
        self.path_label.setToolTip(path)
        # A new session is a new file; the previous session's rows do not
        # belong in a view labelled with this file's name.
        self.clear()

    def clear(self) -> None:
        self._pending.clear()
        self.model.clear()
        self._update_status()

    # -- rendering ---------------------------------------------------------

    def _at_bottom(self) -> bool:
        bar = self.view.verticalScrollBar()
        return bar.value() >= bar.maximum() - STICKY_BOTTOM_PX

    def _flush(self) -> None:
        if not self._pending:
            return
        # Decide *before* inserting: appending rows moves the maximum, so the
        # test has to be made against the pre-insert scroll position.
        stick = self.follow_check.isChecked() and self._at_bottom()
        rows, self._pending = self._pending, []
        self.model.append_rows(rows)
        if stick:
            self.view.scrollToBottom()
        self._update_status()

    def _on_filter(self, text: str) -> None:
        self.model.set_filter(text)
        self._update_status()
        if self.follow_check.isChecked():
            self.view.scrollToBottom()

    def _update_status(self) -> None:
        shown, total = self.model.rowCount(), self.model.total_rows
        if shown == total:
            self.status.setText("%d rows" % total)
        else:
            self.status.setText("%d of %d rows match the filter" % (shown, total))

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        self._flush()
        self.view.resizeColumnsToContents()
        if self.follow_check.isChecked():
            self.view.scrollToBottom()

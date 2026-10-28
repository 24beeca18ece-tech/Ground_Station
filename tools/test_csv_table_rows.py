"""
tools/test_csv_table_rows.py
============================

The live CSV table stores each row as one compact string. What it shows,
filters and numbers must be exactly what it did when rows were lists.

    python -m pytest tools/test_csv_table_rows.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv_table_window as ctw  # noqa: E402
from telemetry_packet import CSV_HEADER  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def row(seq, state="DESCENT", valid=1):
    values = {name: "" for name in CSV_HEADER}
    values.update(seq=seq, checksum_valid=valid, fsm_state_name=state,
                  team_id="2025ASI001", altitude_m=412.5 + seq,
                  gyro_x=-0.35, raw_frame='$2025ASI001,CANSAT,%d,"q",1.0*4D' % seq)
    return [values[name] for name in CSV_HEADER]


def cell(model, r, name, role=None):
    from PyQt5.QtCore import Qt
    index = model.index(r, CSV_HEADER.index(name))
    return model.data(index, role if role is not None else Qt.DisplayRole)


def test_cells_show_the_same_text_as_before(qapp):
    model = ctw.CsvTableModel(list(CSV_HEADER))
    original = row(7)
    model.append_rows([original])
    for c, value in enumerate(original):
        assert model.data(model.index(0, c)) == str(value)


def test_newest_first_and_the_seq_header(qapp):
    from PyQt5.QtCore import Qt
    model = ctw.CsvTableModel(list(CSV_HEADER))
    model.append_rows([row(1), row(2)])
    model.append_rows([row(3)])
    assert [cell(model, r, "seq") for r in range(3)] == ["3", "2", "1"]
    assert model.headerData(0, Qt.Vertical) == "3"


def test_filter_matches_any_cell_case_insensitively_never_across_cells(qapp):
    model = ctw.CsvTableModel(list(CSV_HEADER))
    model.append_rows([row(1, "ASCENT"), row(2, "DESCENT"), row(3, "LANDED")])
    model.set_filter("Descent")
    assert [cell(model, r, "fsm_state_name") for r in range(model.rowCount())] \
        == ["DESCENT"]
    model.set_filter("scent")                        # ASCENT and DESCENT
    assert sorted(cell(model, r, "fsm_state_name")
                  for r in range(model.rowCount())) == ["ASCENT", "DESCENT"]
    model.set_filter("")
    assert model.rowCount() == 3
    # A filter made of the end of one cell and the start of the next matched
    # nothing when rows were lists, and must not now that they are joined.
    two = ctw.CsvTableModel(["a", "b"])
    two.append_rows([["xy", "zw"]])
    for text, expected in (("xy", 1), ("zw", 1), ("yz", 0), ("y z", 0)):
        two.set_filter(text)
        assert two.rowCount() == expected, text


def test_failed_checksum_is_still_flagged(qapp):
    from PyQt5.QtCore import Qt
    model = ctw.CsvTableModel(list(CSV_HEADER))
    model.append_rows([row(1, valid=0)])
    colour = cell(model, 0, "checksum_valid", Qt.ForegroundRole)
    assert colour.name() == ctw.QColor(ctw.COL_ALERT).name()


def test_bounded_at_max_rows(qapp, monkeypatch):
    monkeypatch.setattr(ctw, "MAX_ROWS", 50)
    model = ctw.CsvTableModel(list(CSV_HEADER))
    for start in range(0, 120, 10):
        model.append_rows([row(s) for s in range(start, start + 10)])
    assert model.total_rows == 50 and model.rowCount() == 50
    assert cell(model, 0, "seq") == "119" and cell(model, 49, "seq") == "70"

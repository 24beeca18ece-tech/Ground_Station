"""
tools/test_ui_cleanup.py
========================

The ``ui-cleanup`` dashboard changes:

* ``SHOW_WHEEL_RPM`` hides the reaction-wheel tile and chart without leaving
  anything half wired, and turning it back on restores them.
* The altitude callout speaks exact whole metres, not the nearest 10 m.
* The link status panel has no antenna box; Signal moves up beside
  Valid/Total and Corrupt/Rej spans the bottom row.

    python -m pytest tools/test_ui_cleanup.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dashboard_ui  # noqa: E402
from telemetry_packet import build_frame, parse_frame  # noqa: E402

_TEMPLATE = ("2025ASI001,CANSAT,{t:.2f},{n},235.94,985.23,27.40,7.85,"
             "09:05:07,32.719876,74.861234,312.4,11,0.12,-0.05,9.79,"
             "0.40,-1.20,0.06,1,4.10,6.20,7.00,7.40,{wheel},{stage}")


def cansat_packet(n, wheel="-480", stage="2"):
    return parse_frame(build_frame(
        _TEMPLATE.format(t=n * 0.1, n=n, wheel=wheel, stage=stage)))


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def make_dashboard(qapp, tmp_path, monkeypatch):
    windows = []

    def make(show_wheel):
        monkeypatch.setattr(dashboard_ui, "SHOW_WHEEL_RPM", show_wheel)
        window = dashboard_ui.Dashboard(log_dir=str(tmp_path))
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.close()


def _feed(window, qapp, packets):
    # Ingestion is off until START TELEMETRY; turn it on as the operator would.
    window.telemetry_active = True
    for packet in packets:
        window.on_packet(packet)
        window._render()
    qapp.processEvents()


def _errors(window):
    return [line for line in window.event_log.toPlainText().splitlines()
            if "error" in line.lower() or "uncaught" in line.lower()]


# ---------------------------------------------------------------------------
# Reaction wheel RPM switch
# ---------------------------------------------------------------------------

def test_wheel_hidden_builds_no_wheel_widgets(make_dashboard):
    window = make_dashboard(False)
    assert window.tile_wheel is None
    assert window.chart_wheel is None
    labels = [tile.caption.text() if hasattr(tile, "caption") else ""
              for tile in window.findChildren(dashboard_ui.ReadoutTile)]
    assert not any("wheel" in text.lower() for text in labels)
    assert "wheel" not in window.reset_sensors_btn.toolTip().lower()


def test_wheel_hidden_recovery_has_the_row_to_itself(make_dashboard, qapp):
    window = make_dashboard(False)
    window.resize(1366, 768)
    window.show()
    _feed(window, qapp, [cansat_packet(n) for n in range(1, 4)])
    page = window.payload_stack.widget(1)
    assert window.payload_stack.currentIndex() == 1
    # Full width of the page, give or take the layout margins.
    assert window.tile_recovery.width() >= page.width() - 2
    # The particulate chart took the wheel chart's height.
    assert window.chart_pm.maximumHeight() > 104
    assert window.chart_pm.height() > 104


@pytest.mark.parametrize("show_wheel", [False, True])
def test_dashboard_takes_cansat_packets_either_way(make_dashboard, qapp,
                                                   show_wheel):
    window = make_dashboard(show_wheel)
    window.show()
    packets = [cansat_packet(n) for n in range(1, 30)]
    # Mix in frames without actuator data, the raw-CSV firmware's shape.
    packets += [cansat_packet(n, wheel="", stage="") for n in range(30, 35)]
    _feed(window, qapp, packets)
    assert window.payload_stack.currentIndex() == 1
    assert window.tile_recovery.value.text().strip() == "--"
    assert _errors(window) == []

    window.reset_sensor_data()
    window.clear_session()
    _feed(window, qapp, [cansat_packet(n) for n in range(40, 45)])
    assert _errors(window) == []

    if show_wheel:
        assert window.tile_wheel.value.text().strip().startswith("-480")
        assert window.chart_wheel._n > 0
        assert "wheel" in window.reset_sensors_btn.toolTip().lower()
    else:
        assert window.tile_wheel is None and window.chart_wheel is None
    # The packet itself is untouched by the display switch.
    assert packets[0].reaction_wheel_rpm == -480
    assert packets[0].has_actuator_data


# ---------------------------------------------------------------------------
# Altitude callout phrases: exact whole metres
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("altitude,phrase", [
    (503, "Altitude 503 meters"),
    (0, "Altitude 0 meters"),
    (9, "Altitude 9 meters"),
    (10, "Altitude 10 meters"),
    (99.5, "Altitude 100 meters"),       # half rounds away from zero
    (-3, "Altitude minus 3 meters"),
    (1250, "Altitude 1250 meters"),
    (-1250, "Altitude minus 1250 meters"),
    (593.4, "Altitude 593 meters"),      # was "590" with 10 m rounding
    (-2.5, "Altitude minus 3 meters"),
    (-0.4, "Altitude 0 meters"),         # never "minus 0"
])
def test_callout_speaks_the_exact_altitude(altitude, phrase):
    from altitude_callout import format_altitude
    assert format_altitude(altitude) == phrase


def test_callout_has_no_rounding_setting_left():
    import altitude_callout
    assert not hasattr(altitude_callout, "ROUND_TO_M")


# ---------------------------------------------------------------------------
# Link status panel: no antenna box, three rows
# ---------------------------------------------------------------------------

def _grid_position(grid, widget):
    index = grid.indexOf(widget)
    assert index >= 0, "widget is not in the link grid"
    return grid.getItemPosition(index)       # (row, column, rowSpan, colSpan)


def test_link_grid_layout(make_dashboard):
    from PyQt5.QtWidgets import QGridLayout
    window = make_dashboard(False)
    grid = window.tile_rate.parentWidget().layout()
    assert isinstance(grid, QGridLayout)
    assert _grid_position(grid, window.tile_rate) == (0, 0, 1, 1)
    assert _grid_position(grid, window.tile_age) == (0, 1, 1, 1)
    assert _grid_position(grid, window.tile_total) == (1, 0, 1, 1)
    assert _grid_position(grid, window.tile_rssi) == (1, 1, 1, 1)
    assert _grid_position(grid, window.tile_corrupt) == (2, 0, 1, 2)
    assert grid.count() == 5
    assert grid.rowCount() == 3


def test_no_antenna_widget_exists(make_dashboard):
    from PyQt5.QtWidgets import QComboBox, QLabel
    window = make_dashboard(False)
    assert not hasattr(dashboard_ui, "ANTENNA_PATHS")
    assert not hasattr(window, "antenna_combo")
    assert not hasattr(window, "_on_antenna_changed")
    panel = window.tile_rate.parentWidget()
    assert not any(label.text().strip().upper() == "ANTENNA"
                   for label in window.findChildren(QLabel))
    assert panel.findChildren(QComboBox) == []


def test_link_tiles_still_update(make_dashboard, qapp):
    window = make_dashboard(False)
    window.show()
    _feed(window, qapp, [cansat_packet(n) for n in range(1, 6)])
    window._update_link_status()
    assert window.tile_total.value.text().strip() not in ("", "--")
    assert window.tile_corrupt.value.text().strip().count("/") == 2
    assert _errors(window) == []

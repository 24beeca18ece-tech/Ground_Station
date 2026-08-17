#!/usr/bin/env python3
"""
main.py — Ground Control Station entry point
============================================

Launches the QApplication and the :class:`dashboard_ui.Dashboard` main window.

THREADING OVERVIEW (the whole picture, in one place)
---------------------------------------------------

    +----------------------+        Qt signals        +--------------------+
    |  SerialWorker        |  packet_received(pkt)    |   GUI thread       |
    |  (QThread)           | -----------------------> |   Dashboard        |
    |                      |  bad_frame(raw, reason)  |                    |
    |  * pyserial handle   | -----------------------> |  * all widgets     |
    |  * RX ring buffer    |  stats_updated(...)      |  * pyqtgraph plots |
    |  * $..*XX framing    | -----------------------> |  * QTimer repaint  |
    |  * XOR checksum      |  connection_changed(...) |                    |
    |  * auto-reconnect    | -----------------------> |                    |
    +----------------------+                          +---------+----------+
                                                                |
                                            queue.Queue (non-blocking put)
                                                                |
                                                      +---------v----------+
                                                      | CsvLoggerThread    |
                                                      | (QThread)          |
                                                      |  * Flight_<ID>.csv |
                                                      |  * errors.log      |
                                                      |  * periodic flush  |
                                                      +--------------------+

Rules that keep this safe:
  * Only the serial thread touches the serial port.
  * Only the logger thread touches the log files.
  * Only the GUI thread touches widgets.
  * Threads communicate through Qt signals (auto-queued across threads) or a
    ``queue.Queue`` — never through shared mutable widget state.

Run with ``python main.py``.  ``--port``/``--baud`` pre-fill the connection bar,
and ``--autoconnect`` opens the link immediately (handy for field checkouts).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox

# Enable High-DPI scaling before the QApplication exists, otherwise Qt ignores it.
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

from dashboard_ui import Dashboard, apply_dark_palette  # noqa: E402


def _install_exception_hook(window: Dashboard) -> None:
    """Route uncaught GUI-thread exceptions to the event log instead of stderr.

    Without this, PyQt5 prints the traceback and (on some versions) aborts the
    process.  A ground station must survive a display bug mid-flight: the serial
    and logger threads keep running and the operator sees what went wrong.
    """
    original_hook = sys.excepthook

    def hook(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        sys.stderr.write(text)
        try:
            window.append_event("UNCAUGHT %s: %s" % (exc_type.__name__, exc_value))
            window.csv_logger.log_error(text.strip(), "uncaught-gui-exception")
        except Exception:
            pass
        original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = hook


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CanSat / Rocketry Ground Control Station",
    )
    parser.add_argument(
        "--port", default=None,
        help="Pre-select a serial device (COM7, /dev/ttyUSB0) or pyserial URL "
             "(socket://127.0.0.1:5555).",
    )
    parser.add_argument(
        "--baud", type=int, default=9600,
        help="Pre-select the baud rate (default: 9600).",
    )
    parser.add_argument(
        "--autoconnect", action="store_true",
        help="Open the link immediately after the window appears.",
    )
    parser.add_argument(
        "--log-dir", default="logs",
        help="Directory for Flight_<TEAM_ID>.csv and errors.log (default: ./logs).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Ground Control Station")
    app.setOrganizationName("Avionics Team")
    apply_dark_palette(app)

    try:
        window = Dashboard(log_dir=os.path.abspath(args.log_dir))
    except Exception as exc:
        # Failing before the window exists means no event log to write to.
        traceback.print_exc()
        QMessageBox.critical(None, "Startup failed", "%s\n\n%s" % (exc, traceback.format_exc()))
        return 1

    _install_exception_hook(window)

    if args.port:
        window.port_combo.setEditText(args.port)
    window.baud_combo.setCurrentText(str(args.baud))

    window.show()

    if args.autoconnect and args.port:
        # Fire once the event loop is running so the window paints first.
        QTimer.singleShot(300, window.toggle_connection)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

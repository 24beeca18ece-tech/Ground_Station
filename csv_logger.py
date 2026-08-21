"""
csv_logger.py
=============

Disk logging thread.

THREADING MODEL
---------------
This is the *third* thread in the application (see ``serial_worker`` for the
overview).  It exists so that no file I/O ever happens on the GUI thread or on
the serial ingestion thread:

    GUI thread            CsvLoggerThread
    ----------            ---------------
    log_packet(pkt)  -->  queue.Queue  -->  csv.writer  -->  logs/Flight_<TEAM_ID>.csv
    log_error(...)   -->               -->  text file   -->  logs/errors.log

``queue.Queue`` is used rather than a Qt signal because it is a plain
producer/consumer handoff with no Qt object affinity involved, and because a
bounded queue gives us explicit back-pressure behaviour: if the disk stalls, we
drop the *oldest* rows and count the loss rather than blocking the producer.
The producers (``log_packet`` / ``log_error``) never block and never raise.

Durability
----------
The writer flushes at most every :data:`FLUSH_INTERVAL_S` seconds *and* at least
every :data:`FLUSH_EVERY_ROWS` rows, so a crash or a yanked USB cable costs at
most a fraction of a second of telemetry.  ``os.fsync`` is deliberately *not*
called on every flush — at 20 Hz that would thrash the disk; flushing the Python
buffer to the OS is enough to survive a process crash, which is the realistic
failure mode in a field tent.
"""

from __future__ import annotations

import csv
import os
import queue
import time
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from PyQt5.QtCore import QThread, pyqtSignal

from telemetry_packet import CSV_HEADER, TelemetryPacket, safe_filename

#: Default output directory, relative to the working directory.
DEFAULT_LOG_DIR = "logs"

#: Maximum queued items before the oldest are dropped (~50 s of backlog at 20 Hz).
QUEUE_MAX = 1000

#: Flush cadence.
FLUSH_INTERVAL_S = 1.0
FLUSH_EVERY_ROWS = 20

# Queue item kinds
_KIND_PACKET = "packet"
_KIND_ERROR = "error"
_KIND_NOTE = "note"
_KIND_SESSION = "session"


class CsvLoggerThread(QThread):
    """Consumes telemetry packets and bad frames, writes them to disk."""

    #: ``(csv_path,)`` emitted once the flight CSV has been opened.
    file_opened = pyqtSignal(str)
    #: Human readable problem (disk full, permission denied, ...).
    error_occurred = pyqtSignal(str)
    #: ``(rows_written, rows_dropped)`` — emitted at most once a second.
    stats_updated = pyqtSignal(int, int)
    #: One CSV record, emitted as it is written. Carries the same list that
    #: went to disk, so a live table view never has to re-read the file and
    #: cannot contend with this thread for it.
    row_written = pyqtSignal(object)

    def __init__(self, log_dir: str = DEFAULT_LOG_DIR, parent=None) -> None:
        super().__init__(parent)
        self.log_dir = log_dir
        self._queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=QUEUE_MAX)
        self._running = True

        # Owned by the logger thread only.
        self._csv_file = None
        self._csv_writer: Optional[Any] = None
        self._csv_path: Optional[str] = None
        self._error_file = None
        self._rows_since_flush = 0
        self._last_flush = 0.0
        self._last_stats_emit = 0.0
        #: Timestamp for the current session's file name, set by
        #: begin_session() and consumed when the next packet arrives.
        self._session_stamp = None

        # Counters (ints; read from the GUI thread for display only).
        self.rows_written = 0
        self.rows_dropped = 0
        self.errors_written = 0

    # ------------------------------------------------------------------
    # Producer API — safe to call from any thread, never blocks, never raises
    # ------------------------------------------------------------------

    def begin_session(self) -> None:
        """Start a new logging session: close the old file, name a new one.

        The timestamp is taken *here*, when the operator presses START
        LOGGING, not when the first packet arrives -- the file name should
        say when the run began, not when the link happened to deliver.

        Routed through the queue rather than applied directly so it is
        ordered against the packets around it: everything queued before this
        call lands in the previous file, everything after in the new one.
        """
        self._put((_KIND_SESSION,
                   datetime.now().strftime("%Y-%m-%d_%H%M%S")))

    def log_packet(self, packet: TelemetryPacket) -> None:
        """Queue one validated packet for writing."""
        self._put((_KIND_PACKET, packet))

    def log_error(self, raw_frame: str, reason: str) -> None:
        """Queue one rejected frame for the errors log."""
        self._put((_KIND_ERROR, (time.time(), reason, raw_frame)))

    def log_note(self, text: str) -> None:
        """Queue a free-form session note (connects, disconnects, ...)."""
        self._put((_KIND_NOTE, (time.time(), text)))

    def _put(self, item: Tuple[str, Any]) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Back-pressure: drop the oldest item so live data keeps flowing.
            try:
                self._queue.get_nowait()
                self.rows_dropped += 1
                self._queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                self.rows_dropped += 1

    def stop(self) -> None:
        """Ask the thread to drain the queue and exit.  Follow with ``wait()``."""
        self._running = False
        # Wake the blocking get() immediately.
        try:
            self._queue.put_nowait((_KIND_NOTE, (time.time(), "logger stopping")))
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Consumer thread
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Drain the queue until stopped, then flush and close cleanly."""
        self._last_flush = time.monotonic()
        while True:
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                item = None

            if item is not None:
                try:
                    self._handle(item)
                except Exception as exc:  # never let a disk error kill the thread
                    self.error_occurred.emit("Log write failed: %s" % exc)

            self._maybe_flush()
            self._maybe_emit_stats()

            if not self._running and self._queue.empty():
                break

        self._close_files()
        self.stats_updated.emit(self.rows_written, self.rows_dropped)

    # ------------------------------------------------------------------
    # Internals (logger thread only)
    # ------------------------------------------------------------------

    def _handle(self, item: Tuple[str, Any]) -> None:
        kind, payload = item
        if kind == _KIND_PACKET:
            self._write_packet(payload)
        elif kind == _KIND_ERROR:
            stamp, reason, raw = payload
            self._write_error(stamp, reason, raw)
        elif kind == _KIND_NOTE:
            stamp, text = payload
            self._write_error(stamp, "NOTE", text)
        elif kind == _KIND_SESSION:
            # Close the previous session's file; the next packet opens a new
            # one under the stamp taken when START LOGGING was pressed.
            self._close_csv()
            self._session_stamp = payload

    def _write_packet(self, packet: TelemetryPacket) -> None:
        if self._csv_writer is None:
            # The file name needs TEAM_ID, which is only known once the first
            # packet arrives — hence the lazy open.
            self._open_csv(packet.team_id)
        if self._csv_writer is None:
            return  # open failed; error already reported
        # to_csv_row() is polymorphic: CanSatPacket and RocketPacket fill in
        # their own sensor columns and leave the other vehicle's columns blank,
        # so one CSV can hold a mixed session and the row width never varies.
        row: List[Any] = packet.to_csv_row()
        self._csv_writer.writerow(row)
        self.rows_written += 1
        self._rows_since_flush += 1
        # Hand the identical record to any live table view.
        self.row_written.emit(row)

    def _write_error(self, stamp: float, reason: str, raw: str) -> None:
        if self._error_file is None:
            self._open_error_log()
        if self._error_file is None:
            return
        iso = datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat(
            timespec="milliseconds"
        )
        # repr() keeps control characters from a corrupt frame out of the log.
        self._error_file.write("%s\t%s\t%s\n" % (iso, reason, repr(raw)))
        self.errors_written += 1
        self._rows_since_flush += 1

    def _ensure_dir(self) -> bool:
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            return True
        except OSError as exc:
            self.error_occurred.emit("Cannot create %s: %s" % (self.log_dir, exc))
            return False

    def _existing_header_matches(self, path: str) -> bool:
        """True when *path* already starts with the current CSV schema.

        A packet-format change (v1 -> v2) widens ``CSV_HEADER``.  Appending new
        rows to a file written under the old schema would silently produce a CSV
        whose columns stop lining up part-way down -- the kind of corruption
        nobody notices until they try to analyse the flight.
        """
        try:
            with open(path, "r", newline="", encoding="utf-8") as handle:
                first = next(csv.reader(handle), None)
        except (OSError, StopIteration, UnicodeDecodeError):
            return False
        return first == CSV_HEADER

    def _rotate_stale_log(self, path: str) -> Optional[str]:
        """Rename a log written under an older schema so a fresh one can start.

        The old data is preserved under a timestamped name rather than
        overwritten -- previous flights are not ours to delete.
        """
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(path)
        archived = "%s_pre-v2_%s%s" % (base, stamp, ext)
        try:
            os.replace(path, archived)
        except OSError as exc:
            self.error_occurred.emit("Cannot archive %s: %s" % (path, exc))
            return None
        return archived

    def _session_filename(self, team_id: str) -> str:
        """``Flight_<TEAM_ID>_<YYYY-MM-DD>_<HHMMSS>.csv``.

        One file per logging session. Every START LOGGING press gets its own
        timestamp, so a new run can never overwrite or be mixed into a
        previous one -- which is exactly what the old TEAM_ID-only name did.
        """
        stamp = self._session_stamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
        return "Flight_%s_%s.csv" % (safe_filename(team_id), stamp)

    def _open_csv(self, team_id: str) -> None:
        if not self._ensure_dir():
            return
        path = os.path.join(self.log_dir, self._session_filename(team_id))
        try:
            # The name carries a per-session timestamp, so in practice this is
            # always a new file. Append mode is kept anyway: if two sessions
            # ever started inside the same second, appending preserves the
            # earlier rows instead of truncating them away.
            exists = os.path.exists(path) and os.path.getsize(path) > 0
            if exists and not self._existing_header_matches(path):
                archived = self._rotate_stale_log(path)
                if archived is not None:
                    self.error_occurred.emit(
                        "Existing %s used an older column layout; archived it as "
                        "%s and started a fresh log."
                        % (os.path.basename(path), os.path.basename(archived))
                    )
                    exists = False
                else:
                    return  # could not archive; do not corrupt the old file

            handle = open(path, "a", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(CSV_HEADER)
                handle.flush()
        except OSError as exc:
            self.error_occurred.emit("Cannot open %s: %s" % (path, exc))
            return

        self._csv_file = handle
        self._csv_writer = writer
        self._csv_path = path
        self.file_opened.emit(path)

    def _open_error_log(self) -> None:
        if not self._ensure_dir():
            return
        path = os.path.join(self.log_dir, "errors.log")
        try:
            handle = open(path, "a", encoding="utf-8")
            handle.write(
                "# --- GCS session started %s ---\n"
                % datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
            )
        except OSError as exc:
            self.error_occurred.emit("Cannot open %s: %s" % (path, exc))
            return
        self._error_file = handle

    def _maybe_flush(self) -> None:
        if self._rows_since_flush == 0:
            return
        now = time.monotonic()
        if (
            self._rows_since_flush < FLUSH_EVERY_ROWS
            and (now - self._last_flush) < FLUSH_INTERVAL_S
        ):
            return
        self._last_flush = now
        self._rows_since_flush = 0
        for handle in (self._csv_file, self._error_file):
            if handle is not None:
                try:
                    handle.flush()
                except OSError as exc:
                    self.error_occurred.emit("Flush failed: %s" % exc)

    def _maybe_emit_stats(self) -> None:
        now = time.monotonic()
        if (now - self._last_stats_emit) < 1.0:
            return
        self._last_stats_emit = now
        self.stats_updated.emit(self.rows_written, self.rows_dropped)

    def _close_csv(self) -> None:
        """Flush and close just the CSV, leaving the error log open."""
        if self._csv_file is None:
            self._csv_writer = None
            self._csv_path = None
            return
        try:
            self._csv_file.flush()
            os.fsync(self._csv_file.fileno())
        except (OSError, ValueError):
            pass
        try:
            self._csv_file.close()
        except OSError:
            pass
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = None

    def _close_files(self) -> None:
        for attr in ("_csv_file", "_error_file"):
            handle = getattr(self, attr)
            if handle is None:
                continue
            try:
                handle.flush()
                os.fsync(handle.fileno())  # final close: make it durable
            except (OSError, ValueError):
                pass
            try:
                handle.close()
            except OSError:
                pass
            setattr(self, attr, None)
        self._csv_writer = None

    # ------------------------------------------------------------------

    @property
    def csv_path(self) -> Optional[str]:
        """Path of the flight CSV, or ``None`` until the first packet arrives."""
        return self._csv_path

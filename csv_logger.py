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

ROW ORDER AND THE TWO FILES
---------------------------
The flight CSV is written **newest-first**: the most recent packet is always the
first data row under the header.  A CSV cannot be prepended to in place, so this
is not something a ``csv.writer`` can do by appending — the file has to be
rewritten.  Rewriting on every packet would be O(n) per row and O(n^2) over a
session, which at 20 Hz is untenable.

So there are two files, with different jobs:

    Flight_<id>_<stamp>.csv.journal   append-only, oldest-first, durability
    Flight_<id>_<stamp>.csv           newest-first, rewritten periodically

Every packet is appended to the journal immediately and flushed on the cadence
below, so the crash guarantee is exactly what it always was.  The presentation
CSV is regenerated from memory and swapped into place with ``os.replace``, which
is atomic: a reader never sees a half-written file, and a crash mid-rewrite
leaves the previous complete version intact.  On a clean close the journal is
removed, leaving one correctly ordered file.  If it survives, the process died —
and it holds every row appended during that session.

Rewrite cadence is amortised (see :data:`REWRITE_AMORTIZE_DIVISOR`): the longer
the log grows, the less often it is rebuilt, so total bytes written stay close to
linear in the number of rows rather than quadratic.  A hard ceiling
(:data:`REWRITE_MAX_INTERVAL_S`) bounds how stale the physical file can get.

Durability
----------
The writer flushes at most every :data:`FLUSH_INTERVAL_S` seconds *and* at least
every :data:`FLUSH_EVERY_ROWS` rows, so a crash or a yanked USB cable costs at
most a fraction of a second of telemetry.  ``os.fsync`` is deliberately *not*
called on every flush — at 20 Hz that would thrash the disk; flushing the Python
buffer to the OS is enough to survive a process crash, which is the realistic
failure mode in a field tent.  The periodic rewrite *does* fsync, because it is
infrequent and because ``os.replace`` is only meaningful once the new content is
actually on the platter.
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

#: Flush cadence for the append-only journal.
FLUSH_INTERVAL_S = 1.0
FLUSH_EVERY_ROWS = 20

#: Suffix for the append-only companion file.
JOURNAL_SUFFIX = ".journal"

#: Newest-first rewrite cadence.  A rewrite needs *both* enough elapsed time and
#: enough new rows, so a quiet link does not spin the disk and a busy one does
#: not rewrite on every packet.
REWRITE_INTERVAL_S = 2.0
REWRITE_MIN_ROWS = 20

#: Amortisation.  The new-row threshold grows as ``len(rows) // DIVISOR``, so
#: rewrites get further apart as the file gets bigger and the total bytes written
#: over a session stay ~O(n) instead of O(n^2).  Lower = fresher file, more I/O.
REWRITE_AMORTIZE_DIVISOR = 20

#: Ceiling on staleness, for a small log. A *fixed* ceiling would defeat the
#: amortisation above -- forcing a full rewrite every N seconds makes total work
#: quadratic again -- so it grows with the log, one extra base-interval per
#: REWRITE_CEILING_SCALE_ROWS rows, up to a hard cap.
#:
#: A competition flight is 10-20 minutes; at 20 Hz that is 12k-24k rows, where
#: the file is never more than ~30-45 s behind and the session costs a few
#: hundred MB of writes. Hour-long bench sessions trade freshness for I/O on the
#: same curve. Raise REWRITE_INTERVAL_S or REWRITE_AMORTIZE_DIVISOR to spend
#: less disk; lower them for a fresher file.
REWRITE_MAX_INTERVAL_S = 15.0
REWRITE_CEILING_SCALE_ROWS = 12000
REWRITE_MAX_INTERVAL_CAP_S = 90.0

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
        #: Append-only companion file: this is the durability path.
        self._journal_file = None
        self._journal_writer: Optional[Any] = None
        self._journal_path: Optional[str] = None
        #: The newest-first CSV the operator actually opens.
        self._csv_path: Optional[str] = None
        #: Every row of the current session, oldest-first. Reversed on rewrite.
        #: ~250 bytes/row, so ~18 MB per hour at 20 Hz -- worth it to avoid
        #: re-reading the journal on every rebuild.
        self._rows: List[List[Any]] = []
        self._error_file = None
        self._rows_since_flush = 0
        self._rows_since_rewrite = 0
        self._last_flush = 0.0
        self._last_rewrite = 0.0
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
            self._maybe_rewrite()
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
        if self._journal_writer is None:
            # The file name needs TEAM_ID, which is only known once the first
            # packet arrives — hence the lazy open.
            self._open_csv(packet.team_id)
        if self._journal_writer is None:
            return  # open failed; error already reported
        # to_csv_row() is polymorphic: CanSatPacket and RocketPacket fill in
        # their own sensor columns and leave the other vehicle's columns blank,
        # so one CSV can hold a mixed session and the row width never varies.
        row: List[Any] = packet.to_csv_row()
        # Journal first: the row is durable before it is anywhere else.
        self._journal_writer.writerow(row)
        self._rows.append(row)
        self.rows_written += 1
        self._rows_since_flush += 1
        self._rows_since_rewrite += 1
        # Hand the identical record to any live table view. Emitted per packet,
        # so the table is real-time regardless of the file rewrite cadence.
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

    def _load_existing_rows(self, path: str) -> List[List[Any]]:
        """Read the data rows of an existing newest-first CSV, oldest-first.

        Only reachable when two sessions start inside the same second and land
        on the same file name. Reading the rows back keeps the earlier ones
        instead of dropping them, which is what append mode used to buy.
        """
        try:
            with open(path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                next(reader, None)                 # discard the header
                rows = [r for r in reader if r]
        except (OSError, UnicodeDecodeError) as exc:
            self.error_occurred.emit("Cannot re-read %s: %s" % (path, exc))
            return []
        rows.reverse()          # file is newest-first; memory is oldest-first
        return rows

    def _open_csv(self, team_id: str) -> None:
        if not self._ensure_dir():
            return
        path = os.path.join(self.log_dir, self._session_filename(team_id))
        journal_path = path + JOURNAL_SUFFIX
        try:
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

            self._rows = self._load_existing_rows(path) if exists else []

            # The journal is append-only and never re-read during a session, so
            # it is opened in append mode and simply grows.
            journal_new = not (os.path.exists(journal_path)
                               and os.path.getsize(journal_path) > 0)
            handle = open(journal_path, "a", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            if journal_new:
                writer.writerow(CSV_HEADER)
                handle.flush()
        except OSError as exc:
            self.error_occurred.emit("Cannot open %s: %s" % (journal_path, exc))
            return

        self._journal_file = handle
        self._journal_writer = writer
        self._journal_path = journal_path
        self._csv_path = path
        self._rows_since_rewrite = len(self._rows)
        self._last_rewrite = 0.0        # force a rewrite on the next tick
        # Put a header-only (or recovered) file in place immediately, so the
        # path emitted below always names a file that exists and parses.
        self._rewrite_csv()
        self.file_opened.emit(path)

    def _rewrite_csv(self) -> bool:
        """Rebuild the newest-first CSV atomically. Returns True on success.

        Written to a sibling temp file and swapped in with ``os.replace``, which
        is atomic on both Windows and POSIX. Anyone reading the CSV -- a text
        editor, a re-import, the operator mid-flight -- sees either the previous
        complete file or the new complete file, never a partial one.
        """
        if self._csv_path is None:
            return False
        tmp = self._csv_path + ".tmp"
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(CSV_HEADER)
                # reversed() is a view, not a copy: no second list is built.
                writer.writerows(reversed(self._rows))
                handle.flush()
                # The swap is only worth anything once the bytes are down.
                os.fsync(handle.fileno())
            os.replace(tmp, self._csv_path)
        except OSError as exc:
            self.error_occurred.emit("CSV rewrite failed: %s" % exc)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return False
        self._last_rewrite = time.monotonic()
        self._rows_since_rewrite = 0
        return True

    def _maybe_rewrite(self) -> None:
        """Rebuild the newest-first file when it has fallen far enough behind."""
        if self._csv_path is None or self._rows_since_rewrite == 0:
            return
        elapsed = time.monotonic() - self._last_rewrite
        # Threshold grows with the log, so cost per row stays roughly constant.
        threshold = max(REWRITE_MIN_ROWS,
                        len(self._rows) // REWRITE_AMORTIZE_DIVISOR)
        due = elapsed >= REWRITE_INTERVAL_S and self._rows_since_rewrite >= threshold
        ceiling = min(
            REWRITE_MAX_INTERVAL_S
            * (1.0 + len(self._rows) / REWRITE_CEILING_SCALE_ROWS),
            REWRITE_MAX_INTERVAL_CAP_S,
        )
        if due or elapsed >= ceiling:
            self._rewrite_csv()

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
        for handle in (self._journal_file, self._error_file):
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
        """Finalise the session's CSV, leaving the error log open.

        The last rewrite happens here unconditionally, so the file on disk is
        always complete and correctly ordered once a session ends -- whatever
        the rewrite cadence had or had not got round to.
        """
        if self._journal_file is None and self._csv_path is None:
            self._journal_writer = None
            return
        # Order matters: rewrite from memory first, and only discard the
        # journal once the real file is known to be good.
        rewritten = self._rewrite_csv()
        if self._journal_file is not None:
            try:
                self._journal_file.flush()
                os.fsync(self._journal_file.fileno())
            except (OSError, ValueError):
                pass
            try:
                self._journal_file.close()
            except OSError:
                pass
        if rewritten and self._journal_path:
            # The CSV now holds everything the journal did; a leftover journal
            # would only be a puzzle for whoever opens the logs directory.
            try:
                os.remove(self._journal_path)
            except OSError:
                pass          # harmless: it is redundant, not wrong
        self._journal_file = None
        self._journal_writer = None
        self._journal_path = None
        self._csv_path = None
        self._rows = []
        self._rows_since_rewrite = 0

    def _close_files(self) -> None:
        # Same finalisation as ending a session: last rewrite, journal removed.
        self._close_csv()
        handle = self._error_file
        if handle is not None:
            try:
                handle.flush()
                os.fsync(handle.fileno())  # final close: make it durable
            except (OSError, ValueError):
                pass
            try:
                handle.close()
            except OSError:
                pass
            self._error_file = None

    # ------------------------------------------------------------------

    @property
    def csv_path(self) -> Optional[str]:
        """Path of the flight CSV, or ``None`` until the first packet arrives."""
        return self._csv_path

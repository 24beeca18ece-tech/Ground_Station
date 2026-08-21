"""
serial_worker.py
================

Serial ingestion thread for the ground control station.

THREADING MODEL (read this first)
---------------------------------
The application runs three threads:

    [SerialWorker QThread]  --Qt signal-->  [GUI thread]  --queue-->  [CsvLogger QThread]
      * owns the pyserial handle             * owns every widget      * owns the CSV/error
      * owns the RX ring buffer              * never blocks on I/O      file handles
      * does frame sync + checksum
      * never touches a widget

`SerialWorker` is a *long-lived* thread: it is started once when the app opens
and stops once when the app closes.  Connecting and disconnecting the radio does
**not** start/stop the thread, it only flips the ``_want_connected`` flag that
the run loop watches.  That is what makes automatic reconnection free: if the
USB dongle is yanked out mid-flight, the loop notices the exception, closes the
handle, and keeps retrying until the port comes back.

Cross-thread communication rules used here:
  * GUI -> worker : plain attributes guarded by a ``threading.Lock`` (never call
    a worker method that touches pyserial from the GUI thread).
  * worker -> GUI : Qt signals only.  Qt queues them automatically because the
    sender and receiver live in different threads, so the GUI never blocks.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import serial
import serial.tools.list_ports
from PyQt5.QtCore import QThread, pyqtSignal

from telemetry_packet import (
    MAX_FRAME_LEN,
    ChecksumError,
    PacketError,
    PacketParseError,
    parse_frame,
)

#: How long the blocking read waits before returning empty, in seconds.  Small
#: enough that :meth:`SerialWorker.stop` is honoured almost immediately.
READ_TIMEOUT_S = 0.05

#: Delay between reconnection attempts after a failure, in seconds.
RECONNECT_DELAY_S = 1.5

#: Minimum interval between ``stats_updated`` emissions, in seconds.  Emitting
#: on every packet would flood the GUI event loop during a burst.
STATS_EMIT_INTERVAL_S = 0.2

#: RX ring buffer capacity in bytes.  ~64 full frames; the link runs at 20 Hz so
#: this is >3 s of backlog, far more than a healthy reader ever accumulates.
RING_CAPACITY = 8192


def list_serial_ports() -> List[Tuple[str, str]]:
    """Return ``[(device, human_readable_description), ...]`` for every port.

    Safe to call from the GUI thread — it does not open anything.
    """
    ports: List[Tuple[str, str]] = []
    try:
        for info in serial.tools.list_ports.comports():
            description = info.description or "n/a"
            if description in (None, "", "n/a") and info.manufacturer:
                description = info.manufacturer
            ports.append((info.device, "%s — %s" % (info.device, description)))
    except Exception:
        # Enumerating ports can fail on odd Windows driver stacks; an empty list
        # is better than taking the UI down.
        return []
    ports.sort(key=lambda item: item[0])
    return ports


class RingBuffer:
    """Fixed-capacity, drop-oldest byte buffer for the receive stream.

    Implemented over a ``bytearray`` with front-trimming rather than a wrapped
    index pair, because the frame scanner needs contiguous memory to run
    ``bytes.find`` over.  The behaviour is what matters: once the buffer is
    full, the *oldest* bytes are discarded so a stalled/garbage stream can never
    grow memory without bound, and ``dropped_bytes`` records how much was lost.
    """

    __slots__ = ("_buf", "_capacity", "dropped_bytes")

    def __init__(self, capacity: int = RING_CAPACITY) -> None:
        self._buf = bytearray()
        self._capacity = int(capacity)
        self.dropped_bytes = 0

    def __len__(self) -> int:
        return len(self._buf)

    def append(self, data: bytes) -> None:
        """Append received bytes, evicting the oldest data on overflow."""
        if not data:
            return
        self._buf.extend(data)
        overflow = len(self._buf) - self._capacity
        if overflow > 0:
            del self._buf[:overflow]
            self.dropped_bytes += overflow

    def find(self, needle: bytes, start: int = 0) -> int:
        return self._buf.find(needle, start)

    def peek(self, length: int) -> bytes:
        return bytes(self._buf[:length])

    def consume(self, length: int) -> None:
        """Drop the first *length* bytes (already-handled or junk)."""
        if length > 0:
            del self._buf[:length]

    def clear(self) -> None:
        self._buf.clear()


class FrameSplitter:
    """Turns an arbitrary byte stream into complete ``$...*XX`` frames.

    Feeding is incremental: bytes arrive in whatever chunk sizes the driver hands
    over (often mid-frame), so state lives in the ring buffer between calls.

    Resync rules, in order:
      1. Anything before the first ``$`` is junk — discard it.
      2. If a *second* ``$`` appears before the terminating ``*``, the first
         frame was truncated on-air; restart at the second ``$``.
      3. A frame that grows past :data:`MAX_FRAME_LEN` without terminating is
         garbage; drop its ``$`` and rescan (this is what stops a lost ``*``
         from wedging the reader forever).
    """

    __slots__ = ("buffer", "resyncs")

    def __init__(self, capacity: int = RING_CAPACITY) -> None:
        self.buffer = RingBuffer(capacity)
        self.resyncs = 0

    def feed(self, data: bytes) -> List[str]:
        """Add raw bytes and return every complete frame now available."""
        self.buffer.append(data)
        return self._extract()

    def reset(self) -> None:
        """Forget any partial frame (used after a reconnect)."""
        self.buffer.clear()

    def _extract(self) -> List[str]:
        frames: List[str] = []
        buf = self.buffer

        while True:
            start = buf.find(b"$")
            if start < 0:
                # No frame start at all: the whole buffer is junk (keep nothing,
                # a '$' split across reads cannot exist because it is one byte).
                buf.clear()
                break
            if start > 0:
                buf.consume(start)  # rule 1: drop leading junk

            star = buf.find(b"*", 1)
            next_dollar = buf.find(b"$", 1)

            if 0 <= next_dollar and (star < 0 or next_dollar < star):
                # rule 2: truncated frame, restart at the newer '$'
                buf.consume(next_dollar)
                self.resyncs += 1
                continue

            if star < 0:
                # Terminator has not arrived yet.
                if len(buf) > MAX_FRAME_LEN:
                    buf.consume(1)  # rule 3
                    self.resyncs += 1
                    continue
                break

            if star + 3 > len(buf):
                # '*' is here but one/both checksum digits are still in flight.
                if len(buf) > MAX_FRAME_LEN:
                    buf.consume(1)  # rule 3
                    self.resyncs += 1
                    continue
                break

            raw = buf.peek(star + 3)
            buf.consume(star + 3)
            frames.append(raw.decode("ascii", errors="replace"))

        return frames


class SerialWorker(QThread):
    """Reads the XBee serial stream, validates frames, emits parsed packets.

    Signals are the *only* way this object talks to the GUI.
    """

    #: A fully validated packet (``TelemetryPacket``) ready to display and log.
    packet_received = pyqtSignal(object)
    #: ``(raw_frame_text, reason)`` for a corrupt or unparseable frame.
    bad_frame = pyqtSignal(str, str)
    #: ``(raw_frame_text, reason)`` for a frame that passed the checksum but
    #: carries physically impossible sensor values. Kept separate from
    #: ``bad_frame`` because the two mean completely different things: one is a
    #: damaged link, the other is a misbehaving sensor on an intact link.
    rejected_frame = pyqtSignal(str, str)
    #: ``(total_frames, valid_packets, corrupt_packets, resyncs, rejected)`` —
    #: throttled to ~5 Hz.  ``resyncs`` counts frame-sync recoveries (truncated
    #: frames and oversized junk); ``rejected`` counts frames that passed the
    #: checksum but failed the physical plausibility check.
    stats_updated = pyqtSignal(int, int, int, int, int)
    #: ``(is_connected, human_readable_message)``
    connection_changed = pyqtSignal(bool, str)
    #: Free-form line for the on-screen event log.
    log_message = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # --- state shared with the GUI thread, guarded by _lock -------------
        self._lock = threading.Lock()
        self._port: Optional[str] = None
        self._baud: int = 9600
        self._want_connected = False
        self._running = True
        self._settings_dirty = False

        # --- state owned exclusively by the worker thread -------------------
        self._serial: Optional[serial.SerialBase] = None
        self._splitter = FrameSplitter()

        # --- counters (written on the worker thread, read via signals) ------
        self.total_frames = 0
        self.valid_packets = 0
        self.corrupt_packets = 0
        self.rejected_packets = 0

        self._last_stats_emit = 0.0
        self._reported_connected = False
        self._last_open_error = ""

    # ------------------------------------------------------------------
    # API called from the GUI thread
    # ------------------------------------------------------------------

    def request_connect(self, port: str, baud: int) -> None:
        """Ask the worker to open *port* at *baud* (returns immediately)."""
        with self._lock:
            self._port = port
            self._baud = int(baud)
            self._want_connected = True
            self._settings_dirty = True

    def request_disconnect(self) -> None:
        """Ask the worker to close the port and stay closed."""
        with self._lock:
            self._want_connected = False
            self._settings_dirty = True

    def stop(self) -> None:
        """Ask the run loop to exit.  Follow with ``wait()`` from the GUI."""
        with self._lock:
            self._running = False
            self._want_connected = False

    def reset_counters(self) -> None:
        """Zero the packet statistics (safe: only ints, worst case a stale read)."""
        self.total_frames = 0
        self.valid_packets = 0
        self.corrupt_packets = 0
        self.rejected_packets = 0
        self._splitter.resyncs = 0
        self._emit_stats(force=True)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 - the state machine is clearer inline
        """Main loop: connect, read, split, parse, emit — and never die."""
        next_retry_at = 0.0

        while True:
            with self._lock:
                running = self._running
                want = self._want_connected
                port = self._port
                baud = self._baud
                dirty = self._settings_dirty
                self._settings_dirty = False

            if not running:
                break

            # --- honour a settings change (port/baud swap or disconnect) ----
            if dirty:
                if self._serial is not None:
                    self._close_port("closed by request" if not want else "reopening")
                # A pending back-off must not delay an explicit user action.
                next_retry_at = 0.0

            # --- disconnected state -----------------------------------------
            if not want:
                if self._serial is not None:
                    self._close_port("closed by request")
                self._announce(False, "Disconnected")
                self.msleep(80)
                continue

            # --- (re)connect -------------------------------------------------
            if self._serial is None:
                now = time.monotonic()
                if now < next_retry_at:
                    self.msleep(50)
                    continue
                if not self._open_port(port, baud):
                    next_retry_at = time.monotonic() + RECONNECT_DELAY_S
                    continue

            # --- read ---------------------------------------------------------
            try:
                waiting = 0
                try:
                    waiting = self._serial.in_waiting
                except Exception:
                    # Some URL handlers do not implement in_waiting reliably;
                    # fall back to a single blocking-with-timeout read.
                    waiting = 0
                chunk = self._serial.read(waiting if waiting else 1)
            except (serial.SerialException, OSError, AttributeError) as exc:
                self._close_port("read failed: %s" % exc)
                self._announce(False, "Link lost — retrying…")
                self.log_message.emit("Serial read error: %s" % exc)
                next_retry_at = time.monotonic() + RECONNECT_DELAY_S
                continue
            except Exception as exc:  # pragma: no cover - defensive catch-all
                self._close_port("unexpected read error: %s" % exc)
                self.log_message.emit("Unexpected serial error: %r" % exc)
                next_retry_at = time.monotonic() + RECONNECT_DELAY_S
                continue

            if not chunk:
                self._emit_stats()
                continue

            # --- split + parse -------------------------------------------------
            try:
                frames = self._splitter.feed(chunk)
            except Exception as exc:  # pragma: no cover - defensive
                self.log_message.emit("Frame splitter fault (buffer reset): %r" % exc)
                self._splitter.reset()
                continue

            for frame in frames:
                self._handle_frame(frame)

            self._emit_stats()

        # --- shutdown ---------------------------------------------------------
        self._close_port("shutting down")
        self._emit_stats(force=True)

    # ------------------------------------------------------------------
    # Internals (worker thread only)
    # ------------------------------------------------------------------

    def _handle_frame(self, frame: str) -> None:
        """Validate one frame; every failure path is counted, never raised."""
        self.total_frames += 1
        try:
            packet = parse_frame(frame, gs_recv_epoch=time.time())
        except ChecksumError as exc:
            self.corrupt_packets += 1
            self.bad_frame.emit(frame, "checksum: %s" % exc)
            return
        except PacketParseError as exc:
            self.corrupt_packets += 1
            self.bad_frame.emit(frame, "parse: %s" % exc)
            return
        except PacketError as exc:  # pragma: no cover - future subclasses
            self.corrupt_packets += 1
            self.bad_frame.emit(frame, "packet: %s" % exc)
            return
        except Exception as exc:
            # A bug in the parser must not kill the ingestion thread mid-flight.
            self.corrupt_packets += 1
            self.bad_frame.emit(frame, "unexpected %s: %s" % (type(exc).__name__, exc))
            return

        # The checksum only proves the bytes survived the link. A sensor that
        # misreads -- a failed I2C transaction, an uninitialised register --
        # yields a frame that is intact on the wire and impossible as physics.
        # Such a packet must never reach a chart: one absurd sample drags the
        # Y autoscale to an absurd range and every later redraw with it.
        reasons = packet.implausible_reasons()
        if reasons:
            self.rejected_packets += 1
            self.rejected_frame.emit(frame, "implausible: " + "; ".join(reasons))
            return

        self.valid_packets += 1
        self.packet_received.emit(packet)

    def _open_port(self, port: Optional[str], baud: int) -> bool:
        """Try to open *port*; report success/failure through signals."""
        if not port:
            self._report_open_error("No port selected")
            return False
        try:
            # serial_for_url handles plain device names ("COM7", "/dev/ttyUSB0")
            # *and* URL forms such as "socket://127.0.0.1:5555", which is how the
            # synthetic packet generator feeds the dashboard without a radio.
            handle = serial.serial_for_url(
                port,
                baudrate=int(baud),
                timeout=READ_TIMEOUT_S,
                write_timeout=1.0,
            )
        except Exception as exc:
            self._report_open_error("Cannot open %s: %s" % (port, exc))
            return False

        try:
            handle.reset_input_buffer()
        except Exception:
            pass  # not supported by every URL handler; harmless

        self._serial = handle
        self._splitter.reset()
        self._last_open_error = ""
        self._announce(True, "Connected to %s @ %d baud" % (port, baud))
        self.log_message.emit("Opened %s at %d baud" % (port, baud))
        return True

    def _report_open_error(self, message: str) -> None:
        """Surface a failed open exactly once per distinct message.

        The reconnect loop retries every ``RECONNECT_DELAY_S`` seconds forever,
        so repeating an identical message would flood the event log while the
        radio is simply unplugged.
        """
        self._announce(False, message)
        if message != self._last_open_error:
            self._last_open_error = message
            self.log_message.emit(message)

    def _close_port(self, reason: str) -> None:
        """Close the handle if open; safe to call repeatedly."""
        if self._serial is None:
            return
        try:
            self._serial.close()
        except Exception:
            pass
        self._serial = None
        self._splitter.reset()
        self.log_message.emit("Serial port closed (%s)" % reason)

    def _announce(self, connected: bool, message: str) -> None:
        """Emit ``connection_changed`` only when the state actually changes."""
        if connected != self._reported_connected:
            self._reported_connected = connected
            self.connection_changed.emit(connected, message)

    def _emit_stats(self, force: bool = False) -> None:
        """Throttled statistics emission so bursts cannot flood the GUI."""
        now = time.monotonic()
        if not force and (now - self._last_stats_emit) < STATS_EMIT_INTERVAL_S:
            return
        self._last_stats_emit = now
        self.stats_updated.emit(
            self.total_frames, self.valid_packets, self.corrupt_packets,
            self._splitter.resyncs, self.rejected_packets,
        )

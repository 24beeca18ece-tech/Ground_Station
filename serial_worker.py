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

RECEIVE PIPELINE
----------------
::

    raw serial bytes
        -> ApiFrameUnwrapper   strips XBee 802.15.4 API framing (AP=1, 0x90),
                               validates the API checksum, passes non-API bytes
                               straight through
        -> FrameSplitter       finds $...*XX boundaries in the payload
        -> parse_frame()       validates the telemetry XOR checksum
        -> plausibility check  rejects physically impossible sensor values

Four independent failure layers, four separate counters: ``api_frame_errors``
(the radio's own framing), ``corrupt_packets`` (the telemetry XOR checksum),
``resyncs`` (ASCII frame-boundary recovery) and ``rejected_packets`` (physical
bounds).  Keeping them apart is what makes a bring-up failure diagnosable: API
errors point at the serial link to the radio, XOR errors at the RF hop, and
bounds rejections at a sensor.
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
    parse_raw_csv,
)

#: How long the blocking read waits before returning empty, in seconds.  Small
#: enough that :meth:`SerialWorker.stop` is honoured almost immediately.
READ_TIMEOUT_S = 0.05

#: Delay between reconnection attempts after a failure, in seconds.
RECONNECT_DELAY_S = 1.5

#: Minimum interval between ``stats_updated`` emissions, in seconds.  Emitting
#: on every packet would flood the GUI event loop during a burst.
STATS_EMIT_INTERVAL_S = 0.2

# --- XBee API mode (AP=1) ---------------------------------------------------
#: Start delimiter of an 802.15.4 API frame.
API_START = 0x7E
#: Frame types we care about.
API_RX_PACKET = 0x90        # Receive Packet: carries the RF payload
API_TX_STATUS = 0x8B        # Transmit Status: normal, and not ours to act on
#: 0x90 header before the payload: type(1) + 64-bit addr(8) + 16-bit addr(2)
#: + options(1).
API_RX_HEADER_LEN = 12

#: Legacy 802.15.4 receive frames. XBee 3 radios running the 802.15.4 firmware
#: emit these instead of 0x90 — which one you get depends on the sender's
#: addressing mode (ATMY/ATDL 16-bit vs. ATDH/ATDL 64-bit), not on anything the
#: ground station controls. Real hardware testing produced 0x81 exclusively, so
#: handling only 0x90 silently discarded every packet. Header layouts:
#:   0x80: type(1) + 64-bit addr(8) + RSSI(1) + options(1) = 11
#:   0x81: type(1) + 16-bit addr(2) + RSSI(1) + options(1) = 5
API_RX_64BIT = 0x80
API_RX_16BIT = 0x81
#: Payload offset for each receive frame type.
API_RX_HEADERS = {
    API_RX_PACKET: API_RX_HEADER_LEN,
    API_RX_64BIT: 11,
    API_RX_16BIT: 5,
}
#: Largest frame_data length accepted. The 802.15.4 payload maxes out near 100
#: bytes; this is generous headroom that still rejects a corrupted length field
#: instead of stalling on bytes that will never arrive.
API_MAX_FRAME_DATA = 512
#: Escape byte used by AP=2 ("API with escapes"). We run AP=1, but seeing these
#: inside frames is the signature of a radio configured the other way, and
#: saying so beats silently counting checksum failures forever.
API_ESCAPE = 0x7D

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


class ApiFrameUnwrapper:
    """Strips XBee 802.15.4 API framing, leaving the RF payload bytes.

    Sits *in front of* :class:`FrameSplitter`, so the pipeline is::

        raw bytes -> ApiFrameUnwrapper -> payload bytes -> FrameSplitter ($..*XX)

    The two layers are independent and so are their checksums. The API frame
    checksum proves the radio delivered the frame intact over the serial line;
    the telemetry XOR checksum proves the flight computer's ASCII survived the
    RF hop. A failure at one says nothing about the other, which is why they are
    counted separately.

    BACKWARD COMPATIBILITY
    ----------------------
    There is deliberately no mode flag and no mode detection. Bytes that are not
    part of a valid API frame are passed straight through, so a transparent-mode
    radio, a direct USB connection or packet_sim.py's plain-ASCII TCP stream all
    work exactly as before. ``$`` and ``*`` never appear inside an API header,
    and ``0x7E`` never appears in the ASCII telemetry alphabet, so the two
    formats cannot be confused for one another.

    RESYNC
    ------
    Every scan starts by finding the next ``0x7E``. A frame with a bad checksum
    is discarded whole and the search resumes; if its length field was the part
    that got corrupted, the next genuine delimiter re-anchors the stream. A
    ``0x7E`` that never completes into a frame is dropped once the buffer grows
    past a full frame's worth of bytes, so a spurious delimiter cannot wedge the
    reader.
    """

    __slots__ = ("buffer", "api_frames_ok", "api_frame_errors",
                 "api_frames_other", "passthrough_bytes", "escape_hints",
                 "records")

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.api_frames_ok = 0
        self.api_frame_errors = 0
        self.api_frames_other = 0
        self.passthrough_bytes = 0
        self.escape_hints = 0
        #: Payloads from the most recent :meth:`feed`, kept one entry per RF
        #: frame. ``feed`` returns them concatenated, which is what the ``$..*XX``
        #: splitter wants; raw-CSV mode instead needs the frame boundaries,
        #: because there the RF frame boundary *is* the record separator.
        self.records: List[bytes] = []

    def reset(self) -> None:
        self.buffer.clear()
        self.records = []

    @staticmethod
    def checksum(frame_data: bytes) -> int:
        """XBee API checksum: 0xFF minus the low byte of the frame_data sum."""
        return 0xFF - (sum(frame_data) & 0xFF)

    def feed(self, data: bytes) -> bytes:
        """Consume raw serial bytes, return bytes for the ASCII splitter.

        Also refreshes :attr:`records` with this call's per-frame payloads.
        """
        out = bytearray()
        self.records = []
        buf = self.buffer
        if data:
            buf.extend(data)

        while buf:
            start = buf.find(API_START)

            if start < 0:
                # No API framing in sight: it is all payload for the ASCII layer.
                out.extend(buf)
                self.passthrough_bytes += len(buf)
                buf.clear()
                break

            if start > 0:
                # Anything before the delimiter is plain stream data.
                out.extend(buf[:start])
                self.passthrough_bytes += start
                del buf[:start]
                continue

            # buf[0] is the delimiter.
            if len(buf) < 3:
                break                                   # need the length field

            length = (buf[1] << 8) | buf[2]
            if length < 1 or length > API_MAX_FRAME_DATA:
                # Implausible length: this delimiter is not a frame start.
                out.append(buf[0])
                self.passthrough_bytes += 1
                del buf[:1]
                continue

            total = 3 + length + 1                      # delim + len + data + cs
            if len(buf) < total:
                if len(buf) > 3 + API_MAX_FRAME_DATA + 1:
                    # Cannot ever complete; drop the delimiter and rescan.
                    self.api_frame_errors += 1
                    del buf[:1]
                    continue
                break                                   # wait for more bytes

            frame_data = bytes(buf[3:3 + length])
            received = buf[3 + length]

            if self.checksum(frame_data) == received:
                del buf[:total]
                self._dispatch(frame_data, out)
                continue

            # --- checksum failed -------------------------------------------
            self.api_frame_errors += 1
            if API_ESCAPE in frame_data:
                # 0x7D inside frame_data means the radio is on AP=2.
                self.escape_hints += 1

            # How much to discard matters more than it looks. A frame truncated
            # on the wire still carries a plausible length field, so the span we
            # just measured runs off the end of it and swallows whatever follows
            # -- including the next perfectly good frame. Resync at the first
            # delimiter *inside* the span if there is one, which is almost
            # certainly that next frame; only when the span is delimiter-free
            # (an intact frame with a genuinely bad checksum) is it safe to drop
            # the whole thing and keep the error count at one per frame.
            nxt = buf.find(API_START, 1, total)
            del buf[:nxt if nxt > 0 else total]

        return bytes(out)

    def _dispatch(self, frame_data: bytes, out: bytearray) -> None:
        """Route one validated API frame by type."""
        frame_type = frame_data[0]

        header_len = API_RX_HEADERS.get(frame_type)
        if header_len is not None:
            if len(frame_data) <= header_len:
                # Well-formed but carries no payload; nothing to hand on.
                self.api_frames_ok += 1
                return
            payload = frame_data[header_len:]
            out.extend(payload)
            self.records.append(payload)
            self.api_frames_ok += 1
            return

        # 0x8B Transmit Status, and anything else the radio emits (modem
        # status, AT command responses). Not errors: a receive-only ground
        # station simply has nothing to do with them.
        self.api_frames_other += 1


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
    #: ``(total_frames, valid, corrupt, resyncs, rejected, api_errors)`` —
    #: throttled to ~5 Hz.  ``resyncs`` counts ASCII frame-sync recoveries;
    #: ``rejected`` counts frames that passed the telemetry checksum but failed
    #: the physical plausibility check; ``api_errors`` counts XBee API frames
    #: whose own checksum failed, a separate layer from all of the above.
    stats_updated = pyqtSignal(int, int, int, int, int, int)
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
        #: Bench-test mode for firmware that sends bare CSV with no ``$..*XX``
        #: framing and no checksum. Off unless explicitly enabled.
        self._raw_csv_mode = False
        self._raw_csv_team_id = "TEST001"

        # --- state owned exclusively by the worker thread -------------------
        self._serial: Optional[serial.SerialBase] = None
        self._unwrapper = ApiFrameUnwrapper()
        self._splitter = FrameSplitter()

        # --- counters (written on the worker thread, read via signals) ------
        self.total_frames = 0
        self.valid_packets = 0
        self.corrupt_packets = 0
        self.rejected_packets = 0

        self._last_stats_emit = 0.0
        self._reported_connected = False
        self._escape_warned = False
        self._last_open_error = ""
        #: Epoch the raw-CSV mission clock counts from (set on first record,
        #: because that format carries no mission time of its own).
        self._raw_csv_epoch: Optional[float] = None

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

    def set_raw_csv_mode(self, enabled: bool, team_id: str = "") -> None:
        """Enable/disable bare-CSV compatibility parsing (GUI thread safe).

        Off by default. When on, each RF frame payload is parsed as one bare
        CSV record instead of being scanned for ``$...*XX`` frames. There is no
        checksum in that format, so corruption on the air link cannot be
        detected -- it is a bench-test aid, not a flight configuration.
        """
        with self._lock:
            self._raw_csv_mode = bool(enabled)
            if team_id:
                self._raw_csv_team_id = team_id
        self._raw_csv_epoch = None

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
        self._unwrapper.api_frame_errors = 0
        self._unwrapper.api_frames_ok = 0
        self._unwrapper.api_frames_other = 0
        self._escape_warned = False
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

            # --- unwrap API framing, then split + parse -------------------------
            # The unwrapper is transparent to a plain-ASCII stream, so this is a
            # no-op on a transparent-mode radio or the TCP simulator.
            try:
                payload = self._unwrapper.feed(chunk)
            except Exception as exc:  # pragma: no cover - defensive
                self.log_message.emit("API unwrapper fault (buffer reset): %r" % exc)
                self._unwrapper.reset()
                continue

            if self._unwrapper.escape_hints and not self._escape_warned:
                self._escape_warned = True
                self.log_message.emit(
                    "API frames contain 0x7D escape bytes: the radio looks like "
                    "it is set to AP=2 (API with escapes). This ground station "
                    "expects AP=1 (API without escapes)."
                )

            with self._lock:
                raw_csv = self._raw_csv_mode

            if raw_csv:
                # The RF frame boundary is the record separator in this format,
                # so records come from the unwrapper, not the ASCII splitter.
                for record in self._unwrapper.records:
                    self._handle_raw_csv(record)
                self._emit_stats()
                continue

            if not payload:
                self._emit_stats()
                continue

            try:
                frames = self._splitter.feed(payload)
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

    def _handle_raw_csv(self, record: bytes) -> None:
        """Parse one bare-CSV record (compatibility mode); never raises.

        Mirrors :meth:`_handle_frame` so the same counters, signals and
        plausibility gate apply. The one difference is that no checksum exists
        in this format, so a damaged record can only be caught by failing to
        parse or by being physically implausible.
        """
        self.total_frames += 1
        try:
            text = record.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError as exc:
            self.corrupt_packets += 1
            self.bad_frame.emit(repr(record[:80]), "non-ASCII payload: %s" % exc)
            return

        if not text:
            self.total_frames -= 1          # empty payload is not a frame
            return

        if self._raw_csv_epoch is None:
            self._raw_csv_epoch = time.time()

        with self._lock:
            team_id = self._raw_csv_team_id

        try:
            packet = parse_raw_csv(
                text, team_id,
                gs_recv_epoch=time.time(),
                mission_epoch=self._raw_csv_epoch,
            )
        except PacketError as exc:
            self.corrupt_packets += 1
            self.bad_frame.emit(text, "raw-csv: %s" % exc)
            return
        except Exception as exc:
            self.corrupt_packets += 1
            self.bad_frame.emit(
                text, "unexpected %s: %s" % (type(exc).__name__, exc))
            return

        reasons = packet.implausible_reasons()
        if reasons:
            self.rejected_packets += 1
            self.rejected_frame.emit(text, "implausible: " + "; ".join(reasons))
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
        self._unwrapper.reset()
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
        self._unwrapper.reset()
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
            self._unwrapper.api_frame_errors,
        )

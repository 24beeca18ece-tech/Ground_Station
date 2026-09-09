"""
packet_merger.py
================

Reassembles one telemetry stream from two radios that each receive only part of
it.

WHY THIS EXISTS
---------------
The flight transmitter alternates destinations per packet: one packet to RX1,
the next to RX2, and so on.  Neither ground radio sees the whole stream — each
gets roughly every other packet — so this is *not* a redundancy setup and there
is nothing to de-duplicate.  The two partial streams have to be interleaved back
into the original order before anything is charted or logged.

Ordering is by ``packet_count``, the flight computer's own monotonically
increasing counter, because that is the only field that states the transmitter's
intended order.  Arrival order cannot be trusted: the two radios have
independent USB and serial latency, so the packet that was sent first is not
necessarily the one that arrives first.

DELIBERATELY NOT ASSUMED
------------------------
Nothing here assumes the alternation is strictly odd/even, or that RX1 gets the
odd ones.  The merge works for any interleaving — including three radios, or an
uneven split — because it only ever asks "what is the next ``packet_count``?".
:meth:`PacketMerger.alternation_report` describes the pattern actually observed
so the real firmware behaviour can be confirmed from live data rather than
guessed at.

HOW ORDER IS RECOVERED
----------------------
Arrivals go into a small buffer keyed by ``packet_count``.  A packet is released
as soon as it is the next one expected.  When the next one is missing, it is
given up on once :data:`MIN_GAP_DEPTH` *later* packets are in hand — evidence
that it is not simply still in transit — with :data:`HOLD_S` as a backstop for
when nothing later arrives either.  Either way, a radio going quiet can never
stall the display.

Depth is the primary rule on purpose.  ``add`` and ``drain`` both run on the GUI
thread, so a rendering stall delays arrivals and the give-up decision equally; a
pure timer would measure GUI load as much as radio latency and invent gaps that
never happened.

Both numbers are a trade-off: too small and a slightly late packet is declared
missing, too large and the display lags behind the link.  ``late_drops`` counts
every packet that arrived after its slot had already been given up on, so it is
the number to watch — if it is not near zero on a healthy link,
:data:`MIN_GAP_DEPTH` is too small for the latency spread between the radios.

GAPS ARE REAL
-------------
A ``packet_count`` that never arrives is simply never emitted.  Nothing is
interpolated, back-filled or invented: a radio dropping out has to look like
missing data, because that is what it is.

THREADING
---------
Pure Python, no Qt, no I/O.  It is driven from the GUI thread in the
application, but nothing here depends on that, which is what makes the
behaviour exhaustively testable without a radio or an event loop.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

#: How many packets *after* a missing one must be in hand before it is given up
#: on.  This is the primary rule, and deliberately not a timer: add() and drain()
#: both run on the GUI thread, so a rendering stall delays arrivals and the
#: decision by the same amount, and a packet merely waiting in Qt's queue is not
#: mistaken for a lost one.  Three is comfortably more than the one-packet skew
#: an alternating pair produces, and is reached within ~150 ms at 20 Hz once a
#: radio genuinely stops.
MIN_GAP_DEPTH = 3

#: Backstop for when nothing later arrives either -- both radios quiet, so depth
#: can never be reached.  Only this path is time-triggered, which is why it is
#: generous: it exists to stop the buffer sitting forever, not to police jitter.
HOLD_S = 1.0

#: Hard cap on buffered packets.  Reached only if a counter jumps wildly (a
#: corrupt count that survived the checksum), and exists so a bad value can
#: never grow memory without bound.
MAX_BUFFER = 256

#: How many recently released counts to remember, for spotting a duplicate that
#: arrives after its slot was already emitted.
RECENT_MEMORY = 512

#: A ``packet_count`` this far below what is expected is read as the flight
#: computer restarting, not as a very late packet.
RESET_BACKWARD = 64


class PacketMerger:
    """Interleaves packets from several sources into one ordered stream.

    Usage is two calls: :meth:`add` as packets arrive from any source, and
    :meth:`drain` on a timer to collect whatever is now safe to emit.

    Parameters
    ----------
    hold_s:
        Seconds to wait for a missing ``packet_count`` before giving up on it.
    on_warning:
        Called with a human-readable string for conditions worth surfacing --
        duplicates, gaps, counter resets.  Optional; counters are kept
        regardless.
    """

    def __init__(self, hold_s: float = HOLD_S,
                 on_warning: Optional[Callable[[str], None]] = None) -> None:
        self.hold_s = float(hold_s)
        self._on_warning = on_warning

        #: packet_count -> (packet, source, arrival_monotonic)
        self._buffer: Dict[int, Tuple[Any, str, float]] = {}
        #: Next packet_count expected; None until the first packet arrives.
        self._next: Optional[int] = None
        #: Recently released counts, for late-duplicate detection.
        self._recent: List[int] = []
        self._recent_set = set()

        # --- counters, all read-only from outside -------------------------
        self.released = 0
        self.duplicates = 0
        self.late_drops = 0
        self.missing = 0          # packet_count values given up on
        self.gaps = 0             # runs of consecutive missing counts
        self.resets = 0
        self.buffered_peak = 0
        #: source -> packets released from that source.
        self.per_source: Dict[str, int] = {}
        #: Last few (packet_count, source) pairs, for reporting the observed
        #: alternation pattern rather than assuming one.
        self._pattern: List[Tuple[int, str]] = []

    # ------------------------------------------------------------------
    # Intake
    # ------------------------------------------------------------------

    def add(self, packet: Any, source: str, now: Optional[float] = None) -> None:
        """Take one packet from *source*.  Never raises, never blocks."""
        if now is None:
            now = time.monotonic()

        try:
            count = int(packet.packet_count)
        except (AttributeError, TypeError, ValueError):
            # A packet with no usable counter cannot be ordered. Dropping it is
            # the honest option: placing it arbitrarily would corrupt the very
            # ordering this class exists to guarantee.
            self.late_drops += 1
            self._warn("packet from %s has no usable packet_count; dropped" % source)
            return

        if self._next is None:
            # The sequence has no baseline yet. Buffer without judging the
            # count: which packet is "first" is decided in drain(), after the
            # reordering window, so a skewed radio cannot set it too high.
            if count not in self._buffer:
                self._buffer[count] = (packet, source, now)
                self.buffered_peak = max(self.buffered_peak, len(self._buffer))
            else:
                self.duplicates += 1
                self._warn("duplicate packet_count %d before the sequence "
                           "started; keeping the first" % count)
            return

        if count < self._next - RESET_BACKWARD:
            # A large backward jump is a flight-computer restart, not a late
            # packet. Release what is held, then rebase on the new sequence.
            self.resets += 1
            self._warn(
                "packet_count jumped backwards (%d -> %d): flight computer "
                "reset? Sequence rebased." % (self._next, count)
            )
            self._flush_all()
            self._next = count

        if count < self._next:
            # Its slot was already emitted or given up on. Emitting it now
            # would break the ordering guarantee, so it is dropped and counted.
            self.late_drops += 1
            self._warn(
                "packet %d from %s arrived after its slot was released "
                "(expecting %d); dropped" % (count, source, self._next)
            )
            return

        if count in self._buffer or count in self._recent_set:
            # Both radios received the same packet. With alternating firmware
            # this should not happen; keep the first and say so rather than
            # silently overwriting or counting it twice.
            self.duplicates += 1
            first = self._buffer.get(count)
            self._warn(
                "duplicate packet_count %d (already had it from %s, now also "
                "from %s); keeping the first"
                % (count, first[1] if first else "an earlier release", source)
            )
            return

        self._buffer[count] = (packet, source, now)
        self.buffered_peak = max(self.buffered_peak, len(self._buffer))

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def drain(self, now: Optional[float] = None) -> List[Tuple[Any, str]]:
        """Return every packet now safe to emit, in ``packet_count`` order."""
        if now is None:
            now = time.monotonic()
        out: List[Tuple[Any, str]] = []

        if self._next is None:
            # Start the sequence at the lowest count seen, but only once the
            # window has passed, so a packet still in flight from the other
            # radio still gets to be the first one.
            if not self._buffer:
                return out
            oldest = min(self._buffer, key=lambda c: self._buffer[c][2])
            settled = (len(self._buffer) >= MIN_GAP_DEPTH
                       or now - self._buffer[oldest][2] >= self.hold_s
                       or len(self._buffer) > MAX_BUFFER)
            if not settled:
                return out
            self._next = min(self._buffer)

        while self._buffer:
            # The straightforward case: the next one we want is here.
            if self._next in self._buffer:
                out.append(self._pop(self._next))
                self._next += 1
                continue

            oldest_count = min(self._buffer)
            waited = now - self._buffer[oldest_count][2]

            # Give up only when later packets prove the missing one is not
            # merely in transit: enough of them are in hand (the reliable
            # signal), or nothing later arrived for a long time either (the
            # backstop), or the buffer is at its cap.
            deep_enough = len(self._buffer) >= MIN_GAP_DEPTH
            timed_out = waited >= self.hold_s
            if not (deep_enough or timed_out or len(self._buffer) > MAX_BUFFER):
                break

            # Give up: everything between here and the oldest buffered packet
            # is not coming. This is the moment a gap becomes real.
            lost = oldest_count - self._next
            if lost > 0:
                self.missing += lost
                self.gaps += 1
                self._warn(
                    "packet_count %s never arrived (%d packet%s); charts will "
                    "show a gap"
                    % (self._describe_range(self._next, oldest_count - 1),
                       lost, "" if lost == 1 else "s")
                )
            self._next = oldest_count

        return out

    def flush(self) -> List[Tuple[Any, str]]:
        """Release everything held, ordered.  For shutdown or disconnect."""
        return self._flush_all()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def alternation_report(self) -> str:
        """Describe the interleaving actually observed.

        Written to be read, not parsed: the point is to let an operator confirm
        the real firmware's pattern from live data instead of trusting an
        assumption baked into this code.
        """
        if not self._pattern:
            return "no packets yet"
        by_source: Dict[str, List[int]] = {}
        for count, source in self._pattern:
            by_source.setdefault(source, []).append(count)

        bits = []
        for source in sorted(by_source):
            counts = by_source[source]
            parities = {c % 2 for c in counts}
            if parities == {0}:
                shape = "even only"
            elif parities == {1}:
                shape = "odd only"
            else:
                shape = "mixed parity"
            bits.append("%s: %d packets, %s" % (source, len(counts), shape))

        seq = "".join(s[-1] for _, s in self._pattern[-24:])
        return "; ".join(bits) + "  |  recent order: " + seq

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    @property
    def next_expected(self) -> Optional[int]:
        return self._next

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pop(self, count: int) -> Tuple[Any, str]:
        packet, source, _ = self._buffer.pop(count)
        self.released += 1
        self.per_source[source] = self.per_source.get(source, 0) + 1
        self._remember(count)
        self._pattern.append((count, source))
        if len(self._pattern) > 64:
            del self._pattern[:-64]
        return packet, source

    def _flush_all(self) -> List[Tuple[Any, str]]:
        counts = sorted(self._buffer)
        out = [self._pop(c) for c in counts]
        if counts:
            self._next = counts[-1] + 1
        return out

    def _remember(self, count: int) -> None:
        self._recent.append(count)
        self._recent_set.add(count)
        if len(self._recent) > RECENT_MEMORY:
            self._recent_set.discard(self._recent.pop(0))

    def _warn(self, text: str) -> None:
        if self._on_warning is not None:
            try:
                self._on_warning(text)
            except Exception:
                # A logging callback must never be able to break ingestion.
                pass

    @staticmethod
    def _describe_range(first: int, last: int) -> str:
        return str(first) if first == last else "%d-%d" % (first, last)

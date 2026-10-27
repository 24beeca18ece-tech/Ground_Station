"""
tools/test_packet_merger.py
===========================

The merger across a flight-computer reboot, and the behaviour the reboot
fix must not disturb: duplicates from both radios, the routine 1-in-3 LoRa
gaps, and late or out-of-order packets that are NOT a reboot.

Time is passed explicitly (``now``), so nothing here sleeps.

    python -m pytest tools/test_packet_merger.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import packet_merger as pm  # noqa: E402
from packet_merger import PacketMerger  # noqa: E402


class P:
    def __init__(self, count, run="a"):
        self.packet_count = count
        self.run = run                       # which flight-computer boot


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self, dt=0.0):
        self.t += dt
        return self.t


def feed(merger, clock, counts, run="a", source=None, dt=0.15):
    """Add *counts* one by one, draining after each; return what came out."""
    out = []
    for count in counts:
        src = source or ("RX1" if count % 2 else "RX2")
        merger.add(P(count, run), src, now=clock(dt))
        out += merger.drain(now=clock())
    return out


def settle(merger, clock):
    """Let the hold time pass with nothing arriving; return what comes out."""
    return merger.drain(now=clock(pm.HOLD_S + 0.5))


# ---------------------------------------------------------------------------
# The reboot
# ---------------------------------------------------------------------------

def test_a_reboot_does_not_drop_the_new_run():
    warnings = []
    merger = PacketMerger(on_warning=warnings.append)
    clock = Clock()
    first = feed(merger, clock, range(1, 201), run="a") + settle(merger, clock)
    assert [p.packet_count for p, _ in first] == list(range(1, 201))

    second = feed(merger, clock, range(1, 201), run="b") + settle(merger, clock)
    new_run = [p.packet_count for p, _ in second if p.run == "b"]
    print("new run: %d of 200 released" % len(new_run))
    assert merger.resets == 1
    assert any("jumped backwards (201 -> 1)" in w for w in warnings)
    assert new_run == list(range(1, 201))
    assert merger.duplicates == 0


def test_a_second_reboot_is_handled_the_same_way():
    merger = PacketMerger()
    clock = Clock()
    for run in "abc":
        out = feed(merger, clock, range(1, 151), run=run) + settle(merger, clock)
        assert [p.packet_count for p, _ in out if p.run == run] == list(range(1, 151))
    assert merger.resets == 2 and merger.duplicates == 0


def test_a_duplicate_straight_after_a_reboot_is_still_caught():
    """Clearing the memory on a rebase must not let the new run's own
    duplicates through."""
    merger = PacketMerger()
    clock = Clock()
    feed(merger, clock, range(1, 201), run="a")
    settle(merger, clock)
    out = feed(merger, clock, [1, 2, 3, 4], run="b")
    merger.add(P(2, "b"), "RX1", now=clock(0.05))        # RX1 copy of 2
    out += feed(merger, clock, [5, 6], run="b") + settle(merger, clock)
    assert [p.packet_count for p, _ in out if p.run == "b"] == [1, 2, 3, 4, 5, 6]
    # 2 was already released, so its copy is caught as late (count below the
    # next expected) before the duplicate check -- dropped once either way,
    # and not mistaken for a second reboot.
    assert merger.duplicates + merger.late_drops == 1
    assert merger.resets == 1


# ---------------------------------------------------------------------------
# What must keep working
# ---------------------------------------------------------------------------

def test_a_real_duplicate_is_dropped_once_while_still_buffered():
    merger = PacketMerger()
    clock = Clock()
    feed(merger, clock, [1, 2, 3])
    # 5 arrives from RX1, then the same 5 from RX2 while 4 is still awaited.
    merger.add(P(5), "RX1", now=clock(0.1))
    merger.add(P(5), "RX2", now=clock(0.05))
    out = feed(merger, clock, [4, 6, 7]) + settle(merger, clock)
    counts = [p.packet_count for p, _ in out]
    assert counts.count(5) == 1
    assert counts == [4, 5, 6, 7]
    assert merger.duplicates == 1


def test_a_real_duplicate_is_dropped_once_after_its_release():
    merger = PacketMerger()
    clock = Clock()
    out = feed(merger, clock, range(1, 11))
    merger.add(P(8), "RX2", now=clock(0.05))            # already emitted
    out += feed(merger, clock, [11, 12]) + settle(merger, clock)
    counts = [p.packet_count for p, _ in out]
    assert counts == list(range(1, 13))
    # Dropped as either a duplicate or late -- once, and not emitted twice.
    assert merger.duplicates + merger.late_drops == 1
    assert merger.resets == 0


def test_simo_pattern_raises_no_false_gap_alarms():
    """Counts 1 and 2 of every three reach RX1 and RX2; 3 goes to LoRa. The
    merger reports each LoRa count as a one-count gap, and the dashboard's
    rule must call every one of them routine -- no alarm."""
    from dashboard_ui import Dashboard
    gaps = []
    merger = PacketMerger(on_gap=lambda f, l, n: gaps.append((f, l, n)))
    clock = Clock()
    sent = [c for c in range(1, 301) if c % 3 != 0]
    out = []
    for c in sent:
        merger.add(P(c), "RX1" if c % 3 == 1 else "RX2", now=clock(0.3))
        out += merger.drain(now=clock())
    out += settle(merger, clock)
    assert [p.packet_count for p, _ in out] == sent
    assert gaps, "the LoRa counts should be reported as gaps"
    assert all(Dashboard._is_routine_gap(f, l, n) for f, l, n in gaps), \
        [g for g in gaps if not Dashboard._is_routine_gap(*g)]
    assert merger.resets == 0 and merger.duplicates == 0
    assert merger.late_drops == 0


def test_a_late_packet_is_not_a_reboot():
    """98 given up on (99, 100, 101 in hand), then 98 turns up: dropped as
    late, sequence continues, no reset."""
    warnings = []
    merger = PacketMerger(on_warning=warnings.append)
    clock = Clock()
    out = feed(merger, clock, range(1, 98))
    out += feed(merger, clock, [99, 100, 101])
    assert merger.next_expected == 102                  # 98 given up on
    merger.add(P(98), "RX2", now=clock(0.1))
    out += feed(merger, clock, [102, 103]) + settle(merger, clock)
    counts = [p.packet_count for p, _ in out]
    assert 98 not in counts and counts[-3:] == [101, 102, 103]
    assert merger.resets == 0
    assert merger.late_drops == 1
    assert not any("jumped backwards" in w for w in warnings)


def test_out_of_order_inside_the_window_is_reordered():
    merger = PacketMerger()
    clock = Clock()
    out = feed(merger, clock, range(1, 98))
    out += feed(merger, clock, [100, 99, 98, 101]) + settle(merger, clock)
    assert [p.packet_count for p, _ in out] == list(range(1, 102))
    assert merger.resets == 0 and merger.late_drops == 0 and merger.missing == 0


@pytest.mark.parametrize("back,is_reset", [
    (pm.RESET_BACKWARD, False),        # exactly the threshold: a late packet
    (pm.RESET_BACKWARD + 1, True),     # one more: a reboot
])
def test_the_reboot_threshold(back, is_reset):
    merger = PacketMerger()
    clock = Clock()
    feed(merger, clock, range(1, 301))
    settle(merger, clock)
    assert merger.next_expected == 301
    merger.add(P(301 - back, "b"), "RX1", now=clock(0.1))
    assert (merger.resets == 1) is is_reset


# ---------------------------------------------------------------------------
# Known gaps around a reboot, NOT fixed here (reported). strict=True makes
# these fail loudly the day they are fixed, so the markers get removed.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "add() discards what _flush_all() returns on a rebase: a packet still "
    "held when the reboot is seen is counted as released but never handed "
    "to the caller"))
def test_a_packet_held_at_the_reboot_still_comes_out():
    merger = PacketMerger()
    clock = Clock()
    out = feed(merger, clock, range(1, 99), run="a")
    merger.add(P(100, "a"), "RX2", now=clock(0.15))     # 99 lost: 100 waits
    out += merger.drain(now=clock())
    assert merger.buffered == 1
    out += feed(merger, clock, range(1, 6), run="b") + settle(merger, clock)
    assert 100 in [p.packet_count for p, _ in out if p.run == "a"]


@pytest.mark.xfail(strict=True, reason=(
    "a reboot is only recognised when the count falls by more than "
    "RESET_BACKWARD (64); a reboot before the old run passed 64 is read as "
    "late packets and the new run is dropped until it catches up"))
def test_a_reboot_early_in_the_flight_is_recognised():
    merger = PacketMerger()
    clock = Clock()
    feed(merger, clock, range(1, 51), run="a")
    settle(merger, clock)
    out = feed(merger, clock, range(1, 21), run="b") + settle(merger, clock)
    assert [p.packet_count for p, _ in out if p.run == "b"] == list(range(1, 21))

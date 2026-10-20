"""
tools/test_cansat_v2_frames.py
==============================

Tests for the CanSat v2 frames the current flight firmware emits: the standard
26-field ``$...*XX`` format, with RAW CSV mode OFF, on an airframe that has no
battery sensor and no actuators fitted yet.

Run from the repository root::

    python -m pytest tools/test_cansat_v2_frames.py -v

What these pin down, and why each one is worth a test:

* The two real frames parse, pass their XOR checksum, and are not thrown out
  by the plausibility envelope. A frame full of ``nan`` and blanks is what a
  partly-populated vehicle legitimately sends; rejecting it would lose the
  telemetry that *is* there.
* ``has_voltage`` distinguishes "no battery sensor" from "flat battery".
  Without it a missing sensor renders as 0.00 V and trips a low-battery alert
  on a vehicle that is fine.
* ``has_actuator_data`` likewise distinguishes "no reaction wheel fitted"
  from "wheel stopped at 0 rpm in stage STOWED" -- the dataclass defaults.
* A 150-182 byte frame does not fit one XBee RF payload and arrives split
  across two API frames. The joining logic (ApiFrameUnwrapper feeding
  FrameSplitter) is NOT changed by this work, so these tests exist to prove
  it already handles the split, and that losing one chunk costs exactly one
  packet and does not desynchronise the stream that follows.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serial_worker import (  # noqa: E402
    API_RX_16BIT, API_START, ApiFrameUnwrapper, FrameSplitter,
)
from telemetry_packet import (  # noqa: E402
    CanSatPacket, PAYLOAD_CANSAT, ChecksumError, build_frame, compute_checksum,
    parse_frame,
)

# --- the two frames exactly as the firmware sends them ----------------------

#: Pad frame: no GPS fix, no particulates yet, battery reported as "nan".
FRAME_PAD = ("$2025ASI001,CANSAT,1234.56,12345,nan,nan,nan,nan,,,,,0,,,,,,,1,"
             ",,,,,*50")

#: In-flight frame: GPS locked, SPS30 reporting, still no battery, still no
#: actuators (the last two fields are blank).
FRAME_FLIGHT = ("$2025ASI001,CANSAT,65.43,218,235.94,985.23,27.40,nan,"
                "09:05:07,32.719876,74.861234,312.4,11,0.12,-0.05,9.79,"
                "0.40,-1.20,0.06,1,4.10,6.20,7.00,7.40,,*01")

#: Body of a flight frame with every optional field under test substitutable.
_TEMPLATE = ("2025ASI001,CANSAT,65.43,218,235.94,985.23,27.40,{volt},"
             "09:05:07,32.719876,74.861234,312.4,11,0.12,-0.05,9.79,"
             "0.40,-1.20,0.06,1,4.10,6.20,7.00,7.40,{wheel},{stage}")


def frame_with(volt="nan", wheel="", stage=""):
    """A correctly-checksummed frame with those three fields substituted."""
    return build_frame(_TEMPLATE.format(volt=volt, wheel=wheel, stage=stage))


# ---------------------------------------------------------------------------
# The two real frames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frame", [FRAME_PAD, FRAME_FLIGHT])
def test_frame_checksum_is_correct_as_transmitted(frame):
    body = frame[1:frame.rindex("*")]
    assert compute_checksum(body) == int(frame[-2:], 16)


@pytest.mark.parametrize("frame", [FRAME_PAD, FRAME_FLIGHT])
def test_frame_parses_as_cansat_without_error(frame):
    packet = parse_frame(frame)
    assert isinstance(packet, CanSatPacket)
    assert packet.payload_type == PAYLOAD_CANSAT
    assert packet.team_id == "2025ASI001"
    assert packet.checksum_valid is True


@pytest.mark.parametrize("frame", [FRAME_PAD, FRAME_FLIGHT])
def test_frame_is_not_rejected_by_the_plausibility_envelope(frame):
    packet = parse_frame(frame)
    assert packet.is_plausible, packet.implausible_reasons()


def test_frame_has_the_26_field_v2_shape():
    for frame in (FRAME_PAD, FRAME_FLIGHT):
        body = frame[1:frame.rindex("*")]
        assert len(body.split(",")) == 26


def test_pad_frame_reports_no_fix_and_no_particulates():
    packet = parse_frame(FRAME_PAD)
    assert packet.has_fix is False
    assert all(math.isnan(v) for v in
               (packet.pm1_0, packet.pm2_5, packet.pm4_0, packet.pm10))
    assert packet.fsm_state == 1               # TEST_MODE


def test_flight_frame_values_land_in_their_named_fields():
    packet = parse_frame(FRAME_FLIGHT)
    assert packet.packet_count == 218
    assert packet.altitude_m == pytest.approx(235.94)
    assert packet.pressure_hpa == pytest.approx(985.23)     # hPa
    assert packet.temp_c == pytest.approx(27.40)
    assert packet.nav_time == "09:05:07"
    assert packet.lat == pytest.approx(32.719876)
    assert packet.lon == pytest.approx(74.861234)
    assert packet.nav_alt_m == pytest.approx(312.4)
    assert packet.sats == 11
    assert packet.has_fix is True
    # ACC is already m/s^2 on this wire format: no g conversion here, unlike
    # the raw-CSV path. 9.79 is gravity, not 9.79 g.
    assert packet.acc_z == pytest.approx(9.79)
    assert packet.gyro_y == pytest.approx(-1.20)            # deg/s
    assert packet.fsm_state == 1
    assert packet.fsm_name == "TEST_MODE"
    assert (packet.pm1_0, packet.pm2_5, packet.pm4_0, packet.pm10) == (
        pytest.approx(4.10), pytest.approx(6.20),
        pytest.approx(7.00), pytest.approx(7.40))


# ---------------------------------------------------------------------------
# VOLTAGE is optional
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["nan", "NaN", "", "   ", "n/a", "-"])
def test_absent_voltage_gives_nan_and_has_voltage_false(raw):
    packet = parse_frame(frame_with(volt=raw))
    assert math.isnan(packet.voltage_v)
    assert packet.has_voltage is False


@pytest.mark.parametrize("raw,expected", [("7.85", 7.85), ("0", 0.0),
                                          ("11.1", 11.1)])
def test_real_voltage_keeps_has_voltage_true(raw, expected):
    packet = parse_frame(frame_with(volt=raw))
    assert packet.voltage_v == pytest.approx(expected)
    assert packet.has_voltage is True


def test_absent_voltage_does_not_make_the_packet_implausible():
    """A missing sensor is not an out-of-range reading."""
    packet = parse_frame(frame_with(volt="nan"))
    assert packet.is_plausible
    assert not any("VOLTAGE" in r for r in packet.implausible_reasons())


def test_a_real_out_of_range_voltage_is_still_caught():
    """Making the field optional must not disable its bounds check."""
    packet = parse_frame(frame_with(volt="99.0"))
    assert packet.has_voltage is True
    assert any("VOLTAGE" in r for r in packet.implausible_reasons())


@pytest.mark.parametrize("field,name", [(2, "ALTITUDE"), (3, "PRESSURE"),
                                        (4, "TEMP"), (1, "PACKET_COUNT")])
def test_other_mandatory_fields_are_still_mandatory(field, name):
    """Only VOLTAGE became optional; blanking any other one must still fail."""
    cells = _TEMPLATE.format(volt="7.85", wheel="", stage="").split(",")
    cells[2 + field] = ""
    with pytest.raises(Exception):
        parse_frame(build_frame(",".join(cells)))


# ---------------------------------------------------------------------------
# Actuator fields
# ---------------------------------------------------------------------------

def test_both_actuator_fields_blank_sets_has_actuator_data_false():
    packet = parse_frame(FRAME_FLIGHT)
    assert packet.has_actuator_data is False
    # The values stay at their defaults; the flag is what the display reads.
    assert packet.reaction_wheel_rpm == 0
    assert packet.recovery_stage == 0


@pytest.mark.parametrize("wheel,stage", [("250", ""), ("", "1"), ("250", "1"),
                                         ("0", "0")])
def test_either_actuator_field_present_keeps_has_actuator_data_true(wheel, stage):
    packet = parse_frame(frame_with(wheel=wheel, stage=stage))
    assert packet.has_actuator_data is True


def test_actuator_values_are_read_when_present():
    packet = parse_frame(frame_with(wheel="-480", stage="2"))
    assert packet.reaction_wheel_rpm == -480
    assert packet.recovery_stage == 2


# ---------------------------------------------------------------------------
# One frame split across two XBee API frames
# ---------------------------------------------------------------------------

def wrap_api_rx_16bit(payload: bytes, rssi: int = 0x2D) -> bytes:
    """Wrap *payload* in a 0x81 RX frame (16-bit address), as AP=1 emits.

    frame_data = type(1) + source addr(2) + RSSI(1) + options(1) + payload,
    which is the 5-byte header :data:`serial_worker.API_RX_HEADERS` declares.
    """
    frame_data = bytes((API_RX_16BIT, 0x12, 0x34, rssi, 0x00)) + payload
    checksum = 0xFF - (sum(frame_data) & 0xFF)
    length = len(frame_data)
    return (bytes((API_START, (length >> 8) & 0xFF, length & 0xFF))
            + frame_data + bytes((checksum,)))


def feed(unwrapper, splitter, chunk):
    """Push one serial read through the real two-stage pipeline."""
    return splitter.feed(unwrapper.feed(chunk))


def split_frame_bytes(frame: str, head: int = 100):
    """The frame as two RF payloads: *head* bytes, then the remainder."""
    data = frame.encode("ascii")
    assert len(data) > head, "frame is too short to split meaningfully"
    return data[:head], data[head:]


def test_frame_split_across_two_api_frames_yields_exactly_one_packet():
    unwrapper, splitter = ApiFrameUnwrapper(), FrameSplitter()
    head, tail = split_frame_bytes(FRAME_FLIGHT, 100)

    first = feed(unwrapper, splitter, wrap_api_rx_16bit(head))
    assert first == [], "half a frame must not be emitted"

    second = feed(unwrapper, splitter, wrap_api_rx_16bit(tail))
    assert len(second) == 1
    packet = parse_frame(second[0])
    assert isinstance(packet, CanSatPacket)
    assert packet.packet_count == 218
    assert packet.has_voltage is False
    assert packet.has_actuator_data is False


@pytest.mark.parametrize("head", [1, 50, 100, 140, 148])
def test_the_split_point_does_not_matter(head):
    unwrapper, splitter = ApiFrameUnwrapper(), FrameSplitter()
    data = FRAME_FLIGHT.encode("ascii")
    frames = feed(unwrapper, splitter, wrap_api_rx_16bit(data[:head]))
    frames += feed(unwrapper, splitter, wrap_api_rx_16bit(data[head:]))
    assert len(frames) == 1
    assert parse_frame(frames[0]).packet_count == 218


def test_dropping_the_second_chunk_loses_only_that_packet():
    """A lost tail must not desynchronise the frames that follow it."""
    unwrapper, splitter = ApiFrameUnwrapper(), FrameSplitter()
    head, _tail = split_frame_bytes(FRAME_FLIGHT, 100)

    # The tail never arrives: the RF hop dropped it.
    assert feed(unwrapper, splitter, wrap_api_rx_16bit(head)) == []

    # The next whole frame, split the same way, still parses. Its leading '$'
    # is what re-anchors the splitter over the orphaned head.
    next_frame = frame_with(volt="nan")
    n_head, n_tail = split_frame_bytes(next_frame, 100)
    out = feed(unwrapper, splitter, wrap_api_rx_16bit(n_head))
    out += feed(unwrapper, splitter, wrap_api_rx_16bit(n_tail))

    assert len(out) == 1, "exactly one packet: the orphan is gone, this one is not"
    packet = parse_frame(out[0])
    assert packet.packet_count == 218
    assert packet.checksum_valid is True


def test_a_corrupted_chunk_is_rejected_and_the_next_frame_survives():
    """The orphaned half must not be glued onto the following frame."""
    unwrapper, splitter = ApiFrameUnwrapper(), FrameSplitter()
    head, tail = split_frame_bytes(FRAME_FLIGHT, 100)
    corrupt = tail[:5] + b"X" + tail[6:]        # one byte flipped in the body

    feed(unwrapper, splitter, wrap_api_rx_16bit(head))
    damaged = feed(unwrapper, splitter, wrap_api_rx_16bit(corrupt))
    assert len(damaged) == 1
    with pytest.raises(ChecksumError):
        parse_frame(damaged[0])                 # caught by the XOR checksum

    good = feed(unwrapper, splitter, wrap_api_rx_16bit(FRAME_PAD.encode()))
    assert len(good) == 1
    assert parse_frame(good[0]).packet_count == 12345


def test_both_frames_back_to_back_in_one_read():
    """Two frames in a single serial read come out as two, in order."""
    unwrapper, splitter = ApiFrameUnwrapper(), FrameSplitter()
    data = (wrap_api_rx_16bit(FRAME_PAD.encode())
            + wrap_api_rx_16bit(FRAME_FLIGHT.encode()))
    frames = feed(unwrapper, splitter, data)
    assert [parse_frame(f).packet_count for f in frames] == [12345, 218]


def test_frame_is_longer_than_one_rf_payload_in_practice():
    """Why the split case is the normal case, not an edge case."""
    assert len(FRAME_FLIGHT) >= 140

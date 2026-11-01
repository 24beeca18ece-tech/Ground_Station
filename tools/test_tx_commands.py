"""
test_tx_commands.py
===================
Unit tests for the ROCKET TX START / TX STOP uplink commands.

Verifies that build_tx_request(b"A", 1) and build_tx_request(b"B", 1) produce
byte-exact expected frames according to the XBee 802.15.4 API (AP=1, 0x10
Transmit Request) spec:

  7E                  start delimiter
  00 0F               length (15 bytes of frame_data)
  10                  frame type: Transmit Request
  01                  frame_id
  00 00 00 00 00 00 FF FF   64-bit broadcast address
  FF FE               16-bit "unknown / broadcast"
  00                  broadcast radius
  00                  options
  41 or 42            RF payload ('A' or 'B')
  XX                  checksum = 0xFF - (sum(frame_data) & 0xFF)

Run with:
    python tools/test_tx_commands.py
or, if pytest.ini is present:
    pytest tools/test_tx_commands.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from serial_worker import SerialWorker, ApiFrameUnwrapper


def _expected_frame(payload_byte: int) -> bytes:
    """Build the expected 0x10 Transmit Request frame by hand."""
    frame_data = bytes([
        0x10,           # frame type
        0x01,           # frame_id
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF,  # addr64 broadcast
        0xFF, 0xFE,     # addr16 unknown
        0x00,           # broadcast radius
        0x00,           # options
        payload_byte,   # RF payload
    ])
    length = len(frame_data)
    checksum = ApiFrameUnwrapper.checksum(frame_data)
    return bytes([0x7E, (length >> 8) & 0xFF, length & 0xFF]) + frame_data + bytes([checksum])


def test_tx_start_frame():
    """build_tx_request(b'A', 1) must be byte-exact."""
    result = SerialWorker.build_tx_request(b"A", frame_id=1)
    expected = _expected_frame(ord("A"))

    assert result == expected, (
        "TX START frame mismatch\n"
        "  got:      %s\n"
        "  expected: %s" % (result.hex(" "), expected.hex(" "))
    )
    # Spot-check the structure
    assert result[0] == 0x7E,       "start delimiter"
    assert result[1:3] == b"\x00\x0F", "length = 15"
    assert result[3] == 0x10,       "frame type 0x10"
    assert result[4] == 0x01,       "frame_id = 1"
    assert result[5:13] == bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF]), "broadcast addr64"
    assert result[13:15] == b"\xFF\xFE", "addr16 unknown"
    assert result[15] == 0x00,      "broadcast radius = 0"
    assert result[16] == 0x00,      "options = 0"
    assert result[17] == ord("A"),  "payload = 'A'"
    print("PASS  TX START: %s" % result.hex(" "))


def test_tx_stop_frame():
    """build_tx_request(b'B', 1) must be byte-exact."""
    result = SerialWorker.build_tx_request(b"B", frame_id=1)
    expected = _expected_frame(ord("B"))

    assert result == expected, (
        "TX STOP frame mismatch\n"
        "  got:      %s\n"
        "  expected: %s" % (result.hex(" "), expected.hex(" "))
    )
    assert result[0] == 0x7E,       "start delimiter"
    assert result[1:3] == b"\x00\x0F", "length = 15"
    assert result[3] == 0x10,       "frame type 0x10"
    assert result[4] == 0x01,       "frame_id = 1"
    assert result[17] == ord("B"),  "payload = 'B'"
    print("PASS  TX STOP:  %s" % result.hex(" "))


def test_payload_is_single_byte():
    """Both commands are exactly 1 byte — the firmware ignores anything else."""
    from dashboard_ui import TX_START_COMMAND, TX_STOP_COMMAND
    assert TX_START_COMMAND == b"A", "TX_START_COMMAND must be b'A'"
    assert TX_STOP_COMMAND  == b"B", "TX_STOP_COMMAND must be b'B'"
    assert len(TX_START_COMMAND) == 1, "must be single byte"
    assert len(TX_STOP_COMMAND)  == 1, "must be single byte"
    print("PASS  payload constants are single bytes: %r %r"
          % (TX_START_COMMAND, TX_STOP_COMMAND))


def test_checksum_correctness():
    """Verify the checksum round-trips: the unwrapper must not reject its own output.

    0x10 Transmit Request frames are dispatched to api_frames_other (they are
    outbound frames the GCS built; the unwrapper has nothing to forward from
    them). What matters is that api_frame_errors stays zero, proving the
    checksum field is correct.
    """
    for payload in (b"A", b"B"):
        frame = SerialWorker.build_tx_request(payload, frame_id=1)
        unwrapper = ApiFrameUnwrapper()
        unwrapper.feed(frame)
        assert unwrapper.api_frame_errors == 0, (
            "ApiFrameUnwrapper rejected frame for payload %r (bad checksum)" % payload)
        # Exactly one frame was dispatched (as api_frames_other, not api_frames_ok,
        # because 0x10 is a transmit type, not a receive type).
        assert unwrapper.api_frames_other == 1, (
            "expected api_frames_other=1 for payload %r, got %d"
            % (payload, unwrapper.api_frames_other))
        print("PASS  checksum round-trip for %r: no errors, dispatched as 'other'"
              % payload)


if __name__ == "__main__":
    test_tx_start_frame()
    test_tx_stop_frame()
    test_payload_is_single_byte()
    test_checksum_correctness()
    print("\nAll tests passed.")

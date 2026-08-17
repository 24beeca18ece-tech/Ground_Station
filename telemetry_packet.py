"""
telemetry_packet.py
===================

Packet definition, checksum maths and the frame parser for the CanSat / Rocketry
ground control station.

WIRE FORMAT
-----------
    $TEAM_ID,TIMESTAMP,PACKET_COUNT,ALTITUDE,PRESSURE,TEMP,VOLTAGE,NAV_TIME,
     LAT,LON,NAV_ALT,SATS,ACC_X,ACC_Y,ACC_Z,GYRO_X,GYRO_Y,GYRO_Z,FSM_STATE*CS

* The frame starts at ``$`` and ends at ``*`` followed by exactly two hex digits.
* ``CS`` is the XOR of every byte *between* ``$`` and ``*`` (NMEA-0183 style).
* Exactly 19 comma separated fields live between the delimiters.

THREADING NOTE
--------------
Nothing in this module touches Qt or the GUI.  It is pure, side-effect free
logic so that it can be called from the serial ingestion thread (and from unit
tests / the fault-injection harness) without any locking.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Flight state machine
# ---------------------------------------------------------------------------

FSM_STATES = {
    0: "BOOT",
    1: "TEST_MODE",
    2: "LAUNCH_PAD",
    3: "ASCENT",
    4: "DEPLOY",
    5: "DESCENT",
    6: "AEROBRAKE_RELEASE",
    7: "IMPACT",
}

#: Background colour used for the big FSM banner, one distinct colour per state.
FSM_COLORS = {
    0: "#6b7785",  # BOOT               - slate grey
    1: "#00b0d8",  # TEST_MODE          - cyan
    2: "#e9c135",  # LAUNCH_PAD         - amber
    3: "#f07419",  # ASCENT             - orange
    4: "#e8384f",  # DEPLOY             - red
    5: "#8f6bef",  # DESCENT            - violet
    6: "#00b39b",  # AEROBRAKE_RELEASE  - teal
    7: "#35c46b",  # IMPACT             - green
}

FSM_UNKNOWN_COLOR = "#8a2be2"

#: Number of comma separated fields expected inside the delimiters.
FIELD_COUNT = 19

#: Largest frame we will ever accept.  Anything longer is treated as garbage and
#: is used to force a buffer resync so a missing '*' cannot wedge the reader.
MAX_FRAME_LEN = 512

_FRAME_RE = re.compile(r"^\$(?P<body>[^$*]*)\*(?P<cs>[0-9A-Fa-f]{2})$")


class PacketError(Exception):
    """Base class for every recoverable frame problem."""


class ChecksumError(PacketError):
    """Raised when the transmitted XOR checksum does not match the payload."""


class PacketParseError(PacketError):
    """Raised when the frame is structurally wrong or a field will not convert."""


# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------

def compute_checksum(payload: str) -> int:
    """XOR every byte of *payload* (the text between ``$`` and ``*``)."""
    checksum = 0
    for byte in payload.encode("ascii", errors="replace"):
        checksum ^= byte
    return checksum & 0xFF


def build_frame(payload: str) -> str:
    """Wrap *payload* into a complete, checksum-correct frame.

    Used by the synthetic packet generator and by the unit / fault-injection
    tests; keeping it next to :func:`compute_checksum` guarantees the generator
    and the parser can never drift apart.
    """
    return "$%s*%02X" % (payload, compute_checksum(payload))


# ---------------------------------------------------------------------------
# Field conversion helpers
# ---------------------------------------------------------------------------

def _req_float(raw: str, name: str) -> float:
    """Convert a mandatory float field, raising :class:`PacketParseError`."""
    text = raw.strip()
    if not text:
        raise PacketParseError("mandatory field %r is empty" % name)
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise PacketParseError("field %r is not a number: %r" % (name, raw)) from exc
    if math.isinf(value):
        raise PacketParseError("field %r is infinite: %r" % (name, raw))
    return value


def _req_int(raw: str, name: str) -> int:
    """Convert a mandatory integer field (tolerates ``"3.0"`` style values)."""
    text = raw.strip()
    if not text:
        raise PacketParseError("mandatory field %r is empty" % name)
    try:
        return int(text)
    except (TypeError, ValueError):
        pass
    # Some flight computers emit integers through a float formatter.
    try:
        return int(float(text))
    except (TypeError, ValueError) as exc:
        raise PacketParseError("field %r is not an integer: %r" % (name, raw)) from exc


def _opt_float(raw: str) -> float:
    """Convert an optional float field; blank / unparseable becomes NaN.

    GPS and NavIC receivers legitimately transmit empty lat/lon/alt fields until
    they have a fix, so those fields must not fail a whole packet.
    """
    text = raw.strip()
    if not text:
        return float("nan")
    try:
        value = float(text)
    except (TypeError, ValueError):
        return float("nan")
    return value if not math.isinf(value) else float("nan")


def _opt_int(raw: str, default: int = 0) -> int:
    """Convert an optional integer field; blank / unparseable becomes *default*."""
    text = raw.strip()
    if not text:
        return default
    try:
        return int(text)
    except (TypeError, ValueError):
        pass
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def parse_mission_time(raw: str) -> float:
    """Interpret the TIMESTAMP field as *seconds since boot*.

    Two encodings are accepted because different flight-computer firmware
    revisions in this team have used both:

    * a plain number of seconds (``"137.42"``) — also accepts milliseconds-style
      integers only if the firmware sends seconds, so no scaling is applied;
    * a wall-clock style ``"HH:MM:SS"`` / ``"HH:MM:SS.sss"`` string.
    """
    text = raw.strip()
    if not text:
        raise PacketParseError("empty TIMESTAMP")
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise PacketParseError("bad HH:MM:SS timestamp: %r" % raw)
        try:
            hours, minutes, seconds = (float(p) for p in parts)
        except (TypeError, ValueError) as exc:
            raise PacketParseError("bad HH:MM:SS timestamp: %r" % raw) from exc
        return hours * 3600.0 + minutes * 60.0 + seconds
    return _req_float(text, "TIMESTAMP")


def format_mission_time(seconds: float) -> str:
    """Render seconds-since-boot as ``HH:MM:SS`` (negative values are clamped)."""
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    return "%02d:%02d:%02d" % (total // 3600, (total // 60) % 60, total % 60)


# ---------------------------------------------------------------------------
# The packet itself
# ---------------------------------------------------------------------------

#: CSV column order.  ``CSV_HEADER`` and :meth:`TelemetryPacket.to_csv_row` are
#: kept adjacent on purpose — if you add a field, change both.
CSV_HEADER: List[str] = [
    "gs_recv_iso",        # ground-station wall clock, ISO-8601 UTC
    "gs_recv_epoch",      # ground-station wall clock, float seconds
    "checksum_valid",     # 1 / 0
    "team_id",
    "timestamp",          # raw TIMESTAMP field as transmitted
    "mission_time_s",     # TIMESTAMP normalised to seconds
    "mission_time_hms",   # TIMESTAMP normalised to HH:MM:SS
    "packet_count",
    "altitude_m",
    "pressure_hpa",
    "temp_c",
    "voltage_v",
    "nav_time",
    "lat",
    "lon",
    "nav_alt_m",
    "sats",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "fsm_state",
    "fsm_state_name",
    "raw_frame",
]


@dataclass(slots=True)
class TelemetryPacket:
    """One validated telemetry frame.

    Instances are created on the serial thread and handed to the GUI thread and
    the CSV logger thread through Qt signals / a queue.  Treat them as
    **immutable** once emitted: two threads read every packet.
    """

    team_id: str
    timestamp_raw: str
    mission_time_s: float
    packet_count: int
    altitude_m: float
    pressure_hpa: float
    temp_c: float
    voltage_v: float
    nav_time: str
    lat: float
    lon: float
    nav_alt_m: float
    sats: int
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    fsm_state: int

    raw_frame: str = ""
    checksum_valid: bool = True
    #: Ground-station receive time (``time.time()``), stamped by the serial thread.
    gs_recv_epoch: float = field(default_factory=time.time)

    # -- derived helpers ---------------------------------------------------

    @property
    def fsm_name(self) -> str:
        return FSM_STATES.get(self.fsm_state, "UNKNOWN(%s)" % self.fsm_state)

    @property
    def fsm_color(self) -> str:
        return FSM_COLORS.get(self.fsm_state, FSM_UNKNOWN_COLOR)

    @property
    def mission_time_hms(self) -> str:
        return format_mission_time(self.mission_time_s)

    @property
    def has_fix(self) -> bool:
        """True when lat/lon are usable numbers and not the null-island default."""
        if not (math.isfinite(self.lat) and math.isfinite(self.lon)):
            return False
        if abs(self.lat) < 1e-9 and abs(self.lon) < 1e-9:
            return False
        return -90.0 <= self.lat <= 90.0 and -180.0 <= self.lon <= 180.0

    # -- serialisation -----------------------------------------------------

    def to_csv_row(self) -> List[Any]:
        """Return one CSV record in exactly :data:`CSV_HEADER` order."""
        iso = datetime.fromtimestamp(self.gs_recv_epoch, tz=timezone.utc).isoformat(
            timespec="milliseconds"
        )
        return [
            iso,
            "%.6f" % self.gs_recv_epoch,
            1 if self.checksum_valid else 0,
            self.team_id,
            self.timestamp_raw,
            "%.3f" % self.mission_time_s,
            self.mission_time_hms,
            self.packet_count,
            self.altitude_m,
            self.pressure_hpa,
            self.temp_c,
            self.voltage_v,
            self.nav_time,
            self.lat,
            self.lon,
            self.nav_alt_m,
            self.sats,
            self.acc_x,
            self.acc_y,
            self.acc_z,
            self.gyro_x,
            self.gyro_y,
            self.gyro_z,
            self.fsm_state,
            self.fsm_name,
            self.raw_frame,
        ]


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------

def parse_frame(frame: str, gs_recv_epoch: Optional[float] = None) -> TelemetryPacket:
    """Validate and parse one complete ``$...*XX`` frame.

    Parameters
    ----------
    frame:
        The full frame *including* the ``$`` and the ``*XX`` suffix.
    gs_recv_epoch:
        Ground-station receive time.  Defaults to ``time.time()``.

    Raises
    ------
    ChecksumError
        The XOR checksum did not match — the frame is corrupt on the air link.
    PacketParseError
        The frame is structurally malformed, has the wrong field count, or a
        mandatory field will not convert.

    Both exceptions derive from :class:`PacketError`, so a caller that only
    cares about "was this frame usable" can catch that single type.
    """
    if gs_recv_epoch is None:
        gs_recv_epoch = time.time()

    if not isinstance(frame, str):
        raise PacketParseError("frame is not a string: %r" % type(frame))

    text = frame.strip()
    if not text:
        raise PacketParseError("empty frame")
    if len(text) > MAX_FRAME_LEN:
        raise PacketParseError("frame too long (%d bytes)" % len(text))

    match = _FRAME_RE.match(text)
    if match is None:
        raise PacketParseError("frame does not match $<body>*<hh>: %r" % text[:120])

    body = match.group("body")
    try:
        transmitted = int(match.group("cs"), 16)
    except ValueError as exc:  # pragma: no cover - regex already guarantees hex
        raise PacketParseError("bad checksum digits: %r" % match.group("cs")) from exc

    calculated = compute_checksum(body)
    if calculated != transmitted:
        raise ChecksumError(
            "checksum mismatch: got %02X, expected %02X" % (transmitted, calculated)
        )

    fields = body.split(",")
    if len(fields) != FIELD_COUNT:
        raise PacketParseError(
            "expected %d fields, got %d" % (FIELD_COUNT, len(fields))
        )

    team_id = fields[0].strip()
    if not team_id:
        raise PacketParseError("empty TEAM_ID")

    packet = TelemetryPacket(
        team_id=team_id,
        timestamp_raw=fields[1].strip(),
        mission_time_s=parse_mission_time(fields[1]),
        packet_count=_req_int(fields[2], "PACKET_COUNT"),
        altitude_m=_req_float(fields[3], "ALTITUDE"),
        pressure_hpa=_req_float(fields[4], "PRESSURE"),
        temp_c=_req_float(fields[5], "TEMP"),
        voltage_v=_req_float(fields[6], "VOLTAGE"),
        nav_time=fields[7].strip(),
        lat=_opt_float(fields[8]),
        lon=_opt_float(fields[9]),
        nav_alt_m=_opt_float(fields[10]),
        sats=_opt_int(fields[11], 0),
        acc_x=_opt_float(fields[12]),
        acc_y=_opt_float(fields[13]),
        acc_z=_opt_float(fields[14]),
        gyro_x=_opt_float(fields[15]),
        gyro_y=_opt_float(fields[16]),
        gyro_z=_opt_float(fields[17]),
        fsm_state=_req_int(fields[18], "FSM_STATE"),
        raw_frame=text,
        checksum_valid=True,
        gs_recv_epoch=gs_recv_epoch,
    )
    return packet


def safe_filename(team_id: str) -> str:
    """Strip anything that Windows/POSIX will not accept in a file name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.\-]+", "_", (team_id or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "UNKNOWN"

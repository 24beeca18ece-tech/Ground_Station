"""
telemetry_packet.py
===================

Packet definition, checksum maths and the frame parser for the CanSat / Rocketry
ground control station.

WIRE FORMAT
-----------
Two generations of the format are accepted.  The parser auto-detects which one
it is looking at, so old logs and old firmware keep working unchanged.

**v1 (legacy, 19 fields)** -- no payload type, generic sensor set::

    $TEAM_ID,TIMESTAMP,PACKET_COUNT,ALTITUDE,PRESSURE,TEMP,VOLTAGE,NAV_TIME,
     LAT,LON,NAV_ALT,SATS,ACC_X,ACC_Y,ACC_Z,GYRO_X,GYRO_Y,GYRO_Z,FSM_STATE*CS

**v2 (current)** -- ``PAYLOAD_TYPE`` inserted directly after ``TEAM_ID``, and
vehicle-specific sensors appended after ``FSM_STATE``::

    $TEAM_ID,PAYLOAD_TYPE,TIMESTAMP,...,FSM_STATE[,<vehicle fields>]*CS

    CANSAT (25 fields)  adds  PM1_0,PM2_5,PM10,REACTION_WHEEL_RPM,RECOVERY_STAGE
    ROCKET (22 fields)  adds  SOLENOID_FIRED,NICHROME_FIRED

* The frame starts at ``$`` and ends at ``*`` followed by exactly two hex digits.
* ``CS`` is the XOR of every byte *between* ``$`` and ``*`` (NMEA-0183 style).
  Because it covers the whole body, the checksum automatically spans the new v2
  fields -- there is no per-field checksum logic to keep in sync.

CLASS HIERARCHY
---------------
``TelemetryPacket`` holds everything both vehicles share.  ``CanSatPacket`` and
``RocketPacket`` subclass it and add their unique sensors.  Every consumer that
only needs common fields (charts, CSV timestamps, GPS track) can keep treating
packets as plain ``TelemetryPacket`` and does not care which vehicle sent them.

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
#: Also reused by the session-summary pie chart so the two always agree.
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

# ---------------------------------------------------------------------------
# Payload types
# ---------------------------------------------------------------------------

PAYLOAD_GENERIC = "GENERIC"   #: legacy v1 frame, no vehicle-specific sensors
PAYLOAD_CANSAT = "CANSAT"
PAYLOAD_ROCKET = "ROCKET"

PAYLOAD_TYPES = (PAYLOAD_GENERIC, PAYLOAD_CANSAT, PAYLOAD_ROCKET)

#: CanSat recovery sequencer stages (SPS30 payload + parafoil recovery).
RECOVERY_STAGES = {
    0: "STOWED",
    1: "DROGUE",
    2: "PARAFOIL",
}

RECOVERY_STAGE_COLORS = {
    0: "#6b7785",  # stowed   - grey
    1: "#e9c135",  # drogue   - amber
    2: "#35c46b",  # parafoil - green
}

# ---------------------------------------------------------------------------
# Field layout
# ---------------------------------------------------------------------------

#: Legacy v1 body, in wire order.
FIELDS_V1: List[str] = [
    "TEAM_ID", "TIMESTAMP", "PACKET_COUNT", "ALTITUDE", "PRESSURE", "TEMP",
    "VOLTAGE", "NAV_TIME", "LAT", "LON", "NAV_ALT", "SATS",
    "ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z", "FSM_STATE",
]

#: v2 common prefix, in wire order.  Identical to v1 with PAYLOAD_TYPE spliced
#: in at index 1, which is why every v2 field index is one higher than v1.
FIELDS_V2_COMMON: List[str] = [
    "TEAM_ID", "PAYLOAD_TYPE", "TIMESTAMP", "PACKET_COUNT", "ALTITUDE",
    "PRESSURE", "TEMP", "VOLTAGE", "NAV_TIME", "LAT", "LON", "NAV_ALT", "SATS",
    "ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z", "FSM_STATE",
]

#: Sensirion SPS30 particulate payload + active stabilisation + recovery stage.
FIELDS_CANSAT_EXTRA: List[str] = [
    "PM1_0", "PM2_5", "PM10", "REACTION_WHEEL_RPM", "RECOVERY_STAGE",
]

#: Dual-stage pyrotechnic / mechanical recovery status flags.
FIELDS_ROCKET_EXTRA: List[str] = [
    "SOLENOID_FIRED", "NICHROME_FIRED",
]

FIELD_COUNT_V1 = len(FIELDS_V1)                                    # 19
FIELD_COUNT_CANSAT = len(FIELDS_V2_COMMON) + len(FIELDS_CANSAT_EXTRA)  # 25
FIELD_COUNT_ROCKET = len(FIELDS_V2_COMMON) + len(FIELDS_ROCKET_EXTRA)  # 22

#: Kept for backwards compatibility with code that imported the old name.
FIELD_COUNT = FIELD_COUNT_V1

#: Largest frame we will ever accept.  Anything longer is treated as garbage and
#: is used to force a buffer resync so a missing '*' cannot wedge the reader.
#: Raised from 512 for v2: a CanSat frame is ~160 bytes, so 512 was still ample,
#: but the headroom costs nothing and protects against future field growth.
MAX_FRAME_LEN = 640

_FRAME_RE = re.compile(r"^\$(?P<body>[^$*]*)\*(?P<cs>[0-9A-Fa-f]{2})$")

#: Sensirion SPS30 mass-concentration measurement range, micrograms/m^3.
SPS30_MIN_UGM3 = 0.0
SPS30_MAX_UGM3 = 1000.0


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
    """XOR every byte of *payload* (the text between ``$`` and ``*``).

    This covers the entire body, so it spans the v2 vehicle-specific fields
    with no change: there is deliberately no per-field checksum logic that
    could fall out of step with the field list.
    """
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


def expected_field_count(payload_type: str) -> int:
    """Number of body fields a frame of *payload_type* must carry."""
    if payload_type == PAYLOAD_CANSAT:
        return FIELD_COUNT_CANSAT
    if payload_type == PAYLOAD_ROCKET:
        return FIELD_COUNT_ROCKET
    return FIELD_COUNT_V1


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
    they have a fix, so those fields must not fail a whole packet.  The same
    applies to the SPS30, which reports nothing during its 8-second fan warm-up.
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


def _opt_bool(raw: str) -> bool:
    """Convert an optional 0/1 flag field.

    Accepts ``1``/``0``, ``true``/``false`` and ``yes``/``no`` in any case, so
    that a firmware change in how the flag is formatted cannot silently turn a
    fired pyrotechnic into a not-fired one.
    """
    text = raw.strip().upper()
    if not text:
        return False
    if text in ("1", "TRUE", "T", "YES", "Y", "FIRED", "HIGH"):
        return True
    if text in ("0", "FALSE", "F", "NO", "N", "SAFE", "LOW"):
        return False
    # Numeric fall-back: any non-zero number counts as fired.
    try:
        return abs(float(text)) > 0.5
    except (TypeError, ValueError):
        return False


def parse_mission_time(raw: str) -> float:
    """Interpret the TIMESTAMP field as *seconds since boot*.

    Two encodings are accepted because different flight-computer firmware
    revisions in this team have used both:

    * a plain number of seconds (``"137.42"``);
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
# CSV schema
# ---------------------------------------------------------------------------

#: Number of vehicle-specific cells every packet contributes to a CSV row.
#: A packet that does not carry a given sensor writes an empty cell for it, so
#: one CSV can hold a mixed CanSat/Rocket session without a schema change.
_VARIANT_CELL_COUNT = 8

#: CSV column order.  ``CSV_HEADER``, :meth:`TelemetryPacket.to_csv_row` and
#: :meth:`TelemetryPacket._variant_cells` are kept adjacent on purpose -- if you
#: add a field, change all three.
CSV_HEADER: List[str] = [
    "gs_recv_iso",        # ground-station wall clock, ISO-8601 UTC
    "gs_recv_epoch",      # ground-station wall clock, float seconds
    "checksum_valid",     # 1 / 0
    "team_id",
    "payload_type",       # CANSAT / ROCKET / GENERIC
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
    # --- vehicle-specific (blank when not applicable) ----------------------
    "pm1_0_ugm3",             # CanSat: Sensirion SPS30
    "pm2_5_ugm3",             # CanSat
    "pm10_ugm3",              # CanSat
    "reaction_wheel_rpm",     # CanSat: active stabilisation
    "recovery_stage",         # CanSat: 0/1/2
    "recovery_stage_name",    # CanSat: STOWED/DROGUE/PARAFOIL
    "solenoid_fired",         # Rocket: 6 V latch at apogee
    "nichrome_fired",         # Rocket: cutter at 400 m AGL
    # -----------------------------------------------------------------------
    "raw_frame",
]


# ---------------------------------------------------------------------------
# The packets
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TelemetryPacket:
    """One validated telemetry frame -- fields common to every vehicle.

    Instances are created on the serial thread and handed to the GUI thread and
    the CSV logger thread through Qt signals / a queue.  Treat them as
    **immutable** once emitted: two threads read every packet.

    .. note::
       Subclasses use ``slots=True`` too, which means ``dataclass`` rebuilds the
       class object.  Zero-argument ``super()`` is therefore unreliable inside
       these classes -- override :meth:`_variant_cells` rather than chaining
       ``to_csv_row`` through ``super()``.
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

    payload_type: str = PAYLOAD_GENERIC
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

    @property
    def is_cansat(self) -> bool:
        return self.payload_type == PAYLOAD_CANSAT

    @property
    def is_rocket(self) -> bool:
        return self.payload_type == PAYLOAD_ROCKET

    # -- serialisation -----------------------------------------------------

    def _variant_cells(self) -> List[Any]:
        """Vehicle-specific CSV cells; blank for a packet with no extra sensors."""
        return [""] * _VARIANT_CELL_COUNT

    def to_csv_row(self) -> List[Any]:
        """Return one CSV record in exactly :data:`CSV_HEADER` order."""
        iso = datetime.fromtimestamp(self.gs_recv_epoch, tz=timezone.utc).isoformat(
            timespec="milliseconds"
        )
        row: List[Any] = [
            iso,
            "%.6f" % self.gs_recv_epoch,
            1 if self.checksum_valid else 0,
            self.team_id,
            self.payload_type,
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
        ]
        row.extend(self._variant_cells())
        row.append(self.raw_frame)
        return row


@dataclass(slots=True)
class CanSatPacket(TelemetryPacket):
    """CanSat frame: Sensirion SPS30 particulate payload + active stabilisation.

    ``pm*`` are mass concentrations in micrograms per cubic metre.  ``NaN``
    means the SPS30 reported nothing for that channel (typically during its fan
    warm-up), which is a valid state and must not invalidate the packet.
    """

    pm1_0: float = float("nan")
    pm2_5: float = float("nan")
    pm10: float = float("nan")
    #: Signed: positive is one direction of wheel spin, negative the other.
    reaction_wheel_rpm: int = 0
    recovery_stage: int = 0

    @property
    def recovery_stage_name(self) -> str:
        return RECOVERY_STAGES.get(self.recovery_stage,
                                   "UNKNOWN(%s)" % self.recovery_stage)

    @property
    def recovery_stage_color(self) -> str:
        return RECOVERY_STAGE_COLORS.get(self.recovery_stage, FSM_UNKNOWN_COLOR)

    def _variant_cells(self) -> List[Any]:
        return [
            self.pm1_0, self.pm2_5, self.pm10,
            self.reaction_wheel_rpm,
            self.recovery_stage, self.recovery_stage_name,
            "", "",          # solenoid / nichrome: not fitted to the CanSat
        ]


@dataclass(slots=True)
class RocketPacket(TelemetryPacket):
    """Rocket frame: dual-stage pyrotechnic / mechanical recovery status."""

    #: 6 V solenoid latch released at apogee (drogue event).
    solenoid_fired: bool = False
    #: Nichrome cutter fired at 400 m AGL (main deployment event).
    nichrome_fired: bool = False

    def _variant_cells(self) -> List[Any]:
        return [
            "", "", "", "", "", "",   # PM / wheel / recovery stage: CanSat only
            1 if self.solenoid_fired else 0,
            1 if self.nichrome_fired else 0,
        ]


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------

def _detect_variant(fields: List[str]) -> str:
    """Decide which format *fields* is, from the payload token and field count.

    Resolution order, chosen so that a corrupted PAYLOAD_TYPE token can never
    cause a rocket frame to be read with the CanSat field layout:

    1. A recognised ``PAYLOAD_TYPE`` token wins, but the field count must agree
       with it -- a mismatch is an error, not a reason to guess.
    2. Otherwise fall back on the field count alone.  Exactly 19 fields is a
       legacy v1 frame, which is what keeps old logs and old firmware working.
    """
    count = len(fields)
    token = fields[1].strip().upper() if count > 1 else ""

    if token in (PAYLOAD_CANSAT, PAYLOAD_ROCKET):
        expected = expected_field_count(token)
        if count != expected:
            raise PacketParseError(
                "%s packet: expected %d fields, got %d" % (token, expected, count)
            )
        return token

    # No usable payload-type token: infer from the field count.
    if count == FIELD_COUNT_V1:
        return PAYLOAD_GENERIC
    if count == FIELD_COUNT_CANSAT:
        return PAYLOAD_CANSAT
    if count == FIELD_COUNT_ROCKET:
        return PAYLOAD_ROCKET

    raise PacketParseError(
        "unrecognised field count %d (expected %d, %d or %d)"
        % (count, FIELD_COUNT_V1, FIELD_COUNT_ROCKET, FIELD_COUNT_CANSAT)
    )


def _common_kwargs(fields: List[str], base: int) -> dict:
    """Build the constructor arguments shared by every packet class.

    *base* is the index of the TIMESTAMP field: 1 for legacy v1 frames, 2 for
    v2 frames where PAYLOAD_TYPE occupies index 1.
    """
    return dict(
        timestamp_raw=fields[base].strip(),
        mission_time_s=parse_mission_time(fields[base]),
        packet_count=_req_int(fields[base + 1], "PACKET_COUNT"),
        altitude_m=_req_float(fields[base + 2], "ALTITUDE"),
        pressure_hpa=_req_float(fields[base + 3], "PRESSURE"),
        temp_c=_req_float(fields[base + 4], "TEMP"),
        voltage_v=_req_float(fields[base + 5], "VOLTAGE"),
        nav_time=fields[base + 6].strip(),
        lat=_opt_float(fields[base + 7]),
        lon=_opt_float(fields[base + 8]),
        nav_alt_m=_opt_float(fields[base + 9]),
        sats=_opt_int(fields[base + 10], 0),
        acc_x=_opt_float(fields[base + 11]),
        acc_y=_opt_float(fields[base + 12]),
        acc_z=_opt_float(fields[base + 13]),
        gyro_x=_opt_float(fields[base + 14]),
        gyro_y=_opt_float(fields[base + 15]),
        gyro_z=_opt_float(fields[base + 16]),
        fsm_state=_req_int(fields[base + 17], "FSM_STATE"),
    )


def parse_frame(frame: str, gs_recv_epoch: Optional[float] = None) -> TelemetryPacket:
    """Validate and parse one complete ``$...*XX`` frame.

    Returns a :class:`CanSatPacket`, a :class:`RocketPacket` or a plain
    :class:`TelemetryPacket` depending on the detected format.  Callers that
    only touch common fields need not care which.

    Parameters
    ----------
    frame:
        The full frame *including* the ``$`` and the ``*XX`` suffix.
    gs_recv_epoch:
        Ground-station receive time.  Defaults to ``time.time()``.

    Raises
    ------
    ChecksumError
        The XOR checksum did not match -- the frame is corrupt on the air link.
    PacketParseError
        The frame is structurally malformed, has an unrecognised field count, or
        a mandatory field will not convert.

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
    variant = _detect_variant(fields)

    team_id = fields[0].strip()
    if not team_id:
        raise PacketParseError("empty TEAM_ID")

    shared = dict(
        team_id=team_id,
        raw_frame=text,
        checksum_valid=True,
        gs_recv_epoch=gs_recv_epoch,
    )

    if variant == PAYLOAD_GENERIC:
        # Legacy v1: TIMESTAMP sits at index 1, no vehicle-specific tail.
        return TelemetryPacket(
            payload_type=PAYLOAD_GENERIC,
            **shared,
            **_common_kwargs(fields, 1),
        )

    common = _common_kwargs(fields, 2)
    extra_at = len(FIELDS_V2_COMMON)   # first index past FSM_STATE

    if variant == PAYLOAD_CANSAT:
        return CanSatPacket(
            payload_type=PAYLOAD_CANSAT,
            **shared,
            **common,
            pm1_0=_opt_float(fields[extra_at]),
            pm2_5=_opt_float(fields[extra_at + 1]),
            pm10=_opt_float(fields[extra_at + 2]),
            reaction_wheel_rpm=_opt_int(fields[extra_at + 3], 0),
            recovery_stage=_opt_int(fields[extra_at + 4], 0),
        )

    return RocketPacket(
        payload_type=PAYLOAD_ROCKET,
        **shared,
        **common,
        solenoid_fired=_opt_bool(fields[extra_at]),
        nichrome_fired=_opt_bool(fields[extra_at + 1]),
    )


def safe_filename(team_id: str) -> str:
    """Strip anything that Windows/POSIX will not accept in a file name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.\-]+", "_", (team_id or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "UNKNOWN"

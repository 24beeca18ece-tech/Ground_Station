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

    CANSAT (26 fields)  adds  PM1_0,PM2_5,PM4_0,PM10,REACTION_WHEEL_RPM,
                              RECOVERY_STAGE   (25-field pre-PM4_0 also read)
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
#:
#: DAYLIGHT THEME: these are banner *fills* carrying white text, on a light UI.
#: Each is therefore saturated but dark enough for white to read on it (>=4.5:1)
#: -- the previous set was tuned to sit on near-black under dark text, and the
#: bright ones (amber #e9c135, green #35c46b, cyan #00b0d8) would have left
#: white text unreadable and the banner glaring against a light panel.
FSM_COLORS = {
    0: "#4a5768",  # BOOT               - slate grey
    1: "#00697d",  # TEST_MODE          - deep cyan
    2: "#8a5000",  # LAUNCH_PAD         - dark amber
    3: "#c1490b",  # ASCENT             - burnt orange
    4: "#c0182b",  # DEPLOY             - red
    5: "#5b3fbe",  # DESCENT            - violet
    6: "#00786a",  # AEROBRAKE_RELEASE  - teal
    7: "#0d7a3d",  # IMPACT             - green
}

FSM_UNKNOWN_COLOR = "#7b1fa2"

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

#: Same daylight calibration as FSM_COLORS: saturated fills under white text.
RECOVERY_STAGE_COLORS = {
    0: "#4a5768",  # stowed   - grey
    1: "#8a5000",  # drogue   - dark amber
    2: "#0d7a3d",  # parafoil - green
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
#: PM4_0 sits between PM2_5 and PM10, matching the order the SPS30 itself
#: reports its four mass concentrations in.
FIELDS_CANSAT_EXTRA: List[str] = [
    "PM1_0", "PM2_5", "PM4_0", "PM10", "REACTION_WHEEL_RPM", "RECOVERY_STAGE",
]

#: The CanSat layout before PM4_0 was added. Frames of this width are still
#: accepted and parsed, with PM4_0 left as NaN -- old logs and any flight
#: firmware not yet reflashed keep working.
FIELDS_CANSAT_EXTRA_LEGACY: List[str] = [
    "PM1_0", "PM2_5", "PM10", "REACTION_WHEEL_RPM", "RECOVERY_STAGE",
]

#: Dual-stage pyrotechnic / mechanical recovery status flags.
FIELDS_ROCKET_EXTRA: List[str] = [
    "SOLENOID_FIRED", "NICHROME_FIRED",
]

FIELD_COUNT_V1 = len(FIELDS_V1)                                    # 19
FIELD_COUNT_CANSAT = len(FIELDS_V2_COMMON) + len(FIELDS_CANSAT_EXTRA)  # 26
#: Pre-PM4_0 CanSat width, still accepted on the wire.
FIELD_COUNT_CANSAT_LEGACY = (len(FIELDS_V2_COMMON)
                             + len(FIELDS_CANSAT_EXTRA_LEGACY))  # 25
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

#: Standard gravity, for expressing accelerometer limits in g.
G_MS2 = 9.80665

#: Physical plausibility envelope, ``field -> (low, high, unit)``.
#:
#: The XOR checksum proves a frame arrived intact; it says nothing about
#: whether the numbers inside it are physically possible.  A sensor read that
#: fails on the vehicle -- a mis-clocked I2C transaction, a barometer returning
#: its uninitialised register contents -- produces a frame that is perfectly
#: valid on the wire and complete nonsense as telemetry.  Such a packet plotted
#: straight onto a chart wrenches the Y axis to an absurd range and drags every
#: subsequent redraw with it.
#:
#: These bounds are deliberately generous: they are a nonsense filter, not a
#: flight-envelope check.  Anything inside them is passed through untouched
#: even if it is unusual, because rejecting real data is far worse than
#: admitting a slightly odd value.
PHYSICAL_BOUNDS = {
    "ALTITUDE":    (-500.0, 15000.0, "m"),
    "PRESSURE":    (300.0, 1100.0, "hPa"),
    "TEMPERATURE": (-80.0, 85.0, "C"),
    "VOLTAGE":     (0.0, 20.0, "V"),
    # +/-50 g, expressed in the m/s^2 the packet actually carries.
    "ACCEL":       (-50.0 * G_MS2, 50.0 * G_MS2, "m/s2"),
    "GYRO":        (-2000.0, 2000.0, "deg/s"),
}


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
_VARIANT_CELL_COUNT = 9

#: CSV column order.  ``CSV_HEADER``, :meth:`TelemetryPacket.to_csv_row` and
#: :meth:`TelemetryPacket._variant_cells` are kept adjacent on purpose -- if you
#: add a field, change all three.
CSV_HEADER: List[str] = [
    "gs_recv_iso",        # ground-station wall clock, ISO-8601 UTC
    "gs_recv_epoch",      # ground-station wall clock, float seconds
    "radio",              # which ground radio received it (RX1/RX2), "" if one
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
    "pm4_0_ugm3",             # CanSat
    "pm10_ugm3",              # CanSat
    "reaction_wheel_rpm",     # CanSat: active stabilisation
    "recovery_stage",         # CanSat: 0/1/2
    "recovery_stage_name",    # CanSat: STOWED/DROGUE/PARAFOIL
    "solenoid_fired",         # Rocket: 6 V latch at apogee
    "nichrome_fired",         # Rocket: cutter at 400 m AGL
    # -----------------------------------------------------------------------
    "raw_frame",
]

#: The vehicle-specific block runs from the first CanSat column to just before
#: raw_frame, which is always last. Checked against _VARIANT_CELL_COUNT at
#: import so a header change that forgets the row builders fails loudly here
#: rather than silently writing a misaligned CSV for one payload type.
_VARIANT_SPAN = (CSV_HEADER.index("pm1_0_ugm3"), CSV_HEADER.index("raw_frame"))
if _VARIANT_SPAN[1] - _VARIANT_SPAN[0] != _VARIANT_CELL_COUNT:
    raise RuntimeError(
        "CSV schema mismatch: header has %d vehicle-specific columns but "
        "_VARIANT_CELL_COUNT is %d. Update CSV_HEADER, _VARIANT_CELL_COUNT and "
        "every _variant_cells() together."
        % (_VARIANT_SPAN[1] - _VARIANT_SPAN[0], _VARIANT_CELL_COUNT)
    )



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
    #: False when the wire format carries no flight state (raw-CSV mode). The
    #: dashboard shows "NO FSM DATA" rather than rendering ``fsm_state`` as if
    #: BOOT had been reported, which would be an invented reading.
    has_fsm_data: bool = True
    #: False when the wire format carries no battery telemetry (raw-CSV mode).
    has_voltage: bool = True
    #: Which ground radio delivered this packet ("RX1"/"RX2"), or "" on a
    #: single-radio session. Stamped by the merge layer, not by the parser --
    #: the wire format carries no such field, and inventing one would be a lie
    #: about what the flight computer sent.
    radio: str = ""

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
        """True when lat/lon describe a real fix, not a marginal or null one.

        Satellite count is part of the test because coordinates alone are not
        trustworthy: a receiver with a flickering indoor fix emits small,
        plausible-looking values (0.083333, 0.016667 were both seen on this
        hardware) before it resets them to zero. A fix with no satellites behind
        it is not a fix, so those never reach the ground track.
        """
        if not (math.isfinite(self.lat) and math.isfinite(self.lon)):
            return False
        if self.sats < 1:
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

    # -- physical validation ------------------------------------------------

    def implausible_reasons(self) -> List[str]:
        """Return every physical-bounds violation in this packet.

        An empty list means the packet is plausible.  This is a *separate*
        judgement from the checksum: the checksum says the bytes survived the
        link, this says the numbers could have come from a working sensor.

        ``NaN`` is never a violation.  Optional fields are legitimately absent
        -- no GPS fix yet, no IMU fitted on this airframe -- and treating
        "missing" as "impossible" would throw away good packets.
        """
        bad: List[str] = []

        def check(name: str, value: float, key: str) -> None:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return                      # missing data is not implausible
            low, high, unit = PHYSICAL_BOUNDS[key]
            if value < low or value > high:
                bad.append("%s=%.2f%s outside [%.1f, %.1f]"
                           % (name, value, unit, low, high))

        check("ALTITUDE", self.altitude_m, "ALTITUDE")
        check("PRESSURE", self.pressure_hpa, "PRESSURE")
        check("TEMP", self.temp_c, "TEMPERATURE")
        check("VOLTAGE", self.voltage_v, "VOLTAGE")
        check("ACC_X", self.acc_x, "ACCEL")
        check("ACC_Y", self.acc_y, "ACCEL")
        check("ACC_Z", self.acc_z, "ACCEL")
        check("GYRO_X", self.gyro_x, "GYRO")
        check("GYRO_Y", self.gyro_y, "GYRO")
        check("GYRO_Z", self.gyro_z, "GYRO")
        return bad

    @property
    def is_plausible(self) -> bool:
        """True when every finite sensor value lies inside its physical bounds."""
        return not self.implausible_reasons()

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
            self.radio,
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
    pm4_0: float = float("nan")
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
            self.pm1_0, self.pm2_5, self.pm4_0, self.pm10,
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
            # PM1.0 / PM2.5 / PM4.0 / PM10 / wheel RPM / recovery stage /
            # recovery stage name: all CanSat-only, blank on a rocket.
            "", "", "", "", "", "", "",
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
        # A CanSat frame one field short is the pre-PM4_0 layout, not an error.
        if token == PAYLOAD_CANSAT and count == FIELD_COUNT_CANSAT_LEGACY:
            return token
        if count != expected:
            raise PacketParseError(
                "%s packet: expected %d fields, got %d" % (token, expected, count)
            )
        return token

    # No usable payload-type token: infer from the field count.
    if count == FIELD_COUNT_V1:
        return PAYLOAD_GENERIC
    if count in (FIELD_COUNT_CANSAT, FIELD_COUNT_CANSAT_LEGACY):
        return PAYLOAD_CANSAT
    if count == FIELD_COUNT_ROCKET:
        return PAYLOAD_ROCKET

    raise PacketParseError(
        "unrecognised field count %d (expected %d, %d, %d or %d)"
        % (count, FIELD_COUNT_V1, FIELD_COUNT_ROCKET,
           FIELD_COUNT_CANSAT_LEGACY, FIELD_COUNT_CANSAT)
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
        legacy_cansat = len(fields) == FIELD_COUNT_CANSAT_LEGACY
        return CanSatPacket(
            payload_type=PAYLOAD_CANSAT,
            **shared,
            **common,
            pm1_0=_opt_float(fields[extra_at]),
            pm2_5=_opt_float(fields[extra_at + 1]),
            # Pre-PM4_0 frames are one field narrower; PM4_0 stays NaN and
            # every field after it shifts back by one.
            pm4_0=(_opt_float(fields[extra_at + 2]) if not legacy_cansat
                   else float("nan")),
            pm10=_opt_float(fields[extra_at + (3 if not legacy_cansat else 2)]),
            reaction_wheel_rpm=_opt_int(
                fields[extra_at + (4 if not legacy_cansat else 3)], 0),
            recovery_stage=_opt_int(
                fields[extra_at + (5 if not legacy_cansat else 4)], 0),
        )

    return RocketPacket(
        payload_type=PAYLOAD_ROCKET,
        **shared,
        **common,
        solenoid_fired=_opt_bool(fields[extra_at]),
        nichrome_fired=_opt_bool(fields[extra_at + 1]),
    )


# ---------------------------------------------------------------------------
# Raw-CSV compatibility mode
# ---------------------------------------------------------------------------
#
# Bench hardware (Teensy 4.1, firmware revised Aug 2026) emits bare CSV inside
# each XBee RF frame -- no '$' prefix, no '*XX' checksum, no TEAM_ID, and no
# newline; the RF frame boundary *is* the record boundary. This mode exists so
# that hardware can be tested before the firmware is brought back to spec. It
# is OFF by default and must be enabled explicitly.
#
# IMPORTANT: there is no checksum in this format, so nothing detects a packet
# corrupted on the air link. Frames that arrive damaged will be accepted as
# valid and only the plausibility filter stands between them and the display.
# This is a bench-test aid, not a flight configuration.

# ---------------------------------------------------------------------------
# !! FIELD MAP -- GROUND TRUTH, verified against the Teensy firmware's snprintf
# ---------------------------------------------------------------------------
# Confirmed from the sender's source (Aug 2026), not inferred from values. An
# earlier statistical pass over ~950 captured frames agreed on indices 0-9 but
# got the tail wrong; both corrections are recorded here so the reasoning is not
# repeated:
#
#   * Index 13 is SATELLITES, not FSM_STATE. This format carries NO flight
#     state at all -- the firmware does not transmit it.
#   * Indices 10/11 are LATITUDE/LONGITUDE. The odd values seen across captures
#     (0.083333, 0.016667, 9.685300) were marginal indoor GPS fixes, not a rate
#     field: the firmware only zeroes them when gps.location.isValid() is false,
#     so a flickering fix emits small nonsense before resetting to 0.0. This is
#     why position is gated on SATELLITES rather than trusted on its own.
#
# TWO INDEPENDENT ALTITUDES, deliberately kept apart:
#   * index 3  BARO_ALTITUDE -- MS5611, against a fixed 1013.25 hPa sea-level
#     reference with no ground zero-set applied, so it carries a constant offset.
#   * index 12 GPS_ALTITUDE  -- from the GPS fix; reads 0.0 with no fix.
# They are different measurements with different failure modes and must not be
# conflated.
#
# Anything mapped to None is absent from this wire format, ignored, and simply
# preserved in ``raw_frame``.
RAW_CSV_FIELD_MAP = {
    "PACKET_COUNT": 0,     # PACKET_ID       (unsigned long)
    "TEMP": 1,             # TEMP_C          (2 dp)
    "PRESSURE": 2,         # PRESSURE_HPA    (2 dp)
    "ALTITUDE": 3,         # BARO_ALTITUDE_M (2 dp, 1013.25 hPa reference)
    "ACC_X": 4,            # ACCEL_X_G       (2 dp)
    "ACC_Y": 5,            # ACCEL_Y_G       (2 dp)
    "ACC_Z": 6,            # ACCEL_Z_G       (2 dp)
    "GYRO_X": 7,           # GYRO_X          (2 dp)
    "GYRO_Y": 8,           # GYRO_Y          (2 dp)
    "GYRO_Z": 9,           # GYRO_Z          (2 dp)
    "LAT": 10,             # LATITUDE        (6 dp)
    "LON": 11,             # LONGITUDE       (6 dp)
    "NAV_ALT": 12,         # GPS_ALTITUDE_M  (2 dp) -- distinct from index 3
    "SATS": 13,            # SATELLITES      (unsigned long)
    # Absent from this wire format entirely:
    "VOLTAGE": None,       # no battery telemetry
    "FSM_STATE": None,     # firmware sends no flight state in raw CSV
}

#: A position fix is only believed when the GPS reports at least this many
#: satellites. Without this, marginal indoor fixes plot as real coordinates.
RAW_CSV_MIN_SATS = 1

#: Number of fields in the observed record.
RAW_CSV_FIELD_COUNT = 14

#: Shortest record we will accept: enough fields to satisfy every mapped index.
RAW_CSV_MIN_FIELDS = max(
    (i for i in RAW_CSV_FIELD_MAP.values() if i is not None), default=0
) + 1


def _raw_csv_get(fields: List[str], name: str, default: float = 0.0) -> str:
    """Return the raw text for *name*, or "" when it is unmapped/absent."""
    idx = RAW_CSV_FIELD_MAP.get(name)
    if idx is None or idx >= len(fields):
        return ""
    return fields[idx]


def parse_raw_csv(record: str, team_id: str,
                  gs_recv_epoch: Optional[float] = None,
                  mission_epoch: Optional[float] = None) -> TelemetryPacket:
    """Parse one bare-CSV record from the pre-spec Teensy firmware.

    The record carries no team ID, no mission clock, no battery voltage and no
    GPS, so those are filled from *team_id* and the ground-station clock rather
    than invented. Everything downstream -- plausibility filtering, CSV logging,
    the dashboard -- then works unchanged.

    Parameters
    ----------
    record:
        One record, without ``$``/``*XX``, e.g.
        ``"5635,24.36,944.42,589.48,0.00,-0.00,1.00,0.01,-0.04,-0.02,..."``.
    team_id:
        Substituted for the missing TEAM_ID field.
    gs_recv_epoch:
        Ground-station receive time. Defaults to ``time.time()``.
    mission_epoch:
        Epoch the mission clock is measured from. Mission time is derived as
        ``gs_recv_epoch - mission_epoch`` because the firmware sends no clock.
        When omitted, mission time is 0.

    Raises
    ------
    PacketParseError
        Too few fields, or a mandatory field will not convert.
    """
    if gs_recv_epoch is None:
        gs_recv_epoch = time.time()

    if not isinstance(record, str):
        raise PacketParseError("record is not a string: %r" % type(record))

    text = record.strip()
    if not text:
        raise PacketParseError("empty record")
    if len(text) > MAX_FRAME_LEN:
        raise PacketParseError("record too long (%d bytes)" % len(text))

    fields = text.split(",")
    if len(fields) < RAW_CSV_MIN_FIELDS:
        raise PacketParseError(
            "raw-CSV record has %d fields, need at least %d: %r"
            % (len(fields), RAW_CSV_MIN_FIELDS, text[:120])
        )

    mission_s = 0.0
    if mission_epoch is not None:
        mission_s = max(gs_recv_epoch - mission_epoch, 0.0)

    # GPS gating. The firmware zeroes LAT/LON only when gps.location.isValid()
    # is false, so a flickering indoor fix emits small nonsense coordinates
    # before it resets them. Satellite count is the reliable discriminator, so
    # position is dropped outright unless the fix is backed by satellites.
    sats = _opt_int(_raw_csv_get(fields, "SATS"), 0)
    lat = _opt_float(_raw_csv_get(fields, "LAT"))
    lon = _opt_float(_raw_csv_get(fields, "LON"))
    nav_alt = _opt_float(_raw_csv_get(fields, "NAV_ALT"))
    if sats < RAW_CSV_MIN_SATS:
        lat = lon = nav_alt = 0.0

    # This format reports acceleration in g; every consumer downstream (the
    # plausibility envelope, the accel strip chart, the attitude estimator)
    # works in m/s^2, matching the $..*XX formats. Convert here so the raw-CSV
    # path is not the one place with different units.
    acc_x = _opt_float(_raw_csv_get(fields, "ACC_X")) * G_MS2
    acc_y = _opt_float(_raw_csv_get(fields, "ACC_Y")) * G_MS2
    acc_z = _opt_float(_raw_csv_get(fields, "ACC_Z")) * G_MS2

    return TelemetryPacket(
        payload_type=PAYLOAD_GENERIC,
        team_id=team_id,
        raw_frame=text,
        # No checksum exists in this format. Saying "valid" would claim an
        # integrity guarantee the wire never provided.
        checksum_valid=False,
        gs_recv_epoch=gs_recv_epoch,
        timestamp_raw=format_mission_time(mission_s),
        mission_time_s=mission_s,
        packet_count=_req_int(_raw_csv_get(fields, "PACKET_COUNT") or "0",
                              "PACKET_COUNT"),
        altitude_m=_opt_float(_raw_csv_get(fields, "ALTITUDE")),
        pressure_hpa=_opt_float(_raw_csv_get(fields, "PRESSURE")),
        temp_c=_opt_float(_raw_csv_get(fields, "TEMP")),
        # Unmapped fields resolve to 0.0, which the tiles read as "no data".
        voltage_v=_opt_float(_raw_csv_get(fields, "VOLTAGE")),
        nav_time="",
        lat=lat,
        lon=lon,
        # GPS altitude (index 12) is a separate measurement from the barometric
        # altitude in index 3 and is kept in its own field, never merged.
        nav_alt_m=nav_alt,
        sats=sats,
        acc_x=acc_x,
        acc_y=acc_y,
        acc_z=acc_z,
        # Gyro units are NOT stated by the firmware. Treated as deg/s to match
        # the $..*XX formats; at rest both deg/s and rad/s read ~0, so this
        # capture could not distinguish them. Confirm before trusting rates.
        gyro_x=_opt_float(_raw_csv_get(fields, "GYRO_X")),
        gyro_y=_opt_float(_raw_csv_get(fields, "GYRO_Y")),
        gyro_z=_opt_float(_raw_csv_get(fields, "GYRO_Z")),
        # This format carries no flight state at all. 0 is BOOT only because the
        # dataclass needs an int; the dashboard shows "NO FSM DATA" instead,
        # driven by has_fsm_data rather than by this value.
        fsm_state=0,
        has_fsm_data=False,
        has_voltage=False,
    )


def safe_filename(team_id: str) -> str:
    """Strip anything that Windows/POSIX will not accept in a file name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.\-]+", "_", (team_id or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "UNKNOWN"

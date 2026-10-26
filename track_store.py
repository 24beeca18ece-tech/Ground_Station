"""
track_store.py
==============

The session's flight path, per vehicle, as plain Python.

WHY THIS IS SEPARATE
--------------------
Two features consume the same positions -- the live map and the KML export --
and they must agree exactly: a track the operator watched must be the track
that lands in Google Earth. Keeping one store, filled once from
``Dashboard.on_packet``, is what makes that true. It also means fixes are
recorded while the map window is closed, so opening it mid-flight shows the
whole path rather than starting from the moment it opened.

No Qt here, and no I/O. Both consumers import it; neither owns it.

WHAT COUNTS AS A FIX
--------------------
:func:`is_usable_fix` is the single gate, applied once on intake. A packet
whose receiver has no lock reports lat/lon as 0.0, and a receiver with a
flickering lock reports small plausible-looking numbers near Null Island;
neither is a position. Everything downstream can then assume every stored
point is real, instead of each consumer re-deriving the rule and drifting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Vehicle keys. These are the ``payload_type`` strings, used as-is so a
#: track can never be filed under a name the rest of the app does not use.
VEHICLE_CANSAT = "CANSAT"
VEHICLE_ROCKET = "ROCKET"
VEHICLE_GENERIC = "GENERIC"

#: Default jump filter: a fix this far from the previous one, this soon after
#: it, is a GPS glitch rather than motion. 500 m in under 1 s is 1800 km/h --
#: far beyond any sounding rocket, and a common single-sample receiver fault.
DEFAULT_JUMP_M = 500.0
DEFAULT_JUMP_WINDOW_S = 1.0

#: Mean Earth radius, metres (WGS-84 authalic). Used for great-circle
#: distance and bearing, which at launch-site scale is far more accuracy
#: than the receiver itself has.
EARTH_RADIUS_M = 6371008.8


def is_usable_fix(packet: Any) -> bool:
    """True when this packet carries a position worth plotting.

    Mirrors ``TelemetryPacket.has_fix`` and then re-checks the numbers
    directly, because this gate guards two outputs that leave the
    application -- a map an operator steers by and a KML someone may fly a
    recovery team from -- and a silent change to the packet class should not
    be able to let a null island coordinate through either of them.
    """
    if packet is None:
        return False
    if not getattr(packet, "has_fix", False):
        return False
    lat = getattr(packet, "lat", float("nan"))
    lon = getattr(packet, "lon", float("nan"))
    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
        return False
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False
    if abs(lat) < 1e-9 and abs(lon) < 1e-9:          # exactly 0,0: no lock
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


@dataclass(slots=True)
class TrackPoint:
    """One recorded fix. Only what the map and the KML actually need."""

    lat: float
    lon: float
    #: GNSS altitude in metres above mean sea level, or NaN when the link
    #: does not carry one. Never the barometric altitude: that is referenced
    #: to a fixed sea-level pressure and carries a constant offset.
    alt_m: float
    #: Ground-station wall clock (``time.time()``) when the packet arrived.
    epoch: float
    mission_time_s: float
    packet_count: int
    sats: int
    fsm_state: str
    #: False when the wire format carries no GNSS altitude at all.
    has_alt: bool = True

    @property
    def has_valid_alt(self) -> bool:
        return self.has_alt and math.isfinite(self.alt_m)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2.0) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial true bearing in degrees (0-360) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


@dataclass
class Track:
    """Every fix recorded for one vehicle this session."""

    vehicle: str
    points: List[TrackPoint] = field(default_factory=list)
    #: Fixes rejected by the jump filter, counted so the KML can say so
    #: rather than quietly showing a shorter path than was flown.
    jumps_dropped: int = 0
    #: Packets whose packet_count had already been recorded.
    duplicates_dropped: int = 0

    def __len__(self) -> int:
        return len(self.points)

    @property
    def last(self) -> Optional[TrackPoint]:
        return self.points[-1] if self.points else None

    @property
    def first(self) -> Optional[TrackPoint]:
        return self.points[0] if self.points else None

    @property
    def apogee(self) -> Optional[TrackPoint]:
        """The highest fix, or None when no point carries an altitude."""
        best = None
        for point in self.points:
            if not point.has_valid_alt:
                continue
            if best is None or point.alt_m > best.alt_m:
                best = point
        return best

    @property
    def has_altitude(self) -> bool:
        """True when at least one point carries a usable GNSS altitude."""
        return any(p.has_valid_alt for p in self.points)

    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """(min_lat, min_lon, max_lat, max_lon), or None when empty."""
        if not self.points:
            return None
        lats = [p.lat for p in self.points]
        lons = [p.lon for p in self.points]
        return (min(lats), min(lons), max(lats), max(lons))


class TrackStore:
    """All vehicles' tracks for the current session.

    Intake is O(1): one append and a couple of comparisons per packet, so a
    20 Hz stream costs nothing measurable on the GUI thread.
    """

    def __init__(self, jump_m: float = DEFAULT_JUMP_M,
                 jump_window_s: float = DEFAULT_JUMP_WINDOW_S) -> None:
        self.jump_m = float(jump_m)
        self.jump_window_s = float(jump_window_s)
        self.tracks: Dict[str, Track] = {}
        #: Packets offered that carried no usable fix. Not an error: a
        #: receiver without a lock is the normal state before launch.
        self.no_fix_packets = 0

    # -- intake ------------------------------------------------------------

    def add_packet(self, packet: Any) -> Optional[TrackPoint]:
        """Record one packet's position. Returns the point, or None.

        None means the packet carried no usable fix, repeated a
        ``packet_count`` already stored, or looked like a receiver glitch.
        """
        if not is_usable_fix(packet):
            self.no_fix_packets += 1
            return None

        vehicle = getattr(packet, "payload_type", VEHICLE_GENERIC) or VEHICLE_GENERIC
        track = self.tracks.get(vehicle)
        if track is None:
            track = self.tracks[vehicle] = Track(vehicle=vehicle)

        count = int(getattr(packet, "packet_count", 0) or 0)
        last = track.last
        if last is not None and count and count == last.packet_count:
            track.duplicates_dropped += 1
            return None

        alt = getattr(packet, "nav_alt_m", float("nan"))
        point = TrackPoint(
            lat=float(packet.lat),
            lon=float(packet.lon),
            alt_m=float(alt) if isinstance(alt, (int, float)) else float("nan"),
            epoch=float(getattr(packet, "gs_recv_epoch", 0.0) or 0.0),
            mission_time_s=float(getattr(packet, "mission_time_s", 0.0) or 0.0),
            packet_count=count,
            sats=int(getattr(packet, "sats", 0) or 0),
            fsm_state=str(getattr(packet, "fsm_name", "") or ""),
            has_alt=bool(getattr(packet, "has_nav_alt", True)),
        )

        if last is not None and self._is_jump(last, point):
            track.jumps_dropped += 1
            return None

        track.points.append(point)
        return point

    def _is_jump(self, previous: TrackPoint, current: TrackPoint) -> bool:
        """True when the step from *previous* to *current* is not motion."""
        if self.jump_m <= 0:
            return False
        dt = current.epoch - previous.epoch
        if dt < 0 or dt > self.jump_window_s:
            # Too long a gap to judge: a genuine signal loss over a real
            # flight legitimately moves a long way between fixes.
            return False
        return haversine_m(previous.lat, previous.lon,
                           current.lat, current.lon) > self.jump_m

    # -- queries -----------------------------------------------------------

    def get(self, vehicle: str) -> Optional[Track]:
        return self.tracks.get(vehicle)

    def non_empty(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.points]

    @property
    def total_points(self) -> int:
        return sum(len(t.points) for t in self.tracks.values())

    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Bounding box over every vehicle, or None when nothing is stored."""
        boxes = [t.bounds() for t in self.tracks.values() if t.points]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def clear(self) -> None:
        """Forget every track. Logs on disk are untouched."""
        self.tracks.clear()
        self.no_fix_packets = 0


def decimate(points: Iterable[TrackPoint], limit: int) -> List[TrackPoint]:
    """Thin *points* to at most *limit*, keeping the first and the last.

    For drawing only. A zoomed-out polyline of 20,000 fixes costs far more to
    stroke than it shows -- most of them land on the same pixel -- but the
    endpoints matter, so they are never the ones dropped. The full list is
    what the KML export and the distance readouts use.
    """
    pts = list(points)
    if limit <= 0 or len(pts) <= limit:
        return pts
    step = len(pts) / float(limit)
    out = [pts[int(i * step)] for i in range(limit)]
    if out[-1] is not pts[-1]:
        out[-1] = pts[-1]
    return out

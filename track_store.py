"""
track_store.py
==============

The session's flight path, per vehicle, stored compactly.

WHY THIS IS SEPARATE
--------------------
Two features consume the same positions -- the live map and the KML export --
and they must agree exactly: a track the operator watched must be the track
that lands in Google Earth. Keeping one store, filled once from
``Dashboard.on_packet``, is what makes that true. It also means fixes are
recorded while the map window is closed, so opening it mid-flight shows the
whole path rather than starting from the moment it opened.

No Qt here, and no I/O. Both consumers import it; neither owns it.

STORAGE: COLUMNS, NOT OBJECTS
-----------------------------
A list of dataclass objects measured **363 bytes per fix**, which is 70 MB
for a 200,000-point track and over 100 MB at the cap. The same data as
parallel ``numpy`` columns is 44 bytes a fix -- an eight-fold reduction --
and, more importantly, lets the map project a whole track with one
vectorised expression instead of a Python loop per redraw.

Columns grow geometrically and are sliced with ``_n``, so an append is
amortised O(1) and never copies on the common path.

WHAT COUNTS AS A FIX
--------------------
:func:`is_usable_fix` is the single gate, applied once on intake. A packet
whose receiver has no lock reports lat/lon as 0.0, and a receiver with a
flickering lock reports small plausible-looking numbers near Null Island;
neither is a position. Everything downstream can then assume every stored
point is real, instead of each consumer re-deriving the rule and drifting.

BOUNDED, WITHOUT LOSING THE FLIGHT
----------------------------------
Two limits, both chosen so that what is dropped is what carries no
information:

* **Stationary thinning.** A vehicle sitting on the pad for twenty minutes
  produces 24,000 fixes of the same spot. Once it has not moved
  :data:`STATIONARY_M` for :data:`STATIONARY_AFTER_S`, fixes are stored at
  most every :data:`STATIONARY_EVERY_S`. Any real movement resumes full
  rate immediately, so nothing in flight is thinned.
* **Hard cap.** At :data:`MAX_POINTS_PER_VEHICLE` the older half of the
  track is decimated 2:1 rather than the track being allowed to grow. The
  endpoints are always kept. This degrades resolution of the distant past
  instead of failing, which is the right trade when memory is the
  alternative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

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

#: Stationary thinning. Under this much movement from the last stored fix...
STATIONARY_M = 1.0
#: ...for this long...
STATIONARY_AFTER_S = 60.0
#: ...store at most one fix this often.
STATIONARY_EVERY_S = 10.0

#: Hard cap per vehicle. 300,000 fixes is over four hours at 20 Hz; past it
#: the older half is thinned 2:1 rather than the store growing.
MAX_POINTS_PER_VEHICLE = 300_000

#: Initial column capacity. Grown geometrically from here.
_INITIAL_CAPACITY = 4096

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
    """One fix, as a value. Built on demand from the columns, never stored.

    The store keeps columns; this exists so callers that want *one* point --
    launch, apogee, the latest -- read named fields instead of indices.
    """

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


class Track:
    """Every fix recorded for one vehicle, as parallel numpy columns.

    ``revision`` increments on every change. Consumers that cache derived
    work -- the map caches projected pixels -- compare it instead of
    re-deriving, which is what keeps redraw cost independent of track length.
    """

    __slots__ = ("vehicle", "_n", "_cap", "lat", "lon", "alt", "epoch",
                 "mission", "count", "sats", "fsm", "_fsm_names",
                 "jumps_dropped", "duplicates_dropped", "stationary_dropped",
                 "thinned", "revision", "has_alt_any")

    def __init__(self, vehicle: str, capacity: int = _INITIAL_CAPACITY) -> None:
        self.vehicle = vehicle
        self._n = 0
        self._cap = int(capacity)
        self.lat = np.empty(self._cap, dtype=np.float64)
        self.lon = np.empty(self._cap, dtype=np.float64)
        self.alt = np.empty(self._cap, dtype=np.float32)
        self.epoch = np.empty(self._cap, dtype=np.float64)
        self.mission = np.empty(self._cap, dtype=np.float32)
        self.count = np.empty(self._cap, dtype=np.int64)
        self.sats = np.empty(self._cap, dtype=np.int16)
        # FSM names repeat endlessly; store a small index into a name table
        # rather than a Python string per fix.
        self.fsm = np.empty(self._cap, dtype=np.int16)
        self._fsm_names: List[str] = [""]
        self.jumps_dropped = 0
        self.duplicates_dropped = 0
        self.stationary_dropped = 0
        self.thinned = 0
        self.revision = 0
        self.has_alt_any = False

    # -- size --------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def _grow(self) -> None:
        self._cap = max(_INITIAL_CAPACITY, self._cap * 2)
        for name in ("lat", "lon", "alt", "epoch", "mission", "count",
                     "sats", "fsm"):
            column = getattr(self, name)
            grown = np.empty(self._cap, dtype=column.dtype)
            grown[:self._n] = column[:self._n]
            setattr(self, name, grown)

    def _fsm_index(self, name: str) -> int:
        try:
            return self._fsm_names.index(name)
        except ValueError:
            self._fsm_names.append(name)
            return len(self._fsm_names) - 1

    def append(self, lat: float, lon: float, alt: float, epoch: float,
               mission: float, count: int, sats: int, fsm: str,
               has_alt: bool) -> None:
        if self._n >= self._cap:
            self._grow()
        i = self._n
        self.lat[i] = lat
        self.lon[i] = lon
        self.alt[i] = alt if has_alt else float("nan")
        self.epoch[i] = epoch
        self.mission[i] = mission
        self.count[i] = count
        self.sats[i] = sats
        self.fsm[i] = self._fsm_index(fsm)
        self._n = i + 1
        if has_alt and math.isfinite(alt):
            self.has_alt_any = True
        self.revision += 1
        if self._n > MAX_POINTS_PER_VEHICLE:
            self._thin_older_half()

    def _thin_older_half(self) -> None:
        """Halve the resolution of the older half, keeping both endpoints.

        Degrading the distant past is the right trade against refusing to
        record the present. The count is reported so the KML can say so.
        """
        half = self._n // 2
        keep = np.concatenate([np.arange(0, half, 2),
                               np.arange(half, self._n)])
        for name in ("lat", "lon", "alt", "epoch", "mission", "count",
                     "sats", "fsm"):
            column = getattr(self, name)
            column[:len(keep)] = column[keep]
        self.thinned += self._n - len(keep)
        self._n = len(keep)
        self.revision += 1

    @classmethod
    def from_points(cls, vehicle: str,
                    points: Iterable[TrackPoint]) -> "Track":
        """Build a Track from TrackPoint values.

        For callers that already hold points -- the CSV-to-KML path and the
        tests. The live path appends straight into the columns.
        """
        track = cls(vehicle)
        for point in points:
            track.append(point.lat, point.lon, point.alt_m, point.epoch,
                         point.mission_time_s, point.packet_count,
                         point.sats, point.fsm_state, point.has_alt)
        return track

    # -- views -------------------------------------------------------------

    @property
    def lats(self) -> np.ndarray:
        """A VIEW of the stored latitudes. Do not write to it."""
        return self.lat[:self._n]

    @property
    def lons(self) -> np.ndarray:
        return self.lon[:self._n]

    @property
    def alts(self) -> np.ndarray:
        return self.alt[:self._n]

    @property
    def epochs(self) -> np.ndarray:
        return self.epoch[:self._n]

    def point(self, index: int) -> Optional[TrackPoint]:
        """One fix as a named value. Built here, not stored."""
        if not -self._n <= index < self._n:
            return None
        if index < 0:
            index += self._n
        alt = float(self.alt[index])
        return TrackPoint(
            lat=float(self.lat[index]), lon=float(self.lon[index]),
            alt_m=alt, epoch=float(self.epoch[index]),
            mission_time_s=float(self.mission[index]),
            packet_count=int(self.count[index]), sats=int(self.sats[index]),
            fsm_state=self._fsm_names[int(self.fsm[index])],
            has_alt=math.isfinite(alt))

    @property
    def points(self) -> List[TrackPoint]:
        """Every fix as TrackPoint objects.

        O(n) and allocating: this is for the KML export, which runs once.
        The map must use the columns, never this.
        """
        return [self.point(i) for i in range(self._n)]

    @property
    def last(self) -> Optional[TrackPoint]:
        return self.point(self._n - 1) if self._n else None

    @property
    def first(self) -> Optional[TrackPoint]:
        return self.point(0) if self._n else None

    @property
    def apogee(self) -> Optional[TrackPoint]:
        """The highest fix, or None when no point carries an altitude."""
        if not self._n or not self.has_alt_any:
            return None
        alts = self.alts
        if not np.any(np.isfinite(alts)):
            return None
        return self.point(int(np.nanargmax(alts)))

    @property
    def has_altitude(self) -> bool:
        """True when at least one point carries a usable GNSS altitude."""
        return bool(self.has_alt_any)

    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """(min_lat, min_lon, max_lat, max_lon), or None when empty."""
        if not self._n:
            return None
        return (float(self.lats.min()), float(self.lons.min()),
                float(self.lats.max()), float(self.lons.max()))


class TrackStore:
    """All vehicles' tracks for the current session.

    Intake is O(1) amortised: a handful of array writes per packet, so a
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
        #: vehicle -> (epoch, lat, lon) of the last fix that actually moved.
        self._moved_at: Dict[str, Tuple[float, float, float]] = {}

    # -- intake ------------------------------------------------------------

    def add_packet(self, packet: Any) -> bool:
        """Record one packet's position. True when a point was stored.

        False means the packet carried no usable fix, repeated a
        ``packet_count`` already stored, looked like a receiver glitch, or
        was thinned because the vehicle is sitting still.
        """
        if not is_usable_fix(packet):
            self.no_fix_packets += 1
            return False

        vehicle = getattr(packet, "payload_type", VEHICLE_GENERIC) or VEHICLE_GENERIC
        track = self.tracks.get(vehicle)
        if track is None:
            track = self.tracks[vehicle] = Track(vehicle)

        count = int(getattr(packet, "packet_count", 0) or 0)
        n = len(track)
        if n and count and count == int(track.count[n - 1]):
            track.duplicates_dropped += 1
            return False

        lat = float(packet.lat)
        lon = float(packet.lon)
        epoch = float(getattr(packet, "gs_recv_epoch", 0.0) or 0.0)

        if n:
            prev_lat = float(track.lat[n - 1])
            prev_lon = float(track.lon[n - 1])
            prev_epoch = float(track.epoch[n - 1])
            moved = haversine_m(prev_lat, prev_lon, lat, lon)

            dt = epoch - prev_epoch
            if self.jump_m > 0 and 0 <= dt <= self.jump_window_s \
                    and moved > self.jump_m:
                track.jumps_dropped += 1
                return False

            if self._is_stationary_duplicate(vehicle, lat, lon, epoch, moved):
                track.stationary_dropped += 1
                return False
        self._moved_at.setdefault(vehicle, (epoch, lat, lon))

        alt = getattr(packet, "nav_alt_m", float("nan"))
        has_alt = bool(getattr(packet, "has_nav_alt", True))
        track.append(
            lat=lat, lon=lon,
            alt=float(alt) if isinstance(alt, (int, float)) else float("nan"),
            epoch=epoch,
            mission=float(getattr(packet, "mission_time_s", 0.0) or 0.0),
            count=count, sats=int(getattr(packet, "sats", 0) or 0),
            fsm=str(getattr(packet, "fsm_name", "") or ""), has_alt=has_alt)
        return True

    def _is_stationary_duplicate(self, vehicle: str, lat: float, lon: float,
                                 epoch: float, moved_m: float) -> bool:
        """True when this fix is the pad, again, and need not be stored.

        A vehicle waiting on the pad emits the same coordinate thousands of
        times. Those carry no information the first one did not, and the
        moment it actually moves, full rate resumes.
        """
        anchor = self._moved_at.get(vehicle)
        if anchor is None:
            self._moved_at[vehicle] = (epoch, lat, lon)
            return False
        anchor_epoch, anchor_lat, anchor_lon = anchor
        if haversine_m(anchor_lat, anchor_lon, lat, lon) > STATIONARY_M:
            self._moved_at[vehicle] = (epoch, lat, lon)   # it moved
            return False
        still_for = epoch - anchor_epoch
        if still_for < STATIONARY_AFTER_S:
            return False
        track = self.tracks[vehicle]
        last_stored = float(track.epoch[len(track) - 1]) if len(track) else 0.0
        return (epoch - last_stored) < STATIONARY_EVERY_S

    # -- queries -----------------------------------------------------------

    def get(self, vehicle: str) -> Optional[Track]:
        return self.tracks.get(vehicle)

    def non_empty(self) -> List[Track]:
        return [t for t in self.tracks.values() if len(t)]

    @property
    def total_points(self) -> int:
        return sum(len(t) for t in self.tracks.values())

    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Bounding box over every vehicle, or None when nothing is stored."""
        boxes = [t.bounds() for t in self.tracks.values() if len(t)]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def clear(self) -> None:
        """Forget every track. Logs on disk are untouched."""
        self.tracks.clear()
        self.no_fix_packets = 0
        self._moved_at.clear()


def decimate(points: Sequence[TrackPoint], limit: int) -> List[TrackPoint]:
    """Thin *points* to at most *limit*, keeping the first and the last.

    Kept for the KML path and for callers holding TrackPoint lists. The map
    does not use it: it thins in pixel space on the projected columns.
    """
    pts = list(points)
    if limit <= 0 or len(pts) <= limit:
        return pts
    step = len(pts) / float(limit)
    out = [pts[int(i * step)] for i in range(limit)]
    if out[-1] is not pts[-1]:
        out[-1] = pts[-1]
    return out

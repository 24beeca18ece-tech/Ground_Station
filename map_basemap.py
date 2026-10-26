"""
map_basemap.py
==============

What the map draws when there are no tiles: a graticule, a north arrow and
range rings, plus the two offline base layers that need no server at all —
an OSM vector pack rendered by this application, and a georeferenced image
the operator supplies.

WHY
---
The launch site has no internet, and a boundary layer from a WMS is not a
basemap. With nothing configured the map must still be worth looking at and
must look deliberate rather than broken: a labelled lat/lon grid with a
scale bar and range rings tells an operator where the vehicle is relative to
them, which is the question they actually have.

Nothing here does I/O on the GUI thread except :func:`load_vector_pack` and
:func:`load_image_overlay`, which are called when a pack is selected, not
while painting. Rendering is from caches keyed by zoom.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPolygonF

#: Range rings drawn around the ground station, metres. Chosen for a
#: 2 km-range link: the outer ring is the edge of the budget.
RANGE_RINGS_M = (250.0, 500.0, 1000.0, 2000.0)

#: Graticule steps in degrees, coarse to fine. The renderer picks the
#: finest one that still leaves the labels legible.
GRATICULE_STEPS = (45.0, 30.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25,
                   1 / 12.0, 1 / 30.0,
                   1 / 60.0, 1 / 120.0, 1 / 300.0, 1 / 600.0, 1 / 1200.0,
                   1 / 3600.0)

#: Minimum pixels between graticule lines; below this the labels collide.
MIN_GRATICULE_PX = 90.0

GRID_BG = "#eef1f5"
GRID_LINE = "#c2cbd8"
GRID_LABEL = "#5b6a80"
RING_COLOUR = "#8fa0b8"

#: Feature colours for the vector pack. Muted on purpose: the tracks are
#: what the operator is reading, and the basemap must not compete.
VECTOR_COLOURS: Dict[str, str] = {
    "water": "#b8d4ea",
    "forest": "#cfe0c4",
    "scrub": "#d9e3cc",
    "grass": "#dde8d2",
    "farmland": "#e6e5cf",
    "bare": "#e4e0d8",
    "residential": "#e8e4e0",
    "building": "#d8d2cc",
    "rail": "#9aa3ad",
}

#: Road casing widths in pixels by class, and the zoom each appears at.
ROAD_STYLE: Dict[str, Tuple[float, str, int]] = {
    # class:           (width, colour, min zoom)
    "motorway":        (5.0, "#e8a33d", 8),
    "trunk":           (4.4, "#e8b85c", 9),
    "primary":         (4.0, "#efc87a", 10),
    "secondary":       (3.2, "#f0d9a0", 11),
    "tertiary":        (2.6, "#ffffff", 12),
    "residential":     (2.2, "#ffffff", 13),
    "unclassified":    (2.0, "#ffffff", 13),
    "service":         (1.4, "#ffffff", 15),
    "track":           (1.6, "#cbb89a", 13),
    "path":            (1.2, "#c4b2a0", 14),
    # Administrative boundaries, for the overview pack. Visible at every
    # zoom: unlike a service road, a state line is what an overview is FOR.
    "boundary":        (1.0, "#b0a79c", 0),
}

#: Version written into a pack, so a future format change is detectable.
PACK_FORMAT = "gcs-osm-pack/1"


# ---------------------------------------------------------------------------
# Grid, north arrow, range rings
# ---------------------------------------------------------------------------

def choose_graticule_step(span_deg: float, span_px: float) -> float:
    """The finest step that still leaves MIN_GRATICULE_PX between lines."""
    if span_px <= 0 or span_deg <= 0:
        return GRATICULE_STEPS[0]
    for step in GRATICULE_STEPS:
        if step / span_deg * span_px < MIN_GRATICULE_PX:
            # One finer than this would collide, so the previous one is it.
            index = GRATICULE_STEPS.index(step)
            return GRATICULE_STEPS[max(0, index - 1)]
    return GRATICULE_STEPS[-1]


def format_lat(deg: float) -> str:
    return _format_dm(abs(deg), "N" if deg >= 0 else "S")


def format_lon(deg: float) -> str:
    return _format_dm(abs(deg), "E" if deg >= 0 else "W")


def _format_dm(value: float, hemisphere: str) -> str:
    """Degrees and minutes -- what a map margin shows, not raw decimals."""
    degrees = int(value)
    minutes = (value - degrees) * 60.0
    if minutes < 0.05:
        return "%d°%s" % (degrees, hemisphere)
    if minutes >= 1.0:
        return "%d°%04.1f'%s" % (degrees, minutes, hemisphere)
    return "%d°%04.2f'%s" % (degrees, minutes, hemisphere)


def draw_graticule(painter: QPainter, width: int, height: int,
                   top_left: Tuple[float, float],
                   bottom_right: Tuple[float, float],
                   to_pixel) -> None:
    """A labelled lat/lon grid over a plain ground.

    *to_pixel* maps (lon, lat) to a QPointF, supplied by the canvas so this
    stays free of projection state.
    """
    painter.fillRect(0, 0, width, height, QColor(GRID_BG))
    west, north = top_left
    east, south = bottom_right
    if not all(math.isfinite(v) for v in (west, north, east, south)):
        return

    lon_step = choose_graticule_step(abs(east - west), width)
    lat_step = choose_graticule_step(abs(north - south), height)

    pen = QPen(QColor(GRID_LINE), 1)
    painter.setPen(pen)
    font = QFont()
    font.setPointSize(8)
    painter.setFont(font)

    start = math.floor(west / lon_step) * lon_step
    value = start
    while value <= east + lon_step:
        point = to_pixel(value, north)
        x = int(point.x())
        if -50 <= x <= width + 50:
            painter.setPen(QPen(QColor(GRID_LINE), 1))
            painter.drawLine(x, 0, x, height)
            painter.setPen(QColor(GRID_LABEL))
            painter.drawText(x + 3, 12, format_lon(value))
        value += lon_step

    start = math.floor(south / lat_step) * lat_step
    value = start
    while value <= north + lat_step:
        point = to_pixel(west, value)
        y = int(point.y())
        if -50 <= y <= height + 50:
            painter.setPen(QPen(QColor(GRID_LINE), 1))
            painter.drawLine(0, y, width, y)
            painter.setPen(QColor(GRID_LABEL))
            painter.drawText(3, y - 3, format_lat(value))
        value += lat_step


def draw_north_arrow(painter: QPainter, width: int, height: int) -> None:
    """A north arrow. Web Mercator is north-up everywhere, so it is fixed."""
    x, y = width - 34, 34
    painter.setPen(QPen(QColor(GRID_LABEL), 1.6))
    painter.setBrush(QColor(GRID_LABEL))
    arrow = QPolygonF([QPointF(x, y - 16), QPointF(x + 6, y + 6),
                       QPointF(x, y + 1), QPointF(x - 6, y + 6)])
    painter.drawPolygon(arrow)
    font = QFont()
    font.setPointSize(8)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(x - 5, y + 20, "N")


def draw_range_rings(painter: QPainter, lat: float, lon: float,
                     to_pixel, metres_per_px: float,
                     rings: Sequence[float] = RANGE_RINGS_M) -> None:
    """Range rings around the ground station, labelled in metres.

    The operator's real question on the pad is "how far away is it, and
    which way", and a ring answers the first at a glance.
    """
    if metres_per_px <= 0 or not math.isfinite(metres_per_px):
        return
    centre = to_pixel(lon, lat)
    pen = QPen(QColor(RING_COLOUR), 1.0, Qt.DashLine)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    font = QFont()
    font.setPointSize(7)
    painter.setFont(font)
    for metres in rings:
        radius = metres / metres_per_px
        if radius < 18 or radius > 6000:
            continue           # too small to read, or far off the screen
        painter.setPen(pen)
        painter.drawEllipse(centre, radius, radius)
        painter.setPen(QColor(GRID_LABEL))
        label = "%d m" % metres if metres < 1000 else "%g km" % (metres / 1000.0)
        painter.drawText(QPointF(centre.x() + 3, centre.y() - radius - 3), label)


# ---------------------------------------------------------------------------
# Vector packs
# ---------------------------------------------------------------------------

class VectorPack:
    """An OSM-derived vector basemap, rendered by this application.

    Geometry is held as numpy arrays of lon/lat and projected per zoom into
    a cache, the same way tracks are: no Python loop over features on the
    paint path.
    """

    def __init__(self, data: Dict[str, Any], path: str = "") -> None:
        self.path = path
        self.name = str(data.get("name") or os.path.basename(path) or "pack")
        self.attribution = str(data.get("attribution")
                               or "© OpenStreetMap contributors, ODbL")
        bounds = data.get("bounds") or {}
        self.bounds = (float(bounds.get("west", -180.0)),
                       float(bounds.get("south", -85.0)),
                       float(bounds.get("east", 180.0)),
                       float(bounds.get("north", 85.0)))
        #: (kind, class, numpy Nx2 lon/lat) for lines; polygons likewise.
        self.lines: List[Tuple[str, np.ndarray]] = []
        self.areas: List[Tuple[str, np.ndarray]] = []
        self.places: List[Tuple[str, float, float, int]] = []
        self._load(data)
        #: zoom -> list of (style, QPolygonF in WORLD pixels)
        self._cache: Dict[int, Tuple[List, List]] = {}

    def _load(self, data: Dict[str, Any]) -> None:
        for feature in data.get("lines", ()):
            coords = feature.get("c")
            if not coords or len(coords) < 2:
                continue
            self.lines.append((str(feature.get("k", "unclassified")),
                               np.asarray(coords, dtype=np.float64)))
        for feature in data.get("areas", ()):
            coords = feature.get("c")
            if not coords or len(coords) < 3:
                continue
            self.areas.append((str(feature.get("k", "bare")),
                               np.asarray(coords, dtype=np.float64)))
        for place in data.get("places", ()):
            try:
                self.places.append((str(place.get("n", "")),
                                    float(place["lon"]), float(place["lat"]),
                                    int(place.get("r", 3))))
            except (KeyError, TypeError, ValueError):
                continue

    @property
    def feature_count(self) -> int:
        return len(self.lines) + len(self.areas) + len(self.places)

    def _project(self, zoom: int, project) -> Tuple[List, List]:
        """World-pixel polygons for this zoom, built once and kept.

        *project* is the canvas's vectorised lon/lat -> world pixel.
        """
        cached = self._cache.get(zoom)
        if cached is not None:
            return cached
        areas = []
        for kind, coords in self.areas:
            xs, ys = project(coords[:, 0], coords[:, 1], zoom)
            areas.append((kind, np.column_stack((xs, ys))))
        lines = []
        for kind, coords in self.lines:
            style = ROAD_STYLE.get(kind)
            if style is not None and zoom < style[2]:
                continue                   # too minor to show at this zoom
            xs, ys = project(coords[:, 0], coords[:, 1], zoom)
            lines.append((kind, np.column_stack((xs, ys))))
        # Only the most recent few zooms are worth keeping: panning does not
        # change zoom, and a long session visits a handful of levels.
        if len(self._cache) > 4:
            self._cache.pop(next(iter(self._cache)))
        self._cache[zoom] = (areas, lines)
        return self._cache[zoom]

    def draw(self, painter: QPainter, zoom: int, origin: Tuple[float, float],
             width: int, height: int, project) -> None:
        """Draw the visible part. Areas first, then roads, then labels."""
        areas, lines = self._project(zoom, project)
        ox, oy = origin
        margin = 128.0

        painter.setPen(Qt.NoPen)
        for kind, xy in areas:
            sx = xy[:, 0] - ox
            sy = xy[:, 1] - oy
            if (sx.max() < -margin or sx.min() > width + margin
                    or sy.max() < -margin or sy.min() > height + margin):
                continue
            painter.setBrush(QColor(VECTOR_COLOURS.get(kind, "#e4e0d8")))
            painter.drawPolygon(QPolygonF(
                [QPointF(float(a), float(b)) for a, b in zip(sx, sy)]))

        painter.setBrush(Qt.NoBrush)
        for kind, xy in lines:
            sx = xy[:, 0] - ox
            sy = xy[:, 1] - oy
            if (sx.max() < -margin or sx.min() > width + margin
                    or sy.max() < -margin or sy.min() > height + margin):
                continue
            if kind == "rail":
                pen = QPen(QColor(VECTOR_COLOURS["rail"]), 1.6, Qt.DashLine)
            else:
                style = ROAD_STYLE.get(kind, (1.6, "#ffffff", 0))
                pen = QPen(QColor(style[1]), style[0])
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline(QPolygonF(
                [QPointF(float(a), float(b)) for a, b in zip(sx, sy)]))

        if self.places:
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)
            painter.setPen(QColor("#4a5668"))
            for name, lon, lat, rank in self.places:
                if rank > 3 and zoom < 12:
                    continue
                xs, ys = project(np.array([lon]), np.array([lat]), zoom)
                x, y = float(xs[0]) - ox, float(ys[0]) - oy
                if -20 <= x <= width + 20 and -20 <= y <= height + 20:
                    painter.drawText(QPointF(x + 4, y + 4), name)


def load_vector_pack(path: str) -> Optional[VectorPack]:
    """Read a .osmpack.json. Returns None on any problem, never raises."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return VectorPack(data, path)
    except Exception:                                  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Image overlays
# ---------------------------------------------------------------------------

class ImageOverlay:
    """A georeferenced image the operator supplies, as the base layer.

    The sidecar gives WGS-84 bounds and the image is stretched to them,
    which treats it as Web Mercator aligned. Over a launch site that is a
    good approximation: the Mercator y-scale changes by about 0.08% over
    10 km of latitude at 33°N, which is under 8 m across the whole image
    and well inside what a phone-camera-grade export is worth.
    """

    def __init__(self, image_path: str, bounds: Dict[str, float],
                 name: str = "") -> None:
        self.path = image_path
        self.name = name or os.path.basename(image_path)
        self.west = float(bounds["west"])
        self.south = float(bounds["south"])
        self.east = float(bounds["east"])
        self.north = float(bounds["north"])
        self.attribution = str(bounds.get("attribution")
                               or "Operator-supplied image")
        self.opacity = float(bounds.get("opacity", 1.0))
        #: Filled by the loader thread; None until then.
        self.image = None
        self.error = ""

    @property
    def is_valid(self) -> bool:
        return (self.west < self.east and self.south < self.north
                and all(math.isfinite(v) for v in
                        (self.west, self.south, self.east, self.north)))

    def draw(self, painter: QPainter, to_pixel, opacity: float = 1.0) -> None:
        if self.image is None or self.image.isNull():
            return
        top_left = to_pixel(self.west, self.north)
        bottom_right = to_pixel(self.east, self.south)
        target = QRectF(top_left, bottom_right).normalized()
        if target.width() < 1 or target.height() < 1:
            return
        previous = painter.opacity()
        painter.setOpacity(max(0.0, min(1.0, opacity)))
        painter.drawImage(target, self.image)
        painter.setOpacity(previous)


def load_image_overlay(sidecar_path: str) -> Optional[ImageOverlay]:
    """Read NAME.json describing NAME.png/jpg. None on any problem."""
    try:
        with open(sidecar_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    base = os.path.splitext(sidecar_path)[0]
    image_path = ""
    for extension in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        candidate = base + extension
        if os.path.isfile(candidate):
            image_path = candidate
            break
    if not image_path:
        named = data.get("image")
        if named:
            candidate = os.path.join(os.path.dirname(sidecar_path), str(named))
            if os.path.isfile(candidate):
                image_path = candidate
    if not image_path:
        return None
    try:
        overlay = ImageOverlay(image_path, data,
                               name=str(data.get("name", "")))
    except (KeyError, TypeError, ValueError):
        return None
    return overlay if overlay.is_valid else None

"""
map_window.py
=============

A live slippy map of where the vehicles actually are, drawn with QPainter.

WHY NOT A WEB VIEW
------------------
QtWebEngine would bring a browser engine, a second process and ~100 MB of
dependency for what is a tiled image and some polylines. This draws the same
thing with ``QPainter`` over 256 px tiles, so the application stays one
process with the dependencies it already had.

OFFLINE IS THE BASE CASE
------------------------
The launch site has no internet. MBTiles files under ``assets/maps/`` are read
straight from disk with the standard library's ``sqlite3`` and are the default
source whenever one exists; online sources are the convenience, not the other
way round. A tile that is missing -- no file, no network, nothing cached --
is drawn as a grey square with its zoom level on it, so a hole in the map
reads as a hole rather than as the vehicle having flown over the sea.

THREADING
---------
The GUI thread never blocks on a tile:

* network tiles go through ``QNetworkAccessManager``, which is asynchronous
  on the GUI thread and calls back when the reply lands;
* MBTiles reads run on ``QThreadPool`` as ``QRunnable``, one sqlite
  connection per worker (sqlite objects are not shareable across threads);
* decoded tiles land in a bounded LRU, so a long session cannot grow memory
  without limit.

Packet intake is O(1) -- one append -- and redraws are coalesced to
:data:`REDRAW_HZ`, so a 20 Hz stream cannot make the map the reason the
dashboard stutters.

NO GOOGLE
---------
No Google Maps or Google tile server is used anywhere here. Google Earth is
only the viewer for the separately exported KML.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from PyQt5.QtCore import (QObject, QPoint, QPointF, QRect, QRunnable, Qt,
                          QThreadPool, QTimer, QUrl, pyqtSignal)
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PyQt5.QtNetwork import (QNetworkAccessManager, QNetworkDiskCache,
                             QNetworkRequest)
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                             QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                             QWidget)

from track_store import (TrackStore, Track, TrackPoint, bearing_deg,
                         decimate, haversine_m)

# --- palette, matched to the rest of the application ------------------------
COL_BG = "#d7dee8"
COL_PANEL = "#f2f5f9"
COL_BORDER = "#7a8ba4"
COL_TEXT = "#0d1520"
COL_DIM = "#41506a"
COL_MISSING = "#c6ced9"
COL_ALERT = "#c0182b"
COL_OK = "#0d7a3d"

#: Vehicle colours, matching the KML export so the map and Google Earth agree.
VEHICLE_COLOURS = {"CANSAT": "#00b4d8", "ROCKET": "#ff1f2e",
                   "GENERIC": "#ffb000"}

TILE_PX = 256
MIN_ZOOM, MAX_ZOOM = 1, 19

#: Repaints per second, coalesced. The map is a situational display, not an
#: instrument: five frames a second is plenty to follow a descent, and it
#: keeps the strip charts' budget intact at 20 Hz packet rates.
REDRAW_HZ = 5

#: Decoded tiles held in memory. 400 x 256 x 256 x 4 bytes is about 100 MB
#: worst case, which is the ceiling, not the normal case.
TILE_CACHE_MAX = 400

#: Concurrent network tile requests. Deliberately small: the tile policies of
#: free providers ask for exactly this restraint, and a launch-site laptop on
#: a phone hotspot does not benefit from more.
MAX_INFLIGHT = 4

#: A fix older than this makes the vehicle label turn red.
STALE_FIX_S = 5.0

#: Points drawn per track. The full list stays in the store for the KML and
#: the distance readouts; this is only what gets stroked.
MAX_DRAWN_POINTS = 2000

#: Where sources and the ground-station position live. Both are gitignored:
#: the first may carry an API key's environment-variable name and local file
#: paths, the second is a physical location.
SOURCES_FILE = "map_sources.json"
SETTINGS_FILE = "map_settings.json"
MAPS_DIR = os.path.join("assets", "maps")

#: Sent on every outbound tile request. Providers' usage policies require an
#: application to identify itself rather than impersonate a browser.
USER_AGENT = ("CU-Jammu-Astro-GroundStation/1.0 "
              "(+https://github.com/24beeca18ece-tech/Ground_Station)")

#: Used when map_sources.json is missing or unreadable. Offline first.
BUILTIN_SOURCES: List[Dict[str, Any]] = [
    {
        "name": "Offline (MBTiles)",
        "type": "mbtiles",
        "path": MAPS_DIR,
        "attribution": "Offline tile pack",
        "min_zoom": 0,
        "max_zoom": 19,
    },
    {
        "name": "ISRO Bhuvan (admin basemap)",
        "type": "wms",
        "url": "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms",
        "layers": "basemap:admin_group_ntl",
        "attribution": "© ISRO / NRSC Bhuvan",
        "min_zoom": 4,
        "max_zoom": 16,
    },
    {
        "name": "OpenStreetMap",
        "type": "xyz",
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
        "min_zoom": 1,
        "max_zoom": 19,
        "bulk_download": False,
    },
]


# ---------------------------------------------------------------------------
# Web Mercator (EPSG:3857)
# ---------------------------------------------------------------------------

#: Metres per pixel at zoom 0 on the equator, for a 256 px tile.
EQUATOR_M = 40075016.68557849
ORIGIN_SHIFT = EQUATOR_M / 2.0          # 20037508.34..., the 3857 half-extent


def clamp_lat(lat: float) -> float:
    """Web Mercator is undefined at the poles; 85.0511 is where it is cut."""
    return max(-85.05112878, min(85.05112878, lat))


def lonlat_to_world(lon: float, lat: float, zoom: int) -> Tuple[float, float]:
    """Longitude/latitude to world pixel coordinates at *zoom*."""
    size = TILE_PX * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * size
    sin_lat = math.sin(math.radians(clamp_lat(lat)))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * size
    return x, y


def world_to_lonlat(x: float, y: float, zoom: int) -> Tuple[float, float]:
    """Inverse of :func:`lonlat_to_world`."""
    size = TILE_PX * (2 ** zoom)
    lon = x / size * 360.0 - 180.0
    n = math.pi * (1 - 2 * y / size)
    lat = math.degrees(math.atan(math.sinh(n)))
    return lon, lat


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    """The tile containing this coordinate."""
    x, y = lonlat_to_world(lon, lat, zoom)
    return int(x // TILE_PX), int(y // TILE_PX)


def tile_bounds_3857(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    """A tile's bounding box in EPSG:3857 metres, as WMS GetMap wants it."""
    span = EQUATOR_M / (2 ** zoom)
    west = -ORIGIN_SHIFT + x * span
    east = west + span
    north = ORIGIN_SHIFT - y * span
    south = north - span
    return west, south, east, north


def metres_per_pixel(lat: float, zoom: int) -> float:
    return EQUATOR_M * math.cos(math.radians(clamp_lat(lat))) / (TILE_PX * 2 ** zoom)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@dataclass
class MapSource:
    name: str
    type: str
    url: str = ""
    path: str = ""
    layers: str = ""
    attribution: str = ""
    min_zoom: int = 0
    max_zoom: int = 19
    #: Name of an environment variable holding a key or token. The key itself
    #: is never stored in the repository or in the JSON.
    key_env: str = ""
    key_param: str = "key"
    #: False marks a provider whose terms forbid bulk tile download, which is
    #: what tools/fetch_tiles.py refuses to do.
    bulk_download: bool = False

    @property
    def api_key(self) -> str:
        return os.environ.get(self.key_env, "") if self.key_env else ""

    @property
    def needs_key(self) -> bool:
        return bool(self.key_env) and not self.api_key


def load_sources(base_dir: str = ".") -> Tuple[List[MapSource], str]:
    """Load ``map_sources.json``; fall back to the built-in list.

    Returns ``(sources, note)`` where *note* is a single line for the event
    log -- empty when the file loaded cleanly. A bad file must not stop the
    map from opening: the operator needs *a* map more than their preferred
    one.
    """
    path = os.path.join(base_dir, SOURCES_FILE)
    note = ""
    raw: Any = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception as exc:
            note = ("%s could not be read (%s: %s); using the built-in map "
                    "sources." % (SOURCES_FILE, type(exc).__name__, exc))
            raw = None
    else:
        note = "%s not found; using the built-in map sources." % SOURCES_FILE

    entries = raw.get("sources") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        if raw is not None and not note:
            note = ("%s has no usable 'sources' list; using the built-in map "
                    "sources." % SOURCES_FILE)
        entries = BUILTIN_SOURCES

    sources: List[MapSource] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        if str(entry.get("type", "")).lower() not in ("mbtiles", "xyz", "wms"):
            continue
        sources.append(MapSource(
            name=str(entry["name"]),
            type=str(entry["type"]).lower(),
            url=str(entry.get("url", "")),
            path=str(entry.get("path", "")),
            layers=str(entry.get("layers", "")),
            attribution=str(entry.get("attribution", "")),
            min_zoom=int(entry.get("min_zoom", 0)),
            max_zoom=int(entry.get("max_zoom", 19)),
            key_env=str(entry.get("key_env", "")),
            key_param=str(entry.get("key_param", "key")),
            bulk_download=bool(entry.get("bulk_download", False)),
        ))
    if not sources:
        sources = [MapSource(**{k: v for k, v in BUILTIN_SOURCES[0].items()})]
    return sources, note


def find_mbtiles(directory: str) -> List[str]:
    """Every ``.mbtiles`` file in *directory*, sorted. Missing dir is fine."""
    try:
        return sorted(os.path.join(directory, n)
                      for n in os.listdir(directory)
                      if n.lower().endswith(".mbtiles"))
    except OSError:
        return []


# ---------------------------------------------------------------------------
# MBTiles
# ---------------------------------------------------------------------------

def read_mbtiles_tile(db_path: str, z: int, x: int, y: int) -> Optional[bytes]:
    """One tile's bytes from an MBTiles file, or None.

    MBTiles stores rows in TMS order, whose origin is the BOTTOM-left, while
    slippy tile y counts from the top. The flip is
    ``tms_y = 2**z - 1 - y`` and getting it wrong yields a map that is
    vertically mirrored but otherwise plausible -- which is exactly the kind
    of bug that survives a casual look, so it is done here, once.
    """
    try:
        conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True,
                               timeout=2.0)
    except sqlite3.Error:
        return None
    try:
        tms_y = (1 << z) - 1 - y
        row = conn.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? "
            "AND tile_row=?", (z, x, tms_y)).fetchone()
        return bytes(row[0]) if row and row[0] is not None else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def read_mbtiles_metadata(db_path: str) -> Dict[str, str]:
    try:
        conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True,
                               timeout=2.0)
    except sqlite3.Error:
        return {}
    try:
        return {str(k): str(v) for k, v in
                conn.execute("SELECT name, value FROM metadata")}
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


class _MbtilesSignals(QObject):
    done = pyqtSignal(object, object)        # (key, bytes or None)


class _MbtilesJob(QRunnable):
    """Reads one tile off the GUI thread. sqlite is opened per job."""

    def __init__(self, key, paths: List[str]) -> None:
        super().__init__()
        self.key = key
        self.paths = paths
        self.signals = _MbtilesSignals()
        self.setAutoDelete(True)

    def run(self) -> None:
        _source, z, x, y = self.key
        data = None
        for path in self.paths:
            data = read_mbtiles_tile(path, z, x, y)
            if data:
                break
        try:
            self.signals.done.emit(self.key, data)
        except RuntimeError:
            pass                 # window closed while the job was in flight


# ---------------------------------------------------------------------------
# Tile provider
# ---------------------------------------------------------------------------

class TileProvider(QObject):
    """Fetches tiles for the active source and caches the decoded images."""

    tile_ready = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.source: Optional[MapSource] = None
        self._cache: "OrderedDict[Any, QImage]" = OrderedDict()
        self._pending: set = set()
        self._failed: set = set()
        self._mbtiles_paths: List[str] = []
        self._pool = QThreadPool(self)
        # Two readers is enough to keep the visible grid filled without
        # competing with the serial and logger threads for CPU.
        self._pool.setMaxThreadCount(2)
        self._net = QNetworkAccessManager(self)
        self._net.finished.connect(self._on_reply)
        self._inflight = 0
        self._queue: List[Any] = []
        self.network_errors = 0
        self.tiles_drawn = 0

    # -- source ------------------------------------------------------------

    def set_source(self, source: MapSource, base_dir: str = ".") -> None:
        self.source = source
        self._cache.clear()
        self._pending.clear()
        self._failed.clear()
        self._queue.clear()
        self._mbtiles_paths = []
        if source.type == "mbtiles":
            directory = source.path or MAPS_DIR
            if not os.path.isabs(directory):
                directory = os.path.join(base_dir, directory)
            if os.path.isfile(directory):
                self._mbtiles_paths = [directory]
            else:
                self._mbtiles_paths = find_mbtiles(directory)

    @property
    def offline_ready(self) -> bool:
        return bool(self._mbtiles_paths)

    # -- lookup ------------------------------------------------------------

    def get(self, z: int, x: int, y: int) -> Optional[QImage]:
        """Return a decoded tile, or None and start fetching it."""
        source = self.source
        if source is None:
            return None
        if z < source.min_zoom or z > source.max_zoom:
            return None
        key = (source.name, z, x, y)
        image = self._cache.get(key)
        if image is not None:
            self._cache.move_to_end(key)
            return image
        if key in self._pending or key in self._failed:
            return None
        self._request(key)
        return None

    def _request(self, key) -> None:
        source = self.source
        self._pending.add(key)
        if source.type == "mbtiles":
            job = _MbtilesJob(key, self._mbtiles_paths)
            job.signals.done.connect(self._on_mbtiles)
            self._pool.start(job)
            return
        if source.needs_key:
            # No key in the environment: fail quietly rather than send a
            # request that will be rejected, and let the grey tile say so.
            self._pending.discard(key)
            self._failed.add(key)
            return
        if self._inflight >= MAX_INFLIGHT:
            self._queue.append(key)
            return
        self._send(key)

    def _send(self, key) -> None:
        url = self._tile_url(key)
        if not url:
            self._pending.discard(key)
            self._failed.add(key)
            return
        request = QNetworkRequest(QUrl(url))
        request.setRawHeader(b"User-Agent", USER_AGENT.encode("ascii"))
        request.setAttribute(QNetworkRequest.FollowRedirectsAttribute, True)
        request.setAttribute(QNetworkRequest.User, key)
        self._inflight += 1
        self._net.get(request)

    def _tile_url(self, key) -> str:
        source = self.source
        _name, z, x, y = key
        if source.type == "xyz":
            url = (source.url.replace("{z}", str(z)).replace("{x}", str(x))
                   .replace("{y}", str(y)))
            if source.key_env and source.api_key:
                sep = "&" if "?" in url else "?"
                url = "%s%s%s=%s" % (url, sep, source.key_param,
                                     quote(source.api_key))
            return url
        if source.type == "wms":
            west, south, east, north = tile_bounds_3857(x, y, z)
            params = [
                ("service", "WMS"), ("version", "1.1.1"), ("request", "GetMap"),
                ("layers", source.layers), ("styles", ""),
                ("srs", "EPSG:3857"),
                ("bbox", "%.6f,%.6f,%.6f,%.6f" % (west, south, east, north)),
                ("width", str(TILE_PX)), ("height", str(TILE_PX)),
                ("format", "image/png"), ("transparent", "true"),
            ]
            if source.key_env and source.api_key:
                params.append((source.key_param, source.api_key))
            sep = "&" if "?" in source.url else "?"
            return source.url + sep + "&".join(
                "%s=%s" % (k, quote(str(v), safe=",:")) for k, v in params)
        return ""

    # -- completion --------------------------------------------------------

    def _on_mbtiles(self, key, data) -> None:
        self._pending.discard(key)
        self._store(key, data)

    def _on_reply(self, reply) -> None:
        self._inflight = max(0, self._inflight - 1)
        key = reply.request().attribute(QNetworkRequest.User)
        data = None
        try:
            if reply.error() == 0:
                payload = bytes(reply.readAll())
                # A WMS answers an error with XML and HTTP 200, so the
                # content type is what decides whether this is a tile.
                ctype = bytes(reply.header(QNetworkRequest.ContentTypeHeader)
                              or b"")
                if isinstance(ctype, bytes):
                    ctype = ctype.decode("ascii", "replace")
                if payload and (not ctype or ctype.startswith("image")):
                    data = payload
                else:
                    self.network_errors += 1
            else:
                self.network_errors += 1
        except Exception:
            self.network_errors += 1
        finally:
            reply.deleteLater()
        if key is not None:
            self._pending.discard(key)
            self._store(key, data)
        while self._queue and self._inflight < MAX_INFLIGHT:
            self._send(self._queue.pop(0))

    def _store(self, key, data) -> None:
        if not data:
            self._failed.add(key)
            self.tile_ready.emit()
            return
        image = QImage()
        if not image.loadFromData(data):
            self._failed.add(key)
            self.tile_ready.emit()
            return
        self._cache[key] = image
        self._cache.move_to_end(key)
        while len(self._cache) > TILE_CACHE_MAX:
            self._cache.popitem(last=False)
        self.tile_ready.emit()

    def clear_failures(self) -> None:
        """Let previously missing tiles be retried (e.g. network returned)."""
        self._failed.clear()

    def shutdown(self) -> None:
        self._queue.clear()
        try:
            self._pool.clear()
            self._pool.waitForDone(3000)
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class MapCanvas(QWidget):
    """The map itself: tiles, tracks, vehicles, scale bar, attribution."""

    cursor_moved = pyqtSignal(float, float)       # lat, lon

    def __init__(self, store: TrackStore, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.store = store
        self.provider = TileProvider(self)
        self.provider.tile_ready.connect(self.request_redraw)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self.setAutoFillBackground(False)

        self.center_lat = 32.7266          # CU Jammu, a sane first view
        self.center_lon = 74.8570
        self.zoom = 13
        self.follow = ""                   # "", "CANSAT" or "ROCKET"
        self.ground_station: Optional[Tuple[float, float]] = None
        self.paint_errors = 0
        self._last_paint_error = ""

        self._drag_origin: Optional[QPoint] = None
        self._drag_center: Tuple[float, float] = (0.0, 0.0)

        # Redraw coalescing: intake sets a flag, the timer repaints.
        self._dirty = False
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / REDRAW_HZ))
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # -- state -------------------------------------------------------------

    def request_redraw(self) -> None:
        """Mark the view stale. O(1); the timer does the actual painting."""
        self._dirty = True

    def _tick(self) -> None:
        if self.follow:
            track = self.store.get(self.follow)
            if track and track.last:
                self.center_lat, self.center_lon = track.last.lat, track.last.lon
                self._dirty = True
        # A stale label counts up in seconds, so repaint while anything is
        # on screen even if no packet arrived.
        if self._dirty or self.store.total_points:
            self._dirty = False
            self.update()

    def set_source(self, source: MapSource, base_dir: str = ".") -> None:
        self.provider.set_source(source, base_dir)
        self.request_redraw()

    def set_zoom(self, zoom: int, anchor: Optional[QPoint] = None) -> None:
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, int(zoom)))
        if zoom == self.zoom:
            return
        if anchor is None:
            self.zoom = zoom
            self.request_redraw()
            return
        # Keep the coordinate under the cursor fixed: convert it to lon/lat
        # at the old zoom, then re-centre so it lands on the same pixel.
        lon, lat = self._pixel_to_lonlat(anchor)
        self.zoom = zoom
        wx, wy = lonlat_to_world(lon, lat, zoom)
        cx, cy = anchor.x() - self.width() / 2.0, anchor.y() - self.height() / 2.0
        self.center_lon, self.center_lat = world_to_lonlat(wx - cx, wy - cy, zoom)
        self.request_redraw()

    def fit_all(self) -> None:
        """Frame every recorded point, plus the ground station if set."""
        box = self.store.bounds()
        if box is None and self.ground_station is None:
            return
        if box is None:
            self.center_lat, self.center_lon = self.ground_station
            self.zoom = 15
            self.request_redraw()
            return
        min_lat, min_lon, max_lat, max_lon = box
        if self.ground_station is not None:
            gs_lat, gs_lon = self.ground_station
            min_lat, max_lat = min(min_lat, gs_lat), max(max_lat, gs_lat)
            min_lon, max_lon = min(min_lon, gs_lon), max(max_lon, gs_lon)
        self.center_lat = (min_lat + max_lat) / 2.0
        self.center_lon = (min_lon + max_lon) / 2.0
        for zoom in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
            x0, y0 = lonlat_to_world(min_lon, max_lat, zoom)
            x1, y1 = lonlat_to_world(max_lon, min_lat, zoom)
            if (abs(x1 - x0) <= self.width() * 0.85
                    and abs(y1 - y0) <= self.height() * 0.85):
                self.zoom = zoom
                break
        else:
            self.zoom = MIN_ZOOM
        self.request_redraw()

    # -- coordinate helpers -------------------------------------------------

    def _origin(self) -> Tuple[float, float]:
        """World pixel at the widget's top-left corner."""
        cx, cy = lonlat_to_world(self.center_lon, self.center_lat, self.zoom)
        return cx - self.width() / 2.0, cy - self.height() / 2.0

    def _lonlat_to_pixel(self, lon: float, lat: float) -> QPointF:
        ox, oy = self._origin()
        wx, wy = lonlat_to_world(lon, lat, self.zoom)
        return QPointF(wx - ox, wy - oy)

    def _pixel_to_lonlat(self, point: QPoint) -> Tuple[float, float]:
        ox, oy = self._origin()
        return world_to_lonlat(ox + point.x(), oy + point.y(), self.zoom)

    # -- input -------------------------------------------------------------

    def wheelEvent(self, event) -> None:          # noqa: N802 - Qt naming
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.set_zoom(self.zoom + (1 if steps > 0 else -1), event.pos())
        event.accept()

    def mousePressEvent(self, event) -> None:     # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_origin = event.pos()
            self._drag_center = (self.center_lon, self.center_lat)
        event.accept()

    def mouseMoveEvent(self, event) -> None:      # noqa: N802
        lon, lat = self._pixel_to_lonlat(event.pos())
        self.cursor_moved.emit(lat, lon)
        if self._drag_origin is not None:
            dx = event.pos().x() - self._drag_origin.x()
            dy = event.pos().y() - self._drag_origin.y()
            cx, cy = lonlat_to_world(self._drag_center[0], self._drag_center[1],
                                     self.zoom)
            self.center_lon, self.center_lat = world_to_lonlat(
                cx - dx, cy - dy, self.zoom)
            # Dragging by hand overrides Follow; otherwise the next packet
            # snaps the view back and the map fights the operator.
            self.follow = ""
            self.request_redraw()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:   # noqa: N802
        self._drag_origin = None
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:   # noqa: N802
        self.center_lon, self.center_lat = self._pixel_to_lonlat(event.pos())
        self.follow = ""
        self.request_redraw()
        event.accept()

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:          # noqa: N802
        painter = QPainter()
        try:
            painter.begin(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.fillRect(self.rect(), QColor(COL_BG))
            self._draw_tiles(painter)
            self._draw_tracks(painter)
            self._draw_ground_station(painter)
            self._draw_vehicles(painter)
            self._draw_scale_bar(painter)
            self._draw_attribution(painter)
        except Exception as exc:                  # noqa: BLE001
            # A map that throws out of paintEvent takes the window with it,
            # and Qt will keep calling paintEvent. Count it, remember the
            # first one for the status line, and keep the frame.
            self.paint_errors += 1
            if not self._last_paint_error:
                self._last_paint_error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            if painter.isActive():
                painter.end()

    def _draw_tiles(self, painter: QPainter) -> None:
        ox, oy = self._origin()
        first_x = int(math.floor(ox / TILE_PX))
        first_y = int(math.floor(oy / TILE_PX))
        cols = int(self.width() / TILE_PX) + 2
        rows = int(self.height() / TILE_PX) + 2
        span = 1 << self.zoom
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        for row in range(rows):
            for col in range(cols):
                tx, ty = first_x + col, first_y + row
                px = int(tx * TILE_PX - ox)
                py = int(ty * TILE_PX - oy)
                rect = QRect(px, py, TILE_PX, TILE_PX)
                if ty < 0 or ty >= span:
                    painter.fillRect(rect, QColor(COL_BG))
                    continue
                wrapped = tx % span                 # the world repeats east-west
                image = self.provider.get(self.zoom, wrapped, ty)
                if image is not None and not image.isNull():
                    painter.drawImage(rect, image)
                    self.provider.tiles_drawn += 1
                else:
                    # A visible hole, not a plausible-looking blank: the
                    # operator must be able to tell "no map here" from
                    # "nothing is there".
                    painter.fillRect(rect, QColor(COL_MISSING))
                    painter.setPen(QColor(COL_DIM))
                    painter.drawRect(rect.adjusted(0, 0, -1, -1))
                    painter.drawText(rect, Qt.AlignCenter,
                                     "no tile\nz%d" % self.zoom)

    def _draw_tracks(self, painter: QPainter) -> None:
        for track in self.store.non_empty():
            colour = QColor(VEHICLE_COLOURS.get(track.vehicle,
                                                VEHICLE_COLOURS["GENERIC"]))
            points = decimate(track.points, MAX_DRAWN_POINTS)
            if len(points) < 2:
                continue
            path = QPainterPath()
            first = self._lonlat_to_pixel(points[0].lon, points[0].lat)
            path.moveTo(first)
            for point in points[1:]:
                path.lineTo(self._lonlat_to_pixel(point.lon, point.lat))
            pen = QPen(colour, 2.6)
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

    def _draw_ground_station(self, painter: QPainter) -> None:
        if self.ground_station is None:
            return
        lat, lon = self.ground_station
        pos = self._lonlat_to_pixel(lon, lat)
        painter.setPen(QPen(QColor(COL_TEXT), 2))
        painter.setBrush(QColor("#ffffff"))
        size = 7
        painter.drawRect(int(pos.x() - size / 2), int(pos.y() - size / 2),
                         size, size)
        painter.drawLine(int(pos.x()), int(pos.y() - 12),
                         int(pos.x()), int(pos.y() - size / 2))
        painter.setPen(QColor(COL_TEXT))
        painter.drawText(int(pos.x()) + 9, int(pos.y()) + 4, "GS")

    def _draw_vehicles(self, painter: QPainter) -> None:
        now = time.time()
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        offset = 0
        for track in self.store.non_empty():
            last = track.last
            if last is None:
                continue
            colour = QColor(VEHICLE_COLOURS.get(track.vehicle,
                                                VEHICLE_COLOURS["GENERIC"]))
            pos = self._lonlat_to_pixel(last.lon, last.lat)
            age = now - last.epoch if last.epoch else 0.0
            stale = age > STALE_FIX_S

            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(colour)
            if track.vehicle == "ROCKET":
                # A triangle for the rocket, a circle for the CanSat: shape
                # as well as colour, so the two are still distinguishable in
                # a screenshot or to a colour-blind operator.
                path = QPainterPath()
                path.moveTo(pos.x(), pos.y() - 9)
                path.lineTo(pos.x() + 8, pos.y() + 7)
                path.lineTo(pos.x() - 8, pos.y() + 7)
                path.closeSubpath()
                painter.drawPath(path)
            else:
                painter.drawEllipse(pos, 7.0, 7.0)

            alt = ("%.0f m" % last.alt_m) if last.has_valid_alt else "no alt"
            label = "%s  %s  %.0fs" % (track.vehicle, alt, age)
            if self.ground_station is not None:
                gs_lat, gs_lon = self.ground_station
                label += "  %.0fm @ %.0f°" % (
                    haversine_m(gs_lat, gs_lon, last.lat, last.lon),
                    bearing_deg(gs_lat, gs_lon, last.lat, last.lon))
            metrics = painter.fontMetrics()
            width = metrics.horizontalAdvance(label) + 10
            box = QRect(int(pos.x()) + 12, int(pos.y()) - 10 + offset,
                        width, 18)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 222))
            painter.drawRect(box)
            painter.setPen(QColor(COL_ALERT) if stale else QColor(COL_TEXT))
            painter.drawText(box.adjusted(5, 0, 0, 0),
                             Qt.AlignVCenter | Qt.AlignLeft, label)
            offset += 20

    def _draw_scale_bar(self, painter: QPainter) -> None:
        mpp = metres_per_pixel(self.center_lat, self.zoom)
        if not math.isfinite(mpp) or mpp <= 0:
            return
        target_px = 120
        raw = mpp * target_px
        # Snap to a 1/2/5 x 10^n metre step, the convention every map uses.
        power = 10 ** int(math.floor(math.log10(raw))) if raw > 0 else 1
        for mult in (1, 2, 5, 10):
            nice = mult * power
            if nice >= raw:
                break
        width = int(nice / mpp)
        x0, y0 = 12, self.height() - 18
        painter.setPen(QPen(QColor(COL_TEXT), 2))
        painter.drawLine(x0, y0, x0 + width, y0)
        painter.drawLine(x0, y0 - 5, x0, y0 + 2)
        painter.drawLine(x0 + width, y0 - 5, x0 + width, y0 + 2)
        text = "%d m" % nice if nice < 1000 else "%.1f km" % (nice / 1000.0)
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(x0, y0 - 8, text)

    def _draw_attribution(self, painter: QPainter) -> None:
        source = self.provider.source
        text = source.attribution if source else ""
        if source and source.type == "mbtiles" and not self.provider.offline_ready:
            text = "No .mbtiles file in %s" % MAPS_DIR
        if source and source.needs_key:
            text = "%s — set %s to use this source" % (text, source.key_env)
        if not text:
            return
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 10
        box = QRect(self.width() - width - 6, self.height() - 20, width, 16)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 200))
        painter.drawRect(box)
        painter.setPen(QColor(COL_DIM))
        painter.drawText(box, Qt.AlignCenter, text)


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

class MapWindow(QDialog):
    """Non-modal live map, opened from the dashboard's MAP button."""

    #: One line for the dashboard's event log.
    log_message = pyqtSignal(str)

    def __init__(self, store: TrackStore, base_dir: str = ".",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.base_dir = base_dir
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Live map")
        self.resize(980, 720)
        self.setStyleSheet(
            "QDialog { background-color: %s; }"
            "QLabel { color: %s; }"
            "QDoubleSpinBox { background: %s; color: %s; border: 1px solid %s;"
            "                 border-radius: 4px; padding: 3px 6px; }"
            % (COL_PANEL, COL_TEXT, COL_BG, COL_TEXT, COL_BORDER))

        self.sources, note = load_sources(base_dir)
        self._pending_note = note

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.canvas = MapCanvas(store, self)
        bar = QHBoxLayout()
        bar.setSpacing(6)

        bar.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        for source in self.sources:
            self.source_combo.addItem(source.name)
        self.source_combo.currentIndexChanged.connect(self._on_source)
        bar.addWidget(self.source_combo)

        bar.addWidget(QLabel("Follow:"))
        self.follow_combo = QComboBox()
        self.follow_combo.addItems(["Off", "CANSAT", "ROCKET"])
        self.follow_combo.currentTextChanged.connect(self._on_follow)
        bar.addWidget(self.follow_combo)

        self.fit_btn = QPushButton("FIT ALL")
        self.fit_btn.clicked.connect(self.canvas.fit_all)
        bar.addWidget(self.fit_btn)

        self.clear_btn = QPushButton("CLEAR TRACKS")
        self.clear_btn.setToolTip(
            "Forget the drawn tracks. The CSV log and the KML export are "
            "untouched; this only clears what is on the map.")
        self.clear_btn.clicked.connect(self._on_clear)
        bar.addWidget(self.clear_btn)

        bar.addStretch(1)
        bar.addWidget(QLabel("GS lat:"))
        self.gs_lat = QDoubleSpinBox()
        self.gs_lat.setRange(-90.0, 90.0)
        self.gs_lat.setDecimals(6)
        self.gs_lat.setSingleStep(0.0001)
        bar.addWidget(self.gs_lat)
        bar.addWidget(QLabel("lon:"))
        self.gs_lon = QDoubleSpinBox()
        self.gs_lon.setRange(-180.0, 180.0)
        self.gs_lon.setDecimals(6)
        self.gs_lon.setSingleStep(0.0001)
        bar.addWidget(self.gs_lon)
        self.gs_check = QCheckBox("show")
        self.gs_check.setToolTip(
            "Draw the ground station and show range and true bearing to each "
            "vehicle. Saved to %s; leave unchecked to show nothing."
            % SETTINGS_FILE)
        bar.addWidget(self.gs_check)
        for widget in (self.gs_lat, self.gs_lon):
            widget.valueChanged.connect(self._on_gs_changed)
        self.gs_check.toggled.connect(self._on_gs_changed)

        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)

        self.status = QLabel("")
        self.status.setStyleSheet("color: %s; font-size: 9pt;" % COL_DIM)
        layout.addWidget(self.status)
        self.canvas.cursor_moved.connect(self._on_cursor)

        self._load_settings()
        self._select_initial_source()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(500)

    # -- sources -----------------------------------------------------------

    def _select_initial_source(self) -> None:
        """Prefer an offline pack: it is the one that works in the field."""
        chosen = 0
        for index, source in enumerate(self.sources):
            if source.type == "mbtiles":
                directory = source.path or MAPS_DIR
                if not os.path.isabs(directory):
                    directory = os.path.join(self.base_dir, directory)
                if os.path.isfile(directory) or find_mbtiles(directory):
                    chosen = index
                    break
        self.source_combo.setCurrentIndex(chosen)
        self._on_source(chosen)

    def _on_source(self, index: int) -> None:
        if 0 <= index < len(self.sources):
            source = self.sources[index]
            self.canvas.set_source(source, self.base_dir)
            if source.needs_key:
                self.log_message.emit(
                    "Map source %r needs an API key: set the %s environment "
                    "variable." % (source.name, source.key_env))

    def _on_follow(self, text: str) -> None:
        self.canvas.follow = "" if text == "Off" else text
        self.canvas.request_redraw()

    def _on_clear(self) -> None:
        self.canvas.store.clear()
        self.canvas.request_redraw()
        self.log_message.emit("Map tracks cleared (logs and CSV untouched).")

    # -- ground station ----------------------------------------------------

    def _settings_path(self) -> str:
        return os.path.join(self.base_dir, SETTINGS_FILE)

    def _load_settings(self) -> None:
        try:
            with open(self._settings_path(), "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return                      # no file, or unreadable: show nothing
        station = data.get("ground_station") if isinstance(data, dict) else None
        if not isinstance(station, dict):
            return
        try:
            lat = float(station.get("lat"))
            lon = float(station.get("lon"))
        except (TypeError, ValueError):
            return
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return
        self.gs_lat.setValue(lat)
        self.gs_lon.setValue(lon)
        self.gs_check.setChecked(bool(station.get("show", True)))

    def _save_settings(self) -> None:
        payload = {"ground_station": {"lat": self.gs_lat.value(),
                                      "lon": self.gs_lon.value(),
                                      "show": self.gs_check.isChecked()}}
        try:
            with open(self._settings_path(), "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError as exc:
            self.log_message.emit("Could not save %s: %s" % (SETTINGS_FILE, exc))

    def _on_gs_changed(self, *_args) -> None:
        if self.gs_check.isChecked():
            self.canvas.ground_station = (self.gs_lat.value(), self.gs_lon.value())
        else:
            self.canvas.ground_station = None
        self.canvas.request_redraw()
        self._save_settings()

    # -- status ------------------------------------------------------------

    def _on_cursor(self, lat: float, lon: float) -> None:
        self._cursor = (lat, lon)

    def _update_status(self) -> None:
        store = self.canvas.store
        bits = ["z%d" % self.canvas.zoom]
        cursor = getattr(self, "_cursor", None)
        if cursor:
            bits.append("cursor %.6f, %.6f" % cursor)
        for track in store.non_empty():
            bits.append("%s %d fixes" % (track.vehicle, len(track.points)))
        if not store.non_empty():
            bits.append("no fixes yet")
        source = self.canvas.provider.source
        if source and source.type == "mbtiles" and not self.canvas.provider.offline_ready:
            bits.append("no offline pack in %s" % MAPS_DIR)
        if self.canvas.provider.network_errors:
            bits.append("%d tile errors" % self.canvas.provider.network_errors)
        if self.canvas.paint_errors:
            bits.append("paint error: %s" % self.canvas._last_paint_error)
        self.status.setText("  ·  ".join(bits))

    # -- lifecycle ---------------------------------------------------------

    def showEvent(self, event) -> None:           # noqa: N802
        super().showEvent(event)
        if self._pending_note:
            self.log_message.emit("Map: %s" % self._pending_note)
            self._pending_note = ""
        self.canvas.request_redraw()

    def closeEvent(self, event) -> None:          # noqa: N802
        self.canvas.provider.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        self._status_timer.stop()
        self.canvas._timer.stop()
        self.canvas.provider.shutdown()

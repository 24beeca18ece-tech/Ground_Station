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

* network tiles go through :mod:`tile_fetcher`, a bounded pool of urllib
  workers using Python's own TLS and the Windows certificate store, which
  hands decoded results back by signal;
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
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np

from PyQt5.QtCore import (QObject, QPoint, QPointF, QRect, QRectF, QRunnable,
                          Qt, QThreadPool, QTimer, pyqtSignal)
from PyQt5.QtGui import (QColor, QFont, QImage, QPainter, QPainterPath, QPen,
                         QPolygonF)
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QSlider, QVBoxLayout, QWidget)

from tile_fetcher import (TileError, TileFetcher, TileResult,
                          fetch_tile)
from map_basemap import (ImageOverlay, VectorPack, draw_graticule,
                          draw_north_arrow, draw_range_rings,
                          load_image_overlay, load_vector_pack)
from track_store import (TrackStore, Track, TrackPoint, bearing_deg,
                         haversine_m)

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
MAX_INFLIGHT = 6

#: Connect timeout. With no network at the site the OS default would leave
#: a socket hanging for tens of seconds with an in-flight slot held.
TILE_CONNECT_TIMEOUT_S = 3.0

#: Default read timeout, separate and longer: Bhuvan renders a WMS tile on
#: demand and a first request legitimately takes several seconds.
#: Overridable per source with "read_timeout" in map_sources.json.
TILE_READ_TIMEOUT_S = 8.0

#: While a source is offline, one probe tile is tried this often.
OFFLINE_RETRY_S = 30.0

#: How long a tile that failed is not asked for again. Without this, every
#: repaint re-requests every missing tile -- which with no network is a
#: request storm that never ends and never succeeds.
NEGATIVE_CACHE_S = 60.0

#: Consecutive failures after which a source is declared offline and stops
#: being asked at all until the operator changes source or it is retried.
OFFLINE_AFTER_FAILURES = 12

#: Fetched tiles kept between runs, capped at 500 MB, least recently used
#: removed first. The OSM tile usage policy permits caching, and it is what
#: makes a view already looked at work with the network gone.
TILE_CACHE_DIR = os.path.join("logs", "tile_cache")

#: A fix older than this makes the vehicle label turn red.
STALE_FIX_S = 5.0

#: Minimum pixel separation between drawn track points. Two points closer
#: than this land on the same stroke, so drawing both costs time and shows
#: nothing. This is what keeps paint cost a function of the SCREEN, not of
#: how long the flight has been.
MIN_POINT_SPACING_PX = 2.0

#: Hard ceiling on points in one polyline after thinning. A 2 px spacing
#: already bounds this to roughly (width + height) / 2 for a space-filling
#: path; the cap is a backstop for a pathological zoom.
MAX_DRAWN_POINTS = 4000

#: Where sources and the ground-station position live. Both are gitignored:
#: the first may carry an API key's environment-variable name and local file
#: paths, the second is a physical location.
SOURCES_FILE = "map_sources.json"
SETTINGS_FILE = "map_settings.json"
MAPS_DIR = os.path.join("assets", "maps")

#: Application name and version for the User-Agent. Providers' usage
#: policies (OpenStreetMap's explicitly) require an application to identify
#: itself and give a way to contact its operator, not impersonate a browser.
APP_IDENT = "CU-Jammu-Astro-GroundStation/1.0"

#: The project page, used when no contact has been configured.
PROJECT_URL = "https://github.com/24beeca18ece-tech/Ground_Station"


def default_user_agent(contact: str) -> str:
    """The User-Agent sent on every tile request.

    *contact* is the ``osm_contact`` field from map_settings.json -- an email
    or URL where the OSM operations team can reach whoever is running this
    station. With it empty, the project page is used, which identifies the
    software but not the person; the map says so and asks for it.
    """
    contact = (contact or "").strip()
    return "%s (+%s)" % (APP_IDENT, contact or PROJECT_URL)


#: Kept for callers that predate the configurable contact.
USER_AGENT = default_user_agent("")

#: The map_settings.json field holding the operator's contact for the
#: OpenStreetMap User-Agent. Deliberately not a placeholder value: an empty
#: field is reported on screen and in the log until it is filled in.
OSM_CONTACT_FIELD = "osm_contact"
OSM_CONTACT_HELP = ("Your email address or a URL where OpenStreetMap's "
                    "operations team can reach you. Sent in the User-Agent "
                    "of every OSM tile request, as their tile usage policy "
                    "asks.")

#: The launch pad, used for the marker and -- when no ground station is set
#: -- as the origin of the range rings and of distance and bearing. Written
#: into map_settings.json ("launch_pad") so it can be edited there.
DEFAULT_LAUNCH_PAD: Dict[str, Any] = {"name": "Launch pad",
                                      "lat": 26.720028, "lon": 84.304583}
LAUNCH_PAD_FIELD = "launch_pad"
COL_PAD = "#e8590c"

#: Zoom levels past a source's or pack's max_zoom that are drawn by scaling
#: up the parent tile. It is also how far in the user can zoom on that map:
#: past it there is nothing left worth drawing.
OVERZOOM_LEVELS = 3
#: How far out past a tile source's lowest zoom the user can zoom.
UNDERZOOM_LEVELS = 2
#: Zoom GO TO PAD puts the view at.
PAD_ZOOM = 13
#: Scaled-up tiles kept in memory (never written to disk).
OVERZOOM_CACHE_MAX = 200

#: The map_settings.json field holding a contact for bulk tile downloads
#: (tools/fetch_tiles.py puts it in the User-Agent). Empty until filled in.
TILE_CONTACT_FIELD = "tile_contact"
TILE_CONTACT_HELP = ("Your email address or a URL where a tile provider can "
                     "reach you. Sent in the User-Agent when tools/"
                     "fetch_tiles.py downloads an offline pack.")

#: Sentinel-2 cloudless by EOX. CC BY-NC-SA 4.0: non-commercial use only,
#: share alike, and this attribution whenever the imagery is shown.
S2CLOUDLESS_NAME = "Sentinel-2 cloudless 2023 (EOX)"
S2CLOUDLESS_ATTRIBUTION = ("Sentinel-2 cloudless - https://s2maps.eu by EOX IT "
                           "Services GmbH (Contains modified Copernicus "
                           "Sentinel data 2023)")
#: Verified against https://tiles.maps.eox.at/wmts/1.0.0/WMTSCapabilities.xml
#: on 2026-10-27: layer s2cloudless-2023_3857, TileMatrixSet "g" (Web
#: Mercator, 256 px, 2^z x 2^z), ResourceURL template
#: .../{TileMatrixSet}/{TileMatrix}/{TileRow}/{TileCol}.jpg -- z, then y,
#: then x. The file gives http://; that answers 302 to the https:// form.
S2CLOUDLESS_URL = ("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/"
                   "default/g/{z}/{y}/{x}.jpg")

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
        # "boundaries only" in the name, so nobody expects imagery or roads
        # from it: this layer draws administrative boundaries and nothing
        # else, and the imagery hosts were unreachable when it was set up.
        "name": "ISRO Bhuvan (boundaries only)",
        "type": "wms",
        "url": "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms",
        "layers": "basemap:admin_group_ntl",
        "attribution": "© ISRO / NRSC Bhuvan",
        "min_zoom": 4,
        "max_zoom": 16,
        # A WMS renders on demand; first tiles measured at 0.9-1.3 s here
        # and can be far slower under load.
        "read_timeout": 12.0,
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
    {
        "name": S2CLOUDLESS_NAME,
        "type": "xyz",
        "url": S2CLOUDLESS_URL,
        "attribution": S2CLOUDLESS_ATTRIBUTION,
        "min_zoom": 0,
        # About 10 m per pixel; z14 is ~9.5 m/px at the launch site. Deeper
        # is overzoomed from z14 by the map rather than fetched.
        "max_zoom": 14,
        # EOX permits offline packs inside an application for non-commercial
        # use; tools/fetch_tiles.py may build one, gently.
        "bulk_download": True,
        "rate_limit": 1.0,
        "max_tiles": 4000,
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


def lonlat_to_world_array(lons: "np.ndarray", lats: "np.ndarray",
                          zoom: int) -> "tuple":
    """Vectorised :func:`lonlat_to_world` for whole columns at once.

    One numpy expression instead of a Python call per fix. This is what
    makes a 200,000-point track cost about the same to project as a
    1,000-point one, and it is why the store keeps columns.
    """
    size = TILE_PX * (2 ** zoom)
    x = (lons + 180.0) / 360.0 * size
    clamped = np.clip(lats, -85.05112878, 85.05112878)
    sin_lat = np.sin(np.radians(clamped))
    y = (0.5 - np.log((1 + sin_lat) / (1 - sin_lat)) / (4 * np.pi)) * size
    return x, y


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
    #: Seconds to wait for a tile body. Per source because a WMS that renders
    #: on demand is far slower than a static tile server.
    read_timeout: float = TILE_READ_TIMEOUT_S
    #: Most requests per second tools/fetch_tiles.py may make; 0 = its default.
    rate_limit: float = 0.0
    #: Hard cap on tiles per fetch_tiles job, not overridable; 0 = none.
    max_tiles: int = 0

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
            read_timeout=float(entry.get("read_timeout",
                                         TILE_READ_TIMEOUT_S)),
            rate_limit=float(entry.get("rate_limit", 0.0) or 0.0),
            max_tiles=int(entry.get("max_tiles", 0) or 0),
        ))
    if not sources:
        sources = [MapSource(**{k: v for k, v in BUILTIN_SOURCES[0].items()})]
    return sources, note


@dataclass
class MapPack:
    """One thing the map can draw underneath the tracks.

    Four kinds, all offline: the always-available grid, an MBTiles tile
    pack, an OSM vector pack this application renders itself, and a
    georeferenced image the operator supplies.
    """

    name: str
    kind: str                 # "grid" | "mbtiles" | "vector" | "image"
    path: str = ""
    attribution: str = ""
    error: str = ""
    #: Tile packs: the zoom range stored, from the MBTiles metadata. Past
    #: max_zoom the map overzooms rather than asking for absent tiles.
    min_zoom: int = 0
    max_zoom: int = 19
    #: Tile packs: (west, south, east, north) from the metadata, or None.
    bounds: Optional[Tuple[float, float, float, float]] = None


#: Always present, always works, needs nothing.
GRID_PACK = MapPack(name="Grid only (no map data)", kind="grid",
                    attribution="Lat/lon grid — no map pack loaded, see README")


def find_packs(base_dir: str = ".") -> List[MapPack]:
    """Every pack in assets/maps/, plus the grid. Never raises.

    The grid is always first so there is something that works even if the
    directory is missing, unreadable or full of rubbish.
    """
    packs = [GRID_PACK]
    directory = os.path.join(base_dir, MAPS_DIR)
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return packs
    for entry in names:
        path = os.path.join(directory, entry)
        lower = entry.lower()
        if lower.endswith(".mbtiles"):
            packs.append(mbtiles_pack(entry, path))
        elif lower.endswith(".osmpack.json"):
            packs.append(MapPack(
                name="%s (vector)" % entry[:-len(".osmpack.json")],
                kind="vector", path=path,
                attribution="© OpenStreetMap contributors, ODbL"))
        elif lower.endswith(".json"):
            # A sidecar for an image overlay, if the image is beside it.
            base = os.path.splitext(path)[0]
            if any(os.path.isfile(base + ext)
                   for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG")):
                packs.append(MapPack(name="%s (image)" % entry[:-5],
                                     kind="image", path=path,
                                     attribution="Operator-supplied image"))
    return packs


def mbtiles_pack(entry: str, path: str) -> MapPack:
    """A tile pack described by its own metadata table.

    The attribution travels in the file, so a pack built from a licensed
    source (Sentinel-2 cloudless, say) carries the exact notice its licence
    requires and the map shows it. Missing or odd metadata falls back to
    plain defaults rather than refusing the pack.
    """
    meta = read_mbtiles_metadata(path)

    def zoom(key: str, default: int) -> int:
        try:
            return max(0, min(MAX_ZOOM, int(float(meta.get(key, default)))))
        except (TypeError, ValueError):
            return default

    bounds = None
    try:
        parts = [float(v) for v in meta.get("bounds", "").split(",")]
        if len(parts) == 4 and all(math.isfinite(v) for v in parts):
            bounds = (parts[0], parts[1], parts[2], parts[3])
    except ValueError:
        bounds = None
    return MapPack(name="%s (tiles)" % entry, kind="mbtiles", path=path,
                   attribution=meta.get("attribution", "") or "Offline tile pack",
                   min_zoom=zoom("minzoom", 0), max_zoom=zoom("maxzoom", 19),
                   bounds=bounds)


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
    """Fetches tiles for the active source and caches the decoded images.

    Network tiles go through :class:`tile_fetcher.TileFetcher` -- a bounded
    pool of urllib workers using Python's own TLS and the Windows
    certificate store -- and MBTiles reads through a QThreadPool. The GUI
    thread only ever enqueues work and receives results by signal.
    """

    tile_ready = pyqtSignal()
    #: ``(message,)`` the first time each distinct problem is seen, so the
    #: event log records a cause once rather than once per tile.
    log_error = pyqtSignal(str)
    #: ``(source name,)`` when a source that was failing delivers a tile.
    recovered = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None,
                 user_agent: str = "", cache_dir: str = "") -> None:
        super().__init__(parent)
        self.source: Optional[MapSource] = None
        self._cache: "OrderedDict[Any, QImage]" = OrderedDict()
        #: key -> url still being fetched. A tile that scrolls off screen is
        #: dropped from here and its worker skips it before connecting.
        self._pending: Dict[Any, str] = {}
        #: key -> monotonic time the failure expires. A plain set would mean
        #: a tile that failed once is never retried in the whole session,
        #: which is wrong when the network comes back; no expiry at all means
        #: re-requesting every missing tile on every repaint.
        self._failed: Dict[Any, float] = {}
        self._consecutive_failures = 0
        self._handler_faults: set = set()
        self._reported_kinds: set = set()
        self._mbtiles_paths: List[str] = []
        self._pool = QThreadPool(self)
        # Two readers is enough to keep the visible grid filled without
        # competing with the serial and logger threads for CPU.
        self._pool.setMaxThreadCount(2)
        self._queue: List[Any] = []
        self._user_agent = user_agent or default_user_agent("")
        self._cache_dir = cache_dir
        self._fetcher: Optional[TileFetcher] = None
        self._last_probe = 0.0
        #: The most recent failure, in plain words, and its kind.
        self.last_error = ""
        self.last_error_kind = ""
        #: Bumped whenever the drawable tile set changes, so the canvas can
        #: tell "nothing changed" from "a tile arrived" without comparing
        #: the cache itself.
        self.generation = 0
        self.network_errors = 0
        self.tiles_drawn = 0
        self.cache_hits = 0
        #: (source, z, x, y, levels up) -> parent crop scaled to a full tile.
        #: Memory only; rebuilt from the parent whenever it is needed again.
        self._scaled: "OrderedDict[Any, QImage]" = OrderedDict()
        self.overzoom_drawn = 0

    # -- configuration -----------------------------------------------------

    @property
    def user_agent(self) -> str:
        return self._user_agent

    @user_agent.setter
    def user_agent(self, value: str) -> None:
        self._user_agent = value
        if self._fetcher is not None:
            self._fetcher.user_agent = value

    def _ensure_fetcher(self) -> TileFetcher:
        if self._fetcher is None:
            cache_dir = self._cache_dir or os.path.join(".", TILE_CACHE_DIR)
            self._fetcher = TileFetcher(cache_dir=cache_dir,
                                        user_agent=self._user_agent,
                                        workers=MAX_INFLIGHT, parent=self)
            self._fetcher.finished.connect(self._on_fetched)
        return self._fetcher

    @property
    def _inflight(self) -> int:
        return self._fetcher.in_flight if self._fetcher is not None else 0

    # -- source ------------------------------------------------------------

    def set_source(self, source: MapSource, base_dir: str = ".") -> None:
        self.source = source
        self._cache.clear()
        self._scaled.clear()
        self.abort_pending()
        self._failed.clear()
        self._consecutive_failures = 0
        self._reported_kinds.clear()
        self.last_error = ""
        self.last_error_kind = ""
        self._queue.clear()
        self._mbtiles_paths = []
        if not self._cache_dir:
            self._cache_dir = os.path.join(base_dir, TILE_CACHE_DIR)
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

    @property
    def is_offline(self) -> bool:
        """True when this source has failed often enough to stop asking it.

        Not permanent: :meth:`probe_if_due` tries one tile every
        :data:`OFFLINE_RETRY_S`, and the first success clears it.
        """
        return (self.source is not None and self.source.type != "mbtiles"
                and self._consecutive_failures >= OFFLINE_AFTER_FAILURES)

    @property
    def status_text(self) -> str:
        """The most recent problem in plain words, or "" when healthy.

        "source unreachable" was not a diagnosis; this says which source and
        what actually happened -- "OpenStreetMap: HTTP 403 (blocked ...)",
        "ISRO Bhuvan (boundaries only): timed out after 8 s".
        """
        if not self.last_error:
            return ""
        name = self.source.name if self.source else "tile source"
        prefix = "OFFLINE — " if self.is_offline else ""
        reason = self.last_error
        if self.last_error_kind == TileError.TIMEOUT and self.source is not None:
            reason = "timed out after %g s" % self.source.read_timeout
        return "%s%s: %s" % (prefix, name, reason)

    def retry_source(self) -> None:
        """Forget the offline verdict and the negative cache, and try again."""
        self._failed.clear()
        self._consecutive_failures = 0
        self._last_probe = time.monotonic()

    def probe_if_due(self) -> bool:
        """While offline, allow one probe tile every OFFLINE_RETRY_S.

        Giving up for good is wrong at a launch site: a hotspot comes back,
        a proxy starts answering. Dropping the counter to one below the
        threshold lets exactly one request through; if it fails the source
        is offline again, and if it succeeds the counter resets to zero.
        """
        if not self.is_offline:
            return False
        now = time.monotonic()
        if now - self._last_probe < OFFLINE_RETRY_S:
            return False
        self._last_probe = now
        self._failed.clear()
        self._consecutive_failures = OFFLINE_AFTER_FAILURES - 1
        return True

    def abort_pending(self, keep: Optional[set] = None) -> None:
        """Forget queued and in-flight tiles, except those in *keep*.

        Panning away from a tile makes its request worthless. A worker that
        has not yet connected checks the wanted set and skips the tile; one
        mid-transfer finishes, which is cheaper than tearing down a TLS
        connection, and its result is simply cached.
        """
        for key in list(self._pending):
            if keep is not None and key in keep:
                continue
            self._pending.pop(key, None)
        if keep is None:
            self._queue.clear()
        else:
            self._queue = [k for k in self._queue if k in keep]
        if self._fetcher is not None:
            self._fetcher.set_wanted(keep if keep is not None else set())

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
        if key in self._pending:
            return None
        expiry = self._failed.get(key)
        if expiry is not None:
            if time.monotonic() < expiry:
                return None
            del self._failed[key]            # the window has passed; retry
        if self.is_offline:
            return None
        self._request(key)
        return None

    def overzoomed(self, z: int, x: int, y: int) -> Optional[QImage]:
        """A stand-in for a tile that is not available: the nearest parent
        that is, cropped to this tile's area and scaled up.

        Every parent level is tried, down to the source's lowest zoom, so a
        square is grey only when no level at all has that place. (Searching
        just 3 levels up is what left the map blank when zoomed in at the
        edge of a small pack, though a coarser tile covered the spot.)

        Past the source's max_zoom this is the only way anything is drawn,
        so the parent at max_zoom is requested if it is not here yet. Inside
        the range -- a tile still loading, or missing from a pack -- only a
        parent that is already cached is used, except for an offline pack,
        where asking costs nothing but a local read. Results are kept in a
        small memory cache, never on disk.
        """
        source = self.source
        if source is None or z <= source.min_zoom:
            return None
        beyond = z - source.max_zoom
        if beyond > OVERZOOM_LEVELS:
            return None
        local = source.type == "mbtiles"
        for up in range(max(1, beyond), z - source.min_zoom + 1):
            pz = z - up
            key = (source.name, z, x, y, up)
            scaled = self._scaled.get(key)
            if scaled is not None:
                self._scaled.move_to_end(key)
                return scaled
            px, py = x >> up, y >> up
            parent_key = (source.name, pz, px, py)
            parent = self._cache.get(parent_key)
            if parent is None:
                if parent_key in self._pending:
                    return None                # being read; wait for it
                expiry = self._failed.get(parent_key)
                if expiry is not None and time.monotonic() < expiry:
                    continue                   # known to be missing: go up
                if local or up == beyond:
                    self.get(pz, px, py)       # ask for it; drawn next frame
                    if local:
                        # One level at a time. Asking for every level at once
                        # flooded the reader pool, and a provider destroyed
                        # with reads in flight deadlocked on the GIL.
                        return None
                continue
            # Below 1 px of the parent (more than 8 levels up) the crop is
            # that single pixel: a flat colour, which is all there is.
            size = max(1, TILE_PX >> up)
            crop = parent.copy(((x - (px << up)) * TILE_PX) >> up,
                               ((y - (py << up)) * TILE_PX) >> up, size, size)
            scaled = crop.scaled(TILE_PX, TILE_PX, Qt.IgnoreAspectRatio,
                                 Qt.SmoothTransformation)
            self._scaled[key] = scaled
            while len(self._scaled) > OVERZOOM_CACHE_MAX:
                self._scaled.popitem(last=False)
            return scaled
        return None

    def _request(self, key) -> None:
        source = self.source
        if source.type == "mbtiles":
            self._pending[key] = ""
            job = _MbtilesJob(key, self._mbtiles_paths)
            job.signals.done.connect(self._on_mbtiles)
            self._pool.start(job)
            return
        if source.needs_key:
            # No key in the environment: do not send a request that will be
            # rejected. Say why, once.
            self._mark_failed(key)
            self._note_failure(TileError.HTTP, "needs an API key: set the %s "
                               "environment variable" % source.key_env)
            return
        if self._inflight >= MAX_INFLIGHT:
            if key not in self._queue:
                self._queue.append(key)
            self._pending[key] = ""
            return
        self._send(key)

    def _send(self, key) -> None:
        url = self._tile_url(key)
        if not url:
            self._pending.pop(key, None)
            self._mark_failed(key)
            return
        fetcher = self._ensure_fetcher()
        fetcher.connect_timeout = TILE_CONNECT_TIMEOUT_S
        fetcher.read_timeout = float(getattr(self.source, "read_timeout",
                                             TILE_READ_TIMEOUT_S))
        fetcher.user_agent = self._user_agent
        self._pending[key] = url
        fetcher.submit(key, url)

    def _report_handler_fault(self, exc: BaseException) -> None:
        """Log a fault inside the result handler once, by type."""
        label = "%s: %s" % (type(exc).__name__, exc)
        if label in self._handler_faults:
            return
        self._handler_faults.add(label)
        self.log_error.emit("tile handler fault (this is a bug, not the "
                            "network): %s" % label)

    def _mark_failed(self, key) -> None:
        self._failed[key] = time.monotonic() + NEGATIVE_CACHE_S
        # Keep the negative cache from becoming its own leak over a long
        # session of panning: drop whatever has already expired.
        if len(self._failed) > 4096:
            now = time.monotonic()
            self._failed = {k: v for k, v in self._failed.items() if v > now}

    def _note_failure(self, kind: str, reason: str) -> None:
        """Record the latest reason; log the first of each kind once."""
        self.last_error = reason or "unknown error"
        self.last_error_kind = kind
        if kind not in self._reported_kinds:
            self._reported_kinds.add(kind)
            self.log_error.emit("%s: %s" % (
                self.source.name if self.source else "tile source",
                self.last_error))

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
        self._pending.pop(key, None)
        self._store(key, data)

    def _on_fetched(self, result: TileResult) -> None:
        """A network worker finished. GUI thread, via the fetcher's signal.

        Nothing here may raise silently: a fault in this handler is exactly
        what made a working network look dead, so it is caught, logged once
        as a bug, and the tile is counted -- never swallowed.
        """
        try:
            self._pending.pop(result.key, None)
            if result.ok:
                if result.from_cache:
                    self.cache_hits += 1
                was_failing = bool(self.last_error) or self.is_offline
                self.last_error = ""
                self.last_error_kind = ""
                if not self._store(result.key, result.data):
                    self._note_failure(TileError.BAD_IMAGE,
                                       "a tile arrived but would not decode")
                elif was_failing:
                    self._reported_kinds.clear()
                    self.recovered.emit(self.source.name if self.source
                                        else "tile source")
            else:
                self.network_errors += 1
                self._note_failure(result.kind, result.reason)
                self._store(result.key, None)
        except Exception as exc:                          # noqa: BLE001
            self.network_errors += 1
            self._report_handler_fault(exc)
        finally:
            while self._queue and self._inflight < MAX_INFLIGHT:
                key = self._queue.pop(0)
                if key in self._pending:
                    self._send(key)

    def _store(self, key, data) -> bool:
        """Decode and cache. True when a usable tile was stored."""
        if not data:
            self._mark_failed(key)
            self._consecutive_failures += 1
            self.generation += 1
            self.tile_ready.emit()
            return False
        image = QImage()
        if not image.loadFromData(data):
            self._mark_failed(key)
            self._consecutive_failures += 1
            self.generation += 1
            self.tile_ready.emit()
            return False
        self._consecutive_failures = 0
        self._cache[key] = image
        self._cache.move_to_end(key)
        self.generation += 1
        while len(self._cache) > TILE_CACHE_MAX:
            self._cache.popitem(last=False)
        self.tile_ready.emit()
        return True

    def clear_failures(self) -> None:
        """Let previously missing tiles be retried (e.g. network returned)."""
        self._failed.clear()
        self._consecutive_failures = 0

    def test_tile(self, z: int, x: int, y: int) -> TileResult:
        """Fetch one tile for the current source, bypassing every cache.

        Runs synchronously and is meant to be called from a worker; it is
        what the Test source button runs, so it must report exactly what the
        server said rather than what the cache remembers.
        """
        source = self.source
        if source is None or source.type == "mbtiles":
            return TileResult(key=None, kind=TileError.OTHER,
                              reason="the selected pack is offline; there is "
                                     "nothing to test over the network")
        url = self._tile_url((source.name, z, x, y))
        return fetch_tile((source.name, z, x, y), url, self._user_agent,
                          TILE_CONNECT_TIMEOUT_S, source.read_timeout,
                          cache=None)

    def shutdown(self) -> None:
        self.abort_pending()
        self._queue.clear()
        if self._fetcher is not None:
            self._fetcher.shutdown()
        try:
            self._pool.clear()
            self._pool.waitForDone(3000)
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class _ImageSignals(QObject):
    done = pyqtSignal(str, object, str)


class _ImageLoadJob(QRunnable):
    """Decodes a georeferenced image, and shrinks it if it is huge.

    A 10,000 x 10,000 px export is 400 MB as ARGB and would stall the GUI
    thread for seconds if decoded there. Nothing on screen can show more
    than a few thousand pixels across, so it is scaled down once, here.
    """

    #: Longest edge kept. Above this the image is scaled down on load.
    MAX_EDGE = 4096

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.signals = _ImageSignals()
        self.setAutoDelete(True)

    def run(self) -> None:
        image, error = None, ""
        try:
            loaded = QImage(self.path)
            if loaded.isNull():
                error = "could not decode %s" % os.path.basename(self.path)
            else:
                if max(loaded.width(), loaded.height()) > self.MAX_EDGE:
                    loaded = loaded.scaled(
                        self.MAX_EDGE, self.MAX_EDGE, Qt.KeepAspectRatio,
                        Qt.SmoothTransformation)
                image = loaded
        except Exception as exc:                          # noqa: BLE001
            error = "%s: %s" % (type(exc).__name__, exc)
        try:
            self.signals.done.emit(self.path, image, error)
        except RuntimeError:
            pass


class _TrackProjection:
    """World-pixel coordinates for one track at one zoom, grown in place.

    Projecting is the expensive part and the result only changes when the
    zoom changes or new fixes arrive, neither of which happens on most
    repaints. Panning does not invalidate it: world pixels are absolute, so
    a pan is a subtraction at draw time.
    """

    __slots__ = ("zoom", "revision", "wx", "wy")

    def __init__(self, zoom: int) -> None:
        self.zoom = zoom
        self.revision = -1
        self.wx = np.empty(0, dtype=np.float64)
        self.wy = np.empty(0, dtype=np.float64)

    def update(self, track: Track) -> None:
        """Project whatever is new. Existing points are never redone."""
        n = len(track)
        have = self.wx.shape[0]
        if self.revision == track.revision and have == n:
            return
        if n < have:
            # The track was thinned or cleared: the old projection no longer
            # corresponds to anything, so redo it.
            have = 0
        if have == 0:
            self.wx, self.wy = lonlat_to_world_array(track.lons, track.lats,
                                                     self.zoom)
        elif n > have:
            new_x, new_y = lonlat_to_world_array(
                track.lons[have:], track.lats[have:], self.zoom)
            self.wx = np.concatenate((self.wx, new_x))
            self.wy = np.concatenate((self.wy, new_y))
        self.revision = track.revision


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
        #: (name, lat, lon) of the launch pad, or None. Marked on the map,
        #: and the range origin whenever no ground station is set.
        self.launch_pad: Optional[Tuple[str, float, float]] = None
        #: Set by the paint that drew the launch-pad marker / rings (tests).
        self.pad_drawn = False
        self.rings_drawn = False
        self.coverage_drawn = False
        self.paint_errors = 0
        self._last_paint_error = ""

        self._drag_origin: Optional[QPoint] = None
        self._drag_center: Tuple[float, float] = (0.0, 0.0)

        #: The base layer under the tracks. None means the grid, which is
        #: the default and needs nothing: no file, no server, no network.
        self.pack: MapPack = GRID_PACK
        self.vector_pack: Optional[VectorPack] = None
        self.overlay: Optional[ImageOverlay] = None
        self.overlay_opacity = 1.0
        self.show_rings = True

        #: vehicle -> projected world pixels at the current zoom.
        self._projections: Dict[str, _TrackProjection] = {}
        #: vehicle -> (cache key, QPolygonF) for the thinned, clipped line.
        self._polylines: Dict[str, Tuple[Any, QPolygonF]] = {}
        #: Last paint's view, so an unchanged view can skip the work.
        self._last_view: Any = None
        self.last_paint_ms = 0.0

        # Redraw coalescing: intake sets a flag, the timer repaints.
        self._dirty = False
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / REDRAW_HZ))
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # One worker for image decoding, separate from the tile pool so a
        # slow image cannot starve the tiles.
        self._pool_image = QThreadPool(self)
        self._pool_image.setMaxThreadCount(1)

    # -- state -------------------------------------------------------------

    def request_redraw(self) -> None:
        """Mark the view stale. O(1); the timer does the actual painting."""
        self._dirty = True

    def _view_key(self) -> Tuple:
        """Everything a frame depends on, cheaply comparable.

        The vehicle labels count seconds, so the age is part of the key --
        at whole-second resolution, which is all the label shows. Without
        that the map would repaint five times a second to redraw an
        identical frame.
        """
        revisions = tuple(sorted((v, t.revision)
                                 for v, t in self.store.tracks.items()))
        last_epoch = 0.0
        for track in self.store.tracks.values():
            if len(track):
                last_epoch = max(last_epoch, float(track.epoch[len(track) - 1]))
        age_s = int(time.time() - last_epoch) if last_epoch else 0
        source = self.provider.source
        return (round(self.center_lat, 9), round(self.center_lon, 9),
                self.zoom, self.width(), self.height(), revisions, age_s,
                self.ground_station, self.launch_pad,
                source.name if source else None,
                self.provider.is_offline, self.provider.generation)

    def _tick(self) -> None:
        if self.follow:
            track = self.store.get(self.follow)
            if track and len(track):
                last = track.last
                self.center_lat, self.center_lon = last.lat, last.lon
        key = self._view_key()
        if self._dirty or key != self._last_view:
            self._dirty = False
            self._last_view = key
            self.update()

    def set_source(self, source: MapSource, base_dir: str = ".") -> None:
        self.provider.set_source(source, base_dir)
        self.clamp_zoom()
        self.request_redraw()

    def set_pack(self, pack: MapPack, base_dir: str = ".") -> str:
        """Switch the base layer. Returns "" or a message for the event log.

        A pack that will not load falls back to the grid rather than
        raising: an unreadable file must not be able to take the map away
        from an operator who is watching a flight.
        """
        self.pack = pack
        self.vector_pack = None
        self.overlay = None
        note = ""
        if pack.kind == "mbtiles":
            source = MapSource(name=pack.name, type="mbtiles", path=pack.path,
                               attribution=pack.attribution,
                               min_zoom=pack.min_zoom, max_zoom=pack.max_zoom)
            self.provider.set_source(source, base_dir)
            if not self.provider.offline_ready:
                note = ("Map pack %r could not be opened; showing the grid."
                        % pack.name)
                self.pack = GRID_PACK
        elif pack.kind == "vector":
            self.vector_pack = load_vector_pack(pack.path)
            self.provider.set_source(
                MapSource(name="none", type="mbtiles", path=""), base_dir)
            if self.vector_pack is None:
                note = ("Map pack %r is not a readable vector pack; showing "
                        "the grid." % pack.name)
                self.pack = GRID_PACK
        elif pack.kind == "image":
            self.overlay = load_image_overlay(pack.path)
            self.provider.set_source(
                MapSource(name="none", type="mbtiles", path=""), base_dir)
            if self.overlay is None:
                note = ("Map pack %r has no readable image or bounds; "
                        "showing the grid." % pack.name)
                self.pack = GRID_PACK
            else:
                self._start_overlay_load()
        else:
            self.provider.set_source(
                MapSource(name="none", type="mbtiles", path=""), base_dir)
        self.clamp_zoom()
        self.request_redraw()
        return note

    def _start_overlay_load(self) -> None:
        """Decode the image off the GUI thread; it may be very large."""
        overlay = self.overlay
        if overlay is None:
            return
        job = _ImageLoadJob(overlay.path)
        job.signals.done.connect(self._on_overlay_loaded)
        self._pool_image.start(job)

    def _on_overlay_loaded(self, path: str, image, error: str) -> None:
        if self.overlay is None or self.overlay.path != path:
            return                      # the operator changed pack meanwhile
        self.overlay.image = image
        self.overlay.error = error
        self.request_redraw()

    def invalidate_tracks(self) -> None:
        """Drop the projection and polyline caches (the store was cleared)."""
        self._projections.clear()
        self._polylines.clear()
        self.request_redraw()

    def set_zoom(self, zoom: int, anchor: Optional[QPoint] = None) -> None:
        low, high = self.zoom_range
        zoom = max(low, min(high, int(zoom)))
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
        low, high = self.zoom_range
        if box is None:
            self.center_lat, self.center_lon = self.ground_station
            self.zoom = max(low, min(high, 15))
            self.request_redraw()
            return
        min_lat, min_lon, max_lat, max_lon = box
        if self.ground_station is not None:
            gs_lat, gs_lon = self.ground_station
            min_lat, max_lat = min(min_lat, gs_lat), max(max_lat, gs_lat)
            min_lon, max_lon = min(min_lon, gs_lon), max(max_lon, gs_lon)
        self.center_lat = (min_lat + max_lat) / 2.0
        self.center_lon = (min_lon + max_lon) / 2.0
        for zoom in range(high, low - 1, -1):
            x0, y0 = lonlat_to_world(min_lon, max_lat, zoom)
            x1, y1 = lonlat_to_world(max_lon, min_lat, zoom)
            if (abs(x1 - x0) <= self.width() * 0.85
                    and abs(y1 - y0) <= self.height() * 0.85):
                self.zoom = zoom
                break
        else:
            self.zoom = low
        self.request_redraw()

    # -- coordinate helpers -------------------------------------------------

    @property
    def range_origin(self) -> Optional[Tuple[float, float]]:
        """Where range rings, distance and bearing are measured from: the
        ground station when one is set, otherwise the launch pad."""
        if self.ground_station is not None:
            return self.ground_station
        if self.launch_pad is not None:
            return self.launch_pad[1], self.launch_pad[2]
        return None

    @property
    def zoom_range(self) -> Tuple[int, int]:
        """The zoom levels the view may be at on the current map.

        For a tile pack or online source: from UNDERZOOM_LEVELS below its
        lowest zoom to OVERZOOM_LEVELS above its highest. Beyond those
        there is nothing worth drawing, so the view is not allowed there.
        Grid, vector and image packs draw at any zoom.
        """
        source = self.provider.source
        if (self.pack.kind != "mbtiles" or source is None
                or source.name == "none"):
            return MIN_ZOOM, MAX_ZOOM
        low = max(MIN_ZOOM, source.min_zoom - UNDERZOOM_LEVELS)
        high = min(MAX_ZOOM, source.max_zoom + OVERZOOM_LEVELS)
        return low, max(low, high)

    def clamp_zoom(self) -> None:
        """Bring the current zoom inside :attr:`zoom_range` (map changed)."""
        low, high = self.zoom_range
        if not low <= self.zoom <= high:
            self.zoom = max(low, min(high, self.zoom))
            self.request_redraw()

    @property
    def pack_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """(west, south, east, north) of the tile pack on screen, or None."""
        return self.pack.bounds if self.pack.kind == "mbtiles" else None

    def view_bounds(self) -> Tuple[float, float, float, float]:
        """(west, south, east, north) of what is on screen."""
        west, north = self._pixel_to_lonlat(QPoint(0, 0))
        east, south = self._pixel_to_lonlat(QPoint(self.width(), self.height()))
        return west, south, east, north

    def view_outside_pack(self) -> bool:
        """True when a tile pack is selected but none of it is in view."""
        bounds = self.pack_bounds
        if bounds is None:
            return False
        west, south, east, north = self.view_bounds()
        return (east < bounds[0] or west > bounds[2]
                or north < bounds[1] or south > bounds[3])

    def outside_pack(self, lat: float, lon: float) -> bool:
        bounds = self.pack_bounds
        return bounds is not None and not (
            bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3])

    def go_to_pad(self) -> None:
        """Centre on the launch pad at PAD_ZOOM (stops Follow)."""
        if self.launch_pad is None:
            return
        self.follow = ""
        _name, lat, lon = self.launch_pad
        self.center_lat, self.center_lon = lat, lon
        low, high = self.zoom_range
        self.zoom = max(low, min(high, PAD_ZOOM))
        self.request_redraw()

    @property
    def overzoom_levels(self) -> int:
        """How far the view is past the tile source's max zoom (0 = not)."""
        source = self.provider.source
        if (self.pack.kind != "mbtiles" or source is None
                or source.name == "none"):
            return 0
        return max(0, self.zoom - source.max_zoom)

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
        started = time.perf_counter()
        try:
            painter.begin(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.fillRect(self.rect(), QColor(COL_BG))
            self._draw_base(painter)
            self._draw_coverage(painter)
            self._draw_rings(painter)
            # The pad under the tracks: a vehicle sits on it before launch,
            # and its track must not be hidden by the marker and label.
            self._draw_launch_pad(painter)
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
            self.last_paint_ms = (time.perf_counter() - started) * 1000.0

    def _draw_base(self, painter: QPainter) -> None:
        """The layer under the tracks, whichever kind of pack is selected.

        The grid is drawn under every kind, so an image or vector pack that
        does not cover the whole view still has a readable background with
        coordinates rather than a blank one.
        """
        self._draw_grid(painter)
        if self.pack.kind == "mbtiles":
            self._draw_tiles(painter)
        elif self.pack.kind == "vector" and self.vector_pack is not None:
            try:
                self.vector_pack.draw(
                    painter, self.zoom, self._origin(), self.width(),
                    self.height(), lonlat_to_world_array)
            except Exception:                              # noqa: BLE001
                # A malformed pack must cost the frame, not the window.
                self.vector_pack = None
        elif self.pack.kind == "image" and self.overlay is not None:
            self.overlay.draw(painter, self._lonlat_to_pixel,
                              self.overlay_opacity)

    def _draw_grid(self, painter: QPainter) -> None:
        """Graticule, north arrow and range rings: always available."""
        top_left = self._pixel_to_lonlat(QPoint(0, 0))
        bottom_right = self._pixel_to_lonlat(QPoint(self.width(), self.height()))
        draw_graticule(painter, self.width(), self.height(),
                       (top_left[0], top_left[1]),
                       (bottom_right[0], bottom_right[1]),
                       lambda lon, lat: self._lonlat_to_pixel(lon, lat))
        draw_north_arrow(painter, self.width(), self.height())

    def _draw_coverage(self, painter: QPainter) -> None:
        """A thin outline of the tile pack's coverage, so a view at the edge
        or outside it shows where the imagery is."""
        bounds = self.pack_bounds
        if bounds is None:
            return
        west, south, east, north = bounds
        top_left = self._lonlat_to_pixel(west, north)
        bottom_right = self._lonlat_to_pixel(east, south)
        pen = QPen(QColor(COL_PAD), 1.5, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(top_left, bottom_right))
        self.coverage_drawn = True

    def _draw_rings(self, painter: QPainter) -> None:
        """Range rings around the ground station, or the launch pad.

        Drawn above the base layer, not with the grid beneath it: under a
        tile pack or an image they were simply painted over.
        """
        origin = self.range_origin
        if origin is None or not self.show_rings:
            return
        lat, lon = origin
        draw_range_rings(painter, lat, lon,
                         lambda a, b: self._lonlat_to_pixel(a, b),
                         metres_per_pixel(self.center_lat, self.zoom),
                         halo=self.pack.kind in ("mbtiles", "image"))
        self.rings_drawn = True

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

        wanted = set()
        source_name = self.provider.source.name if self.provider.source else ""
        bounds = self.pack_bounds
        covered = None
        if bounds is not None:
            # Tiles wholly outside the pack's bounds are not asked for or
            # drawn: the grid shows there, not a field of "no tile" squares.
            bx0, by0 = lonlat_to_world(bounds[0], bounds[3], self.zoom)
            bx1, by1 = lonlat_to_world(bounds[2], bounds[1], self.zoom)
            covered = (int(bx0 // TILE_PX), int(by0 // TILE_PX),
                       int(bx1 // TILE_PX), int(by1 // TILE_PX))
        for row in range(rows):
            for col in range(cols):
                tx, ty = first_x + col, first_y + row
                if 0 <= ty < span:
                    wanted.add((source_name, self.zoom, tx % span, ty))
        # Anything still in flight that is no longer on screen is wasted
        # work holding one of the few request slots.
        self.provider.abort_pending(keep=wanted)

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
                if covered is not None and not (
                        covered[0] <= wrapped <= covered[2]
                        and covered[1] <= ty <= covered[3]):
                    continue
                image = self.provider.get(self.zoom, wrapped, ty)
                if image is None or image.isNull():
                    image = self.provider.overzoomed(self.zoom, wrapped, ty)
                    if image is not None:
                        self.provider.overzoom_drawn += 1
                if image is not None and not image.isNull():
                    painter.drawImage(rect, image)
                    self.provider.tiles_drawn += 1
                else:
                    # A visible hole, not a plausible-looking blank: the
                    # operator must be able to tell "no map here" from
                    # "nothing is there". Semi-transparent so the grid
                    # underneath still gives coordinates.
                    painter.fillRect(rect, QColor(198, 206, 217, 210))
                    painter.setPen(QColor(COL_DIM))
                    # Outline only. drawRect also fills with whatever brush
                    # the previous layer left set (the north arrow's), which
                    # painted every missing tile dark over the graticule.
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(rect.adjusted(0, 0, -1, -1))
                    painter.drawText(rect, Qt.AlignCenter,
                                     "no tile\nz%d" % self.zoom)

    def _polyline_for(self, track: Track) -> Optional[QPolygonF]:
        """The track as screen-space points: projected, clipped, thinned.

        Three steps, all vectorised and all cached:

        1. project to world pixels once per zoom, appending only new fixes
           (:class:`_TrackProjection`);
        2. keep only what is inside the view, padded by a margin so the line
           still enters and leaves the edge correctly;
        3. drop points closer together than :data:`MIN_POINT_SPACING_PX`,
           since they land on the same stroke.

        After this the number of points handed to Qt depends on the size of
        the window, not on the length of the flight.
        """
        projection = self._projections.get(track.vehicle)
        if projection is None or projection.zoom != self.zoom:
            projection = _TrackProjection(self.zoom)
            self._projections[track.vehicle] = projection
        projection.update(track)
        if projection.wx.shape[0] < 2:
            return None

        ox, oy = self._origin()
        cache_key = (self.zoom, round(ox, 1), round(oy, 1), self.width(),
                     self.height(), track.revision)
        cached = self._polylines.get(track.vehicle)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        sx = projection.wx - ox
        sy = projection.wy - oy

        # Clip, with a margin so a segment crossing the edge is still drawn.
        margin = 256.0
        visible = ((sx >= -margin) & (sx <= self.width() + margin)
                   & (sy >= -margin) & (sy <= self.height() + margin))
        if not visible.any():
            polygon = QPolygonF()
            self._polylines[track.vehicle] = (cache_key, polygon)
            return polygon
        # Widen the kept run by one on each side so the line enters and
        # leaves the edge rather than stopping short of it.
        indices = np.flatnonzero(visible)
        lo = max(0, int(indices[0]) - 1)
        hi = min(sx.shape[0], int(indices[-1]) + 2)
        sx = sx[lo:hi]
        sy = sy[lo:hi]

        if sx.shape[0] > 2:
            # Thin by pixel distance from the last KEPT point. cumsum on the
            # step distances is an approximation of that and is one pass;
            # exactness is not needed to decide which sub-pixel points to
            # drop.
            step = np.hypot(np.diff(sx), np.diff(sy))
            travelled = np.concatenate(([0.0], np.cumsum(step)))
            buckets = np.floor(travelled / MIN_POINT_SPACING_PX)
            keep = np.concatenate(([True], np.diff(buckets) > 0))
            keep[-1] = True                     # never drop the live end
            sx = sx[keep]
            sy = sy[keep]
            if sx.shape[0] > MAX_DRAWN_POINTS:
                stride = int(np.ceil(sx.shape[0] / MAX_DRAWN_POINTS))
                sx = np.concatenate((sx[::stride], sx[-1:]))
                sy = np.concatenate((sy[::stride], sy[-1:]))

        polygon = QPolygonF([QPointF(float(x), float(y))
                             for x, y in zip(sx, sy)])
        self._polylines[track.vehicle] = (cache_key, polygon)
        return polygon

    def _draw_tracks(self, painter: QPainter) -> None:
        for track in self.store.non_empty():
            polygon = self._polyline_for(track)
            if polygon is None or polygon.size() < 2:
                continue
            colour = QColor(VEHICLE_COLOURS.get(track.vehicle,
                                                VEHICLE_COLOURS["GENERIC"]))
            pen = QPen(colour, 2.6)
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPolyline(polygon)

    def _draw_launch_pad(self, painter: QPainter) -> None:
        """An orange diamond with the pad's name on a white label."""
        if self.launch_pad is None:
            return
        name, lat, lon = self.launch_pad
        pos = self._lonlat_to_pixel(lon, lat)
        size = 8.0
        diamond = QPolygonF([QPointF(pos.x(), pos.y() - size),
                             QPointF(pos.x() + size, pos.y()),
                             QPointF(pos.x(), pos.y() + size),
                             QPointF(pos.x() - size, pos.y())])
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.setBrush(QColor(COL_PAD))
        painter.drawPolygon(diamond)
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        box = QRect(int(pos.x()) + 12, int(pos.y()) - 9,
                    metrics.horizontalAdvance(name) + 10, 18)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 222))
        painter.drawRect(box)
        painter.setPen(QColor(COL_PAD))
        painter.drawText(box.adjusted(5, 0, 0, 0),
                         Qt.AlignVCenter | Qt.AlignLeft, name)
        self.pad_drawn = True

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
            origin = self.range_origin
            if origin is not None:
                gs_lat, gs_lon = origin
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
        # A pale backing, as the attribution has: over satellite imagery a
        # bare dark line and label are close to invisible.
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 200))
        painter.drawRect(x0 - 6, y0 - 22, width + 12, 30)
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
        text = self.attribution_text()
        if not text:
            return
        self._paint_attribution(painter, text)

    def attribution_text(self) -> str:
        """The bottom-corner notice: provider attribution, plus any problem."""
        source = self.provider.source
        online = source is not None and source.type in ("xyz", "wms")
        text = self.pack.attribution or (source.attribution if source else "")
        if self.vector_pack is not None:
            text = self.vector_pack.attribution
        elif self.overlay is not None:
            text = ("%s — you are responsible for this image's licence"
                    % self.overlay.attribution)
        elif online:
            # An online provider's attribution is a condition of its licence
            # (OpenStreetMap's explicitly), so it is shown whatever else is
            # going on. Online sources travel as pack kind "mbtiles", which
            # is why this has to be decided by the source type: testing the
            # pack kind showed "could not be opened" over a working OSM map.
            text = source.attribution
        elif self.pack.kind == "mbtiles" and not self.provider.offline_ready:
            text = "Map pack could not be opened — showing the grid"
        if source and source.needs_key:
            text = "%s — set %s to use this source" % (text, source.key_env)
        if online and self.provider.is_offline:
            # The reason, and the attribution, together: replacing the
            # attribution with an error would drop the licence notice.
            text = "%s   ·   OFFLINE — %s" % (
                source.attribution, self.provider.last_error
                or "not reachable")
        return text or ""

    def _paint_attribution(self, painter: QPainter, text: str) -> None:
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
    #: ``(TileResult, url_label)`` from the Test source worker thread. Qt
    #: queues a signal emitted from another thread onto this one, so the
    #: dialog is always built on the GUI thread.
    test_result = pyqtSignal(object)

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

        bar.addWidget(QLabel("Map pack:"))
        self.pack_combo = QComboBox()
        self.pack_combo.setToolTip(
            "Everything found in %s, plus the grid.\n\n"
            "Grid only needs nothing and always works: a lat/lon graticule, "
            "scale bar, north arrow and range rings.\n"
            "Tiles (.mbtiles), vector (.osmpack.json) and image overlays "
            "are all offline. Online sources are listed last and are the "
            "only ones that touch the network." % MAPS_DIR)
        self.pack_combo.currentIndexChanged.connect(self._on_pack_chosen)
        bar.addWidget(self.pack_combo)

        bar.addWidget(QLabel("Follow:"))
        self.follow_combo = QComboBox()
        self.follow_combo.addItems(["Off", "CANSAT", "ROCKET"])
        self.follow_combo.currentTextChanged.connect(self._on_follow)
        bar.addWidget(self.follow_combo)

        self.fit_btn = QPushButton("FIT ALL")
        self.fit_btn.clicked.connect(self.canvas.fit_all)
        bar.addWidget(self.fit_btn)

        self.pad_btn = QPushButton("GO TO PAD")
        self.pad_btn.setToolTip(
            "Centre the map on the launch pad at zoom %d (stops Follow). The "
            "pad's position is \"launch_pad\" in %s." % (PAD_ZOOM, SETTINGS_FILE))
        self.pad_btn.clicked.connect(self._on_go_to_pad)
        bar.addWidget(self.pad_btn)

        self.opacity_label = QLabel("Opacity:")
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(100)
        self.opacity.setFixedWidth(90)
        self.opacity.setToolTip(
            "Fade the image overlay so the tracks stay readable over it.")
        self.opacity.valueChanged.connect(self._on_opacity)
        bar.addWidget(self.opacity_label)
        bar.addWidget(self.opacity)

        self.clear_btn = QPushButton("CLEAR TRACKS")
        self.clear_btn.setToolTip(
            "Forget the drawn tracks. The CSV log and the KML export are "
            "untouched; this only clears what is on the map.")
        self.clear_btn.clicked.connect(self._on_clear)
        bar.addWidget(self.clear_btn)

        self.retry_btn = QPushButton("RETRY NOW")
        self.retry_btn.setToolTip(
            "Forget the OFFLINE verdict and the 60 s negative cache for the "
            "selected online source, and fetch the visible tiles again.\n"
            "An offline source also retries by itself every %.0f s."
            % OFFLINE_RETRY_S)
        self.retry_btn.clicked.connect(self._on_retry)
        bar.addWidget(self.retry_btn)

        self.test_btn = QPushButton("TEST SOURCE")
        self.test_btn.setToolTip(
            "Fetch one tile for the current view from the selected source, "
            "bypassing every cache, and show the URL, HTTP status, size, "
            "time and the exact error if any. Runs off the GUI thread.")
        self.test_btn.clicked.connect(self._on_test_source)
        bar.addWidget(self.test_btn)

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

        # The first failure of each kind, and recovery, reach the event log;
        # the latest reason is on the status line.
        self.canvas.provider.log_error.connect(
            lambda text: self.log_message.emit("Map: %s" % text))
        self.canvas.provider.recovered.connect(
            lambda name: self.log_message.emit(
                "Map: %s is answering again — online." % name))
        self.test_result.connect(self._show_test_result)
        self._contact_warned = False

        self._load_settings()
        self._populate_packs()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(500)

    # -- sources -----------------------------------------------------------

    def _populate_packs(self) -> None:
        """List the offline packs first, then any online source.

        Nothing here touches the network: the entry selected at startup is
        always an offline one, so opening the map at the launch site makes
        no outbound request at all.
        """
        self.packs = find_packs(self.base_dir)
        self._online_offset = len(self.packs)
        self.pack_combo.blockSignals(True)
        self.pack_combo.clear()
        for pack in self.packs:
            self.pack_combo.addItem(pack.name)
        for source in self.sources:
            if source.type != "mbtiles":
                self.pack_combo.addItem("%s  (online)" % source.name)
        self.pack_combo.blockSignals(False)

        # Signals blocked: this is the startup default, not the operator's
        # choice, and must not be remembered as one.
        index = self._default_pack_index()
        self.pack_combo.blockSignals(True)
        self.pack_combo.setCurrentIndex(index)
        self.pack_combo.blockSignals(False)
        self._on_pack(index)

    def _default_pack_index(self) -> int:
        """Which pack to show at startup. Never an online source.

        A pack the operator chose from the list, if it is still there.
        Otherwise the best offline map: a tile pack covering the launch pad,
        then any tile pack, then a vector or image pack, and only then the
        grid. (A pack picked by this rule is not "chosen", so a tile pack
        added later still wins over one picked automatically before.)
        """
        if self._saved_pack and self._pack_chosen:
            for i, pack in enumerate(self.packs):
                if pack.name == self._saved_pack:
                    return i
        pad = self.canvas.launch_pad

        def covers_pad(pack: MapPack) -> bool:
            if pad is None or pack.bounds is None:
                return False
            west, south, east, north = pack.bounds
            return west <= pad[2] <= east and south <= pad[1] <= north

        for rule in (lambda p: p.kind == "mbtiles" and covers_pad(p),
                     lambda p: p.kind == "mbtiles",
                     lambda p: p.kind in ("vector", "image")):
            for i, pack in enumerate(self.packs):
                if rule(pack):
                    return i
        return 0

    def _on_pack_chosen(self, index: int) -> None:
        """The operator picked from the list: remember it as their choice."""
        self._pack_chosen = True
        self._on_pack(index)

    def _on_pack(self, index: int) -> None:
        if index < 0:
            return
        if index < len(self.packs):
            pack = self.packs[index]
            note = self.canvas.set_pack(pack, self.base_dir)
            if note:
                self.log_message.emit("Map: %s" % note)
            self._saved_pack = self.canvas.pack.name
            self._update_opacity_visibility()
            self._save_settings()
            return
        # An online source: the only path that makes a network request, and
        # only because the operator chose it.
        source_index = index - len(self.packs)
        online = [s for s in self.sources if s.type != "mbtiles"]
        if 0 <= source_index < len(online):
            source = online[source_index]
            self.canvas.pack = MapPack(name=source.name, kind="mbtiles",
                                       attribution=source.attribution)
            self.canvas.vector_pack = None
            self.canvas.overlay = None
            self.canvas.set_source(source, self.base_dir)
            self.canvas.provider.retry_source()
            self.log_message.emit(
                "Map source %r selected: this one uses the network, and the "
                "provider sees which tiles you view." % source.name)
            if source.needs_key:
                self.log_message.emit(
                    "Map source %r needs an API key: set the %s environment "
                    "variable." % (source.name, source.key_env))
        self._update_opacity_visibility()

    def _update_opacity_visibility(self) -> None:
        showing_image = self.canvas.overlay is not None
        self.opacity_label.setVisible(showing_image)
        self.opacity.setVisible(showing_image)

    def _on_opacity(self, value: int) -> None:
        self.canvas.overlay_opacity = value / 100.0
        self.canvas.request_redraw()

    def _on_go_to_pad(self) -> None:
        self.follow_combo.setCurrentText("Off")
        self.canvas.go_to_pad()

    def _on_follow(self, text: str) -> None:
        self.canvas.follow = "" if text == "Off" else text
        self.canvas.request_redraw()

    def _on_clear(self) -> None:
        self.canvas.store.clear()
        self.canvas.invalidate_tracks()
        self.log_message.emit("Map tracks cleared (logs and CSV untouched).")

    # -- online source tools -----------------------------------------------

    def _online_selected(self) -> bool:
        source = self.canvas.provider.source
        return (source is not None and source.type in ("xyz", "wms")
                and self.canvas.pack.kind == "mbtiles"
                and self.canvas.pack.name == source.name)

    def _on_retry(self) -> None:
        if not self._online_selected():
            self.log_message.emit("Map: Retry now applies to an online source; "
                                  "the selected pack is offline.")
            return
        self.canvas.provider.retry_source()
        self.canvas.request_redraw()
        self.log_message.emit("Map: retrying %s now."
                              % self.canvas.provider.source.name)

    def _on_test_source(self) -> None:
        """Fetch one tile for the centre of the view, off the GUI thread."""
        if not self._online_selected():
            QMessageBox.information(
                self, "Test source",
                "The selected map pack is offline, so there is no server to "
                "test.\n\nChoose an online source from the Map pack list "
                "first.")
            return
        canvas = self.canvas
        z = canvas.zoom
        x, y = lonlat_to_tile(canvas.center_lon, canvas.center_lat, z)
        provider = canvas.provider
        self.test_btn.setEnabled(False)
        self.test_btn.setText("TESTING…")

        def work():
            try:
                result = provider.test_tile(z, x, y)
            except Exception as exc:                      # noqa: BLE001
                result = TileResult(key=None, kind=TileError.OTHER,
                                    reason="%s: %s" % (type(exc).__name__, exc))
            try:
                self.test_result.emit(result)
            except RuntimeError:
                pass                     # window closed while testing

        threading.Thread(target=work, name="map-test-source",
                         daemon=True).start()

    def _show_test_result(self, result: TileResult) -> None:
        self.test_btn.setEnabled(True)
        self.test_btn.setText("TEST SOURCE")
        source = self.canvas.provider.source
        lines = [
            "Source:  %s" % (source.name if source else "?"),
            "URL:  %s" % (result.url or "(none)"),
            "HTTP status:  %s" % (result.status or "no response"),
            "Bytes:  %d" % (len(result.data) if result.data else 0),
            "Time:  %.0f ms" % result.elapsed_ms,
            "User-Agent:  %s" % self.canvas.provider.user_agent,
            "",
            "Result:  %s" % ("OK — a valid image tile" if result.ok
                             else "FAILED — %s" % (result.reason
                                                   or "unknown error")),
        ]
        self.last_test_text = "\n".join(lines)
        self.log_message.emit("Map: Test source %s — %s" % (
            source.name if source else "?",
            "OK, %d bytes in %.0f ms" % (len(result.data), result.elapsed_ms)
            if result.ok else result.reason))
        self._present_test_dialog(self.last_test_text, result.ok)

    def _present_test_dialog(self, text: str, ok: bool) -> None:
        """Show the Test source result. Non-modal, text selectable.

        Separate from the result handling so tests can check everything up
        to here: Qt's offscreen platform faults inside QMessageBox.show()
        (measured -- even a plain box), while the real Windows platform
        shows the same box without trouble.
        """
        box = QMessageBox(self)
        box.setWindowTitle("Test source")
        box.setIcon(QMessageBox.Information if ok else QMessageBox.Warning)
        box.setText(text)
        box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.setModal(False)
        box.show()
        self._last_test_box = box

    # -- ground station ----------------------------------------------------

    def _settings_path(self) -> str:
        return os.path.join(self.base_dir, SETTINGS_FILE)

    def _apply_contact(self) -> None:
        """Put the operator's contact into the User-Agent."""
        self.canvas.provider.user_agent = default_user_agent(self.osm_contact)

    def _load_settings(self) -> None:
        self._saved_pack = ""
        #: True when the remembered pack was picked by the operator rather
        #: than by the startup default.
        self._pack_chosen = False
        self.tile_contact = ""
        self._settings_data: Dict[str, Any] = {}
        self._apply_launch_pad(None)
        #: An email or URL where the OpenStreetMap operations team can reach
        #: whoever runs this station. Their tile usage policy asks for it in
        #: the User-Agent; it is the operator's to fill in, in
        #: map_settings.json, and is never guessed.
        self.osm_contact = ""
        try:
            with open(self._settings_path(), "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            self._apply_contact()
            return                      # no file, or unreadable: show nothing
        if isinstance(data, dict):
            self._settings_data = data
            self._saved_pack = str(data.get("map_pack", "") or "")
            self._pack_chosen = bool(data.get("map_pack_chosen", False))
            self.osm_contact = str(data.get(OSM_CONTACT_FIELD, "") or "").strip()
            self.tile_contact = str(data.get(TILE_CONTACT_FIELD, "") or "").strip()
            self._apply_launch_pad(data.get(LAUNCH_PAD_FIELD))
        self._apply_contact()
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

    def _apply_launch_pad(self, entry: Any) -> None:
        """Use *entry* ({"name", "lat", "lon"}) as the launch pad, or the
        default when it is missing or not a valid position."""
        pad = dict(DEFAULT_LAUNCH_PAD)
        if isinstance(entry, dict):
            try:
                lat, lon = float(entry.get("lat")), float(entry.get("lon"))
                if (math.isfinite(lat) and math.isfinite(lon)
                        and -90 <= lat <= 90 and -180 <= lon <= 180):
                    pad = {"name": str(entry.get("name") or pad["name"]),
                           "lat": lat, "lon": lon}
            except (TypeError, ValueError):
                pass
        self.launch_pad = pad
        self.canvas.launch_pad = (pad["name"], pad["lat"], pad["lon"])
        # Open over the pad: that is where the map pack is and where the
        # flight will be, rather than at an unrelated default.
        if not self.canvas.store.non_empty():
            self.canvas.center_lat, self.canvas.center_lon = pad["lat"], pad["lon"]
        self.canvas.request_redraw()

    def _save_settings(self) -> None:
        # Merged into what is already in the file, so keys this window does
        # not own (the attitude panel's SMOOTH choice, anything added by
        # hand) survive a save instead of being dropped.
        payload: Dict[str, Any] = {}
        try:
            with open(self._settings_path(), "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict):
                payload.update(existing)
        except Exception:
            pass
        payload.update({
            "ground_station": {"lat": self.gs_lat.value(),
                               "lon": self.gs_lon.value(),
                               "show": self.gs_check.isChecked()},
            LAUNCH_PAD_FIELD: dict(getattr(self, "launch_pad",
                                           DEFAULT_LAUNCH_PAD)),
            "map_pack": getattr(self, "_saved_pack", ""),
            "map_pack_chosen": bool(getattr(self, "_pack_chosen", False)),
            # Written back even when empty, so the fields are there to be
            # found and filled in rather than having to be known.
            OSM_CONTACT_FIELD: getattr(self, "osm_contact", ""),
            OSM_CONTACT_FIELD + "_help": OSM_CONTACT_HELP,
            TILE_CONTACT_FIELD: getattr(self, "tile_contact", ""),
            TILE_CONTACT_FIELD + "_help": TILE_CONTACT_HELP,
        })
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
            # len(track), not len(track.points): .points builds an object per
            # fix (38 ms at 40,000), and this runs twice a second even with
            # the map window closed.
            bits.append("%s %d fixes" % (track.vehicle, len(track)))
            last = track.last
            if last is not None and self.canvas.outside_pack(last.lat, last.lon):
                bits.append("%s outside map pack" % track.vehicle)
        if not store.non_empty():
            bits.append("no GPS fix yet")
        low, high = self.canvas.zoom_range
        if self.canvas.zoom >= high and high < MAX_ZOOM:
            bits.append("max zoom for this map")
        elif self.canvas.zoom <= low and low > MIN_ZOOM:
            bits.append("min zoom for this map")
        levels = self.canvas.overzoom_levels
        if levels:
            source = self.canvas.provider.source
            bits.append("overzoomed x%d from z%d" % (1 << levels,
                                                     source.max_zoom))
        if self.canvas.view_outside_pack():
            bits.append("outside map pack — GO TO PAD to return")
        if self.canvas.pack.kind == "grid":
            bits.append("grid only — no map pack loaded, see README")
        elif self.canvas.vector_pack is not None:
            bits.append("%s: %d features"
                        % (self.canvas.vector_pack.name,
                           self.canvas.vector_pack.feature_count))
        elif self.canvas.overlay is not None:
            if self.canvas.overlay.image is None:
                bits.append("image loading…")
            elif self.canvas.overlay.error:
                bits.append(self.canvas.overlay.error)
        provider = self.canvas.provider
        # The automatic retry rides on this 500 ms timer: one probe tile every
        # OFFLINE_RETRY_S while offline, never a retry storm.
        if provider.probe_if_due():
            self.canvas.request_redraw()
        reason = provider.status_text
        if reason:
            # The real reason, not "source unreachable".
            bits.append(reason)
        if provider.network_errors:
            bits.append("%d tile errors" % provider.network_errors)
        if (self._online_selected() and not self.osm_contact
                and "openstreetmap" in (provider.source.url or "").lower()):
            bits.append("set %s in %s (OSM asks for a contact)"
                        % (OSM_CONTACT_FIELD, SETTINGS_FILE))
            if not self._contact_warned:
                self._contact_warned = True
                self.log_message.emit(
                    "Map: OpenStreetMap's tile policy asks for a contact in the "
                    "User-Agent. Put your email or URL in \"%s\" in %s; until "
                    "then the project page is sent." % (OSM_CONTACT_FIELD,
                                                        SETTINGS_FILE))
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

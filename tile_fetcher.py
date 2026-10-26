"""
tile_fetcher.py
===============

Fetches map tiles over HTTP with the standard library, on a bounded thread
pool, and hands decoded bytes back to the GUI thread through a Qt signal.

WHY NOT QNetworkAccessManager
-----------------------------
Qt's networking works on this machine, but it puts HTTPS on Qt's own
OpenSSL build, which on Windows is a separate pair of DLLs that may simply
not be present -- the classic "TLS initialization failed" with no further
explanation. ``ssl.create_default_context()`` uses Python's own OpenSSL and
the Windows certificate store, so HTTPS here does not depend on a Qt build
detail. It also makes redirects, proxies and error classification ordinary
Python rather than Qt attributes that are easy to get subtly wrong.

The GUI thread never waits: every request runs on a worker, and results
arrive as a signal.

ERRORS ARE CLASSIFIED
---------------------
"source unreachable" is not a diagnosis. Every failure is sorted into a
:class:`TileError` kind with a sentence an operator can act on -- a 403 is
a blocked User-Agent, a timeout is a slow server, a certificate error is a
clock or a proxy. The map shows the most recent reason and logs the first
of each kind.
"""

from __future__ import annotations

import email.utils
import hashlib
import http.client
import os
import shutil
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

#: Connect timeout: how long to wait for the TCP/TLS handshake.
DEFAULT_CONNECT_TIMEOUT_S = 3.0

#: Read timeout, separate and longer. Bhuvan's WMS renders on demand and a
#: first tile can legitimately take several seconds.
DEFAULT_READ_TIMEOUT_S = 8.0

#: Disk cache ceiling. The OSM tile policy explicitly permits caching, and a
#: view already looked at then works with the network gone.
DISK_CACHE_MAX_BYTES = 500 * 1024 * 1024

#: Trim to this fraction of the cap when pruning, so pruning is occasional
#: rather than on every write once full.
DISK_CACHE_TRIM_TO = 0.85


class TileError:
    """The kinds of failure, so callers match on a constant, not a string."""

    NONE = "ok"
    DNS = "dns"
    REFUSED = "refused"
    TIMEOUT = "timeout"
    TLS = "tls"
    HTTP = "http"
    PROXY = "proxy"
    BAD_IMAGE = "bad-image"
    OTHER = "other"


@dataclass
class TileResult:
    """One attempt: what came back, or precisely why nothing did."""

    key: Any
    url: str = ""
    data: Optional[bytes] = None
    status: int = 0
    kind: str = TileError.NONE
    reason: str = ""
    elapsed_ms: float = 0.0
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.data is not None and self.kind == TileError.NONE


#: Leading bytes of the formats a tile server may legitimately send.
_IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n",        # PNG
    b"\xff\xd8\xff",              # JPEG
    b"GIF87a", b"GIF89a",         # GIF
    b"RIFF",                      # WebP (RIFF....WEBP)
)


def _is_image(content_type: str, payload: bytes) -> bool:
    """True when this really is an image, judged by its bytes.

    The Content-Type header is not enough either way. A WMS reports errors
    as XML with HTTP 200, and a captive portal or a broken proxy will serve
    an HTML page labelled image/png -- the test server does exactly that,
    and trusting the header let it through as a tile. Some servers send a
    real tile with no type at all. So the magic bytes decide, always.
    """
    return any(payload.startswith(magic) for magic in _IMAGE_MAGIC)


def classify(exc: BaseException) -> Tuple[str, str]:
    """Turn an exception into (kind, a sentence worth showing an operator)."""
    if isinstance(exc, urllib.error.HTTPError):
        return TileError.HTTP, http_reason(exc.code)
    if isinstance(exc, ssl.SSLCertVerificationError):
        return (TileError.TLS,
                "certificate not trusted (%s) — check the system clock or a "
                "proxy that intercepts HTTPS" % exc.verify_message)
    if isinstance(exc, ssl.SSLError):
        return TileError.TLS, "TLS error: %s" % (exc.reason or exc)
    if isinstance(exc, socket.timeout):
        return TileError.TIMEOUT, "timed out"
    if isinstance(exc, urllib.error.URLError):
        inner = exc.reason
        if isinstance(inner, BaseException):
            kind, reason = classify(inner)
            # A proxy that is configured but down looks like a refused
            # connection; say so, because the fix is different.
            if kind == TileError.REFUSED and urllib.request.getproxies():
                return (TileError.PROXY,
                        "cannot reach the configured proxy (%s)" % reason)
            return kind, reason
        return TileError.OTHER, str(inner or exc)
    if isinstance(exc, socket.gaierror):
        return (TileError.DNS,
                "host not found (DNS) — no network, or the name is wrong")
    if isinstance(exc, ConnectionRefusedError):
        return TileError.REFUSED, "connection refused"
    if isinstance(exc, ConnectionResetError):
        return TileError.REFUSED, "connection reset by the server"
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 10013:
        return TileError.REFUSED, "blocked locally (firewall)"
    return TileError.OTHER, "%s: %s" % (type(exc).__name__, exc)


def http_reason(status: int) -> str:
    """A plain-words reason for an HTTP status, not just the number."""
    if status == 403:
        return ("HTTP 403 (blocked — the provider rejected this client; "
                "check the User-Agent and their usage policy)")
    if status == 404:
        return "HTTP 404 (no tile here at this zoom)"
    if status == 429:
        return ("HTTP 429 (rate limited — too many requests; back off or "
                "use an offline pack)")
    if status in (401, 407):
        return "HTTP %d (authentication required)" % status
    if 500 <= status <= 599:
        return "HTTP %d (the server failed)" % status
    return "HTTP %d" % status


class _ReadTimeoutHTTPSConnection(http.client.HTTPSConnection):
    """Separate connect and read timeouts.

    ``urllib`` applies one timeout to every socket operation. A launch-site
    laptop wants to find out quickly that nothing is listening (3 s), while
    still allowing a slow renderer to finish a tile (8 s).
    """

    read_timeout = DEFAULT_READ_TIMEOUT_S

    def connect(self) -> None:
        super().connect()
        if self.sock is not None:
            self.sock.settimeout(self.read_timeout)


class _ReadTimeoutHTTPConnection(http.client.HTTPConnection):
    read_timeout = DEFAULT_READ_TIMEOUT_S

    def connect(self) -> None:
        super().connect()
        if self.sock is not None:
            self.sock.settimeout(self.read_timeout)


def build_opener(read_timeout: float,
                 context: Optional[ssl.SSLContext] = None):
    """An opener that follows redirects, honours the system proxy and TLS."""
    context = context or ssl.create_default_context()

    https_conn = type("_Conn", (_ReadTimeoutHTTPSConnection,),
                      {"read_timeout": read_timeout})
    http_conn = type("_Conn", (_ReadTimeoutHTTPConnection,),
                     {"read_timeout": read_timeout})

    class _HTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(https_conn, req, context=context)

    class _HTTPHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(http_conn, req)

    return urllib.request.build_opener(
        # getproxies() reads the Windows registry settings, so a laptop
        # configured for a campus proxy works without being told twice.
        urllib.request.ProxyHandler(urllib.request.getproxies()),
        _HTTPHandler(), _HTTPSHandler(),
        # Redirects are followed by default; Qt5 did not follow them unless
        # asked, which is one of the things that made this fragile.
        urllib.request.HTTPRedirectHandler(),
    )


class DiskCache:
    """A bounded least-recently-used cache of tile bytes on disk.

    The OSM tile usage policy permits caching, and it is what makes a view
    already looked at usable with no network at all. Entries are plain files
    named by a hash of the URL, so the cache survives restarts and can be
    deleted by hand without breaking anything.
    """

    def __init__(self, directory: str,
                 max_bytes: int = DISK_CACHE_MAX_BYTES) -> None:
        self.directory = directory
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _path(self, url: str) -> str:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        # Two-level fan-out: a flat directory of 100k files is slow to list
        # on Windows and awkward to inspect.
        return os.path.join(self.directory, digest[:2], digest[2:] + ".tile")

    def get(self, url: str) -> Optional[bytes]:
        path = self._path(url)
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            self.misses += 1
            return None
        if not data:
            self.misses += 1
            return None
        try:
            # Touch, so least-recently-used means what it says.
            os.utime(path, None)
        except OSError:
            pass
        self.hits += 1
        return data

    def put(self, url: str, data: bytes) -> None:
        if not data:
            return
        path = self._path(url)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
            self.writes += 1
        except OSError:
            return
        if self.writes % 200 == 0:
            self.prune()

    def total_bytes(self) -> int:
        total = 0
        for root, _dirs, files in os.walk(self.directory):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
        return total

    def prune(self) -> int:
        """Drop the least recently used entries down to the trim fraction."""
        with self._lock:
            entries: List[Tuple[float, int, str]] = []
            total = 0
            for root, _dirs, files in os.walk(self.directory):
                for name in files:
                    path = os.path.join(root, name)
                    try:
                        stat = os.stat(path)
                    except OSError:
                        continue
                    entries.append((stat.st_mtime, stat.st_size, path))
                    total += stat.st_size
            if total <= self.max_bytes:
                return 0
            target = int(self.max_bytes * DISK_CACHE_TRIM_TO)
            removed = 0
            for _mtime, size, path in sorted(entries):
                if total <= target:
                    break
                try:
                    os.remove(path)
                except OSError:
                    continue
                total -= size
                removed += 1
            return removed

    def clear(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


class _FetchSignals(QObject):
    done = pyqtSignal(object)            # TileResult


class _FetchJob(QRunnable):
    """One tile, on a worker thread. Never raises into the pool."""

    def __init__(self, key: Any, url: str, user_agent: str,
                 connect_timeout: float, read_timeout: float,
                 cache: Optional[DiskCache], is_cancelled) -> None:
        super().__init__()
        self.key = key
        self.url = url
        self.user_agent = user_agent
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.cache = cache
        self.is_cancelled = is_cancelled
        self.signals = _FetchSignals()
        self.setAutoDelete(True)

    def run(self) -> None:
        result = fetch_tile(self.key, self.url, self.user_agent,
                            self.connect_timeout, self.read_timeout,
                            self.cache, self.is_cancelled)
        try:
            self.signals.done.emit(result)
        except RuntimeError:
            pass                       # the window closed mid-flight


def fetch_tile(key: Any, url: str, user_agent: str,
               connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_S,
               read_timeout: float = DEFAULT_READ_TIMEOUT_S,
               cache: Optional[DiskCache] = None,
               is_cancelled=None) -> TileResult:
    """Fetch one tile. Pure enough to call straight from a test."""
    started = time.perf_counter()
    result = TileResult(key=key, url=url)

    if cache is not None:
        cached = cache.get(url)
        if cached is not None:
            result.data = cached
            result.from_cache = True
            result.status = 200
            result.elapsed_ms = (time.perf_counter() - started) * 1000.0
            return result

    if is_cancelled is not None and is_cancelled(key):
        result.kind = TileError.OTHER
        result.reason = "no longer on screen"
        return result

    opener = build_opener(read_timeout)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with opener.open(request, timeout=connect_timeout) as response:
            payload = response.read()
            result.status = getattr(response, "status", 200) or 200
            content_type = response.headers.get("Content-Type", "") or ""
        if not payload:
            result.kind = TileError.BAD_IMAGE
            result.reason = "empty response"
        elif not _is_image(content_type, payload):
            result.kind = TileError.BAD_IMAGE
            snippet = payload[:120].decode("utf-8", "replace").strip()
            result.reason = ("not an image (%s) — %s"
                             % (content_type or "no content type", snippet[:80]))
        else:
            result.data = payload
            if cache is not None:
                cache.put(url, payload)
    except BaseException as exc:                        # noqa: BLE001
        kind, reason = classify(exc)
        result.kind = kind
        result.reason = reason
        if isinstance(exc, urllib.error.HTTPError):
            result.status = exc.code
    result.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result


class TileFetcher(QObject):
    """Bounded pool of tile workers. The GUI thread only ever enqueues."""

    #: ``(TileResult,)`` on the GUI thread, for every attempt.
    finished = pyqtSignal(object)

    def __init__(self, cache_dir: str, user_agent: str,
                 workers: int = 6, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.user_agent = user_agent
        self.cache = DiskCache(cache_dir)
        self.connect_timeout = DEFAULT_CONNECT_TIMEOUT_S
        self.read_timeout = DEFAULT_READ_TIMEOUT_S
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(int(workers))
        self._wanted: set = set()
        self._lock = threading.Lock()
        self.in_flight = 0

    # -- what is still worth fetching --------------------------------------

    def set_wanted(self, keys) -> None:
        """The tiles currently on screen. Anything else is abandoned.

        A worker already running checks this before it opens a socket, so a
        pan does not leave six workers fetching tiles nobody will see.
        """
        with self._lock:
            self._wanted = set(keys)

    def _cancelled(self, key) -> bool:
        with self._lock:
            return bool(self._wanted) and key not in self._wanted

    # -- submitting --------------------------------------------------------

    def submit(self, key: Any, url: str) -> None:
        job = _FetchJob(key, url, self.user_agent, self.connect_timeout,
                        self.read_timeout, self.cache, self._cancelled)
        job.signals.done.connect(self._on_done)
        self.in_flight += 1
        self._pool.start(job)

    def _on_done(self, result: TileResult) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.finished.emit(result)

    def shutdown(self) -> None:
        try:
            self._pool.clear()
            self._pool.waitForDone(5000)
        except RuntimeError:
            pass

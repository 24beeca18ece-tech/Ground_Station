"""
tools/test_tile_network.py
==========================

Tests for the online tile path, against a local HTTP server rather than a
real provider -- so they are deterministic, need no internet, and never
spend someone else's bandwidth.

    python -m pytest tools/test_tile_network.py -v

The server serves one real PNG and a route per failure mode: redirects,
403, 404, 429, 500, a response slower than the read timeout, and garbage
bytes with an image content type. Each case asserts the *reason* an
operator would be shown, not merely that something failed.
"""

from __future__ import annotations

import http.server
import os
import socket
import sys
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import map_window as mw  # noqa: E402
import tile_fetcher as tf  # noqa: E402
from track_store import TrackStore  # noqa: E402

def _make_png() -> bytes:
    """A genuinely valid 1x1 PNG, built here rather than pasted.

    Decoding is part of what these tests exercise, so the bytes have to be
    real: a PNG with a bad IDAT checksum fails in libpng and would make the
    provider look broken when it is not. (The signature is written as a
    list of ints so no escape sequence can be mangled on the way in.)
    """
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    signature = bytes([137, 80, 78, 71, 13, 10, 26, 10])
    return (signature
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(2)))
            + chunk(b"IEND", b""))


PNG = _make_png()

UA = "GCS-test/1.0 (+test)"


class _Handler(http.server.BaseHTTPRequestHandler):
    """One route per failure mode. Quiet: no stderr log per request."""

    slow_seconds = 5.0

    def log_message(self, *args):                       # noqa: A003
        pass

    def _send(self, status, body=b"", ctype="image/png"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):                                   # noqa: N802
        path = self.path.split("?")[0]
        if path.startswith("/ok"):
            self._send(200, PNG)
        elif path == "/redirect301":
            self.send_response(301)
            self.send_header("Location", "/ok.png")
            self.end_headers()
        elif path == "/redirect302":
            self.send_response(302)
            self.send_header("Location", "/ok.png")
            self.end_headers()
        elif path == "/403":
            self._send(403, b"forbidden", "text/plain")
        elif path == "/404":
            self._send(404, b"nope", "text/plain")
        elif path == "/429":
            self._send(429, b"slow down", "text/plain")
        elif path == "/500":
            self._send(500, b"boom", "text/plain")
        elif path == "/slow":
            time.sleep(self.slow_seconds)
            self._send(200, PNG)
        elif path == "/garbage":
            # An image content type with bytes that are not an image: what a
            # broken proxy or a captive portal actually sends.
            self._send(200, b"<html>not a tile at all</html>", "image/png")
        elif path == "/xml":
            # What a WMS sends for an error: HTTP 200 and XML.
            self._send(200, b"<ServiceExceptionReport/>", "text/xml")
        else:
            self._send(404, b"", "text/plain")


@pytest.fixture(scope="module")
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:%d" % httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def cache(tmp_path):
    return tf.DiskCache(str(tmp_path / "tiles"))


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def free_port() -> int:
    """A port with nothing listening, for the refused-connection case."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# ---------------------------------------------------------------------------
# The regression this all started from
# ---------------------------------------------------------------------------

def test_a_successful_reply_is_cached_and_not_counted_as_an_error(qapp, server):
    """The bug: a perfectly good tile was recorded as a failure.

    Qt's header() returns a str, bytes(str) raised TypeError, and the bare
    except turned every arriving tile into a counted error -- twelve of
    which tripped OFFLINE. This drives a real reply through the provider.
    """
    provider = mw.TileProvider()
    provider.set_source(mw.MapSource(name="local", type="xyz",
                                     url=server + "/ok/{z}/{x}/{y}.png",
                                     min_zoom=0, max_zoom=19), ".")
    provider.get(13, 5800, 3308)
    deadline = time.time() + 15
    while time.time() < deadline and provider._pending:
        qapp.processEvents()
        time.sleep(0.02)

    assert len(provider._cache) == 1, "a good tile was not cached"
    assert provider.network_errors == 0, "a good tile was counted as an error"
    assert provider._consecutive_failures == 0
    assert provider._failed == {}
    assert provider.is_offline is False
    provider.shutdown()


def test_twelve_good_tiles_do_not_trip_offline(qapp, server):
    provider = mw.TileProvider()
    provider.set_source(mw.MapSource(name="local", type="xyz",
                                     url=server + "/ok/{z}/{x}/{y}.png",
                                     min_zoom=0, max_zoom=19), ".")
    for i in range(14):
        provider.get(13, 5800 + i, 3308)
    deadline = time.time() + 25
    while time.time() < deadline and provider._pending:
        qapp.processEvents()
        time.sleep(0.02)
    assert provider.is_offline is False
    assert provider.network_errors == 0
    provider.shutdown()


def test_a_wms_xml_error_with_http_200_is_still_a_failure(qapp, server):
    """A WMS reports failure as XML with status 200; status alone is no test."""
    provider = mw.TileProvider()
    provider.set_source(mw.MapSource(name="local", type="xyz",
                                     url=server + "/xml", min_zoom=0,
                                     max_zoom=19), ".")
    provider.get(13, 1, 1)
    deadline = time.time() + 15
    while time.time() < deadline and provider._pending:
        qapp.processEvents()
        time.sleep(0.02)
    assert len(provider._cache) == 0
    assert provider.network_errors >= 1
    provider.shutdown()


def test_a_handler_fault_is_reported_not_swallowed(qapp):
    """A bug in the reply handler must not look like a dead network."""
    provider = mw.TileProvider()
    messages = []
    provider.log_error.connect(messages.append)
    provider._report_handler_fault(TypeError("string argument without an encoding"))
    provider._report_handler_fault(TypeError("string argument without an encoding"))
    assert len(messages) == 1, "the same fault was reported twice"
    assert "bug, not the network" in messages[0]
    assert "TypeError" in messages[0]


# ---------------------------------------------------------------------------
# fetch_tile: one case per failure mode
# ---------------------------------------------------------------------------

def test_fetch_succeeds_and_returns_the_bytes(server, cache):
    result = tf.fetch_tile("k", server + "/ok.png", UA, cache=cache)
    assert result.ok and result.data == PNG
    assert result.status == 200 and result.kind == tf.TileError.NONE
    assert result.elapsed_ms >= 0


@pytest.mark.parametrize("route", ["/redirect301", "/redirect302"])
def test_redirects_are_followed(server, cache, route):
    result = tf.fetch_tile("k", server + route, UA, cache=cache)
    assert result.ok, result.reason
    assert result.data == PNG


def test_403_says_blocked_and_names_the_user_agent(server, cache):
    result = tf.fetch_tile("k", server + "/403", UA, cache=cache)
    assert not result.ok
    assert result.kind == tf.TileError.HTTP
    assert result.status == 403
    assert "403" in result.reason and "User-Agent" in result.reason


def test_404_says_no_tile_here(server, cache):
    result = tf.fetch_tile("k", server + "/404", UA, cache=cache)
    assert result.kind == tf.TileError.HTTP and result.status == 404
    assert "404" in result.reason


def test_429_says_rate_limited(server, cache):
    result = tf.fetch_tile("k", server + "/429", UA, cache=cache)
    assert result.kind == tf.TileError.HTTP and result.status == 429
    assert "rate limited" in result.reason


def test_500_says_the_server_failed(server, cache):
    result = tf.fetch_tile("k", server + "/500", UA, cache=cache)
    assert result.kind == tf.TileError.HTTP and result.status == 500
    assert "server failed" in result.reason


def test_a_slow_response_times_out_on_the_read_timeout(server, cache):
    started = time.perf_counter()
    result = tf.fetch_tile("k", server + "/slow", UA, connect_timeout=3.0,
                           read_timeout=1.0, cache=cache)
    elapsed = time.perf_counter() - started
    assert result.kind == tf.TileError.TIMEOUT, result.reason
    assert "timed out" in result.reason
    # The read timeout, not the server's 5 s sleep, is what ends it.
    assert elapsed < 4.0, "waited %.1f s despite a 1 s read timeout" % elapsed


def test_a_closed_port_is_reported_as_refused_or_timeout(cache):
    """Windows does not always send RST for a closed loopback port.

    Measured here: the connect sat until the 2 s connect timeout instead of
    being refused. Both are correct diagnoses of "nothing is listening", and
    both carry a reason an operator can act on; what must not happen is a
    bare "unreachable".
    """
    port = free_port()
    result = tf.fetch_tile("k", "http://127.0.0.1:%d/ok.png" % port, UA,
                           connect_timeout=2.0, cache=cache)
    assert result.kind in (tf.TileError.REFUSED, tf.TileError.PROXY,
                           tf.TileError.TIMEOUT), result.reason
    assert result.reason, "a failure with no reason is what we are fixing"


def test_garbage_bytes_are_rejected_as_a_bad_image(server, cache):
    result = tf.fetch_tile("k", server + "/garbage", UA, cache=cache)
    assert result.kind == tf.TileError.BAD_IMAGE
    assert "not an image" in result.reason


def test_an_unresolvable_host_is_reported_as_dns(cache):
    result = tf.fetch_tile(
        "k", "http://no-such-host.invalid./ok.png", UA,
        connect_timeout=3.0, cache=cache)
    assert result.kind in (tf.TileError.DNS, tf.TileError.OTHER)


def test_http_reason_wording():
    assert "blocked" in tf.http_reason(403)
    assert "rate limited" in tf.http_reason(429)
    assert "server failed" in tf.http_reason(503)
    assert tf.http_reason(418) == "HTTP 418"


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def test_a_cached_tile_is_served_with_the_network_gone(server, cache):
    url = server + "/ok.png"
    first = tf.fetch_tile("k", url, UA, cache=cache)
    assert first.ok and not first.from_cache

    # Point at a dead port under the SAME url key by fetching again after
    # the server is unreachable: the cache is keyed on the url, so the
    # second call must not touch the network at all.
    second = tf.fetch_tile("k", url, UA, connect_timeout=0.1,
                           read_timeout=0.1, cache=cache)
    assert second.ok and second.from_cache
    assert second.data == PNG


def test_the_cache_survives_a_new_cache_object(server, tmp_path):
    directory = str(tmp_path / "tiles")
    url = server + "/ok.png"
    tf.fetch_tile("k", url, UA, cache=tf.DiskCache(directory))
    fresh = tf.DiskCache(directory)
    assert fresh.get(url) == PNG


def test_the_cache_prunes_to_its_cap(tmp_path):
    cache = tf.DiskCache(str(tmp_path / "tiles"), max_bytes=4096)
    blob = b"x" * 1024
    for i in range(20):
        cache.put("http://e/%d" % i, blob)
        time.sleep(0.002)          # distinct mtimes, so LRU order is defined
    cache.prune()
    assert cache.total_bytes() <= 4096


def test_the_cache_ignores_an_empty_payload(cache):
    cache.put("http://e/x", b"")
    assert cache.get("http://e/x") is None


def test_a_corrupt_cache_directory_does_not_raise(tmp_path):
    path = tmp_path / "notadir"
    path.write_text("I am a file", encoding="utf-8")
    cache = tf.DiskCache(str(path))
    assert cache.get("http://e/x") is None
    cache.put("http://e/x", b"data")          # must not raise


# ---------------------------------------------------------------------------
# The GUI thread must not block
# ---------------------------------------------------------------------------

def test_the_gui_thread_is_not_blocked_by_a_slow_source(qapp, server):
    """A 5 s server response must not stall the GUI thread.

    The heartbeat is sampled while a fetch is outstanding; the worst gap is
    what an operator would feel as a freeze.
    """
    from PyQt5.QtCore import QTimer
    fetcher = tf.TileFetcher(cache_dir=None or "", user_agent=UA, workers=6)
    gaps = []
    last = [time.perf_counter()]

    def beat():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now

    timer = QTimer()
    timer.timeout.connect(beat)
    timer.start(10)

    done = []
    fetcher.finished.connect(lambda r: done.append(r))
    fetcher.read_timeout = 2.0
    for i in range(6):
        fetcher.submit(("k", i), server + "/slow")

    deadline = time.time() + 10
    while time.time() < deadline and len(done) < 6:
        qapp.processEvents()
        time.sleep(0.005)
    timer.stop()
    fetcher.shutdown()

    assert len(done) == 6, "only %d of 6 finished" % len(done)
    worst = max(gaps) if gaps else 0.0
    assert worst < 0.1, "GUI thread blocked for %.0f ms" % (worst * 1000)


def test_tiles_no_longer_on_screen_are_skipped(server):
    fetcher = tf.TileFetcher(cache_dir="", user_agent=UA, workers=2)
    fetcher.set_wanted({("want", 1)})
    result = tf.fetch_tile(("gone", 2), server + "/ok.png", UA,
                           is_cancelled=fetcher._cancelled)
    assert not result.ok
    assert "no longer on screen" in result.reason
    fetcher.shutdown()

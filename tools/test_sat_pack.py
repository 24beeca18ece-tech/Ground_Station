"""
tools/test_sat_pack.py
======================

The launch-pad satellite pack: Sentinel-2 cloudless (EOX) tiles in an
MBTiles file, the launch pad marker and range origin, overzoom past the
pack's max zoom, the polite downloader -- and that the whole thing runs with
no network at all.

The real pack (assets/maps/launchpad_s2.mbtiles) is gitignored and rebuilt
locally, so these tests build small packs of their own; the one test that
reads the real pack is skipped when it is absent.

    python -m pytest tools/test_sat_pack.py -v
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import map_window as mw  # noqa: E402
from track_store import TrackStore  # noqa: E402

PAD_LAT, PAD_LON = 26.720028, 84.304583
EOX = mw.S2CLOUDLESS_ATTRIBUTION
REAL_PACK = os.path.join(ROOT, "assets", "maps", "launchpad_s2.mbtiles")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def jpeg_tile(colour, quadrants=None) -> bytes:
    """A 256 px JPEG, one colour, or four quadrant colours (TL, TR, BL, BR)."""
    from PyQt5.QtCore import QBuffer, QByteArray, QIODevice
    from PyQt5.QtGui import QColor, QImage, QPainter
    image = QImage(256, 256, QImage.Format_RGB32)
    image.fill(QColor(colour))
    if quadrants:
        painter = QPainter(image)
        for (qx, qy), col in zip(((0, 0), (128, 0), (0, 128), (128, 128)),
                                 quadrants):
            painter.fillRect(qx, qy, 128, 128, QColor(col))
        painter.end()
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "JPG", 95)
    return bytes(data)


def make_pack(path, tiles, zmin=0, zmax=14, bounds=(83.8012, 26.2704,
                                                       84.8080, 27.1697)):
    """An MBTiles file written by the same code fetch_tiles uses."""
    import fetch_tiles
    conn = fetch_tiles.open_mbtiles(
        str(path), "Launch pad test pack", "test", bounds, zmin, zmax,
        fmt="jpg", attribution=EOX)
    for (z, x, y), data in tiles.items():
        fetch_tiles.store_tile(conn, z, x, y, data)
    conn.commit()
    conn.close()
    return str(path)


def pad_tile(z):
    return mw.lonlat_to_tile(PAD_LON, PAD_LAT, z)


def wait_for(qapp, predicate, frames=200):
    import time
    for _ in range(frames):
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def settled(qapp, window):
    """Show the window and let its layout size the canvas, so coordinates
    taken afterwards stay valid (a hidden window lays out on first render)."""
    window.resize(980, 720)
    window.show()
    qapp.processEvents()
    return window.canvas


def grab(canvas):
    from PyQt5.QtGui import QImage
    image = QImage(canvas.width(), canvas.height(), QImage.Format_ARGB32)
    canvas.render(image)
    return image


def pixel(image, x, y):
    from PyQt5.QtGui import QColor
    return QColor(image.pixel(x, y))


def close_enough(colour, rgb, tol=40):
    return all(abs(a - b) <= tol for a, b in
               zip((colour.red(), colour.green(), colour.blue()), rgb))


# ---------------------------------------------------------------------------
# Pack reading
# ---------------------------------------------------------------------------

def test_pack_reading_flips_y_and_reads_the_metadata(qapp, tmp_path):
    z = 3
    x, y = 5, 2                      # slippy y 2 is TMS row 5 at z3
    path = make_pack(tmp_path / "p.mbtiles", {(z, x, y): jpeg_tile("#ff0000"),
                                              (z, x, 5): jpeg_tile("#0000ff")})
    row = sqlite3.connect(path).execute(
        "SELECT tile_row FROM tiles WHERE zoom_level=3 AND tile_column=5 "
        "ORDER BY tile_row").fetchall()
    assert [r[0] for r in row] == [2, 5]          # stored as TMS rows
    from PyQt5.QtGui import QImage
    image = QImage()
    assert image.loadFromData(mw.read_mbtiles_tile(path, z, x, y))
    assert close_enough(pixel(image, 128, 128), (255, 0, 0))   # not mirrored
    pack = mw.mbtiles_pack("p.mbtiles", path)
    assert pack.attribution == EOX
    assert (pack.min_zoom, pack.max_zoom) == (0, 14)
    assert pack.bounds == pytest.approx((83.8012, 26.2704, 84.8080, 27.1697))
    meta = mw.read_mbtiles_metadata(path)
    assert meta["format"] == "jpg"


@pytest.mark.skipif(not os.path.exists(REAL_PACK),
                    reason="launchpad_s2.mbtiles not built on this machine")
def test_the_real_launch_pad_pack():
    meta = mw.read_mbtiles_metadata(REAL_PACK)
    assert meta["format"] == "jpg"
    assert meta["attribution"] == EOX
    assert (meta["minzoom"], meta["maxzoom"]) == ("8", "14")
    for z in (8, 12, 14):
        x, y = pad_tile(z)
        data = mw.read_mbtiles_tile(REAL_PACK, z, x, y)
        assert data and data[:3] == b"\xff\xd8\xff", "pad tile missing at z%d" % z


# ---------------------------------------------------------------------------
# Overzoom
# ---------------------------------------------------------------------------

QUADS = ("#ff0000", "#00ff00", "#0000ff", "#ffff00")
QUAD_RGB = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))


def provider_for(path, max_zoom):
    provider = mw.TileProvider()
    provider.set_source(mw.MapSource(name="pack", type="mbtiles", path=path,
                                     min_zoom=0, max_zoom=max_zoom))
    return provider


def test_overzoom_picks_the_parent_and_the_right_quarter(qapp, tmp_path):
    path = make_pack(tmp_path / "q.mbtiles",
                     {(10, 300, 400): jpeg_tile("#000000", QUADS)}, zmax=10)
    provider = provider_for(path, 10)
    assert provider.get(11, 601, 800) is None          # past max: never asked
    image = None
    assert wait_for(qapp, lambda: provider.overzoomed(11, 601, 800) is not None)
    image = provider.overzoomed(11, 601, 800)          # x odd, y even: TR
    assert (image.width(), image.height()) == (256, 256)
    assert close_enough(pixel(image, 128, 128), QUAD_RGB[1])
    for (cx, cy), rgb in zip(((600, 800), (601, 800), (600, 801), (601, 801)),
                             QUAD_RGB):
        assert close_enough(pixel(provider.overzoomed(11, cx, cy), 128, 128), rgb)


def test_overzoom_two_and_three_levels_and_no_further(qapp, tmp_path):
    path = make_pack(tmp_path / "q.mbtiles",
                     {(10, 300, 400): jpeg_tile("#000000", QUADS)}, zmax=10)
    provider = provider_for(path, 10)
    # z12: the parent's quarter is 64 px; tile (1203, 1603) sits in its BR.
    assert wait_for(qapp, lambda: provider.overzoomed(12, 1203, 1603) is not None)
    assert close_enough(pixel(provider.overzoomed(12, 1203, 1603), 128, 128),
                        QUAD_RGB[3])
    # z13 = 3 levels past max: still drawn, from 32 px of the parent.
    assert close_enough(pixel(provider.overzoomed(13, 2400, 3200), 128, 128),
                        QUAD_RGB[0])
    # z14 = 4 levels: beyond the limit, grey instead.
    assert provider.overzoomed(14, 4800, 6400) is None
    # Memory cache only, and bounded.
    assert len(provider._scaled) <= mw.OVERZOOM_CACHE_MAX
    assert not os.path.exists(os.path.join(str(tmp_path), "logs"))


def test_a_pack_with_a_missing_tile_draws_overzoom_or_grey(qapp, tmp_path):
    from PyQt5.QtGui import QImage
    z, (x, y) = 13, pad_tile(13)
    # The z13 tile at the pad is missing; its z12 parent is present.
    path = make_pack(tmp_path / "maps" / "pad.mbtiles",
                     {(12, x >> 1, y >> 1): jpeg_tile("#00a000")})
    os.makedirs(os.path.join(str(tmp_path), "assets"), exist_ok=True)
    os.replace(os.path.join(str(tmp_path), "maps"),
               os.path.join(str(tmp_path), "assets", "maps"))
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    canvas = settled(qapp, window)
    canvas.zoom = 13
    canvas.center_lat, canvas.center_lon = PAD_LAT, PAD_LON
    for _ in range(4):
        grab(canvas)
        wait_for(qapp, lambda: False, frames=20)
    image = grab(canvas)
    assert canvas.paint_errors == 0
    assert canvas.provider.overzoom_drawn > 0
    # The pixel under the pad is the parent's green, not the grey fallback.
    p = canvas._lonlat_to_pixel(PAD_LON, PAD_LAT)
    assert close_enough(pixel(image, int(p.x()) + 30, int(p.y()) + 30),
                        (0, 160, 0), tol=60)
    # Far from anything in the pack: grey, and still no paint error.
    canvas.center_lat, canvas.center_lon = 10.0, 10.0
    grab(canvas)
    assert canvas.paint_errors == 0
    window.shutdown()
    window.close()


def test_status_says_overzoomed(qapp, tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "assets", "maps"))
    make_pack(os.path.join(str(tmp_path), "assets", "maps", "pad.mbtiles"), {})
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.canvas.zoom = 14
    window._update_status()
    assert "overzoomed" not in window.status.text()
    window.canvas.zoom = 16
    window._update_status()
    assert "overzoomed x4 from z14" in window.status.text()
    window.canvas.zoom = 18
    window._update_status()
    assert "past the 3-level limit" in window.status.text()
    window.shutdown()
    window.close()


# ---------------------------------------------------------------------------
# Default pack, attribution
# ---------------------------------------------------------------------------

def pack_dir(tmp_path):
    maps = os.path.join(str(tmp_path), "assets", "maps")
    os.makedirs(maps, exist_ok=True)
    return maps


def test_the_tile_pack_is_the_default_over_grid_and_vector(qapp, tmp_path):
    maps = pack_dir(tmp_path)
    with open(os.path.join(maps, "aaa.osmpack.json"), "w") as handle:
        json.dump({"name": "v", "bounds": [83, 26, 85, 28], "features": []},
                  handle)
    make_pack(os.path.join(maps, "launchpad_s2.mbtiles"), {})
    # A previous version auto-saved the vector pack without it being chosen.
    with open(os.path.join(str(tmp_path), mw.SETTINGS_FILE), "w") as handle:
        json.dump({"map_pack": "aaa (vector)"}, handle)
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    assert window.canvas.pack.kind == "mbtiles"
    assert window.canvas.pack.name == "launchpad_s2.mbtiles (tiles)"
    window.shutdown()
    window.close()


def test_an_operator_choice_still_wins(qapp, tmp_path):
    maps = pack_dir(tmp_path)
    make_pack(os.path.join(maps, "launchpad_s2.mbtiles"), {})
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    window.pack_combo.setCurrentIndex(0)                    # the grid
    window.shutdown()
    window.close()
    again = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    assert again.canvas.pack.kind == "grid"
    again.shutdown()
    again.close()


def test_attribution_is_the_exact_eox_text_and_fits(qapp, tmp_path):
    from PyQt5.QtGui import QImage
    maps = pack_dir(tmp_path)
    make_pack(os.path.join(maps, "launchpad_s2.mbtiles"), {})
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    canvas = settled(qapp, window)
    assert canvas.attribution_text() == EOX
    grab(canvas)
    from PyQt5.QtGui import QFont, QFontMetrics
    font = QFont()
    font.setPointSize(8)
    assert QFontMetrics(font).horizontalAdvance(EOX) + 16 < canvas.width()
    window.shutdown()
    window.close()
    # The online source carries the same text.
    source = next(s for s in mw.load_sources(str(tmp_path))[0]
                  if s.name == mw.S2CLOUDLESS_NAME)
    assert source.attribution == EOX and source.max_zoom == 14
    assert source.bulk_download and source.rate_limit == 1.0
    assert source.max_tiles == 4000


def test_example_sources_file_lists_the_eox_source():
    with open(os.path.join(ROOT, "map_sources.example.json"), encoding="utf-8") as h:
        data = json.load(h)
    entry = next(s for s in data["sources"] if s["name"] == mw.S2CLOUDLESS_NAME)
    assert entry["url"] == mw.S2CLOUDLESS_URL
    assert entry["attribution"] == EOX
    assert (entry["type"], entry["max_zoom"], entry["bulk_download"]) == (
        "xyz", 14, True)
    # Only this source may be bulk downloaded.
    assert [s["name"] for s in data["sources"] if s.get("bulk_download")] == [
        mw.S2CLOUDLESS_NAME]


# ---------------------------------------------------------------------------
# Launch pad
# ---------------------------------------------------------------------------

def test_launch_pad_defaults_are_written_and_drawn(qapp, tmp_path):
    from PyQt5.QtGui import QImage
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    assert window.canvas.launch_pad == ("Launch pad", PAD_LAT, PAD_LON)
    window._save_settings()
    saved = json.loads((tmp_path / mw.SETTINGS_FILE).read_text())
    assert saved["launch_pad"] == {"name": "Launch pad", "lat": PAD_LAT,
                                   "lon": PAD_LON}
    assert saved[mw.TILE_CONTACT_FIELD] == ""
    canvas = settled(qapp, window)
    image = grab(canvas)
    assert canvas.pad_drawn and canvas.paint_errors == 0
    # The map opened over the pad, so the marker is at the centre.
    p = canvas._lonlat_to_pixel(PAD_LON, PAD_LAT)
    cx, cy = canvas.width() // 2, canvas.height() // 2
    assert abs(p.x() - cx) <= 1 and abs(p.y() - cy) <= 1
    assert close_enough(pixel(image, cx, cy), (0xe8, 0x59, 0x0c))
    window.shutdown()
    window.close()


def test_launch_pad_settings_load_and_set_the_range_origin(qapp, tmp_path):
    (tmp_path / mw.SETTINGS_FILE).write_text(json.dumps({
        "launch_pad": {"name": "Pad B", "lat": 26.5, "lon": 84.1},
        "ground_station": {"lat": 26.7, "lon": 84.3, "show": False},
        "attitude_smooth": False}))
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    assert window.canvas.launch_pad == ("Pad B", 26.5, 84.1)
    # No ground station shown: rings, distance and bearing use the pad.
    assert window.canvas.ground_station is None
    assert window.canvas.range_origin == (26.5, 84.1)
    window.gs_check.setChecked(True)
    assert window.canvas.range_origin == pytest.approx((26.7, 84.3))
    saved = json.loads((tmp_path / mw.SETTINGS_FILE).read_text())
    # Keys the map does not own survive its save.
    assert saved["attitude_smooth"] is False
    assert saved["launch_pad"]["name"] == "Pad B"
    window.shutdown()
    window.close()


@pytest.mark.parametrize("bad", [None, "x", {"lat": "n"}, {"lat": 99, "lon": 1}])
def test_a_bad_launch_pad_entry_falls_back_to_the_default(qapp, tmp_path, bad):
    (tmp_path / mw.SETTINGS_FILE).write_text(json.dumps({"launch_pad": bad}))
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    assert window.canvas.launch_pad == ("Launch pad", PAD_LAT, PAD_LON)
    window.shutdown()
    window.close()


def test_vehicle_label_measures_from_the_pad(qapp, tmp_path):
    from PyQt5.QtGui import QImage
    store = TrackStore()

    class Fix:
        lat, lon, nav_alt_m = PAD_LAT + 0.009, PAD_LON, 500.0
        has_fix = has_nav_alt = True
        packet_count, payload_type = 1, "CANSAT"
        gs_recv_epoch, mission_time_s, sats, fsm_name = 1.7e9, 1.0, 9, "ASCENT"

    store.add_packet(Fix()) if hasattr(store, "add_packet") else store.append(Fix())
    window = mw.MapWindow(store, base_dir=str(tmp_path))
    from track_store import haversine_m
    assert window.canvas.range_origin == (PAD_LAT, PAD_LON)
    assert 990 < haversine_m(PAD_LAT, PAD_LON, Fix.lat, Fix.lon) < 1010
    grab(settled(qapp, window))
    assert window.canvas.paint_errors == 0
    window.shutdown()
    window.close()


# ---------------------------------------------------------------------------
# fetch_tiles
# ---------------------------------------------------------------------------

BOX = ["--bbox", "83.8012", "26.2704", "84.8080", "27.1697"]


def test_fetch_tiles_counts_the_launch_pad_pack():
    import fetch_tiles
    per_zoom = []
    for z in range(8, 15):
        x0, y0, x1, y1 = fetch_tiles.tile_range(83.8012, 26.2704, 84.8080,
                                                27.1697, z)
        per_zoom.append((x1 - x0 + 1) * (y1 - y0 + 1))
    assert per_zoom == [4, 6, 16, 49, 156, 576, 2209]
    assert sum(per_zoom) == 3016


def test_fetch_tiles_refuses_jobs_over_the_cap_even_with_force(capsys, tmp_path):
    import fetch_tiles
    out = str(tmp_path / "x.mbtiles")
    code = fetch_tiles.main(BOX + ["--zoom", "8", "15", "--source",
                                   mw.S2CLOUDLESS_NAME, "--out", out, "--force"])
    assert code == 3
    assert "hard cap of 4000" in capsys.readouterr().err
    assert not os.path.exists(out)
    # Within the cap it is allowed (dry run: nothing downloaded).
    assert fetch_tiles.main(BOX + ["--zoom", "8", "14", "--source",
                                   mw.S2CLOUDLESS_NAME, "--out", out,
                                   "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "tiles      : 3016" in text and "1.00 req/s" in text


def test_fetch_tiles_rate_never_exceeds_the_source_limit(capsys, tmp_path):
    import fetch_tiles
    fetch_tiles.main(BOX + ["--zoom", "8", "9", "--source", mw.S2CLOUDLESS_NAME,
                            "--out", str(tmp_path / "x.mbtiles"), "--rate", "8",
                            "--dry-run"])
    assert "1.00 req/s" in capsys.readouterr().out


def _fake_source():
    return mw.MapSource(name="t", type="xyz", url="https://e/{z}/{y}/{x}.jpg",
                        bulk_download=True)


def test_download_backs_off_on_429_and_resumes(tmp_path):
    import fetch_tiles
    path = str(tmp_path / "b.mbtiles")
    conn = fetch_tiles.open_mbtiles(path, "t", "d", (84.3, 26.7, 84.31, 26.71),
                                    10, 10, fmt="jpg")
    tile = jpeg_tile("#336699")
    answers = [(None, 429, 7.0), (None, 429, 0.0), (tile, 200, 0.0)]
    calls, sleeps = [], []
    clock = [0.0]

    def fetcher(url, timeout, ua):
        calls.append(url)
        return answers.pop(0) if answers else (tile, 200, 0.0)

    def sleep(s):
        sleeps.append(round(s, 3))
        clock[0] += s

    stored, skipped, failed, stopped = fetch_tiles.download(
        conn, _fake_source(), (84.3, 26.7, 84.31, 26.71), 10, 10, 1.0, "UA",
        1, fetcher=fetcher, sleep=sleep, clock=lambda: clock[0])
    assert (stored, failed, stopped) == (1, 0, "")
    assert len(calls) == 3 and len(set(calls)) == 1      # same tile retried
    assert 7.0 in sleeps                                 # honoured Retry-After
    assert fetch_tiles.BACKOFF_START_S * 2 in sleeps     # then doubled backoff
    # A second run skips what is stored.
    again = fetch_tiles.download(conn, _fake_source(), (84.3, 26.7, 84.31, 26.71),
                                 10, 10, 1.0, "UA", 1, fetcher=fetcher,
                                 sleep=sleep, clock=lambda: clock[0])
    assert again[:2] == (0, 1)
    conn.close()


def test_download_stops_resumably_after_repeated_429(tmp_path):
    import fetch_tiles
    conn = fetch_tiles.open_mbtiles(str(tmp_path / "c.mbtiles"), "t", "d",
                                    (84.3, 26.7, 84.4, 26.8), 12, 12)
    clock = [0.0]
    stored, skipped, failed, stopped = fetch_tiles.download(
        conn, _fake_source(), (84.3, 26.7, 84.4, 26.8), 12, 12, 1.0, "UA", 4,
        fetcher=lambda u, t, a: (None, 429, 0.0),
        sleep=lambda s: clock.__setitem__(0, clock[0] + s),
        clock=lambda: clock[0])
    assert stored == 0 and "HTTP 429" in stopped
    conn.close()


def test_download_is_one_request_per_second(tmp_path):
    import fetch_tiles
    conn = fetch_tiles.open_mbtiles(str(tmp_path / "r.mbtiles"), "t", "d",
                                    (84.3, 26.7, 84.4, 26.8), 12, 12)
    clock = [100.0]
    starts = []

    def fetcher(url, timeout, ua):
        starts.append(clock[0])
        clock[0] += 0.2                                 # a fast server
        return jpeg_tile("#123456"), 200, 0.0

    fetch_tiles.download(conn, _fake_source(), (84.3, 26.7, 84.4, 26.8), 12, 12,
                         1.0, "UA", 4, fetcher=fetcher,
                         sleep=lambda s: clock.__setitem__(0, clock[0] + s),
                         clock=lambda: clock[0])
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert gaps and min(gaps) >= 1.0 - 1e-9
    conn.close()


def test_tile_contact_goes_into_the_user_agent(tmp_path, monkeypatch):
    import fetch_tiles
    (tmp_path / mw.SETTINGS_FILE).write_text(
        json.dumps({mw.TILE_CONTACT_FIELD: "team@uni.ac.in"}))
    assert fetch_tiles.read_tile_contact(str(tmp_path)) == "team@uni.ac.in"
    assert "team@uni.ac.in" in mw.default_user_agent("team@uni.ac.in")
    assert fetch_tiles.read_tile_contact(str(tmp_path / "nowhere")) == ""


# ---------------------------------------------------------------------------
# No network
# ---------------------------------------------------------------------------

class NetworkBlocked(AssertionError):
    pass


@pytest.fixture
def no_network(monkeypatch):
    """Every way Python opens a connection raises, and is recorded."""
    attempts = []

    def blocked(name):
        def guard(*args, **kwargs):
            attempts.append((name, args[:2]))
            raise NetworkBlocked("network used: %s %r" % (name, args[:2]))
        return guard

    monkeypatch.setattr(socket.socket, "connect", blocked("connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", blocked("connect_ex"))
    monkeypatch.setattr(socket, "create_connection", blocked("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", blocked("getaddrinfo"))
    yield attempts


def test_offline_start_with_the_pack_makes_no_network_request(qapp, tmp_path,
                                                              no_network):
    from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QImage, QMouseEvent
    maps = pack_dir(tmp_path)
    tiles = {}
    for z in (12, 13):
        x, y = pad_tile(z)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                tiles[(z, x + dx, y + dy)] = jpeg_tile("#2e6b30")
    make_pack(os.path.join(maps, "launchpad_s2.mbtiles"), tiles)
    window = mw.MapWindow(TrackStore(), base_dir=str(tmp_path))
    canvas = settled(qapp, window)
    canvas.zoom = 13
    for _ in range(5):
        grab(canvas)
        wait_for(qapp, lambda: False, frames=20)
    image = grab(canvas)
    # Imagery around the pad, the marker, the attribution.
    assert window.canvas.pack.kind == "mbtiles"
    assert canvas.provider.tiles_drawn > 0
    assert canvas.pad_drawn
    assert canvas.attribution_text() == EOX
    p = canvas._lonlat_to_pixel(PAD_LON, PAD_LAT)
    assert close_enough(pixel(image, int(p.x()) + 60, int(p.y()) + 60),
                        (0x2e, 0x6b, 0x30), tol=30)
    # Scale bar: its dark line at the bottom left.
    bar = pixel(image, 14, canvas.height() - 18)
    assert bar.lightness() < 80
    # Cursor readout.
    move = QMouseEvent(QEvent.MouseMove, QPointF(100, 100), Qt.NoButton,
                       Qt.NoButton, Qt.NoModifier)
    qapp.sendEvent(canvas, move)
    window._update_status()
    assert "cursor 26." in window.status.text()
    assert canvas.paint_errors == 0
    assert no_network == [], no_network
    window.shutdown()
    window.close()


def test_the_dashboard_starts_with_no_network(qapp, tmp_path, no_network,
                                              monkeypatch):
    monkeypatch.chdir(tmp_path)
    maps = pack_dir(tmp_path)
    make_pack(os.path.join(maps, "launchpad_s2.mbtiles"),
              {(12,) + pad_tile(12): jpeg_tile("#2e6b30")})
    from dashboard_ui import Dashboard
    window = Dashboard(log_dir=str(tmp_path / "logs"))
    window.map_window.show()
    qapp.processEvents()
    assert window.map_window.canvas.pack.name == "launchpad_s2.mbtiles (tiles)"
    assert no_network == [], no_network
    window.map_window.shutdown()
    window.close()

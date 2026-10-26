# Ground Control Station — CanSat / Rocketry Telemetry

A multi-threaded PyQt5 dashboard for receiving, displaying and logging avionics
telemetry from an XBee 3 PRO link.

## Quick start

```powershell
# 1. Create a virtual environment (PyQt5 needs Python 3.10-3.12; 3.13+ has no wheels)
py -3.11 -m venv .venv

# 2. Install dependencies
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Run
.\.venv\Scripts\python.exe main.py
```

On Linux/macOS: `python3.11 -m venv .venv && source .venv/bin/activate &&
pip install -r requirements.txt && python main.py`.

Optional flags:

```
python main.py --port COM7 --baud 9600 --autoconnect
python main.py --log-dir D:\flight_logs
```

## Testing without a radio

`packet_sim.py` flies a scripted mission and emits real checksum-correct frames.
It defaults to a **TCP server**, and the GCS reaches it through pyserial's
`socket://` URL handler — no virtual COM port driver required (see the module
docstring for why this beats com0com on Windows).

```powershell
# terminal 1
.\.venv\Scripts\python.exe packet_sim.py --rate 20

# terminal 2
.\.venv\Scripts\python.exe main.py
#   → pick "socket://127.0.0.1:5555  (packet_sim.py)" in the Port dropdown
#   → CONNECT, then START LOGGING
```

Fault injection (for the robustness test requirement):

```powershell
.\.venv\Scripts\python.exe packet_sim.py --rate 20 --chaos
```

`--chaos` = 6 % bad checksums, 3 % junk bytes, 3 % truncated frames, 5 % dropped
frames, and a 40-packet burst after 3 s of silence every 10 s. Individual rates
are also settable: `--corrupt-rate`, `--garbage-rate`, `--truncate-rate`,
`--drop-rate`, `--burst`. Run `packet_sim.py --help` for everything.

If you do have com0com installed (or socat on Linux), use a real port pair
instead: `packet_sim.py --serial COM11` and connect the GCS to `COM12`.

## Packet format

```
$TEAM_ID,TIMESTAMP,PACKET_COUNT,ALTITUDE,PRESSURE,TEMP,VOLTAGE,NAV_TIME,
 LAT,LON,NAV_ALT,SATS,ACC_X,ACC_Y,ACC_Z,GYRO_X,GYRO_Y,GYRO_Z,FSM_STATE*CS
```

19 comma-separated fields; `CS` is the 2-hex-digit XOR of every byte between
`$` and `*`. FSM states: 0 BOOT, 1 TEST_MODE, 2 LAUNCH_PAD, 3 ASCENT,
4 DEPLOY, 5 DESCENT, 6 AEROBRAKE_RELEASE, 7 IMPACT.

## Output files

| File | Contents |
|------|----------|
| `logs/Flight_<TEAM_ID>.csv` | Every valid packet, with GS receive timestamp and checksum-valid flag. Appended, never truncated. |
| `logs/errors.log` | Every rejected frame (raw bytes + reason) plus session notes. Always written, even with CSV logging off. |

## Live map

**MAP** (next to CSV TABLE) opens a separate window showing every GNSS fix of
the session, per vehicle. Fixes are recorded whether or not the window is
open, so opening it mid-flight shows the whole track.

* CanSat is a cyan circle, rocket a red triangle - different shape as well as
  colour, so they stay distinguishable in a screenshot or to a colour-blind
  operator. Each label carries altitude and the age of its last fix, turning
  red after 5 s.
* Wheel zooms around the cursor, drag pans, double-click centres. **FOLLOW**
  keeps a vehicle centred (dragging cancels it); **FIT ALL** frames
  everything; **CLEAR TRACKS** clears only the drawing - the CSV and the KML
  export are untouched.
* Tracks reset when START LOGGING begins a new session file, and on
  RESET SENSOR DATA.

**The map works with no network and no files at all.** Nothing is downloaded
at startup, and the pack selected on open is always an offline one.

### Map packs

The **Map pack** dropdown lists everything found in `assets/maps/`, plus the
grid. Your choice is remembered in `map_settings.json`. A pack that will not
load falls back to the grid with a line in the event log - never an error
dialog, never an exception.

| Pack kind | What it is | Needs |
|---|---|---|
| **Grid only** | Lat/lon graticule labelled in degrees and minutes, scale bar, north arrow, and range rings at 250 m / 500 m / 1 km / 2 km around the ground station. Always available. | nothing |
| **Vector** (`*.osmpack.json`) | OpenStreetMap roads, buildings, water, rail, landuse and place names, rendered by this application. | `tools/fetch_osm_vectors.py`, once, at home |
| **Tiles** (`*.mbtiles`) | A raster tile pack. | `tools/fetch_tiles.py`, QGIS or MOBAC |
| **Image** (`NAME.png` + `NAME.json`) | Your own satellite or scanned-map image, georeferenced. | you supply it |
| *online sources* | Listed last and labelled `(online)`. The only entries that touch the network. | a connection |

A bundled **India overview** (Natural Earth, public domain, 196 KB) ships in
`assets/maps/`, so a fresh clone shows the country outline, state boundaries
and major cities when zoomed out.

### Vector pack for your launch site (recommended)

One Overpass query, no tiles, a few hundred kilobytes:

```bash
python tools/fetch_osm_vectors.py \
    --bbox 74.8305 32.7041 74.8835 32.7491 \
    --out assets/maps/jammu.osmpack.json
```

`--dry-run` prints the query and the area without sending anything. Areas
over 10 km a side are refused: Overpass is donated infrastructure for small
extracts, not regional downloads - for something larger, take a Geofabrik
extract and convert it to the same JSON format, which the map also accepts.
The tool makes exactly **one** request per run and identifies itself; if the
server rate-limits you, wait rather than retrying in a loop.

Data is © OpenStreetMap contributors, ODbL. The map shows that attribution
whenever such a pack is drawn; keep it if you redistribute a pack.

### Raster tile pack

```bash
# 5 x 5 km around the launch site, zoom 12-18 -> 2,218 tiles
python tools/fetch_tiles.py \
    --bbox 74.8305 32.7041 74.8835 32.7491 \
    --zoom 12 18 \
    --source "ISRO Bhuvan (admin basemap)" \
    --out assets/maps/jammu.mbtiles
```

Rate limited (2 req/s), resumable, refuses jobs over 20,000 tiles without
`--force`, and **refuses any source not explicitly marked
`"bulk_download": true`** in `map_sources.json`, naming the reason. The OSMF
Tile Usage Policy forbids bulk downloading from `tile.openstreetmap.org`, so
that source is refused by default. Read a provider's terms yourself before
setting that flag.

**Or build the pack in a GUI**, which is often easier and is the only route
for providers this tool refuses:

* **QGIS** - add the layer (Browser -> XYZ Tiles / WMS), zoom to the site,
  then *Processing -> Toolbox -> Raster tools -> Generate XYZ tiles
  (MBTiles)*. Set the extent, zoom 12-18, PNG, output into `assets/maps/`.
* **Mobile Atlas Creator (MOBAC)** - pick the source, select the area, choose
  **MBTiles SQLite** as the atlas format, add zoom 12-18, create into
  `assets/maps/`.

Zoom 12 gives a few kilometres of context; 18 is roughly 0.6 m per pixel,
enough to pick out tracks and buildings for recovery.

### Your own image overlay

Put an image in `assets/maps/` with a sidecar of the same name:

```
assets/maps/launchsite.png
assets/maps/launchsite.json
```
```json
{
  "name": "Launch site, 2026 survey",
  "west": 74.8305, "south": 32.7041,
  "east": 74.8835, "north": 32.7491,
  "attribution": "My own survey photo"
}
```

Produce the image by exporting a view from an imagery source **you are
allowed to use**, or from QGIS (*Project -> Import/Export -> Export Map to
Image*, with "Draw annotations" off; QGIS reports the extent you then copy
into the sidecar). **The image's licence is your responsibility** - the map
says so on screen, and nothing is downloaded on your behalf.

The image is treated as Web Mercator aligned and stretched to those bounds.
Over an area under 10 km that is a good approximation: the Mercator y-scale
changes by about 0.08% over 10 km of latitude at 33°N, under 8 m across the
whole image. Use the opacity slider to fade it so the tracks stay readable.
Very large images are downsampled on a worker thread, so the GUI never
stalls while one loads.

### Offline checklist - do this at home, before you travel

1. **Vector pack for the site, with a 2 km margin.** If the pad is at
   32.7266 N, 74.8570 E, a 2 km margin is about ±0.022° of latitude and
   ±0.027° of longitude:
   ```bash
   python tools/fetch_osm_vectors.py \
       --bbox 74.8300 32.7046 74.8840 32.7486 \
       --out assets/maps/launchsite.osmpack.json
   ```
2. **A tile pack**, if you have a source whose terms allow it (see above).
3. **Your image overlay**, if you have one, plus its sidecar.
4. **Verify the packs load**, without starting the whole application:
   ```bash
   python -c "import map_basemap, os; \
   [print(f, '->', (lambda p: p.name + ', %d features' % p.feature_count if p else 'FAILED TO LOAD')(map_basemap.load_vector_pack(os.path.join('assets/maps', f)))) \
    for f in os.listdir('assets/maps') if f.endswith('.osmpack.json')]"
   ```
   Every pack must print a feature count. `FAILED TO LOAD` means a corrupt
   or truncated download - fetch it again while you still have a network.
5. **The night before: disconnect the network and start the application.**
   Turn off Wi-Fi and unplug the cable, then:
   ```bash
   python main.py
   ```
   Open **MAP**. You should see your pack drawn, the scale bar, the north
   arrow and - once you enter the ground station position - the range rings.
   The status line must *not* say `OFFLINE`, because nothing should be
   trying to reach a server. If it does, you have an online source selected:
   pick an offline pack from the dropdown, and it will be remembered.
6. **Set the ground station position** in the map window (GS lat / lon,
   then tick *show*) so range and bearing to each vehicle are live. It is
   saved to `map_settings.json`, so do it once.

**Privacy note.** Any online source receives the tile coordinates you are
looking at - the approximate location and zoom of your launch site - along
with your IP address and the application's User-Agent. The offline packs
above send nothing at all, which is the other reason to prepare them.

## KML export for Google Earth

**EXPORT KML** (next to START LOGGING) writes the session's full tracks -
not the thinned drawing data - through a save dialog, defaulting to the same
`logs/` folder and the same `Flight_<team>_<YYYY-MM-DD_HHMMSS>` naming as the
CSV. It works while logging is running. With no fixes recorded it says so and
writes nothing, rather than leaving an empty file that looks like a failed
flight.

Each vehicle gets a Folder containing a `gx:Track` (so Google Earth's time
slider replays the flight, with fsm_state, sats and packet_count available at
the cursor's moment), an extruded `LineString` of the same path, and
Launch / Apogee / Last-known placemarks with a telemetry table each.

Altitude is `nav_alt_m` - GNSS height above mean sea level - with
`altitudeMode absolute`. The barometric altitude is deliberately *not* used:
it is referenced to a fixed 1013.25 hPa sea level with no ground zero-set, so
it carries a constant offset that would put the whole track at the wrong
height. A link that transmits no GNSS altitude produces a `clampToGround`
path, and the document description says so.

Before writing, fixes with no lock and exact 0,0 are dropped, duplicate packet
counts are dropped, and single-sample GPS jumps (default: more than 500 m in
under 1 s, which is 1800 km/h) are dropped. All three counts go into the
document description, so a shortened path is never silent.

### After the flight

```bash
python tools/csv_to_kml.py logs/Flight_2025ASI001_*.csv
```

Converts a CSV the ground station already wrote, offline, through the same
builder and the same filters as the live export - so a post-flight KML is
identical to one made live. `--jump-m` and `--jump-window-s` tune the glitch
filter; `--out-dir` chooses where to write.

## Threading model

| Thread | Owns | Talks to others via |
|--------|------|---------------------|
| `SerialWorker` (QThread) | pyserial handle, RX ring buffer, frame sync, XOR check, auto-reconnect | Qt signals (auto-queued) |
| GUI (main) | all widgets, pyqtgraph plots, QTimer repaint | — |
| `CsvLoggerThread` (QThread) | CSV + errors file handles, periodic flush | `queue.Queue` (non-blocking put) |
| `VoiceCallout` (QThread) | the OS speech engine | one altitude slot + a short state queue, never blocking |
| Map tile workers (`QThreadPool`) | one sqlite connection per MBTiles read | Qt signals; network tiles are asynchronous on the GUI thread via `QNetworkAccessManager` |

Key invariant: **packet arrival rate and repaint rate are decoupled.** The
`packet_received` slot only appends numbers to lists; a 15 Hz `QTimer` does all
the drawing. A burst of 500 packets therefore costs the same number of repaints
as a single packet, which is why the UI never freezes.

The map obeys the same invariant: packet intake is one append, and the canvas
repaints on its own timer at 5 Hz regardless of the packet rate.

Files: `main.py` (entry point), `serial_worker.py` (ingestion thread),
`telemetry_packet.py` (dataclass + parser + checksum), `csv_logger.py` (logging
thread), `dashboard_ui.py` (window and widgets), `packet_sim.py` (test source),
`track_store.py` (the session's fixes, shared by the map and the KML),
`map_window.py` (slippy map), `kml_export.py` (KML builder),
`tools/fetch_tiles.py` and `tools/csv_to_kml.py`.

## Tests

```bash
pip install pytest
python -m pytest tools/ -v
```

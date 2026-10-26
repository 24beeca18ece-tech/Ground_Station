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
* A tile that cannot be had is drawn as a grey square labelled with its zoom.
  That is deliberate: a hole in the map must read as a hole, not as the
  vehicle having flown over blank ground.

### Choosing a source

The dropdown lists the entries in `map_sources.json`. That file is
**gitignored** (it may name an API key's environment variable and local
paths); copy the tracked `map_sources.example.json` to start. A missing or
unreadable file is not an error - the map falls back to a built-in list and
writes one line to the event log.

An offline pack is selected automatically whenever one exists, because the
launch site has no internet. Shipped entries:

| Source | Notes |
|---|---|
| **Offline (MBTiles)** | Any `.mbtiles` file in `assets/maps/`. The default when present. Works with no network at all. |
| **ISRO Bhuvan (admin basemap)** | WMS at `bhuvan-vec1.nrsc.gov.in`. Verified working anonymously - no token, no registration. It is a *vector* server: administrative boundaries, **not** satellite imagery. |
| **OpenStreetMap** | Light interactive use only. Bulk download is forbidden by the OSMF Tile Usage Policy, and `tools/fetch_tiles.py` refuses it. |

A commented Mappls / MapmyIndia entry sits in `_disabled_examples` in the
example file. It needs an API key and **their terms have not been checked for
this project** - read them before enabling it. Keys are read from the
environment variable named by `key_env`; no key is ever stored in the repo.

No Google Maps or Google tile server is used anywhere in the map. Google
Earth is only the viewer for the exported KML.

**Privacy note.** Any online source receives the tile coordinates you are
looking at - which is to say the approximate location and zoom of your launch
site - along with your IP address and the application's User-Agent. If that
matters for a given site, build an offline pack beforehand and fly with the
offline source selected; then nothing leaves the machine.

### Building an offline tile pack

```bash
# 5 x 5 km around the launch site, zoom 12-18 -> 2,218 tiles
python tools/fetch_tiles.py \
    --bbox 74.8305 32.7041 74.8835 32.7491 \
    --zoom 12 18 \
    --source "ISRO Bhuvan (admin basemap)" \
    --out assets/maps/jammu.mbtiles
```

`--dry-run` counts the tiles and checks permissions without downloading. The
tool is rate limited (2 req/s), resumable (re-run the same command to continue
an interrupted job), identifies itself by User-Agent, and refuses jobs over
20,000 tiles without `--force`.

It also **refuses any source not explicitly marked `"bulk_download": true`**
in `map_sources.json`, and says why. Bulk downloading is a sustained load on
someone else's server and most providers' terms govern it - the OSMF policy
forbids it outright for `tile.openstreetmap.org`. Read the provider's terms
yourself before setting that flag.

**Or build the pack in a GUI instead**, which is often easier and is the only
route for providers this tool refuses:

* **QGIS** - add the layer (Browser -> XYZ Tiles / WMS), zoom to the launch
  site, then *Processing -> Toolbox -> Raster tools -> Generate XYZ tiles
  (MBTiles)*. Set the extent, zoom 12-18, tile format PNG, and write the
  output into `assets/maps/`.
* **Mobile Atlas Creator (MOBAC)** - pick the map source, select the area,
  choose **MBTiles SQLite** as the atlas format, add zoom levels 12-18, and
  create the atlas into `assets/maps/`.

Zoom 12 gives context for a few kilometres around the site; 18 is roughly
0.6 m per pixel, enough to pick out tracks and buildings for recovery.
Below 12 is rarely useful at a single site, and above 18 is usually not
available.

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

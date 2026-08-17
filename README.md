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

## Threading model

| Thread | Owns | Talks to others via |
|--------|------|---------------------|
| `SerialWorker` (QThread) | pyserial handle, RX ring buffer, frame sync, XOR check, auto-reconnect | Qt signals (auto-queued) |
| GUI (main) | all widgets, pyqtgraph plots, QTimer repaint | — |
| `CsvLoggerThread` (QThread) | CSV + errors file handles, periodic flush | `queue.Queue` (non-blocking put) |

Key invariant: **packet arrival rate and repaint rate are decoupled.** The
`packet_received` slot only appends numbers to lists; a 15 Hz `QTimer` does all
the drawing. A burst of 500 packets therefore costs the same number of repaints
as a single packet, which is why the UI never freezes.

Files: `main.py` (entry point), `serial_worker.py` (ingestion thread),
`telemetry_packet.py` (dataclass + parser + checksum), `csv_logger.py` (logging
thread), `dashboard_ui.py` (window and widgets), `packet_sim.py` (test source).

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Set up (Python 3.10–3.12 required; PyQt5 has no wheels for 3.13+)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the GCS
python main.py
python main.py --port COM7 --baud 9600 --autoconnect
python main.py --raw-csv          # bare-CSV bench-test mode (no checksum)
python main.py --log-dir /path/to/logs

# Test without hardware (two terminals)
python packet_sim.py --rate 20            # terminal 1: TCP server on :5555
python main.py                            # terminal 2: pick "socket://127.0.0.1:5555" in Port dropdown

# Fault injection
python packet_sim.py --rate 20 --chaos    # 6% bad checksums, 3% junk, etc.

# Regenerate the 3D mesh (only needed if the CAD geometry changes)
pip install gmsh
python tools/step_to_mesh.py 0.012
```

There is no test suite or linter configured; no `pytest`, `ruff`, or `mypy` invocations exist in the project.

## Architecture

**Three-thread model — the central invariant:**
- `SerialWorker` (QThread) owns the pyserial handle, ring buffer, frame sync, and XOR checksum. It communicates to the GUI only via Qt signals.
- GUI thread (main) owns every widget and the pyqtgraph plots. It never touches I/O.
- `CsvLoggerThread` (QThread) owns all file handles. The GUI hands packets to it through a `queue.Queue` — never through a Qt signal.

**Render decoupling:** `Dashboard.on_packet` only appends numbers to lists; a 15 Hz `QTimer` (`RENDER_HZ`) does all repainting. A 500-packet burst after a dropout costs the same number of repaints as a single packet.

**Receive pipeline** (all inside `serial_worker.py`):
```
raw bytes → ApiFrameUnwrapper (strips XBee 802.15.4 API framing / passes plain ASCII through)
          → FrameSplitter (finds $...*XX boundaries)
          → parse_frame() (XOR checksum + field parsing)
          → implausible_reasons() (physical bounds gate)
          → packet_received signal → GUI + CSV logger
```
Four separate failure counters: `api_frame_errors` (radio framing), `corrupt_packets` (telemetry XOR), `resyncs` (ASCII frame sync), `rejected_packets` (physical bounds). They are independent layers by design.

**Dual-radio merge** (`packet_merger.py`): The flight transmitter alternates destinations (one packet to RX1, next to RX2). `PacketMerger` reorders by `packet_count` with a small depth-first buffer (`MIN_GAP_DEPTH = 3`). Time is the backstop only; depth is the primary release rule because both `add` and `drain` run on the GUI thread.

**Packet class hierarchy** (`telemetry_packet.py`):
- `TelemetryPacket` — all common fields; every consumer that only needs common data treats packets as this type.
- `CanSatPacket(TelemetryPacket)` — adds SPS30 particulate fields (pm1_0/pm2_5/pm4_0/pm10), reaction_wheel_rpm, recovery_stage.
- `RocketPacket(TelemetryPacket)` — adds solenoid_fired, nichrome_fired.
- Auto-detected from field count and PAYLOAD_TYPE token; v1 (19-field, no PAYLOAD_TYPE) is still accepted.

**Wire formats:**
- v1 (legacy, 19 fields): `$TEAM_ID,TIMESTAMP,...,FSM_STATE*CS`
- v2 CanSat (26 fields): PAYLOAD_TYPE inserted after TEAM_ID, 6 extra fields after FSM_STATE
- v2 Rocket (22 fields): same common prefix, 2 extra fields after FSM_STATE
- Raw-CSV (--raw-csv flag): bare 16-field CSV, no `$`/`*XX`, no checksum — bench-test mode only. **Gyro at indices 4-6, accel at 7-9** (verified against Teensy firmware source Sep 2026; swapping them produces plausible-looking wrong data). GPS altitude and satellite count are absent from this link (sent over a separate LoRa radio).

**CSV schema** (`telemetry_packet.py:CSV_HEADER`): `CSV_HEADER`, `to_csv_row()`, and `_variant_cells()` are coupled. If a field is added, all three must be updated together — a mismatch raises `RuntimeError` at import time.

**Altitude callout** (`altitude_callout.py`): Speaks altitude during flight using pyttsx3 (offline OS TTS). Altitude is *not* queued — a stale callout is silently skipped. FSM state transitions *are* queued (short FIFO) so they are never dropped. A fresh pyttsx3 engine is created per utterance to work around an SAPI5 reuse bug.

**Attitude widget** (`attitude_widget.py`): Quaternion integration with a Mahony complementary filter gated on ‖accel‖ ≈ 1 g (`ACCEL_GATE_LO` / `ACCEL_GATE_HI`). Degrades to a QPainter 2D artificial horizon if OpenGL is unavailable. Yaw drifts (no magnetometer); roll and pitch are bounded by the gravity reference.

**Output files** (default: `logs/`):
- `Flight_<TEAM_ID>_<stamp>.csv` — newest-first presentation file, rebuilt periodically with `os.replace` for atomicity.
- `Flight_<TEAM_ID>_<stamp>.csv.journal` — append-only oldest-first durability file; removed on clean close.
- `errors.log` — every rejected frame plus session notes; always written even with CSV logging off.

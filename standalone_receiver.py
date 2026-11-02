#!/usr/bin/env python3
"""
standalone_receiver.py
======================
Minimal, always-on XBee 802.15.4 API-1 receiver for the 2025ASI001 rocket.

  - No Qt, no threads, no GUI.
  - Auto-reconnects if the serial port drops or errors.
  - Built-in watchdog: sends PING automatically if no packet arrives for
    WATCHDOG_S seconds and prints a warning.
  - Logs every valid packet to a timestamped CSV file.
  - Interactive: type  p  + Enter  to send a manual PING.
                 type  e  + Enter  to send EJECT (single-byte 'E').
                 type  a  + Enter  to send TX-START ('A').
                 type  b  + Enter  to send TX-STOP  ('B').
                 type  q  + Enter  to quit cleanly.

Usage:
    python standalone_receiver.py                     # auto-detect port
    python standalone_receiver.py --port COM3         # Windows
    python standalone_receiver.py --port /dev/ttyUSB0 # Linux/Mac
    python standalone_receiver.py --port COM3 --baud 115200
"""

import sys
import os
import time
import struct
import select
import argparse
import datetime
import csv as _csv

# ── optional colour output ───────────────────────────────────────────────────
try:
    import colorama
    colorama.init()
    C_OK    = "\033[92m"   # green
    C_WARN  = "\033[93m"   # yellow
    C_ERR   = "\033[91m"   # red
    C_DIM   = "\033[2m"
    C_CYAN  = "\033[96m"
    C_RESET = "\033[0m"
except ImportError:
    C_OK = C_WARN = C_ERR = C_DIM = C_CYAN = C_RESET = ""

# ── runtime constants ────────────────────────────────────────────────────────
DEFAULT_BAUD   = 115200
WATCHDOG_S     = 10      # seconds of silence before auto-PING
RECONNECT_S    = 2       # seconds to wait before retrying a dead port
READ_TIMEOUT   = 0.05    # serial read timeout (s) — keeps the loop snappy
MAX_FRAME_LEN  = 300     # drop any frame claiming to be longer than this

XBEE_BROADCAST = b"\x00\x00\x00\x00\x00\x00\xff\xff"

# ── CSV header ───────────────────────────────────────────────────────────────
CSV_HEADER = [
    "gs_recv_iso", "radio_frame_id", "team_id", "payload_type",
    "timestamp_s", "packet_count", "altitude_m", "pressure_hpa", "temp_c",
    "voltage_v", "nav_time", "lat", "lon", "nav_alt_m", "sats",
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
    "fsm_state", "solenoid_fired", "raw_body",
]


# ═══════════════════════════════════════════════════════════════════════════
# XBee API-1 helpers
# ═══════════════════════════════════════════════════════════════════════════

def _xbee_checksum(data: bytes) -> int:
    return 0xFF - (sum(data) & 0xFF)


def build_tx_request(payload: bytes, frame_id: int = 1) -> bytes:
    """Build an XBee 0x10 Transmit Request frame (broadcast)."""
    header = bytes([0x10, frame_id]) + XBEE_BROADCAST + b"\xff\xfe\x00\x00"
    body   = header + payload
    length = len(body)
    frame  = bytes([0x7E, length >> 8, length & 0xFF]) + body
    frame += bytes([_xbee_checksum(body)])
    return frame


def build_local_at_cmd(at_cmd: bytes, frame_id: int = 0x41) -> bytes:
    """Build an XBee 0x08 Local AT Command frame.

    Sent to the LOCAL XBee dongle (not over RF).  Used to issue AT+FR
    (firmware restart) when the module enters an unresponsive state.
    No parameter = execute the command immediately.
    """
    body = bytes([0x08, frame_id & 0xFF]) + at_cmd[:2]
    return (bytes([0x7E, 0x00, len(body)]) + body
            + bytes([0xFF - (sum(body) & 0xFF)]))


class XBeeFrameParser:
    """
    State-machine XBee API-1 frame parser.
    Feed it bytes one at a time via feed(); it calls on_frame(frame_bytes)
    when a complete, checksum-correct frame arrives.
    """
    def __init__(self, on_frame):
        self._on_frame = on_frame
        self._reset()
        self.bad_checksums = 0
        self.good_frames   = 0

    def _reset(self):
        self._state  = 0   # 0=wait_start, 1=len_hi, 2=len_lo, 3=data, 4=cksum
        self._length = 0
        self._buf    = bytearray()
        self._csum   = 0

    def feed(self, b: int):
        s = self._state
        if s == 0:
            if b == 0x7E:
                self._state = 1
        elif s == 1:
            self._length = b << 8
            self._state  = 2
        elif s == 2:
            self._length |= b
            if self._length == 0 or self._length > MAX_FRAME_LEN:
                self._reset()
                return
            self._buf   = bytearray()
            self._csum  = 0
            self._state = 3
        elif s == 3:
            self._buf.append(b)
            self._csum = (self._csum + b) & 0xFF
            if len(self._buf) >= self._length:
                self._state = 4
        elif s == 4:
            self._reset()
            if (self._csum + b) & 0xFF != 0xFF:
                self.bad_checksums += 1
                return
            self.good_frames += 1
            self._on_frame(bytes(self._buf))


# ═══════════════════════════════════════════════════════════════════════════
# NMEA-style packet parser
# ═══════════════════════════════════════════════════════════════════════════

def _xor(body: str) -> int:
    v = 0
    for c in body:
        v ^= ord(c)
    return v


def parse_nmea_frame(raw: bytes) -> dict | None:
    """
    Parse a raw XBee RF-data payload that contains a $...*XX frame.
    Returns a dict of fields or None if invalid.
    """
    try:
        text = raw.decode("ascii", errors="replace").strip()
    except Exception:
        return None

    # find the outermost $...*XX
    s = text.find("$")
    star = text.rfind("*")
    if s == -1 or star == -1 or star <= s + 1:
        return None
    body = text[s + 1 : star]
    cs_str = text[star + 1 : star + 3]
    if len(cs_str) < 2:
        return None
    try:
        cs_rx = int(cs_str, 16)
    except ValueError:
        return None
    if _xor(body) != cs_rx:
        return None

    fields = body.split(",")
    if len(fields) < 10:
        return None

    d = {"raw_body": body, "checksum_ok": True}
    idx = 0

    d["team_id"]      = fields[idx]; idx += 1
    # detect payload type token
    if fields[idx].upper() in ("CANSAT", "ROCKET", "RDF"):
        d["payload_type"] = fields[idx].upper(); idx += 1
    else:
        d["payload_type"] = "V1"

    if d["payload_type"] == "RDF":
        # Short LoRa beacon — just store raw
        d["beacon"] = True
        return d

    d["beacon"] = False
    _safe = lambda i, fn=float: _try(fields, i, fn)

    d["timestamp_s"]  = _safe(idx);   idx += 1
    d["packet_count"] = _safe(idx, int); idx += 1
    d["altitude_m"]   = _safe(idx);   idx += 1
    d["pressure_hpa"] = _safe(idx);   idx += 1
    d["temp_c"]       = _safe(idx);   idx += 1
    d["voltage_v"]    = _safe(idx);   idx += 1
    d["nav_time"]     = fields[idx] if idx < len(fields) else ""; idx += 1
    d["lat"]          = _safe(idx);   idx += 1
    d["lon"]          = _safe(idx);   idx += 1
    d["nav_alt_m"]    = _safe(idx);   idx += 1
    d["sats"]         = _safe(idx, int); idx += 1
    d["acc_x"]        = _safe(idx);   idx += 1
    d["acc_y"]        = _safe(idx);   idx += 1
    d["acc_z"]        = _safe(idx);   idx += 1
    d["gyro_x"]       = _safe(idx);   idx += 1
    d["gyro_y"]       = _safe(idx);   idx += 1
    d["gyro_z"]       = _safe(idx);   idx += 1
    d["fsm_state"]    = _safe(idx, int); idx += 1
    d["solenoid_fired"] = _safe(idx, int) if idx < len(fields) else 0

    return d


def _try(fields, i, fn):
    try:
        return fn(fields[i]) if i < len(fields) and fields[i].strip() else None
    except (ValueError, TypeError):
        return None


FSM_NAMES = {
    0: "BOOT", 1: "TEST", 2: "PAD", 3: "ASCENT",
    4: "DEPLOY", 5: "DESCENT", 6: "MAIN", 7: "IMPACT",
}


# ═══════════════════════════════════════════════════════════════════════════
# Port auto-detection
# ═══════════════════════════════════════════════════════════════════════════

def find_xbee_port() -> str | None:
    import serial.tools.list_ports
    candidates = list(serial.tools.list_ports.comports())
    # prefer anything mentioning XBee / FTDI / CH340 / CP210x
    keywords = ["xbee", "ftdi", "ch340", "cp210", "usb serial", "uart"]
    for p in candidates:
        desc = (p.description or "").lower()
        if any(k in desc for k in keywords):
            return p.device
    return candidates[0].device if candidates else None


# ═══════════════════════════════════════════════════════════════════════════
# CSV logger
# ═══════════════════════════════════════════════════════════════════════════

class CsvLog:
    def __init__(self, path: str):
        self._path = path
        self._f    = open(path, "w", newline="")
        self._w    = _csv.DictWriter(self._f, fieldnames=CSV_HEADER, extrasaction="ignore")
        self._w.writeheader()
        self._f.flush()

    def write(self, row: dict):
        self._w.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()


# ═══════════════════════════════════════════════════════════════════════════
# Pretty printer
# ═══════════════════════════════════════════════════════════════════════════

def fmt(v, digits=2, unit=""):
    if v is None:
        return "--"
    return f"{v:.{digits}f}{unit}"


def print_packet(d: dict, frame_id: int):
    now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:12]
    if d.get("beacon"):
        print(f"{C_DIM}[{now}] BEACON  {d['raw_body']}{C_RESET}")
        return

    state = d.get("fsm_state")
    sname = FSM_NAMES.get(state, str(state))
    col   = C_OK if state in (2, 3, 4, 5, 6) else C_WARN if state == 7 else C_DIM

    print(
        f"{col}[{now}] PKT#{d.get('packet_count','?'):>5}  "
        f"FSM={sname:<16}"
        f"ALT={fmt(d.get('altitude_m'),1,'m'):>8}  "
        f"P={fmt(d.get('pressure_hpa'),1,'hPa'):>10}  "
        f"T={fmt(d.get('temp_c'),1,'°C'):>7}  "
        f"V={fmt(d.get('voltage_v'),2,'V'):>6}  "
        f"SATS={d.get('sats','?'):>2}  "
        f"LAT={fmt(d.get('lat'),5):>10}  LON={fmt(d.get('lon'),5):>10}"
        f"{C_RESET}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main receiver loop
# ═══════════════════════════════════════════════════════════════════════════

def run(port: str, baud: int, log_dir: str):
    stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"recv_{stamp}.csv")
    log      = CsvLog(csv_path)
    print(f"{C_CYAN}Logging to: {csv_path}{C_RESET}")

    frame_id_counter = [1]

    def next_fid():
        fid = frame_id_counter[0]
        frame_id_counter[0] = fid + 1 if fid < 255 else 1
        return fid

    pkt_count     = 0
    bad_checksums = 0
    last_pkt_time = time.time()
    silent_pings  = 0    # consecutive watchdog PINGs with no response
    ser           = None

    # ── keyboard input (non-blocking on stdin) ───────────────────────────
    pending_cmd = [None]

    def check_keyboard():
        """Non-blocking stdin check. Works on Unix; on Windows falls back."""
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                line = sys.stdin.readline().strip().lower()
                if line:
                    pending_cmd[0] = line
        except Exception:
            pass  # Windows: select on stdin not supported; skip

    def send_cmd(payload: bytes, label: str):
        if ser is None or not ser.is_open:
            print(f"{C_ERR}  ✗ Not connected — cannot send {label}{C_RESET}")
            return
        fid   = next_fid()
        frame = build_tx_request(payload, fid)
        try:
            ser.write(frame)
            ser.flush()
            print(f"{C_CYAN}  → {label} sent (frame_id={fid}){C_RESET}")
        except Exception as e:
            print(f"{C_ERR}  ✗ Send failed: {e}{C_RESET}")

    def handle_keyboard():
        cmd = pending_cmd[0]
        if cmd is None:
            return
        pending_cmd[0] = None
        if cmd.startswith("p"):
            send_cmd(b"PING", "PING")
        elif cmd.startswith("e"):
            send_cmd(b"E",    "EJECT")
        elif cmd.startswith("a"):
            send_cmd(b"A",    "TX-START")
        elif cmd.startswith("b"):
            send_cmd(b"B",    "TX-STOP")
        elif cmd.startswith("q"):
            raise KeyboardInterrupt

    # ── XBee frame callback ──────────────────────────────────────────────
    def on_xbee_frame(data: bytes):
        nonlocal pkt_count, last_pkt_time, silent_pings

        frame_type = data[0] if data else 0

        # Any frame from the XBee (receive, TX status, modem status) means the
        # module is alive — reset the stuck-ping counter.
        silent_pings = 0

        # 0x90 = Receive Packet (extended / ZB)
        # 0x81 = Receive Packet (16-bit, legacy API1)
        # 0x80 = Receive Packet (64-bit, legacy API1)
        if frame_type not in (0x90, 0x81, 0x80):
            return

        if frame_type == 0x90:
            rf_data = data[12:]          # skip type+64bit+16bit+options
        elif frame_type == 0x80:
            rf_data = data[11:]          # skip type+64bit+rssi+options
        else:                            # 0x81
            rf_data = data[5:]           # skip type+16bit+rssi+options

        if not rf_data:
            return

        # Check for PONG reply
        text = rf_data.decode("ascii", errors="replace")
        if "PONG" in text:
            print(f"{C_CYAN}  ← PONG received: {text.strip()}{C_RESET}")
            last_pkt_time = time.time()
            return

        parsed = parse_nmea_frame(rf_data)
        if parsed is None:
            return

        last_pkt_time = time.time()
        pkt_count    += 1

        print_packet(parsed, 0)

        row = {
            "gs_recv_iso":  datetime.datetime.utcnow().isoformat(),
            "radio_frame_id": frame_type,
            **{k: parsed.get(k, "") for k in CSV_HEADER if k not in ("gs_recv_iso", "radio_frame_id")},
        }
        log.write(row)

    parser = XBeeFrameParser(on_xbee_frame)

    print(f"\n{C_OK}╔══════════════════════════════════════════════════════╗")
    print(  f"║  2025ASI001 Standalone Receiver                      ║")
    print(  f"║  p=PING  e=EJECT  a=TX-START  b=TX-STOP  q=quit     ║")
    print(  f"╚══════════════════════════════════════════════════════╝{C_RESET}\n")

    while True:
        # ── (re)connect serial port ─────────────────────────────────────
        if ser is None or not ser.is_open:
            try:
                print(f"{C_WARN}Connecting to {port} @ {baud} baud …{C_RESET}")
                ser = serial.Serial(
                    port=port,
                    baudrate=baud,
                    timeout=READ_TIMEOUT,
                    write_timeout=1.0,
                )
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                last_pkt_time = time.time()
                parser._reset()
                print(f"{C_OK}Connected. Listening …{C_RESET}\n")
            except serial.SerialException as e:
                print(f"{C_ERR}Cannot open {port}: {e} — retrying in {RECONNECT_S}s{C_RESET}")
                time.sleep(RECONNECT_S)
                continue

        # ── read available bytes ─────────────────────────────────────────
        try:
            waiting = ser.in_waiting
            if waiting > 0:
                chunk = ser.read(min(waiting, 256))
                for b in chunk:
                    parser.feed(b)
            else:
                # nothing in buffer — brief sleep to avoid busy-spin
                time.sleep(0.005)
        except serial.SerialException as e:
            print(f"\n{C_ERR}Serial error: {e} — reconnecting …{C_RESET}")
            try:
                ser.close()
            except Exception:
                pass
            ser = None
            time.sleep(RECONNECT_S)
            continue
        except OSError as e:
            print(f"\n{C_ERR}OS error on serial read: {e} — reconnecting …{C_RESET}")
            try:
                ser.close()
            except Exception:
                pass
            ser = None
            time.sleep(RECONNECT_S)
            continue

        # ── watchdog: PING then AT+FR if stream goes silent ─────────────
        silence = time.time() - last_pkt_time
        if silence > WATCHDOG_S:
            if silent_pings >= 3 and ser is not None and ser.is_open:
                # 3 consecutive PINGs with no response: the local XBee module
                # has likely entered an unresponsive state (known hardware issue
                # under heavy RF load or after interference).  AT+FR sends a
                # firmware-restart command to the dongle over the serial line —
                # it does NOT close the port.  The module reboots in ~1 s and
                # comes back ready to receive without any USB reconnection.
                print(f"{C_ERR}[WDT] {silent_pings} unanswered PINGs — "
                      f"AT+FR to local XBee module …{C_RESET}")
                try:
                    ser.write(build_local_at_cmd(b"FR"))
                    ser.flush()
                except Exception as exc:
                    print(f"{C_ERR}  ✗ AT+FR write failed: {exc}{C_RESET}")
                silent_pings  = 0
                last_pkt_time = time.time() + 1.5   # allow module 1.5 s to restart
            else:
                print(f"{C_WARN}[WDT] No packet for {silence:.0f}s — "
                      f"sending PING ({silent_pings+1}/3 before AT+FR) …{C_RESET}")
                send_cmd(b"PING", "auto-PING")
                silent_pings += 1
                last_pkt_time = time.time()   # reset so we don't spam

        # ── keyboard commands ────────────────────────────────────────────
        check_keyboard()
        handle_keyboard()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Standalone 2025ASI001 XBee receiver")
    ap.add_argument("--port",    default=None,         help="Serial port (auto-detected if omitted)")
    ap.add_argument("--baud",    default=DEFAULT_BAUD, type=int, help=f"Baud rate (default {DEFAULT_BAUD})")
    ap.add_argument("--log-dir", default="logs",       help="Directory for CSV log files (default: logs/)")
    args = ap.parse_args()

    port = args.port
    if port is None:
        port = find_xbee_port()
        if port is None:
            print(f"{C_ERR}No serial port found. Plug in the XBee or pass --port.{C_RESET}")
            sys.exit(1)
        print(f"Auto-detected port: {C_OK}{port}{C_RESET}")

    os.makedirs(args.log_dir, exist_ok=True)

    try:
        run(port, args.baud, args.log_dir)
    except KeyboardInterrupt:
        print(f"\n{C_WARN}Stopped by user.{C_RESET}")


if __name__ == "__main__":
    main()

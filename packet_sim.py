#!/usr/bin/env python3
"""
packet_sim.py — synthetic telemetry generator (no radio required)
=================================================================

Flies a scripted mission (BOOT → TEST_MODE → LAUNCH_PAD → ASCENT → DEPLOY →
DESCENT → AEROBRAKE_RELEASE → IMPACT) and emits real, checksum-correct frames in
the competition packet format so the dashboard can be exercised end-to-end.

WHY A TCP SOCKET IS THE DEFAULT ON WINDOWS
------------------------------------------
Windows has no built-in way to create a virtual COM port pair (Linux has
``socat``/``pty``, Windows does not).  The usual answer is the third-party
``com0com`` driver, which is unsigned-driver territory and a genuine hassle to
install on a locked-down competition laptop.

pyserial solves this for us: ``serial.serial_for_url("socket://host:port")``
speaks the same ``Serial`` API over a TCP connection.  The ground station opens
its port through ``serial_for_url``, so pointing it at ``socket://127.0.0.1:5555``
exercises **the entire real code path** — ring buffer, frame sync, checksum,
threading, logging — with zero drivers installed.  That entry is pre-loaded in
the port dropdown.

If you *do* have com0com (or you are on Linux with a socat pty pair), use
``--serial COM11`` to write into the paired port instead and the dashboard reads
the other half as a genuine COM device.

USAGE
-----
    python packet_sim.py                        # TCP server on 127.0.0.1:5555
    python packet_sim.py --rate 20              # 20 Hz (competition nominal)
    python packet_sim.py --serial COM11         # write to a real/virtual COM port
    python packet_sim.py --stdout               # just print frames to the console

FAULT INJECTION (for the robustness / fault-injection test requirement)
-----------------------------------------------------------------------
    --corrupt-rate 0.05    5 % of frames get a deliberately wrong checksum
    --garbage-rate 0.02    2 % of the time, inject random non-frame bytes
    --truncate-rate 0.02   2 % of frames are cut off mid-way (tests resync)
    --drop-rate 0.05       5 % of frames are simply not sent (tests staleness)
    --burst 40             every ~10 s, pause 3 s then dump a 40-packet burst
    --chaos                shorthand for a nasty-but-realistic mix of the above

Example — prove the dashboard survives a hostile link:

    python packet_sim.py --rate 20 --chaos
"""

from __future__ import annotations

import argparse
import math
import os
import random
import socket
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telemetry_packet import build_frame  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555
DEFAULT_TEAM_ID = "2025ASI001"

# Ground-station reference position (Jammu) — the sim walks away from here.
BASE_LAT = 32.7266
BASE_LON = 74.8570
BASE_ALT = 320.0

GARBAGE_ALPHABET = b"abcdefghijklmnopqrstuvwxyz0123456789 ,.;:!?#@%&/\\<>[]{}"


# ---------------------------------------------------------------------------
# Flight profile
# ---------------------------------------------------------------------------

class MissionSim:
    """A simple but physically plausible sounding-rocket / CanSat profile.

    Timeline (seconds after start), chosen so a full mission runs in ~75 s:
        0–3     BOOT               on the pad, altitude 0
        3–6     TEST_MODE
        6–12    LAUNCH_PAD
        12–22   ASCENT             burn then coast to apogee (780 m AGL)
        22–24   DEPLOY             drogue out at apogee, −20 m/s
        24–55   DESCENT            drogue descent at −20 m/s → ~120 m
        55–75   AEROBRAKE_RELEASE  main out, −6 m/s to touchdown
        75+     IMPACT             on the ground, altitude flat at 0
    """

    APOGEE_M = 780.0
    T_LAUNCH = 12.0
    T_APOGEE = 22.0
    T_DROGUE_END = 24.0     # end of the DEPLOY transient
    T_MAIN = 55.0           # aerobrake / main release
    DROGUE_MPS = 20.0
    MAIN_MPS = 6.0

    def __init__(self, team_id: str = DEFAULT_TEAM_ID) -> None:
        self.team_id = team_id
        self.packet_count = 0
        self.t0 = time.time()
        self.landed_alt: Optional[float] = None

    # -- physics-ish helpers ------------------------------------------------

    @staticmethod
    def _pressure_from_alt(alt_m: float) -> float:
        """International Standard Atmosphere, troposphere form, in hPa."""
        return 1013.25 * (1.0 - 2.25577e-5 * max(alt_m, 0.0)) ** 5.25588

    @staticmethod
    def _temp_from_alt(alt_m: float) -> float:
        """ISA lapse rate, ~6.5 °C per 1000 m, from a 28 °C field day."""
        return 28.0 - 6.5e-3 * max(alt_m, 0.0)

    def _profile(self, t: float):
        """Return ``(agl_altitude_m, fsm_state)`` at mission time *t* seconds."""
        if t < 3.0:
            return 0.0, 0
        if t < 6.0:
            return 0.0, 1
        if t < self.T_LAUNCH:
            return 0.0, 2
        if t < self.T_APOGEE:
            # Quadratic climb that flattens out exactly at apogee.
            frac = (t - self.T_LAUNCH) / (self.T_APOGEE - self.T_LAUNCH)
            return self.APOGEE_M * (2 * frac - frac * frac), 3

        # Altitude at the moment the drogue transient ends.
        alt_drogue_start = self.APOGEE_M - self.DROGUE_MPS * (
            self.T_DROGUE_END - self.T_APOGEE
        )
        if t < self.T_DROGUE_END:
            return self.APOGEE_M - self.DROGUE_MPS * (t - self.T_APOGEE), 4

        # Altitude at main / aerobrake release.
        alt_main = alt_drogue_start - self.DROGUE_MPS * (
            self.T_MAIN - self.T_DROGUE_END
        )
        if t < self.T_MAIN:
            alt = alt_drogue_start - self.DROGUE_MPS * (t - self.T_DROGUE_END)
            return max(alt, 0.0), 5

        alt = alt_main - self.MAIN_MPS * (t - self.T_MAIN)
        if alt <= 0.0:
            return 0.0, 7          # IMPACT — on the ground and staying there
        return alt, 6

    # -- packet assembly -----------------------------------------------------

    def next_payload(self) -> str:
        """Build the comma-separated body (everything between ``$`` and ``*``)."""
        t = time.time() - self.t0
        alt_agl, state = self._profile(t)
        self.packet_count += 1

        altitude = BASE_ALT + alt_agl + random.gauss(0.0, 0.35)
        pressure = self._pressure_from_alt(altitude) + random.gauss(0.0, 0.08)
        temperature = self._temp_from_alt(altitude) + random.gauss(0.0, 0.15)

        # Battery: slow linear droop plus a sag under motor burn.
        voltage = 8.20 - 0.0035 * t - (0.25 if 12.0 <= t < 16.0 else 0.0)
        voltage += random.gauss(0.0, 0.008)

        # IMU: 1 g resting on Z, big kick on burn, buffeting under descent.
        if 12.0 <= t < 16.0:
            acc_z = 9.81 * 7.0 + random.gauss(0.0, 1.8)
            acc_x, acc_y = random.gauss(0, 2.2), random.gauss(0, 2.2)
        elif state in (5, 6):
            acc_z = 9.81 + random.gauss(0.0, 2.6)
            acc_x, acc_y = random.gauss(0, 1.9), random.gauss(0, 1.9)
        elif state == 7:
            acc_z = 9.81 + random.gauss(0.0, 0.05)
            acc_x, acc_y = random.gauss(0, 0.05), random.gauss(0, 0.05)
        else:
            acc_z = 9.81 + random.gauss(0.0, 0.12)
            acc_x, acc_y = random.gauss(0, 0.1), random.gauss(0, 0.1)

        spin = 0.0 if state in (0, 1, 2, 7) else 140.0 * math.exp(-(t - 12.0) / 25.0)
        gyro_x = random.gauss(0, 6.0) + (spin * 0.15)
        gyro_y = random.gauss(0, 6.0) - (spin * 0.10)
        gyro_z = spin + random.gauss(0, 4.0)

        # GPS/NavIC: drifts downrange with the wind once it leaves the pad.
        downrange = max(0.0, t - 12.0)
        lat = BASE_LAT + 3.1e-5 * downrange + random.gauss(0, 1.5e-6)
        lon = BASE_LON + 5.4e-5 * downrange + random.gauss(0, 1.5e-6)
        nav_alt = altitude + random.gauss(0.0, 2.0)
        sats = 4 if t < 6 else (9 + random.randint(-1, 2))

        nav_time = time.strftime("%H%M%S", time.gmtime()) + ".00"

        return (
            "%s,%.2f,%d,%.2f,%.2f,%.2f,%.2f,%s,"
            "%.6f,%.6f,%.2f,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d"
            % (
                self.team_id, t, self.packet_count,
                altitude, pressure, temperature, voltage, nav_time,
                lat, lon, nav_alt, sats,
                acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, state,
            )
        )

    def next_frame(self) -> str:
        """A complete, checksum-correct ``$...*XX`` frame."""
        return build_frame(self.next_payload())


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------

class FaultInjector:
    """Deliberately damages the outgoing byte stream to test GCS robustness."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.corrupt_rate = args.corrupt_rate
        self.garbage_rate = args.garbage_rate
        self.truncate_rate = args.truncate_rate
        self.drop_rate = args.drop_rate

    def apply(self, frame: str) -> Optional[bytes]:
        """Return the bytes to transmit, or ``None`` to drop the frame."""
        if random.random() < self.drop_rate:
            return None

        out = bytearray()

        if random.random() < self.garbage_rate:
            # Junk *before* the frame: exercises the "discard leading noise" path.
            length = random.randint(1, 24)
            out.extend(random.choice(GARBAGE_ALPHABET) for _ in range(length))

        if random.random() < self.corrupt_rate:
            # Flip one payload byte but keep the original checksum, so the frame
            # is structurally perfect and only the XOR check catches it.
            body, _, checksum = frame[1:].rpartition("*")
            if body:
                index = random.randrange(len(body))
                original = body[index]
                replacement = chr((ord(original) - 32 + 1) % 94 + 32)
                if replacement == ",":  # keep the field count intact
                    replacement = "."
                body = body[:index] + replacement + body[index + 1:]
            frame = "$%s*%s" % (body, checksum)

        data = frame.encode("ascii", errors="replace")

        if random.random() < self.truncate_rate and len(data) > 8:
            # Cut the frame short: the next '$' must trigger a resync.
            data = data[: random.randint(4, len(data) - 4)]
            out.extend(data)
            return bytes(out)

        out.extend(data)
        out.extend(b"\r\n")
        return bytes(out)


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------

def _emit_loop(send, args: argparse.Namespace) -> None:
    """Generate frames at ``args.rate`` Hz and hand the bytes to *send*.

    ``send(data: bytes) -> None`` may raise to signal that the peer went away.
    """
    sim = MissionSim(args.team_id)
    faults = FaultInjector(args)
    period = 1.0 / max(args.rate, 0.1)
    next_due = time.monotonic()
    sent = dropped = 0
    next_burst = time.monotonic() + args.burst_interval if args.burst else None
    last_report = time.monotonic()

    while True:
        now = time.monotonic()

        # --- burst mode: go quiet, then dump a backlog all at once ----------
        if next_burst is not None and now >= next_burst:
            print("[sim] burst: %.1f s of silence, then %d packets back-to-back"
                  % (args.burst_gap, args.burst))
            time.sleep(args.burst_gap)
            for _ in range(args.burst):
                data = faults.apply(sim.next_frame())
                if data is None:
                    dropped += 1
                    continue
                send(data)
                sent += 1
            next_burst = time.monotonic() + args.burst_interval
            next_due = time.monotonic()
            continue

        if now < next_due:
            time.sleep(min(next_due - now, 0.01))
            continue
        # Absolute schedule, so a slow write does not permanently skew the rate.
        next_due += period
        if next_due < now - period:
            next_due = now + period

        data = faults.apply(sim.next_frame())
        if data is None:
            dropped += 1
        else:
            send(data)
            sent += 1

        if args.duration and (time.time() - sim.t0) >= args.duration:
            print("[sim] duration reached — %d sent, %d dropped" % (sent, dropped))
            return

        if now - last_report >= 5.0:
            last_report = now
            print("[sim] t=%6.1fs  pkt=%-6d sent=%-6d dropped=%d"
                  % (time.time() - sim.t0, sim.packet_count, sent, dropped))


def run_tcp_server(args: argparse.Namespace) -> int:
    """Serve frames to whoever connects (the GCS via ``socket://host:port``)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((args.host, args.port))
    except OSError as exc:
        print("[sim] cannot bind %s:%d — %s" % (args.host, args.port, exc))
        return 1
    server.listen(1)
    print("[sim] listening on %s:%d" % (args.host, args.port))
    print("[sim] in the GCS, connect to:  socket://%s:%d" % (args.host, args.port))

    try:
        while True:
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("[sim] ground station connected from %s:%d" % addr)
            try:
                _emit_loop(conn.sendall, args)
                if args.duration:
                    break
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                print("[sim] client disconnected (%s) — waiting for a new one" % exc)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
            if args.once:
                break
    except KeyboardInterrupt:
        print("\n[sim] stopped.")
    finally:
        server.close()
    return 0


def run_serial(args: argparse.Namespace) -> int:
    """Write frames into a real or virtual COM port (com0com / socat pty)."""
    try:
        import serial
    except ImportError:
        print("[sim] pyserial is required for --serial (pip install pyserial)")
        return 1

    try:
        port = serial.serial_for_url(args.serial, baudrate=args.baud, timeout=1.0)
    except Exception as exc:
        print("[sim] cannot open %s — %s" % (args.serial, exc))
        return 1

    print("[sim] writing to %s @ %d baud" % (args.serial, args.baud))
    print("[sim] point the GCS at the *paired* port of this virtual pair")
    try:
        _emit_loop(port.write, args)
    except KeyboardInterrupt:
        print("\n[sim] stopped.")
    except Exception as exc:
        print("[sim] write failed: %s" % exc)
        return 1
    finally:
        try:
            port.close()
        except Exception:
            pass
    return 0


def run_stdout(args: argparse.Namespace) -> int:
    """Print frames to the console — useful for eyeballing the wire format."""
    def send(data: bytes) -> None:
        sys.stdout.write(data.decode("ascii", errors="replace"))
        sys.stdout.flush()

    try:
        _emit_loop(send, args)
    except KeyboardInterrupt:
        print("\n[sim] stopped.")
    return 0


# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic CanSat/rocketry telemetry generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Default transport is a TCP server; connect the GCS to "
               "socket://127.0.0.1:5555 (no virtual COM driver needed).",
    )
    transport = parser.add_argument_group("transport")
    transport.add_argument("--host", default=DEFAULT_HOST)
    transport.add_argument("--port", type=int, default=DEFAULT_PORT)
    transport.add_argument("--serial", default=None,
                           metavar="DEVICE",
                           help="Write to this COM port instead of TCP "
                                "(needs com0com on Windows, socat on Linux).")
    transport.add_argument("--baud", type=int, default=9600)
    transport.add_argument("--stdout", action="store_true",
                           help="Print frames to the console instead.")
    transport.add_argument("--once", action="store_true",
                           help="Exit after the first client disconnects.")

    flight = parser.add_argument_group("flight")
    flight.add_argument("--team-id", default=DEFAULT_TEAM_ID)
    flight.add_argument("--rate", type=float, default=10.0,
                        help="Packets per second (default 10; competition is 20).")
    flight.add_argument("--duration", type=float, default=0.0,
                        help="Stop after N seconds (0 = run forever).")
    flight.add_argument("--seed", type=int, default=None,
                        help="Seed the RNG for a repeatable run.")

    faults = parser.add_argument_group("fault injection")
    faults.add_argument("--corrupt-rate", type=float, default=0.0,
                        help="Fraction of frames with a bad checksum.")
    faults.add_argument("--garbage-rate", type=float, default=0.0,
                        help="Fraction of frames preceded by random junk bytes.")
    faults.add_argument("--truncate-rate", type=float, default=0.0,
                        help="Fraction of frames cut off mid-transmission.")
    faults.add_argument("--drop-rate", type=float, default=0.0,
                        help="Fraction of frames never sent at all.")
    faults.add_argument("--burst", type=int, default=0,
                        help="Packets to dump in a burst (0 = no bursts).")
    faults.add_argument("--burst-interval", type=float, default=10.0,
                        help="Seconds between bursts.")
    faults.add_argument("--burst-gap", type=float, default=3.0,
                        help="Seconds of silence before each burst.")
    faults.add_argument("--chaos", action="store_true",
                        help="Preset: a realistically hostile link.")

    args = parser.parse_args(argv)

    if args.chaos:
        args.corrupt_rate = max(args.corrupt_rate, 0.06)
        args.garbage_rate = max(args.garbage_rate, 0.03)
        args.truncate_rate = max(args.truncate_rate, 0.03)
        args.drop_rate = max(args.drop_rate, 0.05)
        if not args.burst:
            args.burst = 40

    if args.seed is not None:
        random.seed(args.seed)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.stdout:
        return run_stdout(args)
    if args.serial:
        return run_serial(args)
    return run_tcp_server(args)


if __name__ == "__main__":
    sys.exit(main())

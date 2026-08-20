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
    python packet_sim.py --payload-type cansat  # SPS30 + reaction wheel (default)
    python packet_sim.py --payload-type rocket  # solenoid + nichrome flags
    python packet_sim.py --payload-type generic # legacy 19-field v1 frames
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

from telemetry_packet import (  # noqa: E402
    FIELD_COUNT_CANSAT,
    FIELD_COUNT_ROCKET,
    FIELD_COUNT_V1,
    PAYLOAD_CANSAT,
    PAYLOAD_GENERIC,
    PAYLOAD_ROCKET,
    SPS30_MAX_UGM3,
    build_frame,
)

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

    Timeline (seconds after start).  The main/aerobrake event is triggered by
    *altitude* (400 m AGL, per the CDR) rather than by a hard-coded time, so the
    timeline below is derived from the descent rates rather than assumed::

        0-3      BOOT               on the pad, altitude 0
        3-6      TEST_MODE
        6-12     LAUNCH_PAD
        12-22    ASCENT             burn then coast to apogee (780 m AGL)
        22-24    DEPLOY             drogue out at apogee, -20 m/s
        24-41    DESCENT            drogue descent at -20 m/s -> 400 m AGL
        41-91    AEROBRAKE_RELEASE  main out at 400 m AGL, -8 m/s to touchdown
        91+      IMPACT             on the ground, altitude flat at 0

    Vehicle-specific sensors are generated according to *payload_type*:

    * ``CANSAT`` -- Sensirion SPS30 particulate channels, reaction-wheel RPM and
      the recovery-stage sequencer.
    * ``ROCKET`` -- solenoid latch and nichrome cutter status flags.
    * ``GENERIC`` -- neither; emits the legacy 19-field v1 frame, which is what
      the pre-v2 firmware and the older log files use.
    """

    APOGEE_M = 780.0
    T_LAUNCH = 12.0
    T_APOGEE = 22.0
    T_DROGUE_END = 24.0        # end of the DEPLOY transient
    DROGUE_MPS = 20.0
    MAIN_DEPLOY_AGL = 400.0    # nichrome cutter altitude, per the CDR
    MAIN_MPS = 8.0

    #: Reaction wheel saturation, per the CanSat CDR.
    WHEEL_MAX_RPM = 1124

    def __init__(self, team_id: str = DEFAULT_TEAM_ID,
                 payload_type: str = PAYLOAD_CANSAT) -> None:
        self.team_id = team_id
        self.payload_type = payload_type
        self.packet_count = 0
        self.t0 = time.time()

        # Derived timeline.  Computed once so _profile() stays branch-cheap.
        self.alt_drogue_start = self.APOGEE_M - self.DROGUE_MPS * (
            self.T_DROGUE_END - self.T_APOGEE
        )
        self.t_main = self.T_DROGUE_END + (
            self.alt_drogue_start - self.MAIN_DEPLOY_AGL
        ) / self.DROGUE_MPS
        self.t_touchdown = self.t_main + self.MAIN_DEPLOY_AGL / self.MAIN_MPS

        # Latched event flags.  Pyrotechnics never un-fire, so these are held as
        # state rather than recomputed from t -- that way a clock glitch in the
        # simulator cannot make an already-fired charge report SAFE again.
        self._solenoid_fired = False
        self._nichrome_fired = False
        self._recovery_stage = 0
        self._wheel_rpm = 0.0

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

        if t < self.T_DROGUE_END:
            return self.APOGEE_M - self.DROGUE_MPS * (t - self.T_APOGEE), 4

        if t < self.t_main:
            alt = self.alt_drogue_start - self.DROGUE_MPS * (t - self.T_DROGUE_END)
            return max(alt, 0.0), 5

        # Main / aerobrake released at MAIN_DEPLOY_AGL.
        alt = self.MAIN_DEPLOY_AGL - self.MAIN_MPS * (t - self.t_main)
        if alt <= 0.0:
            return 0.0, 7          # IMPACT — on the ground and staying there
        return alt, 6

    # -- vehicle-specific sensor models -------------------------------------

    def _particulates(self, t: float, alt_agl: float, state: int):
        """Sensirion SPS30 mass concentrations, micrograms/m^3.

        The scientific case for flying the SPS30 is measuring the particulate
        column on the way down, so the interesting signal is the descent: the
        CanSat falls through the ejection-charge smoke, then through the dust
        layer that sits near the ground.  Ambient air on a clear field day is
        only a few micrograms/m^3, so the plume is what dominates.

        SPS30 channels are *cumulative* mass concentrations, so the sensor can
        never report PM1.0 > PM2.5 > PM10.  The model builds them additively to
        guarantee that ordering holds for every sample.
        """
        # Clean-air baseline with slow drift.
        base = 4.0 + 1.5 * math.sin(t / 23.0)

        plume = 0.0
        if state == 4:
            # Ejection charge fires right beside the inlet.
            plume = 190.0
        elif state in (5, 6):
            # Descending through smoke aloft plus ground dust below ~250 m.
            plume = 420.0 * math.exp(-max(alt_agl, 0.0) / 250.0)
        elif state == 7:
            # Landing kicks up dust, which then settles over ~20 s.
            plume = 70.0 * math.exp(-max(0.0, t - self.t_touchdown) / 20.0)

        pm1 = base + 0.30 * plume + random.gauss(0.0, 0.4)
        pm25 = pm1 + 3.5 + 0.30 * plume + abs(random.gauss(0.0, 0.5))
        pm10 = pm25 + 5.0 + 0.40 * plume + abs(random.gauss(0.0, 0.8))

        clamp = lambda v: max(0.0, min(SPS30_MAX_UGM3, v))  # noqa: E731
        return clamp(pm1), clamp(pm25), clamp(pm10)

    def _reaction_wheel(self, state: int, gyro_z: float) -> int:
        """Reaction-wheel RPM for the CanSat active stabilisation system.

        The wheel counter-spins against the body rate, so its RPM tracks the
        negated yaw rate.  It is held at rest while the vehicle is on the pad
        (states BOOT / TEST_MODE / LAUNCH_PAD) and after touchdown, and it
        saturates at the CDR figure of ~1124 RPM.  A first-order lag keeps the
        commanded value from stepping instantaneously, which is what a real
        wheel with rotor inertia does.
        """
        if state in (0, 1, 2, 7):
            target = 0.0
        else:
            target = -gyro_z * 7.5

        target = max(-self.WHEEL_MAX_RPM, min(self.WHEEL_MAX_RPM, target))
        # Lag towards the target: 35 % of the error per packet.
        self._wheel_rpm += 0.35 * (target - self._wheel_rpm)
        return int(round(self._wheel_rpm + random.gauss(0.0, 4.0)))

    def _update_events(self, state: int) -> None:
        """Latch the recovery events off FSM state transitions.

        CanSat recovery stage and rocket pyro flags are driven by the same two
        events, so they stay in step with the FSM banner on the dashboard:

        * ``DEPLOY`` (state 4)            -> drogue out  / solenoid latch fires
        * ``AEROBRAKE_RELEASE`` (state 6) -> parafoil out / nichrome cutter fires
        """
        if state >= 4 and state != 7:
            self._solenoid_fired = True
            self._recovery_stage = max(self._recovery_stage, 1)
        if state >= 6 and state != 7:
            self._nichrome_fired = True
            self._recovery_stage = max(self._recovery_stage, 2)
        if state == 7:
            # Landed: both events are long since latched.
            self._solenoid_fired = True
            self._nichrome_fired = True
            self._recovery_stage = 2

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

        # Gyro noise floor. A MEMS gyro on a vehicle that is physically
        # stationary (on the pad, or landed) reads a few tenths of a degree per
        # second, not several degrees -- the old flat 6 deg/s noise made the
        # attitude display wander visibly while the rocket was sitting still,
        # which looked like a bug in the estimator rather than the honest
        # integral of the transmitted rates that it was.
        stationary = state in (0, 1, 2, 7)
        noise = 0.35 if stationary else 6.0
        spin = 0.0 if stationary else 140.0 * math.exp(-(t - 12.0) / 25.0)
        gyro_x = random.gauss(0, noise) + (spin * 0.15)
        gyro_y = random.gauss(0, noise) - (spin * 0.10)
        gyro_z = spin + random.gauss(0, noise * 0.7)

        # GPS/NavIC: drifts downrange with the wind once it leaves the pad.
        downrange = max(0.0, t - 12.0)
        lat = BASE_LAT + 3.1e-5 * downrange + random.gauss(0, 1.5e-6)
        lon = BASE_LON + 5.4e-5 * downrange + random.gauss(0, 1.5e-6)
        nav_alt = altitude + random.gauss(0.0, 2.0)
        sats = 4 if t < 6 else (9 + random.randint(-1, 2))

        nav_time = time.strftime("%H%M%S", time.gmtime()) + ".00"

        self._update_events(state)

        # Fields shared by every format, minus the leading TEAM_ID/PAYLOAD_TYPE.
        common = (
            "%.2f,%d,%.2f,%.2f,%.2f,%.2f,%s,"
            "%.6f,%.6f,%.2f,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d"
            % (
                t, self.packet_count,
                altitude, pressure, temperature, voltage, nav_time,
                lat, lon, nav_alt, sats,
                acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, state,
            )
        )

        if self.payload_type == PAYLOAD_CANSAT:
            pm1, pm25, pm10 = self._particulates(t, alt_agl, state)
            rpm = self._reaction_wheel(state, gyro_z)
            body = "%s,%s,%s,%.2f,%.2f,%.2f,%d,%d" % (
                self.team_id, PAYLOAD_CANSAT, common,
                pm1, pm25, pm10, rpm, self._recovery_stage,
            )
            expected = FIELD_COUNT_CANSAT
        elif self.payload_type == PAYLOAD_ROCKET:
            body = "%s,%s,%s,%d,%d" % (
                self.team_id, PAYLOAD_ROCKET, common,
                1 if self._solenoid_fired else 0,
                1 if self._nichrome_fired else 0,
            )
            expected = FIELD_COUNT_ROCKET
        else:
            # Legacy v1: no PAYLOAD_TYPE, no vehicle-specific tail.
            body = "%s,%s" % (self.team_id, common)
            expected = FIELD_COUNT_V1

        # Guard against the generator and the parser drifting apart: a field
        # count mismatch here is a bug in this file, not a link problem, and it
        # would otherwise surface as a flood of "corrupt" packets in the GCS.
        actual = body.count(",") + 1
        if actual != expected:
            raise AssertionError(
                "packet_sim built a %s body with %d fields, expected %d"
                % (self.payload_type, actual, expected)
            )
        return body

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
    sim = MissionSim(args.team_id, payload_type=args.payload_type)
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
    flight.add_argument(
        "--payload-type", default="cansat",
        choices=["cansat", "rocket", "generic"],
        help="Vehicle to simulate. 'cansat' adds the SPS30 particulate "
             "channels, reaction-wheel RPM and recovery stage; 'rocket' adds "
             "the solenoid and nichrome status flags; 'generic' emits the "
             "legacy 19-field v1 frame (default: cansat).",
    )
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

    # Map the friendly CLI spelling onto the wire token.
    args.payload_type = {
        "cansat": PAYLOAD_CANSAT,
        "rocket": PAYLOAD_ROCKET,
        "generic": PAYLOAD_GENERIC,
    }[args.payload_type]

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

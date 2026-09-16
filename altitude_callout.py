"""
altitude_callout.py
===================

Speaks the vehicle's altitude aloud during flight, so the operator can watch
the sky instead of the screen.

WHY A THREAD
------------
Speech synthesis blocks for the length of the utterance -- two to three
seconds for "Altitude 590 meters" on Windows SAPI5 -- and the GUI thread
cannot be away that long: the charts would freeze and, worse, the merge
buffer would stall. So the actual speaking happens here, on its own QThread,
the same way serial ingestion and CSV logging do. The GUI thread only ever
hands over a number and returns immediately.

TWO KINDS OF CALLOUT, TWO POLICIES
----------------------------------
*Altitude* is periodic and perishable. There is deliberately no queue for
it: if the GUI asks for an altitude while something is still being spoken,
the request is refused (``request`` returns False) and the GUI tries again
on its next tick. A queue would do the wrong thing in exactly the situation
it is meant for -- if speech ever ran slower than the cadence it would
accumulate a backlog of altitudes the vehicle left behind seconds ago, and
the operator would hear the past. Silence for one tick and then the current
altitude is strictly better than a stale number.

*State changes* are events, and each one is information the operator does
not get again. Those go into a short FIFO (``announce_state``) and are
spoken as soon as the engine is free, ahead of any waiting altitude. A
transition that lands mid-utterance is therefore delayed by at most one
utterance, never dropped. The queue is bounded only as a guard against a
runaway source; transitions are seconds apart in any real flight.

ONE ENGINE PER UTTERANCE
------------------------
pyttsx3's SAPI5 driver is not reliably reusable: measured here, the first
``runAndWait()`` on an engine spoke for 3.3 s, and the second and third
returned in 0.2 s having spoken nothing (their started-utterance events never
fired). Creating the engine fresh for each utterance costs a few hundred
milliseconds of COM setup on this thread -- which nobody notices -- and every
utterance then completes, verified through the engine's own
finished-utterance event. Do not "optimise" this back to a shared engine
without re-running that measurement.

OFFLINE
-------
pyttsx3 drives the operating system's own speech engine (SAPI5 on Windows,
NSSpeechSynthesizer on macOS, espeak on Linux). Nothing leaves the machine,
and it works with no network at all, which is the point in the field.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Optional

from PyQt5.QtCore import QThread, pyqtSignal

try:
    import pyttsx3
    TTS_AVAILABLE = True
    TTS_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the host
    pyttsx3 = None
    TTS_AVAILABLE = False
    TTS_IMPORT_ERROR = "%s: %s" % (type(exc).__name__, exc)

#: Words per minute. The SAPI5 default is 200, at which "Altitude 1250 meters"
#: takes ~3.2 s -- longer than the 3 s cadence, so every other callout would
#: be skipped as overlapping. At 240 every phrase measured here finishes in
#: 1.7-2.6 s including engine setup, and numbers are still perfectly clear.
SPEECH_RATE_WPM = 240

#: Altitude is rounded to this many metres before it is spoken. At flight
#: speeds the vehicle moves tens of metres between callouts, so "five hundred
#: ninety-three" is false precision that costs syllables; "five hundred
#: ninety" is what the operator actually needs.
ROUND_TO_M = 10


def format_altitude(altitude_m: float) -> str:
    """The spoken phrase for *altitude_m*, e.g. ``"Altitude 590 meters"``.

    Digits, not words: SAPI reads "590" as "five hundred ninety" on its own,
    and the digit form is shorter to synthesise than spelling it out.
    """
    rounded = int(round(altitude_m / ROUND_TO_M)) * ROUND_TO_M
    if rounded < 0:
        return "Altitude minus %d meters" % -rounded
    return "Altitude %d meters" % rounded


#: Sentinel queued by warm_up(): build an engine, say nothing, discard it.
_WARM_UP = object()

#: Upper bound on queued state announcements. Transitions in a flight are
#: seconds to minutes apart, so this is a guard against a flapping source,
#: not a working limit. If it is ever hit the OLDEST is dropped and counted,
#: because by then the newer states are the truth.
STATE_QUEUE_MAX = 4


def format_state(fsm_name: str) -> str:
    """The spoken phrase for an FSM state, e.g. ``"State: Aerobrake release"``.

    The wire names are SHOUTED_WITH_UNDERSCORES, which SAPI may spell out
    letter by letter; lower-cased words with one capital read as words.
    """
    words = str(fsm_name).replace("_", " ").strip().lower()
    return "State: %s" % (words[:1].upper() + words[1:])


class VoiceCallout(QThread):
    """Owns the speech engine: periodic altitude, plus state changes on demand."""

    #: ``(phrase,)`` as an utterance begins.
    started = pyqtSignal(str)
    #: ``(phrase,)`` after an utterance has been spoken to completion.
    spoken = pyqtSignal(str)
    #: ``(message,)`` once per distinct failure; the callout then stays quiet.
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cond = threading.Condition()
        #: The one altitude waiting to be spoken, if any. A slot, not a queue.
        self._pending: Optional[float] = None
        #: State-change phrases waiting to be spoken, oldest first.
        self._states: Deque[str] = deque()
        self._speaking = False
        self._warmed = False
        self._running = True
        self._last_error = ""
        #: Name of the OS voice actually in use, for the operator's log.
        self.voice_name = ""
        self.engine_name = ""
        #: Altitude requests refused because an utterance was in progress.
        self.skipped = 0
        #: State announcements lost to a full queue (expected to stay 0).
        self.states_dropped = 0
        self.spoken_count = 0

    # ------------------------------------------------------------------
    # GUI-thread API: never blocks, never raises
    # ------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        with self._cond:
            return (self._speaking or self._pending is not None
                    or bool(self._states))

    def request(self, altitude_m: float) -> bool:
        """Ask for *altitude_m* to be spoken now.

        Returns False -- and speaks nothing -- if an utterance is still in
        progress or a state announcement is waiting. That is the whole
        overlap policy for altitude: the caller's next tick will offer a
        fresher number, which is the one worth hearing.
        """
        if not TTS_AVAILABLE:
            return False
        with self._cond:
            if self._speaking or self._pending is not None or self._states:
                self.skipped += 1
                return False
            self._pending = float(altitude_m)
            self._cond.notify()
        return True

    def announce_state(self, fsm_name: str) -> None:
        """Speak a state change as soon as the engine is free. Never refused.

        Unlike an altitude, a transition is not superseded by the next one:
        missing "State: Deploy" is a real loss even if "State: Descent"
        follows. So this queues -- briefly, behind at most the utterance in
        progress -- and is spoken ahead of any waiting altitude.
        """
        if not TTS_AVAILABLE:
            return
        with self._cond:
            if len(self._states) >= STATE_QUEUE_MAX:
                self._states.popleft()
                self.states_dropped += 1
            self._states.append(format_state(fsm_name))
            self._cond.notify()

    def warm_up(self) -> None:
        """Pay the one-time engine set-up cost now, silently.

        The first engine of a process costs ~1 s more than the rest (comtypes
        builds its SAPI type library on first use), which made the first
        callout of every session run long enough for the second to be
        skipped as overlapping. Called when telemetry is started -- the link
        is up by then, and a burst of set-up work mid-stream was measured to
        cost the merge nothing -- so the first callout comes out on time.
        Idempotent; never speaks.
        """
        if not TTS_AVAILABLE:
            return
        with self._cond:
            if (self._warmed or self._speaking or self._pending is not None
                    or self._states):
                return
            self._warmed = True
            self._pending = _WARM_UP
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._pending = None
            self._states.clear()
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def run(self) -> None:
        if not TTS_AVAILABLE:
            self._fail("pyttsx3 is not installed (%s); altitude callouts "
                       "are unavailable" % TTS_IMPORT_ERROR)
            return
        # Deliberately NO engine work here. Creating a SAPI engine costs
        # ~0.4 s of CPU, and at start-up that lands exactly while the
        # operator is connecting the radios; measured on the dual-radio bench
        # it delayed one radio's connection enough for the merge to give up
        # on its first packets. The voice is identified from the engine that
        # speaks the first real callout instead, well after the link is up.
        while True:
            with self._cond:
                while (self._running and self._pending is None
                       and not self._states):
                    self._cond.wait(timeout=0.5)
                if not self._running:
                    return
                # State changes first: they are the rarer, more valuable
                # event, and an altitude is at most one tick from a retry.
                if self._states:
                    job = self._states.popleft()
                else:
                    job = self._pending
                    self._pending = None
                self._speaking = True
            try:
                if job is _WARM_UP:
                    self._warm()
                elif isinstance(job, str):
                    self._speak(job)
                else:
                    self._speak(format_altitude(job))
            finally:
                with self._cond:
                    self._speaking = False

    def _warm(self) -> None:
        try:
            engine = pyttsx3.init()
            self._identify(engine)
            engine.stop()
            del engine
        except Exception as exc:
            self._fail("speech engine unavailable: %s: %s"
                       % (type(exc).__name__, exc))

    def _identify(self, engine) -> None:
        """Record which driver and OS voice are speaking, once, for the log."""
        if self.voice_name:
            return
        try:
            self.engine_name = type(engine.proxy._driver).__name__
        except Exception:
            self.engine_name = "unknown driver"
        try:
            voice_id = engine.getProperty("voice")
            for voice in engine.getProperty("voices"):
                if voice.id == voice_id:
                    self.voice_name = voice.name
                    break
        except Exception:
            self.voice_name = "unknown voice"

    def _speak(self, phrase: str) -> None:
        completed_flags = []
        try:
            # Fresh engine every time -- see the module docstring for why.
            engine = pyttsx3.init()
            self._identify(engine)
            engine.setProperty("rate", SPEECH_RATE_WPM)
            # pyttsx3 invokes this with KEYWORD arguments (name=, completed=)
            # and silently swallows any exception a callback raises -- so a
            # callback with the wrong parameter names never runs and the
            # utterance looks unconfirmed. The names below are load-bearing.
            engine.connect("finished-utterance",
                           lambda name, completed: completed_flags.append(
                               bool(completed)))
            self.started.emit(phrase)
            engine.say(phrase)
            engine.runAndWait()
            try:
                engine.stop()
            except Exception:
                pass
            del engine
        except Exception as exc:
            self._fail("speech failed: %s: %s" % (type(exc).__name__, exc))
            return
        if completed_flags and completed_flags[-1]:
            self.spoken_count += 1
            self.spoken.emit(phrase)
        else:
            # The engine returned without confirming the utterance. Count it
            # as not spoken rather than pretend; the next tick tries again.
            self._fail("speech engine returned without completing an "
                       "utterance; the next callout will retry")

    def _fail(self, message: str) -> None:
        # Repeating the same failure every 3 s would be its own problem.
        if message != self._last_error:
            self._last_error = message
            self.error_occurred.emit(message)

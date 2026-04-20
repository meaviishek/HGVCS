"""
WakeWordDetector – Continuously listens for "Hey V" using:
  1. Energy-based VAD to detect speech segments
  2. Whisper tiny.en to transcribe short chunks
  3. Fuzzy matching to catch many "hey v" variants

Fixes in this version:
  - Lower ENERGY_THRESH so quiet voices are captured
  - More wake-word fuzzy patterns (covers Indian accent variations)
  - Whisper params tuned for short wake-word detection
  - ALWAYS_AWAKE mode: after wake, stays listening until user says "sleep" / "goodbye"
  - Post-wake capture no longer cut short by TTS "Yes?" audio artefact
  - manual_trigger() for button-driven listening
  - Persistent sounddevice stream that NEVER closes (camera-equivalent keep-alive)
"""

import logging
import threading
import time
import queue
import re
import numpy as np

log = logging.getLogger("hgvcs.wakeword")

# ── sounddevice ────────────────────────────────────────────
try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False
    log.warning("sounddevice not installed — microphone disabled")

# ── whisper ────────────────────────────────────────────────
try:
    import whisper as _whisper_mod
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False
    log.warning("openai-whisper not installed — STT disabled")

# Audio settings
SAMPLE_RATE    = 16_000
CHANNELS       = 1
CHUNK_FRAMES   = 1600          # 100 ms chunks
ENERGY_THRESH  = 0.012         # Raised back: filters keyboard/fan noise
MIN_SPEECH_RMS = 0.020         # Must hit this level at least once to count as real speech
SILENCE_CHUNKS = 12            # 1.2 s of silence = end of utterance
MAX_RECORD_S   = 10            # Max utterance length (seconds)
MIN_RECORD_CS  = 6             # Minimum chunks before we accept utterance (0.6 s)
POST_WAKE_FLUSH_S = 1.0        # Flush audio after wake to skip any TTS playback
TTS_GUARD_S       = 1.5        # Extra silence guard after TTS speech ends

# Wake word variants – broad fuzzy set covering Indian accent + Hindi variants
WAKE_PATTERNS = [
    # English
    r"\bhey\s+v\b",
    r"\bhey\s+vee\b",
    r"\bhey\s+vi\b",
    r"\bhey\s+bee\b",
    r"\bhi\s+v\b",
    r"\bhi\s+vee\b",
    r"\bhey\s+we\b",
    r"\bhey\s+me\b",
    r"\bhey\s+be\b",
    r"\bhey\s+de\b",
    r"\bhey\s+the\b",
    r"\bh+[ei]+\s+v\b",  # slurred variants
    # Hindi-transliterated wake words
    r"\bsuno\s+v\b",      # "suno v" (listen v)
    r"\baro\b",           # "aro" (sounds like V in some accents)
    r"\bv\s+bot\b",
    r"\bvee\s+bot\b",
]

# Phrases that put V back to idle (sleep)
SLEEP_PATTERNS = [
    r"\bsleep\b", r"\bgoodbye\b", r"\bbye\b",
    r"\bstop listening\b", r"\bgo to sleep\b",
    r"\bshut up\b", r"\bquiet\b", r"\bband kar\b",  # Hindi: shut off
    r"\bsona\b", r"\brup ja\b",
]


def _is_wake(text: str) -> bool:
    t = text.lower().strip()
    for pat in WAKE_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def _is_sleep(text: str) -> bool:
    t = text.lower().strip()
    for pat in SLEEP_PATTERNS:
        if re.search(pat, t):
            return True
    return False


class WakeWordDetector:
    """
    Background listener.  Call start() once; it runs a daemon thread that
    keeps one persistent sounddevice stream open FOREVER (never closes).

    Modes
    -----
    always_awake=True  (default) – V stays active after first wake-up; only
        goes idle if user explicitly says "sleep" / "goodbye".
    always_awake=False – classic one-shot: detect wake word, record, go idle.

    manual_trigger() simulates a wake word — from the UI "Listen" button.
    """

    def __init__(self,
                 on_wake: callable = None,
                 on_speech: callable = None,
                 on_state_change: callable = None,
                 whisper_model: str = "tiny.en",
                 always_awake: bool = True):
        self._on_wake         = on_wake or (lambda: None)
        self._on_speech       = on_speech or (lambda t: None)
        self._on_state_change = on_state_change or (lambda s: None)
        self._model_name      = whisper_model
        self._model           = None
        self._running         = False
        self._state           = "idle"
        self._always_awake    = always_awake
        self._is_awake        = False   # True once user activated V (always_awake mode)

        self._audio_q: queue.Queue = queue.Queue(maxsize=300)
        # event to skip straight to utterance capture (manual trigger)
        self._manual_evt = threading.Event()
        # event: user asked V to sleep (re-enter idle detection)
        self._sleep_evt  = threading.Event()
        # TTS guard: set True while V is speaking so we don't hear ourselves
        self._tts_active = False
        self._tts_end_t  = 0.0   # timestamp when TTS finished

    def notify_tts_start(self):
        """Call this when TTS starts speaking so mic is muted."""
        self._tts_active = True

    def notify_tts_end(self):
        """Call this when TTS finishes speaking — imposes a TTS_GUARD_S silence."""
        self._tts_active = False
        self._tts_end_t  = time.time()

    # ── lifecycle ──────────────────────────────────────────
    def start(self):
        if not _SD_OK or not _WHISPER_OK:
            missing = []
            if not _SD_OK:      missing.append("sounddevice")
            if not _WHISPER_OK: missing.append("openai-whisper")
            log.warning(f"WakeWordDetector disabled — missing: {missing}")
            self._set_state("error")
            return

        self._running = True
        t = threading.Thread(target=self._load_and_listen, daemon=True)
        t.start()
        log.info("WakeWordDetector starting...")

    def stop(self):
        self._running = False
        log.info("WakeWordDetector stopped")

    def manual_trigger(self):
        """Simulate a wake word event — called from the UI 'Listen' button."""
        log.info("Manual trigger activated")
        self._is_awake = True
        self._sleep_evt.clear()
        self._manual_evt.set()

    @property
    def state(self) -> str:
        return self._state

    def set_always_awake(self, value: bool):
        self._always_awake = value

    # ── internal ──────────────────────────────────────────

    def _set_state(self, state: str):
        self._state = state
        try:
            self._on_state_change(state)
        except Exception:
            pass

    def _load_and_listen(self):
        log.info(f"Loading Whisper model: {self._model_name}")
        try:
            self._model = _whisper_mod.load_model(self._model_name)
            log.info("Whisper model loaded — listening for 'Hey V'")
        except Exception as e:
            log.error(f"Whisper load failed: {e}")
            self._set_state("error")
            return

        # ONE persistent stream — never close it while _running
        while self._running:
            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="float32",
                    blocksize=CHUNK_FRAMES,
                    callback=self._audio_callback,
                ):
                    log.info("Audio stream open — listening forever")
                    self._set_state("idle")
                    self._listen_loop()
            except Exception as e:
                log.error(f"Audio stream error: {e} — retrying in 2 s")
                self._set_state("error")
                time.sleep(2)   # retry indefinitely

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.debug(f"Audio status: {status}")
        try:
            self._audio_q.put_nowait(indata.copy().flatten())
        except queue.Full:
            pass   # drop frame if lagging

    def _drain_queue(self, seconds: float):
        max_chunks = int(seconds * SAMPLE_RATE / CHUNK_FRAMES)
        chunks = []
        while not self._audio_q.empty() and len(chunks) < max_chunks:
            try:
                chunks.append(self._audio_q.get_nowait())
            except queue.Empty:
                break
        return chunks

    def _capture_segment(self, max_seconds: float = 5.0,
                         stop_on_silence: bool = True,
                         min_seconds: float = 0.0) -> np.ndarray | None:
        """
        Capture from the persistent queue until silence or timeout.
        Returns numpy float32 array at SAMPLE_RATE, or None if:
          - No chunks captured, or
          - Peak RMS never exceeded MIN_SPEECH_RMS (background noise only)
        Automatically skips frames while TTS is active or within TTS_GUARD_S.
        """
        chunks    = []
        silence_c = 0
        max_c     = int(max_seconds * SAMPLE_RATE / CHUNK_FRAMES)
        speaking  = False
        peak_rms  = 0.0   # track loudest frame to filter pure-noise captures

        for _ in range(max_c):
            if not self._running:
                break
            if self._manual_evt.is_set() and self._state == "idle":
                break

            # Skip frames while TTS is speaking or within guard window
            if self._tts_active or (time.time() - self._tts_end_t) < TTS_GUARD_S:
                try:
                    self._audio_q.get(timeout=0.1)
                except queue.Empty:
                    pass
                continue

            try:
                data = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(data ** 2)))
            peak_rms = max(peak_rms, rms)

            if rms > ENERGY_THRESH:
                speaking  = True
                silence_c = 0
                chunks.append(data)
            else:
                if speaking:
                    chunks.append(data)  # include trailing silence for natural pacing
                    silence_c += 1
                    if stop_on_silence and silence_c >= SILENCE_CHUNKS:
                        break

        if not chunks:
            return None
        # Reject if the loudest frame never reached MIN_SPEECH_RMS
        # (this filters out keyboard clicks, fan noise, breathing)
        if peak_rms < MIN_SPEECH_RMS:
            log.debug(f"Capture rejected: peak_rms={peak_rms:.4f} < MIN_SPEECH_RMS")
            return None
        # Also reject very short captures (less than MIN_RECORD_CS chunks)
        if len(chunks) < MIN_RECORD_CS:
            return None
        return np.concatenate(chunks, axis=0)

    def _listen_loop(self):
        """Main loop: idle wake-detection → awake utterance capture, forever."""
        while self._running:

            # ──────────────────────────────────────────────
            # ALWAYS-AWAKE: skip idle detection if already active
            # ──────────────────────────────────────────────
            if self._always_awake and self._is_awake and not self._sleep_evt.is_set():
                # Go straight to utterance capture without waiting for wake word
                self._handle_wake(skip_ack=True)
                continue

            # ── Phase 1: idle — look for wake word ─────────
            self._drain_queue(0.3)

            if self._manual_evt.is_set():
                self._manual_evt.clear()
                self._handle_wake(skip_ack=False)
                continue

            chunk = self._capture_segment(max_seconds=4, stop_on_silence=True)
            if chunk is None or len(chunk) < SAMPLE_RATE * 0.2:
                if self._manual_evt.is_set():
                    self._manual_evt.clear()
                    self._handle_wake(skip_ack=False)
                continue

            text = self._transcribe_wake(chunk)
            log.debug(f"[Idle STT] {text!r}")

            if _is_wake(text):
                log.info("🎙️  Wake word 'Hey V' detected!")
                self._is_awake = True
                self._sleep_evt.clear()
                self._handle_wake(skip_ack=False)
            elif self._manual_evt.is_set():
                self._manual_evt.clear()
                self._handle_wake(skip_ack=False)

    def _handle_wake(self, skip_ack: bool = False):
        """Called on wake word or manual trigger or always-awake loop."""
        self._set_state("awake")
        try:
            self._on_wake()
        except Exception:
            pass

        # Flush stale audio — wait for any TTS to play + guard time
        # The TTS engine plays async; we wait then drain the queue
        tts_wait = POST_WAKE_FLUSH_S
        time.sleep(tts_wait)
        self._drain_queue(0.3)

        # ── Phase 2: capture utterance ─────────────────────
        self._set_state("recording")
        utterance = self._capture_segment(
            max_seconds=MAX_RECORD_S, stop_on_silence=True,
            min_seconds=0.3
        )

        if utterance is not None and len(utterance) > SAMPLE_RATE * 0.2:
            # Extra guard: if TTS just ended, wait a moment and drain again
            if self._tts_active or (time.time() - self._tts_end_t) < TTS_GUARD_S:
                time.sleep(0.5)
                self._drain_queue(0.5)
                return  # skip this utterance — it was likely TTS echo

            full_text = self._transcribe_command(utterance)
            log.info(f"Utterance: {full_text!r}")

            # Check for sleep command
            if _is_sleep(full_text):
                log.info("😴  Sleep command detected — going idle")
                self._is_awake = False
                self._sleep_evt.set()
                self._set_state("idle")
                try:
                    self._on_speech("__sleep__")
                except Exception:
                    pass
                return

            if full_text:
                try:
                    self._on_speech(full_text)
                except Exception as e:
                    log.error(f"on_speech callback error: {e}")

        # After handling, return to idle momentarily so UI can update
        self._set_state("idle")

    def _transcribe_wake(self, audio: np.ndarray) -> str:
        """Whisper tuned for wake-word detection. Auto-detects language."""
        if self._model is None:
            return ""
        try:
            result = self._model.transcribe(
                audio,
                language=None,             # auto-detect: handles Hindi/English mix
                fp16=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.35,  # less strict for short wake words
                logprob_threshold=-2.5,
                temperature=0.0,
                word_timestamps=False,
            )
            return result.get("text", "").strip()
        except Exception as e:
            log.error(f"Whisper wake transcribe error: {e}")
            return ""

    def _transcribe_command(self, audio: np.ndarray) -> str:
        """Whisper tuned for full command transcription. Auto-detects language."""
        if self._model is None:
            return ""
        try:
            result = self._model.transcribe(
                audio,
                language=None,             # auto-detect Hindi / English / mixed
                fp16=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.5,
                logprob_threshold=-1.5,
                temperature=0.0,
                word_timestamps=False,
            )
            text = result.get("text", "").strip()
            lang = result.get("language", "en")
            if lang not in ("en", "hi"):
                log.debug(f"Detected language: {lang} — transcript: {text!r}")
            return text
        except Exception as e:
            log.error(f"Whisper command transcribe error: {e}")
            return ""


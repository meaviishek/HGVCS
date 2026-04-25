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

# ── whisper backend: try faster-whisper first (no PyTorch/AVX) ───────────
_FASTER_WHISPER = False
_OPENAI_WHISPER = False

# Pre-load torch before ctranslate2 imports it — prevents DLL-state corruption
# from mediapipe's TF probe running first
try:
    import torch as _torch_pre  # noqa: F401
except Exception:
    pass  # if torch itself fails, faster-whisper will also fail — handled below

try:
    from faster_whisper import WhisperModel as _FasterWhisperModel
    _FASTER_WHISPER = True
    log.info("Using faster-whisper backend (ctranslate2, no PyTorch required)")
except ImportError:
    log.debug("faster-whisper not installed")
except Exception as _e:
    log.warning(f"faster-whisper unavailable: {_e}")

if not _FASTER_WHISPER:
    try:
        import whisper as _whisper_mod
        _OPENAI_WHISPER = True
        log.info("Using openai-whisper backend (torch-based)")
    except Exception as _e:
        log.warning(f"openai-whisper unavailable: {_e}")

_WHISPER_OK = _FASTER_WHISPER or _OPENAI_WHISPER


# Audio settings
SAMPLE_RATE    = 16_000
CHANNELS       = 1
CHUNK_FRAMES   = 1600          # 100 ms chunks
ENERGY_THRESH  = 0.018         # Raised: stronger filter for fan/keyboard noise
MIN_SPEECH_RMS = 0.030         # Raised: only accept clearly audible speech
SILENCE_CHUNKS = 10            # 1.0 s of silence = end of utterance
MAX_RECORD_S   = 12            # Max utterance length (seconds)
MIN_RECORD_CS  = 8             # Minimum chunks before we accept utterance (0.8 s)
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
                 whisper_model: str = "tiny",
                 cmd_model: str = "base",
                 always_awake: bool = True):
        self._on_wake         = on_wake or (lambda: None)
        self._on_speech       = on_speech or (lambda t: None)
        self._on_state_change = on_state_change or (lambda s: None)
        # Wake-word model: fast/tiny — just needs to catch "hey v"
        self._model_name      = whisper_model
        # Command model: larger/more accurate — transcribes actual commands
        self._cmd_model_name  = cmd_model
        self._model           = None    # wake model
        self._cmd_model       = None    # command model (loaded lazily)
        self._running         = False
        self._state           = "idle"
        self._always_awake    = always_awake
        self._is_awake        = False

        self._audio_q: queue.Queue = queue.Queue(maxsize=300)
        self._manual_evt = threading.Event()
        self._sleep_evt  = threading.Event()
        self._tts_active = False
        self._tts_end_t  = 0.0

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
        log.info(f"Loading wake model: {self._model_name}")
        try:
            if _FASTER_WHISPER:
                # Wake model: tiny — fast for detecting 'hey v'
                self._model = _FasterWhisperModel(
                    self._model_name,
                    device="cpu",
                    compute_type="int8",
                )
                self._use_faster = True
                log.info(f"Wake model '{self._model_name}' loaded (CPU/int8)")

                # Command model: larger — more accurate for actual commands
                log.info(f"Loading command model: {self._cmd_model_name}")
                try:
                    self._cmd_model = _FasterWhisperModel(
                        self._cmd_model_name,
                        device="cpu",
                        compute_type="int8",
                    )
                    log.info(f"Command model '{self._cmd_model_name}' loaded (CPU/int8)")
                except Exception as ce:
                    log.warning(f"Command model failed, falling back to wake model: {ce}")
                    self._cmd_model = self._model   # fallback
            else:
                self._model = _whisper_mod.load_model(self._model_name)
                self._cmd_model = self._model
                self._use_faster = False
                log.info(f"Whisper model '{self._model_name}' loaded")
        except Exception as e:
            log.error(f"Whisper load failed: {e}")
            self._set_state("error")
            return

        # ONE persistent stream
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
                time.sleep(2)

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

    def _transcribe(self, audio: np.ndarray, is_wake: bool) -> str:
        """Unified transcribe using whichever backend loaded."""
        model = self._model if is_wake else (self._cmd_model or self._model)
        if model is None:
            return ""

        # Initial prompt: primes the model with common command vocabulary
        # Dramatically improves recognition of app names and action words
        COMMAND_PROMPT = (
            "open YouTube, open Chrome, open browser, search for, Google, "
            "screenshot, scroll up, scroll down, volume up, volume down, "
            "next slide, previous slide, close window, minimize, maximize, "
            "mute, play, pause, switch app, lock screen, "
            "upar, neeche, band karo, screenshot lo, volume badhao, "
            "agli slide, pichli slide, browser kholo, dhundho"
        )

        try:
            if getattr(self, "_use_faster", False):
                if is_wake:
                    # Wake: fast, tiny model, beam=1
                    segments, _info = model.transcribe(
                        audio,
                        language=None,
                        beam_size=1,
                        vad_filter=True,
                        vad_parameters={"min_silence_duration_ms": 400},
                        condition_on_previous_text=False,
                    )
                else:
                    # Command: larger model, beam=5, prompt for accuracy
                    segments, _info = model.transcribe(
                        audio,
                        language=None,
                        beam_size=5,
                        best_of=5,
                        initial_prompt=COMMAND_PROMPT,
                        vad_filter=True,
                        vad_parameters={"min_silence_duration_ms": 300},
                        condition_on_previous_text=False,
                        temperature=0.0,
                    )
                text = " ".join(seg.text for seg in segments).strip()
            else:
                # openai-whisper API
                result = model.transcribe(
                    audio,
                    language=None,
                    fp16=False,
                    initial_prompt=None if is_wake else COMMAND_PROMPT,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.35 if is_wake else 0.5,
                    logprob_threshold=-2.5 if is_wake else -1.5,
                    temperature=0.0,
                    word_timestamps=False,
                )
                text = result.get("text", "").strip()

            # Reject suspiciously short transcriptions (< 2 words) that aren't
            # wake-word candidates — these are usually noise artifacts
            if not is_wake and text:
                words = text.split()
                if len(words) < 2 and not any(
                    kw in text.lower() for kw in [
                        "screenshot", "mute", "pause", "play", "save", "copy",
                        "paste", "undo", "redo", "maximize", "minimize",
                        "sleep", "goodbye", "bye"
                    ]
                ):
                    log.debug(f"Rejected short transcription: {text!r}")
                    return ""

            return text

        except Exception as e:
            log.error(f"Whisper transcribe error: {e}")
            return ""

    def _transcribe_wake(self, audio: np.ndarray) -> str:
        """Fast wake-word detection using tiny model."""
        return self._transcribe(audio, is_wake=True)

    def _transcribe_command(self, audio: np.ndarray) -> str:
        """Accurate command transcription using base model."""
        return self._transcribe(audio, is_wake=False)

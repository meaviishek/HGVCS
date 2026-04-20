"""
TTSEngine – Text-to-Speech with Microsoft Edge neural voice (female).

Primary: edge-tts  →  en-US-AriaNeural  (warm, natural female voice)
Fallback: pyttsx3  →  selects first female voice found

Usage:
    engine = TTSEngine()
    engine.speak("Hello, I am V, your voice assistant.")
"""

import asyncio
import logging
import threading
import tempfile
import os

log = logging.getLogger("hgvcs.tts")

# ── Try edge-tts ───────────────────────────────────────────
try:
    import edge_tts
    _EDGE_OK = True
except ImportError:
    _EDGE_OK = False
    log.warning("edge-tts not installed — falling back to pyttsx3")

# ── Try pyttsx3 ────────────────────────────────────────────
try:
    import pyttsx3
    _PYTTSX_OK = True
except ImportError:
    _PYTTSX_OK = False

# ── Try pygame for audio playback ─────────────────────────
try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False

EDGE_VOICE  = "en-US-AriaNeural"     # Natural female, US English
EDGE_RATE   = "+0%"                   # Normal rate
EDGE_PITCH  = "+0Hz"                  # Normal pitch


class TTSEngine:
    """
    Thread-safe TTS engine.
    Calls `speak(text)` from any thread; audio plays asynchronously.
    """

    def __init__(self, voice: str = EDGE_VOICE, rate: str = EDGE_RATE,
                 pitch: str = EDGE_PITCH):
        self._voice   = voice
        self._rate    = rate
        self._pitch   = pitch
        self._lock    = threading.Lock()
        self._busy    = False
        self._pyttsx  = None

        # Init pygame mixer for audio playback
        if _PYGAME_OK:
            try:
                pygame.mixer.init()
                log.info("pygame mixer ready for TTS playback")
            except Exception as e:
                log.warning(f"pygame mixer init failed: {e}")

        # Init pyttsx3 fallback
        if not _EDGE_OK and _PYTTSX_OK:
            try:
                self._pyttsx = pyttsx3.init()
                self._select_female_voice()
                log.info("pyttsx3 TTS engine ready (fallback mode)")
            except Exception as e:
                log.error(f"pyttsx3 init error: {e}")

        engine_name = f"edge-tts ({voice})" if _EDGE_OK else "pyttsx3"
        log.info(f"TTSEngine ready — using {engine_name}")

    def speak(self, text: str, blocking: bool = False):
        """Speak *text*.  Non-blocking by default (runs in daemon thread)."""
        if not text or not text.strip():
            return
        if blocking:
            self._do_speak(text)
        else:
            t = threading.Thread(target=self._do_speak, args=(text,), daemon=True)
            t.start()

    def is_speaking(self) -> bool:
        return self._busy

    # ── internal ──────────────────────────────────────────

    def _do_speak(self, text: str):
        with self._lock:
            self._busy = True
        try:
            if _EDGE_OK:
                self._speak_edge(text)
            elif _PYTTSX_OK and self._pyttsx:
                self._speak_pyttsx(text)
            else:
                log.warning(f"[TTS] No engine available. Would say: {text}")
        except Exception as e:
            log.error(f"TTS error: {e}")
        finally:
            with self._lock:
                self._busy = False

    def _speak_edge(self, text: str):
        """Use edge-tts to generate audio and play via pygame."""
        async def _async_tts():
            communicate = edge_tts.Communicate(
                text, self._voice, rate=self._rate, pitch=self._pitch
            )
            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False, dir=tempfile.gettempdir()
            ) as f:
                tmp_path = f.name

            try:
                await communicate.save(tmp_path)
                if _PYGAME_OK:
                    try:
                        pygame.mixer.music.load(tmp_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            await asyncio.sleep(0.05)
                    except Exception as e:
                        log.error(f"pygame playback error: {e}")
                        self._fallback_play(tmp_path)
                else:
                    self._fallback_play(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_async_tts())
        finally:
            loop.close()

    def _fallback_play(self, path: str):
        """Play mp3 via system command if pygame unavailable."""
        import subprocess
        import sys
        if sys.platform == "win32":
            subprocess.Popen(
                ["powershell", "-c", f"Add-Type -AssemblyName presentationCore; "
                 f"$player=[Windows.Media.Playback.MediaPlayer,Windows.Media,"
                 f"ContentType=WindowsRuntime]::new(); "
                 f"$player.Source=[Windows.Media.Core.MediaSource]::CreateFromUri("
                 f"[uri]::new('{path}')); $player.Play(); Start-Sleep 5"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    def _speak_pyttsx(self, text: str):
        """Fallback: use pyttsx3."""
        try:
            self._pyttsx.say(text)
            self._pyttsx.runAndWait()
        except Exception as e:
            log.error(f"pyttsx3 speak error: {e}")

    def _select_female_voice(self):
        """Select first female voice in pyttsx3."""
        if not self._pyttsx:
            return
        voices = self._pyttsx.getProperty("voices")
        for v in voices:
            if "female" in v.name.lower() or "zira" in v.name.lower() or \
               "hazel" in v.name.lower() or "aria" in v.name.lower():
                self._pyttsx.setProperty("voice", v.id)
                log.info(f"pyttsx3 voice selected: {v.name}")
                return
        # fallback: second voice (often female on Windows)
        if len(voices) > 1:
            self._pyttsx.setProperty("voice", voices[1].id)
        self._pyttsx.setProperty("rate", 160)

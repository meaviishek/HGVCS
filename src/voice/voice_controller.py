"""
VoiceController – Full "Hey V" voice assistant pipeline.

Flow:
  WakeWordDetector ──► (on_wake)   → play chime, update UI state
                  ──► (on_speech)  → OllamaClient.ask()
                                   → SystemController action
                                   → TTSEngine.speak(reply)

Improvements in this version:
  - Handles __sleep__ special command (go idle)
  - Feedback learning: user can say "that's wrong, it's <X>" to teach V
  - Faster: skips Ollama for simple conversational phrases (direct reply)
  - always_awake mode: V never sleeps unless explicitly told
"""

import logging
import threading
import time
import json
import os

log = logging.getLogger("hgvcs.voice")

# ── Feedback / learning store ──────────────────────────────
_FEEDBACK_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "v_learned.json"
)
os.makedirs(os.path.dirname(_FEEDBACK_FILE), exist_ok=True)


def _load_learned() -> dict:
    try:
        if os.path.exists(_FEEDBACK_FILE):
            with open(_FEEDBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_learned(data: dict):
    try:
        with open(_FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Could not save learned data: {e}")


class VoiceController:
    """
    Orchestrates wake-word detection → LLM understanding → TTS response.
    Thread-safe — all deps injected after construction.
    """

    # Map Ollama action names → SystemController execute()-compatible names
    _ACTION_MAP = {
        # Media
        "volume_up":       "three_fingers",
        "volume_down":     "four_fingers",
        "mute":            "__mute__",
        "pause_media":     "open_palm",
        "next_track":      "__next_track__",
        "prev_track":      "__prev_track__",
        # Navigation
        "screenshot":      "peace_sign",
        "scroll_up":       "swipe_up",
        "scroll_down":     "swipe_down",
        "zoom_in":         "pinch_out",
        "zoom_out":        "pinch_in",
        "swipe_left":      "swipe_left",
        "swipe_right":     "swipe_right",
        # Browser
        "open_browser":    "__browser__",
        "search_web":      "__search_web__",
        "close_tab":       "__close_tab__",
        "new_tab":         "__new_tab__",
        "reload":          "circular_cw",
        "go_back":         "__go_back__",
        "go_forward":      "__go_forward__",
        # Window
        "close_window":    "__close_win__",
        "minimize_window": "__minimize__",
        "maximize_window": "__maximize__",
        "show_desktop":    "__show_desktop__",
        "task_switch":     "__task_switch__",
        # System
        "copy":            "__copy__",
        "paste":           "__paste__",
        "undo":            "circular_ccw",
        "redo":            "__redo__",
        "save":            "__save__",
        "lock_screen":     "__lock__",
        "type_text":       "__type__",
        # Presentation
        "ppt_next":        "__ppt_next__",
        "ppt_prev":        "__ppt_prev__",
        "ppt_start":       "__ppt_start__",
        "ppt_end":         "__ppt_end__",
        "ppt_fullscreen":  "__ppt_full__",
        "accept":          "thumbs_up",
        "reject":          "thumbs_down",
        "none":            None,
    }

    # Quick direct-reply patterns (no LLM needed) — English + Hindi
    _DIRECT_REPLIES = {
        # English greetings
        "what is your name": "I'm V, your voice assistant! Aapki seva mein hazir hun.",
        "who are you": "I'm V, your voice assistant! Ready to help.",
        "hello": "Hey! Bolo, kya karna hai? (Say a command!)",
        "hi": "Hi! I'm listening — what do you need?",
        "hey": "Haan bolo! (Yes, go ahead!)",
        "what can you do": (
            "Main volume control, screenshot, browser open karna, slides chalana, "
            "scroll karna aur bahut kuch kar sakti hun! Just boliye."
        ),
        "help": (
            "Aap bol sakte hain: volume badhao, screenshot lo, browser kholo, "
            "agli slide, neeche scroll karo, window band karo, aur bahut kuch!"
        ),
        "thanks": "Koi baat nahi! Aur kuch kaam hai?",
        "thank you": "Any time! Aur kuch help chahiye?",
        "how are you": "Main bilkul theek hun aur ready hun! Kya karna hai?",
        "good morning": "Good morning! Aaj kya kara sakte hain?",
        "good night": "Good night! Agar zaroorat pade toh 'Hey V' boliye.",
        # Hindi greetings
        "namaste": "Namaste! Kya help chahiye aapko?",
        "kya hal hai": "Main theek hun! Aap batao kya karna hai.",
        "shukriya": "Koi baat nahi! Koi aur kaam?",
        "theek hai": "Bahut badhiya! Aur kuch karna hai?",
        "acha": "Theek hai! Koi aur command?",
    }

    def __init__(self, config: dict, event_bus=None):
        self.config    = config
        self.event_bus = event_bus
        self._enabled  = False
        self._sys_ctrl = None

        # Voice subsystems
        self._wake   = None
        self._ollama = None
        self._tts    = None

        # UI state callbacks
        self._on_state_change: callable = None
        self._on_transcript:   callable = None

        # Feedback learning
        self._learned = _load_learned()
        self._last_text   = ""   # last thing user said — for feedback context
        self._last_reply  = ""   # V's last reply

    # ── dependency injection ───────────────────────────────
    def set_system_controller(self, ctrl):
        self._sys_ctrl = ctrl

    def set_state_callback(self, cb):
        self._on_state_change = cb

    def set_transcript_callback(self, cb):
        self._on_transcript = cb

    # ── lifecycle ──────────────────────────────────────────
    def start(self):
        if self._enabled:
            return
        self._enabled = True

        try:
            from src.voice.tts_engine import TTSEngine
            self._tts = TTSEngine()
        except Exception as e:
            log.error(f"TTSEngine init failed: {e}")

        try:
            from src.voice.ollama_client import OllamaClient
            model   = self.config.get("ollama_model",   "llama3.2:3b")
            url     = self.config.get("ollama_url",     "http://localhost:11434/api/generate")
            timeout = self.config.get("ollama_timeout", 20.0)   # Reduced timeout
            use_gpu = self.config.get("ollama_use_gpu", False)
            self._ollama = OllamaClient(
                model=model, url=url, timeout=timeout, use_gpu=use_gpu
            )
        except Exception as e:
            log.error(f"OllamaClient init failed: {e}")

        try:
            from src.voice.wake_word import WakeWordDetector
            whisper_m    = self.config.get("whisper_model", "tiny.en")
            always_awake = self.config.get("always_awake", True)
            self._wake = WakeWordDetector(
                on_wake         = self._on_wake,
                on_speech       = self._on_speech,
                on_state_change = self._on_hw_state,
                whisper_model   = whisper_m,
                always_awake    = always_awake,
            )
            self._wake.start()
        except Exception as e:
            log.error(f"WakeWordDetector init failed: {e}")
            self._pub_state("error")

        log.info("VoiceController started — listening for 'Hey V'")

    def stop(self):
        self._enabled = False
        if self._wake:
            self._wake.stop()
        log.info("VoiceController stopped")

    def manual_listen(self):
        if not self._enabled:
            self.start()
        if self._wake:
            self._wake.manual_trigger()
        else:
            log.warning("manual_listen: WakeWordDetector not ready")

    # ── callbacks from WakeWordDetector ───────────────────

    def _on_hw_state(self, state: str):
        self._pub_state(state)

    def _on_wake(self):
        log.info("Wake word detected — waiting for command…")
        self._pub_state("awake")
        self._pub_event("voice_state", {"state": "awake"})
        # NO TTS ACK — "Yes?" was echoing back into the mic and causing loops

    def _on_speech(self, text: str):
        """Full utterance captured after wake word."""

        # ── Sleep command ────────────────────────────────
        if text == "__sleep__":
            if self._tts:
                self._tts.speak("Okay, going to sleep. Say Hey V to wake me up.", blocking=False)
            self._pub_state("idle")
            return

        log.info(f"Processing: {text!r}")
        self._pub_state("recording")

        def _process():
            action, reply = self._resolve(text)

            log.info(f"Action={action!r}  Reply={reply!r}")

            # Store for feedback context
            self._last_text  = text
            self._last_reply = reply

            # Execute OS action
            if action and action != "none":
                self._execute_action(action, {})

            # Speak reply
            if reply:
                self._pub_state("speaking")
                self._speak_reply(reply)

            self._pub_state("idle")
            self._pub_event("voice_command", {"text": text, "action": action, "reply": reply})
            if self._on_transcript:
                try:
                    self._on_transcript(text, reply)
                except Exception:
                    pass

        threading.Thread(target=_process, daemon=True).start()

    # ── fast-path + LLM resolver ──────────────────────────

    def _resolve(self, text: str) -> tuple[str, str]:
        """
        Returns (action, reply).
        Priority:
          1. Feedback correction command ("that's wrong, it's X")
          2. Learned overrides from user feedback
          3. Direct reply (no LLM) for simple phrases
          4. LLM (Ollama)
        """
        text_lower = text.lower().strip()

        # ── 1. Feedback correction detection ──────────────
        feedback_reply = self._handle_feedback(text_lower)
        if feedback_reply:
            return "none", feedback_reply

        # ── 2. Learned overrides ──────────────────────────
        for pattern, (saved_action, saved_reply) in self._learned.items():
            if pattern in text_lower:
                log.info(f"Learned override for '{pattern}': action={saved_action}")
                return saved_action, saved_reply

        # ── 3. Direct fast-path ───────────────────────────
        for phrase, reply in self._DIRECT_REPLIES.items():
            if phrase in text_lower or text_lower == phrase:
                return "none", reply

        # ── 4. LLM ───────────────────────────────────────
        if not self._ollama:
            return "none", "I'm sorry, my language model isn't available right now."

        result = self._ollama.ask(text)
        return result.get("action", "none"), result.get("reply", "")

    def _handle_feedback(self, text_lower: str) -> str | None:
        """
        Detect user corrections like:
          "that's wrong" / "no that's wrong" / "incorrect" / "wrong answer"
          "the answer is X" / "it should be X" / "correct answer is X"
        Returns a spoken acknowledgement if feedback was recorded, else None.
        """
        is_correction = any(p in text_lower for p in [
            "that's wrong", "thats wrong", "wrong answer",
            "incorrect", "not right", "that is wrong",
            "no that's", "no thats",
        ])
        is_teaching = any(p in text_lower for p in [
            "the answer is", "it should be", "correct is",
            "you should say", "say instead", "actually it's",
            "actually its", "the right answer",
        ])

        if (is_correction or is_teaching) and self._last_text:
            # Extract what the user wants V to say/do
            correction = text_lower
            for prefix in [
                "that's wrong", "thats wrong", "wrong answer", "incorrect",
                "not right", "that is wrong", "no that's", "no thats",
                "the answer is", "it should be", "correct is",
                "you should say", "say instead", "actually it's",
                "actually its", "the right answer is", "the right answer",
            ]:
                correction = correction.replace(prefix, "").strip()
            correction = correction.lstrip(",:- ").strip()

            if correction:
                # Save: pattern = first few words of last utterance → correction reply
                pattern = " ".join(self._last_text.lower().split()[:4])
                self._learned[pattern] = ("none", correction)
                _save_learned(self._learned)
                log.info(f"Learned: '{pattern}' → '{correction}'")
                return f"Got it! I'll remember that. Next time you say that, I'll say: {correction}"
            else:
                return (
                    "I understand I was wrong, but could you tell me what the "
                    "correct answer should be? Say: 'The answer is...' or "
                    "'You should say...'"
                )

        return None

    # ── action execution ───────────────────────────────────

    def _execute_action(self, action: str, params: dict):
        import pyautogui, subprocess, webbrowser, urllib.parse

        gesture_name = self._ACTION_MAP.get(action)
        if gesture_name is None:
            return

        def _open_browser():
            """Open default browser (works without Ctrl+Alt+B which may not exist)."""
            try:
                webbrowser.open("https://www.google.com")
            except Exception:
                pyautogui.hotkey("win", "r")
                import time; time.sleep(0.3)
                pyautogui.typewrite("chrome", interval=0.05)
                pyautogui.press("enter")

        def _search_web():
            query = params.get("query", "")
            if query:
                url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                try:
                    webbrowser.open(url)
                except Exception:
                    pass

        def _type_text():
            text = params.get("text", "")
            if text:
                import time; time.sleep(0.2)
                pyautogui.typewrite(text, interval=0.05)

        special = {
            # Media
            "__mute__":        lambda: pyautogui.press("volumemute"),
            "__next_track__":  lambda: pyautogui.press("nexttrack"),
            "__prev_track__":  lambda: pyautogui.press("prevtrack"),
            # Browser
            "__browser__":     _open_browser,
            "__search_web__":  _search_web,
            "__close_tab__":   lambda: pyautogui.hotkey("ctrl", "w"),
            "__new_tab__":     lambda: pyautogui.hotkey("ctrl", "t"),
            "__go_back__":     lambda: pyautogui.hotkey("alt", "left"),
            "__go_forward__":  lambda: pyautogui.hotkey("alt", "right"),
            # Window
            "__show_desktop__":lambda: pyautogui.hotkey("win", "d"),
            "__close_win__":   lambda: pyautogui.hotkey("alt", "f4"),
            "__minimize__":    lambda: pyautogui.hotkey("win", "down"),
            "__maximize__":    lambda: pyautogui.hotkey("win", "up"),
            "__task_switch__": lambda: pyautogui.hotkey("alt", "tab"),
            # System
            "__copy__":        lambda: pyautogui.hotkey("ctrl", "c"),
            "__paste__":       lambda: pyautogui.hotkey("ctrl", "v"),
            "__redo__":        lambda: pyautogui.hotkey("ctrl", "y"),
            "__save__":        lambda: pyautogui.hotkey("ctrl", "s"),
            "__lock__":        lambda: pyautogui.hotkey("win", "l"),
            "__type__":        _type_text,
            # Presentation (PowerPoint / LibreOffice Impress)
            "__ppt_next__":    lambda: pyautogui.press("right"),
            "__ppt_prev__":    lambda: pyautogui.press("left"),
            "__ppt_start__":   lambda: pyautogui.press("f5"),
            "__ppt_end__":     lambda: (pyautogui.press("escape"), pyautogui.hotkey("ctrl", "end")),
            "__ppt_full__":    lambda: pyautogui.press("f5"),
        }

        sp = special.get(gesture_name)
        if sp:
            try:
                sp()
            except Exception as e:
                log.error(f"Special action {gesture_name} failed: {e}")
            return

        if self._sys_ctrl:
            try:
                self._sys_ctrl.execute(gesture_name, confidence=1.0)
            except Exception as e:
                log.error(f"SystemController.execute({gesture_name}) failed: {e}")

    # ── helpers ───────────────────────────────────────────

    def _speak_reply(self, text: str):
        if self._tts:
            # Notify wake detector that TTS is about to start
            if self._wake:
                self._wake.notify_tts_start()
            self._tts.speak(text, blocking=True)  # blocking so notify_tts_end fires after
            if self._wake:
                self._wake.notify_tts_end()

    def _pub_state(self, state: str):
        if self._on_state_change:
            try:
                self._on_state_change(state)
            except Exception:
                pass

    def _pub_event(self, event: str, data: dict):
        if self.event_bus:
            try:
                self.event_bus.publish(event, data)
            except Exception:
                pass

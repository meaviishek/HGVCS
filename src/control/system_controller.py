"""
SystemController – maps gesture names to OS-level actions.

Called by GestureController after each confirmed gesture.
All actions are non-blocking and wrapped in try/except so
a failed OS action never crashes the gesture pipeline.

Modes:
  normal       – default OS control
  game         – gestures map to gaming key bindings
  presentation – gestures control slideshow
"""

import os
import time
import logging
import threading
from datetime import datetime
from typing import Optional, Callable, Dict

import pyautogui

log = logging.getLogger("hgvcs.system")

# ── optional imports ───────────────────────────────────────
try:
    from pynput.keyboard import Controller as KBD, Key
    _KBD = KBD()
    _Key = Key
    _PYNPUT_OK = True
except Exception:
    _PYNPUT_OK = False
    log.warning("pynput not available – some hotkeys may not work")

# screenshots go to Desktop\HGVCS-screenshots\
_SCREENSHOT_DIR = os.path.join(
    os.path.expanduser("~"), "Desktop", "HGVCS-screenshots"
)

# ── game controller key bindings ───────────────────────────
# Maps gesture → key press (fully remappable)
_GAME_BINDINGS: Dict[str, str] = {
    "open_palm":     "space",      # Jump
    "closed_fist":   "ctrl",       # Crouch / sneak
    "thumbs_up":     "e",          # Interact / use
    "thumbs_down":   "q",          # Drop item
    "pointing":      "f",          # Aim / fire
    "peace_sign":    "r",          # Reload
    "rock_on":       "g",          # Grenade
    "ok_sign":       "tab",        # Inventory
    "swipe_left":    "a",          # Strafe left
    "swipe_right":   "d",          # Strafe right
    "swipe_up":      "w",          # Move forward
    "swipe_down":    "s",          # Move back
    "three_fingers": "1",          # Weapon slot 1
    "four_fingers":  "2",          # Weapon slot 2
    "circular_cw":   "3",          # Weapon slot 3
    "wave":          "escape",     # Pause / menu
    "pinch_out":     "z",          # Zoom in
    "pinch_in":      "x",          # Zoom out
    "phone_sign":    "m",          # Map
}

# ── presentation mode bindings ─────────────────────────────
_PRESENTATION_BINDINGS: Dict[str, str] = {
    "swipe_right":   "right",      # Next slide (pointing + right snap)
    "swipe_left":    "left",       # Previous slide (pointing + left snap)
    "swipe_up":      "prior",      # Scroll up (peace_sign + up)
    "swipe_down":    "next",       # Scroll down (peace_sign + down)
    "open_palm":     "b",          # Black screen
    "thumbs_up":     "f5",         # Start presentation
    "thumbs_down":   "escape",     # End presentation
    "wave":          "escape",     # Exit
    "three_fingers": "f",          # Toggle fullscreen
    "pinch_out":     "=",           # Zoom in slide
    "pinch_in":      "-",          # Zoom out slide
}

MODE_NORMAL       = "normal"
MODE_GAME         = "game"
MODE_PRESENTATION = "presentation"


class SystemController:
    """
    Dispatches gesture → OS action.

    Thread-safe – all heavy ops run in daemon threads so
    they don't block the camera/gesture loop.
    """

    def __init__(self):
        os.makedirs(_SCREENSHOT_DIR, exist_ok=True)
        self._last_action_t: float = 0.0
        self._cooldown:      float = 0.5
        self._last_gesture:  str   = ""
        self._on_settings_cb:  Optional[Callable] = None
        self._on_macro_cb:     Optional[Callable] = None
        self._mode: str = MODE_NORMAL
        self._game_bindings: Dict[str, str] = dict(_GAME_BINDINGS)
        self._pres_bindings: Dict[str, str] = dict(_PRESENTATION_BINDINGS)
        self._cross_minimized: bool = False   # toggle: first cross=minimize, second=maximize
        log.info("SystemController ready")

    # ── mode control ──────────────────────────────────────
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        """Switch between 'normal', 'game', 'presentation'."""
        if mode not in (MODE_NORMAL, MODE_GAME, MODE_PRESENTATION):
            log.warning(f"Unknown mode: {mode}")
            return
        self._mode = mode
        log.info(f"SystemController mode → {mode}")

    def set_game_binding(self, gesture: str, key: str):
        """Remap a gesture in game controller mode."""
        self._game_bindings[gesture] = key

    def set_macro_callback(self, cb: Callable):
        self._on_macro_cb = cb

    # ── callbacks so UI can hook in ───────────────────────
    def set_on_settings(self, cb: Callable):
        """Register callback to open the Settings tab in the UI."""
        self._on_settings_cb = cb


    # ── main dispatch ──────────────────────────────────────
    def execute(self, gesture: str, confidence: float = 1.0) -> bool:
        """
        Execute the OS action for *gesture*.
        Behaviour depends on current mode:
          normal       → OS actions (default)
          game         → game key bindings
          presentation → slide control keys
        Returns True if an action was taken.
        """
        now = time.time()
        if now - self._last_action_t < self._cooldown:
            return False
        if gesture == self._last_gesture and (now - self._last_action_t) < 2.0:
            return False

        # ── game / presentation modes ──────────────────────
        if self._mode == MODE_GAME:
            key = self._game_bindings.get(gesture)
            if key:
                self._last_action_t = now
                self._last_gesture  = gesture
                log.info(f"[GAME] {gesture} → key:{key}")
                threading.Thread(
                    target=lambda: pyautogui.press(key), daemon=True
                ).start()
                return True
            return False

        if self._mode == MODE_PRESENTATION:
            key = self._pres_bindings.get(gesture)
            if key:
                self._last_action_t = now
                self._last_gesture  = gesture
                log.info(f"[PRESENTATION] {gesture} → key:{key}")
                threading.Thread(
                    target=lambda: pyautogui.press(key), daemon=True
                ).start()
                return True
            return False

        # ── normal mode ────────────────────────────────────
        handler = self._DISPATCH.get(gesture)
        if handler is None:
            return False

        self._last_action_t = now
        self._last_gesture  = gesture
        threading.Thread(target=self._run, args=(handler,), daemon=True).start()
        log.info(f"Gesture action: {gesture} (conf={confidence:.2f})")
        return True

    def execute_two_hand(self, left_gesture: str, right_gesture: str = "") -> bool:
        """
        Handle combined two-hand gestures.
        Called from HandEngine when both hands are detected.
        If right_gesture is empty, left_gesture contains a pre-computed combo name.
        """
        if not right_gesture:
            combo = left_gesture
        else:
            combo = f"{left_gesture}+{right_gesture}"
        
        log.info(f"Two-hand combo: {combo}")
        now = time.time()
        if now - self._last_action_t < self._cooldown:
            return False
        self._last_action_t = now

        # two-hand combo table
        combos = {
            # both_hands_spread (both open palms) → show desktop
            "open_palm+open_palm":       lambda: pyautogui.hotkey("win", "d"),
            # left fist + right thumbs up → confirm and send
            "closed_fist+thumbs_up":     lambda: pyautogui.hotkey("ctrl", "s"),
            # both pointing → zoom to fit / reset zoom
            "pointing+pointing":         lambda: pyautogui.hotkey("ctrl", "0"),
            # left swipe + right pointing → switch + click
            "thumbs_up+thumbs_down":     lambda: pyautogui.hotkey("win", "l"),
            # rock on both → toggle dark mode (Win+period opens emoji / toggle)
            "rock_on+rock_on":           lambda: pyautogui.hotkey("win", "."),
            # left peace + right peace → multi-screenshot (Win+Shift+S)
            "peace_sign+peace_sign":     lambda: pyautogui.hotkey("win", "shift", "s"),
            # left open palm + right pointing → copy (Ctrl+C)
            "open_palm+pointing":        lambda: pyautogui.hotkey("ctrl", "c"),
            # left fist + right pointing → paste (Ctrl+V)
            "closed_fist+pointing":      lambda: pyautogui.hotkey("ctrl", "v"),
            # two hands point → reset zoom
            "pointing+pointing":         lambda: pyautogui.hotkey("ctrl", "0"),
            "thumbs_up+thumbs_down":     lambda: pyautogui.hotkey("win", "l"),
            "rock_on+rock_on":           lambda: pyautogui.hotkey("win", "."),
            "peace_sign+peace_sign":     lambda: pyautogui.hotkey("win", "shift", "s"),
            "open_palm+pointing":        lambda: pyautogui.hotkey("ctrl", "c"),
            "closed_fist+pointing":      lambda: pyautogui.hotkey("ctrl", "v"),
            "thumbs_up+peace_sign":      lambda: pyautogui.hotkey("alt", "tab"),
            "wave+wave":                 lambda: None,
            # ── semantic two-hand gestures ──────────────────────────
            "two_hand_zoom_in":          lambda: pyautogui.hotkey("ctrl", "="),
            "two_hand_zoom_out":         lambda: pyautogui.hotkey("ctrl", "-"),
            # Cross (X) gesture — toggles minimize ↔ maximize
            "two_hand_cross":            self._do_cross,
        }
        action = combos.get(combo)
        if action:
            threading.Thread(target=action, daemon=True).start()
            return True
        return False

    def _run(self, fn):
        try:
            fn(self)
        except Exception as e:
            log.error(f"SystemController action error: {e}")

    # ── action implementations ─────────────────────────────

    def _pause(self):
        """Toggle media play/pause."""
        pyautogui.press("playpause")

    def _confirm(self):
        """Press Enter (confirm dialogs)."""
        pyautogui.press("enter")

    def _accept(self):
        """Thumbs-up → Enter / Yes."""
        pyautogui.press("enter")

    def _reject(self):
        """Thumbs-down → Escape / No."""
        pyautogui.press("escape")

    def _screenshot(self):
        """OK-sign → save full-screen screenshot.
        (peace_sign is now used exclusively for scroll up/down)
        """
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(_SCREENSHOT_DIR, f"hgvcs_{ts}.png")
        img   = pyautogui.screenshot()
        img.save(fname)
        log.info(f"Screenshot saved → {fname}")

    def _toggle_media(self):
        """Rock-on → play/pause media."""
        pyautogui.press("playpause")

    def _open_settings(self):
        """Phone-sign → open Settings tab in UI if callback registered.
        (ok_sign is now screenshot; phone_sign takes the settings role)
        """
        if self._on_settings_cb:
            self._on_settings_cb()

    def _swipe_left(self):
        """
        Swipe Left → Previous slide (PPT) / previous image / browser Alt+Left.
        Arrow key works in: PowerPoint, image viewers, PDF viewers, browsers.
        """
        pyautogui.press("left")

    def _swipe_right(self):
        """
        Swipe Right → Next slide (PPT) / next image / browser Alt+Right.
        Arrow key works in: PowerPoint, image viewers, PDF viewers, browsers.
        """
        pyautogui.press("right")

    def _swipe_up(self):
        """Scroll up — 3 wheel clicks up."""
        pyautogui.scroll(8)

    def _swipe_down(self):
        """Scroll down — 3 wheel clicks down."""
        pyautogui.scroll(-8)

    def _refresh(self):
        """Circular CW → Ctrl+R."""
        pyautogui.hotkey("ctrl", "r")

    def _undo(self):
        """Circular CCW → Ctrl+Z."""
        pyautogui.hotkey("ctrl", "z")

    def _zoom_in(self):
        """Pinch out → Ctrl+=."""
        pyautogui.hotkey("ctrl", "=")

    def _zoom_out(self):
        """Pinch in → Ctrl+-."""
        pyautogui.hotkey("ctrl", "-")

    def _volume_up(self):
        """Three fingers → media volume up × 3."""
        for _ in range(3):
            pyautogui.press("volumeup")
            time.sleep(0.05)

    def _volume_down(self):
        """Four fingers → media volume down × 3."""
        for _ in range(3):
            pyautogui.press("volumedown")
            time.sleep(0.05)


    def _wave(self):
        """Wave → press Escape (cancel)."""
        pyautogui.press("escape")

    def _phone_sign(self):
        """Phone sign → open communication (Win+C or fallback)."""
        try:
            pyautogui.hotkey("win", "c")
        except Exception:
            pass

    def _do_cross(self):
        """
        Two-hand X (cross) → toggle minimize ↔ maximize.
        First cross: minimize current window (Win+Down).
        Second cross: maximize it back   (Win+Up).
        """
        if not self._cross_minimized:
            pyautogui.hotkey("win", "down")   # minimize
            self._cross_minimized = True
            log.info("Cross gesture → Minimize")
        else:
            pyautogui.hotkey("win", "up")     # maximize / restore
            self._cross_minimized = False
            log.info("Cross gesture → Maximize / Restore")

    # ── dispatch table ────────────────────────────────────
    # NOTE: peace_sign removed — it is exclusively the scroll trigger shape
    # (hold-exempt, fires _detect_swipe before any static action could run).
    # ok_sign → screenshot  |  phone_sign → settings
    _DISPATCH = {
        "open_palm":     _pause,
        "closed_fist":   _confirm,
        "thumbs_up":     _accept,
        "thumbs_down":   _reject,
        "ok_sign":       _screenshot,    # ✌ screenshot (was peace_sign)
        "phone_sign":    _open_settings, # 🤙 settings  (was ok_sign)
        "rock_on":       _toggle_media,
        "swipe_left":    _swipe_left,
        "swipe_right":   _swipe_right,
        "swipe_up":      _swipe_up,
        "swipe_down":    _swipe_down,
        "circular_cw":   _refresh,
        "circular_ccw":  _undo,
        "pinch_out":     _zoom_in,
        "pinch_in":      _zoom_out,
        "three_fingers": _volume_up,
        "four_fingers":  _volume_down,
        "wave":          _wave,
    }

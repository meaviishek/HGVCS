"""
MacroEngine – Gesture Sequence Macros for HGVCS.

Detects sequences of gestures and fires a named macro action.

Example macros (fully configurable):
  thumbs_up → peace_sign → open_palm   → "launch_browser"
  closed_fist × 2                       → "select_all"
  swipe_right × 3 (rapid)              → "media_next_fast"
  thumbs_up + thumbs_down              → "lock_screen"

Usage:
    engine = MacroEngine()
    # in your gesture handler:
    macro = engine.feed("thumbs_up")
    if macro:
        print(f"Macro fired: {macro}")
"""

import time
import json
import os
import logging
from collections import deque
from typing import Optional, List, Dict, Any

log = logging.getLogger("hgvcs.macros")

# ── default macro definitions ──────────────────────────────
_DEFAULT_MACROS: List[Dict] = [
    # Format:
    # { "name": macro_id, "sequence": [gesture, ...],
    #   "max_gap": secs_between_each, "description": str }

    # Double tap gestures
    {
        "name": "double_fist",
        "sequence": ["closed_fist", "closed_fist"],
        "max_gap": 0.9,
        "description": "Double fist → Select All (Ctrl+A)",
    },
    {
        "name": "double_thumbs_up",
        "sequence": ["thumbs_up", "thumbs_up"],
        "max_gap": 0.9,
        "description": "Double thumbs up → Maximize window",
    },
    {
        "name": "double_peace",
        "sequence": ["peace_sign", "peace_sign"],
        "max_gap": 0.9,
        "description": "DISABLED — peace_sign is exclusively the scroll gesture",
        "disabled": True,
    },

    # 3-gesture sequences
    {
        "name": "browser_launch",
        "sequence": ["thumbs_up", "peace_sign", "open_palm"],
        "max_gap": 1.2,
        "description": "Thumbs Up + Peace + Palm → Open browser",
    },
    {
        "name": "file_manager",
        "sequence": ["thumbs_up", "rock_on", "open_palm"],
        "max_gap": 1.2,
        "description": "Thumbs Up + Rock On + Palm → Open file manager",
    },
    {
        "name": "lock_screen",
        "sequence": ["thumbs_up", "thumbs_down"],
        "max_gap": 0.9,
        "description": "Thumbs Up → Thumbs Down → Lock screen",
    },
    {
        "name": "show_desktop",
        "sequence": ["open_palm", "wave"],
        "max_gap": 1.0,
        "description": "Open Palm → Wave → Show desktop",
    },
    {
        "name": "task_manager",
        "sequence": ["closed_fist", "ok_sign", "open_palm"],
        "max_gap": 1.2,
        "description": "Fist + OK + Palm → Open Task Manager",
    },
    {
        "name": "media_next_fast",
        "sequence": ["swipe_right", "swipe_right", "swipe_right"],
        "max_gap": 0.8,
        "description": "Swipe Right × 3 → Skip 3 tracks",
    },
    {
        "name": "media_prev_fast",
        "sequence": ["swipe_left", "swipe_left", "swipe_left"],
        "max_gap": 0.8,
        "description": "Swipe Left × 3 → Back 3 tracks",
    },
    {
        "name": "volume_max",
        "sequence": ["three_fingers", "three_fingers", "three_fingers"],
        "max_gap": 0.7,
        "description": "Three Fingers × 3 → Volume to max",
    },
    {
        "name": "mute_toggle",
        "sequence": ["four_fingers", "four_fingers"],
        "max_gap": 0.7,
        "description": "Four Fingers × 2 → Mute/unmute",
    },
    {
        "name": "save_file",
        "sequence": ["ok_sign", "closed_fist"],
        "max_gap": 1.0,
        "description": "OK + Fist → Save (Ctrl+S)",
    },
    {
        "name": "new_tab",
        "sequence": ["peace_sign", "pointing"],
        "max_gap": 1.0,
        "description": "Peace + Pointing → New Tab (Ctrl+T)",
    },
    {
        "name": "close_tab",
        "sequence": ["thumbs_down", "wave"],
        "max_gap": 0.9,
        "description": "Thumbs Down + Wave → Close Tab (Ctrl+W)",
    },
    {
        "name": "undo_redo",
        "sequence": ["circular_ccw", "circular_cw"],
        "max_gap": 1.2,
        "description": "CCW + CW → Undo then Redo",
    },
]

# ── macro OS action mapping ────────────────────────────────
# Maps macro name → callable
import pyautogui
import subprocess
import sys

def _open_browser():
    try:
        import webbrowser
        webbrowser.open("https://www.google.com")
    except Exception:
        pass

def _open_file_manager():
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer.exe"])
        else:
            subprocess.Popen(["nautilus"])
    except Exception:
        pass

def _lock_screen():
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.LockWorkStation()
        else:
            subprocess.Popen(["loginctl", "lock-session"])
    except Exception:
        pass

def _open_task_manager():
    try:
        if sys.platform == "win32":
            subprocess.Popen(["taskmgr.exe"])
    except Exception:
        pass

_MACRO_ACTIONS = {
    "double_fist":        lambda: pyautogui.hotkey("ctrl", "a"),
    "double_thumbs_up":   lambda: pyautogui.hotkey("win", "up"),
    "double_peace":       lambda: None,   # DISABLED — peace_sign = scroll only
    "browser_launch":     _open_browser,
    "file_manager":       _open_file_manager,
    "lock_screen":        _lock_screen,
    "show_desktop":       lambda: pyautogui.hotkey("win", "d"),
    "task_manager":       _open_task_manager,
    "media_next_fast":    lambda: [pyautogui.press("nexttrack") for _ in range(3)],
    "media_prev_fast":    lambda: [pyautogui.press("prevtrack") for _ in range(3)],
    "volume_max":         lambda: [pyautogui.press("volumeup") for _ in range(20)],
    "mute_toggle":        lambda: pyautogui.press("volumemute"),
    "save_file":          lambda: pyautogui.hotkey("ctrl", "s"),
    "new_tab":            lambda: pyautogui.hotkey("ctrl", "t"),
    "close_tab":          lambda: pyautogui.hotkey("ctrl", "w"),
    "undo_redo":          lambda: (pyautogui.hotkey("ctrl","z"), time.sleep(0.1),
                                   pyautogui.hotkey("ctrl","y")),
}


class MacroEngine:
    """
    Processes a stream of gesture names and fires macros when
    a configured sequence is matched.

    Thread-safe: call feed() from any thread.
    """

    MACRO_FILE = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "macros.json"
    )

    def __init__(self, macros: Optional[List[Dict]] = None):
        self._macros  = self._load(macros)
        self._history: deque = deque(maxlen=8)   # (gesture, timestamp)
        self._last_macro_t: float = 0.0
        self._macro_cooldown: float = 2.5        # global secs between ANY macro fires
        self._per_macro_t: Dict[str, float] = {} # per-macro last fired time
        self._per_macro_cd: float = 3.0          # per-macro cooldown (prevents spam)
        log.info(f"MacroEngine ready — {len(self._macros)} macros loaded")

    # ── public ────────────────────────────────────────────
    def feed(self, gesture: str) -> Optional[str]:
        """
        Feed a recognised gesture.
        Returns the macro name if a sequence just completed, else None.
        """
        if gesture in ("none", "unknown", ""):
            return None

        now = time.time()
        self._history.append((gesture, now))
        self._prune(now)

        matched = self._match(now)
        if matched:
            self._fire(matched)
            self._history.clear()
            return matched
        return None

    def all_macros(self) -> List[Dict]:
        return list(self._macros)

    def add_macro(self, name: str, sequence: List[str],
                  max_gap: float = 1.0, description: str = ""):
        self._macros.append({
            "name": name, "sequence": sequence,
            "max_gap": max_gap, "description": description,
        })
        self._save()
        log.info(f"Macro added: {name} → {sequence}")

    def remove_macro(self, name: str):
        self._macros = [m for m in self._macros if m["name"] != name]
        self._save()

    # ── internal ──────────────────────────────────────────
    def _load(self, override) -> List[Dict]:
        if override is not None:
            return list(override)
        if os.path.exists(self.MACRO_FILE):
            try:
                with open(self.MACRO_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return list(_DEFAULT_MACROS)

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.MACRO_FILE), exist_ok=True)
            with open(self.MACRO_FILE, "w") as f:
                json.dump(self._macros, f, indent=2)
        except Exception as e:
            log.warning(f"Could not save macros: {e}")

    def _prune(self, now: float):
        """Remove entries older than the longest possible gap chain."""
        MAX_WINDOW = 6.0
        while self._history and now - self._history[0][1] > MAX_WINDOW:
            self._history.popleft()

    def _match(self, now: float) -> Optional[str]:
        # Global cooldown: no macro within 2.5s of ANY macro
        if now - self._last_macro_t < self._macro_cooldown:
            return None

        history_gestures = [g for g, _ in self._history]
        history_times    = [t for _, t in self._history]

        for macro in self._macros:
            # Skip disabled macros
            if macro.get("disabled", False):
                continue

            seq     = macro["sequence"]
            max_gap = macro.get("max_gap", 1.0)
            n       = len(seq)
            name    = macro["name"]

            if len(history_gestures) < n:
                continue

            # check the last n gestures
            tail_g = history_gestures[-n:]
            tail_t = history_times[-n:]

            if tail_g != seq:
                continue

            # check inter-gesture gaps
            valid = all(
                tail_t[k+1] - tail_t[k] <= max_gap
                for k in range(n - 1)
            )
            if not valid:
                continue

            # Per-macro cooldown: this specific macro can't fire again within 3s
            last_this = self._per_macro_t.get(name, 0.0)
            if now - last_this < self._per_macro_cd:
                continue

            self._last_macro_t = now
            self._per_macro_t[name] = now
            return name

        return None

    def _fire(self, macro_name: str):
        import threading
        action = _MACRO_ACTIONS.get(macro_name)
        if action:
            log.info(f"Macro fired: {macro_name}")
            threading.Thread(target=self._safe_call, args=(action,),
                             daemon=True).start()
        else:
            log.warning(f"No action for macro: {macro_name}")

    @staticmethod
    def _safe_call(fn):
        try:
            fn()
        except Exception as e:
            log.error(f"Macro action error: {e}")

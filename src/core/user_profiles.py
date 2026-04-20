"""
UserProfiles – per-user calibration and preference storage for HGVCS.

Each profile stores:
  - Preferred gesture → action overrides
  - Personal confidence thresholds
  - UI preferences
  - Gesture history summary (used by analytics)
  - Custom macros

Profiles are stored as JSON in data/users/{username}.json
"""

import json
import os
import time
import logging
from typing import Dict, Any, Optional, List

log = logging.getLogger("hgvcs.profiles")

_PROFILES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "users"
)
_ACTIVE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "active_profile.txt"
)

_DEFAULT_PROFILE = {
    "version":  1,
    "name":     "Default",
    "created":  None,
    "updated":  None,

    # per-gesture confidence overrides (None = use system default)
    "confidence_overrides": {},

    # gesture → action remapping (None = use system default)
    "action_overrides": {},

    # personal macro sequences
    "custom_macros": [],

    # calibration data (wrist-normalised landmark means per gesture)
    "calibration": {},

    # analytics totals
    "stats": {
        "total_gestures":  0,
        "gesture_counts":  {},   # name → count
        "session_count":   0,
        "total_runtime_s": 0,
    },

    # UI preferences
    "prefs": {
        "theme":          "dark",
        "show_hud":       True,
        "cursor_enabled": True,
        "hud_opacity":    0.8,
    },
}


class UserProfile:
    """Represents a single user profile."""

    def __init__(self, data: Dict, path: str):
        self._data = data
        self._path = path

    # ── properties ────────────────────────────────────────
    @property
    def name(self) -> str:
        return self._data.get("name", "Default")

    @property
    def stats(self) -> Dict:
        return self._data.get("stats", {})

    @property
    def prefs(self) -> Dict:
        return self._data.get("prefs", {})

    @property
    def custom_macros(self) -> List[Dict]:
        return self._data.get("custom_macros", [])

    @property
    def calibration(self) -> Dict:
        return self._data.get("calibration", {})

    # ── analytics recording ───────────────────────────────
    def record_gesture(self, gesture_name: str):
        stats = self._data["stats"]
        stats["total_gestures"] += 1
        counts = stats.setdefault("gesture_counts", {})
        counts[gesture_name] = counts.get(gesture_name, 0) + 1

    def record_session(self, duration_s: float):
        stats = self._data["stats"]
        stats["session_count"]   += 1
        stats["total_runtime_s"] += duration_s

    # ── calibration ───────────────────────────────────────
    def add_calibration_sample(self, gesture: str, landmarks_flat: List[float]):
        """
        Store wrist-normalised landmark flat list as calibration data.
        Computes a running mean over up to 50 samples per gesture.
        """
        cal = self._data.setdefault("calibration", {})
        entry = cal.setdefault(gesture, {"mean": None, "n": 0})
        import numpy as np
        new = np.array(landmarks_flat, dtype=float)
        if entry["mean"] is None:
            entry["mean"] = new.tolist()
            entry["n"] = 1
        else:
            n   = entry["n"]
            old = np.array(entry["mean"])
            # incremental mean, cap at 50
            n_new = min(n + 1, 50)
            updated = (old * min(n, 49) + new) / n_new
            entry["mean"] = updated.tolist()
            entry["n"]    = n_new

    # ── preferences ───────────────────────────────────────
    def set_pref(self, key: str, value: Any):
        self._data["prefs"][key] = value

    def get_pref(self, key: str, default=None):
        return self._data["prefs"].get(key, default)

    # ── persistence ───────────────────────────────────────
    def save(self):
        self._data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)
        log.debug(f"Profile saved: {self._path}")

    def to_dict(self) -> Dict:
        return dict(self._data)


class ProfileManager:
    """
    Manages multiple user profiles.
    Always has an 'active' profile.
    """

    def __init__(self):
        os.makedirs(_PROFILES_DIR, exist_ok=True)
        self._active: Optional[UserProfile] = None
        self._session_start = time.time()
        self._load_active()
        log.info(f"ProfileManager: active profile = '{self._active.name}'")

    # ── active profile ────────────────────────────────────
    def active(self) -> UserProfile:
        return self._active

    def switch_to(self, name: str) -> UserProfile:
        profile = self.load(name) or self.create(name)
        self._active = profile
        self._save_active_marker(name)
        log.info(f"Switched to profile: {name}")
        return profile

    # ── CRUD ──────────────────────────────────────────────
    def create(self, name: str) -> UserProfile:
        import copy
        data = copy.deepcopy(_DEFAULT_PROFILE)
        data["name"]    = name
        data["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        path    = self._path_for(name)
        profile = UserProfile(data, path)
        profile.save()
        log.info(f"Created profile: {name}")
        return profile

    def load(self, name: str) -> Optional[UserProfile]:
        path = self._path_for(name)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return UserProfile(data, path)
        except Exception as e:
            log.error(f"Failed to load profile {name}: {e}")
            return None

    def list_profiles(self) -> List[str]:
        names = []
        for fn in os.listdir(_PROFILES_DIR):
            if fn.endswith(".json"):
                names.append(fn[:-5])
        return sorted(names)

    def delete(self, name: str):
        path = self._path_for(name)
        if os.path.exists(path):
            os.remove(path)
            log.info(f"Deleted profile: {name}")

    # ── session recording ─────────────────────────────────
    def end_session(self):
        """Call on app close to record session duration."""
        if self._active:
            duration = time.time() - self._session_start
            self._active.record_session(duration)
            self._active.save()

    def record_gesture(self, gesture: str):
        """Convenience: record a gesture on the active profile."""
        if self._active:
            self._active.record_gesture(gesture)

    # ── internal ──────────────────────────────────────────
    def _path_for(self, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
        return os.path.join(_PROFILES_DIR, f"{safe}.json")

    def _load_active(self):
        active_name = "Default"
        if os.path.exists(_ACTIVE_FILE):
            try:
                with open(_ACTIVE_FILE) as f:
                    active_name = f.read().strip() or "Default"
            except Exception:
                pass
        profile = self.load(active_name)
        if profile is None:
            profile = self.create(active_name)
        self._active = profile

    def _save_active_marker(self, name: str):
        try:
            with open(_ACTIVE_FILE, "w") as f:
                f.write(name)
        except Exception:
            pass

"""
GestureController - bridge between HandEngine gesture results and SystemController.

Runs in the same Qt thread as the camera widget because it just receives
already-processed GestureResult objects via the EventBus / direct connection.

New in this version:
  - MacroEngine integration - detects multi-gesture sequences
  - ProfileManager integration - records gesture history per user
  - Two-hand gesture dispatch to SystemController.execute_two_hand()
  - Mode switching (normal / game / presentation) via set_mode()
  - Hold-guard awareness: only fires OS actions when gesture is confirmed
  - Focused daily-life gesture set for high accuracy
"""

import time
import logging
from typing import Optional

log = logging.getLogger("hgvcs.gesture_ctrl")

# Focused daily-life gestures dispatched to SystemController
# NOTE: pinch_in / pinch_out removed — zoom is now two-hand only (open-palm spread)
_SYSTEM_GESTURES = {
    "open_palm", "closed_fist", "thumbs_up", "thumbs_down",
    "peace_sign",
    "swipe_left", "swipe_right", "swipe_up", "swipe_down",
    "circular_cw", "circular_ccw",
    "three_fingers", "four_fingers",
    "wave",
}

# Network-related gestures forwarded to NetworkManager
_NETWORK_GESTURES = {"wave"}


class GestureController:
    """
    Receives GestureResult objects from the camera thread and
    dispatches them to:
      - SystemController  (OS actions + mode-specific bindings)
      - NetworkManager    (file transfer gestures)
      - MacroEngine       (multi-gesture sequences)
      - ProfileManager    (analytics, per-user stats)

    Hold-guard: Only fires when GestureResult.confirmed == True,
    meaning the gesture was stable for HOLD_FRAMES consecutive frames.
    """

    def __init__(self, config, event_bus):
        self.config      = config
        self.event_bus   = event_bus
        self._sys_ctrl   = None
        self._net_mgr    = None
        self._macro_eng  = None
        self._profiles   = None
        self._enabled    = True
        self._last_t:    float = 0.0
        self._cooldown:  float = float(config.get("gesture_cooldown", 0.5))
        # Two-hand combo cooldown (separate, longer)
        self._last_two_hand_t: float = 0.0
        self._two_hand_cd: float = 1.0
        log.info("GestureController initialised")

    # -- dependency injection --
    def set_system_controller(self, ctrl):
        self._sys_ctrl = ctrl

    def set_network_manager(self, mgr):
        self._net_mgr = mgr

    def set_macro_engine(self, eng):
        self._macro_eng = eng

    def set_profile_manager(self, pm):
        self._profiles = pm

    # -- mode forwarding --
    def set_mode(self, mode: str):
        """Forward mode change to SystemController."""
        if self._sys_ctrl:
            self._sys_ctrl.set_mode(mode)

    # -- lifecycle --
    def start(self):
        self._enabled = True
        log.info("GestureController started")

    def stop(self):
        self._enabled = False
        log.info("GestureController stopped")

    # -- main entry point --
    def on_gesture(self, gesture_name: str, confidence: float,
                   two_hand_gesture: str = "", confirmed: bool = True):
        """
        Call this whenever HandEngine produces a gesture.
        confirmed = True means the hold-guard has passed.
        Accepts both single-hand and two-hand gesture names.
        """
        if not self._enabled:
            return

        # Only fire if hold-guard confirmed (motion gestures already confirmed)
        if not confirmed:
            return

        now = time.time()

        # -- two-hand combo first (higher priority) --
        if two_hand_gesture:
            if now - self._last_two_hand_t > self._two_hand_cd:
                if self._sys_ctrl:
                    if "+" in two_hand_gesture:
                        parts = two_hand_gesture.split("+", 1)
                        left_g, right_g = parts[0], parts[1]
                    else:
                        left_g, right_g = two_hand_gesture, ""  # semantic combos like zoom
                    fired = self._sys_ctrl.execute_two_hand(left_g, right_g)
                    if fired:
                        self._last_two_hand_t = now
                        log.info(f"Two-hand combo: {two_hand_gesture}")
                        self._publish("two_hand_gesture", {
                            "combo": two_hand_gesture, "timestamp": now
                        })
                        return  # don't also fire single-hand action

        # -- single-hand gesture --
        if gesture_name in ("none", "unknown", "pointing", "pinch", ""):
            return
        # Raised confidence threshold for accuracy (0.75 vs old 0.65)
        if confidence < float(self.config.get("confidence_threshold", 0.75)):
            return
        if now - self._last_t < self._cooldown:
            return
        self._last_t = now

        log.debug(f"Dispatching gesture: {gesture_name} ({confidence:.2f})")

        # -- record in profile --
        if self._profiles:
            self._profiles.record_gesture(gesture_name)

        # -- macro engine --
        if self._macro_eng:
            macro = self._macro_eng.feed(gesture_name)
            if macro:
                log.info(f"Macro triggered: {macro}")
                self._publish("macro", {"name": macro, "timestamp": now})

        # -- network gestures --
        if gesture_name in _NETWORK_GESTURES and self._net_mgr is not None:
            self._net_mgr.on_gesture(gesture_name)

        # -- system actions --
        if gesture_name in _SYSTEM_GESTURES and self._sys_ctrl is not None:
            self._sys_ctrl.execute(gesture_name, confidence)

        # -- event bus --
        self._publish("gesture", {
            "name":       gesture_name,
            "confidence": confidence,
            "timestamp":  now,
        })

    # -- helpers --
    def _publish(self, event: str, data: dict):
        if self.event_bus:
            try:
                self.event_bus.publish(event, data)
            except Exception:
                pass

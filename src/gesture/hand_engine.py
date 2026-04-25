"""
HandEngine – MediaPipe hand detection, gesture recognition, and cursor control.
Updated for MediaPipe 0.10+ Tasks API.

Accuracy improvements in this version
───────────────────────────────────────
  • Hold-guard: gesture must be stable for HOLD_FRAMES (~8 frames ≈ 0.27s)
    before it fires → eliminates accidental triggers completely.
    Cursor (pointing) and motion gestures (swipe/pinch) are exempt.
  • Normalised pinch distance (relative to palm width) → works at any
    camera distance.
  • Angle-based thumb detection → separates thumbs_up/down from
    closed_fist reliably.
  • Swipe uses minimum velocity check, not just displacement.
  • Stricter confidence thresholds per gesture.
  • Removed rarely-used / ambiguous gestures from the recognition path.

Daily-life gesture set (16 gestures)
──────────────────────────────────────
  pointing        → cursor mode + SWIPE L/R  (index up, others folded)
  pinch           → left click               (thumb ≈ index)
  open_palm       → pause/play               (all extended)
  closed_fist     → confirm/Enter            (all folded, no thumb direction)
  thumbs_up       → accept                   (thumb up, others folded)
  thumbs_down     → reject                   (thumb down, others folded)
  peace_sign      → V-SIGN: SCROLL UP/DOWN   (index + middle up) ✌️
  three_fingers   → volume up                (index + middle + ring)
  four_fingers    → volume down              (all except thumb)
  swipe_left      → previous/back     ☝️ POINTING + move LEFT
  swipe_right     → next/forward      ☝️ POINTING + move RIGHT
  swipe_up        → scroll up         ✌️ V-SIGN  + move UP
  swipe_down      → scroll down       ✌️ V-SIGN  + move DOWN
  circular_cw     → refresh/reload           (pointing circular motion CW)
  circular_ccw    → undo                     (pointing circular motion CCW)
  wave            → cancel                   (oscillation)
"""

import math
import time
import os
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pyautogui

# ── pyautogui safety ───────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# ── Hold-guard configuration ───────────────────────────
HOLD_FRAMES_REQUIRED = 10         # raised to 10 ≈ 0.33s: better stability
                                   # lower = faster but more false fires
                                   # higher = slower but very accurate
# Gestures exempt from hold-guard (motion-based or immediate-response needed)
# pinch_in / pinch_out REMOVED — they are now two-hand-only via open-palm spread
_HOLD_EXEMPT = {
    # Motion gestures fire instantly — no hold needed
    "pointing",   # cursor mode + swipe L/R trigger shape
    "peace_sign", # scroll U/D trigger shape (no hold static action)
    "swipe_left", "swipe_right", "swipe_up", "swipe_down",
    "circular_cw", "circular_ccw", "wave",
    # Pinch fires instantly — click/select must feel immediate
    "pinch",
    "none", "unknown", "",
}

# ── Model auto-download ────────────────────────────────────
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "hand_landmarker.task")


def _ensure_model():
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"[HandEngine] Downloading model → {_MODEL_PATH} …")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[HandEngine] Model ready.")
    return _MODEL_PATH


# ── Hand skeleton connections ──────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── Landmark indices ───────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4;  THUMB_IP   = 3;  THUMB_MCP  = 2
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_TIP  = 20

# ── HUD colours (BGR) ──────────────────────────────────────
CLR_GREEN  = (34,  211, 165)
CLR_PURPLE = (255,  99, 108)
CLR_AMBER  = (11,  158, 245)
CLR_RED    = (68,   68, 239)
CLR_WHITE  = (230, 232, 226)
CLR_CYAN   = (200, 220, 50)


# ══════════════════════════════════════════════════════════
# DATA CLASS
# ══════════════════════════════════════════════════════════
@dataclass
class GestureResult:
    name:             str   = "none"
    confidence:       float = 0.0
    confirmed:        bool  = False    # True when hold-guard passed
    hold_progress:    float = 0.0     # 0.0–1.0 fill for UI
    hand_count:       int   = 0
    cursor_x:         int   = 0
    cursor_y:         int   = 0
    index_tip:        Tuple[float, float] = (0.0, 0.0)
    annotated_frame:  Optional[np.ndarray] = None
    two_hand_gesture: str   = ""      # only set when zoom/cross FIRES (not every frame)
    two_hand_active:  bool  = False   # True whenever 2 hands are visible (suppress single-hand)


# ══════════════════════════════════════════════════════════
# LANDMARK HELPERS
# ══════════════════════════════════════════════════════════
def _lm(lms, idx):
    l = lms[idx]; return l.x, l.y, l.z


def _dist2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def _finger_up(lms, tip, pip):
    """Finger extended = tip.y is significantly above pip.y (image coords)."""
    return lms[tip].y < lms[pip].y - 0.01


def _palm_width(lms) -> float:
    """Approximate palm width = dist between index MCP and pinky MCP."""
    i = lms[INDEX_MCP]; p = lms[PINKY_MCP]
    return _dist2((i.x, i.y), (p.x, p.y)) or 0.001


def _thumb_direction(lms) -> str:
    """
    Returns 'up', 'down', or 'side'.
    Uses angle of thumb tip relative to wrist, normalised by hand orientation.
    Thresholds tightened for fewer false thumbs_up/down from closed_fist.
    """
    tip   = lms[THUMB_TIP]
    wrist = lms[WRIST]
    mid_mcp = lms[MIDDLE_MCP]  # hand direction reference

    # Hand vertical axis (wrist → middle MCP)
    hand_dy = mid_mcp.y - wrist.y  # negative = hand pointing up
    # Thumb tip relative to wrist
    ty = tip.y - wrist.y

    # Normalise by hand length
    hand_len = abs(_dist2((mid_mcp.x, mid_mcp.y), (wrist.x, wrist.y))) or 0.001
    relative = ty / hand_len

    if relative < -0.35:     # stricter: was -0.3
        return "up"
    elif relative > 0.20:    # stricter: was 0.15
        return "down"
    return "side"


def _thumb_extended(lms) -> bool:
    """
    Thumb is 'extended' (away from palm) when its tip is far from  
    the index finger MCP landmark, normalised by palm width.
    """
    pw = _palm_width(lms)
    tip = lms[THUMB_TIP]; ref = lms[INDEX_MCP]
    return _dist2((tip.x, tip.y), (ref.x, ref.y)) > pw * 0.6


def _states(lms):
    return (
        _thumb_extended(lms),
        _finger_up(lms, INDEX_TIP,  INDEX_PIP),
        _finger_up(lms, MIDDLE_TIP, MIDDLE_PIP),
        _finger_up(lms, RING_TIP,   RING_PIP),
        _finger_up(lms, PINKY_TIP,  PINKY_PIP),
    )


def _pinch_norm(lms) -> float:
    """Normalised pinch distance (0 = touching, 1 = fully open)."""
    pw   = _palm_width(lms)
    tip  = _lm(lms, THUMB_TIP)[:2]
    itip = _lm(lms, INDEX_TIP)[:2]
    return _dist2(tip, itip) / pw


# ══════════════════════════════════════════════════════════
# GESTURE CLASSIFIER — improved accuracy
# ══════════════════════════════════════════════════════════
def classify_gesture(lms, prev_pinch_norm: float = 0.5) -> Tuple[str, float]:
    """
    Returns (gesture_name, confidence).
    Priority-ordered to avoid overlapping matches.
    """
    t, i, m, r, p = _states(lms)
    pinch_n = _pinch_norm(lms)     # normalised 0–1

    # ── pinch / ok (high priority) ─────────────────────────────────────
    # Raised threshold to 0.20 → easier to pinch naturally without straining
    if pinch_n < 0.20:
        if m and r and p:
            return "ok_sign", 0.90      # thumb+index circle, rest extended
        if i and not m and not r:
            return "pinch", 0.93        # click / select
        if not i and not m:
            return "pinch", 0.90        # click / select (all fingers folded)

    # ── open palm (all 5 extended) ───────────────────────────────────
    if t and i and m and r and p:
        return "open_palm", 0.90

    # ── four fingers (all up, thumb IN) ───────────────────────
    if not t and i and m and r and p:
        return "four_fingers", 0.87

    # ── three fingers (index+middle+ring) ─────────────────
    if i and m and r and not p and not t:
        return "three_fingers", 0.85

    # ── peace sign / V-sign (index+middle up, ring+pinky down) ──────────
    # Accept thumb in any state — natural hand position while making V-sign
    if i and m and not r and not p:
        return "peace_sign", 0.87

    # ── pointing (only index up, others clearly folded) ───
    if i and not m and not r and not p:
        # Stricter: middle tip must be visibly below its MCP joint
        middle_folded = lms[MIDDLE_TIP].y > lms[MIDDLE_MCP].y - 0.005
        if middle_folded:
            return "pointing", 0.87
        return "pointing", 0.72   # lower conf when middle is borderline

    # ── thumbs up/down/closed_fist (all fingers folded) ───
    if not i and not m and not r and not p:
        td = _thumb_direction(lms)
        if t:
            if td == "up":
                return "thumbs_up", 0.90
            elif td == "down":
                return "thumbs_down", 0.90
        # thumb curled too → closed fist
        return "closed_fist", 0.88

    # ── phone sign (thumb + pinky) ────────────────────────
    if t and not i and not m and not r and p:
        return "phone_sign", 0.78

    return "unknown", 0.0


# ══════════════════════════════════════════════════════════
# SMOOTH MOUSE
# ══════════════════════════════════════════════════════════
class SmoothMouse:
    def __init__(self, alpha=0.20):
        self._a = alpha
        self._sx, self._sy = pyautogui.position()

    def move(self, tx, ty):
        self._sx = self._a*tx + (1-self._a)*self._sx
        self._sy = self._a*ty + (1-self._a)*self._sy
        pyautogui.moveTo(int(self._sx), int(self._sy))

    @property
    def pos(self): return int(self._sx), int(self._sy)


# ══════════════════════════════════════════════════════════
# CIRCULAR MOTION DETECTOR
# ══════════════════════════════════════════════════════════
class CircularDetector:
    """Detects clockwise / counter-clockwise circular motion."""
    def __init__(self, window=0.9, min_points=14, min_radius=0.05):
        self._hist:    deque = deque(maxlen=50)
        self._window   = window
        self._min_pts  = min_points
        self._min_r    = min_radius

    def update(self, x: float, y: float) -> Optional[str]:
        now = time.time()
        self._hist.append((x, y, now))
        pts = [(x, y) for x, y, t in self._hist if now - t < self._window]
        if len(pts) < self._min_pts:
            return None
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        radii = [math.hypot(p[0]-cx, p[1]-cy) for p in pts]
        if sum(radii)/len(radii) < self._min_r:
            return None
        total_cross = 0.0
        for k in range(1, len(pts)):
            ax, ay = pts[k-1][0]-cx, pts[k-1][1]-cy
            bx, by = pts[k][0]-cx,   pts[k][1]-cy
            total_cross += ax*by - ay*bx
        if abs(total_cross) < 0.006:
            return None
        self._hist.clear()
        return "circular_cw" if total_cross < 0 else "circular_ccw"


# ══════════════════════════════════════════════════════════
# MAIN ENGINE
# ══════════════════════════════════════════════════════════
class HandEngine:
    """
    Full gesture recognition engine with hold-guard accuracy enhancement.
    Compatible with MediaPipe >= 0.10.x.
    """

    def __init__(self,
                 max_hands:           int   = 2,
                 detection_conf:      float = 0.75,   # raised from 0.70
                 tracking_conf:       float = 0.70,   # raised from 0.65
                 cursor_smoothing:    float = 0.18,
                 cursor_enabled:      bool  = True,
                 click_cooldown_s:    float = 0.45,
                 gesture_cooldown_s:  float = 0.50,
                 hold_frames:         int   = HOLD_FRAMES_REQUIRED):

        model_path = _ensure_model()
        base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._lmk   = mp_vision.HandLandmarker.create_from_options(opts)
        # ── Strictly-monotonic timestamp for MediaPipe VIDEO mode ────────────
        # Wall-clock time is UNSAFE: NTP syncs, suspend/resume, or any system
        # jitter can produce a duplicate or decreasing timestamp which causes
        # MediaPipe to silently drop all subsequent frames forever.
        # Solution: use a simple incrementing counter × fixed frame period.
        self._frame_index: int = 0
        self._MS_PER_FRAME: int = 33   # ~30 fps; slightly under 1000/30 is fine

        self._mouse          = SmoothMouse(cursor_smoothing)
        self._cursor_enabled = cursor_enabled
        self._cursor_active  = False

        self._last_click_t   = 0.0
        self._click_cd       = click_cooldown_s
        self._last_gest      = ""
        self._last_gest_t    = 0.0
        self._gest_cd        = gesture_cooldown_s

        # ── PINCH / CLICK state ──────────────────────────────────────────
        self._pinch_active:   bool  = False   # True while thumb≈index
        self._pinch_start_t:  float = 0.0     # time pinch began
        self._pinch_drag:     bool  = False   # True once drag started
        _DRAG_HOLD_S          = 0.55          # seconds held to start drag
        self._DRAG_HOLD_S     = _DRAG_HOLD_S

        # motion tracking history: (x, y, t)
        # Uses index-tip position for wider range of movement detection
        self._wrist_hist: deque = deque(maxlen=30)  # expanded buffer

        # normalised pinch history
        self._prev_pinch_norm: float = 0.5
        self._pinch_hist: deque = deque([0.5]*6, maxlen=6)

        # circular motion
        self._circle = CircularDetector()

        # wave detector
        self._wave_last_x: float = 0.0
        self._wave_dirs:   List[str] = []

        # ── TWO-HAND ZOOM state ────────────────────────────────────────
        # History stores (avg_dist, timestamp)
        self._two_hand_dist_hist: deque = deque(maxlen=20)  # expanded buffer
        self._prev_two_hand_dist: float  = -1.0
        self._zoom_cooldown_t: float = 0.0
        self._zoom_cd = 0.35           # reduced: snappier zoom response
        self._zoom_step_count: int = 1 # proportional presses per fire

        # ── TWO-HAND CROSS (X) state ────────────────────────
        # Crossed = Right hand wrist is to the LEFT of Left hand wrist
        self._cross_hold_count: int   = 0
        self._cross_cooldown_t: float = 0.0
        self._cross_cd: float = 1.2   # seconds between cross fires

        # ── HOLD GUARD ──────────────────────────────────────
        self._hold_frames   = hold_frames
        self._hold_gesture  = ""
        self._hold_count    = 0

    # ── public ────────────────────────────────────────────
    def process(self, bgr: np.ndarray) -> GestureResult:
        frame = cv2.flip(bgr, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Strictly-monotonic timestamp: frame_index × 33ms
        # This is SAFE regardless of system clock jitter/NTP/suspend.
        self._frame_index += 1
        ts_ms = self._frame_index * self._MS_PER_FRAME
        det = self._lmk.detect_for_video(mp_img, ts_ms)

        result = GestureResult(hand_count=0)
        now = time.time()

        if det.hand_landmarks:
            result.hand_count = len(det.hand_landmarks)
            hand = det.hand_landmarks[0]
            hd   = det.handedness

            # ── classify primary hand ──────────────────────────────────────
            pinch_n = _pinch_norm(hand)
            gesture, conf = classify_gesture(hand, self._prev_pinch_norm)

            # ── dynamic gesture overrides ──────────────────────────────────
            # Track INDEX TIP (not wrist) — gives ~2-3x more motion range
            # for reliable swipe/scroll detection
            _itip_lm  = hand[INDEX_TIP]
            _mtip_lm  = hand[MIDDLE_TIP]
            # Use midpoint of index+middle tips for V-sign scroll,
            # or pure index tip for pointing swipe
            if gesture == "peace_sign":
                wx = (_itip_lm.x + _mtip_lm.x) / 2.0
                wy = (_itip_lm.y + _mtip_lm.y) / 2.0
            else:
                wx = _itip_lm.x
                wy = _itip_lm.y
            self._wrist_hist.append((wx, wy, now))

            swipe = self._detect_swipe(now, gesture)
            if swipe:
                gesture, conf = swipe, 0.82

            if gesture == "pointing":
                tip    = hand[INDEX_TIP]
                circle = self._circle.update(tip.x, tip.y)
                if circle:
                    gesture, conf = circle, 0.82
            else:
                self._circle._hist.clear()

            # wave
            wave = self._detect_wave(wx, now, gesture)
            if wave:
                gesture, conf = "wave", 0.78

            self._prev_pinch_norm = pinch_n

            # ── HOLD GUARD ────────────────────────────────────────────────
            if gesture in _HOLD_EXEMPT:
                result.confirmed     = True
                result.hold_progress = 1.0
                self._hold_gesture   = gesture
                self._hold_count     = self._hold_frames
            else:
                if gesture == self._hold_gesture and gesture not in ("none", "unknown", ""):
                    self._hold_count = min(self._hold_count + 1, self._hold_frames)
                else:
                    self._hold_gesture = gesture
                    self._hold_count   = 1

                result.hold_progress = self._hold_count / self._hold_frames
                result.confirmed     = (self._hold_count >= self._hold_frames)

            result.name       = gesture
            result.confidence = conf

            idx_tip = hand[INDEX_TIP]
            result.index_tip = (idx_tip.x, idx_tip.y)

            # ── two-hand gestures ─────────────────────────────────────────
            if result.hand_count >= 2:
                result.two_hand_active = True   # suppress single-hand OS actions downstream
                hand2 = det.hand_landmarks[1]
                g2, _ = classify_gesture(hand2, 0.5)

                # CRITICAL FIX: Do NOT set result.two_hand_gesture here every frame.
                # It is ONLY set when zoom / cross actually fires below.
                # Setting it every frame caused 30fps signal spam.
                # Just always draw the second hand skeleton for visual feedback.
                self._draw_hand(frame, hand2, w, h, hd[1:] if len(hd) > 1 else [])

                # ── Two-hand ZOOM (both open_palm or four_fingers) ───────────
                # ACCURACY: Measure average distance across ALL 5 fingertip
                # pairs (thumb-thumb, index-index, ... pinky-pinky). This is
                # 5× more stable than a single index-tip measurement.
                both_open = (
                    gesture in ("open_palm", "four_fingers") and
                    g2      in ("open_palm", "four_fingers")
                )
                if both_open:
                    _TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
                    dists = [
                        _dist2((hand[tid].x, hand[tid].y),
                               (hand2[tid].x, hand2[tid].y))
                        for tid in _TIPS
                    ]
                    dist = sum(dists) / len(dists)   # average of 5 pairs
                    self._two_hand_dist_hist.append((dist, now))

                    n = len(self._two_hand_dist_hist)
                    if (self._prev_two_hand_dist > 0 and n >= 8 and
                            now - self._zoom_cooldown_t > self._zoom_cd):

                        recent = list(self._two_hand_dist_hist)
                        # Compare oldest 4 vs newest 4 for a stable delta
                        old_d  = sum(x[0] for x in recent[:4]) / 4
                        new_d  = sum(x[0] for x in recent[-4:]) / 4
                        delta  = new_d - old_d

                        # Dead-zone: ignore deltas smaller than 0.030
                        # (filters out natural hand tremor / breathing sway)
                        DEAD_ZONE = 0.030

                        if delta > DEAD_ZONE:         # hands APART → zoom in
                            # Proportional: bigger spread = more zoom steps
                            steps = min(int(delta / 0.025), 3)
                            self._zoom_step_count = max(steps, 1)
                            result.two_hand_gesture = "two_hand_zoom_in"
                            result.confidence = 0.92
                            result.confirmed  = True
                            self._zoom_cooldown_t = now
                            self._two_hand_dist_hist.clear()
                        elif delta < -DEAD_ZONE:      # hands TOGETHER → zoom out
                            steps = min(int(abs(delta) / 0.025), 3)
                            self._zoom_step_count = max(steps, 1)
                            result.two_hand_gesture = "two_hand_zoom_out"
                            result.confidence = 0.92
                            result.confirmed  = True
                            self._zoom_cooldown_t = now
                            self._two_hand_dist_hist.clear()

                    self._prev_two_hand_dist = dist
                else:
                    self._prev_two_hand_dist = -1.0
                    self._two_hand_dist_hist.clear()

                # ── Two-hand CROSS (X shape) → minimize / maximize ──────────
                # Detect when Right-hand wrist crosses to the LEFT of Left-hand wrist.
                # After cv2.flip the image is mirrored, so "Right" hand has larger x.
                # Cross = right_wrist.x LESS than left_wrist.x significantly.
                right_wrist_x = None
                left_wrist_x  = None
                for idx_h, hand_lm in enumerate(det.hand_landmarks[:2]):
                    if idx_h < len(hd) and hd[idx_h]:
                        label = hd[idx_h][0].category_name  # "Left" or "Right"
                        wx_h  = hand_lm[WRIST].x
                        if label == "Right":
                            right_wrist_x = wx_h
                        else:
                            left_wrist_x  = wx_h

                if right_wrist_x is not None and left_wrist_x is not None:
                    # Crossed when right wrist x < left wrist x by ≥0.10
                    is_crossed = (right_wrist_x - left_wrist_x) < -0.10
                    if is_crossed:
                        self._cross_hold_count += 1
                        # Show progress while holding cross
                        cross_prog = min(self._cross_hold_count / self._hold_frames, 1.0)
                        # Draw X overlay hint on frame
                        cv2.putText(frame, f"CROSS {int(cross_prog*100)}%",
                                    (w//2 - 60, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLR_AMBER, 2)
                        if (self._cross_hold_count >= self._hold_frames and
                                now - self._cross_cooldown_t > self._cross_cd):
                            result.two_hand_gesture = "two_hand_cross"
                            result.confidence = 0.87
                            result.confirmed  = True
                            self._cross_cooldown_t = now
                            self._cross_hold_count = 0
                    else:
                        self._cross_hold_count = 0
                else:
                    self._cross_hold_count = 0

            else:
                # No second hand — clear zoom state
                self._prev_two_hand_dist = -1.0
                self._two_hand_dist_hist.clear()

            # ── cursor ────────────────────────────────────────────────────
            if self._cursor_enabled:
                self._handle_cursor(hand, gesture, result, now)

            # ── draw skeleton + HUD ───────────────────────────────────────
            self._draw_hand(frame, hand, w, h, hd)
            self._draw_hud(frame, result)

        else:
            self._cursor_active = False
            self._wrist_hist.clear()
            self._wave_dirs  = []
            self._hold_gesture = ""
            self._hold_count   = 0
            self._prev_two_hand_dist = -1.0
            self._two_hand_dist_hist.clear()
            self._cross_hold_count = 0
            self._draw_hud_idle(frame)

        result.annotated_frame = frame
        return result

    def set_cursor_enabled(self, enabled: bool):
        self._cursor_enabled = enabled
        if not enabled:
            self._cursor_active = False

    def close(self):
        self._lmk.close()

    # ── internal ──────────────────────────────────────────

    def _handle_cursor(self, lms, gesture, result, now):
        """
        Cursor control + pinch-to-click / pinch-to-drag.

        Pinch behaviour
        ───────────────
        • Finger together (pinch_n < 0.20)   → gesture == "pinch"
        • On first frame of pinch            → LEFT CLICK at current cursor pos
        • Pinch held > 0.55 s               → DRAG mode (mouseDown until release)
        • Fingers open again                 → click released / drag dropped

        Cursor follows the INDEX TIP in ALL modes (pointing AND pinch)
        so you can aim, then pinch to click — no mode switch needed.
        """
        idx    = lms[INDEX_TIP]
        thumb  = lms[THUMB_TIP]
        MARGIN = 0.15

        # Use midpoint of thumb+index as click position (feels more natural)
        cx_norm = (idx.x + thumb.x) / 2.0
        cy_norm = (idx.y + thumb.y) / 2.0
        nx = max(0.0, min(1.0, (cx_norm - MARGIN) / (1 - 2*MARGIN)))
        ny = max(0.0, min(1.0, (cy_norm - MARGIN) / (1 - 2*MARGIN)))
        tx = int(nx * SCREEN_W)
        ty = int(ny * SCREEN_H)
        result.cursor_x = tx
        result.cursor_y = ty

        if gesture == "pointing":
            self._cursor_active = True
            self._mouse.move(tx, ty)
            # Release any ongoing drag when user switches to pointing
            if self._pinch_drag:
                pyautogui.mouseUp()
                self._pinch_drag   = False
                self._pinch_active = False

        elif gesture == "pinch":
            self._cursor_active = True
            # Always keep cursor on the pinch midpoint
            self._mouse.move(tx, ty)

            if not self._pinch_active:
                # ── Pinch just closed → LEFT CLICK ──────────────────────
                self._pinch_active  = True
                self._pinch_start_t = now
                self._pinch_drag    = False
                if now - self._last_click_t > self._click_cd:
                    pyautogui.click()
                    self._last_click_t = now
            else:
                # ── Pinch held continuously ──────────────────────────────
                held = now - self._pinch_start_t
                if held >= self._DRAG_HOLD_S and not self._pinch_drag:
                    # Transition to DRAG: press and hold the mouse button
                    pyautogui.mouseDown()
                    self._pinch_drag = True
                elif self._pinch_drag:
                    # Dragging — cursor already moved above, nothing extra needed
                    pass

        else:
            # ── Pinch released / gesture changed ────────────────────────
            if self._pinch_active:
                if self._pinch_drag:
                    pyautogui.mouseUp()   # drop whatever was being dragged
                self._pinch_drag   = False
                self._pinch_active = False

            if gesture in ("open_palm", "closed_fist"):
                self._cursor_active = False

    def _detect_swipe(self, now: float, gesture: str = "") -> Optional[str]:
        """
        ACCURATE gesture-gated swipe/scroll detection.

        Sign guide
        ──────────
          ☝️  POINTING  (only index up)  →  SWIPE LEFT / RIGHT
              Move hand LEFT or RIGHT quickly while pointing.
              Thresholds: |dx| ≥ 0.07, |vx| ≥ 0.18, horizontal dominant 1.4×

          ✌️  V-SIGN / PEACE (index + middle up, ring + pinky down)
              →  SCROLL UP / DOWN
              Move hand UP or DOWN while making a V-sign.
              Thresholds: |dy| ≥ 0.06, |vy| ≥ 0.12, vertical dominant 1.3×

          Any other shape — very strict fallback (≥0.20) to avoid accidents.

        Note: MediaPipe image coords have y increasing downward, so:
          dy > 0  means hand moved DOWN in camera  → scroll DOWN
          dy < 0  means hand moved UP   in camera  → scroll UP
        """
        # Capture recent motion samples within a 0.75s window
        recent = [(x, y, t) for x, y, t in self._wrist_hist
                  if now - t < 0.75]
        if len(recent) < 4:
            return None

        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        dt = recent[-1][2] - recent[0][2]
        if dt < 0.01:
            return None

        vx = dx / dt
        vy = dy / dt

        # ── SWIPE LEFT / RIGHT: ☝️ pointing (index only) ─────────────────────
        if gesture == "pointing":
            # Lower threshold: index-tip tracking gives wider motion range
            if abs(dx) >= 0.07 and abs(vx) >= 0.18 and abs(dx) > abs(dy) * 1.4:
                self._wrist_hist.clear()
                return "swipe_right" if dx > 0 else "swipe_left"

        # ── SCROLL UP / DOWN: ✌️ V-sign / peace_sign ─────────────────────────
        elif gesture == "peace_sign":
            # Lower threshold: midpoint of two fingertips gives smooth tracking
            if abs(dy) >= 0.06 and abs(vy) >= 0.12 and abs(dy) > abs(dx) * 1.3:
                self._wrist_hist.clear()
                # dy > 0 = hand moved down in camera = scroll DOWN
                return "swipe_down" if dy > 0 else "swipe_up"

        # ── FALLBACK: any other shape — strict to avoid accidental fires ──────
        else:
            if abs(dx) >= 0.20 and abs(vx) >= 0.38 and abs(dx) > abs(dy) * 2.5:
                self._wrist_hist.clear()
                return "swipe_right" if dx > 0 else "swipe_left"
            if abs(dy) >= 0.20 and abs(vy) >= 0.38 and abs(dy) > abs(dx) * 2.5:
                self._wrist_hist.clear()
                return "swipe_down" if dy > 0 else "swipe_up"

        return None

    def _detect_wave(self, wx: float, now: float, gesture: str) -> bool:
        """Wave = rapid left-right oscillation with open palm.
        NOTE: open_palm is no longer used for swipe, so wave detection
        remains exclusively on open_palm/four_fingers without conflict.
        """
        if gesture not in ("open_palm", "four_fingers"):
            self._wave_dirs.clear()
            self._wave_last_x = wx
            return False

        dx = wx - self._wave_last_x
        self._wave_last_x = wx

        if abs(dx) > 0.035:
            direction = "R" if dx > 0 else "L"
            if not self._wave_dirs or self._wave_dirs[-1] != direction:
                self._wave_dirs.append(direction)

        if len(self._wave_dirs) > 8:
            self._wave_dirs = self._wave_dirs[-8:]

        changes = sum(1 for k in range(1, len(self._wave_dirs))
                      if self._wave_dirs[k] != self._wave_dirs[k-1])
        if changes >= 5:
            self._wave_dirs.clear()
            return True
        return False

    # ── drawing ───────────────────────────────────────────

    def _draw_hand(self, frame, lms, w, h, hd_list):
        pts = [(int(l.x*w), int(l.y*h)) for l in lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], CLR_PURPLE, 2)
        for cx, cy in pts:
            cv2.circle(frame, (cx, cy), 3, CLR_GREEN, -1)
        for tid in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
            cx, cy = pts[tid]
            cv2.circle(frame, (cx, cy), 8,  CLR_GREEN, -1)
            cv2.circle(frame, (cx, cy), 10, CLR_WHITE, 1)
        wx, wy = pts[WRIST]
        label = "Right"
        if hd_list and hd_list[0]:
            label = hd_list[0][0].category_name
        cv2.putText(frame, label, (wx-10, wy+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_PURPLE, 2)

    _HUD = {
        # ☝️ pointing = DUAL PURPOSE: cursor move + SWIPE L/R trigger
        "pointing":      ("\u261d POINT \u2194  SWIPE L/R",   CLR_PURPLE),
        "pinch":         ("\U0001f90f PINCH = CLICK / DRAG",  CLR_GREEN),
        # ✌️ peace_sign / V-sign = DEDICATED SCROLL trigger
        "peace_sign":    ("\u270c V-SIGN \u2195  SCROLL U/D", CLR_CYAN),
        "open_palm":     ("PAUSE/PLAY",               CLR_WHITE),
        "closed_fist":   ("CONFIRM",                  CLR_GREEN),
        "thumbs_up":     ("\u2713 ACCEPT",             CLR_GREEN),
        "thumbs_down":   ("\u2717 REJECT",             CLR_RED),
        "three_fingers": ("VOL UP\u25b2",              CLR_GREEN),
        "four_fingers":  ("VOL DOWN\u25bc",            CLR_AMBER),
        "swipe_left":    ("\u25c4 SWIPE LEFT",         CLR_PURPLE),
        "swipe_right":   ("SWIPE RIGHT \u25ba",        CLR_PURPLE),
        "swipe_up":      ("\u25b2 SCROLL UP",          CLR_CYAN),
        "swipe_down":    ("\u25bc SCROLL DOWN",        CLR_CYAN),
        "circular_cw":   ("\u21bb REFRESH",            CLR_GREEN),
        "circular_ccw":  ("\u21ba UNDO",               CLR_AMBER),
        "pinch_in":      ("\u25c4 ZOOM OUT",           CLR_AMBER),
        "pinch_out":     ("\u25ba ZOOM IN",            CLR_GREEN),
        "wave":          ("\u2715 CANCEL",             CLR_RED),
        "ok_sign":       ("SCREENSHOT",               CLR_AMBER),
        "phone_sign":    ("SETTINGS",                 CLR_GREEN),
        # Two hand gestures
        "two_hand_zoom_in":  ("\u25ba\u25ba ZOOM IN [2H]",    CLR_GREEN),
        "two_hand_zoom_out": ("\u25c4\u25c4 ZOOM OUT [2H]",   CLR_AMBER),
        "two_hand_cross":    ("\u2715 CROSS \u2192 MIN/MAX [2H]", CLR_RED),
    }

    def _draw_hud(self, frame, result: GestureResult):
        h, w = frame.shape[:2]
        display_name = result.two_hand_gesture if (result.two_hand_gesture and "zoom" in result.two_hand_gesture) else result.name
        label_text, label_color = self._HUD.get(
            display_name,
            (display_name.upper().replace("_"," "), CLR_WHITE)
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = 0.75
        th_  = 2

        # Fade label if hold not yet confirmed
        alpha_mul = 0.45 + 0.55 * result.hold_progress
        fade_c = tuple(int(c * alpha_mul) for c in label_color)

        (tw, text_h), _ = cv2.getTextSize(label_text, font, fs, th_)
        pad  = 12
        bx   = (w - tw)//2 - pad
        by   = h - 56 - text_h - pad
        ov   = frame.copy()
        cv2.rectangle(ov, (bx, by), (bx+tw+pad*2, by+text_h+pad*2), (15,12,10), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx+tw+pad*2, by+text_h+pad*2), fade_c, 2)
        cv2.putText(frame, label_text, (bx+pad, by+text_h+pad//2), font, fs, fade_c, th_)

        # ── pinch drag progress bar ──────────────────────────────────────
        # Shows how long the pinch has been held; fills → DRAG ACTIVE
        if result.name == "pinch" and self._pinch_active:
            held = time.time() - self._pinch_start_t
            drag_prog = min(held / self._DRAG_HOLD_S, 1.0)
            bar_total = tw + pad * 2
            bar_filled = int(bar_total * drag_prog)
            bar_y = by + text_h + pad * 2 + 5
            # background
            cv2.rectangle(frame, (bx, bar_y), (bx + bar_total, bar_y + 6),
                          (40, 35, 30), -1)
            # fill (amber → red as it fills)
            bar_col = CLR_RED if drag_prog >= 1.0 else CLR_AMBER
            cv2.rectangle(frame, (bx, bar_y), (bx + bar_filled, bar_y + 6),
                          bar_col, -1)
            if self._pinch_drag:
                cv2.putText(frame, "DRAG ACTIVE",
                            (bx + pad, bar_y + 22),
                            font, 0.50, CLR_RED, 2)

        # ── hold-guard progress arc ──
        if result.name not in _HOLD_EXEMPT and result.name not in ("none","unknown",""):
            cx_arc = bx + tw//2 + pad
            cy_arc = by - 16
            radius = 12
            progress_angle = int(360 * result.hold_progress)
            cv2.ellipse(frame, (cx_arc, cy_arc), (radius, radius),
                        -90, 0, progress_angle, CLR_GREEN, 3)
            cv2.ellipse(frame, (cx_arc, cy_arc), (radius, radius),
                        -90, progress_angle, 360, (40,40,40), 1)

        # hand count
        cv2.putText(frame, f"Hands: {result.hand_count}", (12, 30),
                    font, 0.55, CLR_WHITE, 1)

        # cursor ON chip
        if self._cursor_active:
            ct = "CURSOR ON"
            (ctw, _), _ = cv2.getTextSize(ct, font, 0.55, 1)
            cv2.rectangle(frame, (w-ctw-22, 8), (w-8, 30), CLR_PURPLE, -1)
            cv2.putText(frame, ct, (w-ctw-14, 24), font, 0.55, CLR_WHITE, 1)

        # confidence bar
        bw = int(result.confidence * 120)
        cv2.rectangle(frame, (10, h-24), (130, h-10), (40,35,30), -1)
        bar_c = CLR_GREEN if result.confidence > 0.75 else CLR_AMBER
        cv2.rectangle(frame, (10, h-24), (10+bw, h-10), bar_c, -1)
        cv2.putText(frame, f"{result.confidence*100:.0f}%",
                    (135, h-12), font, 0.45, CLR_WHITE, 1)

    def _draw_hud_idle(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, "No hand detected", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,80,80), 1)

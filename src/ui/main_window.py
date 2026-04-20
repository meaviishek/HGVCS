"""
HGVCS Main Window
Full-featured dark-themed UI:
  • Dashboard  – live camera, gesture log, stats
  • Gestures   – reference guide (all 20+ gestures)
  • Network    – LAN peer list, file transfer, progress
  • Settings   – config display

New in this version:
  • All gestures wired to SystemController OS actions
  • GestureController bridge initialised from window
  • Network tab with peer list, transfer progress, drag-hold send
  • Toast notification overlay for each gesture
"""

import sys
import os
import math
import random
import time
import threading

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGridLayout, QProgressBar, QTabWidget,
    QSizePolicy, QSpacerItem, QGraphicsDropShadowEffect, QApplication,
    QStatusBar, QToolBar, QAction, QSystemTrayIcon, QMenu, QSlider,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QListWidget,
    QListWidgetItem, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QPropertyAnimation,
    QEasingCurve, QPoint, QRect, pyqtProperty, QMetaObject, Q_ARG,
    pyqtSlot
)
from PyQt5.QtGui import (
    QColor, QPalette, QFont, QFontDatabase, QIcon, QPainter,
    QPainterPath, QLinearGradient, QBrush, QPen, QPixmap,
    QRadialGradient, QPolygon, QImage
)

# ── COLOUR TOKENS ──────────────────────────────────────────
BG       = "#0d0f14"
SURFACE  = "#161a24"
SURFACE2 = "#1e2435"
BORDER   = "#2a3245"
ACCENT   = "#6c63ff"
ACCENT2  = "#a78bfa"
GREEN    = "#22d3a5"
AMBER    = "#f59e0b"
RED      = "#ef4444"
TEXT     = "#e2e8f0"
DIMTEXT  = "#64748b"
CARD     = "#1a1f2e"


def _shadow(radius=18, color="#6c63ff", opacity=80, dx=0, dy=4):
    e = QGraphicsDropShadowEffect()
    e.setBlurRadius(radius)
    c = QColor(color); c.setAlpha(opacity)
    e.setColor(c); e.setOffset(dx, dy)
    return e


def _font(size=12, weight=QFont.Normal, family="Segoe UI"):
    return QFont(family, size, weight)


# ══════════════════════════════════════════════════════════
# TOAST NOTIFICATION
# ══════════════════════════════════════════════════════════
class ToastOverlay(QWidget):
    """Semi-transparent gesture / voice notification that fades out."""
    _COLORS = {
        "pointing":      ACCENT2,
        "pinch":         GREEN,
        "peace_sign":    AMBER,
        "open_palm":     TEXT,
        "closed_fist":   GREEN,
        "thumbs_up":     GREEN,
        "thumbs_down":   RED,
        "three_fingers": GREEN,
        "four_fingers":  AMBER,
        "rock_on":       ACCENT,
        "ok_sign":       AMBER,
        "swipe_left":    ACCENT,
        "swipe_right":   ACCENT,
        "swipe_up":      ACCENT2,
        "swipe_down":    ACCENT2,
        "circular_cw":   GREEN,
        "circular_ccw":  AMBER,
        "pinch_in":      AMBER,
        "pinch_out":     GREEN,
        "wave":          RED,
        "phone_sign":    GREEN,
        # Voice toasts use purple
        "__voice__":     ACCENT,
    }
    _LABELS = {
        "pointing":      "☝️  Swipe: Move wrist L/R  |  Cursor Mode",
        "pinch":         "🤏  Click",
        # peace_sign is now exclusively a scroll trigger (no static action)
        "peace_sign":    "✌️  Scroll: Move wrist Up/Down",
        "open_palm":     "✋  Pause / Play",
        "closed_fist":   "✊  Confirm",
        "thumbs_up":     "👍  Accept",
        "thumbs_down":   "👎  Reject",
        "three_fingers": "🖖  Volume Up",
        "four_fingers":  "🖐️  Volume Down",
        "swipe_left":    "👈  Previous",
        "swipe_right":   "👉  Next",
        "swipe_up":      "👆  Scroll Up",
        "swipe_down":    "👇  Scroll Down",
        "circular_cw":   "🔄  Refresh",
        "circular_ccw":  "🔃  Undo",
        "pinch_in":      "🔍  Zoom Out",
        "pinch_out":     "🔍+  Zoom In",
        "wave":          "👋  Cancel",
        "ok_sign":       "👌  Screenshot!",
        "phone_sign":    "🤙  Settings",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._alpha   = 0
        self._text    = ""
        self._color   = QColor(ACCENT)
        self._timer   = QTimer(self)
        self._timer.timeout.connect(self._fade)
        self._timer.setInterval(30)

    def show_gesture(self, gesture: str):
        self._text  = self._LABELS.get(gesture, gesture.replace("_", " ").title())
        self._color = QColor(self._COLORS.get(gesture, ACCENT))
        self._alpha = 240
        self._timer.start()
        self.update()

    def _fade(self):
        self._alpha = max(0, self._alpha - 8)
        self.update()
        if self._alpha == 0:
            self._timer.stop()

    def paintEvent(self, event):
        if self._alpha == 0 or not self._text:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # pill background
        p.setFont(_font(14, QFont.Bold))
        fm  = p.fontMetrics()
        tw  = fm.horizontalAdvance(self._text)
        pad = 18
        pw  = tw + pad*2
        ph  = 44
        px  = (w - pw) // 2
        py  = 20

        bg = QColor(15, 12, 20)
        bg.setAlpha(int(self._alpha * 0.85))
        p.setBrush(QBrush(bg))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(px, py, pw, ph, 22, 22)

        border_c = QColor(self._color)
        border_c.setAlpha(self._alpha)
        p.setPen(QPen(border_c, 2))
        p.drawRoundedRect(px, py, pw, ph, 22, 22)

        text_c = QColor(TEXT)
        text_c.setAlpha(self._alpha)
        p.setPen(text_c)
        p.drawText(QRect(px, py, pw, ph), Qt.AlignCenter, self._text)


# ══════════════════════════════════════════════════════════
# WIDGETS (unchanged from original)
# ══════════════════════════════════════════════════════════
class PulseDot(QWidget):
    def __init__(self, color=GREEN, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._alpha = 255
        self.setFixedSize(14, 14)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pulse)
        self._timer.start(40)
        self._t = 0

    def _pulse(self):
        self._t += 0.12
        self._alpha = int(140 + 115 * abs(math.sin(self._t)))
        self.update()

    def set_color(self, color):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        outer = QColor(self._color); outer.setAlpha(50)
        p.setBrush(QBrush(outer)); p.setPen(Qt.NoPen)
        p.drawEllipse(0, 0, 14, 14)
        inner = QColor(self._color); inner.setAlpha(self._alpha)
        p.setBrush(QBrush(inner))
        p.drawEllipse(2, 2, 10, 10)


# Keep the alias so old references still compile
WaveformWidget = None  # replaced by VoiceOrbWidget below


class VoiceOrbWidget(QWidget):
    """
    Cinematic teal concentric-rings orb — inspired by the holographic sphere image.
    States
    -------
    idle      : dim slow-breathing rings
    awake     : bright teal, rings start rotating
    listen    : fast rotation + particle stream
    think     : amber glow, slow rotation
    speak     : full brightness, fast spin, inner bloom
    error     : rings fade red — voice unavailable
    """
    STATE_IDLE     = "idle"
    STATE_AWAKE    = "awake"
    STATE_LISTEN   = "listen"
    STATE_THINKING = "think"
    STATE_SPEAKING = "speak"
    STATE_ERROR    = "error"

    # (ring_color, glow_color, speed_factor, brightness)
    _STATE_CFG = {
        STATE_IDLE:     ("#0d6b6b", "#0a3d3d", 0.4,  0.35),
        STATE_AWAKE:    ("#00e5cc", "#006655", 1.0,  0.75),
        STATE_LISTEN:   ("#22d3ea", "#0891b2", 1.8,  0.90),
        STATE_THINKING: ("#f59e0b", "#78350f", 0.7,  0.60),
        STATE_SPEAKING: ("#00ffe0", "#00756b", 2.5,  1.00),
        STATE_ERROR:    ("#6b1a1a", "#3d0a0a", 0.3,  0.25),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(180, 180)
        self._state      = self.STATE_IDLE
        self._t          = 0.0          # animation clock
        self._brightness = 0.35         # lerped toward target
        self._speed      = 0.4
        self._ring_color = QColor("#0d6b6b")
        self._glow_color = QColor("#0a3d3d")
        # Particles: (angle_offset, radial_frac, angular_speed, size)
        rng = random.Random(42)
        self._particles = [
            [rng.uniform(0, 360), rng.uniform(0.45, 0.75),
             rng.uniform(0.3, 1.2) * (1 if rng.random() > 0.5 else -1),
             rng.uniform(1.5, 3.5)]
            for _ in range(28)
        ]
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(16)   # ~60 fps

    # ── public API (same as old WaveformWidget) ─────────────
    def set_active(self, a: bool):
        self._state = self.STATE_AWAKE if a else self.STATE_IDLE

    def set_voice_state(self, state: str):
        self._state = state

    # ── animation ───────────────────────────────────────────
    def _tick(self):
        cfg = self._STATE_CFG.get(self._state, self._STATE_CFG[self.STATE_IDLE])
        rc, gc, spd, tgt_bright = cfg
        self._speed      = spd
        self._ring_color = QColor(rc)
        self._glow_color = QColor(gc)
        # smooth brightness
        self._brightness += (tgt_bright - self._brightness) * 0.06
        self._t += 0.016 * spd
        # update particles
        for pt in self._particles:
            pt[0] += pt[2] * spd * 0.5
        self.update()

    # ── painting ─────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        base_r = min(w, h) * 0.42
        t      = self._t
        bright = self._brightness
        rc     = self._ring_color
        gc     = self._glow_color

        # ── background glow ──
        bg_grad = QRadialGradient(cx, cy, base_r * 1.1)
        gc_inner = QColor(gc); gc_inner.setAlpha(int(110 * bright))
        gc_outer = QColor(gc); gc_outer.setAlpha(0)
        bg_grad.setColorAt(0.0, gc_inner)
        bg_grad.setColorAt(1.0, gc_outer)
        p.setBrush(QBrush(bg_grad)); p.setPen(Qt.NoPen)
        p.drawEllipse(int(cx - base_r * 1.1), int(cy - base_r * 1.1),
                      int(base_r * 2.2), int(base_r * 2.2))

        # ── concentric oval rings (5 rings, tilted like the image) ──
        ring_defs = [
            # (rx_frac, ry_frac, tilt_deg, phase_offset, alpha_frac)
            (1.00, 0.28, 0,   0.0, 0.80),
            (0.92, 0.36, 15,  0.4, 0.70),
            (0.80, 0.44, -10, 0.8, 0.60),
            (0.65, 0.52, 20,  1.2, 0.50),
            (0.48, 0.58, -5,  1.6, 0.40),
        ]
        for rx_f, ry_f, tilt, phase, alpha_f in ring_defs:
            rx = base_r * rx_f
            ry = base_r * ry_f
            # breathing: vary ry slightly
            breathe = 1.0 + 0.06 * math.sin(t * 1.8 + phase)
            ry *= breathe
            # rotation wobble on rx
            rx_actual = rx * (1.0 + 0.03 * math.cos(t + phase))

            ring_c = QColor(rc)
            ring_c.setAlpha(int(255 * alpha_f * bright))

            p.save()
            p.translate(cx, cy)
            p.rotate(tilt + t * 18 * self._speed * (1 if phase < 1 else -1))
            pen = QPen(ring_c, 1.5 + bright * 1.2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(int(-rx_actual), int(-ry), int(rx_actual * 2), int(ry * 2))
            p.restore()

        # ── inner sphere glow ──
        sphere_r = base_r * 0.22
        sph_grad = QRadialGradient(cx, cy, sphere_r)
        c_center = QColor(rc); c_center.setAlpha(int(220 * bright))
        c_mid    = QColor(rc); c_mid.setAlpha(int(80 * bright))
        c_edge   = QColor(rc); c_edge.setAlpha(0)
        sph_grad.setColorAt(0.0, c_center)
        sph_grad.setColorAt(0.5, c_mid)
        sph_grad.setColorAt(1.0, c_edge)
        p.setBrush(QBrush(sph_grad)); p.setPen(Qt.NoPen)
        p.drawEllipse(int(cx - sphere_r), int(cy - sphere_r),
                      int(sphere_r * 2), int(sphere_r * 2))

        # ── flower-of-life inner grid (tiny hexagon dots at centre) ──
        if bright > 0.45:
            dot_r = 1.2
            dot_c = QColor(rc); dot_c.setAlpha(int(160 * bright))
            p.setBrush(QBrush(dot_c)); p.setPen(Qt.NoPen)
            hex_pts = [(0,0)] + [(8*math.cos(a*math.pi/3), 8*math.sin(a*math.pi/3))
                                  for a in range(6)]
            for dx, dy in hex_pts:
                p.drawEllipse(int(cx+dx-dot_r), int(cy+dy-dot_r),
                              int(dot_r*2), int(dot_r*2))

        # ── floating particles ──
        for pt in self._particles:
            ang = math.radians(pt[0])
            rf  = pt[1]
            psize = pt[3]
            jitter = 0.04 * math.sin(t * 2.2 + ang)
            px = cx + base_r * (rf + jitter) * math.cos(ang)
            py = cy + base_r * (rf + jitter) * 0.35 * math.sin(ang)
            pc = QColor(rc)
            dist_fade = 1.0 - abs(rf - 0.6) * 1.5
            pc.setAlpha(max(0, int(200 * bright * dist_fade)))
            p.setBrush(QBrush(pc)); p.setPen(Qt.NoPen)
            p.drawEllipse(int(px - psize/2), int(py - psize/2),
                          int(psize), int(psize))

        # ── outer pulse ring (speaking / awake) ──
        if self._state in (self.STATE_SPEAKING, self.STATE_AWAKE, self.STATE_LISTEN):
            pulse_r  = base_r * (1.05 + 0.10 * math.sin(t * 3.5))
            pulse_c  = QColor(rc)
            pulse_c.setAlpha(int(80 * bright * abs(math.sin(t * 3.5))))
            p.setPen(QPen(pulse_c, 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(int(cx - pulse_r), int(cy - pulse_r * 0.30),
                          int(pulse_r * 2), int(pulse_r * 0.60))

        # ── error overlay — dim red X shimmer ──
        if self._state == self.STATE_ERROR:
            err_c = QColor("#ef4444"); err_c.setAlpha(int(100 * bright + 40))
            p.setPen(QPen(err_c, 2))
            p.setBrush(Qt.NoBrush)
            sz = base_r * 0.25
            p.drawLine(int(cx - sz), int(cy - sz), int(cx + sz), int(cy + sz))
            p.drawLine(int(cx + sz), int(cy - sz), int(cx - sz), int(cy + sz))


class StatCard(QFrame):
    def __init__(self, title, value, unit="", icon="", color=ACCENT, parent=None):
        super().__init__(parent)
        self._color = color
        self.setObjectName("statcard")
        self.setFixedHeight(100)
        self.setStyleSheet(f"""
            QFrame#statcard {{
                background: {CARD};
                border: 1px solid {BORDER};
                border-radius: 12px;
            }}
        """)
        self.setGraphicsEffect(_shadow(20, color, 60, 0, 3))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(4)
        head = QHBoxLayout()
        icon_lbl = QLabel(icon); icon_lbl.setFont(_font(18))
        head.addWidget(icon_lbl); head.addStretch()
        t = QLabel(title.upper()); t.setFont(_font(9, QFont.Bold))
        t.setStyleSheet(f"color:{DIMTEXT}; letter-spacing:1px;")
        head.addWidget(t); lay.addLayout(head)
        val_lay = QHBoxLayout()
        self.val_lbl = QLabel(value)
        self.val_lbl.setFont(_font(26, QFont.Bold))
        self.val_lbl.setStyleSheet(f"color:{color};")
        val_lay.addWidget(self.val_lbl)
        u = QLabel(unit); u.setFont(_font(11))
        u.setStyleSheet(f"color:{DIMTEXT};"); u.setAlignment(Qt.AlignBottom)
        val_lay.addWidget(u); val_lay.addStretch(); lay.addLayout(val_lay)

    def set_value(self, v): self.val_lbl.setText(str(v))


class GestureChip(QFrame):
    def __init__(self, name, action, emoji="🤚", category_color=ACCENT, parent=None):
        super().__init__(parent)
        self.setFixedHeight(68)
        self.setObjectName("gchip")
        self.setStyleSheet(f"""
            QFrame#gchip {{
                background: {SURFACE2};
                border: 1px solid {BORDER};
                border-left: 3px solid {category_color};
                border-radius: 10px;
            }}
            QFrame#gchip:hover {{
                background: #242a3d;
                border-color: {category_color};
            }}
        """)
        lay = QHBoxLayout(self); lay.setContentsMargins(10, 8, 10, 8)
        emo = QLabel(emoji); emo.setFont(_font(22)); emo.setFixedWidth(36)
        emo.setAlignment(Qt.AlignCenter); lay.addWidget(emo)
        info = QVBoxLayout(); info.setSpacing(1)
        n = QLabel(name); n.setFont(_font(11, QFont.Bold))
        n.setStyleSheet(f"color:{TEXT};"); info.addWidget(n)
        a = QLabel(action); a.setFont(_font(9))
        a.setStyleSheet(f"color:{DIMTEXT};"); info.addWidget(a)
        lay.addLayout(info); lay.addStretch()


class LogEntry(QLabel):
    def __init__(self, msg, level="info", parent=None):
        super().__init__(parent)
        colors  = {"info": TEXT, "ok": GREEN, "warn": AMBER, "err": RED}
        badges  = {"info": "●", "ok":  "✓",  "warn": "⚠",   "err": "✗"}
        color   = colors.get(level, TEXT)
        badge   = badges.get(level, "●")
        ts = time.strftime("%H:%M:%S")
        self.setText(f'<span style="color:{DIMTEXT};">[{ts}]</span> '
                     f'<span style="color:{color};">{badge} {msg}</span>')
        self.setFont(_font(10))
        self.setTextFormat(Qt.RichText)
        self.setWordWrap(True)


class ToggleButton(QPushButton):
    def __init__(self, label, active=True, color=GREEN, parent=None):
        super().__init__(label, parent)
        self._active = active; self._color = color
        self.setCheckable(True); self.setChecked(active)
        self.setFixedHeight(36); self._refresh()
        self.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked):
        self._active = checked; self._refresh()

    def _refresh(self):
        col = self._color if self._active else DIMTEXT
        self.setStyleSheet(f"""
            QPushButton {{
                background: {'rgba(34,211,165,15)' if self._active else SURFACE2};
                color: {col};
                border: 1px solid {col};
                border-radius: 8px;
                font-size: 11px;
                font-weight: bold;
                padding: 0 16px;
            }}
            QPushButton:hover {{ background: rgba(108,99,255,30); }}
        """)


# ══════════════════════════════════════════════════════════
# HAND CAPTURE THREAD
# ══════════════════════════════════════════════════════════
class HandCaptureThread(QThread):
    frame_ready      = pyqtSignal(object, object)
    camera_error     = pyqtSignal(str)
    two_hand_gesture = pyqtSignal(str, float)  # for two-hand combos

    def __init__(self, camera_id=0, cursor_enabled=True):
        # NOTE: NO parent=self to prevent Qt from destroying thread with widget
        super().__init__(None)
        self._camera_id      = camera_id
        self._cursor_enabled = cursor_enabled
        self._running        = False
        self._display_paused = False   # pause UI frame painting only; gesture signals always flow
        self._engine         = None
        self._fail_count     = 0
        self._last_frame_t   = 0.0    # watchdog: track when last good frame arrived
        # Keep thread alive — will be stopped explicitly
        self.setObjectName("hgvcs-cam-thread")

    def set_cursor_enabled(self, enabled: bool):
        self._cursor_enabled = enabled
        if self._engine:
            self._engine.set_cursor_enabled(enabled)

    def set_paused(self, paused: bool):
        """Pause/resume UI frame PAINTING without blocking gesture signals."""
        self._display_paused = paused

    # Kept for back-compat with any external callers
    @property
    def _paused(self):
        return self._display_paused

    def seconds_since_last_frame(self) -> float:
        """How long (seconds) since the last successfully processed frame."""
        if self._last_frame_t == 0.0:
            return 0.0
        return time.time() - self._last_frame_t

    def run(self):
        self._running = True
        try:
            from src.gesture.hand_engine import HandEngine
        except Exception as e:
            self.camera_error.emit(f"HandEngine import error: {e}")
            return

        def _open_cap():
            cap = cv2.VideoCapture(self._camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self._camera_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS,          30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize buffer lag
            return cap

        self._engine = HandEngine(
            max_hands=2,
            detection_conf=0.72,
            tracking_conf=0.65,
            cursor_smoothing=0.18,
            cursor_enabled=self._cursor_enabled,
            hold_frames=8,
        )

        # Outer loop: retry camera FOREVER until stop() is called
        while self._running:
            cap = _open_cap()
            if not cap.isOpened():
                self.camera_error.emit(f"Cannot open camera #{self._camera_id} — retrying…")
                self.msleep(2000)
                continue   # retry indefinitely

            self._fail_count  = 0
            self._last_frame_t = time.time()
            self.camera_error.emit(f"Camera #{self._camera_id} opened")

            # ── Inner loop: always read + process every frame ──────────────────
            # IMPORTANT: We ALWAYS call engine.process() so MediaPipe's internal
            # timestamp counter advances monotonically.  If the display is paused
            # we just skip painting — but gesture signals still flow.
            while self._running:
                ret, frame = cap.read()
                if ret:
                    self._fail_count  = 0
                    self._last_frame_t = time.time()
                    result = self._engine.process(frame)

                    # Gesture signals are ALWAYS emitted regardless of display pause
                    self.frame_ready.emit(
                        result.annotated_frame if not self._display_paused else None,
                        result
                    )
                    # Emit two-hand gesture if detected
                    if result.two_hand_gesture:
                        self.two_hand_gesture.emit(
                            result.two_hand_gesture,
                            result.confidence
                        )
                else:
                    self._fail_count += 1
                    self.msleep(30)
                    if self._fail_count >= 20:   # ~0.6 s of failures → reconnect
                        self._fail_count = 0
                        self.camera_error.emit(
                            f"Camera #{self._camera_id} lost — reconnecting…")
                        cap.release()
                        self.msleep(1500)
                        break   # break inner to retry outer

            cap.release()

        if self._engine:
            try:
                self._engine.close()
            except Exception:
                pass

    def stop(self):
        self._running = False
        self.wait(5000)   # wait up to 5s for clean exit


# ══════════════════════════════════════════════════════════
# CAMERA WIDGET
# ══════════════════════════════════════════════════════════
class CameraWidget(QWidget):
    status_changed     = pyqtSignal(str, str)
    gesture_detected   = pyqtSignal(str, float)
    two_hand_detected  = pyqtSignal(str, float)

    def __init__(self, camera_id=0, cursor_enabled=True, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 200)
        self._pixmap      = None
        self._fps         = 0.0
        self._frame_count = 0
        self._fps_timer   = time.time()
        self._error_msg   = None
        self._cam_on      = False
        self._t           = 0.0

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_tick)

        if _CV2_OK:
            # No parent=self — prevents Qt from garbage-collecting the thread
            # when the widget is re-parented or resized
            self._thread = HandCaptureThread(camera_id, cursor_enabled)
            self._thread.frame_ready.connect(self._on_frame)
            self._thread.camera_error.connect(self._on_error)
            self._thread.two_hand_gesture.connect(self._on_two_hand)
            self._thread.start()
        else:
            self._error_msg = "opencv-python not installed"
            self._anim_timer.start(33)

    def _on_two_hand(self, combo: str, conf: float):
        """Forward two-hand combo gesture to dashboard."""
        self.two_hand_detected.emit(combo, conf)

    def _on_frame(self, bgr, gesture_result=None):
        # bgr may be None when display is paused — still process gesture signals
        if bgr is not None:
            self._frame_count += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                self._fps = self._frame_count / (now - self._fps_timer)
                self._frame_count = 0
                self._fps_timer   = now

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(bytes(rgb.data), w, h, ch * w, QImage.Format_RGB888)
            self._pixmap = QPixmap.fromImage(img)

            if not self._cam_on:
                self._cam_on = True
                self._anim_timer.stop()
                self.status_changed.emit("Camera active — hand tracking ON", "ok")

            self.update()

        # Always forward confirmed gesture signals, even when display is paused
        if gesture_result is not None and gesture_result.name not in ("none", "unknown", ""):
            if gesture_result.confirmed:
                self.gesture_detected.emit(gesture_result.name, gesture_result.confidence)

    def _on_error(self, msg):
        self._error_msg = msg
        self._cam_on    = False
        self._anim_timer.start(33)
        self.status_changed.emit(msg, "err")
        self.update()

    def _anim_tick(self):
        self._t += 0.05; self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        if self._pixmap and self._cam_on:
            scaled = self._pixmap.scaled(w, h, Qt.KeepAspectRatioByExpanding,
                                         Qt.SmoothTransformation)
            dx = (scaled.width()  - w) // 2
            dy = (scaled.height() - h) // 2
            p.drawPixmap(0, 0, scaled, dx, dy, w, h)
            vig = QRadialGradient(w/2, h/2, max(w,h)*0.65)
            vig.setColorAt(0.5, QColor(0,0,0,0))
            vig.setColorAt(1.0, QColor(0,0,0,100))
            p.fillRect(0, 0, w, h, vig)
            fps_text = f"FPS  {self._fps:.0f}"
            p.setFont(_font(9, QFont.Bold))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(fps_text) + 14
            chip = QRect(w-tw-10, 10, tw, 22)
            p.fillRect(chip, QColor(0,0,0,140))
            p.setPen(QColor(GREEN))
            p.drawText(chip, Qt.AlignCenter, fps_text)
        else:
            p.fillRect(0, 0, w, h, QColor("#080b11"))
            grid_pen = QPen(QColor("#1a2035"), 1)
            p.setPen(grid_pen)
            for gx in range(0, w, 24): p.drawLine(gx, 0, gx, h)
            for gy in range(0, h, 24): p.drawLine(0, gy, w, gy)
            sy = int((math.sin(self._t)*0.5+0.5)*(h-4))
            grad = QLinearGradient(0, sy-12, 0, sy+12)
            grad.setColorAt(0, QColor(0,0,0,0))
            sc = QColor(ACCENT); sc.setAlpha(180)
            grad.setColorAt(0.5, sc); grad.setColorAt(1, QColor(0,0,0,0))
            p.fillRect(0, sy-12, w, 24, grad)
            p.setFont(_font(32)); p.setPen(QColor(ACCENT2))
            p.drawText(QRect(0, h//2-50, w, 50), Qt.AlignCenter, "✋")
            err_line = self._error_msg or "Starting camera…"
            p.setFont(_font(10))
            p.setPen(QColor(RED if self._error_msg else DIMTEXT))
            p.drawText(QRect(0, h//2+4, w, 24), Qt.AlignCenter, err_line)

        bracket_pen = QPen(QColor(ACCENT), 2)
        p.setPen(bracket_pen)
        bs = 22
        for cx, cy, fx, fy in [
            (10, 10, 1, 1), (w-10, 10, -1, 1),
            (10, h-10, 1, -1), (w-10, h-10, -1, -1)
        ]:
            p.drawLine(cx, cy, cx+fx*bs, cy)
            p.drawLine(cx, cy, cx, cy+fy*bs)

    def closeEvent(self, event):
        if _CV2_OK and hasattr(self, '_thread'):
            self._thread.stop()
        super().closeEvent(event)

    def stop_camera(self):
        """Pause display only — camera + gesture thread keeps running."""
        if _CV2_OK and hasattr(self, '_thread'):
            self._thread.set_paused(True)
            # Don't set _cam_on=False here: gesture signals still flow

    def resume_camera(self):
        """Resume display painting after stop_camera()."""
        if _CV2_OK and hasattr(self, '_thread'):
            self._thread.set_paused(False)
            if not self._cam_on:
                self._cam_on = True
                self._anim_timer.stop()

    def is_thread_alive(self) -> bool:
        """True if the capture thread is still running."""
        return hasattr(self, '_thread') and self._thread.isRunning()

    def seconds_since_last_frame(self) -> float:
        """Seconds since the last good camera frame (for watchdog)."""
        if hasattr(self, '_thread'):
            return self._thread.seconds_since_last_frame()
        return 0.0


# ══════════════════════════════════════════════════════════
# NETWORK TAB
# ══════════════════════════════════════════════════════════
class NetworkTab(QWidget):
    """LAN peer list, file transfer progress, and manual-send controls."""

    request_send = pyqtSignal(str)   # filepath

    def __init__(self, network_manager=None, parent=None):
        super().__init__(parent)
        self._mgr = network_manager
        self._build()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_peers)
        self._refresh_timer.start(3000)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        hdr = QLabel("🌐  LAN File Sharing")
        hdr.setFont(_font(15, QFont.Bold))
        hdr.setStyleSheet(f"color:{TEXT};")
        root.addWidget(hdr)

        desc = QLabel(
            "Peers on the same network are discovered automatically via mDNS.\n"
            "Use  ✊ Hold  to grab a file,  ✋ Release  to send it,  "
            "or click the Send button below."
        )
        desc.setFont(_font(10))
        desc.setStyleSheet(f"color:{DIMTEXT};")
        desc.setWordWrap(True)
        root.addWidget(desc)

        # ── peers ──
        peer_frame = QFrame()
        peer_frame.setStyleSheet(f"""
            QFrame {{ background:{CARD}; border:1px solid {BORDER};
                     border-radius:12px; }}
        """)
        pf_lay = QVBoxLayout(peer_frame)
        pf_lay.setContentsMargins(14, 10, 14, 10)
        pt = QLabel("📡  Discovered Peers")
        pt.setFont(_font(11, QFont.Bold))
        pt.setStyleSheet(f"color:{ACCENT2};")
        pf_lay.addWidget(pt)

        self._peer_list = QListWidget()
        self._peer_list.setFixedHeight(120)
        self._peer_list.setStyleSheet(f"""
            QListWidget {{
                background: {SURFACE2};
                border: 1px solid {BORDER};
                border-radius: 8px;
                color: {TEXT};
                font-size: 10px;
            }}
            QListWidget::item:selected {{ background: rgba(108,99,255,40); }}
            QListWidget::item:hover    {{ background: rgba(255,255,255,5); }}
        """)
        pf_lay.addWidget(self._peer_list)
        root.addWidget(peer_frame)

        # ── send controls ──
        send_frame = QFrame()
        send_frame.setStyleSheet(f"""
            QFrame {{ background:{CARD}; border:1px solid {BORDER};
                     border-radius:12px; }}
        """)
        sf_lay = QVBoxLayout(send_frame)
        sf_lay.setContentsMargins(14, 10, 14, 14)
        st = QLabel("📤  Send File")
        st.setFont(_font(11, QFont.Bold))
        st.setStyleSheet(f"color:{ACCENT2};")
        sf_lay.addWidget(st)

        self._file_label = QLabel("No file selected")
        self._file_label.setFont(_font(10))
        self._file_label.setStyleSheet(f"color:{DIMTEXT};")
        sf_lay.addWidget(self._file_label)

        btn_row = QHBoxLayout()
        browse_btn = QPushButton("📁  Browse File")
        browse_btn.setFixedHeight(34)
        browse_btn.setStyleSheet(self._btn_style(ACCENT))
        browse_btn.clicked.connect(self._browse_file)
        btn_row.addWidget(browse_btn)

        self._send_btn = QPushButton("🚀  Send to Peer")
        self._send_btn.setFixedHeight(34)
        self._send_btn.setStyleSheet(self._btn_style(GREEN))
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._do_send)
        btn_row.addWidget(self._send_btn)
        sf_lay.addLayout(btn_row)
        root.addWidget(send_frame)

        # ── transfer status ──
        prog_frame = QFrame()
        prog_frame.setStyleSheet(f"""
            QFrame {{ background:{CARD}; border:1px solid {BORDER};
                     border-radius:12px; }}
        """)
        pg_lay = QVBoxLayout(prog_frame)
        pg_lay.setContentsMargins(14, 10, 14, 14)
        pg_t = QLabel("📊  Transfer Progress")
        pg_t.setFont(_font(11, QFont.Bold))
        pg_t.setStyleSheet(f"color:{ACCENT2};")
        pg_lay.addWidget(pg_t)

        self._prog_label = QLabel("No active transfer")
        self._prog_label.setFont(_font(10))
        self._prog_label.setStyleSheet(f"color:{DIMTEXT};")
        pg_lay.addWidget(self._prog_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(14)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {SURFACE2};
                border: 1px solid {BORDER};
                border-radius: 7px;
                text-align: center;
                font-size: 9px;
                color: {TEXT};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT}, stop:1 {GREEN});
                border-radius: 7px;
            }}
        """)
        pg_lay.addWidget(self._progress)
        root.addWidget(prog_frame)

        # ── transfer log ──
        log_frame = QFrame()
        log_frame.setStyleSheet(f"""
            QFrame {{ background:{CARD}; border:1px solid {BORDER};
                     border-radius:12px; }}
        """)
        lg_lay = QVBoxLayout(log_frame)
        lg_lay.setContentsMargins(14, 10, 14, 10)
        lg_t = QLabel("🗂  Transfer Log")
        lg_t.setFont(_font(11, QFont.Bold))
        lg_t.setStyleSheet(f"color:{ACCENT2};")
        lg_lay.addWidget(lg_t)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedHeight(140)
        scroll.setStyleSheet("""
            QScrollArea { border:none; background:transparent; }
            QScrollBar:vertical { background:#1a1f2e; width:4px; border-radius:2px; }
            QScrollBar::handle:vertical { background:#2a3245; border-radius:2px; }
        """)
        self._log_w   = QWidget(); self._log_w.setStyleSheet("background:transparent;")
        self._log_lay = QVBoxLayout(self._log_w)
        self._log_lay.setContentsMargins(0, 4, 0, 4)
        self._log_lay.setSpacing(3)
        self._log_lay.addStretch()
        scroll.setWidget(self._log_w)
        lg_lay.addWidget(scroll)
        root.addWidget(log_frame, 1)

        self._selected_file: str = ""

    # ── helpers ────────────────────────────────────────────
    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{
                background: rgba(0,0,0,0);
                color: {color};
                border: 1px solid {color};
                border-radius: 8px;
                font-size: 11px;
                font-weight: bold;
                padding: 0 14px;
            }}
            QPushButton:hover {{ background: rgba(108,99,255,25); }}
            QPushButton:disabled {{ color: {DIMTEXT}; border-color: {DIMTEXT}; }}
        """

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File to Send", "",
                                               "All Files (*.*)")
        if path:
            self._selected_file = path
            name = os.path.basename(path)
            size = os.path.getsize(path)
            self._file_label.setText(
                f"📄 {name}  ({self._human_size(size)})"
            )
            self._file_label.setStyleSheet(f"color:{TEXT};")
            self._send_btn.setEnabled(True)

    def _do_send(self):
        if not self._selected_file:
            return
        if self._mgr is None:
            self._log_entry("Network manager not available", "err")
            return
        peer = self._mgr.registry.first()
        if peer is None:
            self._log_entry("No peers found on LAN", "warn")
            return
        self._log_entry(
            f"Sending {os.path.basename(self._selected_file)} → {peer['name']}…",
            "info"
        )
        self._mgr.send_file(self._selected_file, peer)

    def _refresh_peers(self):
        if self._mgr is None:
            return
        peers = self._mgr.registry.all()
        self._peer_list.clear()
        if not peers:
            item = QListWidgetItem("  No peers found – make sure HGVCS is running on other devices")
            item.setForeground(QColor(DIMTEXT))
            self._peer_list.addItem(item)
        else:
            for p in peers:
                item = QListWidgetItem(f"  💻  {p['name']}   ({p['ip']}:{p['port']})")
                item.setForeground(QColor(GREEN))
                self._peer_list.addItem(item)

    def on_progress(self, filename: str, done: int, total: int, direction: str):
        pct = int(done / total * 100) if total > 0 else 0
        arrow = "↑ Sending" if direction == "tx" else "↓ Receiving"
        self._prog_label.setText(
            f"{arrow}  {os.path.basename(filename)}  — "
            f"{self._human_size(done)} / {self._human_size(total)}"
        )
        self._prog_label.setStyleSheet(
            f"color:{GREEN if direction == 'tx' else AMBER};"
        )
        self._progress.setValue(pct)

    def on_transfer_done(self, filename: str, success: bool, direction: str):
        if success:
            self._log_entry(
                f"{'Sent' if direction == 'tx' else 'Received'}  {filename}", "ok"
            )
        else:
            self._log_entry(f"Transfer failed: {filename}", "err")
        self._progress.setValue(0)
        self._prog_label.setText("No active transfer")
        self._prog_label.setStyleSheet(f"color:{DIMTEXT};")

    def on_incoming_request(self, filename: str, size: int, ip: str) -> bool:
        reply = QMessageBox.question(
            self,
            "Incoming File Transfer",
            f"Accept file from {ip}?\n\n"
            f"  File:  {filename}\n"
            f"  Size:  {self._human_size(size)}\n\n"
            f"File will be saved to:  ~/Downloads/HGVCS/",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        accepted = reply == QMessageBox.Yes
        level = "ok" if accepted else "warn"
        self._log_entry(
            f"{'Accepted' if accepted else 'Rejected'} {filename} from {ip}",
            level
        )
        return accepted


    def _log_entry(self, msg: str, level: str = "info"):
        entry = LogEntry(msg, level)
        self._log_lay.insertWidget(self._log_lay.count()-1, entry)
        while self._log_lay.count() > 20:
            item = self._log_lay.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    @staticmethod
    def _human_size(n: int) -> str:
        for unit in ("B","KB","MB","GB"):
            if n < 1024: return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"


# ══════════════════════════════════════════════════════════
# DASHBOARD TAB
# ══════════════════════════════════════════════════════════
class DashboardTab(QWidget):
    def __init__(self, config, gesture_controller=None,
                 system_controller=None, network_manager=None,
                 switch_to_network=None, parent=None):
        super().__init__(parent)
        self.config               = config
        self._gesture_ctrl        = gesture_controller
        self._sys_ctrl            = system_controller
        self._net_mgr             = network_manager
        self._switch_to_network   = switch_to_network
        self._voice_ctrl          = None   # set later via set_voice_controller()
        self._macro_log: list     = []  # recent macro fires
        self._build()
        self._sim_timer = QTimer(self)
        self._sim_timer.timeout.connect(self._simulate)
        self._sim_timer.start(1000)

        # ── Gesture watchdog: restart camera if frozen for > 8 s ──────────
        self._watchdog_timer = QTimer(self)
        self._watchdog_timer.timeout.connect(self._gesture_watchdog)
        self._watchdog_timer.start(4000)   # check every 4 s

    def set_voice_controller(self, vc):
        """Inject voice controller so the Listen button can trigger it."""
        self._voice_ctrl = vc

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(16)

        # ─ STAT CARDS ─
        cards_row = QHBoxLayout(); cards_row.setSpacing(12)
        self._cards = {
            "gestures": StatCard("Gestures Today", "0",  "",  "✋",  ACCENT),
            "commands": StatCard("Voice Commands",  "0",  "",  "🎙️", GREEN),
            "accuracy": StatCard("Accuracy",        "—",  "%", "🎯",  AMBER),
            "uptime":   StatCard("Uptime",          "00:00", "", "⏱️", ACCENT2),
        }
        for c in self._cards.values():
            cards_row.addWidget(c)
        root.addLayout(cards_row)

        # ─ MIDDLE SPLIT ─
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background: " + BORDER + "; }")
        splitter.setChildrenCollapsible(False)   # prevent either panel from collapsing to 0

        # camera
        cam_frame = QFrame()
        cam_frame.setStyleSheet(f"""
            QFrame {{ background:{CARD}; border:1px solid {BORDER};
                     border-radius:12px; }}
        """)
        cam_lay = QVBoxLayout(cam_frame)
        cam_lay.setContentsMargins(0, 0, 0, 0)
        cam_title = QLabel("  📷  Live Camera")
        cam_title.setFont(_font(11, QFont.Bold))
        cam_title.setStyleSheet(f"color:{TEXT}; padding:10px 14px 0px;")
        cam_lay.addWidget(cam_title)

        cam_id = 0
        if isinstance(self.config, dict):
            cam_id = self.config.get("gesture", {}).get("camera_id", 0)
        self._camera = CameraWidget(cam_id, cursor_enabled=True)
        self._camera.setMinimumWidth(300)
        self._camera.setMinimumHeight(200)
        self._camera.status_changed.connect(self._on_cam_status)
        self._camera.gesture_detected.connect(self._on_gesture_detected)
        self._camera.two_hand_detected.connect(self._on_two_hand_detected)
        cam_lay.addWidget(self._camera, 1)

        # toast overlaid on camera
        self._toast = ToastOverlay(self._camera)
        self._toast.setGeometry(0, 0, 700, 80)
        self._toast.setAttribute(Qt.WA_TransparentForMouseEvents)

        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(12, 6, 12, 10)
        self._cam_dot = PulseDot(GREEN)
        status_bar.addWidget(self._cam_dot)
        self._cam_status = QLabel("Opening camera…")
        self._cam_status.setFont(_font(10))
        self._cam_status.setStyleSheet(f"color:{DIMTEXT};")
        status_bar.addWidget(self._cam_status)
        status_bar.addStretch()
        cam_lay.addLayout(status_bar)
        splitter.addWidget(cam_frame)

        # right panel
        right     = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(12)

        # ── Voice Assistant card with orb ──────────────────
        vcard = QFrame()
        vcard.setStyleSheet(f"""
            QFrame {{ background:{CARD}; border:1px solid {BORDER};
                     border-radius:14px; }}
        """)
        vlay = QVBoxLayout(vcard)
        vlay.setContentsMargins(12, 10, 12, 10)
        vlay.setSpacing(6)

        # Title row
        vtitle_row = QHBoxLayout()
        self._mic_dot = PulseDot("#00e5cc")
        vtitle_row.addWidget(self._mic_dot)
        vt = QLabel("V  Voice Assistant")
        vt.setFont(_font(11, QFont.Bold))
        vt.setStyleSheet(f"color:{TEXT};")
        vtitle_row.addWidget(vt)
        vtitle_row.addStretch()
        self._voice_status = QLabel("Listening for Hey V")
        self._voice_status.setFont(_font(9))
        self._voice_status.setStyleSheet(f"color:{DIMTEXT};")
        vtitle_row.addWidget(self._voice_status)
        vlay.addLayout(vtitle_row)

        # Orb + transcript side by side
        orb_row = QHBoxLayout()
        orb_row.setSpacing(10)
        self._wave = VoiceOrbWidget()          # keep name _wave for compat
        self._wave.setFixedSize(180, 180)
        orb_row.addWidget(self._wave, 0, Qt.AlignVCenter)

        right_info = QVBoxLayout()
        right_info.setSpacing(8)
        right_info.addStretch()

        # State badge
        self._voice_badge = QLabel("IDLE")
        self._voice_badge.setFont(_font(9, QFont.Bold))
        self._voice_badge.setAlignment(Qt.AlignCenter)
        self._voice_badge.setFixedHeight(22)
        self._voice_badge.setStyleSheet(
            f"color:#00e5cc; background:rgba(0,229,204,12);"
            f"border:1px solid #00e5cc; border-radius:11px; padding:0 8px;"
        )
        right_info.addWidget(self._voice_badge)

        # Transcript
        self._voice_transcript = QLabel("")
        self._voice_transcript.setFont(_font(9))
        self._voice_transcript.setStyleSheet(
            f"color:{ACCENT2}; font-style:italic;")
        self._voice_transcript.setWordWrap(True)
        self._voice_transcript.setMaximumWidth(160)
        right_info.addWidget(self._voice_transcript)
        right_info.addStretch()

        # ── Manual Listen button ────────────────────────────
        self._listen_btn = QPushButton("🎙  Listen Now")
        self._listen_btn.setFixedHeight(32)
        self._listen_btn.setCursor(Qt.PointingHandCursor)
        self._listen_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 rgba(0,229,204,25), stop:1 rgba(108,99,255,25));
                color: #00e5cc;
                border: 1px solid #00e5cc;
                border-radius: 10px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 rgba(0,229,204,55), stop:1 rgba(108,99,255,55));
                border-color: #22d3a5;
            }}
            QPushButton:pressed {{
                background: rgba(0,229,204,80);
            }}
        """)
        self._listen_btn.clicked.connect(self._on_manual_listen)
        right_info.addWidget(self._listen_btn)

        orb_row.addLayout(right_info, 1)
        vlay.addLayout(orb_row)
        right_lay.addWidget(vcard)

        log_frame = QFrame()
        log_frame.setStyleSheet(f"""
            QFrame {{ background:{CARD}; border:1px solid {BORDER};
                     border-radius:12px; }}
        """)
        log_lay = QVBoxLayout(log_frame); log_lay.setContentsMargins(14, 10, 14, 10)
        log_t = QLabel("🗒  Activity Log"); log_t.setFont(_font(11, QFont.Bold))
        log_t.setStyleSheet(f"color:{TEXT};"); log_lay.addWidget(log_t)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border:none; background:transparent; }
            QScrollBar:vertical { background:#1a1f2e; width:4px; border-radius:2px; }
            QScrollBar::handle:vertical { background:#2a3245; border-radius:2px; }
        """)
        self._log_container = QWidget()
        self._log_container.setStyleSheet("background:transparent;")
        self._log_layout = QVBoxLayout(self._log_container)
        self._log_layout.setContentsMargins(0, 4, 0, 4); self._log_layout.setSpacing(4)
        self._log_layout.addStretch()
        scroll.setWidget(self._log_container); log_lay.addWidget(scroll)
        right_lay.addWidget(log_frame, 1)
        splitter.addWidget(right)
        splitter.setSizes([560, 340])   # 60% camera, 40% right panel — both always visible
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        # ─ CONTROLS ROW ─
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        self._btn_gesture = ToggleButton("✋  Gesture ON", True, GREEN)
        self._btn_voice   = ToggleButton("🎤️  Voice ON",   True, AMBER)
        self._btn_network = ToggleButton("🌐  Network ON", True, ACCENT)

        # Wire toggle buttons to actually enable/disable the subsystems
        self._btn_gesture.toggled.connect(self._toggle_gesture)
        self._btn_voice.toggled.connect(self._toggle_voice)

        for b in [self._btn_gesture, self._btn_voice, self._btn_network]:
            ctrl.addWidget(b)

        # ── Manual Restart Gesture Camera button ─────────────────
        self._restart_cam_btn = QPushButton("🔄  Restart Camera")
        self._restart_cam_btn.setFixedHeight(36)
        self._restart_cam_btn.setCursor(Qt.PointingHandCursor)
        self._restart_cam_btn.setToolTip(
            "Force-restart the gesture camera thread — use if gesture control stops responding")
        self._restart_cam_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(34,211,165,12);
                color: {GREEN};
                border: 1px solid {GREEN};
                border-radius: 8px;
                font-size: 11px; font-weight: bold;
                padding: 0 14px;
            }}
            QPushButton:hover  {{ background: rgba(34,211,165,30); }}
            QPushButton:pressed {{ background: rgba(34,211,165,55); }}
        """)
        self._restart_cam_btn.clicked.connect(self._restart_gesture_camera)
        ctrl.addWidget(self._restart_cam_btn)

        # Mode switcher
        ctrl.addSpacing(12)
        mode_lbl = QLabel("Mode:")
        mode_lbl.setFont(_font(10))
        mode_lbl.setStyleSheet(f"color:{DIMTEXT};")
        ctrl.addWidget(mode_lbl)

        self._mode_btns = {}
        for mode_name, mode_key, color in [
            ("🖥  Normal",       "normal",       ACCENT),
            ("🎮  Game",         "game",         GREEN),
            ("📊  Presentation", "presentation", AMBER),
        ]:
            b = QPushButton(mode_name)
            b.setFixedHeight(32)
            b.setCheckable(True)
            b.setChecked(mode_key == "normal")
            b.setStyleSheet(self._mode_btn_style(color, mode_key == "normal"))
            b.clicked.connect(lambda checked, k=mode_key, c=color, btn=b: self._set_mode(k, c))
            self._mode_btns[mode_key] = (b, color)
            ctrl.addWidget(b)

        ctrl.addStretch()
        stop_btn = QPushButton("⏹  Emergency Stop")
        stop_btn.setFixedHeight(36)
        stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(239,68,68,15); color: {RED};
                border: 1px solid {RED}; border-radius: 8px;
                font-size: 11px; font-weight: bold; padding: 0 20px;
            }}
            QPushButton:hover {{ background: rgba(239,68,68,35); }}
        """)
        stop_btn.clicked.connect(self._emergency_stop)
        ctrl.addWidget(stop_btn)
        root.addLayout(ctrl)

        self._start_time = time.time()
        self._gesture_count = 0; self._cmd_count = 0
        self._add_log("System started successfully", "ok")
        self._add_log("Camera module initialised", "info")
        self._add_log("Hey V ready — always listening (say 'sleep' to pause)", "ok")
        self._add_log("16 daily-use gestures active with hold-guard", "ok")


    @staticmethod
    def _mode_btn_style(color: str, active: bool) -> str:
        bg = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},20)" if active else "transparent"
        return f"""
            QPushButton {{
                background: {bg}; color: {color if active else DIMTEXT};
                border: 1px solid {color if active else BORDER};
                border-radius: 8px; font-size: 10px;
                font-weight: {'bold' if active else 'normal'};
                padding: 0 10px;
            }}
            QPushButton:hover {{ background: rgba(108,99,255,20); color:{TEXT}; }}
        """

    def _set_mode(self, mode_key: str, color: str):
        """Switch system mode and update button styles."""
        for k, (btn, c) in self._mode_btns.items():
            btn.setChecked(k == mode_key)
            btn.setStyleSheet(self._mode_btn_style(c, k == mode_key))
        if self._gesture_ctrl:
            self._gesture_ctrl.set_mode(mode_key)
        self._add_log(f"Mode → {mode_key.title()}", "info")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_toast') and hasattr(self, '_camera'):
            self._toast.setGeometry(0, 0, self._camera.width(), 80)

    # ── gesture names displayed in log / status bar ──────
    _GESTURE_NAMES = {
        # Swipe/scroll have UNIQUE shapes — no conflict with other gestures
        "pointing":      "Pointing ☝️  →  [Swipe L/R]: snap wrist  |  Cursor Mode",
        "peace_sign":    "Peace Sign ✌️  →  [Scroll]: move wrist Up / Down",
        "pinch":         "Pinch  →  Click",
        "open_palm":     "Open Palm  →  Pause ⏸",
        "closed_fist":   "Closed Fist  →  Confirm ✅",
        "thumbs_up":     "Thumbs Up  →  Accept 👍",
        "thumbs_down":   "Thumbs Down  →  Reject 👎",
        "three_fingers": "Three Fingers  →  Volume Up 🔊",
        "four_fingers":  "Four Fingers  →  Volume Down 🔉",
        "ok_sign":       "OK Sign 👌  →  Screenshot 📸",
        "phone_sign":    "Phone Sign 🤙  →  Settings ⚙️",
        "swipe_left":    "👈 Swipe Left  →  Previous",
        "swipe_right":   "👉 Swipe Right  →  Next",
        "swipe_up":      "👆 Scroll Up  →  Scroll Up ↑",
        "swipe_down":    "👇 Scroll Down  →  Scroll Down ↓",
        "circular_cw":   "Circular CW  →  Refresh 🔄",
        "circular_ccw":  "Circular CCW  →  Undo 🔃",
        "pinch_in":      "Pinch In  →  Zoom Out 🔍-",
        "pinch_out":     "Pinch Out  →  Zoom In 🔍+",
        "wave":          "Wave  →  Cancel 👋",
        "rock_on":       "Rock On  →  Media Toggle 🎵",
    }

    def _on_gesture_detected(self, name: str, confidence: float,
                              two_hand: str = ""):
        """Called only when hold-guard is confirmed (gesture is stable)."""
        self._gesture_count += 1
        self._cards["gestures"].set_value(str(self._gesture_count))
        self._cards["accuracy"].set_value(f"{confidence*100:.0f}")
        label = self._GESTURE_NAMES.get(name, name.replace("_"," ").title())
        self._cam_status.setText(f"✋  {label}")
        self._cam_status.setStyleSheet(f"color:{GREEN};")
        self._add_log(f"Gesture: {label}  ({confidence*100:.0f}%)", "ok")
        self._toast.show_gesture(name)

        # two-hand overlay label
        if two_hand:
            self._add_log(f"🤲  Two-hand: {two_hand.replace('+', ' + ')}", "ok")

        # dispatch to GestureController (confirmed=True since hold-guard passed)
        if self._gesture_ctrl:
            self._gesture_ctrl.on_gesture(name, confidence, two_hand, confirmed=True)


    def _on_two_hand_detected(self, name: str, confidence: float):
        """
        Called when a two-hand semantic gesture is detected (zoom in/out).
        Pass 'name' as both gesture_name and two_hand so that
        GestureController routes it through execute_two_hand().
        """
        # Log it visibly
        label = name.replace("_", " ").title()
        self._add_log(f"\U0001f932  Two-hand: {label}  ({confidence*100:.0f}%)", "ok")
        self._gesture_count += 1
        self._cards["gestures"].set_value(str(self._gesture_count))
        # Dispatch through gesture controller with two_hand=name
        # so execute_two_hand(name, "") finds it in the combo table
        if self._gesture_ctrl:
            self._gesture_ctrl.on_gesture(
                "none",       # single-hand gesture (ignored)
                confidence,
                name,         # two_hand = the semantic combo name
                confirmed=True
            )

    def _on_cam_status(self, message, level):
        color_map = {"ok": GREEN, "warn": AMBER, "err": RED}
        color = color_map.get(level, DIMTEXT)
        self._cam_status.setText(message)
        self._cam_status.setStyleSheet(f"color:{color};")
        self._cam_dot.set_color(color)
        self._add_log(f"Camera: {message}", level)

    def _add_log(self, msg, level="info"):
        entry = LogEntry(msg, level)
        self._log_layout.insertWidget(self._log_layout.count()-1, entry)
        while self._log_layout.count() > 32:
            item = self._log_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    def _emergency_stop(self):
        """Emergency stop: disables gesture ACTIONS (OS commands) but keeps
        the camera + MediaPipe pipeline running so it can instantly resume.
        Does NOT pause the camera thread — that would freeze MediaPipe timestamps."""
        if self._gesture_ctrl:
            self._gesture_ctrl.stop()   # block OS actions
        self._add_log("⛔  Gesture actions paused (camera still tracking)", "warn")
        # Auto-resume gesture controller in 5 s
        QTimer.singleShot(5000, self._resume_gesture_actions)

    def _resume_gesture_actions(self):
        if self._gesture_ctrl:
            self._gesture_ctrl.start()
        self._add_log("✅  Gesture actions resumed", "ok")

    def _on_manual_listen(self):
        """Manual Listen button — bypass wake word and go straight to listening."""
        if self._voice_ctrl:
            self._voice_ctrl.manual_listen()
            self._add_log("🎙️  Manual listen triggered", "info")
            self.set_voice_state("awake")
        else:
            self._add_log("⚠️  Voice controller not available", "warn")

    def _toggle_gesture(self, enabled: bool):
        """Enable or disable gesture OS actions via the toggle button.
        The camera + MediaPipe pipeline is NEVER stopped — only the dispatch
        of OS actions is blocked so MediaPipe timestamps stay healthy."""
        if self._gesture_ctrl:
            if enabled:
                self._gesture_ctrl.start()
                self._add_log("✋  Gesture recognition ENABLED", "ok")
            else:
                self._gesture_ctrl.stop()
                self._add_log("✋  Gesture recognition DISABLED (camera still running)", "warn")
        # Always keep camera display running to maintain MediaPipe health
        self._camera.resume_camera()

    def _toggle_voice(self, enabled: bool):
        """Enable or disable voice assistant via the toggle button."""
        if self._voice_ctrl:
            if enabled:
                self._voice_ctrl.start()
                self._add_log("🎙️  Voice assistant ENABLED", "ok")
            else:
                self._voice_ctrl.stop()
                self._add_log("🎙️  Voice assistant DISABLED", "warn")

    def _restart_gesture_camera(self):
        """Force-restart the camera + HandEngine (full hard reset)."""
        self._add_log("🔄  Restarting gesture camera…", "info")
        # Stop the existing thread cleanly
        if hasattr(self._camera, '_thread') and self._camera._thread.isRunning():
            self._camera._thread.stop()
        # Recreate and start a fresh HandCaptureThread
        cam_id = 0
        if isinstance(self.config, dict):
            cam_id = self.config.get("gesture", {}).get("camera_id", 0)
        new_thread = HandCaptureThread(cam_id, cursor_enabled=True)
        new_thread.frame_ready.connect(self._camera._on_frame)
        new_thread.camera_error.connect(self._camera._on_error)
        new_thread.two_hand_gesture.connect(self._camera._on_two_hand)
        self._camera._thread = new_thread
        self._camera._cam_on = False
        new_thread.start()
        # Ensure gesture controller is running
        if self._gesture_ctrl:
            self._gesture_ctrl.start()
        self._btn_gesture.setChecked(True)
        self._add_log("✅  Gesture camera hard-restarted", "ok")

    def _gesture_watchdog(self):
        """Auto-recover if the gesture camera thread freezes."""
        if not hasattr(self._camera, '_thread'):
            return
        # Thread died entirely → restart
        if not self._camera.is_thread_alive():
            self._add_log("⚠️  Gesture thread died — auto-restarting…", "warn")
            self._restart_gesture_camera()
            return
        # Thread alive but no frames for > 8 s → reconnect
        stale = self._camera.seconds_since_last_frame()
        if stale > 8.0:
            self._add_log(
                f"⚠️  No gesture frames for {stale:.0f}s — reconnecting camera…", "warn")
            # Force inner loop to break by releasing and retrying
            if hasattr(self._camera, '_thread'):
                self._camera._thread._fail_count = 25   # trigger reconnect branch

    def _simulate(self):
        elapsed = int(time.time() - self._start_time)
        mins, secs = divmod(elapsed, 60)
        hrs, mins  = divmod(mins, 60)
        self._cards["uptime"].set_value(f"{hrs:02d}:{mins:02d}:{secs:02d}")

    # ── Voice state callbacks (called from VoiceController) ──
    def set_voice_state(self, state: str):
        """Update the voice waveform and status chip. Thread-safe."""
        QMetaObject.invokeMethod(self, "_update_voice_state",
                                 Qt.QueuedConnection, Q_ARG(str, state))

    @pyqtSlot(str)
    def _update_voice_state(self, state: str):
        # Map external state strings → VoiceOrbWidget states + UI strings
        _ORB_TEAL   = "#00e5cc"
        _ORB_PURPLE = "#a78bfa"
        _ORB_AMBER  = "#f59e0b"
        _ORB_RED    = "#ef4444"
        state_cfg = {
            "idle":      ("Listening for Hey V",   _ORB_TEAL,   VoiceOrbWidget.STATE_IDLE,     "IDLE"),
            "awake":     ("Hey V — speak now 🎙️",  _ORB_TEAL,   VoiceOrbWidget.STATE_AWAKE,    "AWAKE"),
            "recording": ("Listening…",             _ORB_PURPLE, VoiceOrbWidget.STATE_LISTEN,   "LISTENING"),
            "thinking":  ("Processing…",            _ORB_AMBER,  VoiceOrbWidget.STATE_THINKING,  "THINKING"),
            "speaking":  ("V is speaking 🔊",        _ORB_TEAL,   VoiceOrbWidget.STATE_SPEAKING, "SPEAKING"),
            "error":     ("Voice unavailable ✗",    _ORB_RED,    VoiceOrbWidget.STATE_ERROR,    "ERROR"),
        }
        label, color, orb_state, badge = state_cfg.get(
            state, ("Processing…", _ORB_AMBER, VoiceOrbWidget.STATE_THINKING, "…"))
        self._voice_status.setText(label)
        self._voice_status.setStyleSheet(f"color:{color};")
        self._mic_dot.set_color(color)
        self._wave.set_voice_state(orb_state)
        # Update badge
        badge_style = {
            "IDLE":      f"color:#0d6b6b; background:rgba(13,107,107,12); border:1px solid #0d6b6b;",
            "AWAKE":     f"color:#00e5cc; background:rgba(0,229,204,15); border:1px solid #00e5cc;",
            "LISTENING": f"color:#a78bfa; background:rgba(167,139,250,15); border:1px solid #a78bfa;",
            "THINKING":  f"color:#f59e0b; background:rgba(245,158,11,15); border:1px solid #f59e0b;",
            "SPEAKING":  f"color:#00ffe0; background:rgba(0,255,224,20); border:1px solid #00ffe0;",
            "ERROR":     f"color:#ef4444; background:rgba(239,68,68,15); border:1px solid #ef4444;",
        }.get(badge, f"color:{color}; background:transparent; border:1px solid {color};")
        self._voice_badge.setText(badge)
        self._voice_badge.setStyleSheet(
            badge_style + " border-radius:11px; padding:0 8px; font-size:9px; font-weight:bold;")

    def set_voice_transcript(self, heard: str, reply: str):
        """Show what was heard and what V replied."""
        QMetaObject.invokeMethod(self, "_show_transcript",
                                 Qt.QueuedConnection,
                                 Q_ARG(str, heard), Q_ARG(str, reply))

    @pyqtSlot(str, str)
    def _show_transcript(self, heard: str, reply: str):
        self._voice_transcript.setText(f"You: {heard[:40]}…" if len(heard)>40 else f"You: {heard}")
        self._cmd_count += 1
        self._cards["commands"].set_value(str(self._cmd_count))
        self._add_log(f'Voice: "{heard[:55]}"', "info")
        if reply:
            self._add_log(f'V: "{reply[:55]}"', "ok")
            # Show voice toast
            self._toast.show_gesture("__voice__")


# ══════════════════════════════════════════════════════════
# GESTURE GUIDE TAB
# ══════════════════════════════════════════════════════════
GESTURE_DATA = [
    # ─── Hold-guard gestures (stable 8+ frames to fire) ─────────────────────
    ("✋",  "Open Palm",       "Pause / Play media",                ACCENT),
    ("✊",  "Closed Fist",     "Confirm / Enter",                   GREEN),
    ("👍",  "Thumbs Up",       "Accept / Yes (Enter)",              GREEN),
    ("👎",  "Thumbs Down",     "Reject / No (Escape)",              RED),
    ("🖖",  "Three Fingers",   "Volume Up ×3",                      GREEN),
    ("🖐️",  "Four Fingers",    "Volume Down ×3",                    RED),
    ("👌",  "OK Sign",         "Screenshot  (was Peace Sign)",      AMBER),
    ("🤙",  "Phone Sign",      "Open Settings  (was OK Sign)",      ACCENT2),
    # ─── Motion gestures ─────────────────────────────────────────────────────
    # SWIPE: point ONE finger, snap wrist LEFT or RIGHT quickly
    ("☝️",  "Pointing + Snap →", "Swipe Right  »  Next / Forward",  ACCENT),
    ("☝️",  "Pointing + Snap ←", "Swipe Left   «  Back / Previous", ACCENT),
    # SCROLL: make V-sign, move wrist UP or DOWN
    ("✌️",  "Peace Sign + ↑",  "Scroll Up (wrist up)",              ACCENT2),
    ("✌️",  "Peace Sign + ↓",  "Scroll Down (wrist down)",          ACCENT2),
    # Cursor: just hold pointing and move your finger
    ("☝️",  "Pointing (still)","Cursor control mode",               ACCENT2),
    # Other motions
    ("🤏",  "Pinch",           "Left click",                        GREEN),
    ("↻",   "Circular CW",    "Refresh (Ctrl+R)",                   GREEN),
    ("↺",   "Circular CCW",   "Undo (Ctrl+Z)",                      AMBER),
    ("👋",  "Wave",            "Cancel (Escape)",                   RED),
    # ─── Two-hand gestures ───────────────────────────────────────────────────
    ("🤲",  "Both Open Palms", "Show Desktop (Win+D)",              ACCENT),
    ("🙌",  "Both Pointing",   "Reset Zoom (Ctrl+0)",               GREEN),
    ("🤲",  "Spread Apart",    "Zoom In (two-hand)",                GREEN),
    ("🤲",  "Pinch Together", "Zoom Out (two-hand)",                AMBER),
    # ─── Voice ───────────────────────────────────────────────────────────────
    ("🎙️",  "Hey V",           "Wake word — say a command",         ACCENT),
]


class GestureGuideTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(16)
        hdr = QLabel("✋  Gesture Reference Guide")
        hdr.setFont(_font(15, QFont.Bold))
        hdr.setStyleSheet(f"color:{TEXT};")
        root.addWidget(hdr)
        desc = QLabel(
            "20 gesture commands — unique hand shapes prevent conflicts.\n"
            "Hold-guard: static gestures must be stable for ~0.3 s before firing.\n"
            "Scroll: Three Fingers + move up/down.  "
            "Swipe: Open Palm + move left/right."
        )
        desc.setFont(_font(10))
        desc.setStyleSheet(f"color:{DIMTEXT};")
        root.addWidget(desc)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border:none; background:transparent; }
            QScrollBar:vertical { background:#1a1f2e; width:5px; border-radius:3px; }
            QScrollBar::handle:vertical { background:#2a3245; border-radius:3px; }
        """)
        inner = QWidget(); inner.setStyleSheet("background:transparent;")
        grid  = QGridLayout(inner); grid.setSpacing(10); grid.setContentsMargins(0,0,0,0)
        for i, (emoji, name, action, cat_color) in enumerate(GESTURE_DATA):
            chip = GestureChip(name, action, emoji, cat_color)
            grid.addWidget(chip, i//2, i%2)
        scroll.setWidget(inner); root.addWidget(scroll, 1)


# ══════════════════════════════════════════════════════════
# SETTINGS TAB
# ══════════════════════════════════════════════════════════
class SettingsTab(QWidget):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16); root.setSpacing(16)
        hdr = QLabel("⚙️  Settings")
        hdr.setFont(_font(15, QFont.Bold)); hdr.setStyleSheet(f"color:{TEXT};")
        root.addWidget(hdr)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border:none; background:transparent; }
            QScrollBar:vertical { background:#1a1f2e; width:5px; }
            QScrollBar::handle:vertical { background:#2a3245; border-radius:3px; }
        """)
        inner = QWidget(); inner.setStyleSheet("background:transparent;")
        lay   = QVBoxLayout(inner); lay.setSpacing(12); lay.setContentsMargins(0,0,0,0)
        sections = [
            ("✋ Gesture Settings", [
                ("Detection Confidence", "0.70", "Minimum confidence to accept a gesture"),
                ("Cooldown Period",      "0.5 s","Seconds between consecutive gestures"),
                ("Camera Resolution",   "1280×720","Camera capture resolution"),
                ("Max Hands",           "2",    "Maximum number of hands to track"),
            ]),
            ("💻 System Actions", [
                ("Swipe L/R",      "Ctrl+Win+Arrow","Switch virtual desktops"),
                ("Swipe Up/Down",  "scroll(±5)",   "Mouse wheel scroll"),
                ("Circular CW",    "Ctrl+R",        "Refresh"),
                ("Circular CCW",   "Ctrl+Z",        "Undo"),
                ("Pinch In/Out",   "Ctrl+- / Ctrl+=","Zoom"),
                ("Peace Sign",     "Screenshot",    "Saved to Desktop/HGVCS-screenshots"),
            ]),
            ("🌐 Network Settings", [
                ("TCP Port",       "9876",         "LAN file transfer port"),
                ("Save Folder",    "~/Downloads/HGVCS","Where received files are saved"),
                ("Chunk Size",     "64 KB",         "Transfer chunk size"),
                ("Discovery",      "mDNS / zeroconf","Peer discovery protocol"),
            ]),
            ("🎙️ Voice Settings", [
                ("Whisper Model",  "base",         "Model size: tiny / base / small"),
                ("Wake Word",      "Hey System",   "Trigger phrase"),
                ("Silence Timeout","2.0 s",        "Stop listening after silence"),
            ]),
        ]
        for section_title, items in sections:
            grp = QFrame()
            grp.setStyleSheet(f"""
                QFrame {{ background:{CARD}; border:1px solid {BORDER};
                         border-radius:12px; }}
            """)
            g = QVBoxLayout(grp); g.setContentsMargins(16,12,16,12); g.setSpacing(8)
            sec_lbl = QLabel(section_title); sec_lbl.setFont(_font(12, QFont.Bold))
            sec_lbl.setStyleSheet(f"color:{ACCENT2};"); g.addWidget(sec_lbl)
            sep = QFrame(); sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(f"background:{BORDER};"); sep.setFixedHeight(1)
            g.addWidget(sep)
            for name, value, tip in items:
                row = QHBoxLayout()
                lbl = QLabel(name); lbl.setFont(_font(10))
                lbl.setStyleSheet(f"color:{TEXT};"); lbl.setFixedWidth(200)
                row.addWidget(lbl)
                val = QLabel(value); val.setFont(_font(10, QFont.Bold))
                val.setStyleSheet(f"color:{ACCENT};"); row.addWidget(val)
                row.addStretch()
                tip_lbl = QLabel(tip); tip_lbl.setFont(_font(9))
                tip_lbl.setStyleSheet(f"color:{DIMTEXT};"); row.addWidget(tip_lbl)
                g.addLayout(row)
            lay.addWidget(grp)
        lay.addStretch()
        scroll.setWidget(inner); root.addWidget(scroll, 1)


# ══════════════════════════════════════════════════════════
# ANALYTICS TAB
# ══════════════════════════════════════════════════════════
class BarChart(QWidget):
    """Mini horizontal bar chart for gesture frequency."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: dict = {}   # label → count
        self.setMinimumHeight(200)

    def update_data(self, data: dict):
        self._data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True)[:12])
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(SURFACE2))

        if not self._data:
            p.setFont(_font(10)); p.setPen(QColor(DIMTEXT))
            p.drawText(QRect(0, 0, w, h), Qt.AlignCenter, "No data yet — perform gestures!")
            return

        max_val = max(self._data.values()) or 1
        n       = len(self._data)
        row_h   = max(16, (h - 10) // n)
        grad_cols = [ACCENT, GREEN, AMBER, ACCENT2, RED]

        for i, (label, count) in enumerate(self._data.items()):
            y      = i * row_h + 4
            bar_w  = int((count / max_val) * (w - 130))
            color  = QColor(grad_cols[i % len(grad_cols)])

            # bar
            bar_rect = QRect(120, y+2, max(bar_w, 4), row_h - 6)
            grad = QLinearGradient(bar_rect.left(), 0, bar_rect.right(), 0)
            c2 = QColor(color); c2.setAlpha(60)
            grad.setColorAt(0, color); grad.setColorAt(1, c2)
            p.setBrush(QBrush(grad)); p.setPen(Qt.NoPen)
            p.drawRoundedRect(bar_rect, 3, 3)

            # label
            p.setFont(_font(9)); p.setPen(QColor(TEXT))
            p.drawText(QRect(4, y, 114, row_h), Qt.AlignVCenter | Qt.AlignRight,
                       label.replace("_", " "))

            # count
            p.setFont(_font(9, QFont.Bold)); p.setPen(QColor(color))
            p.drawText(QRect(bar_rect.right() + 6, y, 40, row_h),
                       Qt.AlignVCenter, str(count))


class AnalyticsTab(QWidget):
    """Analytics & insights tab — shows gesture stats, session info, macros."""

    def __init__(self, profile_manager=None, macro_engine=None, parent=None):
        super().__init__(parent)
        self._pm    = profile_manager
        self._me    = macro_engine
        self._build()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.start(2000)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        hdr = QLabel("📊  Analytics & Insights")
        hdr.setFont(_font(15, QFont.Bold))
        hdr.setStyleSheet(f"color:{TEXT};")
        root.addWidget(hdr)

        # ── session stats ──
        sess_frame = QFrame()
        sess_frame.setStyleSheet(f"""
            QFrame {{ background:{CARD}; border:1px solid {BORDER};
                     border-radius:12px; }}
        """)
        sf = QVBoxLayout(sess_frame); sf.setContentsMargins(14, 10, 14, 12)
        sf_title = QLabel("⏱  Current Session")
        sf_title.setFont(_font(11, QFont.Bold))
        sf_title.setStyleSheet(f"color:{ACCENT2};")
        sf.addWidget(sf_title)
        self._sess_cards_lay = QHBoxLayout()
        self._total_lbl  = self._stat_pill("Total Gestures", "—",  ACCENT)
        self._uniq_lbl   = self._stat_pill("Unique Types",   "—",  GREEN)
        self._top_lbl    = self._stat_pill("Most Used",      "—",  AMBER)
        self._sess_cards_lay.addWidget(self._total_lbl)
        self._sess_cards_lay.addWidget(self._uniq_lbl)
        self._sess_cards_lay.addWidget(self._top_lbl)
        self._sess_cards_lay.addStretch()
        sf.addLayout(self._sess_cards_lay)
        root.addWidget(sess_frame)

        # ── gesture frequency chart ──
        chart_frame = QFrame()
        chart_frame.setStyleSheet(f"""
            QFrame {{ background:{CARD}; border:1px solid {BORDER};
                     border-radius:12px; }}
        """)
        cf = QVBoxLayout(chart_frame); cf.setContentsMargins(14, 10, 14, 10)
        cf_title = QLabel("📈  Gesture Frequency")
        cf_title.setFont(_font(11, QFont.Bold))
        cf_title.setStyleSheet(f"color:{ACCENT2};")
        cf.addWidget(cf_title)
        self._chart = BarChart()
        self._chart.setMinimumHeight(220)
        cf.addWidget(self._chart)
        root.addWidget(chart_frame, 1)

        # ── macro log + data collection ──
        bottom = QHBoxLayout(); bottom.setSpacing(12)

        macro_frame = QFrame()
        macro_frame.setStyleSheet(f"""
            QFrame {{ background:{CARD}; border:1px solid {BORDER};
                     border-radius:12px; }}
        """)
        mf = QVBoxLayout(macro_frame); mf.setContentsMargins(14, 10, 14, 10)
        mf_t = QLabel("⚡  Active Macros")
        mf_t.setFont(_font(11, QFont.Bold))
        mf_t.setStyleSheet(f"color:{ACCENT2};")
        mf.addWidget(mf_t)
        scroll_m = QScrollArea(); scroll_m.setWidgetResizable(True)
        scroll_m.setFixedHeight(140)
        scroll_m.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        self._macro_inner = QWidget()
        self._macro_inner.setStyleSheet("background:transparent;")
        self._macro_lout  = QVBoxLayout(self._macro_inner)
        self._macro_lout.setSpacing(2); self._macro_lout.setContentsMargins(0,0,0,0)
        self._macro_lout.addStretch()
        scroll_m.setWidget(self._macro_inner)
        mf.addWidget(scroll_m)
        bottom.addWidget(macro_frame, 1)

        # data collection launcher
        dc_frame = QFrame()
        dc_frame.setStyleSheet(f"""
            QFrame {{ background:{CARD}; border:1px solid {BORDER};
                     border-radius:12px; }}
        """)
        dc = QVBoxLayout(dc_frame); dc.setContentsMargins(14, 10, 14, 14); dc.setSpacing(10)
        dc_t = QLabel("🎯  Collect Training Data")
        dc_t.setFont(_font(11, QFont.Bold))
        dc_t.setStyleSheet(f"color:{ACCENT2};")
        dc.addWidget(dc_t)
        dc_desc = QLabel(
            "Record gesture samples for ML model training.\n"
            "200 samples per gesture → ~97% accuracy."
        )
        dc_desc.setFont(_font(9))
        dc_desc.setStyleSheet(f"color:{DIMTEXT};")
        dc_desc.setWordWrap(True)
        dc.addWidget(dc_desc)
        dc_btn = QPushButton("📹  Launch Data Collector")
        dc_btn.setFixedHeight(36)
        dc_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(108,99,255,20); color:{ACCENT2};
                border: 1px solid {ACCENT}; border-radius:8px;
                font-size:11px; font-weight:bold;
            }}
            QPushButton:hover {{ background: rgba(108,99,255,40); }}
        """)
        dc_btn.clicked.connect(self._open_data_collector)
        dc.addWidget(dc_btn)

        train_btn = QPushButton("🧠  Train Model (Background)")
        train_btn.setFixedHeight(36)
        train_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(34,211,165,15); color:{GREEN};
                border: 1px solid {GREEN}; border-radius:8px;
                font-size:11px; font-weight:bold;
            }}
            QPushButton:hover {{ background: rgba(34,211,165,30); }}
        """)
        train_btn.clicked.connect(self._launch_training)
        dc.addWidget(train_btn)
        dc.addStretch()
        bottom.addWidget(dc_frame)
        root.addLayout(bottom)

    def _stat_pill(self, label: str, value: str, color: str) -> QFrame:
        f = QFrame()
        f.setFixedSize(140, 60)
        f.setStyleSheet(f"""
            QFrame {{ background:{SURFACE2}; border:1px solid {BORDER};
                     border-radius:10px; }}
        """)
        lay = QVBoxLayout(f); lay.setContentsMargins(10, 6, 10, 6); lay.setSpacing(1)
        v = QLabel(value); v.setFont(_font(18, QFont.Bold))
        v.setStyleSheet(f"color:{color};"); v.setAlignment(Qt.AlignCenter)
        l = QLabel(label.upper()); l.setFont(_font(8))
        l.setStyleSheet(f"color:{DIMTEXT}; letter-spacing:1px;")
        l.setAlignment(Qt.AlignCenter)
        lay.addWidget(v); lay.addWidget(l)
        f._val_lbl = v
        return f

    def _refresh(self):
        """Update chart and stats from profile data."""
        counts = {}

        # try profile manager
        if self._pm:
            try:
                stats  = self._pm.active().stats
                counts = stats.get("gesture_counts", {})
                total  = stats.get("total_gestures", 0)
                self._total_lbl._val_lbl.setText(str(total))
                self._uniq_lbl._val_lbl.setText(str(len(counts)))
                if counts:
                    top = max(counts, key=counts.get)
                    self._top_lbl._val_lbl.setText(top.replace("_", " ").title()[:10])
            except Exception:
                pass

        self._chart.update_data(counts)

        # refresh macro list
        if self._me:
            # clear old
            while self._macro_lout.count() > 1:
                item = self._macro_lout.takeAt(0)
                if item and item.widget():
                    item.widget().deleteLater()
            for m in self._me.all_macros()[:10]:
                row = QLabel(
                    f'<span style="color:{AMBER};">⚡</span> '
                    f'<span style="color:{TEXT};">{" → ".join(m["sequence"])}</span>'
                    f'  <span style="color:{DIMTEXT};">{m["description"]}</span>'
                )
                row.setFont(_font(9))
                row.setTextFormat(Qt.RichText)
                self._macro_lout.insertWidget(self._macro_lout.count()-1, row)

    def log_macro(self, macro_name: str):
        """Called when a macro fires."""
        ts  = time.strftime("%H:%M:%S")
        lbl = QLabel(
            f'<span style="color:{DIMTEXT};">[{ts}]</span> '
            f'<span style="color:{GREEN};">⚡ Macro fired: {macro_name}</span>'
        )
        lbl.setFont(_font(9))
        lbl.setTextFormat(Qt.RichText)
        self._macro_lout.insertWidget(0, lbl)

    def _open_data_collector(self):
        dlg = DataCollectionDialog(self)
        dlg.exec_()

    def _launch_training(self):
        import subprocess, sys
        QMessageBox.information(
            self, "Training Started",
            "Training is running in background.\n\n"
            "Requirements: collect data first with the Data Collector.\n"
            "Output: models/gesture_classifier.tflite\n\n"
            "Check the terminal for progress."
        )
        try:
            subprocess.Popen(
                [sys.executable, "scripts/train_gesture_model.py"],
                cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            )
        except Exception as e:
            QMessageBox.warning(self, "Training Error", str(e))


# ══════════════════════════════════════════════════════════
# DATA COLLECTION DIALOG
# ══════════════════════════════════════════════════════════
class DataCollectionDialog(QDialog):
    """In-app gesture data collection interface."""

    _GESTURES = [
        "open_palm", "closed_fist", "pointing", "peace_sign",
        "thumbs_up", "thumbs_down", "ok_sign", "rock_on",
        "three_fingers", "four_fingers", "swipe_left", "swipe_right",
        "wave", "phone_sign", "pinch",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gesture Data Collector")
        self.setMinimumSize(480, 420)
        self.setStyleSheet(f"""
            QDialog {{ background:{BG}; color:{TEXT}; }}
            QLabel {{ color:{TEXT}; }}
            QListWidget {{ background:{SURFACE2}; border:1px solid {BORDER};
                           border-radius:8px; color:{TEXT}; font-size:11px; }}
            QListWidget::item:selected {{ background:rgba(108,99,255,40); }}
        """)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 20)
        root.setSpacing(14)

        hdr = QLabel("🎯  Collect Training Data")
        hdr.setFont(_font(14, QFont.Bold))
        hdr.setStyleSheet(f"color:{ACCENT2};")
        root.addWidget(hdr)

        desc = QLabel(
            "Select gestures to record. Stand in front of camera and press "
            "SPACE to start recording each gesture (200 samples each)."
        )
        desc.setFont(_font(10))
        desc.setStyleSheet(f"color:{DIMTEXT};")
        desc.setWordWrap(True)
        root.addWidget(desc)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.MultiSelection)
        for g in self._GESTURES:
            item = QListWidgetItem(g.replace("_", " ").title())
            item.setData(Qt.UserRole, g)
            self._list.addItem(item)
        # select all by default
        self._list.selectAll()
        root.addWidget(self._list)

        # samples selector
        samples_row = QHBoxLayout()
        samples_row.addWidget(QLabel("Samples per gesture:"))
        self._samples_btn_group = []
        for n in [50, 100, 200, 300]:
            b = QPushButton(str(n))
            b.setFixedSize(50, 30)
            b.setCheckable(True)
            b.setChecked(n == 200)
            b.setStyleSheet(f"""
                QPushButton {{ background:{SURFACE2}; color:{DIMTEXT};
                              border:1px solid {BORDER}; border-radius:6px; font-size:10px; }}
                QPushButton:checked {{ background:rgba(108,99,255,30); color:{ACCENT2};
                                       border-color:{ACCENT}; font-weight:bold; }}
            """)
            b.clicked.connect(lambda _, btn=b: self._select_samples(btn))
            self._samples_btn_group.append((n, b))
            samples_row.addWidget(b)
        samples_row.addStretch()
        root.addLayout(samples_row)

        self._status = QLabel("Ready to collect data.")
        self._status.setFont(_font(10))
        self._status.setStyleSheet(f"color:{GREEN};")
        root.addWidget(self._status)

        btns = QHBoxLayout()
        start_btn = QPushButton("▶  Start Collection")
        start_btn.setFixedHeight(38)
        start_btn.setStyleSheet(f"""
            QPushButton {{ background:rgba(34,211,165,20); color:{GREEN};
                          border:1px solid {GREEN}; border-radius:8px;
                          font-size:12px; font-weight:bold; }}
            QPushButton:hover {{ background:rgba(34,211,165,35); }}
        """)
        start_btn.clicked.connect(self._start_collection)
        close_btn = QPushButton("✕  Close")
        close_btn.setFixedHeight(38)
        close_btn.setStyleSheet(f"""
            QPushButton {{ background:transparent; color:{DIMTEXT};
                          border:1px solid {BORDER}; border-radius:8px;
                          font-size:12px; }}
            QPushButton:hover {{ color:{TEXT}; border-color:{TEXT}; }}
        """)
        close_btn.clicked.connect(self.accept)
        btns.addWidget(start_btn); btns.addWidget(close_btn)
        root.addLayout(btns)

        self._selected_n = 200

    def _select_samples(self, clicked_btn):
        for n, b in self._samples_btn_group:
            b.setChecked(b == clicked_btn)
            if b == clicked_btn:
                self._selected_n = n

    def _start_collection(self):
        selected = []
        for item in self._list.selectedItems():
            selected.append(item.data(Qt.UserRole))
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select at least one gesture.")
            return

        import subprocess, sys
        gestures_arg = ",".join(selected)
        script = os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts", "collect_gesture_data.py"
        )
        try:
            subprocess.Popen(
                [sys.executable, script, "--gestures", gestures_arg,
                 "--samples", str(self._selected_n)],
                cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            )
            self._status.setText(
                f"✅  Collection started for: {', '.join(selected[:3])}"
                + ("..." if len(selected) > 3 else "")
            )
            self._status.setStyleSheet(f"color:{GREEN};")
        except Exception as e:
            self._status.setText(f"❌  Error: {e}")
            self._status.setStyleSheet(f"color:{RED};")


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
class Sidebar(QWidget):
    tab_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(f"background:{SURFACE}; border-right:1px solid {BORDER};")
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        logo_frame = QFrame(); logo_frame.setFixedHeight(80)
        logo_frame.setStyleSheet(f"background:{SURFACE}; border-bottom:1px solid {BORDER};")
        logo_lay = QVBoxLayout(logo_frame); logo_lay.setAlignment(Qt.AlignCenter)
        logo = QLabel("✋  HGVCS"); logo.setFont(_font(16, QFont.Bold))
        logo.setStyleSheet(f"color:{ACCENT2};"); logo.setAlignment(Qt.AlignCenter)
        sub  = QLabel("v1.0.0"); sub.setFont(_font(8))
        sub.setStyleSheet(f"color:{DIMTEXT};"); sub.setAlignment(Qt.AlignCenter)
        logo_lay.addWidget(logo); logo_lay.addWidget(sub); root.addWidget(logo_frame)

        nav_frame = QWidget()
        nav_lay   = QVBoxLayout(nav_frame)
        nav_lay.setContentsMargins(12,16,12,0); nav_lay.setSpacing(4)
        self._nav_buttons = []
        nav_items = [
            ("🏠", "Dashboard",    0),
            ("✋", "Gestures",     1),
            ("🌐", "Network",      2),
            ("📊", "Analytics",   3),
            ("⚙️", "Settings",     4),
        ]
        for icon, label, idx in nav_items:
            btn = QPushButton(f"  {icon}  {label}")
            btn.setFixedHeight(40); btn.setCheckable(True)
            btn.setFont(_font(11))
            btn.setStyleSheet(self._nav_style(False))
            btn.clicked.connect(lambda checked, i=idx: self._select(i))
            self._nav_buttons.append(btn); nav_lay.addWidget(btn)
        nav_lay.addStretch(); root.addWidget(nav_frame, 1)

        footer = QFrame(); footer.setFixedHeight(70)
        footer.setStyleSheet(f"border-top:1px solid {BORDER};")
        foot_lay = QHBoxLayout(footer); foot_lay.setContentsMargins(14,0,14,0)
        self._sys_dot = PulseDot(GREEN); foot_lay.addWidget(self._sys_dot)
        foot_info = QVBoxLayout(); foot_info.setSpacing(2)
        sys_lbl = QLabel("System Active"); sys_lbl.setFont(_font(10, QFont.Bold))
        sys_lbl.setStyleSheet(f"color:{GREEN};"); foot_info.addWidget(sys_lbl)
        mode_lbl = QLabel("Adaptive mode"); mode_lbl.setFont(_font(9))
        mode_lbl.setStyleSheet(f"color:{DIMTEXT};"); foot_info.addWidget(mode_lbl)
        foot_lay.addLayout(foot_info); root.addWidget(footer)
        self._select(0)

    def _nav_style(self, active):
        if active:
            return f"""
                QPushButton {{
                    background: rgba(108,99,255,20); color: {ACCENT2};
                    border: none; border-radius: 8px;
                    text-align: left; padding-left: 12px; font-weight: bold;
                }}
            """
        return f"""
            QPushButton {{
                background: transparent; color: {DIMTEXT};
                border: none; border-radius: 8px;
                text-align: left; padding-left: 12px;
            }}
            QPushButton:hover {{ background: rgba(255,255,255,5); color:{TEXT}; }}
        """

    def _select(self, idx):
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == idx)
            btn.setStyleSheet(self._nav_style(i == idx))
        self.tab_selected.emit(idx)

    def select(self, idx):
        self._select(idx)


# ══════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self, config=None, gesture_controller=None, voice_controller=None,
                 network_manager=None, fusion_engine=None, system_controller=None,
                 profile_manager=None, macro_engine=None):
        super().__init__()
        self.config             = config or {}
        self.gesture_controller = gesture_controller
        self.voice_controller   = voice_controller
        self.network_manager    = network_manager
        self.fusion_engine      = fusion_engine
        self.system_controller  = system_controller
        self.profile_manager    = profile_manager
        self.macro_engine       = macro_engine

        self._apply_palette()
        self.setWindowTitle("HGVCS – Hand Gesture & Voice Control System")
        self.resize(1150, 720)
        self.setMinimumSize(950, 620)
        self._build_ui()
        self._wire_controllers()

    def _apply_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.Window,          QColor(BG))
        pal.setColor(QPalette.WindowText,      QColor(TEXT))
        pal.setColor(QPalette.Base,            QColor(SURFACE))
        pal.setColor(QPalette.AlternateBase,   QColor(SURFACE2))
        pal.setColor(QPalette.Text,            QColor(TEXT))
        pal.setColor(QPalette.Button,          QColor(SURFACE2))
        pal.setColor(QPalette.ButtonText,      QColor(TEXT))
        pal.setColor(QPalette.Highlight,       QColor(ACCENT))
        pal.setColor(QPalette.HighlightedText, QColor(TEXT))
        self.setPalette(pal); self.setAutoFillBackground(True)

    def _build_ui(self):
        central = QWidget(); central.setStyleSheet(f"background:{BG};")
        self.setCentralWidget(central)
        outer = QHBoxLayout(central); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        self._sidebar = Sidebar()
        outer.addWidget(self._sidebar)

        self._stack = QWidget(); self._stack.setStyleSheet(f"background:{BG};")
        stack_lay = QVBoxLayout(self._stack)
        stack_lay.setContentsMargins(0,0,0,0); stack_lay.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.tabBar().hide()
        self._tabs.setStyleSheet("QTabWidget::pane { border:none; }")

        self._net_tab = NetworkTab(self.network_manager)
        self._analytics_tab = AnalyticsTab(
            profile_manager=self.profile_manager,
            macro_engine=self.macro_engine,
        )

        self._dash_tab  = DashboardTab(
            self.config,
            gesture_controller=self.gesture_controller,
            system_controller=self.system_controller,
            network_manager=self.network_manager,
            switch_to_network=lambda: self._sidebar.select(2),
        )
        self._guide_tab = GestureGuideTab()
        self._set_tab   = SettingsTab(self.config)

        self._tabs.addTab(self._dash_tab,       "Dashboard")
        self._tabs.addTab(self._guide_tab,      "Gestures")
        self._tabs.addTab(self._net_tab,        "Network")
        self._tabs.addTab(self._analytics_tab,  "Analytics")
        self._tabs.addTab(self._set_tab,        "Settings")
        stack_lay.addWidget(self._tabs, 1)

        # status bar
        sb = self.statusBar()
        sb.setStyleSheet(f"""
            QStatusBar {{
                background:{SURFACE}; color:{DIMTEXT};
                border-top:1px solid {BORDER}; font-size:9px;
            }}
        """)
        self._status_dot = PulseDot(GREEN); sb.addWidget(self._status_dot)
        self._status_label = QLabel(
            "  HGVCS Active  |  16 Gestures + Hold-Guard  |  Hey V Ready  |  LAN Ready"
        )
        self._status_label.setFont(_font(9)); sb.addWidget(self._status_label)
        sb.addPermanentWidget(QLabel("  v1.0.0  "))
        outer.addWidget(self._stack, 1)
        self._sidebar.tab_selected.connect(self._tabs.setCurrentIndex)

    def _wire_controllers(self):
        """Connect SystemController, NetworkManager, VoiceController, MacroEngine callbacks to UI."""
        if self.system_controller:
            self.system_controller.set_on_settings(
                lambda: self._sidebar.select(4)   # Settings is now tab 4
            )
            # Macro callback → analytics tab log
            self.system_controller.set_macro_callback(
                self._analytics_tab.log_macro
            )

        if self.network_manager:
            self.network_manager.set_progress_cb(self._net_tab.on_progress)
            self.network_manager.set_transfer_done_cb(self._net_tab.on_transfer_done)
            self.network_manager.set_rx_accept_cb(self._safe_incoming)

        if self.gesture_controller:
            if self.system_controller:
                self.gesture_controller.set_system_controller(self.system_controller)
            if self.network_manager:
                self.gesture_controller.set_network_manager(self.network_manager)

        # Wire voice controller UI callbacks
        if self.voice_controller:
            try:
                if self.system_controller:
                    self.voice_controller.set_system_controller(self.system_controller)
                # Inject VC into dashboard so the Listen button works
                self._dash_tab.set_voice_controller(self.voice_controller)
                # State callback → waveform animation
                self.voice_controller.set_state_callback(
                    self._dash_tab.set_voice_state
                )
                # Transcript callback → log + transcript label
                self.voice_controller.set_transcript_callback(
                    self._dash_tab.set_voice_transcript

                )
            except Exception as e:
                import logging
                logging.getLogger("hgvcs.ui").warning(f"Voice UI wire error: {e}")

    def _safe_incoming(self, filename: str, size: int, ip: str) -> bool:
        """Run incoming-file dialog on Qt thread (NetworkManager is off-thread)."""
        result = [True]
        done   = threading.Event()
        def _ask():
            result[0] = self._net_tab.on_incoming_request(filename, size, ip)
            done.set()
        QTimer.singleShot(0, _ask)
        done.wait(timeout=60)
        return result[0]

    def run_calibration(self):
        pass

"""Replace _detect_swipe with gesture-gated version. Uses unicode ≥ in the comment."""
import sys

path = "src/gesture/hand_engine.py"
src  = open(path, encoding="utf-8").read()

# Locate the function by its start/end markers
start_marker = "    def _detect_swipe(self, now: float) -> Optional[str]:\n"
end_marker   = "    def _detect_wave("

i_start = src.find(start_marker)
i_end   = src.find(end_marker)

if i_start == -1 or i_end == -1:
    print(f"Markers not found: start={i_start} end={i_end}")
    sys.exit(1)

OLD = src[i_start:i_end]

NEW = (
    "    def _detect_swipe(self, now: float, gesture: str = \"\") -> Optional[str]:\n"
    "        \"\"\"\n"
    "        Gesture-GATED motion detection: scroll and swipe CANNOT conflict.\n"
    "\n"
    "          SCROLL  (swipe_up / swipe_down)\n"
    "            Trigger: THREE_FINGERS hand shape + slow vertical wrist motion.\n"
    "            Action : pyautogui.scroll  (mouse wheel up/down)\n"
    "\n"
    "          SWIPE L/R (swipe_left / swipe_right)\n"
    "            Trigger: OPEN_PALM or FOUR_FINGERS + fast horizontal motion.\n"
    "            Action : Left/Right arrow key  (PPT slide / image / PDF nav)\n"
    "\n"
    "          FALLBACK (any other hand shape)\n"
    "            Very strict thresholds to prevent accidental firing.\n"
    "        \"\"\"\n"
    "        recent = [(x, y, t) for x, y, t in self._wrist_hist\n"
    "                  if now - t < 0.55]\n"
    "        if len(recent) < 5:\n"
    "            return None\n"
    "\n"
    "        dx = recent[-1][0] - recent[0][0]\n"
    "        dy = recent[-1][1] - recent[0][1]\n"
    "        dt = recent[-1][2] - recent[0][2]\n"
    "        if dt < 0.01:\n"
    "            return None\n"
    "\n"
    "        vx = dx / dt\n"
    "        vy = dy / dt\n"
    "\n"
    "        # SCROLL: three_fingers + vertical wrist motion\n"
    "        if gesture == \"three_fingers\":\n"
    "            if abs(dy) >= 0.13 and abs(vy) >= 0.20 and abs(dy) > abs(dx) * 1.6:\n"
    "                self._wrist_hist.clear()\n"
    "                return \"swipe_down\" if dy > 0 else \"swipe_up\"\n"
    "\n"
    "        # SWIPE L/R: open_palm / four_fingers + horizontal motion\n"
    "        elif gesture in (\"open_palm\", \"four_fingers\"):\n"
    "            if abs(dx) >= 0.18 and abs(vx) >= 0.30 and abs(dx) > abs(dy) * 2.0:\n"
    "                self._wrist_hist.clear()\n"
    "                return \"swipe_right\" if dx > 0 else \"swipe_left\"\n"
    "\n"
    "        # FALLBACK: all other hand shapes -- very strict to avoid false fires\n"
    "        else:\n"
    "            if abs(dx) >= 0.23 and abs(vx) >= 0.42 and abs(dx) > abs(dy) * 2.5:\n"
    "                self._wrist_hist.clear()\n"
    "                return \"swipe_right\" if dx > 0 else \"swipe_left\"\n"
    "            if abs(dy) >= 0.23 and abs(vy) >= 0.42 and abs(dy) > abs(dx) * 2.5:\n"
    "                self._wrist_hist.clear()\n"
    "                return \"swipe_down\" if dy > 0 else \"swipe_up\"\n"
    "\n"
    "        return None\n"
    "\n"
)

out = src[:i_start] + NEW + src[i_end:]
open(path, "w", encoding="utf-8").write(out)
print("OK: gesture-gated _detect_swipe written successfully.")

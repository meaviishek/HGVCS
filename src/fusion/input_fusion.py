"""
InputFusionEngine – stub implementation.

Coordinates gesture and voice inputs into unified system actions.
Currently a lightweight placeholder; extend as the fusion layer matures.
"""

import logging

log = logging.getLogger("hgvcs.fusion")


class InputFusionEngine:
    """
    Lightweight fusion coordinator.

    Receives processed gesture and voice events from their respective
    controllers via the EventBus and can arbitrate conflicts or
    combine multi-modal inputs into single system actions.

    Currently a pass-through stub; no active fusion logic is applied.
    """

    def __init__(self, config: dict, event_bus):
        self.config    = config
        self.event_bus = event_bus
        self._mode     = config.get("mode", "combined")
        self._enabled  = True
        log.info(f"InputFusionEngine ready (mode={self._mode})")

    # ── mode control ──────────────────────────────────────
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        """Switch fusion mode: 'combined', 'gesture_only', 'voice_only'."""
        self._mode = mode
        log.info(f"FusionEngine mode → {mode}")

    # ── lifecycle ─────────────────────────────────────────
    def start(self):
        self._enabled = True
        log.info("InputFusionEngine started")

    def stop(self):
        self._enabled = False
        log.info("InputFusionEngine stopped")

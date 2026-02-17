"""
Emergency vehicle preemption manager.

Handles the lifecycle of a preemption event:
  1. Detect the emergency vehicle (simulated here, real: siren/strobe)
  2. Safely transition to all-red
  3. Give green to the emergency vehicle's approach
  4. Hold until the vehicle clears
  5. Resume normal cycling

Multiple simultaneous preemption requests are queued and served
in FIFO order (real-world spec: priority-based, but FIFO is
reasonable for a first implementation).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

from traffic_controller.config import Direction
from traffic_controller.models.signal import SignalController

logger = logging.getLogger(__name__)


@dataclass
class PreemptionEvent:
    """A single preemption request."""
    direction: Direction
    requested_at: float = field(default_factory=time.monotonic)
    activated_at: float | None = None
    cleared_at: float | None = None
    min_hold_s: float = 10.0   # Minimum time to hold green for EV

    @property
    def is_active(self) -> bool:
        return self.activated_at is not None and self.cleared_at is None

    @property
    def hold_elapsed_s(self) -> float:
        if self.activated_at is None:
            return 0.0
        return time.monotonic() - self.activated_at


@dataclass
class PreemptionManager:
    """
    Manages emergency vehicle preemption for the intersection.

    Usage:
      - Call request(direction) when an EV is detected.
      - Call tick() every controller tick to manage lifecycle.
      - Call clear() when the EV has passed (manual or auto).
    """

    signal_controller: SignalController
    max_hold_s: float = 30.0  # Maximum preemption hold before auto-clear

    # --- state ---
    pending: deque[PreemptionEvent] = field(default_factory=deque)
    active_event: PreemptionEvent | None = None
    history: list[PreemptionEvent] = field(default_factory=list)

    def request(self, direction: Direction, min_hold_s: float = 10.0) -> None:
        """
        Request preemption for an emergency vehicle approaching from `direction`.

        If a preemption is already active, the request is queued.
        """
        event = PreemptionEvent(direction=direction, min_hold_s=min_hold_s)

        if self.active_event is not None:
            logger.warning(
                "Preemption already active for %s — queueing request for %s",
                self.active_event.direction.value, direction.value,
            )
            self.pending.append(event)
        else:
            self._activate(event)

    def tick(self) -> None:
        """
        Called every controller tick.

        Checks if the active preemption should auto-clear (timeout)
        and activates the next queued event if needed.
        """
        if self.active_event is None:
            # Check for pending requests
            if self.pending:
                self._activate(self.pending.popleft())
            return

        # Auto-clear if max hold time exceeded
        if self.active_event.hold_elapsed_s >= self.max_hold_s:
            logger.warning(
                "Preemption for %s exceeded max hold (%.0fs) — auto-clearing",
                self.active_event.direction.value, self.max_hold_s,
            )
            self.clear()

    def clear(self) -> None:
        """
        Clear the active preemption event.

        The signal controller will resume normal cycling.
        """
        if self.active_event is None:
            return

        self.active_event.cleared_at = time.monotonic()
        logger.info(
            "Preemption cleared for %s (held %.1fs)",
            self.active_event.direction.value,
            self.active_event.hold_elapsed_s,
        )
        self.history.append(self.active_event)
        self.active_event = None
        self.signal_controller.clear_preemption()

        # Activate next pending if any
        if self.pending:
            self._activate(self.pending.popleft())

    def _activate(self, event: PreemptionEvent) -> None:
        """Activate a preemption event."""
        event.activated_at = time.monotonic()
        self.active_event = event
        self.signal_controller.request_preemption(event.direction)
        logger.info("Preemption ACTIVATED for %s", event.direction.value)

    @property
    def is_active(self) -> bool:
        return self.active_event is not None

    @property
    def queue_depth(self) -> int:
        return len(self.pending)

    def get_status(self) -> dict:
        """Status summary for dashboard / logging."""
        return {
            "active": self.active_event.direction.value if self.active_event else None,
            "hold_elapsed_s": (
                round(self.active_event.hold_elapsed_s, 1)
                if self.active_event
                else 0.0
            ),
            "queue_depth": self.queue_depth,
            "total_events": len(self.history),
        }

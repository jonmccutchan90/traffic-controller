"""
Conflict monitor — independent safety watchdog.

In real traffic controllers, the conflict monitor is a SEPARATE
hardware board that monitors the signal outputs independently of the
controller CPU. If it detects conflicting greens, it forces the
intersection to all-way flashing red.

This module simulates that behavior in software.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from traffic_controller.config import Direction
from traffic_controller.models.signal import SignalController, SignalState

logger = logging.getLogger(__name__)


# Pairs of directions that must NEVER both have green/arrow simultaneously
CONFLICTING_PAIRS: list[tuple[Direction, Direction]] = [
    (Direction.NORTH, Direction.EAST),
    (Direction.NORTH, Direction.WEST),
    (Direction.SOUTH, Direction.EAST),
    (Direction.SOUTH, Direction.WEST),
]


@dataclass
class ConflictMonitor:
    """
    Watches signal states and triggers fault mode if conflicts are detected.

    This is intentionally independent of the SignalController's internal
    conflict check — defense in depth. In production, this would run on
    separate hardware.
    """

    signal_controller: SignalController
    fault_active: bool = False
    conflict_count: int = 0
    last_check_time: float = field(default_factory=time.monotonic)

    # Consecutive clean checks required to clear a fault
    clean_checks_to_clear: int = 50
    _consecutive_clean: int = 0

    def check(self) -> bool:
        """
        Run one conflict check.

        Returns True if the intersection is healthy, False if a
        conflict was detected.
        """
        self.last_check_time = time.monotonic()

        green_directions: set[Direction] = set()

        for direction, head in self.signal_controller.signal_heads.items():
            if head.vehicle_signal in (SignalState.GREEN, SignalState.YELLOW):
                green_directions.add(direction)
            if head.left_turn_signal == SignalState.GREEN_ARROW:
                green_directions.add(direction)

        for d1, d2 in CONFLICTING_PAIRS:
            if d1 in green_directions and d2 in green_directions:
                self._on_conflict_detected(d1, d2)
                return False

        # No conflict
        if self.fault_active:
            self._consecutive_clean += 1
            if self._consecutive_clean >= self.clean_checks_to_clear:
                logger.info(
                    "Conflict monitor: %d consecutive clean checks — clearing fault",
                    self._consecutive_clean,
                )
                self.fault_active = False
                self._consecutive_clean = 0
        return True

    def _on_conflict_detected(self, d1: Direction, d2: Direction) -> None:
        """Handle a detected conflict."""
        self.conflict_count += 1
        self._consecutive_clean = 0

        if not self.fault_active:
            self.fault_active = True
            logger.critical(
                "CONFLICT MONITOR FAULT: %s and %s both green! "
                "Forcing all-way flashing red. (conflict #%d)",
                d1.value, d2.value, self.conflict_count,
            )
            # Force the controller into fault mode
            self.signal_controller._enter_fault_mode()

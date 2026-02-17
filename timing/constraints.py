"""
Timing constraint enforcement.

All phase durations must pass through these checks before being
applied. This module ensures that safety-critical minimums and
maximums are never violated, regardless of what the adaptive
algorithm requests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from traffic_controller.config import TimingConstraints
from traffic_controller.models.signal import Phase

logger = logging.getLogger(__name__)


@dataclass
class TimingEnforcer:
    """Clamps phase timings to safe limits."""

    constraints: TimingConstraints

    def enforce(self, phase: Phase) -> Phase:
        """
        Apply all timing constraints to a phase.

        Modifies the phase in place and returns it for chaining.
        Logs warnings when values are clamped.
        """
        c = self.constraints

        # --- Green time ---
        if phase.is_left_turn:
            min_g = c.min_protected_left_green_s
            max_g = c.max_protected_left_green_s
        else:
            min_g = c.min_green_s
            max_g = c.max_green_s

        original_green = phase.green_time_s
        phase.green_time_s = max(min_g, min(max_g, phase.green_time_s))

        if phase.green_time_s != original_green:
            logger.debug(
                "Phase %d green clamped: %.1f → %.1f (limits: %.1f–%.1f)",
                phase.phase_id, original_green, phase.green_time_s, min_g, max_g,
            )

        # --- Yellow clearance (fixed, not adaptive) ---
        phase.yellow_time_s = c.yellow_clearance_s

        # --- All-red clearance (fixed, not adaptive) ---
        phase.all_red_time_s = c.all_red_clearance_s

        # --- Pedestrian timing ---
        if not phase.is_left_turn:
            phase.walk_time_s = max(c.min_walk_s, phase.walk_time_s)
            phase.ped_clearance_time_s = c.ped_clearance_s

            # Green must be at least walk + ped_clearance for pedestrians
            min_ped_green = phase.walk_time_s + phase.ped_clearance_time_s
            if phase.green_time_s < min_ped_green:
                logger.debug(
                    "Phase %d green extended for ped timing: %.1f → %.1f",
                    phase.phase_id, phase.green_time_s, min_ped_green,
                )
                phase.green_time_s = min_ped_green

        return phase

    def enforce_cycle(self, phases: list[Phase]) -> list[Phase]:
        """Apply constraints to all phases in a cycle."""
        for phase in phases:
            self.enforce(phase)

        # Verify total cycle time is within bounds
        total = sum(p.total_phase_time_s for p in phases)
        if total < self.constraints.min_cycle_s:
            logger.warning(
                "Cycle time %.1fs below minimum %.1fs — extending greens proportionally",
                total, self.constraints.min_cycle_s,
            )
            self._extend_to_minimum(phases, total)
        elif total > self.constraints.max_cycle_s:
            logger.warning(
                "Cycle time %.1fs exceeds maximum %.1fs — reducing greens proportionally",
                total, self.constraints.max_cycle_s,
            )
            self._reduce_to_maximum(phases, total)

        return phases

    def _extend_to_minimum(self, phases: list[Phase], current_total: float) -> None:
        """Proportionally extend green times to meet minimum cycle."""
        fixed_time = sum(p.yellow_time_s + p.all_red_time_s for p in phases)
        green_time = current_total - fixed_time
        target_green = self.constraints.min_cycle_s - fixed_time

        if green_time <= 0:
            return

        ratio = target_green / green_time
        for p in phases:
            p.green_time_s *= ratio
            self.enforce(p)  # Re-clamp after scaling

    def _reduce_to_maximum(self, phases: list[Phase], current_total: float) -> None:
        """Proportionally reduce green times to meet maximum cycle."""
        fixed_time = sum(p.yellow_time_s + p.all_red_time_s for p in phases)
        green_time = current_total - fixed_time
        target_green = self.constraints.max_cycle_s - fixed_time

        if green_time <= 0:
            return

        ratio = target_green / green_time
        for p in phases:
            p.green_time_s *= ratio
            self.enforce(p)  # Re-clamp after scaling

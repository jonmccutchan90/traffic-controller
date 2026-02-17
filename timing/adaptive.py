"""
SCATS-style adaptive timing algorithm.

Recalculates phase green times at the START of each cycle based on
observed queue lengths and saturation flow rates.

Algorithm summary:
  1. Compute degree of saturation (DS) for each phase.
  2. Compute ideal green time to clear each queue.
  3. Allocate green splits proportionally based on demand.
  4. Adjust total cycle length based on overall demand.
  5. Decide protected vs. permissive left turn per approach.
  6. Enforce all safety constraints.

Reference: SCATS (Sydney Coordinated Adaptive Traffic System)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from traffic_controller.config import Direction, LaneType, TimingConstraints
from traffic_controller.models.intersection import Intersection
from traffic_controller.models.signal import Phase, PhaseRing, PhaseType
from traffic_controller.timing.constraints import TimingEnforcer

logger = logging.getLogger(__name__)


@dataclass
class PhaseDemand:
    """Computed demand metrics for a single phase."""
    phase_id: int
    total_queue: int               # Total vehicles queued for this phase
    ideal_green_s: float           # Green time needed to clear queue
    degree_of_saturation: float    # Current DS ratio
    needs_protected_left: bool     # Whether to use protected arrow


@dataclass
class CyclePlan:
    """Output of the adaptive algorithm — the plan for the next cycle."""
    cycle_length_s: float
    phase_demands: list[PhaseDemand]
    phase_greens: dict[int, float]  # phase_id → green time
    computed_at: float = 0.0


@dataclass
class AdaptiveTimingEngine:
    """
    Computes green splits for each cycle based on real-time demand.

    Call compute_cycle_plan() at the start of each cycle, then
    apply the results to the PhaseRing.
    """

    intersection: Intersection
    timing: TimingConstraints
    enforcer: TimingEnforcer = field(init=False)

    # Smoothing factor for demand changes (0-1, higher = more reactive)
    smoothing_alpha: float = 0.6

    # Previous cycle's DS values for smoothing
    _prev_ds: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.enforcer = TimingEnforcer(self.timing)

    def compute_cycle_plan(self, phase_ring: PhaseRing) -> CyclePlan:
        """
        Compute the green splits for the next cycle.

        This is the core adaptive algorithm.
        """
        demands = self._compute_demands(phase_ring)
        cycle_length = self._compute_cycle_length(demands)
        phase_greens = self._allocate_green_splits(demands, cycle_length, phase_ring)

        plan = CyclePlan(
            cycle_length_s=cycle_length,
            phase_demands=demands,
            phase_greens=phase_greens,
        )

        logger.info(
            "Cycle plan: length=%.0fs, greens=%s",
            cycle_length,
            {pid: f"{g:.0f}s" for pid, g in phase_greens.items()},
        )

        return plan

    def apply_plan(self, plan: CyclePlan, phase_ring: PhaseRing) -> None:
        """
        Apply the computed plan to the phase ring.

        Sets green times and left-turn mode for each phase,
        then enforces safety constraints.
        """
        for phase in phase_ring.phases:
            if phase.phase_id in plan.phase_greens:
                phase.green_time_s = plan.phase_greens[phase.phase_id]

            # Apply left-turn mode decision
            for demand in plan.phase_demands:
                if demand.phase_id == phase.phase_id:
                    phase.use_protected_left = demand.needs_protected_left
                    break

        # Enforce all safety constraints
        self.enforcer.enforce_cycle(phase_ring.phases)

    # --- internal computation ---

    def _compute_demands(self, phase_ring: PhaseRing) -> list[PhaseDemand]:
        """Compute demand metrics for each phase."""
        demands: list[PhaseDemand] = []

        for phase in phase_ring.phases:
            total_queue = 0
            max_ideal_green = 0.0
            left_turn_total = 0

            for direction in phase.served_directions:
                approach = self.intersection.approach(direction)

                if phase.is_left_turn:
                    lane = approach.left_turn_lane
                    left_turn_total += lane.queue_count
                else:
                    lane = approach.through_lane

                total_queue += lane.queue_count
                ideal = lane.green_time_to_clear(self.timing.startup_lost_time_s)
                max_ideal_green = max(max_ideal_green, ideal)

            # Compute DS with smoothing
            current_green = phase.green_time_s
            ds = total_queue / max(1, current_green * (
                self.intersection.approaches[phase.served_directions[0]]
                .through_lane.saturation_flow_per_sec
            )) if not phase.is_left_turn else (
                total_queue / max(1, current_green * (
                    self.intersection.approaches[phase.served_directions[0]]
                    .left_turn_lane.saturation_flow_per_sec
                ))
            )

            # Exponential smoothing
            prev_ds = self._prev_ds.get(phase.phase_id, ds)
            smoothed_ds = self.smoothing_alpha * ds + (1 - self.smoothing_alpha) * prev_ds
            self._prev_ds[phase.phase_id] = smoothed_ds

            # Left-turn mode decision
            needs_protected = (
                phase.is_left_turn
                and left_turn_total >= self.timing.left_turn_queue_threshold
            )

            demands.append(PhaseDemand(
                phase_id=phase.phase_id,
                total_queue=total_queue,
                ideal_green_s=max_ideal_green,
                degree_of_saturation=smoothed_ds,
                needs_protected_left=needs_protected,
            ))

        return demands

    def _compute_cycle_length(self, demands: list[PhaseDemand]) -> float:
        """
        Determine total cycle length based on overall demand.

        Low demand → shorter cycle (more responsive)
        High demand → longer cycle (more efficient green ratio)
        """
        # Total lost time per cycle (yellow + all-red for each phase)
        total_lost = len(demands) * (
            self.timing.yellow_clearance_s + self.timing.all_red_clearance_s
        )

        # Average degree of saturation across all phases
        avg_ds = sum(d.degree_of_saturation for d in demands) / max(1, len(demands))

        # Webster's optimal cycle formula (simplified):
        # C_opt = (1.5 * L + 5) / (1 - Y)
        # where L = total lost time, Y = sum of critical flow ratios
        # We approximate Y with avg_ds clamped to avoid division by zero
        y = min(0.90, avg_ds)  # Cap at 0.90 to avoid unreasonable cycles

        if y < 0.05:
            # Very low demand — use minimum cycle
            cycle = self.timing.min_cycle_s
        else:
            cycle = (1.5 * total_lost + 5.0) / (1.0 - y)

        # Clamp to bounds
        cycle = max(self.timing.min_cycle_s, min(self.timing.max_cycle_s, cycle))

        logger.debug("Cycle length: avg_DS=%.2f, lost=%.1fs → cycle=%.0fs", avg_ds, total_lost, cycle)
        return cycle

    def _allocate_green_splits(
        self,
        demands: list[PhaseDemand],
        cycle_length: float,
        phase_ring: PhaseRing,
    ) -> dict[int, float]:
        """
        Distribute available green time proportionally based on demand.

        Available green = cycle_length - total fixed time (yellow + all-red)
        Each phase gets a share proportional to its ideal green time.
        """
        # Total fixed time (yellow + all-red for each phase)
        total_fixed = sum(
            p.yellow_time_s + p.all_red_time_s for p in phase_ring.phases
        )
        available_green = max(0.0, cycle_length - total_fixed)

        # Compute weight for each phase (based on ideal green or minimum)
        weights: dict[int, float] = {}
        for demand in demands:
            # Use ideal green time as weight, with a floor of min_green
            if demand.total_queue == 0:
                # No demand — assign minimum weight
                weight = self.timing.min_green_s
            else:
                weight = max(self.timing.min_green_s, demand.ideal_green_s)

            # Left-turn phases with no protected need less weight
            phase = next(p for p in phase_ring.phases if p.phase_id == demand.phase_id)
            if phase.is_left_turn and not demand.needs_protected_left:
                weight = self.timing.min_protected_left_green_s * 0.5  # Minimal time for permissive

            weights[demand.phase_id] = weight

        total_weight = sum(weights.values())
        if total_weight <= 0:
            total_weight = 1.0

        # Proportional split
        greens: dict[int, float] = {}
        for phase_id, weight in weights.items():
            greens[phase_id] = (weight / total_weight) * available_green

        return greens

    def get_diagnostics(self, phase_ring: PhaseRing) -> dict:
        """Return diagnostic info for dashboard display."""
        demands = self._compute_demands(phase_ring)
        return {
            "phase_demands": [
                {
                    "phase_id": d.phase_id,
                    "queue": d.total_queue,
                    "ideal_green_s": round(d.ideal_green_s, 1),
                    "ds": round(d.degree_of_saturation, 2),
                    "protected_left": d.needs_protected_left,
                }
                for d in demands
            ],
            "smoothed_ds": {k: round(v, 2) for k, v in self._prev_ds.items()},
        }

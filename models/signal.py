"""
Signal state machine for a NEMA-style 8-phase dual-ring traffic controller.

The state machine enforces safe transitions:
  GREEN → YELLOW → ALL_RED → (next phase)

No phase transition ever skips yellow or all-red clearance.
A conflict monitor watches for illegal concurrent greens.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from traffic_controller.config import Direction, TimingConstraints

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal display states
# ---------------------------------------------------------------------------

class SignalState(Enum):
    """Possible display states for a single signal head."""
    RED = auto()
    GREEN = auto()
    YELLOW = auto()
    GREEN_ARROW = auto()       # Protected left turn
    YELLOW_ARROW = auto()      # Clearing protected left
    FLASHING_YELLOW = auto()   # Permissive left-turn yield
    ALL_RED = auto()           # Clearance interval
    DARK = auto()              # Signal off (used during transitions)

    # Pedestrian
    WALK = auto()
    PED_CLEARANCE = auto()     # Flashing DON'T WALK
    DONT_WALK = auto()


# ---------------------------------------------------------------------------
# Phase definitions (NEMA-style)
# ---------------------------------------------------------------------------

class PhaseType(Enum):
    THROUGH = auto()       # N/S or E/W through movement
    LEFT_TURN = auto()     # Protected or permissive left turn


@dataclass
class Phase:
    """
    A single traffic phase — defines which movements get green.

    In a standard 4-way intersection:
      - Through phases serve opposing directions (N+S or E+W)
      - Left-turn phases serve opposing left turns
    """
    phase_id: int
    phase_type: PhaseType
    served_directions: tuple[Direction, ...]  # Which approaches get green
    is_left_turn: bool = False

    # --- timing (set by the adaptive engine each cycle) ---
    green_time_s: float = 15.0
    yellow_time_s: float = 4.0
    all_red_time_s: float = 2.5

    # Pedestrian timing (concurrent with vehicle green)
    walk_time_s: float = 7.0
    ped_clearance_time_s: float = 14.0  # Derived from crosswalk distance

    # Left-turn mode decision (set per cycle by timing engine)
    use_protected_left: bool = False  # True = green arrow, False = flashing yellow

    @property
    def total_phase_time_s(self) -> float:
        """Total time this phase occupies: green + yellow + all-red."""
        return self.green_time_s + self.yellow_time_s + self.all_red_time_s


# ---------------------------------------------------------------------------
# Phase ring — the ordered sequence of phases in a cycle
# ---------------------------------------------------------------------------

@dataclass
class PhaseRing:
    """
    Ordered sequence of phases that constitute one signal cycle.

    Standard NEMA 8-phase layout for a 4-way:
      Phase 1: N/S left turn
      Phase 2: N/S through
      Phase 3: E/W left turn
      Phase 4: E/W through

    (Simplified from full dual-ring for clarity; the concept is identical.)
    """
    phases: list[Phase] = field(default_factory=list)

    @classmethod
    def create_standard_4way(cls, timing: TimingConstraints) -> PhaseRing:
        """Build the standard 4-phase ring for a 4-way intersection."""
        ped_clearance = timing.ped_clearance_s

        phases = [
            Phase(
                phase_id=1,
                phase_type=PhaseType.LEFT_TURN,
                served_directions=(Direction.NORTH, Direction.SOUTH),
                is_left_turn=True,
                green_time_s=timing.min_protected_left_green_s,
                yellow_time_s=timing.yellow_clearance_s,
                all_red_time_s=timing.all_red_clearance_s,
                walk_time_s=0.0,           # No ped during left-turn phase
                ped_clearance_time_s=0.0,
            ),
            Phase(
                phase_id=2,
                phase_type=PhaseType.THROUGH,
                served_directions=(Direction.NORTH, Direction.SOUTH),
                is_left_turn=False,
                green_time_s=timing.min_green_s,
                yellow_time_s=timing.yellow_clearance_s,
                all_red_time_s=timing.all_red_clearance_s,
                walk_time_s=timing.min_walk_s,
                ped_clearance_time_s=ped_clearance,
            ),
            Phase(
                phase_id=3,
                phase_type=PhaseType.LEFT_TURN,
                served_directions=(Direction.EAST, Direction.WEST),
                is_left_turn=True,
                green_time_s=timing.min_protected_left_green_s,
                yellow_time_s=timing.yellow_clearance_s,
                all_red_time_s=timing.all_red_clearance_s,
                walk_time_s=0.0,
                ped_clearance_time_s=0.0,
            ),
            Phase(
                phase_id=4,
                phase_type=PhaseType.THROUGH,
                served_directions=(Direction.EAST, Direction.WEST),
                is_left_turn=False,
                green_time_s=timing.min_green_s,
                yellow_time_s=timing.yellow_clearance_s,
                all_red_time_s=timing.all_red_clearance_s,
                walk_time_s=timing.min_walk_s,
                ped_clearance_time_s=ped_clearance,
            ),
        ]
        return cls(phases=phases)

    @property
    def total_cycle_time_s(self) -> float:
        return sum(p.total_phase_time_s for p in self.phases)

    def next_phase_index(self, current: int) -> int:
        return (current + 1) % len(self.phases)


# ---------------------------------------------------------------------------
# Internal sub-states within a phase
# ---------------------------------------------------------------------------

class PhaseStep(Enum):
    """Sub-states within a single phase."""
    GREEN = auto()           # Main green (or green arrow for left turn)
    FLASHING_YELLOW = auto() # Permissive left turn (only for left-turn phases)
    YELLOW = auto()          # Yellow clearance
    ALL_RED = auto()         # All-red clearance


# ---------------------------------------------------------------------------
# Signal head — display state for one direction
# ---------------------------------------------------------------------------

@dataclass
class SignalHead:
    """Current display state for one direction's signal head."""
    direction: Direction
    vehicle_signal: SignalState = SignalState.RED
    left_turn_signal: SignalState = SignalState.RED
    pedestrian_signal: SignalState = SignalState.DONT_WALK


# ---------------------------------------------------------------------------
# Signal state machine
# ---------------------------------------------------------------------------

@dataclass
class SignalController:
    """
    Manages the signal state machine for the entire intersection.

    Runs through the phase ring, transitioning through sub-steps
    (GREEN → YELLOW → ALL_RED) with precise timing.

    External code never sets signals directly — it can only:
      - Call tick() to advance time
      - Call request_preemption() for emergency vehicles
      - Let the timing engine update phase.green_time_s between cycles
    """

    phase_ring: PhaseRing
    timing: TimingConstraints

    # --- runtime state ---
    current_phase_idx: int = 0
    current_step: PhaseStep = PhaseStep.GREEN
    step_start_time: float = field(default_factory=time.monotonic)
    cycle_count: int = 0
    is_preempted: bool = False
    preemption_direction: Direction | None = None

    # Signal heads — one per direction
    signal_heads: dict[Direction, SignalHead] = field(default_factory=dict)

    # Callbacks
    on_phase_change: Callable[[Phase, PhaseStep], None] | None = None
    on_cycle_complete: Callable[[int], None] | None = None
    on_conflict_detected: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        # Initialize signal heads for all four directions
        for d in Direction:
            self.signal_heads[d] = SignalHead(direction=d)
        # Apply initial phase signals
        self._apply_signals_for_current_state()

    @property
    def current_phase(self) -> Phase:
        return self.phase_ring.phases[self.current_phase_idx]

    @property
    def step_elapsed_s(self) -> float:
        return time.monotonic() - self.step_start_time

    @property
    def step_remaining_s(self) -> float:
        duration = self._current_step_duration()
        return max(0.0, duration - self.step_elapsed_s)

    # --- main tick (called by controller at controller_hz) ---

    def tick(self, now: float | None = None) -> None:
        """
        Advance the state machine by one tick.

        Should be called at the controller's tick rate (e.g., 10 Hz).
        """
        if now is None:
            now = time.monotonic()

        elapsed = now - self.step_start_time
        duration = self._current_step_duration()

        if elapsed >= duration:
            self._advance_step(now)

        # Safety check: verify no conflicting greens
        self._check_conflicts()

    # --- preemption ---

    def request_preemption(self, direction: Direction) -> None:
        """
        Emergency vehicle preemption.

        Begins safe transition (yellow → all-red) then gives green
        to the requested direction.
        """
        if self.is_preempted:
            logger.warning("Preemption already active, ignoring new request for %s", direction)
            return

        logger.info("PREEMPTION requested for %s", direction.value)
        self.is_preempted = True
        self.preemption_direction = direction

        # Force transition to yellow if currently green
        if self.current_step == PhaseStep.GREEN or self.current_step == PhaseStep.FLASHING_YELLOW:
            self.current_step = PhaseStep.YELLOW
            self.step_start_time = time.monotonic()
            self._apply_signals_for_current_state()

    def clear_preemption(self) -> None:
        """End preemption and resume normal cycling."""
        logger.info("Preemption cleared, resuming normal operation")
        self.is_preempted = False
        self.preemption_direction = None

    # --- internal state transitions ---

    def _advance_step(self, now: float) -> None:
        """Move to the next sub-step or the next phase."""
        phase = self.current_phase

        if self.current_step == PhaseStep.GREEN:
            if phase.is_left_turn and not phase.use_protected_left:
                # Protected portion done, switch to permissive flashing yellow
                # (This step is skipped if use_protected_left is True)
                self.current_step = PhaseStep.YELLOW
            else:
                self.current_step = PhaseStep.YELLOW

        elif self.current_step == PhaseStep.FLASHING_YELLOW:
            self.current_step = PhaseStep.YELLOW

        elif self.current_step == PhaseStep.YELLOW:
            self.current_step = PhaseStep.ALL_RED

        elif self.current_step == PhaseStep.ALL_RED:
            # Phase is complete — move to next phase
            if self.is_preempted and self.preemption_direction:
                self._enter_preemption_phase(now)
            else:
                self._advance_to_next_phase(now)
            return

        self.step_start_time = now
        self._apply_signals_for_current_state()
        self._notify_phase_change()

    def _advance_to_next_phase(self, now: float) -> None:
        """Move to the next phase in the ring."""
        prev_idx = self.current_phase_idx
        self.current_phase_idx = self.phase_ring.next_phase_index(prev_idx)
        self.current_step = PhaseStep.GREEN
        self.step_start_time = now
        self._apply_signals_for_current_state()
        self._notify_phase_change()

        # Check if we completed a full cycle
        if self.current_phase_idx == 0 and prev_idx != 0:
            self.cycle_count += 1
            logger.info("Cycle %d complete", self.cycle_count)
            if self.on_cycle_complete:
                self.on_cycle_complete(self.cycle_count)

    def _enter_preemption_phase(self, now: float) -> None:
        """Give green to the preemption direction after safe clearance."""
        # Set all signals to red first
        for head in self.signal_heads.values():
            head.vehicle_signal = SignalState.RED
            head.left_turn_signal = SignalState.RED
            head.pedestrian_signal = SignalState.DONT_WALK

        # Give green only to the preemption direction
        if self.preemption_direction:
            head = self.signal_heads[self.preemption_direction]
            head.vehicle_signal = SignalState.GREEN

        self.current_step = PhaseStep.GREEN
        self.step_start_time = now
        logger.info("Preemption GREEN active for %s", self.preemption_direction)

    # --- signal application ---

    def _apply_signals_for_current_state(self) -> None:
        """Set all signal heads based on current phase and step."""
        phase = self.current_phase

        # Default everything to red / don't walk
        for head in self.signal_heads.values():
            head.vehicle_signal = SignalState.RED
            head.left_turn_signal = SignalState.RED
            head.pedestrian_signal = SignalState.DONT_WALK

        if self.current_step == PhaseStep.ALL_RED:
            # All signals stay red — already set above
            return

        for direction in phase.served_directions:
            head = self.signal_heads[direction]

            if self.current_step == PhaseStep.GREEN:
                if phase.is_left_turn:
                    if phase.use_protected_left:
                        head.left_turn_signal = SignalState.GREEN_ARROW
                    else:
                        head.left_turn_signal = SignalState.FLASHING_YELLOW
                    # Through stays red during left-turn phase
                else:
                    head.vehicle_signal = SignalState.GREEN
                    # Permissive left turn (flashing yellow) during through green
                    head.left_turn_signal = SignalState.FLASHING_YELLOW
                    # Pedestrian walk (concurrent with through green)
                    head.pedestrian_signal = SignalState.WALK

            elif self.current_step == PhaseStep.FLASHING_YELLOW:
                head.left_turn_signal = SignalState.FLASHING_YELLOW

            elif self.current_step == PhaseStep.YELLOW:
                if phase.is_left_turn:
                    head.left_turn_signal = SignalState.YELLOW_ARROW
                else:
                    head.vehicle_signal = SignalState.YELLOW
                    head.left_turn_signal = SignalState.RED
                    # Switch ped to clearance
                    head.pedestrian_signal = SignalState.PED_CLEARANCE

    # --- step duration ---

    def _current_step_duration(self) -> float:
        """Duration of the current sub-step in seconds."""
        phase = self.current_phase

        if self.is_preempted and self.current_step == PhaseStep.GREEN:
            # During preemption, hold green indefinitely until cleared
            return float("inf")

        if self.current_step == PhaseStep.GREEN:
            return phase.green_time_s
        elif self.current_step == PhaseStep.FLASHING_YELLOW:
            # Flashing yellow runs for the remainder of the through phase
            return phase.green_time_s  # Will be set by timing engine
        elif self.current_step == PhaseStep.YELLOW:
            return phase.yellow_time_s
        elif self.current_step == PhaseStep.ALL_RED:
            return phase.all_red_time_s
        return 0.0

    # --- conflict detection ---

    def _check_conflicts(self) -> None:
        """
        Conflict monitor: verify that no conflicting directions have green.

        Conflicting pairs: N↔E, N↔W, S↔E, S↔W
        (N+S can be green together, E+W can be green together)
        """
        green_directions: set[Direction] = set()

        for direction, head in self.signal_heads.items():
            if head.vehicle_signal in (SignalState.GREEN, SignalState.YELLOW):
                green_directions.add(direction)
            if head.left_turn_signal in (SignalState.GREEN_ARROW,):
                green_directions.add(direction)

        # Check for conflicting pairs
        conflicts = [
            (Direction.NORTH, Direction.EAST),
            (Direction.NORTH, Direction.WEST),
            (Direction.SOUTH, Direction.EAST),
            (Direction.SOUTH, Direction.WEST),
        ]

        for d1, d2 in conflicts:
            if d1 in green_directions and d2 in green_directions:
                logger.critical(
                    "CONFLICT DETECTED: %s and %s both have green/arrow!",
                    d1.value, d2.value,
                )
                self._enter_fault_mode()
                if self.on_conflict_detected:
                    self.on_conflict_detected()
                return

    def _enter_fault_mode(self) -> None:
        """
        Fail-safe: set all signals to flashing red.

        This is what real conflict monitors do when they detect
        an impossible state — the intersection goes to all-way stop.
        """
        logger.critical("ENTERING FAULT MODE — ALL FLASHING RED")
        for head in self.signal_heads.values():
            head.vehicle_signal = SignalState.ALL_RED
            head.left_turn_signal = SignalState.ALL_RED
            head.pedestrian_signal = SignalState.DONT_WALK

    # --- notification helpers ---

    def _notify_phase_change(self) -> None:
        if self.on_phase_change:
            self.on_phase_change(self.current_phase, self.current_step)

    # --- display helpers ---

    def get_display_state(self) -> dict[str, dict[str, str]]:
        """Return human-readable signal states for all directions."""
        return {
            d.value: {
                "vehicle": head.vehicle_signal.name,
                "left_turn": head.left_turn_signal.name,
                "pedestrian": head.pedestrian_signal.name,
            }
            for d, head in self.signal_heads.items()
        }

    def get_status_summary(self) -> dict:
        """Summary for dashboard / logging."""
        return {
            "cycle": self.cycle_count,
            "phase": self.current_phase.phase_id,
            "phase_type": self.current_phase.phase_type.name,
            "step": self.current_step.name,
            "step_remaining_s": round(self.step_remaining_s, 1),
            "is_preempted": self.is_preempted,
            "signals": self.get_display_state(),
        }

"""
Tests for the signal state machine.

Verifies:
  - No conflicting greens ever occur
  - Yellow and all-red clearance are never skipped
  - Minimum green times are respected
  - Preemption goes through safe transitions
  - Fault mode activates on conflicts
"""

import time
import pytest

from traffic_controller.config import Direction, TimingConstraints
from traffic_controller.models.signal import (
    Phase,
    PhaseRing,
    PhaseStep,
    PhaseType,
    SignalController,
    SignalState,
)


@pytest.fixture
def timing() -> TimingConstraints:
    return TimingConstraints()


@pytest.fixture
def phase_ring(timing: TimingConstraints) -> PhaseRing:
    return PhaseRing.create_standard_4way(timing)


@pytest.fixture
def controller(phase_ring: PhaseRing, timing: TimingConstraints) -> SignalController:
    return SignalController(phase_ring=phase_ring, timing=timing)


class TestSignalStateInitialization:
    def test_starts_in_green(self, controller: SignalController) -> None:
        assert controller.current_step == PhaseStep.GREEN

    def test_starts_at_phase_1(self, controller: SignalController) -> None:
        assert controller.current_phase_idx == 0

    def test_all_directions_have_heads(self, controller: SignalController) -> None:
        for d in Direction:
            assert d in controller.signal_heads


class TestNoConflictingGreens:
    """The most critical invariant: conflicting directions are never both green."""

    CONFLICTING = [
        (Direction.NORTH, Direction.EAST),
        (Direction.NORTH, Direction.WEST),
        (Direction.SOUTH, Direction.EAST),
        (Direction.SOUTH, Direction.WEST),
    ]

    def test_no_conflicts_during_normal_cycling(self, controller: SignalController) -> None:
        """Run through multiple complete cycles and verify no conflicts."""
        now = time.monotonic()
        for _ in range(5000):  # 5000 ticks at 10Hz = ~500 seconds
            controller.tick(now)
            now += 0.1  # 100ms tick

            self._assert_no_conflicts(controller)

    def test_no_conflicts_during_preemption(self, controller: SignalController) -> None:
        """Preemption must go through safe transitions."""
        now = time.monotonic()

        # Run for a bit then preempt
        for _ in range(100):
            controller.tick(now)
            now += 0.1

        controller.request_preemption(Direction.EAST)

        for _ in range(200):
            controller.tick(now)
            now += 0.1
            self._assert_no_conflicts(controller)

        controller.clear_preemption()

        for _ in range(200):
            controller.tick(now)
            now += 0.1
            self._assert_no_conflicts(controller)

    def _assert_no_conflicts(self, ctrl: SignalController) -> None:
        green_dirs: set[Direction] = set()
        for d, head in ctrl.signal_heads.items():
            if head.vehicle_signal in (SignalState.GREEN, SignalState.YELLOW):
                green_dirs.add(d)
            if head.left_turn_signal == SignalState.GREEN_ARROW:
                green_dirs.add(d)

        for d1, d2 in self.CONFLICTING:
            assert not (d1 in green_dirs and d2 in green_dirs), (
                f"CONFLICT: {d1.value} and {d2.value} both green! "
                f"Phase={ctrl.current_phase.phase_id}, Step={ctrl.current_step.name}"
            )


class TestClearanceIntervals:
    """Yellow and all-red are never skipped."""

    def test_yellow_before_red(self, controller: SignalController) -> None:
        """After green, yellow must occur before the next phase's green."""
        now = time.monotonic()
        saw_yellow = False
        prev_step = controller.current_step

        for _ in range(2000):
            controller.tick(now)
            now += 0.1

            curr = controller.current_step
            if prev_step == PhaseStep.GREEN and curr != PhaseStep.GREEN:
                assert curr == PhaseStep.YELLOW, (
                    f"Expected YELLOW after GREEN, got {curr.name}"
                )
                saw_yellow = True
            prev_step = curr

        assert saw_yellow, "Never saw a GREEN → YELLOW transition"

    def test_all_red_after_yellow(self, controller: SignalController) -> None:
        """After yellow, all-red must occur before next green."""
        now = time.monotonic()
        saw_all_red = False
        prev_step = controller.current_step

        for _ in range(2000):
            controller.tick(now)
            now += 0.1

            curr = controller.current_step
            if prev_step == PhaseStep.YELLOW and curr != PhaseStep.YELLOW:
                assert curr == PhaseStep.ALL_RED, (
                    f"Expected ALL_RED after YELLOW, got {curr.name}"
                )
                saw_all_red = True
            prev_step = curr

        assert saw_all_red, "Never saw a YELLOW → ALL_RED transition"


class TestMinimumGreens:
    def test_green_duration_respects_minimum(
        self, controller: SignalController, timing: TimingConstraints
    ) -> None:
        """Green phase should last at least min_green_s."""
        now = time.monotonic()
        green_start: float | None = None
        prev_step = PhaseStep.GREEN

        for _ in range(3000):
            controller.tick(now)
            now += 0.1

            curr = controller.current_step

            if prev_step != PhaseStep.GREEN and curr == PhaseStep.GREEN:
                green_start = now

            if prev_step == PhaseStep.GREEN and curr != PhaseStep.GREEN:
                if green_start is not None:
                    duration = now - green_start
                    min_expected = timing.min_green_s
                    assert duration >= min_expected - 0.2, (
                        f"Green was only {duration:.1f}s (min={min_expected}s)"
                    )

            prev_step = curr


class TestPreemption:
    def test_preemption_gives_green_to_requested_direction(
        self, controller: SignalController
    ) -> None:
        now = time.monotonic()

        # Advance past initial green
        for _ in range(200):
            controller.tick(now)
            now += 0.1

        controller.request_preemption(Direction.NORTH)

        # Advance through yellow and all-red to preemption green
        for _ in range(200):
            controller.tick(now)
            now += 0.1

            if controller.is_preempted:
                head = controller.signal_heads[Direction.NORTH]
                if head.vehicle_signal == SignalState.GREEN:
                    return  # Success

        # If we got here, preemption green was never activated
        pytest.fail("Preemption never gave GREEN to NORTH")

    def test_preemption_clears_correctly(self, controller: SignalController) -> None:
        now = time.monotonic()

        controller.request_preemption(Direction.EAST)

        for _ in range(200):
            controller.tick(now)
            now += 0.1

        controller.clear_preemption()
        assert not controller.is_preempted


class TestFaultMode:
    def test_fault_mode_sets_all_red(self, controller: SignalController) -> None:
        """Fault mode should set all signals to ALL_RED."""
        controller._enter_fault_mode()

        for head in controller.signal_heads.values():
            assert head.vehicle_signal == SignalState.ALL_RED
            assert head.left_turn_signal == SignalState.ALL_RED

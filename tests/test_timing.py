"""
Tests for the adaptive timing engine.

Verifies:
  - Green splits respond to demand changes
  - Cycle length adjusts to traffic volume
  - Left-turn mode switches correctly
  - Safety constraints are never violated
  - Zero-traffic and heavy-traffic edge cases
"""

import pytest

from traffic_controller.config import Direction, FlowDefaults, TimingConstraints
from traffic_controller.models.intersection import Intersection
from traffic_controller.models.signal import PhaseRing
from traffic_controller.timing.adaptive import AdaptiveTimingEngine
from traffic_controller.timing.constraints import TimingEnforcer


@pytest.fixture
def timing() -> TimingConstraints:
    return TimingConstraints()


@pytest.fixture
def intersection() -> Intersection:
    return Intersection.create_standard()


@pytest.fixture
def phase_ring(timing: TimingConstraints) -> PhaseRing:
    return PhaseRing.create_standard_4way(timing)


@pytest.fixture
def engine(intersection: Intersection, timing: TimingConstraints) -> AdaptiveTimingEngine:
    return AdaptiveTimingEngine(intersection=intersection, timing=timing)


class TestGreenSplitResponsiveness:
    def test_heavier_approach_gets_more_green(
        self, engine: AdaptiveTimingEngine, intersection: Intersection,
        phase_ring: PhaseRing
    ) -> None:
        """The approach with more cars should get a larger green split."""
        # Heavy N/S through traffic
        intersection.approach(Direction.NORTH).through_lane.update(15)
        intersection.approach(Direction.SOUTH).through_lane.update(12)
        # Light E/W traffic
        intersection.approach(Direction.EAST).through_lane.update(2)
        intersection.approach(Direction.WEST).through_lane.update(1)

        plan = engine.compute_cycle_plan(phase_ring)

        # Phase 2 = N/S through, Phase 4 = E/W through
        ns_green = plan.phase_greens.get(2, 0)
        ew_green = plan.phase_greens.get(4, 0)

        assert ns_green > ew_green, (
            f"N/S (queue=27) got {ns_green:.0f}s but E/W (queue=3) got {ew_green:.0f}s"
        )

    def test_zero_traffic_all_phases_get_minimum(
        self, engine: AdaptiveTimingEngine, intersection: Intersection,
        phase_ring: PhaseRing, timing: TimingConstraints
    ) -> None:
        """With zero demand, all phases should get minimum green."""
        plan = engine.compute_cycle_plan(phase_ring)
        engine.apply_plan(plan, phase_ring)

        for phase in phase_ring.phases:
            if phase.is_left_turn:
                # Left-turn phases may get very short times when permissive
                assert phase.green_time_s >= timing.min_protected_left_green_s * 0.4
            else:
                assert phase.green_time_s >= timing.min_green_s


class TestCycleLengthAdaptation:
    def test_heavy_traffic_extends_cycle(
        self, engine: AdaptiveTimingEngine, intersection: Intersection,
        phase_ring: PhaseRing, timing: TimingConstraints
    ) -> None:
        """Heavy traffic should push cycle length toward maximum."""
        for d in Direction:
            intersection.approach(d).through_lane.update(20)
            intersection.approach(d).left_turn_lane.update(5)

        plan = engine.compute_cycle_plan(phase_ring)

        # With heavy traffic, cycle should be well above default
        assert plan.cycle_length_s > timing.default_cycle_s

    def test_light_traffic_shrinks_cycle(
        self, engine: AdaptiveTimingEngine, intersection: Intersection,
        phase_ring: PhaseRing, timing: TimingConstraints
    ) -> None:
        """Light traffic should allow cycle to shrink."""
        for d in Direction:
            intersection.approach(d).through_lane.update(1)
            intersection.approach(d).left_turn_lane.update(0)

        plan = engine.compute_cycle_plan(phase_ring)

        assert plan.cycle_length_s <= timing.default_cycle_s


class TestLeftTurnModeDecision:
    def test_low_left_turn_queue_uses_permissive(
        self, engine: AdaptiveTimingEngine, intersection: Intersection,
        phase_ring: PhaseRing, timing: TimingConstraints
    ) -> None:
        """Below threshold, left turn should be permissive (flashing yellow)."""
        # 1 car turning left — below threshold of 3
        intersection.approach(Direction.NORTH).left_turn_lane.update(1)
        intersection.approach(Direction.SOUTH).left_turn_lane.update(0)

        plan = engine.compute_cycle_plan(phase_ring)

        # Phase 1 = N/S left turn
        ns_left_demand = next(d for d in plan.phase_demands if d.phase_id == 1)
        assert not ns_left_demand.needs_protected_left

    def test_high_left_turn_queue_uses_protected(
        self, engine: AdaptiveTimingEngine, intersection: Intersection,
        phase_ring: PhaseRing, timing: TimingConstraints
    ) -> None:
        """Above threshold, left turn should be protected (green arrow)."""
        # 5 cars turning left — above threshold of 3
        intersection.approach(Direction.NORTH).left_turn_lane.update(5)
        intersection.approach(Direction.SOUTH).left_turn_lane.update(3)

        plan = engine.compute_cycle_plan(phase_ring)

        ns_left_demand = next(d for d in plan.phase_demands if d.phase_id == 1)
        assert ns_left_demand.needs_protected_left


class TestTimingConstraintEnforcement:
    def test_green_clamped_to_minimum(self, timing: TimingConstraints) -> None:
        """Green time below minimum should be clamped up."""
        from traffic_controller.models.signal import Phase, PhaseType

        enforcer = TimingEnforcer(timing)
        phase = Phase(
            phase_id=1, phase_type=PhaseType.THROUGH,
            served_directions=(Direction.NORTH, Direction.SOUTH),
            green_time_s=1.0,  # Way below minimum
        )
        enforcer.enforce(phase)
        # Through green must be at least walk + ped_clearance
        min_ped = timing.min_walk_s + timing.ped_clearance_s
        assert phase.green_time_s >= max(timing.min_green_s, min_ped)

    def test_green_clamped_to_maximum(self, timing: TimingConstraints) -> None:
        """Green time above maximum should be clamped down."""
        from traffic_controller.models.signal import Phase, PhaseType

        enforcer = TimingEnforcer(timing)
        phase = Phase(
            phase_id=1, phase_type=PhaseType.THROUGH,
            served_directions=(Direction.NORTH, Direction.SOUTH),
            green_time_s=999.0,  # Way above maximum
        )
        enforcer.enforce(phase)
        assert phase.green_time_s <= timing.max_green_s

    def test_yellow_is_fixed(self, timing: TimingConstraints) -> None:
        """Yellow clearance should always equal the configured value."""
        from traffic_controller.models.signal import Phase, PhaseType

        enforcer = TimingEnforcer(timing)
        phase = Phase(
            phase_id=1, phase_type=PhaseType.THROUGH,
            served_directions=(Direction.NORTH, Direction.SOUTH),
            yellow_time_s=99.0,
        )
        enforcer.enforce(phase)
        assert phase.yellow_time_s == timing.yellow_clearance_s

"""
Tests for safety subsystems: conflict monitor and preemption manager.
"""

import time
import pytest

from traffic_controller.config import Direction, TimingConstraints
from traffic_controller.models.signal import (
    PhaseRing,
    SignalController,
    SignalState,
)
from traffic_controller.safety.conflict import ConflictMonitor
from traffic_controller.safety.preemption import PreemptionManager


@pytest.fixture
def timing() -> TimingConstraints:
    return TimingConstraints()


@pytest.fixture
def signal_controller(timing: TimingConstraints) -> SignalController:
    ring = PhaseRing.create_standard_4way(timing)
    return SignalController(phase_ring=ring, timing=timing)


class TestConflictMonitor:
    def test_healthy_state_returns_true(
        self, signal_controller: SignalController
    ) -> None:
        monitor = ConflictMonitor(signal_controller=signal_controller)
        assert monitor.check() is True

    def test_detects_forced_conflict(
        self, signal_controller: SignalController
    ) -> None:
        """Manually force conflicting greens and verify detection."""
        monitor = ConflictMonitor(signal_controller=signal_controller)

        # Force an illegal state: N and E both green
        signal_controller.signal_heads[Direction.NORTH].vehicle_signal = SignalState.GREEN
        signal_controller.signal_heads[Direction.EAST].vehicle_signal = SignalState.GREEN

        result = monitor.check()
        assert result is False
        assert monitor.fault_active is True
        assert monitor.conflict_count == 1

    def test_fault_clears_after_consecutive_clean_checks(
        self, signal_controller: SignalController
    ) -> None:
        monitor = ConflictMonitor(
            signal_controller=signal_controller,
            clean_checks_to_clear=5,
        )

        # Trigger a fault
        signal_controller.signal_heads[Direction.NORTH].vehicle_signal = SignalState.GREEN
        signal_controller.signal_heads[Direction.EAST].vehicle_signal = SignalState.GREEN
        monitor.check()
        assert monitor.fault_active is True

        # Fix the state
        signal_controller.signal_heads[Direction.EAST].vehicle_signal = SignalState.RED

        # Need consecutive clean checks to clear
        for _ in range(4):
            monitor.check()
            assert monitor.fault_active is True  # Not yet cleared

        monitor.check()  # 5th clean check
        assert monitor.fault_active is False


class TestPreemptionManager:
    def test_single_preemption(
        self, signal_controller: SignalController
    ) -> None:
        manager = PreemptionManager(signal_controller=signal_controller)
        manager.request(Direction.NORTH)

        assert manager.is_active
        assert manager.active_event is not None
        assert manager.active_event.direction == Direction.NORTH

    def test_clear_preemption(
        self, signal_controller: SignalController
    ) -> None:
        manager = PreemptionManager(signal_controller=signal_controller)
        manager.request(Direction.SOUTH)
        manager.clear()

        assert not manager.is_active
        assert len(manager.history) == 1

    def test_queued_preemption(
        self, signal_controller: SignalController
    ) -> None:
        """Second preemption request while one is active should queue."""
        manager = PreemptionManager(signal_controller=signal_controller)
        manager.request(Direction.NORTH)
        manager.request(Direction.EAST)

        assert manager.is_active
        assert manager.active_event.direction == Direction.NORTH
        assert manager.queue_depth == 1

        # Clearing first should activate second
        manager.clear()
        assert manager.is_active
        assert manager.active_event.direction == Direction.EAST
        assert manager.queue_depth == 0

    def test_auto_clear_on_timeout(
        self, signal_controller: SignalController
    ) -> None:
        manager = PreemptionManager(
            signal_controller=signal_controller,
            max_hold_s=0.1,  # Very short for testing
        )
        manager.request(Direction.WEST)

        # Wait for timeout
        time.sleep(0.15)
        manager.tick()

        assert not manager.is_active

    def test_status_report(
        self, signal_controller: SignalController
    ) -> None:
        manager = PreemptionManager(signal_controller=signal_controller)
        status = manager.get_status()

        assert status["active"] is None
        assert status["queue_depth"] == 0

        manager.request(Direction.NORTH)
        status = manager.get_status()
        assert status["active"] == "N"

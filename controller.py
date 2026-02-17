"""
Main controller loop — orchestrates vision, timing, state machine, and safety.

This is the "main thread" of the traffic controller. It runs at a
configurable tick rate and coordinates all subsystems.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from traffic_controller.config import (
    Direction,
    IntersectionConfig,
    LaneType,
)
from traffic_controller.models.intersection import Intersection
from traffic_controller.models.signal import (
    Phase,
    PhaseRing,
    PhaseStep,
    SignalController,
)
from traffic_controller.safety.conflict import ConflictMonitor
from traffic_controller.safety.preemption import PreemptionManager
from traffic_controller.timing.adaptive import AdaptiveTimingEngine
from traffic_controller.vision.provider import VehicleDetectionProvider

logger = logging.getLogger(__name__)


def create_provider(config: IntersectionConfig) -> VehicleDetectionProvider:
    """Factory: create the configured detection provider."""
    if config.vision.provider_type == "yolov8":
        from traffic_controller.vision.yolov8 import YOLOv8Provider
        return YOLOv8Provider()
    elif config.vision.provider_type == "mock":
        from traffic_controller.vision.mock import MockProvider
        return MockProvider()
    else:
        raise ValueError(f"Unknown provider type: {config.vision.provider_type}")


@dataclass
class TrafficController:
    """
    Top-level controller for a single intersection.

    Lifecycle:
      1. Construct with an IntersectionConfig
      2. Call setup() to initialize all subsystems
      3. Call run() to start the main loop (blocking)
         OR call tick() manually for step-by-step control
      4. Call teardown() on shutdown
    """

    config: IntersectionConfig

    # --- subsystems (initialized in setup()) ---
    intersection: Intersection = field(init=False)
    phase_ring: PhaseRing = field(init=False)
    signal_controller: SignalController = field(init=False)
    timing_engine: AdaptiveTimingEngine = field(init=False)
    conflict_monitor: ConflictMonitor = field(init=False)
    preemption_manager: PreemptionManager = field(init=False)
    provider: VehicleDetectionProvider = field(init=False)

    # --- runtime state ---
    is_running: bool = False
    tick_count: int = 0
    last_vision_time: float = 0.0
    last_cycle_count: int = 0

    # Dashboard callback (set by dashboard module)
    on_tick: list[Any] = field(default_factory=list)

    def setup(self) -> None:
        """Initialize all subsystems."""
        logger.info("Setting up TrafficController for '%s'", self.config.name)

        # 1. Intersection model
        self.intersection = Intersection.create_standard(
            name=self.config.name,
            flow=self.config.flow,
            crosswalk_distance_ft=self.config.timing.default_crosswalk_distance_ft,
        )

        # 2. Phase ring
        self.phase_ring = PhaseRing.create_standard_4way(self.config.timing)

        # 3. Signal controller
        self.signal_controller = SignalController(
            phase_ring=self.phase_ring,
            timing=self.config.timing,
        )
        self.signal_controller.on_cycle_complete = self._on_cycle_complete

        # 4. Adaptive timing engine
        self.timing_engine = AdaptiveTimingEngine(
            intersection=self.intersection,
            timing=self.config.timing,
        )

        # 5. Safety subsystems
        self.conflict_monitor = ConflictMonitor(
            signal_controller=self.signal_controller,
        )
        self.preemption_manager = PreemptionManager(
            signal_controller=self.signal_controller,
        )

        # 6. Vision provider
        self.provider = create_provider(self.config)
        self.provider.initialize(self.config.vision.to_dict())

        # Run initial timing computation
        plan = self.timing_engine.compute_cycle_plan(self.phase_ring)
        self.timing_engine.apply_plan(plan, self.phase_ring)

        logger.info("TrafficController setup complete")

    def teardown(self) -> None:
        """Shut down all subsystems."""
        self.is_running = False
        if hasattr(self, "provider"):
            self.provider.shutdown()
        logger.info("TrafficController torn down")

    # --- main loop ---

    def run(self, max_ticks: int | None = None) -> None:
        """
        Run the main control loop.

        Blocks until is_running is set to False or max_ticks is reached.
        Uses monotonic timing to avoid drift.
        """
        self.is_running = True
        interval = self.config.tick_interval_s
        next_tick = time.monotonic()

        logger.info("Controller running at %.0f Hz", self.config.controller_hz)

        try:
            while self.is_running:
                now = time.monotonic()

                if now >= next_tick:
                    self.tick(now)
                    next_tick += interval

                    if max_ticks is not None and self.tick_count >= max_ticks:
                        logger.info("Reached max_ticks (%d), stopping", max_ticks)
                        break

                # Sleep briefly to avoid busy-waiting
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time * 0.9)  # Sleep slightly less to avoid overshooting

        except KeyboardInterrupt:
            logger.info("Controller interrupted by user")
        finally:
            self.is_running = False

    def tick(self, now: float | None = None) -> None:
        """
        Execute one controller tick.

        This is the heartbeat of the system:
          1. Run vision (at target FPS, not every tick)
          2. Update queue counts
          3. Advance signal state machine
          4. Run safety checks
          5. Handle preemption lifecycle
          6. Notify dashboard
        """
        if now is None:
            now = time.monotonic()

        self.tick_count += 1

        # 1. Vision — run at target FPS, not every tick
        vision_interval = 1.0 / self.config.vision.target_fps
        if now - self.last_vision_time >= vision_interval:
            self._run_vision(now)
            self.last_vision_time = now

        # 2. Signal state machine
        self.signal_controller.tick(now)

        # 3. Safety checks
        self.conflict_monitor.check()
        self.preemption_manager.tick()

        # 4. Notify listeners (dashboard, etc.)
        for callback in self.on_tick:
            callback(self)

    # --- vision pipeline ---

    def _run_vision(self, now: float) -> None:
        """Run one frame through the detection pipeline and update queues."""
        # In simulation mode with MockProvider, we don't need a real frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = self.provider.detect(dummy_frame)

        # Update intersection queue counts from detections
        # MockProvider maintains its own queue state; for real providers,
        # we'd use ROI counting here
        from traffic_controller.vision.mock import MockProvider
        if isinstance(self.provider, MockProvider):
            counts = self.provider.get_queue_counts()
            for (dir_val, lane_val), count in counts.items():
                direction = Direction(dir_val)
                approach = self.intersection.approach(direction)
                if lane_val == LaneType.THROUGH.value:
                    approach.through_lane.update(count)
                elif lane_val == LaneType.LEFT_TURN.value:
                    approach.left_turn_lane.update(count)
        else:
            # For real providers, use simple quadrant counting
            from traffic_controller.vision.counting import count_vehicles_simple
            counts_simple = count_vehicles_simple(result)
            for dir_val, count in counts_simple.items():
                direction = Direction(dir_val)
                approach = self.intersection.approach(direction)
                # Split roughly: 85% through, 15% left turn
                approach.through_lane.update(int(count * 0.85))
                approach.left_turn_lane.update(max(0, count - int(count * 0.85)))

    # --- cycle-level callbacks ---

    def _on_cycle_complete(self, cycle_count: int) -> None:
        """Called at the end of each cycle — recompute timing."""
        if self.preemption_manager.is_active:
            logger.debug("Skipping timing recompute during preemption")
            return

        plan = self.timing_engine.compute_cycle_plan(self.phase_ring)
        self.timing_engine.apply_plan(plan, self.phase_ring)
        self.last_cycle_count = cycle_count

    # --- public API ---

    def trigger_preemption(self, direction: Direction) -> None:
        """Trigger emergency vehicle preemption from the given direction."""
        self.preemption_manager.request(direction)

    def clear_preemption(self) -> None:
        """Clear the current preemption."""
        self.preemption_manager.clear()

    def get_full_status(self) -> dict:
        """Return complete system status for dashboard / API."""
        return {
            "tick": self.tick_count,
            "intersection": self.intersection.snapshot(),
            "signals": self.signal_controller.get_status_summary(),
            "preemption": self.preemption_manager.get_status(),
            "conflict_monitor": {
                "fault_active": self.conflict_monitor.fault_active,
                "conflict_count": self.conflict_monitor.conflict_count,
            },
            "timing": self.timing_engine.get_diagnostics(self.phase_ring),
            "cycle_time_s": self.phase_ring.total_cycle_time_s,
        }

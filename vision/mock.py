"""
Mock vehicle detection provider for simulation / testing.

Generates synthetic queue counts using a Poisson arrival process
so the timing engine can be developed and tested without a real
camera or ML model.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any

import numpy as np

from traffic_controller.config import Direction, LaneType
from traffic_controller.vision.provider import (
    DetectedVehicle,
    DetectionResult,
    VehicleDetectionProvider,
)

logger = logging.getLogger(__name__)


class MockProvider(VehicleDetectionProvider):
    """
    Generates fake detections using configurable traffic patterns.

    Config keys used:
      - base_arrival_rate    : float  (vehicles/sec per approach, default 0.3)
      - peak_multiplier      : float  (rush-hour multiplier, default 2.5)
      - left_turn_fraction   : float  (fraction of traffic turning left, default 0.15)
      - enable_surge         : bool   (inject random surges, default True)
      - random_seed          : int|None (for reproducible runs)
    """

    def __init__(self) -> None:
        self._base_rate: float = 0.3
        self._peak_mult: float = 2.5
        self._left_frac: float = 0.15
        self._enable_surge: bool = True
        self._rng: random.Random = random.Random()
        self._start_time: float = 0.0
        self._initialized: bool = False

        # Persistent queue state per (direction, lane_type)
        self._queues: dict[tuple[str, str], int] = {}

    def initialize(self, config: dict[str, Any]) -> None:
        self._base_rate = config.get("base_arrival_rate", 0.3)
        self._peak_mult = config.get("peak_multiplier", 2.5)
        self._left_frac = config.get("left_turn_fraction", 0.15)
        self._enable_surge = config.get("enable_surge", True)
        seed = config.get("random_seed", None)
        self._rng = random.Random(seed)
        self._start_time = time.monotonic()

        # Initialize queues for all approaches
        for d in Direction:
            for lt in LaneType:
                self._queues[(d.value, lt.value)] = 0

        self._initialized = True
        logger.info("MockProvider initialized (rate=%.2f, peak=%.1fx)",
                     self._base_rate, self._peak_mult)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Ignore the frame entirely — generate synthetic detections
        based on a Poisson arrival model with time-varying rates.
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        t0 = time.monotonic()
        elapsed = t0 - self._start_time

        # Time-varying rate: simulate a rush-hour pattern (sinusoidal)
        # Peak at ~60s intervals in simulation time
        cycle_position = (elapsed % 120.0) / 120.0  # 0-1 over 2 minutes
        time_mult = 1.0 + (self._peak_mult - 1.0) * max(0, math.sin(cycle_position * math.pi))

        # Optional random surge on one approach
        surge_dir = None
        if self._enable_surge and self._rng.random() < 0.02:  # 2% chance per frame
            surge_dir = self._rng.choice(list(Direction))
            logger.debug("Traffic surge on %s approach", surge_dir.value)

        vehicles: list[DetectedVehicle] = []

        MAX_QUEUE = 25  # Realistic cap

        for d in Direction:
            rate = self._base_rate * time_mult
            if surge_dir == d:
                rate *= 3.0  # Surge triples the arrival rate

            # Poisson arrivals for through lane (~0.3-0.9 cars/sec)
            through_arrivals = 1 if self._rng.random() < rate else 0
            # Departures happen at ~0.5 cars/sec only conceptually
            # (actual discharge depends on signal, but mock has no signal access)
            through_departures = 1 if self._rng.random() < 0.4 else 0

            key_through = (d.value, LaneType.THROUGH.value)
            self._queues[key_through] = min(
                MAX_QUEUE,
                max(0, self._queues[key_through] + through_arrivals - through_departures),
            )

            # Left turn arrivals (fraction of through)
            left_arrivals = 1 if self._rng.random() < rate * self._left_frac else 0
            left_departures = 1 if self._rng.random() < 0.3 else 0

            key_left = (d.value, LaneType.LEFT_TURN.value)
            self._queues[key_left] = min(
                MAX_QUEUE,
                max(0, self._queues[key_left] + left_arrivals - left_departures),
            )

            # Generate DetectedVehicle objects at plausible positions
            for i in range(self._queues[key_through]):
                vehicles.append(self._make_vehicle(d, LaneType.THROUGH, i))
            for i in range(self._queues[key_left]):
                vehicles.append(self._make_vehicle(d, LaneType.LEFT_TURN, i))

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        return DetectionResult(
            vehicles=vehicles,
            frame_timestamp=t0,
            processing_time_ms=elapsed_ms,
            confidence_threshold=0.95,  # Mock always "confident"
            provider_name="mock",
        )

    def shutdown(self) -> None:
        self._initialized = False
        logger.info("MockProvider shut down")

    @property
    def name(self) -> str:
        return "mock"

    # --- queue access for the controller ---

    def get_queue_counts(self) -> dict[tuple[str, str], int]:
        """Direct access to synthetic queue counts (bypasses ROI logic)."""
        return dict(self._queues)

    def set_queue(self, direction: Direction, lane_type: LaneType, count: int) -> None:
        """Manually set a queue count — useful for testing specific scenarios."""
        self._queues[(direction.value, lane_type.value)] = max(0, count)

    # --- helpers ---

    def _make_vehicle(
        self, direction: Direction, lane_type: LaneType, index: int
    ) -> DetectedVehicle:
        """
        Generate a plausible bounding-box position for a queued car.

        The positions are laid out as if viewed from above, with cars
        spacing back from the stop line.
        """
        # Map direction to a region of the frame
        base_positions: dict[str, tuple[float, float]] = {
            "N": (0.45, 0.1),   # Top of frame, heading south
            "S": (0.55, 0.9),   # Bottom of frame, heading north
            "E": (0.9, 0.45),   # Right of frame, heading west
            "W": (0.1, 0.55),   # Left of frame, heading east
        }

        bx, by = base_positions[direction.value]

        # Offset for lane type (left-turn lane is adjacent)
        if lane_type == LaneType.LEFT_TURN:
            if direction in (Direction.NORTH, Direction.SOUTH):
                bx -= 0.05
            else:
                by -= 0.05

        # Stack cars back from stop line
        spacing = 0.03
        if direction == Direction.NORTH:
            by -= index * spacing
        elif direction == Direction.SOUTH:
            by += index * spacing
        elif direction == Direction.EAST:
            bx += index * spacing
        elif direction == Direction.WEST:
            bx -= index * spacing

        # Clamp to [0, 1]
        bx = max(0.02, min(0.98, bx))
        by = max(0.02, min(0.98, by))

        vtype = self._rng.choices(
            ["car", "car", "car", "truck", "bus", "motorcycle"],
            weights=[60, 60, 60, 10, 5, 15],
        )[0]

        return DetectedVehicle(
            x=bx,
            y=by,
            width=0.03 + self._rng.random() * 0.01,
            height=0.05 + self._rng.random() * 0.02,
            confidence=0.90 + self._rng.random() * 0.10,
            vehicle_type=vtype,
        )

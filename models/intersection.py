"""
Data model for a 4-way intersection: approaches, lanes, and queue state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from traffic_controller.config import Direction, FlowDefaults, LaneType


@dataclass
class Lane:
    """A single lane within an approach (through or left-turn)."""

    direction: Direction
    lane_type: LaneType
    saturation_flow: float              # veh/hr/lane

    # --- live state (updated by vision) ---
    queue_count: int = 0                # vehicles currently queued
    arrival_rate: float = 0.0           # estimated vehicles/sec arriving
    last_updated: float = field(default_factory=time.monotonic)

    # --- derived helpers ---

    @property
    def saturation_flow_per_sec(self) -> float:
        """Saturation flow converted to vehicles per second."""
        return self.saturation_flow / 3600.0

    def green_time_to_clear(self, startup_lost_time_s: float = 2.0) -> float:
        """
        Seconds of green needed to discharge the current queue.

        Formula: queue_count / (saturation_flow / 3600) + startup_lost_time
        """
        if self.queue_count <= 0:
            return 0.0
        return (self.queue_count / self.saturation_flow_per_sec) + startup_lost_time_s

    def degree_of_saturation(self, green_time_s: float) -> float:
        """
        Ratio of demand to capacity for a given green time.

        v/c = queue_count / (saturation_flow_per_sec * green_time)
        Values near 1.0 mean the lane is at capacity.
        """
        if green_time_s <= 0:
            return float("inf")
        capacity = self.saturation_flow_per_sec * green_time_s
        if capacity <= 0:
            return float("inf")
        return self.queue_count / capacity

    def update(self, queue_count: int, arrival_rate: float = 0.0) -> None:
        """Update live state from the vision pipeline."""
        self.queue_count = max(0, queue_count)
        self.arrival_rate = max(0.0, arrival_rate)
        self.last_updated = time.monotonic()


@dataclass
class Approach:
    """
    One of the four cardinal approaches to the intersection.

    Each approach has one through lane (may represent multiple physical
    lanes aggregated) and one left-turn lane.
    """

    direction: Direction
    through_lane: Lane
    left_turn_lane: Lane

    # Pedestrian crossing associated with this approach
    has_pedestrian_crossing: bool = True
    crosswalk_distance_ft: float = 48.0

    @property
    def total_queue(self) -> int:
        return self.through_lane.queue_count + self.left_turn_lane.queue_count

    @property
    def through_queue(self) -> int:
        return self.through_lane.queue_count

    @property
    def left_turn_queue(self) -> int:
        return self.left_turn_lane.queue_count


@dataclass
class Intersection:
    """
    A complete 4-way intersection with four approaches.

    This is the central data model that the controller, timing engine,
    and dashboard all read from / write to.
    """

    name: str
    approaches: dict[Direction, Approach] = field(default_factory=dict)

    @classmethod
    def create_standard(
        cls,
        name: str = "Main & 1st",
        flow: FlowDefaults | None = None,
        crosswalk_distance_ft: float = 48.0,
    ) -> Intersection:
        """Factory: build a standard 4-way intersection with default lanes."""
        if flow is None:
            flow = FlowDefaults()

        approaches: dict[Direction, Approach] = {}
        for direction in Direction:
            through = Lane(
                direction=direction,
                lane_type=LaneType.THROUGH,
                saturation_flow=flow.through_lane,
            )
            left = Lane(
                direction=direction,
                lane_type=LaneType.LEFT_TURN,
                saturation_flow=flow.left_turn_lane,
            )
            approaches[direction] = Approach(
                direction=direction,
                through_lane=through,
                left_turn_lane=left,
                crosswalk_distance_ft=crosswalk_distance_ft,
            )

        return cls(name=name, approaches=approaches)

    # --- convenience accessors ---

    def approach(self, d: Direction) -> Approach:
        return self.approaches[d]

    @property
    def all_lanes(self) -> list[Lane]:
        lanes: list[Lane] = []
        for a in self.approaches.values():
            lanes.extend([a.through_lane, a.left_turn_lane])
        return lanes

    @property
    def total_queue(self) -> int:
        return sum(a.total_queue for a in self.approaches.values())

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return a simple dict of queue counts for logging/dashboard."""
        return {
            d.value: {
                "through": a.through_queue,
                "left_turn": a.left_turn_queue,
            }
            for d, a in self.approaches.items()
        }

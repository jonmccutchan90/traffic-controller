"""
Central configuration for the adaptive traffic light controller.

All timing values are in seconds unless noted otherwise.
All flow values are in vehicles per hour per lane (veh/hr/lane).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Direction / approach enums
# ---------------------------------------------------------------------------

class Direction(Enum):
    """Cardinal directions for the four approaches."""
    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"


class LaneType(Enum):
    THROUGH = "through"
    LEFT_TURN = "left_turn"


# ---------------------------------------------------------------------------
# Timing constraints (safety-critical — values from ITE / MUTCD standards)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimingConstraints:
    """Hard limits that the adaptive algorithm must never violate."""

    # Vehicle phase minimums
    min_green_s: float = 7.0          # Minimum green for vehicle phase
    max_green_s: float = 60.0         # Maximum green for any single phase
    yellow_clearance_s: float = 4.0   # Yellow interval (3–5s typical)
    all_red_clearance_s: float = 2.5  # All-red between every phase change
    startup_lost_time_s: float = 2.0  # Time for first car to react at green

    # Left-turn specifics
    min_protected_left_green_s: float = 8.0   # Minimum protected arrow
    max_protected_left_green_s: float = 25.0  # Maximum protected arrow
    left_turn_queue_threshold: int = 3        # Cars needed to trigger protected

    # Pedestrian timing
    min_walk_s: float = 7.0                   # WALK signal minimum
    ped_clearance_speed_ft_per_s: float = 3.5 # Walking speed for clearance calc
    default_crosswalk_distance_ft: float = 48.0  # Typical 4-lane crossing

    # Cycle-level bounds
    min_cycle_s: float = 45.0    # Shortest cycle under low traffic
    max_cycle_s: float = 150.0   # Longest cycle under heavy traffic
    default_cycle_s: float = 90.0

    @property
    def ped_clearance_s(self) -> float:
        """Flashing DON'T WALK duration derived from crosswalk distance."""
        return self.default_crosswalk_distance_ft / self.ped_clearance_speed_ft_per_s


# ---------------------------------------------------------------------------
# Saturation flow defaults (veh/hr/lane)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlowDefaults:
    """
    Saturation flow = max vehicles that can discharge per hour of green
    through a single lane when there is a standing queue.

    Typical values for standard US intersections.
    """
    through_lane: float = 1_800.0    # veh/hr/lane
    left_turn_lane: float = 1_600.0  # Lower due to turning movement


# ---------------------------------------------------------------------------
# Vision / detection provider config
# ---------------------------------------------------------------------------

@dataclass
class VisionConfig:
    """Configuration passed to the VehicleDetectionProvider."""
    provider_type: str = "mock"              # "yolov8" | "mock"
    confidence_threshold: float = 0.5
    device: str = "cpu"                      # "cpu" | "cuda" | "mps"
    input_resolution: int = 640
    target_fps: float = 3.0                  # Frames to process per second
    model_path: str = "yolov8n.pt"           # Only used by yolov8 provider

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for passing to provider.initialize()."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "device": self.device,
            "input_resolution": self.input_resolution,
            "model_path": self.model_path,
        }


# ---------------------------------------------------------------------------
# Lane geometry (for ROI-based counting)
# ---------------------------------------------------------------------------

@dataclass
class LaneROI:
    """
    Region of interest polygon for a single lane, defined as normalized
    (0-1) coordinates relative to the camera frame.

    Vehicles whose bounding-box center falls inside this polygon are
    counted as being in this lane.
    """
    direction: Direction
    lane_type: LaneType
    polygon: list[tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Intersection-level config
# ---------------------------------------------------------------------------

@dataclass
class IntersectionConfig:
    """Top-level configuration for a single 4-way intersection."""
    name: str = "Main & 1st"
    timing: TimingConstraints = field(default_factory=TimingConstraints)
    flow: FlowDefaults = field(default_factory=FlowDefaults)
    vision: VisionConfig = field(default_factory=VisionConfig)
    lane_rois: list[LaneROI] = field(default_factory=list)

    # Approach speed for yellow/all-red calculations
    approach_speed_mph: float = 35.0

    # Controller tick rate
    controller_hz: float = 10.0

    # Dashboard
    dashboard_enabled: bool = True

    @property
    def tick_interval_s(self) -> float:
        return 1.0 / self.controller_hz

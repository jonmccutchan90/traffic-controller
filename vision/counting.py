"""
ROI-based vehicle counting.

Takes raw DetectionResults from any provider and counts how many
vehicles fall within each lane's region of interest polygon.
"""

from __future__ import annotations

from dataclasses import dataclass

from traffic_controller.config import Direction, LaneROI, LaneType
from traffic_controller.vision.provider import DetectedVehicle, DetectionResult


@dataclass
class LaneCount:
    """Vehicle count for a single lane after ROI filtering."""
    direction: Direction
    lane_type: LaneType
    count: int


def point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    """
    Ray-casting algorithm to test if a point is inside a polygon.

    Works with normalized (0-1) coordinates.
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def count_vehicles_by_lane(
    result: DetectionResult,
    lane_rois: list[LaneROI],
) -> list[LaneCount]:
    """
    Filter detections by lane ROI polygons and return counts per lane.

    Each vehicle's bounding-box center (x, y) is tested against each
    lane polygon. A vehicle is counted in the first matching lane.
    """
    counts: dict[tuple[str, str], int] = {}
    for roi in lane_rois:
        key = (roi.direction.value, roi.lane_type.value)
        counts[key] = 0

    for vehicle in result.vehicles:
        for roi in lane_rois:
            if not roi.polygon:
                continue
            if point_in_polygon(vehicle.x, vehicle.y, roi.polygon):
                key = (roi.direction.value, roi.lane_type.value)
                counts[key] = counts.get(key, 0) + 1
                break  # Each vehicle counted once

    return [
        LaneCount(
            direction=roi.direction,
            lane_type=roi.lane_type,
            count=counts.get((roi.direction.value, roi.lane_type.value), 0),
        )
        for roi in lane_rois
    ]


def count_vehicles_simple(
    result: DetectionResult,
) -> dict[str, int]:
    """
    Simple quadrant-based counting when no ROI polygons are defined.

    Splits the frame into four quadrants and maps them to directions
    based on a top-down camera view convention:
      - Top    (y < 0.3) → NORTH approach (cars heading south)
      - Bottom (y > 0.7) → SOUTH approach (cars heading north)
      - Right  (x > 0.7) → EAST approach  (cars heading west)
      - Left   (x < 0.3) → WEST approach  (cars heading east)

    Vehicles in the center are not counted (they're in the intersection).
    """
    counts = {d.value: 0 for d in Direction}

    for v in result.vehicles:
        if v.y < 0.3:
            counts["N"] += 1
        elif v.y > 0.7:
            counts["S"] += 1
        elif v.x > 0.7:
            counts["E"] += 1
        elif v.x < 0.3:
            counts["W"] += 1
        # else: vehicle is in the intersection, skip

    return counts

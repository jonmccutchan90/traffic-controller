"""
Abstract interface for vehicle detection providers.

Any CV backend (YOLO, SSD, background subtraction, mock) implements
this interface so the controller is decoupled from the detection model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Provider-agnostic data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectedVehicle:
    """
    A single detected vehicle.

    All coordinates are normalized to [0, 1] relative to the frame
    so that providers with different input resolutions produce
    comparable output.
    """
    x: float              # bounding-box center X (0-1)
    y: float              # bounding-box center Y (0-1)
    width: float          # bounding-box width (0-1)
    height: float         # bounding-box height (0-1)
    confidence: float     # detection confidence (0-1)
    vehicle_type: str     # "car", "truck", "bus", "motorcycle", "unknown"


@dataclass(frozen=True)
class DetectionResult:
    """
    One detection pass over a single frame.

    Every provider returns exactly this shape regardless of the
    underlying model or technique.
    """
    vehicles: list[DetectedVehicle]
    frame_timestamp: float        # time.monotonic() when frame was captured
    processing_time_ms: float     # wall-clock inference duration
    confidence_threshold: float   # threshold that was applied
    provider_name: str            # e.g. "yolov8-nano", "mock"

    @property
    def count(self) -> int:
        return len(self.vehicles)


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class VehicleDetectionProvider(ABC):
    """
    Interface that any vehicle-detection backend must implement.

    Lifecycle:
      1. Construct the provider
      2. Call initialize(config) once at startup
      3. Call detect(frame) repeatedly from the controller loop
      4. Call shutdown() on teardown

    Config dict contract (providers SHOULD support these keys and
    MUST silently ignore keys they don't understand):
      - confidence_threshold : float (default 0.5)
      - device              : str   ("cpu", "cuda", "mps")
      - input_resolution    : int   (e.g. 640)
      - model_path          : str   (path to weights file)
    """

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """
        Load model weights, allocate resources, warm up.

        Called exactly once before the first detect() call.
        """
        ...

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a single BGR frame (OpenCV convention).

        Returns ALL vehicles found in the frame. Spatial filtering
        (e.g. by lane ROI) is the caller's responsibility.

        The frame's dtype is uint8, shape is (H, W, 3).
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release model and free resources."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this provider instance."""
        ...

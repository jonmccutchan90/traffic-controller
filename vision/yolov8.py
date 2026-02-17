"""
YOLOv8 vehicle detection provider.

Uses the `ultralytics` package. Supports nano/small/medium/large model
variants via the model_path config key.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from traffic_controller.vision.provider import (
    DetectedVehicle,
    DetectionResult,
    VehicleDetectionProvider,
)

logger = logging.getLogger(__name__)


class YOLOv8Provider(VehicleDetectionProvider):
    """
    Concrete provider wrapping Ultralytics YOLOv8.

    Config keys used:
      - model_path           : str   (default "yolov8n.pt")
      - confidence_threshold : float (default 0.5)
      - device               : str   (default "cpu")
      - input_resolution     : int   (default 640)
    """

    # COCO class IDs for vehicle types
    VEHICLE_CLASSES: dict[int, str] = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self) -> None:
        self._model: Any = None
        self._conf: float = 0.5
        self._device: str = "cpu"
        self._imgsz: int = 640
        self._model_name: str = "yolov8n"

    def initialize(self, config: dict[str, Any]) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "ultralytics package is required for YOLOv8Provider. "
                "Install with: pip install ultralytics"
            ) from e

        model_path = config.get("model_path", "yolov8n.pt")
        self._device = config.get("device", "cpu")
        self._conf = config.get("confidence_threshold", 0.5)
        self._imgsz = config.get("input_resolution", 640)
        self._model_name = model_path.replace(".pt", "")

        logger.info(
            "Loading YOLOv8 model: %s (device=%s, conf=%.2f, imgsz=%d)",
            model_path, self._device, self._conf, self._imgsz,
        )
        self._model = YOLO(model_path)

        # Warm-up inference to load weights into memory / compile graph
        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._model.predict(dummy, device=self._device, verbose=False)
        logger.info("YOLOv8 warm-up complete")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self._model is None:
            raise RuntimeError("Provider not initialized — call initialize() first")

        t0 = time.monotonic()
        results = self._model.predict(
            frame,
            device=self._device,
            conf=self._conf,
            imgsz=self._imgsz,
            verbose=False,
        )[0]
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        h, w = frame.shape[:2]
        vehicles: list[DetectedVehicle] = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            vehicles.append(
                DetectedVehicle(
                    x=(x1 + x2) / 2.0 / w,
                    y=(y1 + y2) / 2.0 / h,
                    width=(x2 - x1) / w,
                    height=(y2 - y1) / h,
                    confidence=float(box.conf[0]),
                    vehicle_type=self.VEHICLE_CLASSES[cls_id],
                )
            )

        return DetectionResult(
            vehicles=vehicles,
            frame_timestamp=time.monotonic(),
            processing_time_ms=elapsed_ms,
            confidence_threshold=self._conf,
            provider_name=self._model_name,
        )

    def shutdown(self) -> None:
        logger.info("Shutting down YOLOv8 provider")
        if self._model is not None:
            del self._model
            self._model = None

    @property
    def name(self) -> str:
        return f"yolov8-{self._model_name}"

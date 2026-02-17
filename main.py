"""
Entry point for the Adaptive Traffic Light Controller.

Usage:
  python main.py                    # Run with dashboard (default)
  python main.py --headless         # Run without dashboard
  python main.py --provider yolov8  # Use YOLOv8 instead of mock
  python main.py --help             # Show all options
"""

from __future__ import annotations

import argparse
import logging
import sys

from traffic_controller.config import IntersectionConfig, VisionConfig
from traffic_controller.controller import TrafficController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Traffic Light Controller with Computer Vision",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the visual dashboard",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "yolov8"],
        default="mock",
        help="Vehicle detection provider (default: mock)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Controller tick rate in Hz (default: 10)",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Stop after N ticks (default: run forever)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("traffic_controller")

    # Build config
    config = IntersectionConfig(
        name="Main & 1st",
        vision=VisionConfig(
            provider_type=args.provider,
            confidence_threshold=args.confidence,
            device=args.device,
            model_path=args.model,
        ),
        controller_hz=args.hz,
        dashboard_enabled=not args.headless,
    )

    # Create and set up the controller
    controller = TrafficController(config=config)
    controller.setup()

    # Set up dashboard if enabled
    dashboard = None
    if config.dashboard_enabled:
        try:
            from traffic_controller.dashboard.display import Dashboard
            dashboard = Dashboard()
            dashboard.setup()
            controller.on_tick.append(dashboard.update)
            logger.info("Dashboard enabled")
        except ImportError:
            logger.warning("pygame not installed — running headless")

    # Run
    logger.info("Starting controller (provider=%s, hz=%.0f)", args.provider, args.hz)
    try:
        controller.run(max_ticks=args.max_ticks)
    finally:
        if dashboard:
            dashboard.teardown()
        controller.teardown()
        logger.info("Controller stopped")


if __name__ == "__main__":
    main()

# Adaptive Traffic Light Controller

Real-time adaptive traffic light controller with computer-vision-inspired elastic timing. A single 4-way intersection manages green splits dynamically based on queue lengths, using a SCATS-style algorithm.

## [Live Demo →](https://jonmccutchan90.github.io/traffic-controller/)

## Features

- **Adaptive timing** — SCATS-style algorithm adjusts green splits and cycle length based on real-time queue demand
- **Protected-permissive left turns** — Automatically switches between green arrow (protected) and flashing yellow (permissive yield) based on left-turn queue depth
- **Emergency vehicle preemption** — Safe transition through yellow → all-red → green for the emergency approach
- **Conflict monitor** — Independent safety watchdog that forces all-way flashing red if conflicting greens are ever detected
- **Pedestrian signals** — WALK / flashing DON'T WALK with timing derived from crosswalk distance
- **Swappable CV providers** — Abstract `VehicleDetectionProvider` interface with YOLOv8 and mock implementations

## Architecture

```
traffic_controller/
├── config.py               # Timing constraints, flow defaults, vision config
├── controller.py            # Main orchestration loop
├── main.py                  # CLI entry point
├── models/
│   ├── intersection.py      # Lane / Approach / Intersection data model
│   └── signal.py            # NEMA 8-phase signal state machine
├── vision/
│   ├── provider.py          # Abstract detection provider interface
│   ├── yolov8.py            # YOLOv8 provider (ultralytics)
│   ├── mock.py              # Synthetic traffic generator
│   └── counting.py          # ROI-based vehicle counting
├── timing/
│   ├── adaptive.py          # SCATS-style adaptive algorithm
│   └── constraints.py       # Safety constraint enforcer
├── safety/
│   ├── conflict.py          # Independent conflict monitor
│   └── preemption.py        # Emergency vehicle preemption manager
├── dashboard/
│   └── display.py           # Pygame visualization
├── tests/                   # 28 tests (state machine, timing, safety)
└── docs/                    # GitHub Pages web demo (HTML5 Canvas)
```

## Quick Start

### Web Demo (no install)

Visit the [live demo](https://jonmccutchan90.github.io/traffic-controller/) — runs entirely in your browser.

### Python (Pygame dashboard)

```bash
pip install numpy pygame
python main.py
```

### Python (headless)

```bash
pip install numpy
python main.py --headless --max-ticks 1000
```

### With YOLOv8

```bash
pip install numpy ultralytics opencv-python pygame
python main.py --provider yolov8
```

## Controls

| Key | Action |
|-----|--------|
| N / S / E / W | Trigger emergency preemption from that direction |
| C | Clear active preemption |
| Space | Pause / resume |
| Q / Esc | Quit (Pygame only) |
| 1-5 | Simulation speed (web demo) |

## Key Concepts

- **Saturation flow**: Max vehicles per hour per lane under ideal green (1,800 veh/hr for through, 1,600 for left turn). Used to convert queue counts into required green time.
- **Degree of saturation (DS)**: Queue / capacity ratio. Near 1.0 = phase is maxed out.
- **Webster's formula**: Optimal cycle = (1.5L + 5) / (1 - Y), where L = total lost time, Y = sum of critical flow ratios.

## Tests

```bash
cd ~/projects
python3 -m pytest traffic_controller/tests/ -v
```

## License

MIT

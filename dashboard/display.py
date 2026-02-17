"""
Pygame-based visual dashboard for the traffic controller.

Renders:
  - Intersection layout with roads and lane markings
  - Signal heads (colored circles) with current state
  - Queue indicators (car count bars) per approach
  - Phase information and countdown timer
  - Real-time statistics panel
  - Keyboard controls for preemption and simulation

Controls:
  - N/S/E/W keys: trigger emergency preemption from that direction
  - C: clear active preemption
  - Q or ESC: quit
  - SPACE: pause/resume
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traffic_controller.controller import TrafficController

logger = logging.getLogger(__name__)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (40, 40, 40)
GRAY = (100, 100, 100)
LIGHT_GRAY = (180, 180, 180)
ROAD_COLOR = (60, 60, 60)
LANE_MARKING = (200, 200, 100)
GRASS_COLOR = (50, 120, 50)

RED = (220, 30, 30)
YELLOW = (240, 220, 40)
GREEN = (30, 200, 50)
DARK_RED = (80, 10, 10)
DARK_YELLOW = (80, 75, 15)
DARK_GREEN = (10, 65, 18)
FLASH_YELLOW = (255, 200, 0)

BLUE = (50, 120, 220)
ORANGE = (240, 140, 30)


def _signal_color(state_name: str, tick: int = 0) -> tuple[int, int, int]:
    """Map a signal state name to a display color."""
    mapping = {
        "RED": RED,
        "GREEN": GREEN,
        "YELLOW": YELLOW,
        "GREEN_ARROW": GREEN,
        "YELLOW_ARROW": YELLOW,
        "FLASHING_YELLOW": FLASH_YELLOW if tick % 20 < 10 else DARK_YELLOW,
        "ALL_RED": RED,
        "DARK": DARK_GRAY,
        "WALK": WHITE,
        "PED_CLEARANCE": ORANGE if tick % 10 < 5 else DARK_GRAY,
        "DONT_WALK": RED,
    }
    return mapping.get(state_name, GRAY)


@dataclass
class Dashboard:
    """
    Pygame dashboard for visualizing the traffic controller.

    Call setup() once, then call update() from the controller's on_tick.
    """

    width: int = 1000
    height: int = 700
    fps: int = 30

    # --- pygame objects (initialized in setup) ---
    _screen: object = field(default=None, repr=False)
    _clock: object = field(default=None, repr=False)
    _font: object = field(default=None, repr=False)
    _font_small: object = field(default=None, repr=False)
    _font_large: object = field(default=None, repr=False)

    _paused: bool = False
    _tick_count: int = 0

    def setup(self) -> None:
        """Initialize Pygame and create the window."""
        try:
            import pygame
        except ImportError:
            logger.error("pygame is required for the dashboard. Install with: pip install pygame")
            raise

        pygame.init()
        self._screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Adaptive Traffic Controller")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14)
        self._font_small = pygame.font.SysFont("monospace", 11)
        self._font_large = pygame.font.SysFont("monospace", 20, bold=True)

        logger.info("Dashboard initialized (%dx%d)", self.width, self.height)

    def update(self, controller: TrafficController) -> None:
        """
        Render one frame. Called from the controller's on_tick callback.

        Also handles Pygame events (keyboard input, quit).
        """
        import pygame

        self._tick_count += 1

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                controller.is_running = False
                return
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key, controller)

        if self._paused:
            return

        screen = self._screen
        screen.fill(GRASS_COLOR)

        status = controller.get_full_status()

        # Draw layers
        self._draw_intersection(screen)
        self._draw_signals(screen, status)
        self._draw_queues(screen, status)
        self._draw_info_panel(screen, status)
        self._draw_controls_help(screen)

        pygame.display.flip()
        self._clock.tick(self.fps)

    def teardown(self) -> None:
        """Clean up Pygame."""
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass

    # --- keyboard handling ---

    def _handle_key(self, key: int, controller: TrafficController) -> None:
        import pygame
        from traffic_controller.config import Direction

        key_map = {
            pygame.K_n: Direction.NORTH,
            pygame.K_s: Direction.SOUTH,
            pygame.K_e: Direction.EAST,
            pygame.K_w: Direction.WEST,
        }

        if key in key_map:
            controller.trigger_preemption(key_map[key])
        elif key == pygame.K_c:
            controller.clear_preemption()
        elif key == pygame.K_SPACE:
            self._paused = not self._paused
        elif key in (pygame.K_q, pygame.K_ESCAPE):
            controller.is_running = False

    # --- drawing methods ---

    def _draw_intersection(self, screen: object) -> None:
        """Draw the road layout."""
        import pygame

        cx, cy = 350, 350  # Intersection center
        road_w = 80        # Road width
        half = road_w // 2

        # Horizontal road
        pygame.draw.rect(screen, ROAD_COLOR, (0, cy - half, 700, road_w))
        # Vertical road
        pygame.draw.rect(screen, ROAD_COLOR, (cx - half, 0, road_w, 700))

        # Lane markings (center dashed lines)
        for i in range(0, 700, 20):
            # Vertical road center line
            if not (cy - half - 5 < i < cy + half + 5):
                pygame.draw.line(screen, LANE_MARKING, (cx, i), (cx, i + 10), 2)
            # Horizontal road center line
            if not (cx - half - 5 < i < cx + half + 5):
                pygame.draw.line(screen, LANE_MARKING, (i, cy), (i + 10, cy), 2)

        # Stop lines
        stop_offset = half + 3
        line_len = half - 2
        # North stop line
        pygame.draw.line(screen, WHITE, (cx - line_len, cy - stop_offset),
                         (cx + line_len, cy - stop_offset), 3)
        # South stop line
        pygame.draw.line(screen, WHITE, (cx - line_len, cy + stop_offset),
                         (cx + line_len, cy + stop_offset), 3)
        # East stop line
        pygame.draw.line(screen, WHITE, (cx + stop_offset, cy - line_len),
                         (cx + stop_offset, cy + line_len), 3)
        # West stop line
        pygame.draw.line(screen, WHITE, (cx - stop_offset, cy - line_len),
                         (cx - stop_offset, cy + line_len), 3)

        # Crosswalk markings
        cw = 8
        for offset in range(-half + 5, half - 5, 12):
            # N crosswalk
            pygame.draw.rect(screen, WHITE, (cx + offset - 3, cy - half - 15, 6, 12))
            # S crosswalk
            pygame.draw.rect(screen, WHITE, (cx + offset - 3, cy + half + 3, 6, 12))
            # E crosswalk
            pygame.draw.rect(screen, WHITE, (cx + half + 3, cy + offset - 3, 12, 6))
            # W crosswalk
            pygame.draw.rect(screen, WHITE, (cx - half - 15, cy + offset - 3, 12, 6))

    def _draw_signals(self, screen: object, status: dict) -> None:
        """Draw signal heads for each direction."""
        import pygame

        cx, cy = 350, 350
        signals = status.get("signals", {}).get("signals", {})

        # Signal head positions (outside the intersection)
        positions = {
            "N": (cx + 55, cy - 70),
            "S": (cx - 55, cy + 70),
            "E": (cx + 70, cy + 55),
            "W": (cx - 70, cy - 55),
        }

        for dir_key, pos in positions.items():
            sig = signals.get(dir_key, {})
            x, y = pos

            # Background box
            pygame.draw.rect(screen, DARK_GRAY, (x - 15, y - 35, 30, 70), 0, 5)
            pygame.draw.rect(screen, GRAY, (x - 15, y - 35, 30, 70), 2, 5)

            # Red light
            veh_state = sig.get("vehicle", "RED")
            r_color = _signal_color("RED") if veh_state == "RED" else DARK_RED
            pygame.draw.circle(screen, r_color, (x, y - 20), 10)

            # Yellow light
            y_color = _signal_color("YELLOW") if veh_state == "YELLOW" else DARK_YELLOW
            pygame.draw.circle(screen, y_color, (x, y), 10)

            # Green light
            g_color = _signal_color("GREEN") if veh_state == "GREEN" else DARK_GREEN
            pygame.draw.circle(screen, g_color, (x, y + 20), 10)

            # Left-turn indicator (small arrow icon next to main signal)
            lt_state = sig.get("left_turn", "RED")
            lt_color = _signal_color(lt_state, self._tick_count)
            pygame.draw.circle(screen, lt_color, (x + 20, y), 6)

            # Direction label
            label = self._font_small.render(dir_key, True, WHITE)
            screen.blit(label, (x - 4, y + 35))

            # Pedestrian signal
            ped_state = sig.get("pedestrian", "DONT_WALK")
            ped_color = _signal_color(ped_state, self._tick_count)
            pygame.draw.rect(screen, DARK_GRAY, (x - 8, y + 48, 16, 16), 0, 3)
            pygame.draw.rect(screen, ped_color, (x - 5, y + 51, 10, 10), 0, 2)

    def _draw_queues(self, screen: object, status: dict) -> None:
        """Draw queue length bars and counts for each approach."""
        import pygame

        cx, cy = 350, 350
        queues = status.get("intersection", {})

        # Bar positions and orientations
        bar_config = {
            "N": {"pos": (cx - 30, 30), "horizontal": False, "label_pos": (cx - 45, 10)},
            "S": {"pos": (cx + 10, cy + 120), "horizontal": False, "label_pos": (cx - 5, cy + 100)},
            "E": {"pos": (cx + 120, cy - 30), "horizontal": True, "label_pos": (cx + 100, cy - 50)},
            "W": {"pos": (30, cy + 10), "horizontal": True, "label_pos": (10, cy - 10)},
        }

        max_bar = 200  # Max bar length in pixels
        max_queue = 20  # Queue count that fills the bar

        for dir_key, cfg in bar_config.items():
            q = queues.get(dir_key, {"through": 0, "left_turn": 0})
            through = q.get("through", 0)
            left_turn = q.get("left_turn", 0)

            x, y = cfg["pos"]
            lx, ly = cfg["label_pos"]

            # Through lane bar
            t_len = min(max_bar, int((through / max_queue) * max_bar))
            t_color = GREEN if through < 5 else (YELLOW if through < 12 else RED)

            if cfg["horizontal"]:
                pygame.draw.rect(screen, DARK_GRAY, (x, y, max_bar, 15), 0, 3)
                pygame.draw.rect(screen, t_color, (x, y, t_len, 15), 0, 3)
                # Left-turn bar below
                lt_len = min(max_bar, int((left_turn / max_queue) * max_bar))
                lt_color = GREEN if left_turn < 3 else (YELLOW if left_turn < 6 else RED)
                pygame.draw.rect(screen, DARK_GRAY, (x, y + 20, max_bar, 10), 0, 3)
                pygame.draw.rect(screen, lt_color, (x, y + 20, lt_len, 10), 0, 3)
            else:
                pygame.draw.rect(screen, DARK_GRAY, (x, y, 15, max_bar), 0, 3)
                pygame.draw.rect(screen, t_color, (x, y + max_bar - t_len, 15, t_len), 0, 3)
                # Left-turn bar beside
                lt_len = min(max_bar, int((left_turn / max_queue) * max_bar))
                lt_color = GREEN if left_turn < 3 else (YELLOW if left_turn < 6 else RED)
                pygame.draw.rect(screen, DARK_GRAY, (x + 20, y, 10, max_bar), 0, 3)
                pygame.draw.rect(screen, lt_color, (x + 20, y + max_bar - lt_len, 10, lt_len), 0, 3)

            # Labels
            label = self._font_small.render(f"{dir_key}: T={through} L={left_turn}", True, WHITE)
            screen.blit(label, (lx, ly))

    def _draw_info_panel(self, screen: object, status: dict) -> None:
        """Draw the information panel on the right side."""
        import pygame

        panel_x = 720
        panel_y = 20
        panel_w = 260
        panel_h = 660

        # Panel background
        pygame.draw.rect(screen, (30, 30, 40), (panel_x, panel_y, panel_w, panel_h), 0, 8)
        pygame.draw.rect(screen, GRAY, (panel_x, panel_y, panel_w, panel_h), 2, 8)

        x = panel_x + 15
        y = panel_y + 15

        # Title
        title = self._font_large.render("Traffic Controller", True, WHITE)
        screen.blit(title, (x, y))
        y += 30

        # Signal status
        sig = status.get("signals", {})
        phase_info = f"Phase {sig.get('phase', '?')} ({sig.get('phase_type', '?')})"
        step_info = f"Step: {sig.get('step', '?')}"
        remain_info = f"Remaining: {sig.get('step_remaining_s', 0):.1f}s"
        cycle_info = f"Cycle: {sig.get('cycle', 0)}"

        for text in [phase_info, step_info, remain_info, cycle_info]:
            surface = self._font.render(text, True, LIGHT_GRAY)
            screen.blit(surface, (x, y))
            y += 18

        y += 10
        pygame.draw.line(screen, GRAY, (x, y), (x + panel_w - 30, y))
        y += 10

        # Cycle time
        cycle_time = status.get("cycle_time_s", 0)
        ct_text = self._font.render(f"Cycle time: {cycle_time:.0f}s", True, LIGHT_GRAY)
        screen.blit(ct_text, (x, y))
        y += 22

        # Phase demands
        timing = status.get("timing", {})
        demands = timing.get("phase_demands", [])
        header = self._font.render("Phase Demands:", True, WHITE)
        screen.blit(header, (x, y))
        y += 18

        for d in demands:
            pid = d.get("phase_id", "?")
            q = d.get("queue", 0)
            ds = d.get("ds", 0)
            ig = d.get("ideal_green_s", 0)
            prot = "*" if d.get("protected_left", False) else " "

            color = GREEN if ds < 0.5 else (YELLOW if ds < 0.85 else RED)
            line = f"P{pid}: Q={q:>3} DS={ds:.2f} G={ig:>4.0f}s{prot}"
            surface = self._font_small.render(line, True, color)
            screen.blit(surface, (x, y))
            y += 16

        y += 10
        pygame.draw.line(screen, GRAY, (x, y), (x + panel_w - 30, y))
        y += 10

        # Preemption status
        preempt = status.get("preemption", {})
        preempt_active = preempt.get("active")
        if preempt_active:
            p_color = RED
            p_text = f"PREEMPTION: {preempt_active} ({preempt.get('hold_elapsed_s', 0):.0f}s)"
        else:
            p_color = DARK_GREEN
            p_text = "No preemption active"
        surface = self._font.render(p_text, True, p_color)
        screen.blit(surface, (x, y))
        y += 20

        # Conflict monitor
        conflict = status.get("conflict_monitor", {})
        if conflict.get("fault_active", False):
            c_color = RED
            c_text = f"FAULT! Conflicts: {conflict.get('conflict_count', 0)}"
        else:
            c_color = DARK_GREEN
            c_text = "Conflict monitor: OK"
        surface = self._font.render(c_text, True, c_color)
        screen.blit(surface, (x, y))
        y += 22

        y += 5
        pygame.draw.line(screen, GRAY, (x, y), (x + panel_w - 30, y))
        y += 10

        # Queue totals
        queues = status.get("intersection", {})
        header = self._font.render("Queue Totals:", True, WHITE)
        screen.blit(header, (x, y))
        y += 18

        total = 0
        for dir_key in ["N", "S", "E", "W"]:
            q = queues.get(dir_key, {"through": 0, "left_turn": 0})
            t = q.get("through", 0)
            lt = q.get("left_turn", 0)
            total += t + lt
            line = f"  {dir_key}: {t:>3} through, {lt:>2} left"
            surface = self._font_small.render(line, True, LIGHT_GRAY)
            screen.blit(surface, (x, y))
            y += 16

        total_line = f"  Total: {total} vehicles"
        surface = self._font.render(total_line, True, WHITE)
        screen.blit(surface, (x, y))
        y += 25

        # Tick count
        tick_text = f"Tick: {status.get('tick', 0)}"
        surface = self._font_small.render(tick_text, True, GRAY)
        screen.blit(surface, (x, y))

    def _draw_controls_help(self, screen: object) -> None:
        """Draw keyboard controls at the bottom."""
        import pygame

        y = self.height - 25
        help_text = "N/S/E/W=Preempt  C=Clear  SPACE=Pause  Q=Quit"
        surface = self._font_small.render(help_text, True, LIGHT_GRAY)
        screen.blit(surface, (20, y))

        if self._paused:
            pause_text = self._font_large.render("PAUSED", True, YELLOW)
            screen.blit(pause_text, (self.width // 2 - 40, self.height // 2 - 15))

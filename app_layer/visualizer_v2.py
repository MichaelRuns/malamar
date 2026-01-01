"""
Enhanced visualizer for tile-based perception
Shows raw frame with grid overlay, alphabetic/numeric labels, regions of interest,
and AI agent status panel
"""

import cv2
import numpy as np
from typing import Optional, List, Dict
from perception_v2 import (
    PerceptionState, GRID_ROWS, GRID_COLS, TILE_SIZE, REGIONS_OF_INTEREST,
    WordInfo, MenuSelection, BattleHP, BattleNames
)


class VisualizerV2:
    """Enhanced visualizer with tile grid and labels"""

    def __init__(self, window_name: str = "Perception V2 - Tile Grid"):
        self.window_name = window_name
        self.scale = 4  # Scale factor for display
        self.display_width = 160 * self.scale
        self.display_height = 144 * self.scale
        self.label_margin = 30  # Space for labels

        # Label grid panel settings
        self.label_grid_char_width = 12  # Pixels per character in label grid
        self.label_grid_char_height = 16  # Pixels per row in label grid
        self.label_grid_width = GRID_COLS * self.label_grid_char_width + 20  # 20 cols + padding
        self.label_grid_height = GRID_ROWS * self.label_grid_char_height + 40  # 18 rows + header

        # State info panel settings
        self.state_panel_width = 280
        self.state_panel_height = 320

        # Agent panel settings
        self.agent_panel_width = 320
        self.agent_panel_height = 400

        # Total canvas size (frame + gap + label grid + state panel + agent panel)
        self.gap = 20  # Gap between panels
        self.canvas_width = (self.display_width + self.label_margin * 2 +
                            self.gap + self.label_grid_width +
                            self.gap + self.state_panel_width +
                            self.gap + self.agent_panel_width)
        self.canvas_height = max(self.display_height + self.label_margin * 2,
                                self.label_grid_height + self.label_margin,
                                self.state_panel_height + self.label_margin,
                                self.agent_panel_height + self.label_margin)

        self.created = False
        self.show_grid = True
        self.show_labels = True
        self.show_regions = True
        self.show_label_grid = True  # Toggle for label grid panel
        self.show_state_panel = True  # Toggle for state info panel
        self.show_agent_panel = True  # Toggle for agent panel

        # Cache for expensive state panel computations
        self._state_cache_hash = None
        self._cached_menu = None
        self._cached_hp = None
        self._cached_names = None
        self._cached_words = None
        self._cached_groups = None

    def visualize(self, frame: np.ndarray, state: Optional[PerceptionState] = None,
                  agent_status: Optional[Dict] = None):
        """
        Visualize frame with tile grid overlay

        Args:
            frame: Grayscale frame (144x160)
            state: Optional PerceptionState for enhanced visualization
            agent_status: Optional dict with agent status for display
        """
        # Create canvas with margins for labels
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        canvas.fill(20)  # Dark background

        # Convert frame to color and scale
        if len(frame.shape) == 2:
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_color = frame

        scaled_frame = cv2.resize(
            frame_color,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_NEAREST
        )

        # Place frame on canvas
        x_offset = self.label_margin
        y_offset = self.label_margin
        canvas[y_offset:y_offset+self.display_height,
               x_offset:x_offset+self.display_width] = scaled_frame

        # Draw tile grid
        if self.show_grid:
            self._draw_grid(canvas, x_offset, y_offset)

        # Draw grid labels (A-T horizontal, 0-17 vertical)
        if self.show_labels:
            self._draw_grid_labels(canvas, x_offset, y_offset)

        # Draw regions of interest
        if self.show_regions and state:
            self._draw_regions(canvas, x_offset, y_offset, state)

        # Draw info overlay
        if state:
            self._draw_info_overlay(canvas, state)

        # Draw label grid panel
        if self.show_label_grid and state:
            label_grid_x = self.display_width + self.label_margin * 2 + self.gap
            self._draw_label_grid_panel(canvas, label_grid_x, self.label_margin, state)

        # Draw state info panel
        if self.show_state_panel and state:
            state_panel_x = self.display_width + self.label_margin * 2 + self.gap + self.label_grid_width + self.gap
            self._draw_state_panel(canvas, state_panel_x, self.label_margin, state)

        # Draw agent panel
        if self.show_agent_panel:
            agent_panel_x = (self.display_width + self.label_margin * 2 +
                           self.gap + self.label_grid_width +
                           self.gap + self.state_panel_width + self.gap)
            self._draw_agent_panel(canvas, agent_panel_x, self.label_margin, agent_status)

        # Display
        cv2.imshow(self.window_name, canvas)
        self.created = True

    def _draw_grid(self, canvas: np.ndarray, x_offset: int, y_offset: int):
        """Draw tile grid overlay"""
        scaled_tile = TILE_SIZE * self.scale

        # Vertical lines
        for col in range(GRID_COLS + 1):
            x = x_offset + col * scaled_tile
            cv2.line(
                canvas,
                (x, y_offset),
                (x, y_offset + self.display_height),
                (0, 255, 0),
                1
            )

        # Horizontal lines
        for row in range(GRID_ROWS + 1):
            y = y_offset + row * scaled_tile
            cv2.line(
                canvas,
                (x_offset, y),
                (x_offset + self.display_width, y),
                (0, 255, 0),
                1
            )

    def _draw_grid_labels(self, canvas: np.ndarray, x_offset: int, y_offset: int):
        """Draw alphabetic column labels and numeric row labels"""
        scaled_tile = TILE_SIZE * self.scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (200, 200, 200)

        # Column labels (A-T) at top
        for col in range(GRID_COLS):
            label = chr(ord('A') + col)
            x = x_offset + col * scaled_tile + scaled_tile // 2 - 5
            y = y_offset - 10
            cv2.putText(canvas, label, (x, y), font, font_scale, color, thickness)

        # Row labels (0-17) on left
        for row in range(GRID_ROWS):
            label = str(row)
            x = x_offset - 25
            y = y_offset + row * scaled_tile + scaled_tile // 2 + 5
            cv2.putText(canvas, label, (x, y), font, font_scale, color, thickness)

    def _draw_regions(self, canvas: np.ndarray, x_offset: int, y_offset: int,
                      state: PerceptionState):
        """Draw regions of interest as colored overlays"""
        scaled_tile = TILE_SIZE * self.scale

        # Color scheme for different regions
        region_colors = {
            "text_box": (255, 100, 100),      # Red
            "player_area": (100, 255, 100),   # Green
            "enemy_hp": (255, 150, 100),      # Orange
            "player_hp": (100, 150, 255),     # Blue
            "battle_menu": (255, 255, 100),   # Yellow
        }

        # Create ONE overlay for all regions (O(1) copy instead of O(n))
        overlay = canvas.copy()

        for name, region in state.regions.items():
            color = region_colors.get(name, (128, 128, 128))

            # Calculate pixel bounds
            x1 = x_offset + region.tile_start[1] * scaled_tile
            y1 = y_offset + region.tile_start[0] * scaled_tile
            x2 = x_offset + (region.tile_end[1] + 1) * scaled_tile
            y2 = y_offset + (region.tile_end[0] + 1) * scaled_tile

            # Draw filled rectangle on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Blend ONCE at the end
        cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)

        # Draw borders and labels (no blending needed)
        for name, region in state.regions.items():
            color = region_colors.get(name, (128, 128, 128))

            x1 = x_offset + region.tile_start[1] * scaled_tile
            y1 = y_offset + region.tile_start[0] * scaled_tile
            x2 = x_offset + (region.tile_end[1] + 1) * scaled_tile
            y2 = y_offset + (region.tile_end[0] + 1) * scaled_tile

            # Draw border
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label_y = y1 - 5 if y1 > 20 else y2 + 15
            cv2.putText(
                canvas,
                name,
                (x1 + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1
            )

    def _draw_info_overlay(self, canvas: np.ndarray, state: PerceptionState):
        """Draw info overlay in top-left corner"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)

        info_lines = [
            f"Frame: {state.frame_number}",
            f"Hash: {state.screen_hash[:12]}...",
            f"Grid: {GRID_COLS}x{GRID_ROWS} ({GRID_COLS*GRID_ROWS} tiles)",
        ]

        y = 15
        for line in info_lines:
            cv2.putText(canvas, line, (10, y), font, font_scale, color, thickness)
            y += 18

        # Draw controls at bottom
        controls = [
            "G: Grid | R: Regions | T: Labels | I: State | O: Agent | Q: Quit"
        ]
        y = self.canvas_height - 10
        for line in controls:
            cv2.putText(canvas, line, (10, y), font, 0.35, (150, 150, 150), 1)
            y += 15

    def _draw_label_grid_panel(self, canvas: np.ndarray, x_offset: int, y_offset: int,
                                state: PerceptionState):
        """Draw the label grid panel showing tile labels as characters"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        # Draw panel background
        panel_width = self.label_grid_width
        panel_height = self.label_grid_height
        cv2.rectangle(canvas, (x_offset, y_offset),
                     (x_offset + panel_width, y_offset + panel_height),
                     (40, 40, 40), -1)  # Dark gray background
        cv2.rectangle(canvas, (x_offset, y_offset),
                     (x_offset + panel_width, y_offset + panel_height),
                     (100, 100, 100), 1)  # Border

        # Draw header
        cv2.putText(canvas, "Label Grid", (x_offset + 5, y_offset + 15),
                   font, 0.5, (255, 255, 255), 1)

        # Get display grid from state
        display_grid = state.get_display_grid()

        # Draw the grid of labels
        char_y = y_offset + 30  # Start below header
        for row, line in enumerate(display_grid):
            char_x = x_offset + 10
            for col, char in enumerate(line):
                # Color based on character type
                if char == "?":
                    color = (60, 60, 60)  # Dark gray for unlabeled
                elif char == " ":
                    # Walkable terrain - skip drawing (blank)
                    char_x += self.label_grid_char_width
                    continue
                elif char == "_":
                    color = (150, 150, 150)  # Light gray for space character
                elif char.isalpha() and char.isupper():
                    if char in "BDT":  # Terrain (B=blocked, D=door, T=water)
                        color = (100, 200, 100)  # Green for terrain
                    else:
                        color = (255, 255, 100)  # Yellow for text characters
                elif char.isdigit():
                    color = (255, 200, 100)  # Orange for numbers
                elif char == ">":
                    color = (255, 255, 0)  # Bright yellow for cursor
                elif char == "U":
                    color = (100, 150, 255)  # Blue for UI elements
                elif char == "@":
                    color = (255, 100, 100)  # Red for player sprite
                elif char == "N":
                    color = (255, 180, 100)  # Orange for NPC sprite
                elif char == "S":
                    color = (255, 150, 150)  # Light red for other sprites
                else:
                    color = (255, 100, 255)  # Magenta for punctuation/symbols

                cv2.putText(canvas, char, (char_x, char_y),
                           font, font_scale, color, thickness)
                char_x += self.label_grid_char_width

            char_y += self.label_grid_char_height

        # Draw row numbers on left
        char_y = y_offset + 30
        for row in range(GRID_ROWS):
            cv2.putText(canvas, f"{row:2d}", (x_offset - 20, char_y),
                       font, 0.3, (120, 120, 120), 1)
            char_y += self.label_grid_char_height

    def _draw_state_panel(self, canvas: np.ndarray, x_offset: int, y_offset: int,
                          state: PerceptionState):
        """Draw the perception state info panel"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        # Cache expensive computations based on screen_hash
        if self._state_cache_hash != state.screen_hash:
            self._state_cache_hash = state.screen_hash
            self._cached_menu = state.get_menu_selection()
            self._cached_hp = state.get_battle_hp()
            self._cached_names = state.get_battle_names()
            self._cached_words = state.extract_words()
            self._cached_groups = state.get_labeled_tiles_by_category()

        # Draw panel background
        panel_width = self.state_panel_width
        panel_height = self.state_panel_height
        cv2.rectangle(canvas, (x_offset, y_offset),
                     (x_offset + panel_width, y_offset + panel_height),
                     (30, 30, 40), -1)  # Dark blue-gray background
        cv2.rectangle(canvas, (x_offset, y_offset),
                     (x_offset + panel_width, y_offset + panel_height),
                     (100, 100, 120), 1)  # Border

        # Draw header
        cv2.putText(canvas, "Perception State", (x_offset + 5, y_offset + 18),
                   font, 0.5, (255, 255, 255), 1)

        y = y_offset + 40
        line_height = 16

        # --- Menu Selection ---
        cv2.putText(canvas, "MENU:", (x_offset + 5, y), font, font_scale, (150, 200, 255), thickness)
        y += line_height

        menu = self._cached_menu
        if menu.option:
            color = (0, 255, 0)  # Green for active selection
            cv2.putText(canvas, f"  > {menu.option}", (x_offset + 5, y), font, font_scale, color, thickness)
        elif menu.cursor_position:
            cv2.putText(canvas, f"  Cursor at {menu.cursor_position}", (x_offset + 5, y), font, font_scale, (180, 180, 180), thickness)
        else:
            cv2.putText(canvas, "  (no cursor)", (x_offset + 5, y), font, font_scale, (100, 100, 100), thickness)
        y += line_height + 8

        # --- Battle HP ---
        cv2.putText(canvas, "HP BARS:", (x_offset + 5, y), font, font_scale, (150, 200, 255), thickness)
        y += line_height

        hp = self._cached_hp

        # Enemy HP
        enemy_pct = hp.enemy.percentage
        if enemy_pct is not None and hp.enemy.hp_bar_tiles > 0:
            bar_width = 100
            filled = int(bar_width * enemy_pct)
            cv2.putText(canvas, "  Enemy:", (x_offset + 5, y), font, font_scale, (255, 150, 100), thickness)
            bar_x = x_offset + 70
            cv2.rectangle(canvas, (bar_x, y - 10), (bar_x + bar_width, y), (60, 60, 60), -1)
            if filled > 0:
                cv2.rectangle(canvas, (bar_x, y - 10), (bar_x + filled, y), (0, 200, 0), -1)
            cv2.putText(canvas, f"{int(enemy_pct * 100)}%", (bar_x + bar_width + 5, y), font, font_scale, (200, 200, 200), thickness)
        else:
            cv2.putText(canvas, "  Enemy: --", (x_offset + 5, y), font, font_scale, (100, 100, 100), thickness)
        y += line_height

        # Player HP
        player_pct = hp.player.percentage
        if player_pct is not None and hp.player.hp_bar_tiles > 0:
            bar_width = 100
            filled = int(bar_width * player_pct)
            cv2.putText(canvas, "  Player:", (x_offset + 5, y), font, font_scale, (100, 150, 255), thickness)
            bar_x = x_offset + 70
            cv2.rectangle(canvas, (bar_x, y - 10), (bar_x + bar_width, y), (60, 60, 60), -1)
            if filled > 0:
                cv2.rectangle(canvas, (bar_x, y - 10), (bar_x + filled, y), (0, 200, 0), -1)
            cv2.putText(canvas, f"{int(player_pct * 100)}%", (bar_x + bar_width + 5, y), font, font_scale, (200, 200, 200), thickness)
        else:
            cv2.putText(canvas, "  Player: --", (x_offset + 5, y), font, font_scale, (100, 100, 100), thickness)
        y += line_height + 8

        # --- Battle Names ---
        cv2.putText(canvas, "POKEMON:", (x_offset + 5, y), font, font_scale, (150, 200, 255), thickness)
        y += line_height

        names = self._cached_names
        if names.enemy_name:
            cv2.putText(canvas, f"  Enemy: {names.enemy_name}", (x_offset + 5, y), font, font_scale, (255, 150, 100), thickness)
        else:
            cv2.putText(canvas, "  Enemy: --", (x_offset + 5, y), font, font_scale, (100, 100, 100), thickness)
        y += line_height

        if names.player_name:
            cv2.putText(canvas, f"  Player: {names.player_name}", (x_offset + 5, y), font, font_scale, (100, 150, 255), thickness)
        else:
            cv2.putText(canvas, "  Player: --", (x_offset + 5, y), font, font_scale, (100, 100, 100), thickness)
        y += line_height + 8

        # --- Extracted Words ---
        cv2.putText(canvas, "WORDS:", (x_offset + 5, y), font, font_scale, (150, 200, 255), thickness)
        y += line_height

        words = self._cached_words
        if words:
            # Show up to 8 words
            word_strs = [w.word for w in words[:8]]
            # Wrap words into lines
            line = "  "
            for w in word_strs:
                if len(line) + len(w) + 1 > 35:
                    cv2.putText(canvas, line, (x_offset + 5, y), font, font_scale, (255, 255, 100), thickness)
                    y += line_height
                    line = "  " + w + " "
                else:
                    line += w + " "
            if line.strip():
                cv2.putText(canvas, line, (x_offset + 5, y), font, font_scale, (255, 255, 100), thickness)
                y += line_height
            if len(words) > 8:
                cv2.putText(canvas, f"  (+{len(words) - 8} more)", (x_offset + 5, y), font, font_scale, (120, 120, 120), thickness)
        else:
            cv2.putText(canvas, "  (none)", (x_offset + 5, y), font, font_scale, (100, 100, 100), thickness)
        y += line_height + 8

        # --- Tile Counts ---
        cv2.putText(canvas, "TILES:", (x_offset + 5, y), font, font_scale, (150, 200, 255), thickness)
        y += line_height

        groups = self._cached_groups
        cv2.putText(canvas, f"  Text: {len(groups.text)}", (x_offset + 5, y), font, font_scale, (255, 255, 100), thickness)
        cv2.putText(canvas, f"Terrain: {len(groups.terrain)}", (x_offset + 90, y), font, font_scale, (100, 200, 100), thickness)
        y += line_height
        cv2.putText(canvas, f"  UI: {len(groups.ui)}", (x_offset + 5, y), font, font_scale, (100, 150, 255), thickness)
        cv2.putText(canvas, f"Sprite: {len(groups.sprite)}", (x_offset + 90, y), font, font_scale, (255, 100, 100), thickness)
        y += line_height
        cv2.putText(canvas, f"  Unlabeled: {len(groups.unlabeled)}", (x_offset + 5, y), font, font_scale, (100, 100, 100), thickness)

    def toggle_grid(self):
        """Toggle grid visibility"""
        self.show_grid = not self.show_grid
        return self.show_grid

    def toggle_labels(self):
        """Toggle label visibility"""
        self.show_labels = not self.show_labels
        return self.show_labels

    def toggle_regions(self):
        """Toggle regions visibility"""
        self.show_regions = not self.show_regions
        return self.show_regions

    def toggle_label_grid(self):
        """Toggle label grid panel visibility"""
        self.show_label_grid = not self.show_label_grid
        return self.show_label_grid

    def toggle_state_panel(self):
        """Toggle state info panel visibility"""
        self.show_state_panel = not self.show_state_panel
        return self.show_state_panel

    def toggle_agent_panel(self):
        """Toggle agent panel visibility"""
        self.show_agent_panel = not self.show_agent_panel
        return self.show_agent_panel

    def _draw_agent_panel(self, canvas: np.ndarray, x_offset: int, y_offset: int,
                          agent_status: Optional[Dict]):
        """Draw the AI agent status and output panel"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        # Draw panel background
        panel_width = self.agent_panel_width
        panel_height = self.agent_panel_height
        cv2.rectangle(canvas, (x_offset, y_offset),
                     (x_offset + panel_width, y_offset + panel_height),
                     (25, 35, 45), -1)  # Dark blue background
        cv2.rectangle(canvas, (x_offset, y_offset),
                     (x_offset + panel_width, y_offset + panel_height),
                     (80, 120, 160), 1)  # Border

        # Draw header
        cv2.putText(canvas, "AI Agent", (x_offset + 5, y_offset + 18),
                   font, 0.5, (100, 180, 255), 1)

        y = y_offset + 40
        line_height = 16

        if agent_status is None:
            # Agent not initialized
            cv2.putText(canvas, "Agent: DISABLED", (x_offset + 5, y),
                       font, font_scale, (100, 100, 100), thickness)
            y += line_height
            cv2.putText(canvas, "Press 'M' then 'A' to enable", (x_offset + 5, y),
                       font, font_scale, (80, 80, 80), thickness)
            return

        # --- Status indicators ---
        cv2.putText(canvas, "STATUS:", (x_offset + 5, y), font, font_scale, (150, 200, 255), thickness)
        y += line_height

        # Processing indicator
        if agent_status.get("is_processing"):
            cv2.putText(canvas, "  Processing...", (x_offset + 5, y),
                       font, font_scale, (255, 255, 0), thickness)
            # Draw spinning indicator
            import time
            spin_chars = "|/-\\"
            spin_idx = int(time.time() * 4) % 4
            cv2.putText(canvas, spin_chars[spin_idx], (x_offset + 120, y),
                       font, font_scale, (255, 255, 0), thickness)
        else:
            cv2.putText(canvas, "  Idle", (x_offset + 5, y),
                       font, font_scale, (100, 200, 100), thickness)
        y += line_height

        # Time since last decision
        time_since = agent_status.get("time_since_decision")
        if time_since is not None:
            cv2.putText(canvas, f"  Last decision: {time_since:.1f}s ago", (x_offset + 5, y),
                       font, font_scale, (180, 180, 180), thickness)
        else:
            cv2.putText(canvas, "  Last decision: --", (x_offset + 5, y),
                       font, font_scale, (100, 100, 100), thickness)
        y += line_height

        # Pending actions
        pending = agent_status.get("pending_actions", 0)
        color = (100, 255, 100) if pending > 0 else (100, 100, 100)
        cv2.putText(canvas, f"  Pending actions: {pending}", (x_offset + 5, y),
                   font, font_scale, color, thickness)
        y += line_height + 8

        # --- Current Plan ---
        cv2.putText(canvas, "PLAN:", (x_offset + 5, y), font, font_scale, (150, 200, 255), thickness)
        y += line_height

        if agent_status.get("has_plan"):
            goal = agent_status.get("plan_goal", "Unknown")
            # Truncate goal if too long
            if len(goal) > 40:
                goal = goal[:37] + "..."
            cv2.putText(canvas, f"  Goal: {goal}", (x_offset + 5, y),
                       font, font_scale, (200, 255, 200), thickness)
            y += line_height

            steps = agent_status.get("plan_steps", 0)
            cv2.putText(canvas, f"  Steps: {steps}", (x_offset + 5, y),
                       font, font_scale, (180, 180, 180), thickness)
            y += line_height

            time_since_plan = agent_status.get("time_since_plan")
            if time_since_plan is not None:
                cv2.putText(canvas, f"  Plan age: {time_since_plan:.0f}s", (x_offset + 5, y),
                           font, font_scale, (150, 150, 150), thickness)
        else:
            cv2.putText(canvas, "  (no plan yet)", (x_offset + 5, y),
                       font, font_scale, (100, 100, 100), thickness)
        y += line_height + 8

        # --- Last Reasoning ---
        cv2.putText(canvas, "REASONING:", (x_offset + 5, y), font, font_scale, (150, 200, 255), thickness)
        y += line_height

        reasoning = agent_status.get("last_reasoning")
        if reasoning:
            # Word wrap the reasoning
            words = reasoning.split()
            line = "  "
            max_chars = 42
            lines_drawn = 0
            max_lines = 6

            for word in words:
                if len(line) + len(word) + 1 > max_chars:
                    if lines_drawn < max_lines:
                        cv2.putText(canvas, line, (x_offset + 5, y),
                                   font, font_scale, (255, 255, 200), thickness)
                        y += line_height
                        lines_drawn += 1
                    line = "  " + word + " "
                else:
                    line += word + " "

            if line.strip() and lines_drawn < max_lines:
                cv2.putText(canvas, line, (x_offset + 5, y),
                           font, font_scale, (255, 255, 200), thickness)
                y += line_height
        else:
            cv2.putText(canvas, "  (waiting for first decision)", (x_offset + 5, y),
                       font, font_scale, (100, 100, 100), thickness)
            y += line_height
        y += 8

        # --- Error Display ---
        error = agent_status.get("last_error")
        if error:
            cv2.putText(canvas, "ERROR:", (x_offset + 5, y), font, font_scale, (100, 100, 255), thickness)
            y += line_height

            # Truncate error if too long
            if len(error) > 45:
                error = error[:42] + "..."
            cv2.putText(canvas, f"  {error}", (x_offset + 5, y),
                       font, font_scale, (100, 100, 255), thickness)

    def close(self):
        """Close the visualization window"""
        if self.created:
            cv2.destroyWindow(self.window_name)


class InteractiveTileSelector:
    """
    Interactive tool for selecting and labeling tiles
    Click on tiles to select them for labeling
    """

    def __init__(self, visualizer: VisualizerV2):
        self.visualizer = visualizer
        self.selected_tiles = []
        self.mouse_callback_set = False

    def setup_mouse_callback(self):
        """Setup mouse callback for tile selection"""
        if not self.mouse_callback_set:
            cv2.setMouseCallback(self.visualizer.window_name, self._mouse_callback)
            self.mouse_callback_set = True

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for tile selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert screen coordinates to tile coordinates
            x_offset = self.visualizer.label_margin
            y_offset = self.visualizer.label_margin

            # Check if click is within the game area
            if (x_offset <= x < x_offset + self.visualizer.display_width and
                y_offset <= y < y_offset + self.visualizer.display_height):

                # Calculate tile coordinates
                tile_x = (x - x_offset) // (TILE_SIZE * self.visualizer.scale)
                tile_y = (y - y_offset) // (TILE_SIZE * self.visualizer.scale)

                if 0 <= tile_x < GRID_COLS and 0 <= tile_y < GRID_ROWS:
                    tile_id = (tile_y, tile_x)
                    if tile_id in self.selected_tiles:
                        self.selected_tiles.remove(tile_id)
                        print(f"Deselected tile: {chr(ord('A')+tile_x)}{tile_y}")
                    else:
                        self.selected_tiles.append(tile_id)
                        print(f"Selected tile: {chr(ord('A')+tile_x)}{tile_y}")

    def get_selected_tiles(self):
        """Get list of selected tiles"""
        return self.selected_tiles.copy()

    def clear_selection(self):
        """Clear all selected tiles"""
        self.selected_tiles.clear()

    def draw_selection_overlay(self, canvas: np.ndarray, x_offset: int, y_offset: int):
        """Draw overlay showing selected tiles"""
        scaled_tile = TILE_SIZE * self.visualizer.scale

        for row, col in self.selected_tiles:
            x1 = x_offset + col * scaled_tile
            y1 = y_offset + row * scaled_tile
            x2 = x1 + scaled_tile
            y2 = y1 + scaled_tile

            # Draw highlight
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

            # Draw border
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)

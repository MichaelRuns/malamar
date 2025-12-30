"""
Real-time perception state visualizer
Displays perception analysis alongside the game window
"""

import cv2
import numpy as np
from typing import Optional
from perception_boy import GameState


class PerceptionVisualizer:
    """Visualize perception state in real-time"""

    def __init__(self, window_name: str = "Perception Analysis"):
        self.window_name = window_name
        self.viz_width = 640
        self.viz_height = 720
        self.created = False

    def visualize(self, frame: np.ndarray, state: GameState):
        """
        Create visualization of perception state

        Args:
            frame: Original grayscale frame (144x160)
            state: GameState object from perception
        """
        # Create visualization canvas
        canvas = np.zeros((self.viz_height, self.viz_width, 3), dtype=np.uint8)

        # Layout sections
        y_offset = 10

        # Section 1: Original Frame (scaled 2x)
        y_offset = self._draw_original_frame(canvas, frame, y_offset)

        # Section 2: Meta Information
        y_offset = self._draw_meta_info(canvas, state, y_offset)

        # Section 3: Grid Visualization
        y_offset = self._draw_grid_viz(canvas, state, y_offset)

        # Section 4: Regions & Sprites
        y_offset = self._draw_regions_sprites(canvas, state, frame, y_offset)

        # Section 5: Changes
        y_offset = self._draw_changes(canvas, state, y_offset)

        # Display
        cv2.imshow(self.window_name, canvas)
        self.created = True

    def _draw_original_frame(self, canvas: np.ndarray, frame: np.ndarray, y_offset: int) -> int:
        """Draw the original game frame"""
        # Title
        cv2.putText(canvas, "ORIGINAL FRAME", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25

        # Scale frame 2x (160x144 -> 320x288)
        if len(frame.shape) == 2:
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_color = frame

        scaled = cv2.resize(frame_color, (320, 288), interpolation=cv2.INTER_NEAREST)

        # Place on canvas
        canvas[y_offset:y_offset+288, 160:160+320] = scaled

        return y_offset + 288 + 15

    def _draw_meta_info(self, canvas: np.ndarray, state: GameState, y_offset: int) -> int:
        """Draw meta information"""
        cv2.putText(canvas, "META INFORMATION", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25

        meta = state.meta
        info_lines = [
            f"Frame: {meta['frame_number']}",
            f"State: {meta['game_state'].upper()}",
            f"Hash: {meta['screen_hash'][:12]}...",
        ]

        if meta['previous_state']:
            info_lines.append(f"Prev: {meta['previous_state']}")

        for line in info_lines:
            color = self._get_state_color(meta['game_state'])
            cv2.putText(canvas, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20

        return y_offset + 10

    def _draw_grid_viz(self, canvas: np.ndarray, state: GameState, y_offset: int) -> int:
        """Draw grid visualization with cell types"""
        cv2.putText(canvas, "GRID VISUALIZATION (18x20)", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25

        # Draw mini grid representation
        cell_size = 8
        grid_width = 20 * cell_size
        grid_height = 18 * cell_size
        x_start = 10

        # Create grid image
        grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        # Color cells by type
        type_colors = {
            'solid': (50, 50, 50),      # Dark gray
            'bg': (100, 100, 150),      # Blue-ish
            'text': (200, 200, 100),    # Yellow-ish
        }

        for cell in state.grid['cells']:
            row, col = cell['pos']
            cell_type = cell['type']
            color = type_colors.get(cell_type, (128, 128, 128))

            y1 = row * cell_size
            x1 = col * cell_size
            y2 = y1 + cell_size
            x2 = x1 + cell_size

            cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, -1)

        # Highlight changed cells
        if 'cells_modified' in state.changes:
            for row, col in state.changes['cells_modified']:
                y1 = row * cell_size
                x1 = col * cell_size
                y2 = y1 + cell_size
                x2 = x1 + cell_size
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Place grid on canvas
        canvas[y_offset:y_offset+grid_height, x_start:x_start+grid_width] = grid_img

        # Draw legend
        legend_x = x_start + grid_width + 20
        legend_items = [
            ("Solid", type_colors['solid']),
            ("Background", type_colors['bg']),
            ("Text", type_colors['text']),
            ("Changed", (0, 255, 0)),
        ]

        legend_y = y_offset
        for label, color in legend_items:
            cv2.rectangle(canvas, (legend_x, legend_y), (legend_x+15, legend_y+15), color, -1)
            cv2.putText(canvas, label, (legend_x+20, legend_y+12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            legend_y += 20

        return y_offset + grid_height + 15

    def _draw_regions_sprites(self, canvas: np.ndarray, state: GameState,
                              frame: np.ndarray, y_offset: int) -> int:
        """Draw detected regions and sprites"""
        cv2.putText(canvas, "REGIONS & SPRITES", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25

        # Draw regions
        if state.regions:
            for region_name, region_data in state.regions.items():
                if region_name == 'text_box' and region_data.get('present'):
                    text = region_data.get('text', '')
                    conf = region_data.get('confidence', 0.0)
                    cv2.putText(canvas, f"Text: '{text}'", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
                    y_offset += 18
                    cv2.putText(canvas, f"Conf: {conf:.2f}", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                    y_offset += 18

                elif region_name == 'hp_bars':
                    cv2.putText(canvas, f"HP Bars: {len(region_data)} detected", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
                    y_offset += 18
        else:
            cv2.putText(canvas, "No regions detected", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
            y_offset += 18

        y_offset += 10

        # Draw sprites
        if state.sprites:
            cv2.putText(canvas, f"Sprites: {len(state.sprites)}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 255), 1)
            y_offset += 18

            for i, sprite in enumerate(state.sprites[:5]):  # Show max 5
                sprite_info = f"  {sprite['id']}: pos={sprite['pos']}, size={sprite['bounds']['w']}x{sprite['bounds']['h']}"
                cv2.putText(canvas, sprite_info, (30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 150, 255), 1)
                y_offset += 15

            if len(state.sprites) > 5:
                cv2.putText(canvas, f"  ... and {len(state.sprites)-5} more", (30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
                y_offset += 15
        else:
            cv2.putText(canvas, "No sprites detected", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
            y_offset += 18

        return y_offset + 10

    def _draw_changes(self, canvas: np.ndarray, state: GameState, y_offset: int) -> int:
        """Draw change information"""
        cv2.putText(canvas, "FRAME CHANGES", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25

        changes = state.changes

        if changes.get('first_frame'):
            cv2.putText(canvas, "First frame - no previous comparison", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            y_offset += 18
        else:
            num_changes = changes.get('num_changes', 0)
            state_trans = changes.get('state_transition', False)

            color = (0, 255, 0) if num_changes > 0 else (128, 128, 128)
            cv2.putText(canvas, f"Cells changed: {num_changes}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 18

            if state_trans:
                cv2.putText(canvas, "STATE TRANSITION!", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)
                y_offset += 18

            # Show some changed cell positions
            if num_changes > 0 and num_changes <= 20:
                changed_cells = changes.get('cells_modified', [])
                cells_str = ', '.join([f"[{r},{c}]" for r, c in changed_cells[:10]])
                if len(changed_cells) > 10:
                    cells_str += f" +{len(changed_cells)-10} more"

                cv2.putText(canvas, f"Positions: {cells_str}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                y_offset += 15

        # Grid stats
        y_offset += 10
        grid = state.grid
        rle_ratio = len(grid['rle_encoded']) / (len(grid['cells']) * 10)

        cv2.putText(canvas, f"Compression: {rle_ratio*100:.1f}% of raw", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y_offset += 15

        return y_offset

    def _get_state_color(self, game_state: str) -> tuple:
        """Get color for game state"""
        colors = {
            'combat': (0, 100, 255),    # Orange-red
            'dialogue': (100, 255, 100), # Green
            'menu': (255, 200, 100),     # Cyan
            'world': (200, 200, 200),    # Gray
        }
        return colors.get(game_state, (255, 255, 255))

    def close(self):
        """Close the visualization window"""
        if self.created:
            cv2.destroyWindow(self.window_name)

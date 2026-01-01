#!/usr/bin/env python3
"""
Interactive Tile Labeller Tool

Usage:
    python tools/tile_labeller_interactive.py

Controls:
    - Click tiles to select/deselect them

    When NO tiles selected (control mode):
        g: toggle grid, r: toggle regions
        s: save labels, l: load labels
        q: quit

    When tiles ARE selected (labelling mode):
        Shift+A through Shift+Z: Label as text character A-Z
        0-9: Label as text character 0-9
        Punctuation: . , ! ? : ; - ' " ( ) / SPACE
        w=walkable, b=blocked, t=water, d=door
        h=hp_bar, m=menu_item, >=cursor
        p=player sprite, n=npc sprite
        c: clear selection (return to control mode)

This tool helps you manually label tile hashes for the perception system.
Note: Use Shift for character labelling to distinguish from terrain keys.
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app_layer'))

from perception_v2 import PerceptionV2, TileLabeller, GRID_COLS, GRID_ROWS, TILE_SIZE
from visualizer_v2 import VisualizerV2


class InteractiveTileLabeller:
    """Interactive tool for labeling tiles"""

    def __init__(self, frame: np.ndarray, labels_file: str = "tile_labels.json"):
        self.frame = frame
        self.labels_file = labels_file

        # Initialize perception and visualizer
        self.labeller = TileLabeller()
        if os.path.exists(labels_file):
            self.labeller.load_labels(labels_file)
            print(f"Loaded {len(self.labeller.hash_to_label)} existing labels")

        self.perception = PerceptionV2(labeller=self.labeller)
        self.visualizer = VisualizerV2(window_name="Interactive Tile Labeller")

        # State
        self.state = None
        self.selected_tiles = []
        self.running = True

        # Setup mouse callback
        cv2.namedWindow(self.visualizer.window_name)
        cv2.setMouseCallback(self.visualizer.window_name, self._mouse_callback)

    def run(self):
        """Run the interactive labeller"""
        print("\n" + "="*60)
        print("INTERACTIVE TILE LABELLER")
        print("="*60)
        print("\nControls:")
        print("  Click tiles to select/deselect")
        print("\nWhen NO tiles selected (control mode):")
        print("  g: toggle grid")
        print("  r: toggle regions")
        print("  s: save labels")
        print("  l: load labels")
        print("  q: quit")
        print("\nWhen tiles ARE selected (labelling mode):")
        print("  SHIFT+letter (A-Z): Label as text character")
        print("  0-9: Label as text character")
        print("  Punctuation: . , ! ? : ; - ' \" ( ) / SPACE")
        print("  w: walkable terrain")
        print("  b: blocked terrain")
        print("  t: water")
        print("  d: door/gate")
        print("  h: hp_bar")
        print("  m: menu_item")
        print("  >: cursor (menu selector)")
        print("  p: player sprite")
        print("  n: npc sprite")
        print("  c: clear selection (return to control mode)")
        print("="*60 + "\n")

        while self.running:
            self._render()
            key = cv2.waitKey(1) & 0xFF

            # Selection-aware key handling
            if self.selected_tiles:
                # LABELLING MODE: Tiles are selected
                # Character labelling (Shift+A-Z = uppercase)
                if ord('A') <= key <= ord('Z'):
                    char = chr(key)
                    self._label_selected(f"char_{char}", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character '{char}'")

                # Character labelling (numbers)
                elif ord('0') <= key <= ord('9'):
                    char = chr(key)
                    self._label_selected(f"char_{char}", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character '{char}'")

                # Punctuation and symbols (common keys that don't conflict)
                elif key == ord('.'):
                    self._label_selected("char_.", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character '.'")
                elif key == ord(','):
                    self._label_selected("char_,", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character ','")
                elif key == ord('!'):
                    self._label_selected("char_!", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character '!'")
                elif key == ord('?'):
                    self._label_selected("char_?", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character '?'")
                elif key == ord(':'):
                    self._label_selected("char_:", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character ':'")
                elif key == ord(';'):
                    self._label_selected("char_;", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character ';'")
                elif key == ord('-'):
                    self._label_selected("char_-", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character '-'")
                elif key == ord('\''):
                    self._label_selected("char_'", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character \"'\"")
                elif key == ord('"'):
                    self._label_selected("char_\"", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character '\"'")
                elif key == ord('('):
                    self._label_selected("char_(", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character '('")
                elif key == ord(')'):
                    self._label_selected("char_)", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character ')'")
                elif key == ord('/'):
                    self._label_selected("char_/", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character '/'")
                elif key == ord(' '):  # Space bar
                    self._label_selected("char_SPACE", "text")
                    print(f"Labeled {len(self.selected_tiles)} tiles as character 'SPACE'")

                # Terrain types
                elif key == ord('w'):
                    self._label_selected("terrain_walkable", "terrain")
                    print(f"Labeled {len(self.selected_tiles)} tiles as walkable terrain")
                elif key == ord('b'):
                    self._label_selected("terrain_blocked", "terrain")
                    print(f"Labeled {len(self.selected_tiles)} tiles as blocked terrain")
                elif key == ord('t'):
                    self._label_selected("terrain_water", "terrain")
                    print(f"Labeled {len(self.selected_tiles)} tiles as water")
                elif key == ord('d'):
                    self._label_selected("terrain_door", "terrain")
                    print(f"Labeled {len(self.selected_tiles)} tiles as door/gate")

                # UI elements
                elif key == ord('h'):
                    self._label_selected("ui_hp_bar", "ui")
                    print(f"Labeled {len(self.selected_tiles)} tiles as HP bar")
                elif key == ord('m'):
                    self._label_selected("ui_menu_item", "ui")
                    print(f"Labeled {len(self.selected_tiles)} tiles as menu item")
                elif key == ord('>'):
                    self._label_selected("ui_cursor", "ui")
                    print(f"Labeled {len(self.selected_tiles)} tiles as cursor")

                # Sprite types
                elif key == ord('p'):
                    self._label_selected("sprite_player", "sprite")
                    print(f"Labeled {len(self.selected_tiles)} tiles as player sprite")
                elif key == ord('n'):
                    self._label_selected("sprite_npc", "sprite")
                    print(f"Labeled {len(self.selected_tiles)} tiles as NPC sprite")

                # Clear selection
                elif key == ord('c'):
                    self._clear_selection()

            else:
                # CONTROL MODE: No tiles selected
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    self._save_labels()
                elif key == ord('l'):
                    self._load_labels()
                elif key == ord('g'):
                    self.visualizer.toggle_grid()
                elif key == ord('r'):
                    self.visualizer.toggle_regions()

        self.visualizer.close()
        print("\nLabeller closed.")

    def _render(self):
        """Render the current state"""
        # Process frame
        self.state = self.perception.perceive(Image.fromarray(self.frame))

        # Create visualization
        canvas = np.zeros(
            (self.visualizer.canvas_height, self.visualizer.canvas_width, 3),
            dtype=np.uint8
        )
        canvas.fill(20)

        # Convert frame to color and scale
        frame_color = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
        scaled_frame = cv2.resize(
            frame_color,
            (self.visualizer.display_width, self.visualizer.display_height),
            interpolation=cv2.INTER_NEAREST
        )

        # Place frame
        x_offset = self.visualizer.label_margin
        y_offset = self.visualizer.label_margin
        canvas[y_offset:y_offset+self.visualizer.display_height,
               x_offset:x_offset+self.visualizer.display_width] = scaled_frame

        # Draw labeled tiles (faint green highlight)
        self._draw_labeled_tiles(canvas, x_offset, y_offset)

        # Draw grid and labels
        if self.visualizer.show_grid:
            self._draw_grid(canvas, x_offset, y_offset)
        if self.visualizer.show_labels:
            self._draw_grid_labels(canvas, x_offset, y_offset)
        if self.visualizer.show_regions:
            self._draw_regions(canvas, x_offset, y_offset)

        # Draw selected tiles
        self._draw_selection(canvas, x_offset, y_offset)

        # Draw info
        self._draw_info(canvas)

        cv2.imshow(self.visualizer.window_name, canvas)

    def _draw_grid(self, canvas, x_offset, y_offset):
        """Draw grid lines"""
        scaled_tile = TILE_SIZE * self.visualizer.scale
        for col in range(GRID_COLS + 1):
            x = x_offset + col * scaled_tile
            cv2.line(canvas, (x, y_offset),
                    (x, y_offset + self.visualizer.display_height), (0, 255, 0), 1)
        for row in range(GRID_ROWS + 1):
            y = y_offset + row * scaled_tile
            cv2.line(canvas, (x_offset, y),
                    (x_offset + self.visualizer.display_width, y), (0, 255, 0), 1)

    def _draw_grid_labels(self, canvas, x_offset, y_offset):
        """Draw A-T and 0-17 labels"""
        scaled_tile = TILE_SIZE * self.visualizer.scale
        for col in range(GRID_COLS):
            label = chr(ord('A') + col)
            cv2.putText(canvas, label,
                       (x_offset + col * scaled_tile + scaled_tile // 2 - 5,
                        y_offset - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        for row in range(GRID_ROWS):
            cv2.putText(canvas, str(row),
                       (x_offset - 25, y_offset + row * scaled_tile + scaled_tile // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def _draw_labeled_tiles(self, canvas, x_offset, y_offset):
        """Draw faint green highlight on tiles that have labels"""
        if not self.state:
            return

        scaled_tile = TILE_SIZE * self.visualizer.scale

        # Create ONE overlay for all labeled tiles (O(1) copies instead of O(n))
        overlay = canvas.copy()
        has_labels = False

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                tile = self.state.get_tile(row, col)
                if tile and tile.label:
                    has_labels = True
                    x1 = x_offset + col * scaled_tile
                    y1 = y_offset + row * scaled_tile
                    x2 = x1 + scaled_tile
                    y2 = y1 + scaled_tile
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)

        # Blend ONCE at the end
        if has_labels:
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

    def _draw_regions(self, canvas, x_offset, y_offset):
        """Draw regions of interest"""
        if not self.state:
            return

        scaled_tile = TILE_SIZE * self.visualizer.scale
        region_colors = {
            "text_box": (255, 100, 100),
            "player_area": (100, 255, 100),
            "enemy_hp": (255, 150, 100),
            "player_hp": (100, 150, 255),
            "battle_menu": (255, 255, 100),
        }

        for name, region in self.state.regions.items():
            color = region_colors.get(name, (128, 128, 128))
            x1 = x_offset + region.tile_start[1] * scaled_tile
            y1 = y_offset + region.tile_start[0] * scaled_tile
            x2 = x_offset + (region.tile_end[1] + 1) * scaled_tile
            y2 = y_offset + (region.tile_end[0] + 1) * scaled_tile

            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

    def _draw_selection(self, canvas, x_offset, y_offset):
        """Draw selected tiles"""
        scaled_tile = TILE_SIZE * self.visualizer.scale

        for row, col in self.selected_tiles:
            x1 = x_offset + col * scaled_tile
            y1 = y_offset + row * scaled_tile
            x2 = x1 + scaled_tile
            y2 = y1 + scaled_tile

            # Highlight
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

            # Border
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)

    def _draw_info(self, canvas):
        """Draw info panel"""
        y = 15
        cv2.putText(canvas, f"Selected: {len(self.selected_tiles)} tiles",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
        cv2.putText(canvas, f"Total labels: {len(self.labeller.hash_to_label)}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show info about selected tiles
        if self.selected_tiles and self.state:
            y += 25
            for i, (row, col) in enumerate(self.selected_tiles[:3]):
                tile = self.state.get_tile(row, col)
                if tile:
                    label_str = tile.label or "unlabeled"
                    cv2.putText(canvas, f"{tile.grid_id()}: {label_str}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 100), 1)
                    y += 18
            if len(self.selected_tiles) > 3:
                cv2.putText(canvas, f"  ... and {len(self.selected_tiles)-3} more",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            x_offset = self.visualizer.label_margin
            y_offset = self.visualizer.label_margin

            if (x_offset <= x < x_offset + self.visualizer.display_width and
                y_offset <= y < y_offset + self.visualizer.display_height):

                tile_col = (x - x_offset) // (TILE_SIZE * self.visualizer.scale)
                tile_row = (y - y_offset) // (TILE_SIZE * self.visualizer.scale)

                if 0 <= tile_col < GRID_COLS and 0 <= tile_row < GRID_ROWS:
                    tile_id = (tile_row, tile_col)

                    if tile_id in self.selected_tiles:
                        self.selected_tiles.remove(tile_id)
                        print(f"Deselected: {chr(ord('A')+tile_col)}{tile_row}")
                    else:
                        self.selected_tiles.append(tile_id)

                        # Show tile info
                        if self.state:
                            tile = self.state.get_tile(tile_row, tile_col)
                            if tile:
                                label_str = tile.label or "unlabeled"
                                print(f"Selected: {tile.grid_id()} | Hash: {tile.hash} | Label: {label_str}")

    def _label_selected(self, label: str, tile_type: str):
        """Apply label to all selected tiles"""
        for row, col in self.selected_tiles:
            tile = self.state.get_tile(row, col)
            if tile:
                self.labeller.label_tile(tile.hash, label, tile_type)

    def _save_labels(self):
        """Save labels to file"""
        self.labeller.save_labels(self.labels_file)
        print(f"✅ Saved {len(self.labeller.hash_to_label)} labels to {self.labels_file}")

    def _load_labels(self):
        """Load labels from file"""
        self.labeller.load_labels(self.labels_file)
        print(f"✅ Loaded {len(self.labeller.hash_to_label)} labels from {self.labels_file}")

    def _clear_selection(self):
        """Clear selection"""
        self.selected_tiles.clear()
        print("Selection cleared")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Tile Labeller")
    parser.add_argument('--image', type=str, help='Path to image file to label')
    parser.add_argument('--labels', type=str, default='tile_labels.json',
                       help='Path to labels file (default: tile_labels.json)')
    args = parser.parse_args()

    # Load frame
    if args.image:
        frame = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Error: Could not load image {args.image}")
            return 1
        print(f"Loaded image: {args.image}")
    else:
        # Create synthetic test frame
        print("No image specified, using synthetic test frame")
        frame = np.zeros((144, 160), dtype=np.uint8)

        # Add gradient
        for y in range(0, 64):
            for x in range(160):
                frame[y, x] = (x + y) % 256

        # Add checkerboard
        for y in range(64, 96):
            for x in range(160):
                if ((x // 8) + (y // 8)) % 2 == 0:
                    frame[y, x] = 200
                else:
                    frame[y, x] = 50

        # Add text box
        frame[128:144, :] = 255
        frame[128, :] = 0
        frame[143, :] = 0

    # Run labeller
    labeller = InteractiveTileLabeller(frame, labels_file=args.labels)
    labeller.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Create a visual example of the grid system
"""

import sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, 'app_layer')
from perception_v2 import PerceptionV2, TileLabeller
from visualizer_v2 import VisualizerV2


def create_example_frame():
    """Create an example frame with labeled regions"""
    frame = np.zeros((144, 160), dtype=np.uint8)

    # Top section - gradient
    for y in range(0, 64):
        for x in range(160):
            frame[y, x] = min(255, (x + y) * 2)

    # Middle section - checkerboard pattern
    for y in range(64, 112):
        for x in range(160):
            if ((x // 8) + (y // 8)) % 2 == 0:
                frame[y, x] = 180
            else:
                frame[y, x] = 80

    # Player area - bright square
    frame[56:88, 64:96] = 220

    # Text box area - white with black border
    frame[128:144, :] = 255
    frame[128:129, :] = 0
    frame[143:144, :] = 0
    frame[128:144, 0:1] = 0
    frame[128:144, 159:160] = 0

    # Add some "text" in text box
    for i in range(5):
        x_start = 8 + i * 16
        frame[132:140, x_start:x_start+6] = 0

    return frame


def main():
    """Create and save example visualization"""
    print("Creating grid example visualization...")

    # Create example frame
    frame = create_example_frame()

    # Initialize perception
    perception = PerceptionV2()
    visualizer = VisualizerV2()

    # Process frame
    state = perception.perceive(Image.fromarray(frame))

    # Create visualization
    visualizer.visualize(frame, state)

    # Get the current display
    import time
    cv2.waitKey(100)  # Give it time to render

    # Capture the window
    print("Visualization created!")
    print("\nGrid layout:")
    print("  Columns: A-T (20 columns)")
    print("  Rows: 0-17 (18 rows)")
    print("  Tile size: 8x8 pixels")
    print("  Total tiles: 360")
    print("\nRegions of interest:")
    for name, region in state.regions.items():
        start = region.tile_start
        end = region.tile_end
        col_start = chr(ord('A') + start[1])
        col_end = chr(ord('A') + end[1])
        print(f"  {name}: {col_start}{start[0]} to {col_end}{end[0]}")

    print("\nPress Q to close window...")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            visualizer.toggle_grid()
            visualizer.visualize(frame, state)
        elif key == ord('l'):
            visualizer.toggle_labels()
            visualizer.visualize(frame, state)
        elif key == ord('r'):
            visualizer.toggle_regions()
            visualizer.visualize(frame, state)

    visualizer.close()
    print("Done!")


if __name__ == "__main__":
    sys.exit(main())

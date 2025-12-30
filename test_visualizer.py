#!/usr/bin/env python3
"""
Test script for perception visualizer
Demonstrates the visualization without running the full game
"""

import sys
import cv2
import numpy as np
from PIL import Image
import time

sys.path.insert(0, 'app_layer')
from perception_boy import PerceptionModule
from perception_visualizer import PerceptionVisualizer


def create_synthetic_frame(frame_num: int) -> Image:
    """Create synthetic game-like frame for testing"""
    # Base frame
    frame = Image.new('L', (160, 144), color=200)
    pixels = frame.load()

    # Simulate HP bars (combat state trigger)
    if frame_num % 120 < 60:  # First half: combat
        for x in range(24, 80):
            pixels[x, 35] = 50
            pixels[x, 36] = 150
            pixels[x, 37] = 50
        for x in range(80, 130):
            pixels[x, 83] = 50
            pixels[x, 84] = 150
            pixels[x, 85] = 50

    # Simulate dialogue box
    if frame_num % 180 > 120:  # Last third: dialogue
        for x in range(160):
            pixels[x, 128] = 50  # Top border
            pixels[x, 143] = 50  # Bottom border
        for y in range(129, 143):
            pixels[0, y] = 50
            pixels[159, y] = 50

    # Animated sprite-like movement
    sprite_x = int(80 + 30 * np.sin(frame_num * 0.05))
    sprite_y = int(72 + 20 * np.cos(frame_num * 0.03))

    for dy in range(16):
        for dx in range(16):
            if 0 <= sprite_x + dx < 160 and 0 <= sprite_y + dy < 144:
                # Simple sprite pattern
                if (dx < 2 or dx > 13 or dy < 2 or dy > 13):
                    pixels[sprite_x + dx, sprite_y + dy] = 50
                else:
                    pixels[sprite_x + dx, sprite_y + dy] = 100

    # Varied background
    for y in range(40, 120):
        for x in range(20, 140):
            if (x + y + frame_num) % 16 < 8:
                pixels[x, y] = 150 + ((x + y) % 50)

    return frame


def main():
    print("=" * 70)
    print("PERCEPTION VISUALIZER TEST")
    print("=" * 70)
    print("\nThis demonstrates the real-time perception visualization.")
    print("\nControls:")
    print("  ESC or Q - Exit")
    print("  SPACE - Pause/Resume")
    print("\nStarting visualization in 2 seconds...")
    time.sleep(2)

    perception = PerceptionModule(use_llm=False)
    visualizer = PerceptionVisualizer()

    frame_num = 0
    paused = False

    print("\n✓ Visualizer running - showing synthetic game frames")
    print("  Watch the grid visualization update in real-time!")

    while True:
        if not paused:
            # Create synthetic frame
            frame = create_synthetic_frame(frame_num)
            frame_array = np.array(frame)

            # Process with perception
            state = perception.perceive(frame)

            # Visualize
            visualizer.visualize(frame_array, state)

            frame_num += 1

            # Print updates occasionally
            if frame_num % 60 == 0:
                print(f"\rFrame {frame_num} | State: {state.meta['game_state']:10} | "
                      f"Changes: {state.changes.get('num_changes', 0):3} cells | "
                      f"Sprites: {len(state.sprites)}", end='', flush=True)

        # Handle keyboard
        key = cv2.waitKey(16) & 0xFF  # ~60 FPS

        if key == 27 or key == ord('q'):  # ESC or Q
            print("\n\nExiting...")
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"\n{status}")

    visualizer.close()
    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f"Processed {frame_num} frames")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()

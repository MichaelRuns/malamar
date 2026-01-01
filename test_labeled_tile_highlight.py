#!/usr/bin/env python3
"""
Test labeled tile highlighting - verify green highlight appears on labeled tiles
"""

import sys
import os
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, 'app_layer')
from perception_v2 import PerceptionV2, TileLabeller

# Import the labeller to test
sys.path.insert(0, 'tools')
from tile_labeller_interactive import InteractiveTileLabeller


def create_test_frame_with_text():
    """Create a test frame with varied patterns"""
    frame = np.zeros((144, 160), dtype=np.uint8)

    # Create a checkerboard pattern in top-left (will be labeled)
    for y in range(64):
        for x in range(80):
            if ((x // 8) + (y // 8)) % 2 == 0:
                frame[y, x] = 200
            else:
                frame[y, x] = 50

    # Create gradient pattern in top-right (unlabeled)
    for y in range(64):
        for x in range(80, 160):
            frame[y, x] = min((x + y) % 256, 255)

    # Create solid pattern in bottom (some labeled, some not)
    frame[64:128, 0:80] = 128  # Will label some
    frame[64:128, 80:160] = 180  # Leave unlabeled

    # Text box at bottom
    frame[128:144, :] = 255
    frame[128, :] = 0
    frame[143, :] = 0

    return frame


def main():
    """Test labeled tile highlighting"""
    print("=== Testing Labeled Tile Highlighting ===\n")

    # Create test frame
    frame = create_test_frame_with_text()
    print("Created test frame (144x160)")

    # Create labeller and label some tiles
    labeller = TileLabeller()
    perception = PerceptionV2(labeller=labeller)
    state = perception.perceive(Image.fromarray(frame))

    # Label some tiles to test highlighting
    print("\nLabeling test tiles:")

    # Label tiles in checkerboard area (rows 0-7, cols 0-9)
    for row in range(0, 8):
        for col in range(0, 10):
            tile = state.get_tile(row, col)
            if tile:
                labeller.label_tile(tile.hash, f"test_tile_{row}_{col}", "test")

    print(f"  ✅ Labeled 80 tiles (rows 0-7, cols 0-9)")

    # Label some tiles in bottom area (rows 8-15, cols 0-4)
    for row in range(8, 16):
        for col in range(0, 5):
            tile = state.get_tile(row, col)
            if tile:
                labeller.label_tile(tile.hash, f"test_tile_{row}_{col}", "test")

    print(f"  ✅ Labeled 40 tiles (rows 8-15, cols 0-4)")

    print(f"\n  Total labeled tiles: {len(labeller.hash_to_label)}")
    print(f"  Total tiles in frame: {20 * 18}")

    print("\n" + "="*60)
    print("✅ LABELED TILE HIGHLIGHTING TEST READY")
    print("="*60)
    print("\nExpected behavior:")
    print("  - Top-left tiles (rows 0-7, cols 0-9) should have FAINT GREEN highlight")
    print("  - Left side tiles (rows 8-15, cols 0-4) should have FAINT GREEN highlight")
    print("  - All other tiles should have NO highlight")
    print("  - Selected tiles should have BRIGHT CYAN highlight (when clicked)")
    print("\nInstructions:")
    print("  1. The labeller window should now open")
    print("  2. Verify labeled tiles have faint green highlight")
    print("  3. Click tiles to see selection vs label highlighting")
    print("  4. Press 'q' to quit when done")
    print("="*60 + "\n")

    # Save to temp file for labeller
    temp_file = "/tmp/test_labeled_highlight.png"
    Image.fromarray(frame).save(temp_file)

    # Create and run interactive labeller
    interactive_labeller = InteractiveTileLabeller(frame, labels_file="/tmp/test_labels.json")

    # Pre-populate the labeller with our labels
    interactive_labeller.labeller = labeller
    interactive_labeller.perception = perception

    print("Opening interactive labeller...")
    interactive_labeller.run()

    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)

    print("\n✅ Test completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

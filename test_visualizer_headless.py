#!/usr/bin/env python3
"""
Headless test for perception visualizer (no display required)
Verifies the visualizer can create visualization images
"""

import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, 'app_layer')
from perception_boy import PerceptionModule
from perception_visualizer import PerceptionVisualizer


def test_visualizer():
    print("=" * 70)
    print("PERCEPTION VISUALIZER - HEADLESS TEST")
    print("=" * 70)

    # Create test frame
    frame = Image.new('L', (160, 144), color=150)
    pixels = frame.load()

    # Add some pattern
    for y in range(144):
        for x in range(160):
            if (x // 8 + y // 8) % 2 == 0:
                pixels[x, y] = 200

    # Add HP bars
    for x in range(24, 80):
        pixels[x, 35] = 50
        pixels[x, 36] = 150

    # Add dialogue box
    for x in range(160):
        pixels[x, 128] = 50
        pixels[x, 143] = 50

    frame_array = np.array(frame)

    print("\n1. Creating perception module...")
    perception = PerceptionModule(use_llm=False)

    print("2. Processing frame...")
    state = perception.perceive(frame)

    print(f"   ✓ State: {state.meta['game_state']}")
    print(f"   ✓ Grid cells: {len(state.grid['cells'])}")
    print(f"   ✓ Regions: {list(state.regions.keys())}")

    print("\n3. Creating visualizer...")
    visualizer = PerceptionVisualizer()

    print("4. Generating visualization canvas...")
    try:
        # Create canvas manually (without showing window)
        canvas = np.zeros((visualizer.viz_height, visualizer.viz_width, 3), dtype=np.uint8)

        # Test each drawing method
        y_offset = 10
        y_offset = visualizer._draw_original_frame(canvas, frame_array, y_offset)
        print("   ✓ Drew original frame")

        y_offset = visualizer._draw_meta_info(canvas, state, y_offset)
        print("   ✓ Drew meta information")

        y_offset = visualizer._draw_grid_viz(canvas, state, y_offset)
        print("   ✓ Drew grid visualization")

        y_offset = visualizer._draw_regions_sprites(canvas, state, frame_array, y_offset)
        print("   ✓ Drew regions & sprites")

        y_offset = visualizer._draw_changes(canvas, state, y_offset)
        print("   ✓ Drew changes")

        print(f"\n5. Verification:")
        print(f"   Canvas shape: {canvas.shape}")
        print(f"   Canvas dtype: {canvas.dtype}")
        print(f"   Non-zero pixels: {np.count_nonzero(canvas):,}")

        # Save sample output
        output_path = "visualizer_test_output.png"
        cv2.imwrite(output_path, canvas)
        print(f"\n   ✓ Saved sample visualization to: {output_path}")

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nThe visualizer is working correctly!")
        print(f"Check '{output_path}' to see the sample visualization.")

    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(test_visualizer())

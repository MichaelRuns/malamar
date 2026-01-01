#!/usr/bin/env python3
"""
Test script for Perception V2 - Tile-based system with enhanced visualizer
"""

import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, 'app_layer')
from perception_v2 import PerceptionV2, TileLabeller, REGIONS_OF_INTEREST
from visualizer_v2 import VisualizerV2


def test_basic_visualization():
    """Test basic visualization with a synthetic frame"""
    print("=== Testing Basic Visualization ===")

    # Create a test frame (160x144 grayscale)
    frame = np.zeros((144, 160), dtype=np.uint8)

    # Add some patterns
    # Top section - gradient
    for y in range(0, 64):
        for x in range(160):
            frame[y, x] = (x + y) % 256

    # Middle section - checkerboard
    for y in range(64, 96):
        for x in range(160):
            if ((x // 8) + (y // 8)) % 2 == 0:
                frame[y, x] = 200
            else:
                frame[y, x] = 50

    # Bottom section (text box area) - white with black border
    frame[128:144, :] = 255
    frame[128, :] = 0
    frame[143, :] = 0
    frame[128:144, 0] = 0
    frame[128:144, 159] = 0

    # Initialize perception and visualizer
    perception = PerceptionV2()
    visualizer = VisualizerV2()

    # Process frame
    state = perception.perceive(Image.fromarray(frame))

    print(f"Frame processed: {state.frame_number}")
    print(f"Screen hash: {state.screen_hash}")
    print(f"Total tiles: {len(state.tiles)}")
    print(f"Regions defined: {list(state.regions.keys())}")

    # Visualize
    print("\nDisplaying visualization...")
    print("Controls:")
    print("  G - Toggle grid")
    print("  L - Toggle labels")
    print("  R - Toggle regions")
    print("  Q - Quit")

    visualizer.visualize(frame, state)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('g'):
            visualizer.toggle_grid()
            visualizer.visualize(frame, state)
            print(f"Grid: {'ON' if visualizer.show_grid else 'OFF'}")
        elif key == ord('l'):
            visualizer.toggle_labels()
            visualizer.visualize(frame, state)
            print(f"Labels: {'ON' if visualizer.show_labels else 'OFF'}")
        elif key == ord('r'):
            visualizer.toggle_regions()
            visualizer.visualize(frame, state)
            print(f"Regions: {'ON' if visualizer.show_regions else 'OFF'}")

    visualizer.close()
    print("✅ Test completed")


def test_region_extraction():
    """Test extracting tiles from specific regions"""
    print("\n=== Testing Region Extraction ===")

    # Create test frame
    frame = np.random.randint(0, 256, (144, 160), dtype=np.uint8)

    perception = PerceptionV2()
    state = perception.perceive(Image.fromarray(frame))

    # Test extracting tiles from each region
    for region_name in state.regions.keys():
        tiles = state.get_tiles_in_region(region_name)
        print(f"\nRegion '{region_name}':")
        print(f"  Tiles: {len(tiles)}")
        if tiles:
            print(f"  First tile: {tiles[0].grid_id()} (hash: {tiles[0].hash})")
            print(f"  Last tile: {tiles[-1].grid_id()} (hash: {tiles[-1].hash})")

    print("✅ Region extraction test completed")


def test_tile_labelling():
    """Test tile labelling functionality"""
    print("\n=== Testing Tile Labelling ===")

    # Create test frame with known patterns
    frame = np.zeros((144, 160), dtype=np.uint8)

    # Create some distinct tiles
    # Top-left tile - all white
    frame[0:8, 0:8] = 255

    # Top-right tile - all black
    frame[0:8, 152:160] = 0

    # Create perception with labeller
    labeller = TileLabeller()
    perception = PerceptionV2(labeller=labeller)

    # Process frame
    state = perception.perceive(Image.fromarray(frame))

    # Get specific tiles
    top_left = state.get_tile(0, 0)
    top_right = state.get_tile(0, 19)

    print(f"Top-left tile (A0): hash={top_left.hash}")
    print(f"Top-right tile (T0): hash={top_right.hash}")

    # Label these tiles
    labeller.label_tile(top_left.hash, "char_A", "text")
    labeller.label_tile(top_right.hash, "terrain_wall", "terrain")

    # Process again to see labels applied
    state2 = perception.perceive(Image.fromarray(frame))
    top_left2 = state2.get_tile(0, 0)
    top_right2 = state2.get_tile(0, 19)

    print(f"\nAfter labelling:")
    print(f"Top-left tile: label={top_left2.label}, type={top_left2.tile_type}")
    print(f"Top-right tile: label={top_right2.label}, type={top_right2.tile_type}")

    # Test saving/loading labels
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        labeller.save_labels(temp_path)
        print(f"\n✅ Labels saved to {temp_path}")

        # Load into new labeller
        new_labeller = TileLabeller()
        new_labeller.load_labels(temp_path)

        print(f"✅ Labels loaded successfully")
        print(f"   Loaded {len(new_labeller.hash_to_label)} labels")

        # Verify
        assert new_labeller.get_label(top_left.hash) == "char_A"
        assert new_labeller.get_type(top_right.hash) == "terrain"
        print("✅ Label persistence verified")

    finally:
        os.unlink(temp_path)

    print("✅ Tile labelling test completed")


def test_json_output():
    """Test JSON serialization"""
    print("\n=== Testing JSON Output ===")

    frame = np.random.randint(0, 256, (144, 160), dtype=np.uint8)

    perception = PerceptionV2()
    state = perception.perceive(Image.fromarray(frame))

    # Get JSON
    json_str = state.to_json()

    # Verify it's valid JSON
    import json
    data = json.loads(json_str)

    print(f"JSON length: {len(json_str)} chars")
    print(f"Keys: {list(data.keys())}")
    print(f"Tiles in JSON: {len(data['tiles'])}")
    print(f"Regions in JSON: {list(data['regions'].keys())}")

    # Print sample
    print(f"\nSample JSON (first 500 chars):")
    print(json_str[:500] + "...")

    print("✅ JSON output test completed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("PERCEPTION V2 TEST SUITE")
    print("Tile-based perception with enhanced visualizer")
    print("=" * 60)

    try:
        # Run tests that don't require user interaction
        test_region_extraction()
        test_tile_labelling()
        test_json_output()

        # Run interactive visualization test
        print("\n" + "=" * 60)
        print("Starting interactive visualization test...")
        print("=" * 60)
        test_basic_visualization()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

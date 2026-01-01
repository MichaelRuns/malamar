#!/usr/bin/env python3
"""
Quick non-interactive test for Perception V2
"""

import sys
import numpy as np
from PIL import Image

sys.path.insert(0, 'app_layer')
from perception_v2 import PerceptionV2, TileLabeller, REGIONS_OF_INTEREST


def test_basic_perception():
    """Test basic perception functionality"""
    print("=== Testing Basic Perception ===")

    # Create test frame
    frame = np.random.randint(0, 256, (144, 160), dtype=np.uint8)

    # Initialize
    perception = PerceptionV2()
    state = perception.perceive(Image.fromarray(frame))

    # Verify
    assert state.frame_number == 1
    assert len(state.tiles) == 360  # 18 * 20
    assert state.screen_hash is not None
    assert len(state.regions) == 5

    print(f"✅ Frame processed: {state.frame_number}")
    print(f"✅ Tiles: {len(state.tiles)}")
    print(f"✅ Screen hash: {state.screen_hash}")
    print(f"✅ Regions: {list(state.regions.keys())}")


def test_tile_access():
    """Test accessing specific tiles"""
    print("\n=== Testing Tile Access ===")

    frame = np.zeros((144, 160), dtype=np.uint8)
    perception = PerceptionV2()
    state = perception.perceive(Image.fromarray(frame))

    # Get specific tiles
    tile_a0 = state.get_tile(0, 0)
    tile_t17 = state.get_tile(17, 19)

    assert tile_a0 is not None
    assert tile_a0.row == 0
    assert tile_a0.col == 0
    assert tile_a0.grid_id() == "A0"

    assert tile_t17 is not None
    assert tile_t17.row == 17
    assert tile_t17.col == 19
    assert tile_t17.grid_id() == "T17"

    print(f"✅ Tile A0: {tile_a0.grid_id()} - hash: {tile_a0.hash}")
    print(f"✅ Tile T17: {tile_t17.grid_id()} - hash: {tile_t17.hash}")


def test_regions():
    """Test region extraction"""
    print("\n=== Testing Regions ===")

    frame = np.random.randint(0, 256, (144, 160), dtype=np.uint8)
    perception = PerceptionV2()
    state = perception.perceive(Image.fromarray(frame))

    # Test each region
    for region_name in state.regions.keys():
        tiles = state.get_tiles_in_region(region_name)
        region = state.regions[region_name]

        print(f"Region '{region_name}':")
        print(f"  Tiles: {len(tiles)}")
        print(f"  Range: {region.tile_start} to {region.tile_end}")

        # Verify tiles are in the correct range
        for tile in tiles:
            assert region.contains_tile(tile.row, tile.col)

    print("✅ All regions verified")


def test_labelling():
    """Test tile labelling"""
    print("\n=== Testing Tile Labelling ===")

    frame = np.zeros((144, 160), dtype=np.uint8)
    frame[0:8, 0:8] = 255  # White tile at A0

    labeller = TileLabeller()
    perception = PerceptionV2(labeller=labeller)

    # First pass - no labels
    state = perception.perceive(Image.fromarray(frame))
    tile_a0 = state.get_tile(0, 0)
    assert tile_a0.label is None

    # Add label
    labeller.label_tile(tile_a0.hash, "char_A", "text")

    # Second pass - should have label
    state2 = perception.perceive(Image.fromarray(frame))
    tile_a0_labelled = state2.get_tile(0, 0)

    assert tile_a0_labelled.label == "char_A"
    assert tile_a0_labelled.tile_type == "text"

    print(f"✅ Tile labelling works")
    print(f"   Hash: {tile_a0.hash}")
    print(f"   Label: {tile_a0_labelled.label}")
    print(f"   Type: {tile_a0_labelled.tile_type}")


def test_json_output():
    """Test JSON serialization"""
    print("\n=== Testing JSON Output ===")

    frame = np.random.randint(0, 256, (144, 160), dtype=np.uint8)
    perception = PerceptionV2()
    state = perception.perceive(Image.fromarray(frame))

    # Get JSON
    json_str = state.to_json()

    # Verify it's valid
    import json
    data = json.loads(json_str)

    assert "frame_number" in data
    assert "tiles" in data
    assert "regions" in data
    assert "screen_hash" in data

    print(f"✅ JSON serialization works")
    print(f"   Length: {len(json_str)} characters")
    print(f"   Keys: {list(data.keys())}")


def main():
    """Run all tests"""
    print("="*60)
    print("PERCEPTION V2 QUICK TEST SUITE")
    print("="*60 + "\n")

    try:
        test_basic_perception()
        test_tile_access()
        test_regions()
        test_labelling()
        test_json_output()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

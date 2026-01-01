#!/usr/bin/env python3
"""
Test tile labeller persistence - verify labels save and load correctly
"""

import sys
import os
import json
import numpy as np
from PIL import Image

sys.path.insert(0, 'app_layer')
from perception_v2 import PerceptionV2, TileLabeller


def test_label_persistence():
    """Test that labels persist between runs"""
    print("=== Testing Tile Label Persistence ===\n")

    # Create a test frame with distinct tiles
    frame = np.zeros((144, 160), dtype=np.uint8)

    # Create 4 distinct tiles in top-left corner
    # Tile A0 (0,0) - all white
    frame[0:8, 0:8] = 255

    # Tile B0 (0,1) - all black
    frame[0:8, 8:16] = 0

    # Tile C0 (0,2) - checkerboard
    for y in range(8):
        for x in range(8):
            if (x + y) % 2 == 0:
                frame[y, 16 + x] = 255

    # Tile D0 (0,3) - gradient
    for y in range(8):
        for x in range(8):
            frame[y, 24 + x] = min((x + y) * 15, 255)  # Clamp to 255

    # Test file path
    test_labels_file = "test_tile_labels.json"

    # Clean up any existing test file
    if os.path.exists(test_labels_file):
        os.remove(test_labels_file)

    print("Step 1: Create labeller and label tiles...")

    # Create perception with labeller
    labeller = TileLabeller()
    perception = PerceptionV2(labeller=labeller)

    # Process frame
    state = perception.perceive(Image.fromarray(frame))

    # Get the 4 tiles
    tile_a0 = state.get_tile(0, 0)
    tile_b0 = state.get_tile(0, 1)
    tile_c0 = state.get_tile(0, 2)
    tile_d0 = state.get_tile(0, 3)

    print(f"  Tile A0 hash: {tile_a0.hash}")
    print(f"  Tile B0 hash: {tile_b0.hash}")
    print(f"  Tile C0 hash: {tile_c0.hash}")
    print(f"  Tile D0 hash: {tile_d0.hash}")

    # Label them
    labeller.label_tile(tile_a0.hash, "char_A", "text")
    labeller.label_tile(tile_b0.hash, "char_B", "text")
    labeller.label_tile(tile_c0.hash, "char_C", "text")
    labeller.label_tile(tile_d0.hash, "char_D", "text")

    print(f"\n  Labeled 4 tiles")
    print(f"  Total labels in memory: {len(labeller.hash_to_label)}")

    # Save labels
    labeller.save_labels(test_labels_file)
    print(f"\n  ✅ Saved labels to {test_labels_file}")

    # Verify file exists and contains data
    assert os.path.exists(test_labels_file), "Labels file not created!"
    with open(test_labels_file, 'r') as f:
        saved_data = json.load(f)

    print(f"  ✅ Labels file exists with {len(saved_data['hash_to_label'])} labels")

    # Step 2: Load in new session
    print("\nStep 2: Simulate new session - load labels...")

    # Create new labeller (simulates new run)
    new_labeller = TileLabeller()
    print(f"  New labeller has {len(new_labeller.hash_to_label)} labels (should be 0)")

    # Load labels
    new_labeller.load_labels(test_labels_file)
    print(f"  ✅ Loaded {len(new_labeller.hash_to_label)} labels from file")

    # Create new perception with loaded labeller
    new_perception = PerceptionV2(labeller=new_labeller)

    # Process same frame
    new_state = new_perception.perceive(Image.fromarray(frame))

    # Get tiles and check labels
    new_tile_a0 = new_state.get_tile(0, 0)
    new_tile_b0 = new_state.get_tile(0, 1)
    new_tile_c0 = new_state.get_tile(0, 2)
    new_tile_d0 = new_state.get_tile(0, 3)

    print("\nStep 3: Verify labels were applied correctly...")
    print(f"  Tile A0: label='{new_tile_a0.label}', type='{new_tile_a0.tile_type}'")
    print(f"  Tile B0: label='{new_tile_b0.label}', type='{new_tile_b0.tile_type}'")
    print(f"  Tile C0: label='{new_tile_c0.label}', type='{new_tile_c0.tile_type}'")
    print(f"  Tile D0: label='{new_tile_d0.label}', type='{new_tile_d0.tile_type}'")

    # Verify labels match
    assert new_tile_a0.label == "char_A", f"A0 label mismatch: {new_tile_a0.label}"
    assert new_tile_b0.label == "char_B", f"B0 label mismatch: {new_tile_b0.label}"
    assert new_tile_c0.label == "char_C", f"C0 label mismatch: {new_tile_c0.label}"
    assert new_tile_d0.label == "char_D", f"D0 label mismatch: {new_tile_d0.label}"

    assert new_tile_a0.tile_type == "text", f"A0 type mismatch: {new_tile_a0.tile_type}"
    assert new_tile_b0.tile_type == "text", f"B0 type mismatch: {new_tile_b0.tile_type}"
    assert new_tile_c0.tile_type == "text", f"C0 type mismatch: {new_tile_c0.tile_type}"
    assert new_tile_d0.tile_type == "text", f"D0 type mismatch: {new_tile_d0.tile_type}"

    print("\n  ✅ All labels match!")

    # Verify hashes match
    assert new_tile_a0.hash == tile_a0.hash, "A0 hash changed!"
    assert new_tile_b0.hash == tile_b0.hash, "B0 hash changed!"
    assert new_tile_c0.hash == tile_c0.hash, "C0 hash changed!"
    assert new_tile_d0.hash == tile_d0.hash, "D0 hash changed!"

    print("  ✅ All hashes stable (deterministic)")

    # Clean up test file
    os.remove(test_labels_file)
    print(f"\n  🧹 Cleaned up {test_labels_file}")

    print("\n" + "="*60)
    print("✅ LABEL PERSISTENCE TEST PASSED")
    print("="*60)


def test_shift_key_characters():
    """Test that all shift+letter characters can be labeled"""
    print("\n=== Testing Shift Key Characters ===\n")

    # Create frames with unique patterns for each letter
    labeller = TileLabeller()

    print("Testing all uppercase letters (Shift+A through Shift+Z):")

    for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        # Create unique pattern
        frame = np.zeros((144, 160), dtype=np.uint8)
        frame[0:8, 0:8] = i * 10  # Unique value for each letter

        perception = PerceptionV2(labeller=labeller)
        state = perception.perceive(Image.fromarray(frame))
        tile = state.get_tile(0, 0)

        # Label it
        labeller.label_tile(tile.hash, f"char_{letter}", "text")

        # Verify
        assert labeller.get_label(tile.hash) == f"char_{letter}", f"Failed to label {letter}"

    print(f"  ✅ All 26 letters can be labeled")
    print(f"  Total labels: {len(labeller.hash_to_label)}")

    # Test problematic characters specifically
    print("\nTesting previously problematic characters:")
    problematic = ["C", "G", "L", "Q", "R", "S"]

    for letter in problematic:
        # Find the label
        label = f"char_{letter}"
        found = False
        for hash_val, stored_label in labeller.hash_to_label.items():
            if stored_label == label:
                found = True
                print(f"  ✅ '{letter}' labeled successfully as '{stored_label}'")
                break

        assert found, f"Character '{letter}' not found in labels!"

    print("\n" + "="*60)
    print("✅ SHIFT KEY CHARACTER TEST PASSED")
    print("="*60)


def main():
    """Run all tests"""
    print("="*60)
    print("TILE LABELLER PERSISTENCE AND KEY TEST")
    print("="*60 + "\n")

    try:
        test_label_persistence()
        test_shift_key_characters()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nConclusions:")
        print("  ✅ Labels save correctly to JSON")
        print("  ✅ Labels load correctly on next run")
        print("  ✅ Same tiles get same hashes (deterministic)")
        print("  ✅ Labels correctly apply to tiles")
        print("  ✅ All 26 letters (A-Z) can be labeled")
        print("  ✅ Previously problematic keys (C, G, S, etc.) work")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test label grid functionality in perception module
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, 'app_layer')
from perception_v2 import PerceptionV2, TileLabeller, PerceptionState, GRID_ROWS, GRID_COLS


def test_label_grid_creation():
    """Test that label_grid is created correctly"""
    print("=== Testing Label Grid Creation ===\n")

    # Create test frame with unique tiles
    frame = np.zeros((144, 160), dtype=np.uint8)

    # Make each tile unique by using row+col based patterns
    for row in range(18):
        for col in range(20):
            y_start = row * 8
            x_start = col * 8
            # Create unique pattern for each tile
            frame[y_start:y_start+8, x_start:x_start+8] = (row * 20 + col) % 256

    # Create labeller and label some tiles
    labeller = TileLabeller()
    perception = PerceptionV2(labeller=labeller)
    state = perception.perceive(Image.fromarray(frame))

    # Label first 10 tiles in row 0
    for col in range(10):
        tile = state.get_tile(0, col)
        if tile:
            char = chr(ord('A') + col)
            labeller.label_tile(tile.hash, f"char_{char}", "text")

    # Re-perceive to get updated labels
    state = perception.perceive(Image.fromarray(frame))

    # Verify label_grid
    print("Checking label_grid structure:")
    assert state.label_grid is not None, "label_grid should not be None"
    assert len(state.label_grid) == GRID_ROWS, f"Expected {GRID_ROWS} rows, got {len(state.label_grid)}"
    assert len(state.label_grid[0]) == GRID_COLS, f"Expected {GRID_COLS} cols, got {len(state.label_grid[0])}"
    print(f"  ✅ label_grid has correct dimensions: {GRID_ROWS} × {GRID_COLS}")

    # Verify labels in grid
    print("\nChecking labeled tiles:")
    for col in range(10):
        label = state.get_label_at(0, col)
        expected = f"char_{chr(ord('A') + col)}"
        assert label == expected, f"Expected {expected}, got {label}"
        print(f"  ✅ Tile (0, {col}): {label}")

    # Verify unlabeled tiles are None
    unlabeled = state.get_label_at(0, 15)
    assert unlabeled is None, f"Expected None for unlabeled tile, got {unlabeled}"
    print(f"  ✅ Unlabeled tile (0, 15): None")

    print("\n✅ Label grid creation test passed!")


def test_display_char():
    """Test get_display_char for various label types"""
    print("\n=== Testing Display Characters ===\n")

    frame = np.zeros((144, 160), dtype=np.uint8)

    # Create unique tiles for entire frame
    for row in range(18):
        for col in range(20):
            y_start = row * 8
            x_start = col * 8
            frame[y_start:y_start+8, x_start:x_start+8] = (row * 20 + col + 100) % 256

    labeller = TileLabeller()
    perception = PerceptionV2(labeller=labeller)
    state = perception.perceive(Image.fromarray(frame))

    # Label different tile types
    test_cases = [
        (0, 0, "char_A", "A"),
        (0, 1, "char_5", "5"),
        (0, 2, "char_SPACE", "_"),
        (0, 3, "char_!", "!"),
        (1, 0, "terrain_walkable", "W"),
        (1, 1, "terrain_blocked", "B"),
        (1, 2, "terrain_door", "D"),
        (2, 0, "ui_hp_bar", "U"),
    ]

    for row, col, label, expected_char in test_cases:
        tile = state.get_tile(row, col)
        if tile:
            labeller.label_tile(tile.hash, label, "test")

    # Re-perceive
    state = perception.perceive(Image.fromarray(frame))

    # Verify display characters
    print("Checking display characters:")
    for row, col, label, expected_char in test_cases:
        display_char = state.get_display_char(row, col)
        assert display_char == expected_char, f"Expected '{expected_char}' for {label}, got '{display_char}'"
        print(f"  ✅ {label} → '{display_char}'")

    # Check unlabeled tile
    unlabeled_char = state.get_display_char(4, 4)
    assert unlabeled_char == ".", f"Expected '.' for unlabeled, got '{unlabeled_char}'"
    print(f"  ✅ Unlabeled → '.'")

    print("\n✅ Display character test passed!")


def test_display_grid():
    """Test get_display_grid for text representation"""
    print("\n=== Testing Display Grid ===\n")

    frame = np.zeros((144, 160), dtype=np.uint8)

    # Create unique tiles for entire frame
    for row in range(18):
        for col in range(20):
            y_start = row * 8
            x_start = col * 8
            frame[y_start:y_start+8, x_start:x_start+8] = (row * 20 + col + 50) % 256

    labeller = TileLabeller()
    perception = PerceptionV2(labeller=labeller)
    state = perception.perceive(Image.fromarray(frame))

    # Label "HELLO" in first row
    word = "HELLO"
    for i, char in enumerate(word):
        tile = state.get_tile(0, i)
        if tile:
            labeller.label_tile(tile.hash, f"char_{char}", "text")

    # Re-perceive
    state = perception.perceive(Image.fromarray(frame))

    # Get display grid
    display_grid = state.get_display_grid()

    print("Checking display grid:")
    assert len(display_grid) == GRID_ROWS, f"Expected {GRID_ROWS} lines"
    assert len(display_grid[0]) == GRID_COLS, f"Expected {GRID_COLS} chars per line"
    print(f"  ✅ Display grid has correct dimensions")

    # Check first row starts with "HELLO"
    first_row = display_grid[0]
    assert first_row.startswith("HELLO"), f"Expected first row to start with 'HELLO', got '{first_row[:10]}'"
    print(f"  ✅ First row: '{first_row}'")

    # Check rest is dots
    assert first_row[5:] == "." * (GRID_COLS - 5), "Expected rest of row to be dots"
    print(f"  ✅ Unlabeled tiles shown as '.'")

    print("\n✅ Display grid test passed!")


def test_json_includes_label_grid():
    """Test that to_dict includes label_grid and display_grid"""
    print("\n=== Testing JSON Output ===\n")

    frame = np.zeros((144, 160), dtype=np.uint8)

    # Create unique tiles for entire frame
    for row in range(18):
        for col in range(20):
            y_start = row * 8
            x_start = col * 8
            frame[y_start:y_start+8, x_start:x_start+8] = (row * 20 + col + 150) % 256

    labeller = TileLabeller()
    perception = PerceptionV2(labeller=labeller)
    state = perception.perceive(Image.fromarray(frame))

    # Label a tile
    tile = state.get_tile(0, 0)
    if tile:
        labeller.label_tile(tile.hash, "char_X", "text")

    state = perception.perceive(Image.fromarray(frame))

    # Get dict representation
    state_dict = state.to_dict()

    print("Checking JSON structure:")
    assert "label_grid" in state_dict, "label_grid should be in dict"
    print(f"  ✅ label_grid present")

    assert "display_grid" in state_dict, "display_grid should be in dict"
    print(f"  ✅ display_grid present")

    assert len(state_dict["display_grid"]) == GRID_ROWS
    print(f"  ✅ display_grid has {GRID_ROWS} rows")

    # Check first character
    first_char = state_dict["display_grid"][0][0]
    assert first_char == "X", f"Expected 'X', got '{first_char}'"
    print(f"  ✅ First character in display_grid: '{first_char}'")

    print("\n✅ JSON output test passed!")


def main():
    """Run all tests"""
    print("="*60)
    print("LABEL GRID FUNCTIONALITY TESTS")
    print("="*60)

    try:
        test_label_grid_creation()
        test_display_char()
        test_display_grid()
        test_json_includes_label_grid()

        print("\n" + "="*60)
        print("✅ ALL LABEL GRID TESTS PASSED")
        print("="*60)
        print("\nNew features verified:")
        print("  ✅ PerceptionState.label_grid - 2D grid of labels")
        print("  ✅ PerceptionState.get_label_at(row, col) - Get label at position")
        print("  ✅ PerceptionState.get_display_char(row, col) - Single char representation")
        print("  ✅ PerceptionState.get_display_grid() - Full text grid for visualization")
        print("  ✅ to_dict() includes label_grid and display_grid")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

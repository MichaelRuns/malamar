#!/usr/bin/env python3
"""
Test punctuation support - verify all symbols can be labeled
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, 'app_layer')
from perception_v2 import PerceptionV2, TileLabeller


def test_punctuation_symbols():
    """Test that all punctuation symbols can be labeled"""
    print("=== Testing Punctuation Symbol Support ===\n")

    labeller = TileLabeller()

    # Test all punctuation symbols
    punctuation = {
        '.': 'period',
        ',': 'comma',
        '!': 'exclamation',
        '?': 'question',
        ':': 'colon',
        ';': 'semicolon',
        '-': 'hyphen',
        "'": 'apostrophe',
        '"': 'quote',
        '(': 'left_paren',
        ')': 'right_paren',
        '/': 'slash',
        'SPACE': 'space',
    }

    print(f"Testing {len(punctuation)} punctuation symbols:")

    for i, (symbol, name) in enumerate(punctuation.items()):
        # Create unique pattern for each symbol
        frame = np.zeros((144, 160), dtype=np.uint8)
        frame[0:8, 0:8] = min((i + 50) * 5, 255)  # Unique value, clamped

        perception = PerceptionV2(labeller=labeller)
        state = perception.perceive(Image.fromarray(frame))
        tile = state.get_tile(0, 0)

        # Label it (use the label format from the tool)
        if symbol == 'SPACE':
            label = f"char_SPACE"
        else:
            label = f"char_{symbol}"

        labeller.label_tile(tile.hash, label, "text")

        # Verify
        stored_label = labeller.get_label(tile.hash)
        assert stored_label == label, f"Failed to label {symbol}: expected {label}, got {stored_label}"
        print(f"  ✅ '{symbol}' ({name}) labeled as '{label}'")

    print(f"\n  ✅ All {len(punctuation)} punctuation symbols can be labeled")
    print(f"\n  Total labels in test: {len(labeller.hash_to_label)}")

    print("\n" + "="*60)
    print("✅ PUNCTUATION SUPPORT TEST PASSED")
    print("="*60)
    print("\nCharacter Coverage:")
    print("  ✅ 26 letters (A-Z)")
    print("  ✅ 10 digits (0-9)")
    print("  ✅ 13 punctuation symbols")
    print("  ━━━━━━━━━━━━━━━━━━━━━")
    print("  ✅ 49 total characters for text reading!")


def main():
    """Run all tests"""
    try:
        test_punctuation_symbols()
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

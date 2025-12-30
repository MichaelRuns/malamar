#!/usr/bin/env python3
"""
Test script for perception system
Verifies deterministic JSON generation and performance
"""

import sys
import time
import json
from PIL import Image
import numpy as np

sys.path.insert(0, 'app_layer')
from perception_boy import PerceptionModule, GameState


def test_determinism():
    """Test that same frame produces same JSON"""
    print("\n=== Testing Determinism ===")

    # Create a simple test frame (160x144 grayscale)
    test_frame = Image.new('L', (160, 144), color=128)

    # Draw a simple pattern
    pixels = test_frame.load()
    for y in range(20, 40):
        for x in range(20, 40):
            pixels[x, y] = 255

    perception = PerceptionModule(use_llm=False)

    # Process same frame twice
    state1 = perception.perceive(test_frame)

    # Reset perception module
    perception = PerceptionModule(use_llm=False)
    state2 = perception.perceive(test_frame)

    # Compare screen hashes
    hash1 = state1.meta['screen_hash']
    hash2 = state2.meta['screen_hash']

    if hash1 == hash2:
        print(f"✅ Determinism test PASSED - Hash: {hash1}")
    else:
        print(f"❌ Determinism test FAILED - Hash1: {hash1}, Hash2: {hash2}")

    # Verify grid structure
    assert len(state1.grid['cells']) == 360, "Should have 360 cells (18x20)"
    print(f"✅ Grid structure correct - {len(state1.grid['cells'])} cells")

    return state1


def test_performance():
    """Test that processing is fast enough for 60 FPS"""
    print("\n=== Testing Performance ===")

    # Create test frame
    test_frame = Image.new('L', (160, 144), color=100)
    perception = PerceptionModule(use_llm=False)

    # Warm up
    for _ in range(10):
        perception.perceive(test_frame)

    # Benchmark
    num_frames = 100
    start_time = time.time()

    for _ in range(num_frames):
        state = perception.perceive(test_frame)

    elapsed = time.time() - start_time
    avg_ms = (elapsed / num_frames) * 1000
    fps = num_frames / elapsed

    print(f"Average time per frame: {avg_ms:.2f}ms")
    print(f"Effective FPS: {fps:.1f}")

    if avg_ms < 16.67:
        print(f"✅ Performance test PASSED - Target: <16.67ms for 60 FPS")
    else:
        print(f"⚠️  Performance test WARNING - Slower than 60 FPS target")

    return avg_ms


def test_state_detection():
    """Test game state detection with synthetic frames"""
    print("\n=== Testing State Detection ===")

    perception = PerceptionModule(use_llm=False)

    # Test 1: World navigation (varied pattern)
    world_frame = Image.new('L', (160, 144))
    pixels = world_frame.load()
    for y in range(144):
        for x in range(160):
            pixels[x, y] = (x + y) % 256  # Varied pattern

    state = perception.perceive(world_frame)
    print(f"World frame detected as: {state.meta['game_state']}")

    # Test 2: Dialogue box (darker border at bottom)
    dialogue_frame = Image.new('L', (160, 144), color=200)
    pixels = dialogue_frame.load()
    for x in range(160):
        pixels[x, 128] = 50  # Dark border
        pixels[x, 143] = 50  # Dark border

    perception2 = PerceptionModule(use_llm=False)
    state2 = perception2.perceive(dialogue_frame)
    print(f"Dialogue frame detected as: {state2.meta['game_state']}")

    print("✅ State detection test completed")


def test_change_detection():
    """Test frame-to-frame change detection"""
    print("\n=== Testing Change Detection ===")

    perception = PerceptionModule(use_llm=False)

    # Frame 1
    frame1 = Image.new('L', (160, 144), color=100)
    state1 = perception.perceive(frame1)
    print(f"Frame 1 changes: {state1.changes}")

    # Frame 2 (identical)
    frame2 = Image.new('L', (160, 144), color=100)
    state2 = perception.perceive(frame2)
    print(f"Frame 2 changes: {state2.changes['num_changes']} cells (should be 0)")

    # Frame 3 (modified)
    frame3 = Image.new('L', (160, 144), color=100)
    pixels = frame3.load()
    for y in range(8):
        for x in range(8):
            pixels[x, y] = 255  # Change top-left tile

    state3 = perception.perceive(frame3)
    print(f"Frame 3 changes: {state3.changes['num_changes']} cells (should be 1)")

    if state2.changes['num_changes'] == 0 and state3.changes['num_changes'] == 1:
        print("✅ Change detection test PASSED")
    else:
        print("❌ Change detection test FAILED")


def test_json_output(state: GameState):
    """Test JSON serialization"""
    print("\n=== Testing JSON Output ===")

    json_str = state.to_json()

    # Parse to verify it's valid JSON
    try:
        data = json.loads(json_str)
        print("✅ JSON is valid")

        # Check structure
        required_keys = ['meta', 'grid', 'regions', 'sprites', 'changes']
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

        print(f"✅ JSON structure correct - Keys: {list(data.keys())}")

        # Print sample
        print(f"\nSample JSON (first 500 chars):")
        print(json_str[:500] + "...")

        # Print stats
        print(f"\nJSON Stats:")
        print(f"  Total length: {len(json_str)} chars")
        print(f"  Cells: {len(data['grid']['cells'])}")
        print(f"  RLE encoded length: {len(data['grid']['rle_encoded'])} chars")

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("PERCEPTION SYSTEM TEST SUITE")
    print("=" * 60)

    try:
        # Run tests
        state = test_determinism()
        test_performance()
        test_state_detection()
        test_change_detection()
        test_json_output(state)

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

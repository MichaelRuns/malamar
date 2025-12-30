#!/usr/bin/env python3
"""
Example script showing the JSON output format
Run this to see a sample of the perception system output
"""

import sys
from PIL import Image
import numpy as np

sys.path.insert(0, 'app_layer')
from perception_boy import PerceptionModule


def create_sample_frame():
    """Create a sample Pokemon Blue-like frame"""
    frame = Image.new('L', (160, 144), color=200)
    pixels = frame.load()

    # Simulate HP bar region (darker horizontal line)
    for x in range(24, 80):
        pixels[x, 35] = 50
        pixels[x, 36] = 150
        pixels[x, 37] = 50

    # Simulate dialogue box at bottom
    for x in range(160):
        pixels[x, 128] = 50  # Top border
        pixels[x, 143] = 50  # Bottom border
    for y in range(129, 143):
        pixels[0, y] = 50
        pixels[159, y] = 50

    # Add some varied content in the middle (world/sprite area)
    for y in range(40, 120):
        for x in range(20, 140):
            pixels[x, y] = (x + y) % 180 + 50

    return frame


def main():
    print("=" * 70)
    print("POKEMON BLUE PERCEPTION SYSTEM - EXAMPLE OUTPUT")
    print("=" * 70)

    # Create sample frame
    frame = create_sample_frame()

    # Initialize perception
    perception = PerceptionModule(use_llm=False)

    # Process frame
    print("\nProcessing sample frame...")
    state = perception.perceive(frame)

    # Display results
    print("\n" + "=" * 70)
    print("STRUCTURED STATE OUTPUT")
    print("=" * 70)

    print(f"\n📊 META INFORMATION")
    print(f"  Frame Number: {state.meta['frame_number']}")
    print(f"  Screen Hash: {state.meta['screen_hash']}")
    print(f"  Game State: {state.meta['game_state']}")
    print(f"  Previous State: {state.meta['previous_state']}")

    print(f"\n🎮 GRID DATA")
    print(f"  Dimensions: {state.grid['dimensions']['rows']}x{state.grid['dimensions']['cols']}")
    print(f"  Cell Size: {state.grid['dimensions']['cell_size']}px")
    print(f"  Total Cells: {len(state.grid['cells'])}")
    print(f"  RLE Compressed: {len(state.grid['rle_encoded'])} chars")
    print(f"  Sample cells:")
    for i in range(min(3, len(state.grid['cells']))):
        cell = state.grid['cells'][i]
        print(f"    [{cell['pos'][0]},{cell['pos'][1]}] hash={cell['hash']} type={cell['type']}")

    print(f"\n🏷️  REGIONS DETECTED")
    if state.regions:
        for region_name, region_data in state.regions.items():
            print(f"  {region_name}: {region_data}")
    else:
        print("  (none)")

    print(f"\n👾 SPRITES DETECTED")
    if state.sprites:
        for sprite in state.sprites:
            print(f"  {sprite['id']}: pos={sprite['pos']}, bounds={sprite['bounds']}")
    else:
        print("  (none - first frame has no previous frame to compare)")

    print(f"\n🔄 CHANGES FROM PREVIOUS FRAME")
    print(f"  {state.changes}")

    print("\n" + "=" * 70)
    print("FULL JSON OUTPUT")
    print("=" * 70)

    json_output = state.to_json()
    print(json_output)

    print("\n" + "=" * 70)
    print("JSON SIZE ANALYSIS")
    print("=" * 70)
    print(f"  Total JSON size: {len(json_output):,} bytes")
    print(f"  Estimated tokens (÷4): ~{len(json_output)//4:,} tokens")
    print(f"  RLE compression ratio: {len(state.grid['rle_encoded']) / (len(state.grid['cells']) * 10):.1%}")

    print("\n💡 USAGE TIP:")
    print("  Use state.to_json() to get JSON string for LLM input")
    print("  Access structured data via state.meta, state.grid, etc.")
    print("  Grid hashes are deterministic - same frame = same hash")


if __name__ == "__main__":
    main()

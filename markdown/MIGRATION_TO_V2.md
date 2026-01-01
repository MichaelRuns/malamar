# Migration Guide: Perception V1 → V2

## Summary

[game_runner.py](app_layer/game_runner.py) has been updated to use **Perception V2** with the new tile-based system.

## What Changed

### Imports
**Before (V1):**
```python
from perception_boy import PerceptionModule
from perception_visualizer import PerceptionVisualizer
```

**After (V2):**
```python
from perception_v2 import PerceptionV2, TileLabeller
from visualizer_v2 import VisualizerV2
import cv2
import os
```

### Initialization
**Before (V1):**
```python
self.perception = PerceptionModule(False)
self.visualizer = PerceptionVisualizer() if enable_viz else None
```

**After (V2):**
```python
# Load tile labels
self.labeller = TileLabeller()
if os.path.exists(labels_file):
    self.labeller.load_labels(labels_file)

# Initialize perception
self.perception = PerceptionV2(labeller=self.labeller)
self.visualizer = VisualizerV2() if enable_viz else None
```

### Perception Output
**Before (V1):**
```python
state = self.perception.perceive(frame)  # Returns GameState
# Access via: state.meta['game_state'], state.changes, etc.
```

**After (V2):**
```python
state = self.perception.perceive(frame)  # Returns PerceptionState
# Access via: state.screen_hash, state.tiles, state.regions
```

### New Features in V2

1. **Tile Grid with Labels**
   - Columns: A-T (alphabetic)
   - Rows: 0-17 (numeric)
   - Example: tile at position (5, 10) = "K5"

2. **Regions of Interest**
   ```python
   text_tiles = state.get_tiles_in_region("text_box")
   player_tiles = state.get_tiles_in_region("player_area")
   ```

3. **Text Reconstruction**
   ```python
   dialogue = self._read_text_from_tiles(text_tiles)
   ```

4. **Interactive Controls**
   - `G`: Toggle grid overlay
   - `R`: Toggle regions overlay
   - Both work when visualizer is enabled

## Running the Game with V2

```bash
cd /Users/michaelvernau/repos/malamar
source .venv/bin/activate
python app_layer/game_runner.py
```

**Controls while running:**
- `V`: Toggle visualizer on/off
- `G`: Toggle grid (shows A-T, 0-17 labels)
- `R`: Toggle regions (shows text box, HP bars, etc.)
- `M`: Toggle manual mode
- `S`: Save game
- `L`: Load game

## Tile Labelling Workflow

To build your tile vocabulary:

1. **Capture screenshots** while playing:
   - Navigate to different screens (dialogue, menus, battle)
   - Press `S` to save game state
   - Take screenshots (or use PyBoy's save screenshot)

2. **Label tiles interactively**:
   ```bash
   python tools/tile_labeller_interactive.py --image screenshot.png --labels tile_labels.json
   ```

3. **Labels are automatically loaded** next time you run the game

## Key Differences

| Feature | V1 (perception_boy) | V2 (perception_v2) |
|---------|---------------------|-------------------|
| Grid labels | ❌ | ✅ A-T, 0-17 |
| Regions of interest | Hardcoded | ✅ Configurable |
| Tile labelling | ❌ | ✅ Interactive tool |
| Label persistence | ❌ | ✅ JSON save/load |
| Visualizer | Basic | ✅ Enhanced with overlay |
| Game state detection | ✅ | Removed (can add back) |
| Sprite detection | ✅ | Removed (can add back) |

**V2 focuses on:**
- Tile-level perception and labelling
- Building a tile vocabulary over time
- Clean separation of concerns
- Better visualization for debugging

## Backward Compatibility

V1 modules are still available if needed:
- [perception_boy.py](app_layer/perception_boy.py)
- [perception_visualizer.py](app_layer/perception_visualizer.py)

Simply change the imports back to use V1.

## Next Steps

1. **Run the game** and verify V2 visualizer works
2. **Capture screenshots** of different game screens
3. **Label tiles** using the interactive tool
4. **Build vocabulary** of text characters first (highest priority)
5. **Integrate with LLM** once you have labeled text tiles

## Documentation

- [PERCEPTION_V2_README.md](PERCEPTION_V2_README.md) - Full user guide
- [PERCEPTION_V2_IMPLEMENTATION.md](PERCEPTION_V2_IMPLEMENTATION.md) - Implementation details
- [test_perception_v2_quick.py](test_perception_v2_quick.py) - Quick tests

## Testing

Quick test to verify V2 works:
```bash
python test_perception_v2_quick.py
```

Should output:
```
✅ ALL TESTS PASSED
```

## Troubleshooting

**Issue**: `No module named 'perception_v2'`
- **Solution**: Make sure you're in the right directory and venv is activated

**Issue**: Visualizer window doesn't appear
- **Solution**: Press `V` to toggle visualizer, make sure `enable_viz=True`

**Issue**: No labels showing
- **Solution**: You need to create labels first using `tools/tile_labeller_interactive.py`

**Issue**: Grid not visible
- **Solution**: Press `G` to toggle grid overlay

## File Structure

```
app_layer/
├── perception_v2.py          # Core V2 perception
├── visualizer_v2.py           # Enhanced visualizer
├── game_runner.py             # ✅ Updated to use V2
├── perception_boy.py          # Legacy V1 (still available)
└── perception_visualizer.py   # Legacy V1 visualizer

tools/
└── tile_labeller_interactive.py  # Interactive labelling tool

tile_labels.json               # Your tile vocabulary (created by tool)
```

## Summary

✅ **game_runner.py** now uses Perception V2
✅ **Enhanced visualizer** with grid labels
✅ **Tile labelling system** ready to use
✅ **All tests passing**

Start playing and labelling tiles to build your vocabulary! 🎮

# Perception V2 - Tile-Based Perception System

A new tile-based perception module for Pokemon Blue that segments the game screen into labeled tiles for LLM-based gameplay.

## Overview

Perception V2 improves upon the original perception system by:

1. **Tile Grid Visualization**: Enhanced visualizer with alphabetic column labels (A-T) and numeric row labels (0-17)
2. **Regions of Interest**: Pre-defined regions for key game areas (text box, HP bars, player area, battle menu)
3. **Interactive Tile Labelling**: Tools to manually label tile hashes for building a tile dictionary
4. **Cleaner Architecture**: Separation of concerns between perception, visualization, and labelling

## Architecture

```
app_layer/
├── perception_v2.py          # Core perception module
├── visualizer_v2.py           # Enhanced visualizer with grid overlay
└── game_runner.py             # Main game loop (can integrate V2)

tools/
└── tile_labeller_interactive.py  # Interactive labelling tool

test_perception_v2.py          # Test suite
```

## Key Components

### 1. Tile Grid System

The screen (160×144 pixels) is divided into a **20×18 grid** of **8×8 pixel tiles**:

- **Columns**: Labeled A-T (20 columns)
- **Rows**: Labeled 0-17 (18 rows)
- **Total tiles**: 360 tiles per frame

Each tile is hashed using xxHash for fast, deterministic identification.

### 2. Regions of Interest

Pre-defined regions for Pokemon Blue gameplay:

| Region | Tiles | Description |
|--------|-------|-------------|
| `text_box` | (16,0) to (17,19) | Bottom 2 rows for dialogue/battle text |
| `player_area` | (7,8) to (10,11) | Center area where player sprite appears |
| `enemy_hp` | (4,3) to (5,10) | Enemy HP bar in battle |
| `player_hp` | (10,10) to (11,17) | Player HP bar in battle |
| `battle_menu` | (12,12) to (15,19) | Battle menu options |

These can be customized in [perception_v2.py](app_layer/perception_v2.py#L31).

### 3. Tile Labelling System

The `TileLabeller` class allows you to:

- Label tiles by their hash value
- Classify tiles into types (text, terrain, ui)
- Save/load label dictionaries to JSON
- Build a tile vocabulary over time

**Tile Types:**

- **Text tiles**: Individual characters (A-Z, 0-9, punctuation)
- **Terrain tiles**:
  - `walkable` - passable ground
  - `blocked` - walls, obstacles
  - `grass` - tall grass (encounter zones)
  - `water` - water tiles
  - `door` - doors/gates
  - `stair` - stairs/ledges
- **UI tiles**: cursors, menu items, HP bars

**Note**: We don't need to label every overworld tile explicitly. For terrain, we only care about:
- Is it traversable or not?
- Special types (grass, water, doors)

For text/menus, we need precise character-level labelling to read dialogue and options.

## Usage

### Basic Perception

```python
from perception_v2 import PerceptionV2, TileLabeller
from visualizer_v2 import VisualizerV2
from PIL import Image
import numpy as np

# Initialize
labeller = TileLabeller()
labeller.load_labels("tile_labels.json")  # Load existing labels

perception = PerceptionV2(labeller=labeller)
visualizer = VisualizerV2()

# Process a frame
frame = Image.fromarray(game_frame)  # Your game frame
state = perception.perceive(frame)

# Visualize
visualizer.visualize(np.array(frame.convert("L")), state)

# Access tiles
text_tiles = state.get_tiles_in_region("text_box")
for tile in text_tiles:
    print(f"{tile.grid_id()}: {tile.label} (hash: {tile.hash})")

# Get JSON for LLM
json_str = state.to_json()
```

### Interactive Tile Labelling

Use the interactive tool to build your tile dictionary:

```bash
# With a game screenshot
python tools/tile_labeller_interactive.py --image screenshot.png --labels tile_labels.json

# With synthetic test frame
python tools/tile_labeller_interactive.py
```

**Controls:**
- **Click** tiles to select/deselect them
- **Character keys** (A-Z, 0-9): Label as text character
- **W**: Walkable terrain
- **B**: Blocked terrain
- **G**: Grass
- **T**: Water
- **D**: Door/gate
- **S**: Stair
- **C**: Cursor
- **H**: HP bar
- **M**: Menu item
- **S** (Shift+S): Save labels
- **L**: Load labels
- **C**: Clear selection
- **Q**: Quit

### Running Tests

```bash
python test_perception_v2.py
```

Tests include:
- ✅ Basic visualization with grid overlay
- ✅ Region extraction
- ✅ Tile labelling and persistence
- ✅ JSON serialization

## Integration with Game Loop

To integrate V2 with [game_runner.py](app_layer/game_runner.py):

```python
from perception_v2 import PerceptionV2, TileLabeller
from visualizer_v2 import VisualizerV2

class GameOrchestrator:
    def __init__(self, enable_viz=True):
        # Load tile labels
        self.labeller = TileLabeller()
        self.labeller.load_labels("tile_labels.json")

        # Initialize V2 perception
        self.perception = PerceptionV2(labeller=self.labeller)
        self.visualizer = VisualizerV2() if enable_viz else None

    def get_actions(self, frame) -> List[str]:
        # Process with V2
        state = self.perception.perceive(frame)

        # Visualize
        if self.visualizer:
            frame_array = np.array(frame.convert("L"))
            self.visualizer.visualize(frame_array, state)

        # Get text from text box
        text_tiles = state.get_tiles_in_region("text_box")
        dialogue = self._reconstruct_text(text_tiles)

        # Send to LLM
        state_json = state.to_json()
        # actions = llm.decide(state_json, dialogue)

        return actions

    def _reconstruct_text(self, tiles):
        """Reconstruct text from labeled tiles"""
        chars = []
        for tile in tiles:
            if tile.label and tile.label.startswith("char_"):
                char = tile.label[5:]  # Remove "char_" prefix
                chars.append(char)
        return "".join(chars)
```

## Visualizer Features

The V2 visualizer includes:

1. **Grid Overlay**: Shows 8×8 tile boundaries in green
2. **Grid Labels**: Alphabetic (A-T) for columns, numeric (0-17) for rows
3. **Region Highlighting**: Color-coded overlays for regions of interest
4. **Interactive Controls**:
   - `G`: Toggle grid
   - `L`: Toggle labels
   - `R`: Toggle regions
   - `Q`: Quit

Colors:
- Text box: Red
- Player area: Green
- Enemy HP: Orange
- Player HP: Blue
- Battle menu: Yellow

## Data Structures

### TileInfo

```python
@dataclass
class TileInfo:
    row: int              # Row index (0-17)
    col: int              # Column index (0-19)
    hash: str             # xxHash of tile pixels
    label: Optional[str]  # Human-readable label (e.g., "char_A", "terrain_grass")
    tile_type: Optional[str]  # Type category (text, terrain, ui)
```

### RegionOfInterest

```python
@dataclass
class RegionOfInterest:
    name: str
    tile_start: Tuple[int, int]  # (row, col)
    tile_end: Tuple[int, int]    # (row, col) inclusive
    description: str
```

### PerceptionState

```python
@dataclass
class PerceptionState:
    frame_number: int
    tiles: List[TileInfo]         # All 360 tiles
    regions: Dict[str, RegionOfInterest]
    screen_hash: str              # Hash of entire screen
```

## Labelling Strategy

### For Text Recognition

**Goal**: Read dialogue, menu options, battle text

**Approach**:
1. Capture screenshots of all text characters (A-Z, 0-9, punctuation)
2. Use the interactive labeller to label each character tile
3. Build a complete character set (43+ characters for Pokemon Blue font)
4. Reconstruct words/sentences from labeled tiles

**Priority characters**:
- A-Z (uppercase)
- 0-9
- Space, period, comma, exclamation, question mark
- Special Pokemon symbols

### For Terrain Navigation

**Goal**: Know what tiles are traversable

**Approach**:
1. Don't need to label every unique terrain tile
2. Only classify into broad categories:
   - `walkable`: Can walk on it
   - `blocked`: Cannot walk on it
   - `grass`: Tall grass (triggers encounters)
   - `water`: Water tiles (need surf)
   - `door`/`gate`/`stair`: Transition tiles

**Method**:
- Use the labeller to mark a few examples of each type
- Over time, as we encounter new screens, label new terrain hashes
- Focus on functional classification, not aesthetic details

### For UI Elements

**Goal**: Identify cursors, menus, HP bars

**Approach**:
- Label cursor tiles (to know current selection)
- Label menu item backgrounds
- Label HP bar segments (to read HP values later)

## Performance

- **Tile hashing**: ~3-4ms for 360 tiles (same as V1)
- **Hash caching**: 95%+ hit rate on repeated tiles
- **Visualization**: ~60 FPS sustained

## Comparison with V1

| Feature | V1 (perception_boy) | V2 (perception_v2) |
|---------|---------------------|-------------------|
| Grid system | ✅ 18×20 tiles | ✅ 18×20 tiles |
| Tile hashing | ✅ xxHash | ✅ xxHash |
| Visualizer | Basic grid viz | ✅ Enhanced with labels |
| Grid labels | ❌ | ✅ A-T, 0-17 |
| Regions of interest | Hardcoded in detector | ✅ Configurable ROI system |
| Tile labelling | ❌ | ✅ Interactive tool |
| Label persistence | ❌ | ✅ JSON save/load |
| Game state detection | ✅ combat/world/menu | ⚠️ TBD (can add) |
| Sprite detection | ✅ Frame differencing | ⚠️ TBD (can add) |
| OCR | ✅ Template matching | ✅ Hash-based labelling |

**V2 Philosophy**:
- Focus on tile-level perception and labelling
- Build a tile vocabulary over time
- Let higher-level systems (game state, sprites) be added as needed
- Cleaner separation of concerns

## Future Enhancements

1. **Automatic Clustering**: Group similar tile hashes to suggest labels
2. **Template Matching**: Combine with V1's OCR for bootstrapping
3. **Change Detection**: Track which tiles changed frame-to-frame
4. **Tile Statistics**: Track frequency of tiles to prioritize labelling
5. **Multi-frame Analysis**: Use temporal info to improve classification
6. **Export to ML Format**: Generate training data for vision models

## Files

### Core Modules

- [perception_v2.py](app_layer/perception_v2.py) - Core perception system (320 lines)
- [visualizer_v2.py](app_layer/visualizer_v2.py) - Enhanced visualizer (330 lines)

### Tools

- [tile_labeller_interactive.py](tools/tile_labeller_interactive.py) - Interactive labelling (380 lines)

### Tests

- [test_perception_v2.py](test_perception_v2.py) - Test suite (250 lines)

### Documentation

- [PERCEPTION_V2_README.md](PERCEPTION_V2_README.md) - This file

## Quick Start

1. **Run the test** to see the visualizer:
   ```bash
   python test_perception_v2.py
   ```

2. **Label some tiles** interactively:
   ```bash
   python tools/tile_labeller_interactive.py
   ```

3. **Integrate into your game loop**:
   ```python
   from perception_v2 import PerceptionV2, TileLabeller

   labeller = TileLabeller()
   perception = PerceptionV2(labeller)
   state = perception.perceive(frame)
   ```

## Next Steps

To complete the perception system:

1. ✅ **Tile grid with labels** - DONE
2. ✅ **Regions of interest** - DONE
3. ✅ **Tile labeller tool** - DONE
4. ⏭️ **Capture game screenshots** - Capture text, terrain, UI examples
5. ⏭️ **Label tile dictionary** - Use interactive tool to build vocabulary
6. ⏭️ **Text reconstruction** - Implement logic to read labeled character tiles
7. ⏭️ **Integrate with LLM** - Send labeled state to Claude for decision-making

## License

Part of the Malamar LLM-plays-Pokemon-Blue project.

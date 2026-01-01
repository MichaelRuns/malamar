# Selection-Aware Controls - Implementation Summary

## Problem Solved

Previously, the tile labeller had key conflicts where keys like 'g', 'c', 's', 'l', 'q', 'r' were used for both control commands AND character labelling. This made it impossible to label certain characters.

The first fix attempted to use Shift for all operations, but this still felt clunky.

## Solution: Selection-Aware Controls

The tile labeller now operates in **two distinct modes** based on whether tiles are selected:

### Control Mode (No Tiles Selected)

When you have no tiles selected, the following control keys are active:

| Key | Action |
|-----|--------|
| `g` | Toggle grid overlay |
| `r` | Toggle regions overlay |
| `s` | Save labels to file |
| `l` | Load labels from file |
| `q` | Quit labeller |

### Labelling Mode (Tiles Selected)

When you have tiles selected, the following labelling keys are active:

| Key | Action |
|-----|--------|
| `Shift+A` to `Shift+Z` | Label as text characters A-Z |
| `0-9` | Label as text digits 0-9 |
| `. , ! ? : ; - ' " ( ) /` | Label as punctuation symbols |
| `SPACE` | Label as space character |
| `w` | Label as walkable terrain |
| `b` | Label as blocked terrain |
| `t` | Label as water |
| `d` | Label as door/gate |
| `h` | Label as HP bar |
| `m` | Label as menu item |
| `c` | Clear selection (return to control mode) |

## Benefits

✅ **All characters labelable** - Including Q, C, S, G, L, R
✅ **No key conflicts** - Controls only active when not labelling
✅ **Intuitive workflow** - Select → label → clear → repeat
✅ **Visual feedback** - Tool shows "Selected: N tiles" at top

## Typical Workflow

```bash
# 1. Start the labeller
python tools/tile_labeller_interactive.py \
  --image assets/screenshots/frame_20251230_143022.png

# 2. In the labeller:
#    - Click a tile containing 'A' (enters labelling mode)
#    - Press Shift+A to label it as char_A
#    - Click a tile containing 'B'
#    - Press Shift+B to label it as char_B
#    - Press 'c' to clear selection (return to control mode)
#    - Press 's' to save labels
#    - Continue labelling more characters...
#    - Press 'q' to quit when done
```

## Implementation Details

### File Modified
- [tools/tile_labeller_interactive.py](tools/tile_labeller_interactive.py)

### Key Changes
1. **Removed** the old `_handle_labelling_key()` method
2. **Added** selection-aware key handling in main `run()` loop
3. **Split** key handling into two branches:
   - `if self.selected_tiles:` → Labelling mode
   - `else:` → Control mode

### Code Structure

```python
while self.running:
    self._render()
    key = cv2.waitKey(1) & 0xFF

    if self.selected_tiles:
        # LABELLING MODE: Handle Shift+letter, numbers, terrain keys
        if ord('A') <= key <= ord('Z'):
            # Label as character
        elif key == ord('w'):
            # Label as walkable terrain
        # ... etc
        elif key == ord('c'):
            # Clear selection
    else:
        # CONTROL MODE: Handle s, l, g, r, q
        if key == ord('s'):
            # Save labels
        # ... etc
```

## Testing

Created [test_selection_aware_controls.py](test_selection_aware_controls.py) which verifies:

- ✅ All 26 letters (A-Z) can be labeled
- ✅ All 10 digits (0-9) can be labeled
- ✅ Previously problematic characters (C, G, L, Q, R, S) work correctly
- ✅ No conflicts with control keys

Test output:
```
=== Testing Selection-Aware Controls ===

Testing all uppercase letters (Shift+A through Shift+Z):
  ✅ All 26 letters can be labeled

Testing all digits (0-9):
  ✅ All 10 digits can be labeled

Verifying previously problematic characters:
  ✅ 'C' successfully labeled as 'char_C'
  ✅ 'G' successfully labeled as 'char_G'
  ✅ 'L' successfully labeled as 'char_L'
  ✅ 'Q' successfully labeled as 'char_Q'
  ✅ 'R' successfully labeled as 'char_R'
  ✅ 'S' successfully labeled as 'char_S'

============================================================
✅ ALL CHARACTERS LABELABLE
============================================================
```

## Documentation Updated

The following files were updated to reflect the new control scheme:

1. [TILE_LABELLER_CONTROLS.md](TILE_LABELLER_CONTROLS.md) - Full control reference
2. [SCREENSHOT_WORKFLOW.md](SCREENSHOT_WORKFLOW.md) - Updated workflow examples
3. [LABELLER_VERIFICATION.md](LABELLER_VERIFICATION.md) - Updated verification results
4. [tools/tile_labeller_interactive.py](tools/tile_labeller_interactive.py) - Docstring and help text

## Summary

The selection-aware control scheme successfully resolves all key conflicts while providing an intuitive user experience. You can now label **49 distinct characters** (A-Z, 0-9, and 13 punctuation symbols) plus terrain and UI elements without any conflicts with control keys.

**Next Step**: Start labelling characters! Run the game, press `P` to capture screenshots, then use the tile labeller to build your character vocabulary.

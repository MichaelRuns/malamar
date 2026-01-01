# Tile Labeller Controls - Quick Reference

## Fixed: Selection-Aware Controls

The tile labeller now uses **selection-aware** controls to avoid conflicts between control keys and character labelling.

## Controls Summary

### When NO Tiles Selected (Control Mode)
| Key | Action |
|-----|--------|
| **g** | Toggle grid overlay |
| **r** | Toggle regions overlay |
| **s** | Save labels to file |
| **l** | Load labels from file |
| **q** | Quit labeller |

### When Tiles ARE Selected (Labelling Mode)

#### Character Labelling
| Key | Label |
|-----|-------|
| **Shift+A** through **Shift+Z** | Label as text character A-Z |
| **0-9** | Label as text character 0-9 |

#### Punctuation & Symbols
| Key | Label |
|-----|-------|
| **.** | Period |
| **,** | Comma |
| **!** | Exclamation mark |
| **?** | Question mark |
| **:** | Colon |
| **;** | Semicolon |
| **-** | Hyphen/dash |
| **'** | Apostrophe/single quote |
| **"** | Double quote |
| **(** | Left parenthesis |
| **)** | Right parenthesis |
| **/** | Slash |
| **SPACE** | Space character |

**Example:** To label a tile as the letter 'G':
1. Click the tile to select it (enters labelling mode)
2. Press **Shift+G**
3. Tile is now labeled as `char_G`

**Example:** To label a tile as a period:
1. Click the tile to select it
2. Press **.** (period key)
3. Tile is now labeled as `char_.`

#### Terrain Labelling
| Key | Label |
|-----|-------|
| **w** | walkable terrain |
| **b** | blocked terrain |
| **t** | water |
| **d** | door/gate |

#### UI Labelling
| Key | Label |
|-----|-------|
| **h** | HP bar |
| **m** | menu item |
| **>** | cursor (menu selector) |

#### Exit Labelling Mode
| Key | Action |
|-----|--------|
| **c** | Clear selection (return to control mode) |

## How It Works

**The key behavior changes based on whether you have tiles selected:**

- **No selection** → Control mode → 's' saves, 'g' toggles grid, 'q' quits
- **Tiles selected** → Labelling mode → 'Shift+S' labels as 'char_S', 'Shift+G' labels as 'char_G'

**This means:**
- You can label ALL characters including Q, C, S, G, L, R
- No conflicts between controls and character labelling
- Intuitive workflow: select → label → clear → repeat

## Workflow Examples

### Example 1: Labelling the word "HELLO"
1. Click tiles containing H, E, L, L, O (enters labelling mode)
2. Press: **Shift+H**, **Shift+E**, **Shift+L**, **Shift+L**, **Shift+O**
3. Each tile gets labeled as `char_H`, `char_E`, `char_L`, etc.
4. Press **c** to clear selection (return to control mode)

### Example 2: Labelling numbers "123"
1. Click tiles containing 1, 2, 3 (enters labelling mode)
2. Press: **1**, **2**, **3** (no shift needed for numbers)
3. Each tile gets labeled as `char_1`, `char_2`, `char_3`
4. Press **c** to clear selection

### Example 3: Labelling terrain
1. Click a grass tile (enters labelling mode)
2. Press: **w** for walkable
3. Tile labeled as `terrain_walkable`
4. Press **c** to clear selection

### Example 4: Saving your work
1. Make sure no tiles are selected (control mode)
2. Press: **s** (lowercase, no shift)
3. Labels saved to `tile_labels.json`

## Visual Feedback

The labeller provides visual cues to help you track your progress:

- **Faint green highlight**: Tiles that have been labeled
- **Bright cyan highlight**: Currently selected tiles
- **Grid overlay**: Shows 8x8 tile boundaries (toggle with 'g')
- **Region overlay**: Shows regions of interest (toggle with 'r')

## Tips

1. **Check selection status:** The tool shows "Selected: N tiles" at the top
2. **Remember Shift for letters:** Character labels always need Shift (A-Z)
3. **Numbers don't need Shift:** Just press 0-9 directly
4. **Control mode vs labelling mode:** The mode depends on whether tiles are selected
5. **Clear to return:** Press 'c' when tiles are selected to return to control mode
6. **Green = labeled:** Look for the faint green highlight to see what you've already labeled

## All Characters Available

You can now label:
- **A-Z**: Use Shift+A through Shift+Z (26 letters)
- **0-9**: Use 0-9 (no shift) (10 digits)
- **Punctuation**: . , ! ? : ; - ' " ( ) / SPACE (13 symbols)

**Total: 49 distinct characters** that can be labeled for text reading!

## See Also

- [SCREENSHOT_WORKFLOW.md](SCREENSHOT_WORKFLOW.md) - Full workflow guide
- [PERCEPTION_V2_README.md](PERCEPTION_V2_README.md) - Perception system overview
- [tools/tile_labeller_interactive.py](tools/tile_labeller_interactive.py) - The tool itself

# Screenshot Capture Workflow

## Quick Start

### 1. Run the Game
```bash
python app_layer/game_runner.py
```

### 2. Take Screenshots
- Ensure you're in **manual mode** (press `M` if not)
- Navigate to interesting screens with your arrow keys
- Press `P` to capture a screenshot
- Screenshot saved to `assets/screenshots/frame_YYYYMMDD_HHMMSS.png`

### 3. Label Tiles
```bash
python tools/tile_labeller_interactive.py --image assets/screenshots/frame_20251230_143022.png
```

## Screenshot Controls

| Key | Action |
|-----|--------|
| `P` | Take screenshot (manual mode only) |
| `M` | Toggle manual mode |
| `V` | Toggle perception visualizer |

## What Gets Captured

- **Format**: Grayscale PNG
- **Size**: 160×144 pixels (native Game Boy resolution)
- **Location**: `assets/screenshots/`
- **Naming**: `frame_YYYYMMDD_HHMMSS.png` (timestamp-based)

## Features

### ✅ Automatic Deduplication
The system prevents saving duplicate frames:
- Uses screen hash to detect identical frames
- If you press `P` twice on the same screen → second press shows warning
- Allows capturing same screen later (only blocks consecutive duplicates)

**Example:**
```
📸 Screenshot saved: assets/screenshots/frame_20251230_143022.png
⚠️  Duplicate frame, not saving  # Pressed P again without moving
📸 Screenshot saved: assets/screenshots/frame_20251230_143045.png  # Moved to new screen
```

### ✅ Manual Mode Only
Screenshots only work in manual mode to avoid accidental captures during AI gameplay.

## Recommended Screens to Capture

### Priority 1: Text Characters
Capture screens with dialogue to label all text characters:

- Opening dialogue (Professor Oak)
- NPC conversations
- Battle text ("PIKACHU used THUNDERBOLT!")
- Menu text (Pokémon names, item names)
- Status messages

**Goal**: Label all characters A-Z, 0-9, and punctuation

### Priority 2: Menu Elements
Capture different menu screens:

- Start menu
- Pokémon menu
- Item menu
- Battle menu
- Save menu

**Goal**: Label cursors, menu backgrounds, selection indicators

### Priority 3: Battle Screens
Capture battle scenarios:

- Battle UI (HP bars, status)
- Different Pokémon
- Different moves
- Different status conditions

**Goal**: Label HP bar segments, status icons

### Priority 4: Overworld Terrain
Capture different areas:

- Grass tiles
- Water tiles
- Buildings/doors
- Walls/obstacles
- NPCs

**Goal**: Label terrain as walkable/blocked/special

## Workflow Example

1. **Start game and load saved state:**
   ```bash
   python app_layer/game_runner.py
   # Press L to load game
   ```

2. **Navigate to dialogue:**
   - Move around until you find an NPC
   - Talk to them (Press Z for A button)
   - Press P to capture the dialogue

3. **Label the screenshot:**
   ```bash
   python tools/tile_labeller_interactive.py \
     --image assets/screenshots/frame_20251230_143022.png \
     --labels tile_labels.json
   ```

4. **In the labeller:**
   - Click tiles in the text box to select them (enters labelling mode)
   - Press **Shift+letter** to label as characters (e.g., Shift+A for 'A', Shift+H for 'H')
   - Press numbers directly for numeric characters (e.g., 0-9)
   - Press punctuation keys for symbols (e.g., ., !, ?, :, etc.)
   - Press **SPACE** to label space characters
   - **Labeled tiles show a faint green highlight** - easy to see what's done!
   - Press **c** to clear selection (return to control mode)
   - Press **s** to save labels (only works in control mode - no tiles selected)
   - Press **q** to quit (only works in control mode)

5. **Repeat:**
   - Next time you run the game, labels are auto-loaded
   - Continue capturing and labelling until you have full character set

## File Structure

```
assets/
└── screenshots/
    ├── .last_hash                      # Hidden deduplication tracker
    ├── frame_20251230_143022.png       # Opening dialogue
    ├── frame_20251230_143045.png       # NPC conversation
    ├── frame_20251230_143112.png       # Battle screen
    └── frame_20251230_143156.png       # Menu screen

tile_labels.json                        # Your tile vocabulary
```

## Troubleshooting

### "Cannot save screenshot: game state not yet initialized"
- **Cause**: Pressed P before the game started processing frames
- **Solution**: Wait a second after starting, then press P

### "Duplicate frame, not saving"
- **Cause**: Screen hasn't changed since last screenshot
- **Solution**: This is normal - move to a different screen first

### Screenshots saving to wrong location
- **Cause**: Running from wrong directory
- **Solution**: Ensure you're in `/Users/michaelvernau/repos/malamar` when running

### Can't press P in AI mode
- **Cause**: Screenshot capture only works in manual mode
- **Solution**: Press M to enter manual mode first

## Tips

1. **Capture systematically:**
   - Start with all text characters
   - Move on to menu elements
   - Finally terrain tiles

2. **Use save states:**
   - Save before important dialogue (`S` key)
   - Load and replay to capture missed text (`L` key)

3. **Check what you've captured:**
   ```bash
   ls -lt assets/screenshots/  # List newest first
   open assets/screenshots/frame_20251230_143022.png  # Preview on macOS
   ```

4. **Clean up duplicates:**
   - The system prevents consecutive duplicates automatically
   - But you can manually delete unwanted screenshots

5. **Batch labelling:**
   - Capture 5-10 screenshots of different text
   - Label them all in one session
   - More efficient than switching constantly

## Next Steps

After capturing and labelling screenshots:

1. **Build character set** (Priority 1):
   - Label all A-Z (26 letters)
   - Label all 0-9 (10 digits)
   - Label punctuation: . , ! ? : ; - ' " ( ) / SPACE (13 symbols)
   - **Total: 49 characters** for complete text reading

2. **Test text reading**:
   - Run game and check console for "Text detected: ..."
   - Verify it reads dialogue correctly

3. **Expand vocabulary**:
   - Add terrain types as you encounter them
   - Add menu elements
   - Add UI components

4. **Integrate with LLM**:
   - Once you can read text, send to Claude
   - LLM can make decisions based on dialogue

## See Also

- [PERCEPTION_V2_README.md](PERCEPTION_V2_README.md) - Full perception system guide
- [MIGRATION_TO_V2.md](MIGRATION_TO_V2.md) - Migration from V1
- [tools/tile_labeller_interactive.py](tools/tile_labeller_interactive.py) - Labelling tool

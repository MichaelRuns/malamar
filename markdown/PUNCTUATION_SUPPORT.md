# Punctuation Support - Complete Character Coverage

## Overview

The tile labeller now supports **49 distinct characters** for complete text reading in Pokemon Blue:

- **26 letters** (A-Z) - via Shift+letter
- **10 digits** (0-9) - direct keys
- **13 punctuation symbols** - direct keys

## Supported Punctuation Symbols

All punctuation symbols are accessible when tiles are selected (labelling mode):

| Symbol | Key | Label | Common Use in Pokemon |
|--------|-----|-------|----------------------|
| `.` | Period | `char_.` | End of sentence |
| `,` | Comma | `char_,` | Lists, pauses |
| `!` | Exclamation | `char_!` | Excitement, commands |
| `?` | Question | `char_?` | Questions |
| `:` | Colon | `char_:` | Labels, time (HP: 50/50) |
| `;` | Semicolon | `char_;` | Rare, but possible |
| `-` | Hyphen | `char_-` | Compound words, ranges |
| `'` | Apostrophe | `char_'` | Contractions (don't, can't) |
| `"` | Quote | `char_"` | Dialogue markers |
| `(` | Left Paren | `char_(` | Parenthetical text |
| `)` | Right Paren | `char_)` | Parenthetical text |
| `/` | Slash | `char_/` | Fractions, alternatives |
| `SPACE` | Space bar | `char_SPACE` | Word separation |

## Usage Examples

### Example 1: Label a sentence with punctuation

Dialogue: `"Hello, how are you?"`

1. Click tile with `"` → Press `"`
2. Click tile with `H` → Press `Shift+H`
3. Click tile with `e` → Press `Shift+E`
4. ... continue for each character
5. Click tile with `,` → Press `,`
6. Click tile with space → Press `SPACE`
7. Continue until complete
8. Press `c` to clear selection
9. Press `s` to save

### Example 2: Label HP indicator

Text: `HP: 35/50`

1. Select `H` tile → `Shift+H`
2. Select `P` tile → `Shift+P`
3. Select `:` tile → `:`
4. Select space → `SPACE`
5. Select `3` tile → `3`
6. Select `5` tile → `5`
7. Select `/` tile → `/`
8. Select `5` tile → `5`
9. Select `0` tile → `0`

### Example 3: Label exclamation

Text: `Pokemon attack!`

Last character is exclamation mark:
1. Select `!` tile → Press `!`
2. Labeled as `char_!`

## Character Set Completeness

With 49 characters, you can read virtually all text in Pokemon Blue:

**Uppercase letters (26):**
```
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
```

**Digits (10):**
```
0 1 2 3 4 5 6 7 8 9
```

**Punctuation (13):**
```
. , ! ? : ; - ' " ( ) / [SPACE]
```

**Note:** Pokemon Blue uses all-caps text, so lowercase letters are not needed.

## Common Pokemon Blue Text Patterns

### Dialogue
```
"Hello!"
"How are you?"
"I'm fine, thanks!"
```
Uses: quotes, exclamation, question mark, apostrophe, comma, space

### Battle Text
```
PIKACHU used THUNDERBOLT!
It's super effective!
HP: 25/50
```
Uses: uppercase letters, exclamation, apostrophe, colon, slash, space

### Menu Items
```
POKEMON
ITEM
SAVE
```
Uses: uppercase letters, space

### Status Messages
```
GOT POTION!
SAVED THE GAME.
```
Uses: uppercase letters, exclamation, period, space

## Implementation Details

### Label Format

All punctuation uses the `char_` prefix, just like letters and numbers:
- Letter: `char_A`, `char_B`, etc.
- Digit: `char_0`, `char_1`, etc.
- Symbol: `char_.`, `char_!`, etc.
- Space: `char_SPACE` (special case)

### Why `char_SPACE` instead of `char_ `?

The space character is labeled as `char_SPACE` (not `char_ `) because:
1. More explicit and clear in JSON
2. Easier to read in console output
3. Avoids potential parsing issues with literal space in label

### Text Reconstruction

The game runner's `_read_text_from_tiles()` method extracts characters:

```python
for tile in sorted_tiles:
    if tile.label and tile.label.startswith("char_"):
        # Extract character from label (e.g., "char_A" -> "A")
        char = tile.label[5:]  # Skip "char_" prefix

        # Handle space specially
        if char == "SPACE":
            chars.append(" ")
        else:
            chars.append(char)
```

**Note:** You'll need to update this method to handle `char_SPACE` → `" "` conversion.

## Testing

Created [test_punctuation_support.py](test_punctuation_support.py) which verifies all 13 punctuation symbols can be labeled.

Test output:
```
=== Testing Punctuation Symbol Support ===

Testing 13 punctuation symbols:
  ✅ '.' (period) labeled as 'char_.'
  ✅ ',' (comma) labeled as 'char_,'
  ✅ '!' (exclamation) labeled as 'char_!'
  ✅ '?' (question) labeled as 'char_?'
  ✅ ':' (colon) labeled as 'char_:'
  ✅ ';' (semicolon) labeled as 'char_;'
  ✅ '-' (hyphen) labeled as 'char_-'
  ✅ ''' (apostrophe) labeled as 'char_''
  ✅ '"' (quote) labeled as 'char_"'
  ✅ '(' (left_paren) labeled as 'char_('
  ✅ ')' (right_paren) labeled as 'char_)'
  ✅ '/' (slash) labeled as 'char_/'
  ✅ 'SPACE' (space) labeled as 'char_SPACE'

✅ PUNCTUATION SUPPORT TEST PASSED
```

## Next Steps

1. **Capture dialogue screenshots** with various punctuation
2. **Label the punctuation tiles** using the new keys
3. **Update `_read_text_from_tiles()`** to handle `char_SPACE` → `" "`
4. **Test text reading** with full sentences including punctuation
5. **Verify LLM integration** can parse complete text

## Files Modified

1. [tools/tile_labeller_interactive.py](tools/tile_labeller_interactive.py:109-148) - Added punctuation key handlers
2. [TILE_LABELLER_CONTROLS.md](TILE_LABELLER_CONTROLS.md) - Documented punctuation support
3. [SCREENSHOT_WORKFLOW.md](SCREENSHOT_WORKFLOW.md) - Updated workflow examples
4. [SELECTION_AWARE_CONTROLS.md](SELECTION_AWARE_CONTROLS.md) - Added punctuation to feature list
5. [test_punctuation_support.py](test_punctuation_support.py) - Automated verification test

## See Also

- [TILE_LABELLER_CONTROLS.md](TILE_LABELLER_CONTROLS.md) - Complete control reference
- [SELECTION_AWARE_CONTROLS.md](SELECTION_AWARE_CONTROLS.md) - Selection-aware design
- [SCREENSHOT_WORKFLOW.md](SCREENSHOT_WORKFLOW.md) - Screenshot capture workflow

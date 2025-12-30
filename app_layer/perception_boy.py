from PIL import Image
import cv2
import numpy as np
import json
import xxhash
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class GameState:
    """Structured game state representation"""
    meta: dict
    grid: dict
    regions: dict
    sprites: List[dict]
    changes: dict

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(asdict(self), indent=2)


class GameStateDetector:
    """Detect high-level game state (combat, world, menu, dialogue)"""

    STATES = ["combat", "world", "menu", "dialogue", "inventory", "pause"]

    def __init__(self):
        self.previous_state = None

    def detect_state(self, arr: np.ndarray, grid_hashes: dict) -> str:
        """
        Classify game state using region analysis
        Priority: combat > dialogue > menu > world
        """
        # Combat detection - check for HP bars
        if self._has_battle_ui(arr):
            return "combat"

        # Dialogue detection - check for text box
        if self._has_dialogue_box(arr):
            return "dialogue"

        # Menu detection - check for menu patterns
        if self._has_menu_ui(arr):
            return "menu"

        # Default to world navigation
        return "world"

    def _has_battle_ui(self, arr: np.ndarray) -> bool:
        """Check for battle UI elements (HP bars)"""
        if arr.shape[0] < 88 or arr.shape[1] < 136:
            return False

        # Pokemon Blue has HP bars at specific positions
        # Enemy HP bar region: Y=32-40, X=24-80
        # Player HP bar region: Y=80-88, X=80-136
        try:
            enemy_hp_region = arr[32:40, 24:80]
            player_hp_region = arr[80:88, 80:136]

            # Check for HP bar patterns (horizontal lines with contrast)
            has_enemy_bar = self._has_hp_bar_pattern(enemy_hp_region)
            has_player_bar = self._has_hp_bar_pattern(player_hp_region)

            return has_enemy_bar or has_player_bar
        except:
            return False

    def _has_hp_bar_pattern(self, region: np.ndarray) -> bool:
        """Detect HP bar by horizontal line structure"""
        if region.shape[0] < 4 or region.shape[1] < 40:
            return False

        # HP bar has horizontal variation (dark edges, lighter middle)
        middle_row = region[region.shape[0]//2, :]
        edges = np.diff(middle_row.astype(int))

        # Look for at least 2 significant transitions
        return np.sum(np.abs(edges) > 30) >= 2

    def _has_dialogue_box(self, arr: np.ndarray) -> bool:
        """Check for dialogue text box at bottom of screen"""
        if arr.shape[0] < 144:
            return False

        # Dialogue box region: Y=128-144 (bottom 16 pixels / 2 rows)
        text_region = arr[128:144, :]

        # Look for border pattern (darker edges)
        top_row = text_region[0, :]
        bottom_row = text_region[-1, :]

        # Check if top/bottom rows are darker (border indicators)
        mean_border = (np.mean(top_row) + np.mean(bottom_row)) / 2
        mean_interior = np.mean(text_region[4:12, :])

        # Border should be darker than interior
        return mean_interior > mean_border + 20

    def _has_menu_ui(self, arr: np.ndarray) -> bool:
        """Check for menu patterns (centered rectangles, low entropy)"""
        # Calculate grid variance - menus have large solid regions
        variance = np.var(arr)

        # Menus typically have lower variance than world navigation
        # but higher than solid black screen
        return 100 < variance < 800


class SpriteDetector:
    """Detect and track sprites (player, NPCs, Pokemon)"""

    def __init__(self):
        self.previous_frame = None
        self.tracked_sprites = {}
        self.next_sprite_id = 0

    def detect_sprites(self, arr: np.ndarray, grid_hashes: dict) -> List[dict]:
        """
        Detect sprites using frame differencing
        Returns list of sprite dictionaries with positions and bounds
        """
        sprites = []

        # Method 1: Frame differencing (only if we have previous frame)
        if self.previous_frame is not None:
            sprites = self._detect_by_difference(arr)

        # Store current frame for next comparison
        self.previous_frame = arr.copy()

        return sprites

    def _detect_by_difference(self, current: np.ndarray) -> List[dict]:
        """Find sprites by comparing with previous frame"""
        # Compute absolute difference
        diff = np.abs(current.astype(int) - self.previous_frame.astype(int))

        # Threshold to binary mask
        change_mask = diff > 15

        # Find connected components using simple blob detection
        sprites = []

        # Simple blob detection: scan for contiguous changed regions
        # (Using scipy would be better but let's keep it simple for now)
        visited = np.zeros_like(change_mask, dtype=bool)

        for y in range(change_mask.shape[0]):
            for x in range(change_mask.shape[1]):
                if change_mask[y, x] and not visited[y, x]:
                    # Found a new blob - flood fill to get bounds
                    blob_pixels = self._flood_fill(change_mask, visited, x, y)

                    if len(blob_pixels) < 10:  # Too small
                        continue

                    # Get bounding box
                    xs = [p[0] for p in blob_pixels]
                    ys = [p[1] for p in blob_pixels]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    width = x_max - x_min + 1
                    height = y_max - y_min + 1

                    # Filter by sprite size (Game Boy sprites are 8x8 or 16x16)
                    if 6 <= width <= 32 and 6 <= height <= 32:
                        sprite_region = current[y_min:y_max+1, x_min:x_max+1]
                        sprite_hash = self._hash_sprite(sprite_region)

                        sprites.append({
                            "id": f"sprite_{self.next_sprite_id}",
                            "pos": [y_min // 8, x_min // 8],  # Tile coordinates
                            "bounds": {"x": x_min, "y": y_min, "w": width, "h": height},
                            "hash": sprite_hash
                        })
                        self.next_sprite_id += 1

        return sprites

    def _flood_fill(self, mask: np.ndarray, visited: np.ndarray, start_x: int, start_y: int) -> List[Tuple[int, int]]:
        """Simple flood fill to find connected pixels"""
        stack = [(start_x, start_y)]
        pixels = []

        while stack and len(pixels) < 1000:  # Limit to prevent runaway
            x, y = stack.pop()

            if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
                continue
            if visited[y, x] or not mask[y, x]:
                continue

            visited[y, x] = True
            pixels.append((x, y))

            # Add 4-connected neighbors
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])

        return pixels

    def _hash_sprite(self, sprite_region: np.ndarray) -> str:
        """Hash sprite region for identification"""
        sprite_bytes = sprite_region.astype(np.uint8).tobytes()
        return xxhash.xxh64(sprite_bytes, seed=0).hexdigest()[:8]


class TextReader:
    """Read text using font templates"""

    def __init__(self, template_dir: str = "assets/font/templates"):
        self.template_dir = template_dir
        self.char_templates = {}
        self._load_templates()

    def _load_templates(self):
        """Load character templates from disk"""
        import os
        if not os.path.exists(self.template_dir):
            print(f"Warning: Font template directory not found: {self.template_dir}")
            return

        # Load all PNG files as character templates
        for filename in os.listdir(self.template_dir):
            if filename.endswith('.png'):
                char = filename[:-4]  # Remove .png extension
                template_path = os.path.join(self.template_dir, filename)
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.char_templates[char] = template

        if self.char_templates:
            print(f"Loaded {len(self.char_templates)} character templates")

    def read_text_region(self, arr: np.ndarray, region: Tuple[int, int, int, int],
                         threshold: float = 0.75) -> Tuple[str, float]:
        """
        Read text from a specific region
        Returns (text, average_confidence)
        """
        if not self.char_templates:
            return "", 0.0

        x, y, w, h = region
        text_region = arr[y:y+h, x:x+w]

        # Split into 8x8 tiles (character size)
        chars = []
        confidences = []

        tile_size = 8
        for tile_x in range(0, w - tile_size + 1, tile_size):
            tile = text_region[:, tile_x:tile_x+tile_size]

            if tile.shape != (h, tile_size):
                continue

            # Match against all character templates
            best_char, best_conf = self._match_character(tile)

            if best_conf > threshold:
                chars.append(best_char)
                confidences.append(best_conf)
            elif np.var(tile) < 10:  # Low variance = likely space
                chars.append(' ')
                confidences.append(1.0)

        text = ''.join(chars).strip()
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return text, avg_confidence

    def _match_character(self, tile: np.ndarray) -> Tuple[str, float]:
        """Match tile against character templates"""
        best_char = ''
        best_score = 0.0

        for char, template in self.char_templates.items():
            # Resize template to match tile height if needed
            if template.shape[0] != tile.shape[0]:
                scale = tile.shape[0] / template.shape[0]
                new_width = int(template.shape[1] * scale)
                template = cv2.resize(template, (new_width, tile.shape[0]))

            if template.shape != tile.shape:
                continue

            # Normalized cross-correlation
            try:
                result = cv2.matchTemplate(tile, template, cv2.TM_CCOEFF_NORMED)
                score = result[0, 0] if result.size > 0 else 0.0

                if score > best_score:
                    best_score = score
                    best_char = char
            except:
                continue

        return best_char, best_score


class PerceptionModule:
    """
    Multi-layer perception system for Pokemon Blue
    Returns deterministic JSON representation of game state
    """

    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        self.previous_frames = []
        self.frame_similarity_threshold = 0.95

        # Frame tracking
        self.frame_count = 0
        self.previous_grid_hashes = None
        self.previous_screen_hash = None

        # Component systems
        self.state_detector = GameStateDetector()
        self.sprite_detector = SpriteDetector()
        self.text_reader = TextReader()

        # Performance optimization: hash cache
        self.hash_cache = {}

    def perceive(self, frame: Image) -> GameState:
        """
        Main perception pipeline: Frame -> Structured JSON
        """
        self.frame_count += 1
        arr = np.array(frame.convert("L"))

        # Step 1: Extract and hash all tiles
        grid_data = self._extract_and_hash_grid(arr)

        # Step 2: Detect high-level game state
        game_state = self.state_detector.detect_state(arr, grid_data["hashes"])

        # Step 3: Extract region-specific information
        regions = self._extract_regions(arr, game_state)

        # Step 4: Detect sprites
        sprites = self.sprite_detector.detect_sprites(arr, grid_data["hashes"])

        # Step 5: Compute changes from previous frame
        changes = self._compute_changes(grid_data, game_state)

        # Step 6: Build structured output
        state = GameState(
            meta={
                "frame_number": self.frame_count,
                "screen_hash": grid_data["screen_hash"],
                "game_state": game_state,
                "previous_state": self.state_detector.previous_state
            },
            grid=grid_data,
            regions=regions,
            sprites=sprites,
            changes=changes
        )

        # Update history
        self.previous_grid_hashes = grid_data["hashes"]
        self.previous_screen_hash = grid_data["screen_hash"]
        self.state_detector.previous_state = game_state

        return state

    def _extract_and_hash_grid(self, arr: np.ndarray) -> dict:
        """
        Extract 18x20 grid of 8x8 tiles and hash each one
        FIXED BUG: Was using col*16, now uses col*tile_width
        """
        tile_width = 8
        tile_height = 8
        rows, cols = 18, 20

        cells = []
        hashes_dict = {}
        all_hashes = []

        for row in range(rows):
            for col in range(cols):
                # FIXED: Now uses col*tile_width instead of col*16
                tile = arr[row*tile_height:(row+1)*tile_height,
                          col*tile_width:(col+1)*tile_width]

                # Generate hash
                tile_hash = self._hash_tile(tile)
                all_hashes.append(tile_hash)

                cells.append({
                    "pos": [row, col],
                    "hash": tile_hash,
                    "type": self._classify_tile_type(tile)
                })

                # Store with string key for JSON compatibility
                hashes_dict[f"{row},{col}"] = tile_hash

        # Compute overall screen hash (for frame deduplication)
        screen_hash = xxhash.xxh64("".join(all_hashes).encode(), seed=0).hexdigest()[:16]

        # RLE encoding for compression
        rle_encoded = self._rle_encode(all_hashes)

        return {
            "dimensions": {"rows": rows, "cols": cols, "cell_size": 8},
            "cells": cells,
            "rle_encoded": rle_encoded,
            "screen_hash": screen_hash,
            "hashes": hashes_dict
        }

    def _hash_tile(self, tile: np.ndarray) -> str:
        """Fast deterministic hash for 8x8 tile using xxHash"""
        tile_bytes = tile.astype(np.uint8).tobytes()

        # Check cache first
        if tile_bytes in self.hash_cache:
            return self.hash_cache[tile_bytes]

        h = xxhash.xxh64(tile_bytes, seed=0).hexdigest()[:8]

        # Cache result (limit cache size)
        if len(self.hash_cache) < 10000:
            self.hash_cache[tile_bytes] = h

        return h

    def _classify_tile_type(self, tile: np.ndarray) -> str:
        """Classify tile as solid, text, or background based on variance"""
        variance = np.var(tile)

        if variance < 10:
            return "solid"
        elif variance > 100:
            return "text"
        else:
            return "bg"

    def _rle_encode(self, hashes: List[str]) -> str:
        """Run-length encode hash sequence for compression"""
        if not hashes:
            return ""

        encoded = []
        current = hashes[0]
        count = 1

        for h in hashes[1:]:
            if h == current:
                count += 1
            else:
                encoded.append(f"{current}*{count}" if count > 1 else current)
                current = h
                count = 1

        encoded.append(f"{current}*{count}" if count > 1 else current)
        return ",".join(encoded)

    def _extract_regions(self, arr: np.ndarray, game_state: str) -> dict:
        """Extract semantic information from known screen regions"""
        regions = {}

        if game_state == "combat":
            regions["hp_bars"] = self._extract_hp_bars(arr)
            # Battle menu would go here

        if game_state in ["dialogue", "combat"]:
            regions["text_box"] = self._extract_text_box(arr)

        return regions

    def _extract_hp_bars(self, arr: np.ndarray) -> List[dict]:
        """Extract HP bar information"""
        hp_bars = []

        # Enemy HP bar: Y=32-40, X=24-80
        # Player HP bar: Y=80-88, X=80-136
        # This is a stub - full implementation would measure bar width

        # For now, just indicate presence
        if arr.shape[0] >= 88 and arr.shape[1] >= 136:
            hp_bars.append({
                "entity": "player",
                "present": True,
                "region": {"x": 80, "y": 80, "width": 56, "height": 8}
            })
            hp_bars.append({
                "entity": "enemy",
                "present": True,
                "region": {"x": 24, "y": 32, "width": 56, "height": 8}
            })

        return hp_bars

    def _extract_text_box(self, arr: np.ndarray) -> dict:
        """Extract text from dialogue box region"""
        if arr.shape[0] < 144:
            return {"present": False}

        # Dialogue box at bottom 2 rows (Y=128-144)
        text, confidence = self.text_reader.read_text_region(
            arr,
            region=(0, 128, 160, 16),
            threshold=0.70
        )

        return {
            "present": True,
            "y": 128,
            "height": 16,
            "text": text,
            "confidence": float(confidence)
        }

    def _compute_changes(self, current_grid: dict, current_state: str) -> dict:
        """Compute delta from previous frame"""
        if self.previous_grid_hashes is None:
            return {"first_frame": True}

        changed_cells = []
        for key, hash_val in current_grid["hashes"].items():
            if self.previous_grid_hashes.get(key) != hash_val:
                # Parse "row,col" back to [row, col]
                row, col = map(int, key.split(','))
                changed_cells.append([row, col])

        return {
            "cells_modified": changed_cells,
            "num_changes": len(changed_cells),
            "state_transition": (current_state != self.state_detector.previous_state
                               if self.state_detector.previous_state else False)
        }





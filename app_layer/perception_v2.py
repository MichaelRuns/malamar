"""
Tile-based perception module for Pokemon Blue
Segments screen into tiles, hashes them, and provides labelling infrastructure
"""

import numpy as np
import xxhash
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
import json


# Game Boy screen dimensions
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144
TILE_SIZE = 8
GRID_COLS = SCREEN_WIDTH // TILE_SIZE  # 20
GRID_ROWS = SCREEN_HEIGHT // TILE_SIZE  # 18


@dataclass
class TileInfo:
    """Information about a single tile"""
    row: int
    col: int
    hash: str
    label: Optional[str] = None
    tile_type: Optional[str] = None

    def grid_id(self) -> str:
        """Return grid identifier like 'A0', 'B5', etc."""
        col_label = chr(ord('A') + self.col)
        return f"{col_label}{self.row}"


@dataclass
class WordInfo:
    """Represents an extracted word with position"""
    word: str
    start_row: int
    start_col: int
    end_col: int


@dataclass
class MenuSelection:
    """Represents the currently selected menu option"""
    option: Optional[str]  # "FIGHT", "PKMN", "ITEM", "RUN"
    cursor_position: Optional[Tuple[int, int]]  # (row, col)
    menu_type: Optional[str]  # "battle_menu" or None


@dataclass
class HPReading:
    """HP bar reading for a combatant"""
    percentage: Optional[float]  # 0.0 to 1.0
    hp_bar_tiles: int
    total_tiles: int


@dataclass
class BattleHP:
    """HP readings for both combatants"""
    player: 'HPReading'
    enemy: 'HPReading'


@dataclass
class BattleNames:
    """Pokemon names in battle"""
    enemy_name: Optional[str]
    player_name: Optional[str]


@dataclass
class LabeledTileGroups:
    """Tiles grouped by category"""
    text: List['TileInfo']
    terrain: List['TileInfo']
    ui: List['TileInfo']
    sprite: List['TileInfo']
    unlabeled: List['TileInfo']


@dataclass
class RegionOfInterest:
    """Defines a region of interest on the screen"""
    name: str
    tile_start: Tuple[int, int]  # (row, col)
    tile_end: Tuple[int, int]    # (row, col) inclusive
    description: str

    def contains_tile(self, row: int, col: int) -> bool:
        """Check if a tile is within this region"""
        return (self.tile_start[0] <= row <= self.tile_end[0] and
                self.tile_start[1] <= col <= self.tile_end[1])

    def get_pixel_bounds(self) -> Tuple[int, int, int, int]:
        """Return pixel bounds as (x, y, width, height)"""
        x = self.tile_start[1] * TILE_SIZE
        y = self.tile_start[0] * TILE_SIZE
        w = (self.tile_end[1] - self.tile_start[1] + 1) * TILE_SIZE
        h = (self.tile_end[0] - self.tile_start[0] + 1) * TILE_SIZE
        return (x, y, w, h)


# Define common regions of interest for Pokemon Blue
REGIONS_OF_INTEREST = {
    "text_box": RegionOfInterest(
        name="text_box",
        tile_start=(13, 1),  # Bottom 2 rows
        tile_end=(16, 18),
        description="Text box for dialogue and battle text"
    ),
    "player_area": RegionOfInterest(
        name="player_area",
        tile_start=(7, 8),   # Center area where player typically is
        tile_end=(9, 9),
        description="Central area where player sprite appears"
    ),
    "enemy_hp": RegionOfInterest(
        name="enemy_hp",
        tile_start=(2, 4),   # Enemy HP bar region
        tile_end=(2, 9),
        description="Enemy HP bar in battle"
    ),
    "player_hp": RegionOfInterest(
        name="player_hp",
        tile_start=(9, 12), # Player HP bar region
        tile_end=(9, 17),
        description="Player HP bar in battle"
    ),
    "battle_menu": RegionOfInterest(
        name="battle_menu",
        tile_start=(13, 9), # Battle menu region
        tile_end=(16, 18),
        description="Battle menu options"
    ),
    "enemy_name": RegionOfInterest(
        name="enemy_name",
        tile_start=(0, 0),
        tile_end=(1, 7),
        description="Enemy Pokemon name in battle"
    ),
    "player_name": RegionOfInterest(
        name="player_name",
        tile_start=(7, 10),
        tile_end=(8, 19),
        description="Player Pokemon name in battle"
    ),
}

# Battle menu option positions (2x2 grid)
BATTLE_MENU_OPTIONS = {
    "FIGHT": {"row_range": (13, 14), "col_range": (10, 13)},
    "PKMN":  {"row_range": (13, 14), "col_range": (15, 18)},
    "ITEM":  {"row_range": (15, 16), "col_range": (10, 13)},
    "RUN":   {"row_range": (15, 16), "col_range": (15, 18)},
}


class TileLabeller:
    """Manages tile labels and classifications"""

    def __init__(self):
        self.hash_to_label: Dict[str, str] = {}
        self.hash_to_type: Dict[str, str] = {}

    def label_tile(self, tile_hash: str, label: str, tile_type: str = None):
        """Assign a label to a tile hash"""
        self.hash_to_label[tile_hash] = label
        if tile_type:
            self.hash_to_type[tile_hash] = tile_type

    def get_label(self, tile_hash: str) -> Optional[str]:
        """Get label for a tile hash"""
        return self.hash_to_label.get(tile_hash)

    def get_type(self, tile_hash: str) -> Optional[str]:
        """Get type for a tile hash"""
        return self.hash_to_type.get(tile_hash)

    def label_character_tiles(self, char: str, hashes: List[str]):
        """Label multiple tiles as representing a character"""
        for h in hashes:
            self.label_tile(h, f"char_{char}", "text")

    def label_terrain_type(self, terrain_type: str, hashes: List[str]):
        """
        Label tiles as terrain types.
        Types: 'walkable', 'blocked', 'grass', 'water', 'door', 'stair', 'gate'
        """
        for h in hashes:
            self.label_tile(h, f"terrain_{terrain_type}", "terrain")

    def save_labels(self, filepath: str):
        """Save labels to JSON file"""
        data = {
            "hash_to_label": self.hash_to_label,
            "hash_to_type": self.hash_to_type
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_labels(self, filepath: str):
        """Load labels from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.hash_to_label = data.get("hash_to_label", {})
                self.hash_to_type = data.get("hash_to_type", {})
        except FileNotFoundError:
            print(f"Label file not found: {filepath}")


@dataclass
class PerceptionState:
    """Complete perception state for a frame"""
    frame_number: int
    tiles: List[TileInfo]
    regions: Dict[str, RegionOfInterest]
    screen_hash: str
    label_grid: List[List[Optional[str]]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize label_grid from tiles if not provided"""
        if not self.label_grid:
            self.label_grid = self._build_label_grid()

    def _build_label_grid(self) -> List[List[Optional[str]]]:
        """Build a 2D grid of labels from tiles (18 rows × 20 cols)"""
        grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        for tile in self.tiles:
            if tile.label:
                grid[tile.row][tile.col] = tile.label
        return grid

    def get_tile(self, row: int, col: int) -> Optional[TileInfo]:
        """Get tile at specific grid position"""
        idx = row * GRID_COLS + col
        if 0 <= idx < len(self.tiles):
            return self.tiles[idx]
        return None

    def get_label_at(self, row: int, col: int) -> Optional[str]:
        """Get label at specific grid position"""
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            return self.label_grid[row][col]
        return None

    def get_display_char(self, row: int, col: int) -> str:
        """
        Get a single display character for a tile (for visualization).

        Returns:
            '?' for unlabeled tiles
            The character itself for char_X labels (e.g., 'A', '5', '!')
            '_' for space characters
            ' ' for walkable terrain, first letter for others (B, D, T)
            '>' for cursor
            'U' for other UI elements
            '.' for unknown labels
        """
        label = self.get_label_at(row, col)
        if label is None:
            return "?"
        elif label.startswith("char_"):
            char = label[5:]
            if char == "SPACE":
                return "_"
            return char if len(char) == 1 else char[0]
        elif label.startswith("terrain_"):
            terrain = label[8:]
            if terrain == "walkable":
                return " "
            return terrain[0].upper() if terrain else "T"
        elif label.startswith("ui_"):
            if label == "ui_cursor":
                return ">"
            return "U"
        elif label.startswith("sprite_"):
            if label == "sprite_player":
                return "@"
            elif label == "sprite_npc":
                return "N"
            return "S"
        else:
            return "?"

    def get_display_grid(self) -> List[str]:
        """
        Get the full display grid as a list of strings (one per row).
        Useful for text-based visualization or LLM consumption.
        """
        lines = []
        for row in range(GRID_ROWS):
            line = ""
            for col in range(GRID_COLS):
                line += self.get_display_char(row, col)
            lines.append(line)
        return lines

    def get_tiles_in_region(self, region_name: str) -> List[TileInfo]:
        """Get all tiles within a region of interest"""
        if region_name not in self.regions:
            return []

        region = self.regions[region_name]
        tiles_in_region = []

        for tile in self.tiles:
            if region.contains_tile(tile.row, tile.col):
                tiles_in_region.append(tile)

        return tiles_in_region

    def get_labeled_tiles_by_category(self) -> LabeledTileGroups:
        """
        Group all tiles by their label category.

        Returns:
            LabeledTileGroups with tiles organized by type (text/terrain/ui/sprite/unlabeled)
        """
        text_tiles = []
        terrain_tiles = []
        ui_tiles = []
        sprite_tiles = []
        unlabeled_tiles = []

        for tile in self.tiles:
            if tile.tile_type == "text":
                text_tiles.append(tile)
            elif tile.tile_type == "terrain":
                terrain_tiles.append(tile)
            elif tile.tile_type == "ui":
                ui_tiles.append(tile)
            elif tile.tile_type == "sprite":
                sprite_tiles.append(tile)
            else:
                unlabeled_tiles.append(tile)

        return LabeledTileGroups(
            text=text_tiles,
            terrain=terrain_tiles,
            ui=ui_tiles,
            sprite=sprite_tiles,
            unlabeled=unlabeled_tiles
        )

    def extract_words(self, region_name: Optional[str] = None) -> List[WordInfo]:
        """
        Extract words from char_* labeled tiles.

        Args:
            region_name: Optional region to limit extraction to (e.g., "text_box")

        Returns:
            List of WordInfo objects with extracted words and positions
        """
        # Get tiles (filtered by region if specified)
        if region_name:
            tiles = self.get_tiles_in_region(region_name)
        else:
            tiles = self.tiles

        # Group tiles by row
        rows: Dict[int, List[TileInfo]] = {}
        for tile in tiles:
            if tile.row not in rows:
                rows[tile.row] = []
            rows[tile.row].append(tile)

        # Sort each row by column
        for row in rows.values():
            row.sort(key=lambda t: t.col)

        words = []

        # Process each row
        for row_num in sorted(rows.keys()):
            row_tiles = rows[row_num]

            current_word = ""
            word_start_col = None
            last_col = None

            for tile in row_tiles:
                # Check if this is a character tile
                if tile.label and tile.label.startswith("char_"):
                    char = tile.label[5:]

                    # Handle space as word boundary
                    if char == "SPACE":
                        if current_word:
                            words.append(WordInfo(
                                word=current_word,
                                start_row=row_num,
                                start_col=word_start_col,
                                end_col=last_col
                            ))
                            current_word = ""
                            word_start_col = None
                    else:
                        # Check for gap (non-consecutive columns)
                        if last_col is not None and tile.col > last_col + 1:
                            # Gap detected - end current word
                            if current_word:
                                words.append(WordInfo(
                                    word=current_word,
                                    start_row=row_num,
                                    start_col=word_start_col,
                                    end_col=last_col
                                ))
                                current_word = ""
                                word_start_col = None

                        # Start new word or continue current
                        if word_start_col is None:
                            word_start_col = tile.col
                        current_word += char
                        last_col = tile.col
                else:
                    # Non-character tile - end current word
                    if current_word:
                        words.append(WordInfo(
                            word=current_word,
                            start_row=row_num,
                            start_col=word_start_col,
                            end_col=last_col
                        ))
                        current_word = ""
                        word_start_col = None
                    last_col = tile.col

            # Don't forget the last word in the row
            if current_word:
                words.append(WordInfo(
                    word=current_word,
                    start_row=row_num,
                    start_col=word_start_col,
                    end_col=last_col
                ))

        return words

    def get_menu_selection(self) -> MenuSelection:
        """
        Detect which menu option is currently selected based on cursor position.

        Returns:
            MenuSelection with detected option and cursor position
        """
        # Find cursor tile
        cursor_tile = None
        for tile in self.tiles:
            if tile.label == "ui_cursor":
                cursor_tile = tile
                break

        if cursor_tile is None:
            return MenuSelection(option=None, cursor_position=None, menu_type=None)

        cursor_pos = (cursor_tile.row, cursor_tile.col)

        # Check if cursor is in battle_menu region
        battle_menu = self.regions.get("battle_menu")
        if battle_menu and battle_menu.contains_tile(cursor_tile.row, cursor_tile.col):
            # Map cursor position to menu option
            for option_name, bounds in BATTLE_MENU_OPTIONS.items():
                row_min, row_max = bounds["row_range"]
                col_min, col_max = bounds["col_range"]
                if (row_min <= cursor_tile.row <= row_max and
                    col_min <= cursor_tile.col <= col_max):
                    return MenuSelection(
                        option=option_name,
                        cursor_position=cursor_pos,
                        menu_type="battle_menu"
                    )

            # Cursor in battle_menu but not matching any option
            return MenuSelection(
                option=None,
                cursor_position=cursor_pos,
                menu_type="battle_menu"
            )

        # Cursor found but not in battle_menu
        return MenuSelection(option=None, cursor_position=cursor_pos, menu_type=None)

    def get_battle_hp(self) -> BattleHP:
        """
        Read HP bar status for player and enemy Pokemon.

        Returns:
            BattleHP with player and enemy HP readings
        """
        def read_hp_region(region_name: str) -> HPReading:
            tiles = self.get_tiles_in_region(region_name)
            total = len(tiles)
            hp_tiles = sum(1 for t in tiles if t.label == "ui_hp_bar")
            percentage = hp_tiles / total if total > 0 else None
            return HPReading(
                percentage=percentage,
                hp_bar_tiles=hp_tiles,
                total_tiles=total
            )

        return BattleHP(
            player=read_hp_region("player_hp"),
            enemy=read_hp_region("enemy_hp")
        )

    def get_battle_names(self) -> BattleNames:
        """
        Extract Pokemon names from battle screen regions.

        Returns:
            BattleNames with enemy and player Pokemon names
        """
        enemy_words = self.extract_words("enemy_name")
        player_words = self.extract_words("player_name")

        enemy_name = enemy_words[0].word if enemy_words else None
        player_name = player_words[0].word if player_words else None

        return BattleNames(enemy_name=enemy_name, player_name=player_name)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "frame_number": self.frame_number,
            "screen_hash": self.screen_hash,
            "grid": {
                "rows": GRID_ROWS,
                "cols": GRID_COLS,
                "tile_size": TILE_SIZE
            },
            "label_grid": self.label_grid,
            "display_grid": self.get_display_grid(),
            "tiles": [asdict(t) for t in self.tiles],
            "regions": {name: asdict(roi) for name, roi in self.regions.items()}
        }

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)


class PerceptionV2:
    """
    New perception module using tile-based hashing approach
    """

    def __init__(self, labeller: Optional[TileLabeller] = None):
        self.frame_count = 0
        self.labeller = labeller or TileLabeller()
        self.hash_cache = {}

    def perceive(self, frame: Image.Image) -> PerceptionState:
        """
        Process a frame and return perception state

        Args:
            frame: PIL Image (160x144)

        Returns:
            PerceptionState with all tiles hashed and labelled
        """
        self.frame_count += 1

        # Convert to grayscale numpy array
        arr = np.array(frame.convert("L"))

        # Extract and hash all tiles
        tiles = self._extract_tiles(arr)

        # Calculate screen hash
        all_hashes = [t.hash for t in tiles]
        screen_hash = xxhash.xxh64("".join(all_hashes).encode(), seed=0).hexdigest()[:16]

        # Create perception state
        state = PerceptionState(
            frame_number=self.frame_count,
            tiles=tiles,
            regions=REGIONS_OF_INTEREST.copy(),
            screen_hash=screen_hash
        )

        return state

    def _extract_tiles(self, arr: np.ndarray) -> List[TileInfo]:
        """Extract all tiles from the frame"""
        tiles = []

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                # Extract tile pixels
                y_start = row * TILE_SIZE
                y_end = y_start + TILE_SIZE
                x_start = col * TILE_SIZE
                x_end = x_start + TILE_SIZE

                tile_pixels = arr[y_start:y_end, x_start:x_end]

                # Hash the tile
                tile_hash = self._hash_tile(tile_pixels)

                # Get label and type from labeller
                label = self.labeller.get_label(tile_hash)
                tile_type = self.labeller.get_type(tile_hash)

                # Create tile info
                tile_info = TileInfo(
                    row=row,
                    col=col,
                    hash=tile_hash,
                    label=label,
                    tile_type=tile_type
                )

                tiles.append(tile_info)

        return tiles

    def _hash_tile(self, tile: np.ndarray) -> str:
        """Hash an 8x8 tile using xxHash"""
        tile_bytes = tile.astype(np.uint8).tobytes()

        # Check cache
        if tile_bytes in self.hash_cache:
            return self.hash_cache[tile_bytes]

        # Compute hash
        h = xxhash.xxh64(tile_bytes, seed=0).hexdigest()[:8]

        # Cache it (with size limit)
        if len(self.hash_cache) < 10000:
            self.hash_cache[tile_bytes] = h

        return h

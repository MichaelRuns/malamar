import numpy as np
from PIL import Image, ImageEnhance
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum
import pytesseract  # pip install pytesseract
# Also need: brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)

class GameState(Enum):
    OVERWORLD = "overworld"
    BATTLE = "battle"
    MENU = "menu"
    DIALOG = "dialog"
    UNKNOWN = "unknown"

@dataclass
class BattlePerception:
    """Battle-specific state information"""
    player_hp_percent: float  # 0.0 to 1.0
    enemy_hp_percent: float
    player_pokemon: Optional[str]
    enemy_pokemon: Optional[str]
    available_moves: List[str]
    in_move_selection: bool
    status_conditions: List[str]
    battle_text: str  # OCR'd text from battle messages

@dataclass
class OverworldPerception:
    """Overworld/map state information"""
    player_position: Optional[Tuple[int, int]]  # Tile coordinates
    facing_direction: Optional[str]  # up, down, left, right
    nearby_trainers: List[Tuple[int, int]]
    nearby_items: List[Tuple[int, int]]
    in_grass: bool
    can_interact: bool  # Sign, NPC, door nearby
    
@dataclass  
class TextPerception:
    """OCR'd text from screen"""
    dialog_text: str  # Text from dialog boxes
    menu_items: List[str]  # Menu options visible
    location_name: Optional[str]  # Place names that appear

@dataclass
class PerceptionState:
    """Complete perception of game state"""
    game_state: GameState
    frame: Image.Image
    frame_hash: str  # For detecting stuck states
    
    # State-specific data
    battle: Optional[BattlePerception]
    overworld: Optional[OverworldPerception]
    text: TextPerception  # OCR'd text
    
    # High-level flags
    text_visible: bool
    menu_open: bool
    
    # For LLM context
    description: str  # Natural language summary


class PerceptionModule:
    """
    Multi-layer perception system for Pokemon Blue
    Layer 1: Fast pixel-based heuristics (no ML)
    Layer 2: OCR for text/numbers
    Layer 3: Vision LLM for complex scenes
    """
    
    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        self.previous_frames = []
        self.frame_similarity_threshold = 0.95
        
        # OCR configuration for Game Boy font
        self.tesseract_config = '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?\'-/:é '
        
    def perceive(self, frame: Image.Image) -> PerceptionState:
        """Main perception pipeline"""
        
        # Layer 1: Fast heuristics
        game_state = self._detect_game_state(frame)
        text_visible = self._detect_text_box(frame)
        menu_open = self._detect_menu(frame)
        
        # Layer 1.5: OCR for all text
        text_info = self._extract_all_text(frame, game_state)
        
        # Layer 2: State-specific perception
        battle_info = None
        overworld_info = None
        
        if game_state == GameState.BATTLE:
            battle_info = self._perceive_battle(frame, text_info)
        elif game_state == GameState.OVERWORLD:
            overworld_info = self._perceive_overworld(frame)
        
        # Layer 3: LLM for complex understanding (optional)
        description = ""
        if self.use_llm and self._should_use_llm(game_state):
            description = self._llm_describe_scene(frame, game_state)
        
        return PerceptionState(
            game_state=game_state,
            frame=frame,
            frame_hash=self._hash_frame(frame),
            battle=battle_info,
            overworld=overworld_info,
            text=text_info,
            text_visible=text_visible,
            menu_open=menu_open,
            description=description
        )
    
    # ===== OCR LAYER =====
    
    def _preprocess_for_ocr(self, image: Image.Image, scale_factor: int = 4) -> Image.Image:
        """Preprocess Game Boy screen for better OCR"""
        # Scale up (Game Boy font is tiny)
        scaled = image.resize(
            (image.width * scale_factor, image.height * scale_factor),
            Image.NEAREST  # Preserve pixel art
        )
        
        # Convert to grayscale
        gray = scaled.convert('L')
        
        # Increase contrast (Game Boy has limited colors)
        enhancer = ImageEnhance.Contrast(gray)
        contrasted = enhancer.enhance(2.0)
        
        # Threshold to pure black and white
        threshold = 128
        bw = contrasted.point(lambda x: 255 if x > threshold else 0, mode='1')
        
        return bw
    
    def _extract_all_text(self, frame: Image.Image, game_state: GameState) -> TextPerception:
        """Extract all text from frame using OCR"""
        
        dialog_text = ""
        menu_items = []
        location_name = None
        
        # Extract dialog text if text box visible
        if self._detect_text_box(frame):
            dialog_text = self._ocr_text_box(frame)
        
        # Extract menu text if menu open
        if self._detect_menu(frame):
            menu_items = self._ocr_menu(frame)
        
        # Extract location name (top of screen)
        if game_state == GameState.OVERWORLD:
            location_name = self._ocr_location(frame)
        
        return TextPerception(
            dialog_text=dialog_text,
            menu_items=menu_items,
            location_name=location_name
        )
    
    def _ocr_text_box(self, frame: Image.Image) -> str:
        """OCR the text dialog box at bottom of screen"""
        # Text box is at bottom, approximately pixels 100-144, full width
        text_region = frame.crop((0, 100, 160, 144))
        
        # Preprocess
        processed = self._preprocess_for_ocr(text_region)
        
        # OCR
        try:
            text = pytesseract.image_to_string(
                processed, 
                config=self.tesseract_config
            )
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def _ocr_menu(self, frame: Image.Image) -> List[str]:
        """OCR menu items"""
        # Menus are typically on the right side or full screen
        # Try right side first
        menu_region = frame.crop((80, 0, 160, 144))
        
        processed = self._preprocess_for_ocr(menu_region)
        
        try:
            text = pytesseract.image_to_string(
                processed,
                config=self.tesseract_config
            )
            # Split into lines and filter empty
            items = [line.strip() for line in text.split('\n') if line.strip()]
            return items
        except Exception as e:
            print(f"OCR error: {e}")
            return []
    
    def _ocr_location(self, frame: Image.Image) -> Optional[str]:
        """OCR location name at top of screen"""
        # Location names appear briefly at top when entering new areas
        top_region = frame.crop((0, 0, 160, 20))
        
        processed = self._preprocess_for_ocr(top_region)
        
        try:
            text = pytesseract.image_to_string(
                processed,
                config=self.tesseract_config
            )
            text = text.strip()
            return text if text and len(text) > 2 else None
        except Exception as e:
            return None
    
    def _ocr_battle_moves(self, frame: Image.Image) -> List[str]:
        """OCR the four move names in battle"""
        # Move menu is in bottom right when visible
        move_region = frame.crop((80, 100, 160, 144))
        
        processed = self._preprocess_for_ocr(move_region)
        
        try:
            text = pytesseract.image_to_string(
                processed,
                config=self.tesseract_config
            )
            moves = [line.strip() for line in text.split('\n') if line.strip()]
            return moves[:4]  # Max 4 moves
        except Exception as e:
            return []
    
    def _ocr_pokemon_name(self, frame: Image.Image, enemy: bool = False) -> Optional[str]:
        """OCR Pokemon name from battle screen"""
        if enemy:
            # Enemy name is at top left
            name_region = frame.crop((5, 5, 80, 20))
        else:
            # Player name is at bottom left  
            name_region = frame.crop((5, 100, 80, 115))
        
        processed = self._preprocess_for_ocr(name_region)
        
        try:
            text = pytesseract.image_to_string(
                processed,
                config=self.tesseract_config
            )
            name = text.strip()
            return name if name and len(name) > 1 else None
        except Exception as e:
            return None
    
    # ===== LAYER 1: FAST HEURISTICS =====
    
    def _detect_game_state(self, frame: Image.Image) -> GameState:
        """Detect game state using pixel patterns"""
        arr = np.array(frame)
        
        # Battle detection: Black bars at top/bottom, HP bars visible
        if self._has_battle_ui(arr):
            return GameState.BATTLE
        
        # Menu detection: Specific color patterns
        if self._has_menu_ui(arr):
            return GameState.MENU
        
        # Text box detection
        if self._detect_text_box(frame):
            return GameState.DIALOG
        
        # Default to overworld
        return GameState.OVERWORLD
    
    def _has_battle_ui(self, arr: np.ndarray) -> bool:
        """Check for battle UI elements"""
        # Battle screen has distinctive layout:
        # - Black/dark bars at top and bottom
        # - HP bars in specific positions
        
        height, width = arr.shape[:2]
        
        # Check top section (enemy pokemon area) - typically darker
        top_section = arr[0:30, :].mean()
        
        # Check for HP bar region (greenish pixels in specific area)
        # Enemy HP bar around y=40-50, x=100-150
        enemy_hp_region = arr[40:50, 100:150]
        has_green_bar = self._has_color_in_region(enemy_hp_region, 
                                                   target_color=[0, 200, 0], 
                                                   tolerance=50)
        
        # Player HP bar around y=110-120, x=50-100
        player_hp_region = arr[110:120, 50:100]
        has_player_bar = self._has_color_in_region(player_hp_region,
                                                     target_color=[0, 200, 0],
                                                     tolerance=50)
        
        return has_green_bar or has_player_bar
    
    def _has_menu_ui(self, arr: np.ndarray) -> bool:
        """Check for menu interface"""
        # Menus have white background with black borders
        # Typically right side or full screen
        
        # Check right section for menu (Pokemon has right-side menus)
        right_section = arr[:, 120:]
        white_ratio = np.mean(right_section > 200)
        
        return white_ratio > 0.5
    
    def _detect_text_box(self, frame: Image.Image) -> bool:
        """Detect if text dialog box is visible"""
        arr = np.array(frame)
        
        # Text boxes in Pokemon are at the bottom
        # White background, black text, specific dimensions
        bottom_section = arr[100:, :]
        
        # Check for white rectangular region
        white_pixels = np.mean(bottom_section > 200)
        
        return white_pixels > 0.3
    
    def _detect_menu(self, frame: Image.Image) -> bool:
        """Detect if any menu is open"""
        return self._has_menu_ui(np.array(frame))
    
    def _has_color_in_region(self, region: np.ndarray, 
                            target_color: List[int], 
                            tolerance: int) -> bool:
        """Check if target color exists in region"""
        if region.size == 0:
            return False
        
        target = np.array(target_color)
        distances = np.abs(region - target).sum(axis=-1)
        return np.any(distances < tolerance)
    
    # ===== LAYER 2: STATE-SPECIFIC PERCEPTION =====
    
    def _perceive_battle(self, frame: Image.Image, text_info: TextPerception) -> BattlePerception:
        """Extract battle-specific information"""
        arr = np.array(frame)
        
        # Extract HP bars using color detection
        player_hp = self._extract_hp_bar(arr, player=True)
        enemy_hp = self._extract_hp_bar(arr, player=False)
        
        # Detect move selection menu
        in_move_selection = self._detect_move_menu(arr)
        
        # OCR Pokemon names
        player_pokemon = self._ocr_pokemon_name(frame, enemy=False)
        enemy_pokemon = self._ocr_pokemon_name(frame, enemy=True)
        
        # OCR moves if menu is visible
        available_moves = []
        if in_move_selection:
            available_moves = self._ocr_battle_moves(frame)
        
        # Get battle text
        battle_text = text_info.dialog_text
        
        return BattlePerception(
            player_hp_percent=player_hp,
            enemy_hp_percent=enemy_hp,
            player_pokemon=player_pokemon,
            enemy_pokemon=enemy_pokemon,
            available_moves=available_moves,
            in_move_selection=in_move_selection,
            status_conditions=[],  # Could parse from text
            battle_text=battle_text
        )
    
    def _extract_hp_bar(self, arr: np.ndarray, player: bool) -> float:
        """Extract HP percentage from HP bar"""
        if player:
            # Player HP bar location (approximate)
            hp_region = arr[115:118, 54:102]
        else:
            # Enemy HP bar location
            hp_region = arr[45:48, 102:134]
        
        if hp_region.size == 0:
            return 1.0
        
        # Green pixels indicate remaining HP
        green_mask = (hp_region[:, :, 1] > 150) & \
                     (hp_region[:, :, 0] < 100) & \
                     (hp_region[:, :, 2] < 100)
        
        total_pixels = hp_region.shape[0] * hp_region.shape[1]
        green_pixels = np.sum(green_mask)
        
        return green_pixels / total_pixels if total_pixels > 0 else 1.0
    
    def _detect_move_menu(self, arr: np.ndarray) -> bool:
        """Detect if move selection menu is visible"""
        # Move menu appears in bottom right
        menu_region = arr[100:, 80:]
        white_ratio = np.mean(menu_region > 200)
        return white_ratio > 0.4
    
    def _perceive_overworld(self, frame: Image.Image) -> OverworldPerception:
        """Extract overworld/map information"""
        arr = np.array(frame)
        
        # Detect player position (center of screen usually)
        player_pos = self._detect_player_position(arr)
        
        # Detect facing direction from sprite
        facing = self._detect_player_direction(arr)
        
        # Detect grass tiles (green pattern)
        in_grass = self._detect_grass(arr)
        
        return OverworldPerception(
            player_position=player_pos,
            facing_direction=facing,
            nearby_trainers=[],  # Requires more sophisticated detection
            nearby_items=[],
            in_grass=in_grass,
            can_interact=False   # Could check for nearby NPCs/signs
        )
    
    def _detect_player_position(self, arr: np.ndarray) -> Tuple[int, int]:
        """Estimate player tile position"""
        # Player is typically centered on screen
        # GB screen is 160x144, tiles are 8x8
        center_tile = (160 // 16, 144 // 16)
        return center_tile
    
    def _detect_player_direction(self, arr: np.ndarray) -> str:
        """Detect which way player sprite is facing"""
        # This is tricky - would need sprite recognition
        # For now, return unknown
        return "unknown"
    
    def _detect_grass(self, arr: np.ndarray) -> bool:
        """Detect if player is in tall grass"""
        # Grass has distinctive pattern - green with darker spots
        center_region = arr[60:80, 70:90]
        
        # Look for green colors
        green_mask = (center_region[:, :, 1] > 100) & \
                     (center_region[:, :, 0] < 100)
        
        green_ratio = np.sum(green_mask) / green_mask.size
        return green_ratio > 0.3
    
    # ===== LAYER 3: LLM PERCEPTION =====
    
    def _should_use_llm(self, game_state: GameState) -> bool:
        """Decide if LLM analysis is needed"""
        # Use LLM sparingly to save compute
        # Only for complex situations or every N frames
        return game_state in [GameState.UNKNOWN, GameState.DIALOG]
    
    def _llm_describe_scene(self, frame: Image.Image, 
                           game_state: GameState) -> str:
        """Use vision LLM to describe scene"""
        # This would call Ollama with llama3.2-vision
        # For now, return placeholder
        return f"Current state: {game_state.value}"
    
    # ===== UTILITIES =====
    
    def _hash_frame(self, frame: Image.Image) -> str:
        """Create hash of frame for similarity detection"""
        # Downsample and hash for stuck detection
        small = frame.resize((16, 16))
        return str(hash(small.tobytes()))
    
    def is_stuck(self, current_hash: str, window: int = 100) -> bool:
        """Detect if agent is stuck (same frames repeating)"""
        self.previous_frames.append(current_hash)
        if len(self.previous_frames) > window:
            self.previous_frames.pop(0)
        
        if len(self.previous_frames) < 50:
            return False
        
        # If last 50 frames are very similar, probably stuck
        unique_frames = len(set(self.previous_frames[-50:]))
        return unique_frames < 5


# ===== USAGE EXAMPLE =====

if __name__ == "__main__":
    # Example usage
    perception = PerceptionModule(use_llm=False)
    
    # Dummy frame
    dummy_frame = Image.new('RGB', (160, 144), color='white')
    
    state = perception.perceive(dummy_frame)
    
    print(f"Game State: {state.game_state}")
    print(f"Text Visible: {state.text_visible}")
    print(f"Menu Open: {state.menu_open}")
    print(f"Dialog Text: {state.text.dialog_text}")
    print(f"Menu Items: {state.text.menu_items}")
    
    if state.battle:
        print(f"\n=== BATTLE INFO ===")
        print(f"Player: {state.battle.player_pokemon} (HP: {state.battle.player_hp_percent:.2%})")
        print(f"Enemy: {state.battle.enemy_pokemon} (HP: {state.battle.enemy_hp_percent:.2%})")
        print(f"Moves: {state.battle.available_moves}")
        print(f"Battle Text: {state.battle.battle_text}")
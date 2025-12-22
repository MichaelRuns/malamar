"""
Pokemon Black/White 2 Perception Layer
Extracts game state from melonDS screenshots
"""

import cv2
import numpy as np
import mss
import pytesseract
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

class GameState(Enum):
    """High-level game state"""
    BATTLE = "battle"
    OVERWORLD = "overworld"
    MENU = "menu"
    DIALOGUE = "dialogue"
    UNKNOWN = "unknown"

@dataclass
class BattleState:
    """Parsed battle information"""
    player_pokemon: str
    player_hp: int
    player_hp_max: int
    player_level: int
    
    opponent_pokemon: str
    opponent_hp_percent: float  # Opponent HP not shown as numbers
    opponent_level: int
    
    available_moves: List[str]
    move_pp: List[int]
    
    can_switch: bool
    party_size: int

@dataclass
class OverworldState:
    """Overworld navigation state"""
    location: str
    party_status: List[dict]  # [{name, hp, status}, ...]
    money: int

@dataclass
class PerceptionOutput:
    """Complete perception output"""
    game_state: GameState
    battle_state: Optional[BattleState]
    overworld_state: Optional[OverworldState]
    raw_frame: np.ndarray

class PerceptionModule:
    """
    Main perception module for Pokemon game state extraction
    """
    
    def __init__(self, window_title="melonDS"):
        self.window_title = window_title
        self.sct = mss.mss()
        
        # Screen regions (adjust based on your melonDS window)
        # These are relative to the DS screen layout
        self.regions = {
            'full': None,  # Will be set dynamically
            'top_screen': (0, 0, 256, 192),
            'bottom_screen': (0, 192, 256, 192),
            'player_hp': (130, 100, 100, 20),  # Approximate
            'opponent_hp': (10, 40, 100, 20),
            'move_menu': (0, 200, 256, 92),
        }
        
        # Template matching cache
        self.templates = {}
        self._load_templates()
        
    def _load_templates(self):
        """Load UI element templates for matching"""
        # TODO: Add template images for:
        # - Battle UI indicators
        # - Menu cursors
        # - HP bars
        # - Dialog boxes
        pass
    
    def get_window_region(self) -> dict:
        """
        Find melonDS window and return its screen region
        """
        # For now, capture entire screen
        # TODO: Implement window detection
        monitor = self.sct.monitors[1]
        return {
            'top': monitor['top'],
            'left': monitor['left'],
            'width': monitor['width'],
            'height': monitor['height']
        }
    
    def capture_frame(self) -> np.ndarray:
        """Capture current frame from melonDS"""
        region = self.get_window_region()
        screenshot = self.sct.grab(region)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    def detect_game_state(self, frame: np.ndarray) -> GameState:
        """
        Detect high-level game state using visual cues
        """
        # Check for battle UI elements
        if self._is_battle_screen(frame):
            return GameState.BATTLE
        
        # Check for dialogue box
        if self._is_dialogue_screen(frame):
            return GameState.DIALOGUE
        
        # Check for menu
        if self._is_menu_screen(frame):
            return GameState.MENU
        
        # Default to overworld
        return GameState.OVERWORLD
    
    def _is_battle_screen(self, frame: np.ndarray) -> bool:
        """
        Detect if we're in battle by looking for HP bars
        Uses color-based detection for the characteristic green HP bars
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green color range for HP bars
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = np.sum(mask > 0)
        
        # If significant green pixels, likely in battle
        return green_pixels > 500
    
    def _is_dialogue_screen(self, frame: np.ndarray) -> bool:
        """Detect dialogue box"""
        # Look for white text box at bottom
        bottom = frame[-100:, :]
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(gray > 200)
        return white_pixels > 5000
    
    def _is_menu_screen(self, frame: np.ndarray) -> bool:
        """Detect menu screen"""
        # Look for menu UI patterns
        # TODO: Implement template matching
        return False
    
    def parse_battle_state(self, frame: np.ndarray) -> BattleState:
        """
        Extract detailed battle information
        """
        # Extract text regions
        player_pokemon = self._ocr_region(frame, (130, 90, 80, 15))
        opponent_pokemon = self._ocr_region(frame, (10, 30, 80, 15))
        
        # Parse HP values
        player_hp_text = self._ocr_region(frame, (150, 105, 50, 15))
        player_hp, player_hp_max = self._parse_hp(player_hp_text)
        
        # Opponent HP is visual only (bar)
        opponent_hp_percent = self._get_hp_bar_percent(frame, (20, 50, 80, 4))
        
        # Parse levels
        player_level = self._parse_level(frame, (180, 95, 20, 10))
        opponent_level = self._parse_level(frame, (70, 35, 20, 10))
        
        # Parse moves from bottom screen
        moves, pp = self._parse_moves(frame)
        
        return BattleState(
            player_pokemon=player_pokemon,
            player_hp=player_hp,
            player_hp_max=player_hp_max,
            player_level=player_level,
            opponent_pokemon=opponent_pokemon,
            opponent_hp_percent=opponent_hp_percent,
            opponent_level=opponent_level,
            available_moves=moves,
            move_pp=pp,
            can_switch=True,  # TODO: Detect
            party_size=6  # TODO: Detect actual party size
        )
    
    def _ocr_region(self, frame: np.ndarray, region: Tuple[int, int, int, int], debug=False) -> str:
        """
        Perform OCR on a specific region with enhanced preprocessing
        region: (x, y, width, height)
        debug: if True, show intermediate processing steps
        """
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return ""
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Try multiple preprocessing approaches
        preprocessed_images = []
        
        # Method 1: Simple binary threshold
        _, binary1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        preprocessed_images.append(("binary_150", binary1))
        
        # Method 2: Otsu's thresholding (automatic)
        _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("otsu", binary2))
        
        # Method 3: Adaptive threshold (good for varying lighting)
        binary3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(("adaptive", binary3))
        
        # Method 4: Invert for white text on dark background
        _, binary4 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        preprocessed_images.append(("inverted", binary4))
        
        # Method 5: High contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, binary5 = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
        preprocessed_images.append(("clahe", binary5))
        
        # Upscale for better OCR (larger text = better recognition)
        scale = 4
        results = []
        
        for method_name, binary in preprocessed_images:
            # Upscale
            upscaled = cv2.resize(binary, None, fx=scale, fy=scale, 
                                 interpolation=cv2.INTER_CUBIC)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(upscaled, None, 10, 7, 21)
            
            # OCR with different configs
            configs = [
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/',  # Pokemon names, levels, HP
                '--psm 8',  # Treat as single word
                '--psm 7',  # Single line
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(denoised, config=config).strip()
                    if text:
                        results.append(text)
                        if debug:
                            print(f"  {method_name} + {config}: '{text}'")
                except:
                    pass
        
        if debug and results:
            # Show the preprocessed images
            for method_name, binary in preprocessed_images[:3]:  # Show first 3
                cv2.imshow(f'OCR Debug - {method_name}', binary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Return most common result (voting)
        if results:
            from collections import Counter
            # Clean results
            cleaned = [r.upper().replace(' ', '') for r in results if len(r) > 0]
            if cleaned:
                most_common = Counter(cleaned).most_common(1)[0][0]
                return most_common
        
        return ""
    
    def _parse_hp(self, hp_text: str) -> Tuple[int, int]:
        """Parse HP text like '45/67' """
        try:
            parts = hp_text.split('/')
            current = int(parts[0].strip())
            max_hp = int(parts[1].strip())
            return current, max_hp
        except:
            return 0, 0
    
    def _get_hp_bar_percent(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> float:
        """
        Calculate HP percentage from visual HP bar
        Uses green color detection
        """
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = np.sum(mask > 0)
        total_pixels = w * h
        
        return green_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def _parse_level(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> int:
        """Parse level number"""
        text = self._ocr_region(frame, region)
        try:
            # Look for "Lv.XX" or just "XX"
            if 'Lv' in text or 'LV' in text:
                return int(''.join(filter(str.isdigit, text)))
            return int(text)
        except:
            return 0
    
    def _parse_moves(self, frame: np.ndarray) -> Tuple[List[str], List[int]]:
        """Parse available moves and PP from battle menu"""
        moves = []
        pp_values = []
        
        # Bottom screen move regions (approximate)
        move_regions = [
            (10, 210, 100, 15),
            (10, 235, 100, 15),
            (130, 210, 100, 15),
            (130, 235, 100, 15)
        ]
        
        for region in move_regions:
            move_text = self._ocr_region(frame, region)
            if move_text:
                moves.append(move_text)
                # TODO: Parse PP values
                pp_values.append(10)  # Placeholder
        
        return moves, pp_values
    
    def perceive(self) -> PerceptionOutput:
        """
        Main perception function - captures and analyzes current frame
        """
        frame = self.capture_frame()
        game_state = self.detect_game_state(frame)
        
        battle_state = None
        overworld_state = None
        
        if game_state == GameState.BATTLE:
            battle_state = self.parse_battle_state(frame)
        elif game_state == GameState.OVERWORLD:
            # TODO: Implement overworld state parsing
            pass
        
        return PerceptionOutput(
            game_state=game_state,
            battle_state=battle_state,
            overworld_state=overworld_state,
            raw_frame=frame
        )

# Example usage and testing
if __name__ == "__main__":
    perception = PerceptionModule()
    
    print("Starting perception test...")
    print("Switch to melonDS window in 3 seconds...")
    import time
    time.sleep(3)
    
    # Capture and analyze 10 frames
    for i in range(10):
        output = perception.perceive()
        print(f"\nFrame {i+1}:")
        print(f"Game State: {output.game_state.value}")
        
        if output.battle_state:
            bs = output.battle_state
            print(f"Battle Info:")
            print(f"  Player: {bs.player_pokemon} Lv.{bs.player_level} ({bs.player_hp}/{bs.player_hp_max} HP)")
            print(f"  Opponent: {bs.opponent_pokemon} Lv.{bs.opponent_level} ({bs.opponent_hp_percent*100:.1f}% HP)")
            print(f"  Moves: {bs.available_moves}")
        
        time.sleep(1)
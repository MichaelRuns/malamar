import pygame
from pyboy import PyBoy
import time
import numpy as np
import subprocess
from typing import List, Optional
from perception_v2 import PerceptionV2, TileLabeller
from visualizer_v2 import VisualizerV2
import cv2
import os

"""
Pokemon Blue Game Runner with Perception V2

Screen Layout:
- 160x144 pixels (Game Boy screen)
- 20x18 grid of 8x8 pixel tiles (360 total)
- Column labels: A-T (20 columns)
- Row labels: 0-17 (18 rows)

Controls:
- M: Toggle manual mode
- V: Toggle perception visualizer
- G: Toggle grid overlay (when visualizer is on)
- R: Toggle regions overlay (when visualizer is on)
- T: Toggle label grid panel (when visualizer is on)
- I: Toggle state info panel (when visualizer is on)
- S: Save game state
- L: Load game state
- P: Take screenshot and open tile labeler (manual mode only)
- Arrow keys: Movement
- Z: A button
- X: B button
- Enter: Start
- Right Shift: Select
"""
# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((320, 290,))
pygame.display.set_caption("Pokemon Blue - Raw Frame")
clock = pygame.time.Clock()

# Initialize PyBoy in headless mode
pyboy = PyBoy('assets/game_files/pokemon_blue.gb', window="null")
pyboy.set_emulation_speed(0)


class GameOrchestrator:
    """Simple orchestrator that decides what actions to take based on frames"""

    def __init__(self, enable_viz=True, labels_file="tile_labels.json"):
        self.frame_count = 0
        self.action_queue = []
        self.current_state = None  # Track latest perception state for screenshots

        # Initialize V2 perception with tile labelling
        self.labeller = TileLabeller()
        if os.path.exists(labels_file):
            self.labeller.load_labels(labels_file)
            print(f"Loaded {len(self.labeller.hash_to_label)} tile labels from {labels_file}")
        else:
            print(f"No labels file found at {labels_file}, starting fresh")

        self.perception = PerceptionV2(labeller=self.labeller)
        self.visualizer = VisualizerV2() if enable_viz else None
    
    def get_actions(self, frame) -> List[str]:
        """
        Given a frame (PIL Image), decide what actions to take.
        Returns a list of button commands: ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']

        Replace this logic with your AI/LLM calls.
        """
        self.frame_count += 1
        state = self.perception.perceive(frame)  # Returns PerceptionState
        self.current_state = state  # Store for screenshot capture

        # Visualize perception state with V2
        if self.visualizer:
            frame_array = np.array(frame.convert("L"))
            self.visualizer.visualize(frame_array, state)

            # Handle visualizer controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('g'):
                self.visualizer.toggle_grid()
            elif key == ord('r'):
                self.visualizer.toggle_regions()
            elif key == ord('t'):
                self.visualizer.toggle_label_grid()
            elif key == ord('i'):
                self.visualizer.toggle_state_panel()

        # Get JSON representation for LLM
        state_json = state.to_json()

        # Extract text from text box region
        text_tiles = state.get_tiles_in_region("text_box")
        dialogue = self._read_text_from_tiles(text_tiles)

        # Optional: Log state for debugging (every 60 frames = 1 second)
        if self.frame_count % 60 == 0:
            print(f"\n=== Frame {self.frame_count} ===")
            print(f"Screen Hash: {state.screen_hash}")
            print(f"Regions: {list(state.regions.keys())}")
            if dialogue:
                print(f"Text detected: '{dialogue}'")

        # TODO: Send state_json to LLM for decision making
        # actions = self.llm.decide(state_json, dialogue)

        # For now, keep existing placeholder logic
        if self.frame_count % 120 == 0:
            return ['a']  # Press A every 2 seconds
        elif self.frame_count % 60 == 0:
            return ['down']  # Press down every second

        return []  # No action

    def _read_text_from_tiles(self, tiles) -> str:
        """
        Reconstruct text from labeled character tiles

        Args:
            tiles: List of TileInfo objects from a region

        Returns:
            Reconstructed text string
        """
        chars = []
        # Sort tiles by position (top-to-bottom, left-to-right)
        sorted_tiles = sorted(tiles, key=lambda t: (t.row, t.col))

        for tile in sorted_tiles:
            if tile.label and tile.label.startswith("char_"):
                # Extract character from label (e.g., "char_A" -> "A")
                char = tile.label[5:]  # Skip "char_" prefix

                # Handle space character specially
                if char == "SPACE":
                    chars.append(" ")
                else:
                    chars.append(char)

        return "".join(chars)
    
    def should_continue(self) -> bool:
        """Decide if we should keep running. Override with your logic."""
        return True


def execute_action(action: str, duration_frames: int = 5):
    """Execute a button action for a specified number of frames"""
    pyboy.button_press(action)
    for _ in range(duration_frames):
        pyboy.tick()
        render_frame()
    pyboy.button_release(action)


def render_frame():
    """Render current game frame to Pygame window"""
    current_frame = pyboy.screen.image
    mode = current_frame.mode
    size = current_frame.size
    data = current_frame.tobytes()
    
    img_surface = pygame.image.fromstring(data, size, mode)
    scaled = pygame.transform.scale(img_surface, (320, 288))
    
    screen.blit(scaled, (0, 0))
    pygame.display.flip()

def save_game():
        # todo: support file choice
        with open("assets/game_files/game_save.state", "wb") as f:
            pyboy.save_state(f)
            print("game saved!!")

def load_game():
    # todo: support file choice
    with open("assets/game_files/game_save.state", "rb") as f:
        pyboy.load_state(f)
        print('game loaded from file. Enjoy!')


def save_screenshot(frame, state, directory="assets/screenshots"):
    """
    Save current frame as grayscale PNG for tile labeling
    Includes deduplication to avoid saving identical frames

    Args:
        frame: PIL Image from PyBoy (160x144 RGB)
        state: PerceptionState object with screen_hash
        directory: Directory to save screenshots

    Returns:
        filepath if saved, None if duplicate or error
    """
    if state is None:
        print("⚠️  Cannot save screenshot: game state not yet initialized")
        return None

    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    # Check for duplicates using screen hash
    last_screenshot_hash_file = os.path.join(directory, ".last_hash")
    current_hash = state.screen_hash

    if os.path.exists(last_screenshot_hash_file):
        with open(last_screenshot_hash_file, 'r') as f:
            last_hash = f.read().strip()
            if last_hash == current_hash:
                print("⚠️  Duplicate frame, not saving")
                return None

    # Convert to grayscale and save
    frame_gray = frame.convert("L")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"frame_{timestamp}.png"
    filepath = os.path.join(directory, filename)
    frame_gray.save(filepath)

    # Update last hash
    with open(last_screenshot_hash_file, 'w') as f:
        f.write(current_hash)

    print(f"📸 Screenshot saved: {filepath}")
    return filepath


# Initialize orchestrator
orchestrator = GameOrchestrator(enable_viz=True)

running = True
manual_mode = True  # Toggle with 'M' key
viz_enabled = True  # Toggle with 'V' key

while running:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                manual_mode = not manual_mode
                print(f"Manual mode: {manual_mode}")
            elif event.key == pygame.K_v:
                viz_enabled = not viz_enabled
                if viz_enabled:
                    if not orchestrator.visualizer:
                        orchestrator.visualizer = VisualizerV2()
                    print("Perception visualizer V2: ENABLED")
                else:
                    if orchestrator.visualizer:
                        orchestrator.visualizer.close()
                        orchestrator.visualizer = None
                    print("Perception visualizer V2: DISABLED")
            elif manual_mode:
                # Manual control when enabled
                if event.key == pygame.K_z: pyboy.button_press('a')
                elif event.key == pygame.K_x: pyboy.button_press('b')
                elif event.key == pygame.K_RETURN: pyboy.button_press('start')
                elif event.key == pygame.K_RSHIFT: pyboy.button_press('select')
                elif event.key == pygame.K_UP: pyboy.button_press('up')
                elif event.key == pygame.K_DOWN: pyboy.button_press('down')
                elif event.key == pygame.K_LEFT: pyboy.button_press('left')
                elif event.key == pygame.K_RIGHT: pyboy.button_press('right')
                elif event.key == pygame.K_s: save_game()
                elif event.key == pygame.K_l: load_game()
                elif event.key == pygame.K_p:
                    filepath = save_screenshot(current_frame, orchestrator.current_state)
                    if filepath:
                        # Launch tile labeler in background
                        print(f"🏷️  Opening tile labeler for {filepath}")
                        subprocess.Popen([
                            "python", "tools/tile_labeller_interactive.py",
                            "--image", filepath, "--labels", "tile_labels.json"
                        ], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        elif event.type == pygame.KEYUP and manual_mode:
            if event.key == pygame.K_z: pyboy.button_release('a')
            elif event.key == pygame.K_x: pyboy.button_release('b')
            elif event.key == pygame.K_RETURN: pyboy.button_release('start')
            elif event.key == pygame.K_RSHIFT: pyboy.button_release('select')
            elif event.key == pygame.K_UP: pyboy.button_release('up')
            elif event.key == pygame.K_DOWN: pyboy.button_release('down')
            elif event.key == pygame.K_LEFT: pyboy.button_release('left')
            elif event.key == pygame.K_RIGHT: pyboy.button_release('right')

    # Update emulator
    pyboy.tick()
    
    # Get current frame
    current_frame = pyboy.screen.image
    
    # AI Orchestrator decides actions (unless in manual mode)
    actions = orchestrator.get_actions(current_frame)
    if not manual_mode:
        for action in actions:
            execute_action(action, duration_frames=5)
    
    # Render
    render_frame()
    clock.tick(60)
    
    # Check if orchestrator wants to continue
    if not orchestrator.should_continue():
        running = False

# Cleanup
if orchestrator.visualizer:
    orchestrator.visualizer.close()
pyboy.stop()
pygame.quit()
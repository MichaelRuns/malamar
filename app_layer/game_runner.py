import pygame
from pyboy import PyBoy
import time
import numpy as np
from typing import List, Optional
from perception_boy import PerceptionModule
from perception_visualizer import PerceptionVisualizer

"""
note: gba tiles are 8px by 8px
there are 20tiles in each direction
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

    def __init__(self, enable_viz=True):
        self.frame_count = 0
        self.action_queue = []
        self.perception = PerceptionModule(False)
        self.visualizer = PerceptionVisualizer() if enable_viz else None
    
    def get_actions(self, frame) -> List[str]:
        """
        Given a frame (PIL Image), decide what actions to take.
        Returns a list of button commands: ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']

        Replace this logic with your AI/LLM calls.
        """
        self.frame_count += 1
        state = self.perception.perceive(frame)  # Now returns GameState object

        # Visualize perception state
        if self.visualizer:
            frame_array = np.array(frame.convert("L"))
            self.visualizer.visualize(frame_array, state)

        # Get JSON representation for LLM (when ready)
        state_json = state.to_json()

        # Optional: Log state for debugging (every 60 frames = 1 second)
        if self.frame_count % 60 == 0:
            print(f"\n=== Frame {self.frame_count} ===")
            print(f"Game State: {state.meta['game_state']}")
            print(f"Changes: {state.changes.get('num_changes', 0)} cells modified")
            print(f"Screen Hash: {state.meta['screen_hash']}")
            if state.regions:
                print(f"Regions: {list(state.regions.keys())}")
            if state.sprites:
                print(f"Sprites: {len(state.sprites)} detected")

        # TODO: Send state_json to LLM for decision making
        # actions = self.llm.decide(state_json)

        # For now, keep existing placeholder logic
        if self.frame_count % 120 == 0:
            return ['a']  # Press A every 2 seconds
        elif self.frame_count % 60 == 0:
            return ['down']  # Press down every second

        return []  # No action
    
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
                        orchestrator.visualizer = PerceptionVisualizer()
                    print("Perception visualizer: ENABLED")
                else:
                    if orchestrator.visualizer:
                        orchestrator.visualizer.close()
                        orchestrator.visualizer = None
                    print("Perception visualizer: DISABLED")
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
    if not manual_mode:
        actions = orchestrator.get_actions(current_frame)
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
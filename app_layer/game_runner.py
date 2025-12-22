import pygame
from pyboy import PyBoy
import time
from typing import List, Optional

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((320, 288))
pygame.display.set_caption("Pokemon Blue - AI Orchestrator")
clock = pygame.time.Clock()

# Initialize PyBoy in headless mode
pyboy = PyBoy('pokemon_blue.gb', window="null")
pyboy.set_emulation_speed(0)


class GameOrchestrator:
    """Simple orchestrator that decides what actions to take based on frames"""
    
    def __init__(self):
        self.frame_count = 0
        self.action_queue = []
    
    def get_actions(self, frame) -> List[str]:
        """
        Given a frame (PIL Image), decide what actions to take.
        Returns a list of button commands: ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']
        
        Replace this logic with your AI/LLM calls.
        """
        self.frame_count += 1
        
        # Example: Simple pattern - press buttons every N frames
        # Replace this with actual AI logic
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


# Initialize orchestrator
orchestrator = GameOrchestrator()

running = True
manual_mode = False  # Toggle with 'M' key

while running:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                manual_mode = not manual_mode
                print(f"Manual mode: {manual_mode}")
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

pyboy.stop()
pygame.quit()
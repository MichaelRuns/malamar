import pygame
from pyboy import PyBoy
import time
import json
import numpy as np
import subprocess
from typing import List, Optional
from perception_v2 import PerceptionV2, TileLabeller
from visualizer_v2 import VisualizerV2
import cv2
import os

"""
Pokemon Blue Game Runner with Perception V2 and AI Agent

Screen Layout:
- 160x144 pixels (Game Boy screen)
- 20x18 grid of 8x8 pixel tiles (360 total)
- Column labels: A-T (20 columns)
- Row labels: 0-17 (18 rows)

Controls:
- M: Toggle manual mode
- A: Toggle AI agent mode (requires Ollama running)
- V: Toggle perception visualizer
- G: Toggle grid overlay (when visualizer is on)
- R: Toggle regions overlay (when visualizer is on)
- T: Toggle label grid panel (when visualizer is on)
- I: Toggle state info panel (when visualizer is on)
- O: Toggle agent output panel (when visualizer is on)
- S: Save game state
- L: Load game state
- P: Take screenshot and open tile labeler (manual mode only)
- Arrow keys: Movement
- Z: A button
- X: B button
- Enter: Start
- Right Shift: Select
"""


def load_settings(settings_file: str = "settings.json") -> dict:
    """Load settings from JSON file."""
    default_settings = {
        "agent": {
            "enabled": False,
            "model": "llama3.2:3b",
            "ollama_url": "http://localhost:11434",
            "timeout": 30,
            "timing": {
                "decision_interval_seconds": 7,
                "button_delay_seconds": 1.0,
                "action_hold_frames": 5
            },
            "roles": {
                "planner": {"enabled": True, "replan_interval_seconds": 30},
                "executor": {"enabled": True, "max_actions_per_decision": 15}
            },
            "context": {"max_history_exchanges": 6}
        },
        "visualization": {
            "enabled": True,
            "show_agent_panel": True
        },
        "game": {
            "emulation_speed": 0,
            "target_fps": 60
        }
    }

    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                loaded = json.load(f)
                # Merge with defaults
                for key in default_settings:
                    if key in loaded:
                        if isinstance(default_settings[key], dict):
                            default_settings[key].update(loaded[key])
                        else:
                            default_settings[key] = loaded[key]
                print(f"Loaded settings from {settings_file}")
                return default_settings
        except json.JSONDecodeError as e:
            print(f"Error loading settings: {e}, using defaults")

    return default_settings


# Load settings
SETTINGS = load_settings()

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((320, 290,))
pygame.display.set_caption("Pokemon Blue - Raw Frame")
clock = pygame.time.Clock()

# Initialize PyBoy in headless mode
pyboy = PyBoy('assets/game_files/pokemon_blue.gb', window="null")
pyboy.set_emulation_speed(SETTINGS["game"]["emulation_speed"])


class GameOrchestrator:
    """Orchestrator that manages game state, perception, and AI agent"""

    def __init__(self, settings: dict, labels_file: str = "tile_labels.json"):
        self.settings = settings
        self.frame_count = 0
        self.current_state = None
        self.use_agent = settings["agent"]["enabled"]

        # Action execution state
        self.pending_actions = []
        self.last_action_time = 0
        self.button_delay = settings["agent"]["timing"]["button_delay_seconds"]
        self.action_hold_frames = settings["agent"]["timing"]["action_hold_frames"]

        # Initialize V2 perception with tile labelling
        self.labeller = TileLabeller()
        if os.path.exists(labels_file):
            self.labeller.load_labels(labels_file)
            print(f"Loaded {len(self.labeller.hash_to_label)} tile labels from {labels_file}")
        else:
            print(f"No labels file found at {labels_file}, starting fresh")

        self.perception = PerceptionV2(labeller=self.labeller)

        # Initialize visualizer
        viz_settings = settings.get("visualization", {})
        self.visualizer = VisualizerV2() if viz_settings.get("enabled", True) else None

        # Initialize AI agent (lazy - only connects when enabled)
        self.agent = None
        self._agent_initialized = False

    def process_frame(self, frame) -> Optional[str]:
        """
        Process a frame through perception and agent.
        Returns the next button action to execute, or None.
        """
        self.frame_count += 1
        state = self.perception.perceive(frame)
        self.current_state = state

        # Visualize perception state
        if self.visualizer:
            frame_array = np.array(frame.convert("L"))

            # Get agent status for visualization
            agent_status = None
            if self.agent:
                agent_status = self.agent.get_status()

            self.visualizer.visualize(frame_array, state, agent_status)

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
            elif key == ord('o'):
                self.visualizer.toggle_agent_panel()

        # AI Agent decision making (only if enabled and no pending actions)
        if self.use_agent and not self.pending_actions:
            self._request_agent_decision(state)

        # Execute pending actions with timing
        return self._get_next_action()

    def _request_agent_decision(self, state):
        """Request a new decision from the agent."""
        if not self._agent_initialized:
            self._initialize_agent()

        if self.agent is None:
            return

        try:
            # Request decision (respects internal timing)
            made_decision = self.agent.request_decision(state)

            if made_decision:
                # Collect all queued actions
                while self.agent.has_pending_actions():
                    action = self.agent.get_next_action()
                    if action:
                        self.pending_actions.append(action)

        except Exception as e:
            print(f"Agent error: {e}")

    def _get_next_action(self) -> Optional[str]:
        """Get the next action to execute, respecting timing."""
        if not self.pending_actions:
            return None

        current_time = time.time()

        # Check if enough time has passed since last action
        if current_time - self.last_action_time < self.button_delay:
            return None

        action = self.pending_actions.pop(0)
        self.last_action_time = current_time

        # Handle wait commands
        if action.action_type == "wait":
            wait_time = float(action.value)
            # Add wait time to last_action_time to delay next action
            self.last_action_time = current_time + wait_time - self.button_delay
            return None

        # Return button action
        return action.value

    def _initialize_agent(self):
        """Initialize the AI agent (lazy loading)."""
        self._agent_initialized = True
        try:
            from agent import OllamaAgent

            agent_settings = self.settings["agent"]
            self.agent = OllamaAgent(
                model=agent_settings["model"],
                base_url=agent_settings["ollama_url"],
                timeout=agent_settings["timeout"],
                settings=agent_settings
            )

            if self.agent.check_connection():
                models = self.agent.list_models()
                print(f"AI Agent initialized (Ollama connected)")
                print(f"   Model: {agent_settings['model']}")
                print(f"   Decision interval: {agent_settings['timing']['decision_interval_seconds']}s")
                print(f"   Button delay: {agent_settings['timing']['button_delay_seconds']}s")

                if agent_settings["model"] not in models and models:
                    print(f"   Model not found, using: {models[0]}")
                    self.agent.model = models[0]

                # Start the agent
                self.agent.start()
            else:
                print("Cannot connect to Ollama. Is it running? (ollama serve)")
                print("   Agent disabled. Press 'A' to retry.")
                self.agent = None
                self.use_agent = False
        except ImportError as e:
            print(f"Failed to import agent module: {e}")
            self.agent = None
            self.use_agent = False

    def toggle_agent(self) -> bool:
        """Toggle AI agent on/off. Returns new state."""
        self.use_agent = not self.use_agent
        if self.use_agent:
            if not self._agent_initialized:
                self._initialize_agent()
            elif self.agent:
                self.agent.start()
        else:
            if self.agent:
                self.agent.stop()
            self.pending_actions = []
        return self.use_agent

    def get_agent_status(self) -> Optional[dict]:
        """Get current agent status for display."""
        if self.agent:
            return self.agent.get_status()
        return None

    def should_continue(self) -> bool:
        """Decide if we should keep running."""
        return True

    def cleanup(self):
        """Clean up resources."""
        if self.agent:
            self.agent.stop()
        if self.visualizer:
            self.visualizer.close()


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
    with open("assets/game_files/game_save.state", "wb") as f:
        pyboy.save_state(f)
        print("game saved!!")


def load_game():
    with open("assets/game_files/game_save.state", "rb") as f:
        pyboy.load_state(f)
        print('game loaded from file. Enjoy!')


def save_screenshot(frame, state, directory="assets/screenshots"):
    """
    Save current frame as grayscale PNG for tile labeling.
    Includes deduplication to avoid saving identical frames.
    """
    if state is None:
        print("Cannot save screenshot: game state not yet initialized")
        return None

    os.makedirs(directory, exist_ok=True)

    last_screenshot_hash_file = os.path.join(directory, ".last_hash")
    current_hash = state.screen_hash

    if os.path.exists(last_screenshot_hash_file):
        with open(last_screenshot_hash_file, 'r') as f:
            last_hash = f.read().strip()
            if last_hash == current_hash:
                print("Duplicate frame, not saving")
                return None

    frame_gray = frame.convert("L")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"frame_{timestamp}.png"
    filepath = os.path.join(directory, filename)
    frame_gray.save(filepath)

    with open(last_screenshot_hash_file, 'w') as f:
        f.write(current_hash)

    print(f"Screenshot saved: {filepath}")
    return filepath


# Initialize orchestrator with settings
orchestrator = GameOrchestrator(SETTINGS)

running = True
manual_mode = True  # Toggle with 'M' key

while running:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                manual_mode = not manual_mode
                print(f"Manual mode: {manual_mode}")
            elif event.key == pygame.K_a and not manual_mode:
                # Toggle AI agent (only when not in manual mode)
                agent_enabled = orchestrator.toggle_agent()
                print(f"AI Agent: {'ENABLED' if agent_enabled else 'DISABLED'}")
            elif event.key == pygame.K_v:
                viz_settings = SETTINGS.get("visualization", {})
                viz_enabled = not (orchestrator.visualizer is not None)
                if viz_enabled:
                    if not orchestrator.visualizer:
                        orchestrator.visualizer = VisualizerV2()
                    print("Perception visualizer: ENABLED")
                else:
                    if orchestrator.visualizer:
                        orchestrator.visualizer.close()
                        orchestrator.visualizer = None
                    print("Perception visualizer: DISABLED")
            elif manual_mode:
                # Manual control when enabled
                if event.key == pygame.K_z:
                    pyboy.button_press('a')
                elif event.key == pygame.K_x:
                    pyboy.button_press('b')
                elif event.key == pygame.K_RETURN:
                    pyboy.button_press('start')
                elif event.key == pygame.K_RSHIFT:
                    pyboy.button_press('select')
                elif event.key == pygame.K_UP:
                    pyboy.button_press('up')
                elif event.key == pygame.K_DOWN:
                    pyboy.button_press('down')
                elif event.key == pygame.K_LEFT:
                    pyboy.button_press('left')
                elif event.key == pygame.K_RIGHT:
                    pyboy.button_press('right')
                elif event.key == pygame.K_s:
                    save_game()
                elif event.key == pygame.K_l:
                    load_game()
                elif event.key == pygame.K_p:
                    current_frame = pyboy.screen.image
                    filepath = save_screenshot(current_frame, orchestrator.current_state)
                    if filepath:
                        print(f"Opening tile labeler for {filepath}")
                        subprocess.Popen([
                            "python", "tools/tile_labeller_interactive.py",
                            "--image", filepath, "--labels", "tile_labels.json"
                        ], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        elif event.type == pygame.KEYUP and manual_mode:
            if event.key == pygame.K_z:
                pyboy.button_release('a')
            elif event.key == pygame.K_x:
                pyboy.button_release('b')
            elif event.key == pygame.K_RETURN:
                pyboy.button_release('start')
            elif event.key == pygame.K_RSHIFT:
                pyboy.button_release('select')
            elif event.key == pygame.K_UP:
                pyboy.button_release('up')
            elif event.key == pygame.K_DOWN:
                pyboy.button_release('down')
            elif event.key == pygame.K_LEFT:
                pyboy.button_release('left')
            elif event.key == pygame.K_RIGHT:
                pyboy.button_release('right')

    # Update emulator
    pyboy.tick()

    # Get current frame
    current_frame = pyboy.screen.image

    # Process frame through orchestrator (perception + agent)
    if not manual_mode:
        action = orchestrator.process_frame(current_frame)
        if action:
            execute_action(action, duration_frames=orchestrator.action_hold_frames)
    else:
        # Still run perception for visualization even in manual mode
        orchestrator.process_frame(current_frame)

    # Render
    render_frame()
    clock.tick(SETTINGS["game"]["target_fps"])

    # Check if orchestrator wants to continue
    if not orchestrator.should_continue():
        running = False

# Cleanup
orchestrator.cleanup()
pyboy.stop()
pygame.quit()

"""
Async role-based AI agent for Pokemon Blue.

Uses a local LLM via Ollama to make game decisions based on perception state.
Supports a two-tier system:
  - Planner: High-level strategy decisions
  - Executor: Frame-by-frame action decisions
"""

import json
import re
import requests
import threading
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from queue import Queue, Empty

from perception_v2 import PerceptionState
from prompts import (
    SYSTEM_PROMPT_BASE, PLANNER_PROMPT, EXECUTOR_PROMPT,
    get_context_for_state
)


# Valid button actions the agent can issue
VALID_ACTIONS = {"up", "down", "left", "right", "a", "b", "start", "select"}


@dataclass
class AgentAction:
    """A single action to execute"""
    action_type: str  # "button" or "wait"
    value: str        # Button name or wait duration in seconds

    def __repr__(self):
        if self.action_type == "wait":
            return f"wait({self.value}s)"
        return self.value


@dataclass
class AgentResponse:
    """Response from the agent containing actions and metadata"""
    actions: List[AgentAction]      # Parsed actions with waits
    reasoning: Optional[str]        # LLM's explanation of decision
    raw_response: str               # Full LLM output for debugging
    role: str = "executor"          # Which role generated this


@dataclass
class PlanStep:
    """A single step in the high-level plan"""
    description: str
    completed: bool = False
    actions_taken: List[str] = field(default_factory=list)


@dataclass
class AgentPlan:
    """High-level plan from the planner"""
    goal: str
    steps: List[PlanStep]
    created_at: float
    reasoning: str


@dataclass
class AgentState:
    """Shared state for the async agent"""
    current_plan: Optional[AgentPlan] = None
    action_queue: Queue = field(default_factory=Queue)
    last_decision_time: float = 0
    last_plan_time: float = 0
    last_screen_hash: Optional[str] = None
    is_processing: bool = False
    last_response: Optional[AgentResponse] = None
    last_error: Optional[str] = None


class OllamaAgent:
    """
    Async AI agent that uses Ollama LLM to make game decisions.

    Supports two roles:
    - Planner: Creates high-level strategy (runs less frequently)
    - Executor: Determines specific button presses (runs more frequently)
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        settings: Optional[Dict] = None
    ):
        """
        Initialize the Ollama agent.

        Args:
            model: Ollama model name (e.g., "llama3.2:3b", "mistral:7b")
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            settings: Optional settings dict from settings.json
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        # Load settings
        self.settings = settings or {}
        timing = self.settings.get("timing", {})
        self.decision_interval = timing.get("decision_interval_seconds", 7)
        self.button_delay = timing.get("button_delay_seconds", 1.0)
        self.action_hold_frames = timing.get("action_hold_frames", 5)

        roles = self.settings.get("roles", {})
        planner_cfg = roles.get("planner", {})
        self.planner_enabled = planner_cfg.get("enabled", True)
        self.replan_interval = planner_cfg.get("replan_interval_seconds", 30)
        self.max_plan_steps = planner_cfg.get("max_plan_steps", 10)

        executor_cfg = roles.get("executor", {})
        self.executor_enabled = executor_cfg.get("enabled", True)
        self.max_actions = executor_cfg.get("max_actions_per_decision", 15)

        context_cfg = self.settings.get("context", {})
        self.max_history = context_cfg.get("max_history_exchanges", 6)
        self.compact_threshold = context_cfg.get("compact_after_exchanges", 20)

        # Conversation histories (separate for each role)
        self.planner_history: List[Dict[str, str]] = []
        self.executor_history: List[Dict[str, str]] = []

        # Agent state
        self.state = AgentState()

        # Threading for async operation
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()

    def start(self):
        """Start the async agent worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """Stop the async agent worker thread."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2)
            self._worker_thread = None

    def _worker_loop(self):
        """Background worker that processes agent decisions."""
        while not self._stop_event.is_set():
            time.sleep(0.1)  # Small sleep to prevent busy-waiting

    def request_decision(self, state: PerceptionState) -> bool:
        """
        Request a new decision based on current game state.
        Non-blocking - queues the request for background processing.

        Returns True if a new decision will be made.
        """
        current_time = time.time()

        # Check if enough time has passed since last decision
        if current_time - self.state.last_decision_time < self.decision_interval:
            return False

        # Skip if screen hasn't changed
        if state.screen_hash == self.state.last_screen_hash:
            return False

        # Update state
        with self._state_lock:
            self.state.last_screen_hash = state.screen_hash
            self.state.is_processing = True

        # Process decision in current thread (could be made async later)
        try:
            response = self._make_decision(state)
            with self._state_lock:
                self.state.last_response = response
                self.state.last_decision_time = current_time
                self.state.last_error = None

                # Queue actions
                for action in response.actions:
                    self.state.action_queue.put(action)

        except Exception as e:
            with self._state_lock:
                self.state.last_error = str(e)
        finally:
            with self._state_lock:
                self.state.is_processing = False

        return True

    def get_next_action(self) -> Optional[AgentAction]:
        """
        Get the next action from the queue.
        Returns None if queue is empty.
        """
        try:
            return self.state.action_queue.get_nowait()
        except Empty:
            return None

    def has_pending_actions(self) -> bool:
        """Check if there are actions waiting to be executed."""
        return not self.state.action_queue.empty()

    def _make_decision(self, state: PerceptionState) -> AgentResponse:
        """
        Make a decision based on current game state.
        Uses planner + executor if both are enabled.
        """
        current_time = time.time()

        # Check if we need to replan
        if (self.planner_enabled and
            (self.state.current_plan is None or
             current_time - self.state.last_plan_time > self.replan_interval)):
            self._update_plan(state)

        # Get executor decision
        return self._get_executor_decision(state)

    def _update_plan(self, state: PerceptionState):
        """Get a new high-level plan from the planner."""
        state_text = self._format_state_for_llm(state, role="planner")

        messages = [
            {"role": "system", "content": PLANNER_PROMPT}
        ]

        # Add history
        messages.extend(self.planner_history[-self.max_history:])
        messages.append({"role": "user", "content": state_text})

        try:
            response_text = self._call_ollama(messages)
            plan = self._parse_plan(response_text)

            with self._state_lock:
                self.state.current_plan = plan
                self.state.last_plan_time = time.time()

            # Update history
            self.planner_history.append({"role": "user", "content": state_text})
            self.planner_history.append({"role": "assistant", "content": response_text})

            # Compact if needed
            if len(self.planner_history) > self.compact_threshold * 2:
                self.planner_history = self.planner_history[-self.max_history * 2:]

        except Exception as e:
            with self._state_lock:
                self.state.last_error = f"Planner error: {e}"

    def _get_executor_decision(self, state: PerceptionState) -> AgentResponse:
        """Get action decision from the executor."""
        state_text = self._format_state_for_llm(state, role="executor")

        # Build system prompt with current plan context
        system_prompt = EXECUTOR_PROMPT
        if self.state.current_plan:
            plan_context = self._format_plan_context()
            system_prompt = f"{EXECUTOR_PROMPT}\n\n{plan_context}"

        # Add game context
        context = get_context_for_state(state)
        system_prompt = f"{system_prompt}\n{context}"

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Add history
        messages.extend(self.executor_history[-self.max_history:])
        messages.append({"role": "user", "content": state_text})

        response_text = self._call_ollama(messages)
        actions, reasoning = self._parse_response(response_text)

        # Update history
        self.executor_history.append({"role": "user", "content": state_text})
        self.executor_history.append({"role": "assistant", "content": response_text})

        # Compact if needed
        if len(self.executor_history) > self.compact_threshold * 2:
            self.executor_history = self.executor_history[-self.max_history * 2:]

        return AgentResponse(
            actions=actions,
            reasoning=reasoning,
            raw_response=response_text,
            role="executor"
        )

    def _format_plan_context(self) -> str:
        """Format the current plan for inclusion in executor prompt."""
        if not self.state.current_plan:
            return ""

        plan = self.state.current_plan
        lines = [
            "## Current Plan",
            f"**Goal:** {plan.goal}",
            "",
            "**Steps:**"
        ]

        for i, step in enumerate(plan.steps):
            status = "✓" if step.completed else "○"
            lines.append(f"{i+1}. [{status}] {step.description}")

        return "\n".join(lines)

    def _format_state_for_llm(self, state: PerceptionState, role: str = "executor") -> str:
        """
        Format perception state as readable text for LLM.

        Args:
            state: PerceptionState object
            role: "planner" or "executor"

        Returns:
            Formatted string describing current game state
        """
        lines = ["## Current Screen (20x18 tile grid)"]

        # Add display grid
        lines.append("```")
        for i, row in enumerate(state.get_display_grid()):
            lines.append(f"{i:2d}|{row}|")
        lines.append("   ABCDEFGHIJKLMNOPQRST")
        lines.append("```")

        # Add structured information
        sections = []

        # Menu selection
        menu = state.get_menu_selection()
        if menu.option:
            sections.append(f"**Menu cursor on: {menu.option}**")
        elif menu.cursor_position:
            sections.append(f"Cursor at position: {menu.cursor_position}")

        # Battle HP
        hp = state.get_battle_hp()
        if hp.enemy.hp_bar_tiles > 0:
            enemy_pct = int((hp.enemy.percentage or 0) * 100)
            player_pct = int((hp.player.percentage or 0) * 100)
            sections.append(f"**Battle HP** - Enemy: {enemy_pct}%, Player: {player_pct}%")

        # Pokemon names
        names = state.get_battle_names()
        if names.enemy_name or names.player_name:
            sections.append(f"Pokemon - Enemy: {names.enemy_name or '?'}, Player: {names.player_name or '?'}")

        # Extracted text
        words = state.extract_words()
        if words:
            text = " ".join(w.word for w in words[:15])
            sections.append(f"**Text on screen:** {text}")

        # Tile counts
        groups = state.get_labeled_tiles_by_category()
        sections.append(
            f"Tiles - Text: {len(groups.text)}, Terrain: {len(groups.terrain)}, "
            f"UI: {len(groups.ui)}, Sprites: {len(groups.sprite)}, Unknown: {len(groups.unlabeled)}"
        )

        if sections:
            lines.append("")
            lines.extend(sections)

        lines.append("")

        if role == "planner":
            lines.append("What should be the high-level plan? Respond with JSON: {\"goal\": \"...\", \"steps\": [\"step1\", \"step2\", ...], \"reasoning\": \"...\"}")
        else:
            lines.append("What action(s) should I take? Respond with JSON: {\"reasoning\": \"...\", \"actions\": [...]}")
            lines.append("Actions can include: up, down, left, right, a, b, start, select, wait(N)")
            lines.append("Example: [\"up\", \"up\", \"a\", \"wait(2)\", \"down\", \"a\"]")

        return "\n".join(lines)

    def _parse_plan(self, response: str) -> AgentPlan:
        """Parse planner response into a plan."""
        goal = "Continue playing"
        steps = []
        reasoning = ""

        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*"goal"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                goal = data.get("goal", goal)
                step_list = data.get("steps", [])
                steps = [PlanStep(description=s) for s in step_list[:self.max_plan_steps]]
                reasoning = data.get("reasoning", "")
            except json.JSONDecodeError:
                pass

        # Fallback: extract steps from bullet points or numbered lists
        if not steps:
            step_patterns = re.findall(r'(?:^|\n)\s*(?:\d+\.|[-*])\s*(.+)', response)
            steps = [PlanStep(description=s.strip()) for s in step_patterns[:self.max_plan_steps]]

        if not steps:
            steps = [PlanStep(description="Explore and progress")]

        return AgentPlan(
            goal=goal,
            steps=steps,
            created_at=time.time(),
            reasoning=reasoning
        )

    def _parse_response(self, response: str) -> Tuple[List[AgentAction], Optional[str]]:
        """
        Parse LLM response to extract actions and reasoning.
        Supports wait(N) commands.

        Args:
            response: Raw LLM response text

        Returns:
            Tuple of (actions list, reasoning string)
        """
        reasoning = None
        actions = []

        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*"actions"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                raw_actions = data.get("actions", [])
                reasoning = data.get("reasoning")
                actions = self._parse_action_list(raw_actions)
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse entire response as JSON
        if not actions:
            try:
                data = json.loads(response)
                raw_actions = data.get("actions", [])
                reasoning = data.get("reasoning")
                actions = self._parse_action_list(raw_actions)
            except json.JSONDecodeError:
                pass

        # Fallback: extract action words from text
        if not actions:
            actions = self._parse_actions_from_text(response)

        # Limit to max actions
        actions = actions[:self.max_actions]

        return actions, reasoning

    def _parse_action_list(self, raw_actions: List) -> List[AgentAction]:
        """Parse a list of action strings into AgentAction objects."""
        actions = []

        for action in raw_actions:
            if not isinstance(action, str):
                continue

            action = action.lower().strip()

            # Check for wait command
            wait_match = re.match(r'wait\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?', action)
            if wait_match:
                wait_time = float(wait_match.group(1))
                actions.append(AgentAction(action_type="wait", value=str(wait_time)))
                continue

            # Check for valid button action
            if action in VALID_ACTIONS:
                actions.append(AgentAction(action_type="button", value=action))

        return actions

    def _parse_actions_from_text(self, response: str) -> List[AgentAction]:
        """Extract actions from plain text response."""
        actions = []

        # Look for wait commands first
        for match in re.finditer(r'wait\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?', response.lower()):
            wait_time = float(match.group(1))
            actions.append(AgentAction(action_type="wait", value=str(wait_time)))

        # Look for button actions
        for word in response.lower().split():
            word = re.sub(r'[^\w]', '', word)
            if word in VALID_ACTIONS:
                actions.append(AgentAction(action_type="button", value=word))

        return actions

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """
        Make HTTP request to Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Response text from LLM

        Raises:
            Exception on connection or API error
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 200  # Slightly more for plans
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}. Is it running?")
        except requests.exceptions.Timeout:
            raise Exception(f"Ollama request timed out after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Ollama HTTP error: {e}")

    def reset(self):
        """Clear all state and history."""
        self.planner_history = []
        self.executor_history = []
        with self._state_lock:
            self.state = AgentState()

    def check_connection(self) -> bool:
        """
        Check if Ollama server is reachable.

        Returns:
            True if connection successful
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        List available models on Ollama server.

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def get_status(self) -> Dict:
        """Get current agent status for visualization."""
        with self._state_lock:
            return {
                "has_plan": self.state.current_plan is not None,
                "plan_goal": self.state.current_plan.goal if self.state.current_plan else None,
                "plan_steps": len(self.state.current_plan.steps) if self.state.current_plan else 0,
                "pending_actions": self.state.action_queue.qsize(),
                "is_processing": self.state.is_processing,
                "last_reasoning": self.state.last_response.reasoning if self.state.last_response else None,
                "last_error": self.state.last_error,
                "time_since_decision": time.time() - self.state.last_decision_time if self.state.last_decision_time else None,
                "time_since_plan": time.time() - self.state.last_plan_time if self.state.last_plan_time else None,
            }

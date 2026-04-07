"""
ShopEasy Customer Support Resolution Gym — HTTP/WebSocket Client.

Provides the CustomerSupportGym2Env client for connecting to the environment
server over HTTP or WebSocket.  Use this to run agents against the env.
"""

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SupportAction, SupportObservation

# Public aliases (backward compat)
CustomerSupportGym2Action = SupportAction
CustomerSupportGym2Observation = SupportObservation


class CustomerSupportGym2Env(
    EnvClient[SupportAction, SupportObservation, State]
):
    """
    Client for the ShopEasy Customer Support Resolution Gym.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step episode collection with low latency.
    Each client instance has its own isolated environment session.

    Example — connect to a running server:
        >>> with CustomerSupportGym2Env(base_url="http://localhost:8000") as env:
        ...     obs_result = env.reset()           # returns StepResult
        ...     obs = obs_result.observation
        ...     print(obs.customer_message)
        ...
        ...     action = SupportAction(
        ...         action_type="tool_call",
        ...         tool_name="lookup_order",
        ...         tool_args={"order_id": "SE-1001"},
        ...     )
        ...     result = env.step(action)
        ...     print(result.observation.tool_result)

    Example — spin up a Docker container automatically:
        >>> env = CustomerSupportGym2Env.from_docker_image(
        ...     "shopeasy-support-gym:latest"
        ... )
        >>> try:
        ...     result = env.reset(task_id="angry_customer")
        ... finally:
        ...     env.close()
    """

    # ------------------------------------------------------------------
    # Internal serialisation helpers (called by EnvClient base class)
    # ------------------------------------------------------------------

    def _step_payload(self, action: SupportAction) -> Dict[str, Any]:
        """Convert SupportAction to JSON payload for the /step endpoint."""
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.action_type == "tool_call":
            payload["tool_name"] = action.tool_name
            payload["tool_args"] = action.tool_args or {}
        elif action.action_type == "send_message":
            payload["message"] = action.message or ""
        elif action.action_type == "close_ticket":
            payload["resolution"] = action.resolution or "unresolved"
        return payload

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[SupportObservation]:
        """Parse server JSON response into StepResult[SupportObservation]."""
        obs_data = payload.get("observation", {})
        observation = SupportObservation(
            customer_message=obs_data.get("customer_message", ""),
            customer_sentiment=obs_data.get("customer_sentiment", "calm"),
            tool_result=obs_data.get("tool_result"),
            tool_error=obs_data.get("tool_error"),
            ticket_id=obs_data.get("ticket_id", ""),
            ticket_status=obs_data.get("ticket_status", "open"),
            issue_type=obs_data.get("issue_type", ""),
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            verified_facts=obs_data.get("verified_facts", {}),
            conversation_history=obs_data.get("conversation_history", []),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 20),
            steps_remaining=obs_data.get("steps_remaining", 20),
            reward_breakdown=obs_data.get("reward_breakdown"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )

    # ------------------------------------------------------------------
    # Convenience — pass reset kwargs through to server
    # ------------------------------------------------------------------

    def reset(  # type: ignore[override]
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> StepResult[SupportObservation]:
        """
        Start a new episode.

        Args:
            task_id   : specific scenario ID (e.g. 'simple_refund').
                        If None, picks randomly based on difficulty.
            difficulty: filter ('easy' | 'medium' | 'hard').
                        Ignored if task_id is provided.
            seed      : random seed for reproducibility.

        Returns:
            StepResult with initial SupportObservation.
        """
        payload: Dict[str, Any] = {}
        if task_id:
            payload["task_id"] = task_id
        if difficulty:
            payload["difficulty"] = difficulty
        if seed is not None:
            payload["seed"] = seed
        return super().reset(**payload)

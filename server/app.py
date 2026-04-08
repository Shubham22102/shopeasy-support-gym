"""
FastAPI application for the ShopEasy Customer Support Resolution Gym.

Exposes the SupportEnvironment over HTTP and WebSocket endpoints,
supporting multi-session concurrent RL training workers.

Endpoints:
    POST /reset    — Start a new episode (optional: task_id, difficulty, seed)
    POST /step     — Execute an action
    GET  /state    — Get current episode state
    GET  /schema   — Get action/observation schemas
    WS   /ws       — WebSocket for persistent multi-step sessions
"""

import os
from typing import Any, Dict

# Load .env FIRST — reads OPENAI_API_KEY, HF_TOKEN, MAX_CONCURRENT_ENVS, etc.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass  # In Docker/HF Spaces, vars are injected directly into environment

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with:\n    uv sync") from e

try:
    from ..models import SupportAction, SupportObservation
    from .Customer_Support_Gym_2_environment import SupportEnvironment
except (ModuleNotFoundError, ImportError):
    import sys
    import os

    # Add project root to sys.path so direct execution (`python server/app.py`) works
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import SupportAction, SupportObservation  # type: ignore
    from server.Customer_Support_Gym_2_environment import SupportEnvironment  # type: ignore

# Backward-compat aliases
CustomerSupportGym2Action = SupportAction
CustomerSupportGym2Observation = SupportObservation
CustomerSupportGym2Environment = SupportEnvironment

# Create the FastAPI app with multi-session support
# MAX_CONCURRENT_ENVS from .env (default 16) allows parallel RL training workers
_max_sessions = int(os.getenv("MAX_CONCURRENT_ENVS", "16"))
app = create_app(
    SupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="shopeasy-support-gym",
    max_concurrent_envs=_max_sessions,
)


def _normalize_grader_inputs(
    task_id: str | None,
    world_state: Dict[str, Any] | None,
    payload: Dict[str, Any] | None = None,
) -> tuple[str, Dict[str, Any]]:
    merged_state: Dict[str, Any] = {}
    if isinstance(world_state, dict):
        merged_state.update(world_state)
    if isinstance(payload, dict):
        merged_state.update(payload)

    resolved_task = (
        task_id
        or merged_state.get("task_id")
        or merged_state.get("current_task_id")
        or os.getenv("DEFAULT_TASK_ID")
        or "simple_refund"
    )
    return resolved_task, merged_state


@app.get("/grader")
def grader_endpoint(
    task_id: str | None = None,
    session_id: str | None = None,
):
    """Runtime grader endpoint for hackathon task validation."""
    from reward.grader import TaskGrader

    resolved_task, state = _normalize_grader_inputs(task_id, {"session_id": session_id})
    result = TaskGrader(default_task_id=resolved_task).grade(
        task_id=resolved_task,
        world_state=state,
    )
    payload = result.as_info()
    payload["endpoint"] = "/grader"
    return payload


@app.post("/grader")
def grader_endpoint_post(payload: Dict[str, Any]):
    """POST variant of the hackathon grader endpoint."""
    from reward.grader import TaskGrader

    task_id = payload.get("task_id")
    world_state = payload.get("world_state")
    resolved_task, state = _normalize_grader_inputs(task_id, world_state, payload)
    result = TaskGrader(default_task_id=resolved_task).grade(
        task_id=resolved_task,
        world_state=state,
    )
    payload = result.as_info()
    payload["endpoint"] = "/grader"
    return payload


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port != 8000:
        main(port=args.port)
    else:
        main()

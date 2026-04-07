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

# Load .env FIRST — reads OPENAI_API_KEY, HF_TOKEN, MAX_CONCURRENT_ENVS, etc.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # In Docker/HF Spaces, vars are injected directly into environment

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with:\n    uv sync"
    ) from e

try:
    from ..models import SupportAction, SupportObservation
    from .Customer_Support_Gym_2_environment import SupportEnvironment
except (ModuleNotFoundError, ImportError):
    from models import SupportAction, SupportObservation
    from server.Customer_Support_Gym_2_environment import SupportEnvironment

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


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port != 8000:
        main(port=args.port)
    else:
        main()

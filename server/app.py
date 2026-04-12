"""
FastAPI application for the ShopEasy Customer Support Resolution Gym.
"""

import os
from typing import Any, Dict

# Load .env FIRST
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with:\n    uv sync") from e

# Simplified import pattern matching working repo
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
_max_sessions = int(os.getenv("MAX_CONCURRENT_ENVS", "16"))
app = create_app(
    SupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="shopeasy-support-gym",
    max_concurrent_envs=_max_sessions,
)

# REMOVED: Custom grader endpoints - create_app() provides these automatically!

def main():
    """Entry point for direct execution."""
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
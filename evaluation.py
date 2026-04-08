import os
import json
import asyncio
from typing import Any

from server.data.scenarios import SCENARIOS
from openai import OpenAI

try:
    from Customer_Support_Gym_2.client import CustomerSupportGym2Env
    from Customer_Support_Gym_2.models import SupportAction
except ImportError:
    from client import CustomerSupportGym2Env  # type: ignore
    from models import SupportAction  # type: ignore

from inference import get_agent_action

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY: str = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "no-key-set"

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
SUCCESS_SCORE_THRESHOLD: float = 0.4


def run_episode_sync(env_url: str, task_id: str, client: OpenAI) -> float:
    """Synchronous implementation of running an episode."""
    conversation = []
    with CustomerSupportGym2Env(base_url=env_url).sync() as env:
        reset_result = env.reset(task_id=task_id)
        obs = reset_result.observation.model_dump()
        max_steps = obs.get("max_steps", 20)

        score = 0.0

        for step in range(1, max_steps + 1):
            if reset_result.done and step == 1:
                break

            action_dict = get_agent_action(client, obs, conversation)

            try:
                step_result = env.step(SupportAction.model_validate(action_dict))
            except Exception as exc:
                print(f"[DEBUG] Error taking step: {exc}")
                break

            obs = step_result.observation.model_dump()
            reward = float(step_result.reward)
            done = bool(step_result.done)

            if done:
                score = reward
                break

        return score


async def run_episode(task_id: str) -> Any:
    """Async wrapper around the episode execution to match the requested signature."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Run the synchronous episode logic in a background thread
    score = await asyncio.to_thread(run_episode_sync, ENV_URL, task_id, client)

    # Create a dummy result object with `.reward`
    class Result:
        def __init__(self, reward: float):
            self.reward = reward

    return Result(score)


async def evaluate_agent(model_name: str, n_episodes: int = 5):
    """Run all tasks and compute success rates"""
    results = {}
    print(
        f"Starting evaluation for model '{model_name}' on {len(SCENARIOS)} tasks, {n_episodes} episodes each.\n"
    )

    for task_def in SCENARIOS:
        print(f"Running Task: {task_def.task_id} ({task_def.title})")
        scores = []
        for ep in range(n_episodes):
            print(f"  Episode {ep + 1}/{n_episodes}...", end="", flush=True)
            result = await run_episode(task_id=task_def.task_id)
            scores.append(result.reward)

            status = "✔️" if result.reward >= SUCCESS_SCORE_THRESHOLD else "❌"
            print(f" Score: {result.reward:.2f} {status}")

        success_rate = sum(1 for s in scores if s >= SUCCESS_SCORE_THRESHOLD) / len(
            scores
        )
        avg_score = sum(scores) / len(scores) if scores else 0.0

        results[task_def.task_id] = {
            "success_rate": success_rate,
            "avg_score": avg_score,
        }
        print(
            f"  -> Task Summary: Success Rate {success_rate * 100:.1f}%, Avg Score {avg_score:.2f}\n"
        )

    print("--- Final Evaluation Results ---")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    n_eps = int(os.getenv("EVAL_EPISODES", "5"))
    asyncio.run(evaluate_agent(model_name=MODEL_NAME, n_episodes=n_eps))

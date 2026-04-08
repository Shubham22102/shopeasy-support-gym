"""
Inference Script — ShopEasy Customer Support Resolution Gym
============================================================
Baseline ReAct-style agent that resolves customer support tickets
using structured JSON tool calls via the OpenAI-compatible API.

REQUIRED ENV VARS (set in .env or export before running):
  API_BASE_URL   — LLM API endpoint (default: https://api.openai.com/v1)
  MODEL_NAME     — model identifier  (default: gpt-4o-mini)
  OPENAI_API_KEY — your API key (also checked as HF_TOKEN for HF inference)
  HF_TOKEN       — Hugging Face token (used as API key for HF endpoints)

STDOUT FORMAT (mandatory for hackathon grader):
  [START] task=<task_id> env=shopeasy-support-gym model=<model>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

USAGE:
  # Run against local server (start server first):
  uvicorn server.app:app --port 8000 &
  python inference.py

  # Run specific task:
  TASK_ID=angry_customer python inference.py

  # Run against Docker image:
  LOCAL_IMAGE_NAME=shopeasy-support-gym python inference.py
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

# ── Load .env FIRST before reading any env vars ────────────────────────────
try:
    from dotenv import load_dotenv

    # load_dotenv() looks for .env in cwd and parents; override=False keeps
    # already-set shell vars untouched (so CI/CD env vars take priority).
    load_dotenv(override=False)
except ImportError:
    pass  # dotenv not installed — rely on shell env vars (fine in Docker/HF)

from openai import OpenAI

try:
    from Customer_Support_Gym_2.client import CustomerSupportGym2Env
    from Customer_Support_Gym_2.models import SupportAction
except ImportError:
    from client import CustomerSupportGym2Env  # type: ignore
    from models import SupportAction  # type: ignore

# ── Configuration (all read from .env / shell env) ─────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
# OpenAI key takes priority; fall back to HF_TOKEN for HF inference endpoints
API_KEY: str = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "no-key-set"
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# Task selection
TASK_ID: str = os.getenv("TASK_ID", os.getenv("DEFAULT_TASK_ID", ""))
DIFFICULTY: str = os.getenv("DIFFICULTY", os.getenv("DEFAULT_DIFFICULTY", ""))
BENCHMARK: str = "shopeasy-support-gym"

# Docker image name (if using from_docker_image instead of local server)
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")

# Agent config
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "20"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "400"))
SUCCESS_SCORE_THRESHOLD: float = 0.4  # reward >= 0.4 counts as "success"

# ── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are a professional customer support agent for ShopEasy, an e-commerce platform.
Your goal is to resolve the customer's issue efficiently and correctly by following company policies.

AVAILABLE TOOLS:
  lookup_order          — look up order details. args: {"order_id": "SE-XXXX"}
  process_refund        — issue a refund. args: {"order_id": "SE-XXXX", "reason": "...", "amount": float}
  search_kb             — search knowledge base. args: {"query": "refund policy electronics"}
  escalate_to_human     — escalate ticket. args: {"reason": "...", "priority": "normal"|"high"}
  cancel_subscription   — cancel a subscription. args: {"subscription_id": "SUB-XXXX", "reason": "..."}
  check_payment         — verify payment/detect duplicate charge. args: {"order_id": "SE-XXXX"}

RULES:
  1. ALWAYS call lookup_order before making any refund promise.
  2. ALWAYS call search_kb before citing a policy to the customer.
  3. NEVER process a refund on a fraud-risk order — escalate instead.
  4. If the return window has EXPIRED, offer store credit (not full refund).
  5. Apologize sincerely when the customer is upset.
  6. Be concise — do not use unnecessary tool calls.

RESPONSE FORMAT — you MUST output EXACTLY ONE of these JSON objects per turn:
  Tool call:
    {"action_type": "tool_call", "tool_name": "<name>", "tool_args": {<args>}}

  Message to customer:
    {"action_type": "send_message", "message": "<your message to customer>"}

  Close ticket:
    {"action_type": "close_ticket", "resolution": "resolved"|"escalated"|"unresolved"}

Output ONLY the JSON object. No explanations, no markdown, no extra text.
""").strip()


# ── Logging helpers (mandatory hackathon format — NO extra spaces) ────────


def log_start(task: str, env: str, model: str) -> None:
    sys.stdout.write(f"[START]task={task}env={env}model={model}\n")
    sys.stdout.flush()


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    # CRITICAL: exact 2-decimal reward, no spaces between fields
    reward_str = f"{reward:.2f}"
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    # Remove newlines from action to guarantee single-line output
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    sys.stdout.write(
        f"[STEP]step={step}action={action_clean}reward={reward_str}done={done_str}error={error_str}\n"
    )
    sys.stdout.flush()


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    # Format all rewards to exactly 2 decimal places, comma-separated, no spaces
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    sys.stdout.write(
        f"[END]success={success_str}steps={steps}score={score:.3f}rewards={rewards_str}\n"
    )
    sys.stdout.flush()


# ── Observation → LLM prompt ───────────────────────────────────────────────


def build_user_prompt(obs_dict: Dict[str, Any]) -> str:
    """Convert a SupportObservation dict into a clear prompt for the LLM."""
    customer_msg = obs_dict.get("customer_message", "")
    sentiment = obs_dict.get("customer_sentiment", "calm")
    tool_result = obs_dict.get("tool_result")
    tool_error = obs_dict.get("tool_error")
    ticket_status = obs_dict.get("ticket_status", "open")
    issue_type = obs_dict.get("issue_type", "")
    verified_facts = obs_dict.get("verified_facts", {})
    steps_remaining = obs_dict.get("steps_remaining", 0)

    # Last few turns of conversation history
    history = obs_dict.get("conversation_history", [])
    history_lines = []
    for turn in history[-6:]:  # last 6 turns
        role = turn.get("role", "?").upper()
        content = turn.get("content", "")[:200]
        history_lines.append(f"  [{role}]: {content}")
    history_block = "\n".join(history_lines) if history_lines else "  (no history)"

    # Tool result block
    tool_block = ""
    if tool_result:
        tool_block = f"\nLAST TOOL RESULT:\n{json.dumps(tool_result, indent=2)}"
    elif tool_error:
        tool_block = f"\nLAST TOOL ERROR: {tool_error}"

    prompt = textwrap.dedent(f"""
    TICKET STATUS: {ticket_status.upper()} | ISSUE: {issue_type} | SENTIMENT: {sentiment.upper()}
    STEPS REMAINING: {steps_remaining}

    CONVERSATION HISTORY:
{history_block}

    CUSTOMER JUST SAID: "{customer_msg}"
{tool_block}

    VERIFIED FACTS SO FAR: {json.dumps(verified_facts) if verified_facts else "none yet"}

    What is your next action? Respond with ONLY a JSON object.
    """).strip()

    return prompt


# ── LLM call ──────────────────────────────────────────────────────────────


def get_agent_action(
    client: OpenAI,
    obs_dict: Dict[str, Any],
    conversation: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Call the LLM and parse its JSON response into an action dict.
    Falls back to send_message if JSON parsing fails.
    """
    user_prompt = build_user_prompt(obs_dict)
    conversation.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        conversation.append({"role": "assistant", "content": raw})

        # Strip markdown code fences if model wraps JSON in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        action = json.loads(raw)
        return action

    except json.JSONDecodeError:
        # Model didn't return valid JSON — extract any text and send as message
        fallback_msg = (
            raw[:300] if raw else "I'm looking into your issue, please hold on."
        )
        return {"action_type": "send_message", "message": fallback_msg}

    except Exception:
        return {
            "action_type": "send_message",
            "message": "I'm sorry, please give me a moment.",
        }


# ── Main episode loop ──────────────────────────────────────────────────────


def run_episode(env_url: str = "http://localhost:8000") -> None:
    """
    Run one full support episode against the ShopEasy environment server.

    Connects to the env via HTTP (REST API), runs until done or max steps,
    and prints [START]/[STEP]/[END] logs to stdout.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    conversation: List[Dict[str, str]] = []  # LLM multi-turn memory

    task_id = TASK_ID or None
    difficulty = DIFFICULTY or None

    try:
        with CustomerSupportGym2Env(base_url=env_url).sync() as env:
            reset_result = env.reset(task_id=task_id, difficulty=difficulty)
            obs_model = reset_result.observation
            obs = obs_model.model_dump()
            actual_task = obs.get("task_id", task_id or "unknown")
            max_steps = obs.get("max_steps", MAX_STEPS)

            log_start(task=actual_task, env=BENCHMARK, model=MODEL_NAME)

            for step in range(1, max_steps + 1):
                if reset_result.done and step == 1:
                    break

                action_dict = get_agent_action(client, obs, conversation)
                action_str = json.dumps(action_dict)

                try:
                    step_result = env.step(SupportAction.model_validate(action_dict))
                except Exception as exc:
                    log_step(
                        step=step,
                        action=action_str,
                        reward=0.0,
                        done=True,
                        error=str(exc),
                    )
                    break

                obs_model = step_result.observation
                obs = obs_model.model_dump()
                reward = float(step_result.reward)
                done = bool(step_result.done)
                error = obs.get("tool_error")

                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step, action=action_str, reward=reward, done=done, error=error
                )

                if done:
                    # Clamp to strict open interval used by hackathon graders
                    score = max(0.02, min(0.98, reward))
                    success = score >= SUCCESS_SCORE_THRESHOLD
                    break

    except Exception as exc:
        sys.stderr.write(f"[ERROR] Failed to connect to env at {env_url}: {exc}\n")
        sys.stderr.write(
            "[ERROR] Is the server running? Start with: uv run server or uvicorn server.app:app --port 8000\n"
        )
        sys.exit(1)
    finally:
        # Final safety clamp — ensures score is never exactly 0.0 or 1.0
        score = max(0.02, min(0.98, score))
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env_url = os.getenv("ENV_URL", "http://localhost:8000")
    run_episode(env_url=env_url)

"""
Inference Script — ShopEasy Customer Support Resolution Gym
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from Customer_Support_Gym_2.client import CustomerSupportGym2Env
    from Customer_Support_Gym_2.models import SupportAction
except ImportError:
    from client import CustomerSupportGym2Env  # type: ignore
    from models import SupportAction  # type: ignore

# Environment variables with defaults
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY: str = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "no-key-set"
TASK_ID: str = os.getenv("TASK_ID", os.getenv("DEFAULT_TASK_ID", ""))
DIFFICULTY: str = os.getenv("DIFFICULTY", os.getenv("DEFAULT_DIFFICULTY", ""))
BENCHMARK: str = "shopeasy-support-gym"
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")

MAX_STEPS: int = int(os.getenv("MAX_STEPS", "20"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "400"))
SUCCESS_SCORE_THRESHOLD: float = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
You are a professional customer support agent for ShopEasy, an e-commerce platform.

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

RESPONSE FORMAT — output EXACTLY ONE JSON object per turn:
  {"action_type": "tool_call", "tool_name": "<name>", "tool_args": {<args>}}
  or
  {"action_type": "send_message", "message": "<your message>"}
  or  
  {"action_type": "close_ticket", "resolution": "resolved"|"escalated"|"unresolved"}

Output ONLY the JSON object. No explanations, no markdown.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    """Format: [START] task=X env=Y model=Z"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Format: [STEP] step=X action=Y reward=Z done=W error=V"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Format: [END] success=X steps=Y score=Z rewards=R1,R2,..."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def clamp_score(score: float) -> float:
    """Ensure final score stays strictly inside (0, 1)."""
    return round(min(max(float(score), 0.02), 0.98), 3)


def build_user_prompt(obs_dict: Dict[str, Any]) -> str:
    """Build prompt from observation."""
    customer_msg = obs_dict.get("customer_message", "")
    sentiment = obs_dict.get("customer_sentiment", "calm")
    tool_result = obs_dict.get("tool_result")
    tool_error = obs_dict.get("tool_error")
    ticket_status = obs_dict.get("ticket_status", "open")
    issue_type = obs_dict.get("issue_type", "")
    verified_facts = obs_dict.get("verified_facts", {})
    steps_remaining = obs_dict.get("steps_remaining", 0)

    history = obs_dict.get("conversation_history", [])
    history_lines = []
    for turn in history[-6:]:
        role = turn.get("role", "?").upper()
        content = turn.get("content", "")[:200]
        history_lines.append(f"  [{role}]: {content}")
    history_block = "\n".join(history_lines) if history_lines else "  (no history)"

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

    VERIFIED FACTS: {json.dumps(verified_facts) if verified_facts else "none"}

    What is your next action? Respond with ONLY a JSON object.
    """).strip()

    return prompt


def get_agent_action(
    client: OpenAI,
    obs_dict: Dict[str, Any],
    conversation: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Call LLM and parse JSON action."""
    user_prompt = build_user_prompt(obs_dict)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation + [{"role": "user", "content": user_prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        conversation.append({"role": "user", "content": user_prompt})
        conversation.append({"role": "assistant", "content": raw})

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError:
        fallback = raw[:300] if raw else "I'm looking into your issue, please hold on."
        return {"action_type": "send_message", "message": fallback}
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return {"action_type": "send_message", "message": "I'm sorry, please give me a moment."}


def run_episode(env_url: str = "http://localhost:8000") -> None:
    """Run one episode."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    conversation: List[Dict[str, str]] = []

    task_id = TASK_ID or None
    difficulty = DIFFICULTY or None

    log_start(task=task_id or "default", env=BENCHMARK, model=MODEL_NAME)

    try:
        with CustomerSupportGym2Env(base_url=env_url).sync() as env:
            result = env.reset(task_id=task_id, difficulty=difficulty)
            obs_model = result.observation
            obs = obs_model.model_dump()
            actual_task = obs.get("task_id", task_id or "unknown")
            max_steps = obs.get("max_steps", MAX_STEPS)

            if actual_task != (task_id or "default"):
                log_start(task=actual_task, env=BENCHMARK, model=MODEL_NAME)

            for step in range(1, max_steps + 1):
                if result.done:
                    break

                action_dict = get_agent_action(client, obs, conversation)
                action_str = json.dumps(action_dict)

                try:
                    result = env.step(SupportAction.model_validate(action_dict))
                except Exception as exc:
                    log_step(step=step, action=action_str, reward=0.00, done=True, error=str(exc))
                    break

                obs_model = result.observation
                obs = obs_model.model_dump()
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                error = obs.get("tool_error")

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                if done:
                    break

            # Use final reward from environment as score
            score = rewards[-1] if rewards else 0.0
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Environment error: {exc}", flush=True)
    finally:
        final_score = clamp_score(score)
        final_success = final_score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=final_success, steps=steps_taken, score=final_score, rewards=rewards)


if __name__ == "__main__":
    env_url = os.getenv("ENV_URL", "http://localhost:8000")
    run_episode(env_url=env_url)
"""
Hackathon-compatible task grader — fully self-contained.

This module intentionally does NOT import from grader.py or server/
so it can be loaded by the submission validator in isolation
(without openenv-core installed in the validator environment).

Scoring contract
----------------
  grade(task_id, world_state, **kwargs) -> (float, dict)

  The returned float is ALWAYS strictly inside (0, 1):
    0.02 ≤ score ≤ 0.98
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRICT_SCORE_MIN = 0.02
STRICT_SCORE_MAX = 0.98
SCORE_RANGE = [STRICT_SCORE_MIN, STRICT_SCORE_MAX]

OUTCOME_MAX = 0.60
PROCESS_MAX = 0.30
EFFICIENCY_MAX = 0.10

# Tasks that require a knowledge-base search
KB_REQUIRED_TASKS = {"expired_return", "kb_policy_question", "vip_warranty_claim", "warranty_claim"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(score: float) -> float:
    """Clamp to the open interval (0, 1) required by hackathon validators."""
    # OpenEnv requirement: scores must be strictly in (0.0, 1.0)
    clamped = min(STRICT_SCORE_MAX, max(STRICT_SCORE_MIN, float(score)))
    return round(clamped, 3)


def _compute_efficiency(step_count: int, max_steps: int) -> float:
    if max_steps <= 0:
        return 0.0
    steps_used_ratio = step_count / max_steps
    return max(0.0, 1.0 - steps_used_ratio) * EFFICIENCY_MAX


def _compute_process(order: Dict[str, Any], task_id: str, facts: Dict[str, Any]) -> float:
    score = 0.0

    if facts.get("order_looked_up"):
        score += 0.10

    if task_id in KB_REQUIRED_TASKS and facts.get("kb_searched"):
        score += 0.05

    if order:
        # Penalty: refunded a fraud-risk order
        if facts.get("refund_processed") and order.get("is_fraud_risk"):
            score -= 0.10

        # Penalty: full refund outside return window on expired_return task
        if (
            facts.get("refund_processed")
            and not order.get("is_damaged")
            and not order.get("is_fraud_risk")
            and order.get("status") == "delivered"
            and not order.get("within_return_window")
            and task_id == "expired_return"
        ):
            score -= 0.05

        # Bonus: correctly escalated fraud risk
        if (
            order.get("is_fraud_risk")
            and facts.get("escalated")
            and not facts.get("refund_processed")
        ):
            score += 0.15

        # Bonus: duplicate charge — payment verified before refund
        if task_id == "duplicate_charge":
            if facts.get("payment_checked") and facts.get("refund_processed"):
                score += 0.10
            elif facts.get("refund_processed") and not facts.get("payment_checked"):
                score -= 0.05

    return max(0.0, min(PROCESS_MAX, score))


def _compute_outcome(
    order: Dict[str, Any],
    task_id: str,
    facts: Dict[str, Any],
    resolution: str,
    customer_mood: float,
    agent_sent_messages: bool,
) -> float:
    """Scenario-specific outcome scoring."""
    if resolution == "timeout":
        return 0.05

    mood_bonus = 0.0
    if customer_mood >= 0.5:
        mood_bonus = 0.10
    elif customer_mood >= 0.0:
        mood_bonus = 0.05
    elif customer_mood < -0.5:
        mood_bonus = -0.05

    base = 0.0

    if task_id == "simple_refund":
        if facts.get("refund_processed") and resolution == "resolved":
            base = 1.0
        elif not facts.get("refund_processed"):
            base = 0.10
        else:
            base = 0.40

    elif task_id == "delivery_tracking":
        if facts.get("order_looked_up") and resolution == "resolved":
            base = 1.0
        elif facts.get("refund_processed"):
            base = 0.0
        else:
            base = 0.20

    elif task_id == "kb_policy_question":
        if resolution == "resolved":
            base = 1.0
        else:
            base = 0.20

    elif task_id == "cancellation_request":
        if facts.get("order_looked_up") and resolution == "resolved":
            base = 1.0
        else:
            base = 0.20

    elif task_id == "expired_return":
        if facts.get("refund_processed") and not (order or {}).get("within_return_window"):
            base = -0.20
        elif facts.get("kb_searched") and not facts.get("refund_processed"):
            base = 0.90
        else:
            base = 0.20

    elif task_id == "wrong_item_sent":
        if facts.get("refund_processed") and resolution == "resolved":
            base = 1.0
        else:
            base = 0.10

    elif task_id == "duplicate_charge":
        if facts.get("payment_checked") and facts.get("refund_processed"):
            base = 1.0
        elif facts.get("refund_processed") and not facts.get("payment_checked"):
            base = 0.35
        else:
            base = 0.10

    elif task_id == "partial_order":
        if facts.get("refund_processed") and resolution == "resolved":
            base = 1.0
        else:
            base = 0.20

    elif task_id == "damaged_item":
        if facts.get("refund_processed") and (order or {}).get("is_damaged") and resolution == "resolved":
            base = 1.0
        elif facts.get("escalated") and not facts.get("refund_processed"):
            base = -0.10
        else:
            base = 0.10

    elif task_id == "angry_customer":
        if facts.get("order_looked_up") and resolution == "resolved" and not facts.get("escalated"):
            base = 1.0
        elif facts.get("escalated"):
            base = 0.30
        else:
            base = 0.10

    elif task_id == "fraud_risk":
        if facts.get("escalated") and not facts.get("refund_processed"):
            base = 1.0
        elif facts.get("refund_processed") and (order or {}).get("is_fraud_risk"):
            base = -0.30
        else:
            base = 0.0

    elif task_id == "vip_warranty_claim":
        base = 0.0
        if facts.get("order_looked_up") and facts.get("is_vip"):
            base += 0.15
        if facts.get("kb_searched"):
            base += 0.30
        if facts.get("order_looked_up") and (order or {}).get("has_warranty") and resolution == "resolved":
            base += 0.55

    else:
        # Generic fallback
        base = 1.0 if resolution == "resolved" else 0.3

    raw = max(0.0, base) * OUTCOME_MAX + mood_bonus
    return max(0.0, min(OUTCOME_MAX, raw))


def _score_task(
    task_id: str,
    facts: Dict[str, Any],
    order: Dict[str, Any],
    resolution: str,
    step_count: int,
    max_steps: int,
    customer_mood: float,
    agent_sent_messages: bool,
) -> float:
    r_outcome = _compute_outcome(order, task_id, facts, resolution, customer_mood, agent_sent_messages)
    r_process = _compute_process(order, task_id, facts)
    r_efficiency = _compute_efficiency(step_count, max_steps)
    total = round(r_outcome + r_process + r_efficiency, 4)
    return max(0.0, min(1.0, total))


# ---------------------------------------------------------------------------
# Public API — TaskGrader class expected by the hackathon validator
# ---------------------------------------------------------------------------


class TaskGrader:
    """
    Hackathon-compatible grader.

    The validator calls:
        grader = TaskGrader()
        score, info = grader.grade(task_id=..., world_state={...})

    The score is always strictly inside (0, 1).
    """

    def __init__(self, default_task_id: str | None = None) -> None:
        self.default_task_id = default_task_id

    def grade(
        self,
        task_id: str | None = None,
        world_state: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "TaskGradeResult":
        state = world_state or {}

        task = (
            task_id
            or self.default_task_id
            or state.get("task_id")
            or state.get("current_task_id")
            or kwargs.get("task_id")
            or "simple_refund"
        )

        facts = (
            kwargs.get("verified_facts")
            or state.get("verified_facts")
            or state.get("facts")
            or {}
        )
        order = kwargs.get("order") or state.get("order") or {}
        resolution = (
            kwargs.get("final_resolution")
            or kwargs.get("resolution")
            or state.get("close_resolution")
            or state.get("final_resolution")
            or "unresolved"
        )
        step_count = int(kwargs.get("step_count", state.get("step_count", 10)))
        max_steps = int(kwargs.get("max_steps", state.get("max_steps", 20)))
        customer_mood = float(kwargs.get("customer_mood", state.get("customer_mood", 0.0)))
        agent_sent_messages = bool(
            kwargs.get("agent_sent_messages", state.get("agent_sent_messages", True))
        )

        raw_score = 0.5

        raw_score = _score_task(
            task_id=task,
            facts=facts,
            order=order,
            resolution=resolution,
            step_count=step_count,
            max_steps=max_steps,
            customer_mood=customer_mood,
            agent_sent_messages=agent_sent_messages,
        )

        score = _clamp(raw_score)
        passed = score >= 0.5

        return TaskGradeResult(
            score=score,
            passed=passed,
            feedback=f"Task '{task}' score: {score:.3f} (resolution={resolution})",
            task_id=task,
        )


@dataclass
class TaskGradeResult:
    """Compatibility result with both attribute access and tuple unpacking."""

    score: float
    passed: bool
    feedback: str
    task_id: str

    def __post_init__(self) -> None:
        self.score = _clamp(self.score)

    def as_info(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": self.score,
            "passed": self.passed,
            "feedback": self.feedback,
            "score_range": SCORE_RANGE,
        }

    def __iter__(self) -> Iterator[Any]:
        yield self.score
        yield self.as_info()


# ---------------------------------------------------------------------------
# Module-level grader discovery — some validators call get_task_graders()
# ---------------------------------------------------------------------------

TASK_IDS = [
    "simple_refund",
    "delivery_tracking",
    "kb_policy_question",
    "cancellation_request",
    "expired_return",
    "wrong_item_sent",
    "duplicate_charge",
    "partial_order",
    "damaged_item",
    "angry_customer",
    "fraud_risk",
    "vip_warranty_claim",
]


def get_task_graders() -> Dict[str, Any]:
    """Return a per-task grader mapping for validator discovery."""
    return {tid: TaskGrader(default_task_id=tid) for tid in TASK_IDS}


# ---------------------------------------------------------------------------
# Module-level grade() shortcut — some validators call reward.grader.grade(...)
# ---------------------------------------------------------------------------

_default_grader = TaskGrader()


def grade(
    task_id: str | None = None,
    world_state: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> TaskGradeResult:
    return _default_grader.grade(task_id=task_id, world_state=world_state, **kwargs)

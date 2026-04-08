"""Compatibility wrapper matching the common `tasks/graders.py` layout."""

from __future__ import annotations

from typing import Any, Dict

from reward.grader import SCORE_RANGE, TaskGrader, get_task_graders


def clamp_score(score: float) -> float:
    """Keep task scores strictly within the declared score range."""
    lower, upper = SCORE_RANGE
    return round(min(max(float(score), lower), upper), 3)


def grade_action(task_id: str, action: str, signals: Dict[str, Any]) -> float:
    """
    Compatibility function for validators that expect tasks/graders.py::grade_action.

    We map the action into a minimal world-state-like structure and delegate to
    the main TaskGrader so the returned score is always valid and consistent.
    """
    normalized_action = (action or "").strip().lower()
    facts: Dict[str, Any] = {}
    resolution = "unresolved"

    if task_id in {"simple_refund", "wrong_item_sent", "duplicate_charge", "damaged_item"}:
        facts["refund_processed"] = normalized_action in {"buy", "refund", "resolve"}
        resolution = "resolved" if facts["refund_processed"] else "unresolved"
    elif task_id == "fraud_risk":
        facts["escalated"] = normalized_action in {"sell", "escalate"}
        resolution = "escalated" if facts["escalated"] else "unresolved"
    elif task_id in {"kb_policy_question", "expired_return", "vip_warranty_claim"}:
        facts["kb_searched"] = normalized_action in {"hold", "search", "resolved"}
        resolution = "resolved" if facts["kb_searched"] else "unresolved"
    else:
        resolution = "resolved" if normalized_action in {"hold", "resolved"} else "unresolved"

    result = TaskGrader(default_task_id=task_id).grade(
        task_id=task_id,
        world_state={
            "verified_facts": facts,
            "facts": facts,
            "signals": signals or {},
            "final_resolution": resolution,
        },
    )
    return clamp_score(result.score)


__all__ = ["TaskGrader", "get_task_graders", "grade_action", "clamp_score"]

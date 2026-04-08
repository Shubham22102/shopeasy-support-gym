"""Hackathon-compatible task grader adapter."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from grader import SupportTaskGrader, TASK_GRADERS, _strict_unit_score


class TaskGrader:
    """
    Compatibility adapter for validators that expect reward/grader.py::TaskGrader.

    Supports multiple calling conventions and always returns a score strictly
    between 0 and 1.
    """

    def __init__(self) -> None:
        self._grader = SupportTaskGrader()

    def grade(
        self,
        task_id: str | None = None,
        world_state: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Tuple[float, Dict[str, Any]]:
        state = world_state or {}
        task = (
            task_id
            or state.get("task_id")
            or state.get("current_task_id")
            or kwargs.get("task_id")
            or "simple_refund"
        )

        result = self._grader.grade(
            task_id=task,
            verified_facts=kwargs.get("verified_facts")
            or state.get("verified_facts")
            or state.get("facts")
            or {},
            conversation_history=kwargs.get("conversation_history")
            or state.get("conversation_history")
            or state.get("history")
            or [],
            final_resolution=kwargs.get("final_resolution")
            or kwargs.get("resolution")
            or state.get("close_resolution")
            or state.get("final_resolution")
            or "unresolved",
            order=kwargs.get("order") or state.get("order") or {},
            step_count=int(kwargs.get("step_count", state.get("step_count", 0))),
            max_steps=int(kwargs.get("max_steps", state.get("max_steps", 20))),
            customer_mood=float(
                kwargs.get("customer_mood", state.get("customer_mood", 0.0))
            ),
            agent_sent_messages=bool(
                kwargs.get(
                    "agent_sent_messages", state.get("agent_sent_messages", True)
                )
            ),
        )
        return _strict_unit_score(result.score), {
            "task_id": task,
            "passed": result.passed,
            "feedback": result.feedback,
        }


def get_task_graders() -> Dict[str, Any]:
    """Expose task graders as a mapping for validator discovery."""
    return dict(TASK_GRADERS)

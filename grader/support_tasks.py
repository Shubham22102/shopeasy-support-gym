"""
Per-task grader classes for OpenEnv validator discovery.
Each task in openenv.yaml references one of these classes.
The validator imports the class, instantiates it, and calls grade().
"""

from __future__ import annotations

import math
from typing import Any, Dict

# CRITICAL: Must be strictly 0 < score < 1, not inclusive!
# Using 0.02-0.98 range to be safe
_SCORE_MIN = 0.02
_SCORE_MAX = 0.98


def _safe_score(raw: float) -> float:
    """Clamp to open interval (0, 1) - strictly excluding 0.0 and 1.0."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return _SCORE_MIN
    score = float(raw)
    # Hard clamp to valid range
    score = max(_SCORE_MIN, min(_SCORE_MAX, score))
    return round(score, 3)


class BaseTaskGrader:
    """Base grader - subclasses override _score()."""
    
    task_id: str = "unknown"

    def grade(
        self,
        task_id: str | None = None,
        world_state: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[float, dict]:
        """
        Grade the task. Returns (score, info_dict).
        CRITICAL: Must return exactly 2 items: (float, dict)
        """
        state = world_state or {}
        
        facts = state.get("verified_facts") or state.get("facts") or {}
        resolution = (
            state.get("close_resolution")
            or state.get("final_resolution")
            or state.get("resolution")
            or "unresolved"
        )
        step_count = int(state.get("step_count", 5))
        max_steps = int(state.get("max_steps", 10))
        order = state.get("order") or {}
        mood = float(state.get("customer_mood", 0.0))

        # Calculate score
        raw_score = self._score(facts, resolution, order, step_count, max_steps, mood)
        score = _safe_score(raw_score)
        passed = score >= 0.5
        
        # Build info dict with required keys
        info = {
            "task_id": self.task_id,
            "score": score,
            "passed": passed,
            "feedback": f"Task '{self.task_id}' score: {score:.3f}",
            "score_range": [_SCORE_MIN, _SCORE_MAX],
        }
        
        # CRITICAL: Return exactly (score, info) - a 2-tuple!
        return score, info

    def _score(self, facts, resolution, order, step_count, max_steps, mood) -> float:
        return 0.5


class SimpleRefundGrader(BaseTaskGrader):
    task_id = "simple_refund"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.15
        if facts.get("order_looked_up"):
            score += 0.15
        if facts.get("refund_processed"):
            score += 0.40
        if resolution == "resolved":
            score += 0.20
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.08
        return score


class DeliveryTrackingGrader(BaseTaskGrader):
    task_id = "delivery_tracking"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.15
        if facts.get("order_looked_up"):
            score += 0.35
        if resolution == "resolved":
            score += 0.30
        if facts.get("refund_processed"):
            score -= 0.20
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.08
        return score


class KbPolicyQuestionGrader(BaseTaskGrader):
    task_id = "kb_policy_question"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.15
        if facts.get("kb_searched"):
            score += 0.40
        if resolution == "resolved":
            score += 0.25
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.08
        return score


class CancellationRequestGrader(BaseTaskGrader):
    task_id = "cancellation_request"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.15
        if facts.get("order_looked_up"):
            score += 0.20
        if resolution == "resolved":
            score += 0.40
        if facts.get("refund_processed"):
            score += 0.10
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.08
        return score


class ExpiredReturnGrader(BaseTaskGrader):
    task_id = "expired_return"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.10
        if facts.get("kb_searched"):
            score += 0.30
        if facts.get("order_looked_up"):
            score += 0.10
        if facts.get("refund_processed"):
            score -= 0.20
        if resolution == "resolved":
            score += 0.25
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.08
        return score


class WrongItemSentGrader(BaseTaskGrader):
    task_id = "wrong_item_sent"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.15
        if facts.get("order_looked_up"):
            score += 0.15
        if facts.get("refund_processed"):
            score += 0.35
        if resolution == "resolved":
            score += 0.20
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.08
        return score


class DuplicateChargeGrader(BaseTaskGrader):
    task_id = "duplicate_charge"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.10
        if facts.get("payment_checked"):
            score += 0.20
        if facts.get("refund_processed") and facts.get("payment_checked"):
            score += 0.40
        elif facts.get("refund_processed") and not facts.get("payment_checked"):
            score += 0.15
        if resolution == "resolved":
            score += 0.15
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.08
        return score


class PartialOrderGrader(BaseTaskGrader):
    task_id = "partial_order"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.15
        if facts.get("order_looked_up"):
            score += 0.15
        if facts.get("refund_processed"):
            score += 0.35
        if resolution == "resolved":
            score += 0.20
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.08
        return score


class DamagedItemGrader(BaseTaskGrader):
    task_id = "damaged_item"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.10
        if facts.get("order_looked_up"):
            score += 0.10
        if facts.get("refund_processed"):
            score += 0.45
        if resolution == "resolved":
            score += 0.20
        if facts.get("escalated") and not facts.get("refund_processed"):
            score -= 0.15
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.08
        return score


class AngryCustomerGrader(BaseTaskGrader):
    task_id = "angry_customer"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.10
        if facts.get("order_looked_up"):
            score += 0.15
        if resolution == "resolved" and not facts.get("escalated"):
            score += 0.50
        elif facts.get("escalated"):
            score += 0.15
        elif resolution == "resolved":
            score += 0.30
        if mood >= 0.5:
            score += 0.10
        elif mood < -0.5:
            score -= 0.05
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.08
        return score


class FraudRiskGrader(BaseTaskGrader):
    task_id = "fraud_risk"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.05
        if facts.get("escalated") and not facts.get("refund_processed"):
            score += 0.80
        elif facts.get("refund_processed"):
            score = 0.02  # Minimum valid score
        elif facts.get("escalated"):
            score += 0.30
        if facts.get("order_looked_up"):
            score += 0.05
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.05
        return score


class VipWarrantyClaimGrader(BaseTaskGrader):
    task_id = "vip_warranty_claim"

    def _score(self, facts, resolution, order, step_count, max_steps, mood):
        score = 0.10
        if facts.get("order_looked_up"):
            score += 0.10
        if facts.get("kb_searched"):
            score += 0.15
        if facts.get("is_vip") or order.get("is_vip"):
            score += 0.10
        has_warranty = facts.get("has_warranty") or order.get("has_warranty")
        if has_warranty and resolution == "resolved":
            score += 0.40
        elif resolution == "resolved":
            score += 0.20
        if max_steps > 0:
            score += max(0, 1 - step_count / max_steps) * 0.08
        return score
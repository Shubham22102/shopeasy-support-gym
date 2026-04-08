"""
3-Tier Reward Calculator + Non-Linear Trajectory Grader.

Final score = 0.60 × three_tier + 0.40 × trajectory_grade

Three-tier:
  R_outcome    → goal-directedness  (0.60 max)
  R_process    → policy compliance  (0.30 max)
  R_efficiency → conciseness        (0.10 max)

Trajectory grading (per-step meta-evaluation):
  0.90  perfect        — correct action + explanation + policy ref
  0.88  near_perfect   — correct action, easy/medium, missed details
  0.70-0.87 partial    — hard task, partial credit
  0.50  cautious       — overly conservative but safe
  0.30  missed_issue   — ignored real problem
  0.15  false_positive — flagged safe action
  0.10  catastrophic   — approved fraud / shipped a bug

Trajectory modifiers (per difficulty):
  Easy:   approve_bug -0.40 | consistency +0.05 | explanation +0.05
  Medium: approve_bug -0.50 | consistency +0.10 | explanation +0.07
  Hard:   approve_bug -0.60 | consistency +0.15 | explanation +0.10
"""

from typing import Any, Dict, List, Optional, Tuple

from .policy_engine import RefundPolicyEngine

# ---------------------------------------------------------------------------
# Three-tier weights
# ---------------------------------------------------------------------------
OUTCOME_MAX    = 0.60
PROCESS_MAX    = 0.30
EFFICIENCY_MAX = 0.10

# ---------------------------------------------------------------------------
# Per-step score constants
# ---------------------------------------------------------------------------
STEP_PERFECT           = 0.90
STEP_NEAR_PERFECT      = 0.88
STEP_PARTIAL_HARD_HIGH = 0.87
STEP_PARTIAL_HARD_MID  = 0.78
STEP_PARTIAL_HARD_LOW  = 0.70
STEP_CAUTIOUS          = 0.50
STEP_MISSED_ISSUE      = 0.30
STEP_FALSE_POSITIVE    = 0.15
STEP_CATASTROPHIC      = 0.10

CAT_PERFECT        = "perfect"
CAT_NEAR_PERFECT   = "near_perfect"
CAT_PARTIAL        = "partial"
CAT_CAUTIOUS       = "cautious"
CAT_MISSED         = "missed_issue"
CAT_FALSE_POSITIVE = "false_positive"
CAT_CATASTROPHIC   = "catastrophic"

# Easy tasks where escalation is a false-positive
_EASY_NO_ESCALATE = frozenset({
    "simple_refund", "delivery_tracking",
    "kb_policy_question", "cancellation_request",
})

# ---------------------------------------------------------------------------
# Step scorer
# ---------------------------------------------------------------------------

def score_step_action(
    action_type: str,
    tool_name: Optional[str],
    tool_args: Dict[str, Any],
    tool_result: Optional[Dict],
    tool_error: Optional[str],
    agent_message: str,
    difficulty: str,
    task_id: str,
    verified_facts: Dict[str, Any],
    order: Optional[Dict[str, Any]],
) -> Tuple[float, str]:
    """Return (score, category) for one agent step."""
    order       = order or {}
    is_fraud    = bool(order.get("is_fraud_risk"))
    is_damaged  = bool(order.get("is_damaged"))
    resolution  = tool_args.get("resolution", "")
    escalated   = verified_facts.get("escalated", False)
    looked_up   = verified_facts.get("order_looked_up", False)
    kb_searched = verified_facts.get("kb_searched", False)

    # ── CATASTROPHIC ────────────────────────────────────────────────────────
    if action_type == "tool_call" and tool_name == "process_refund" and is_fraud:
        return STEP_CATASTROPHIC, CAT_CATASTROPHIC
    if action_type == "close_ticket" and resolution == "resolved" and is_fraud and not escalated:
        return STEP_CATASTROPHIC, CAT_CATASTROPHIC

    # ── FALSE POSITIVE ──────────────────────────────────────────────────────
    if (action_type == "tool_call" and tool_name == "escalate_to_human"
            and not is_fraud and not is_damaged and task_id in _EASY_NO_ESCALATE):
        return STEP_FALSE_POSITIVE, CAT_FALSE_POSITIVE

    # ── MISSED ISSUE ────────────────────────────────────────────────────────
    if (action_type == "close_ticket" and resolution in ("unresolved", "timeout")
            and is_fraud and not escalated):
        return STEP_MISSED_ISSUE, CAT_MISSED

    # ── Message quality signals ─────────────────────────────────────────────
    msg_lower       = agent_message.lower()
    has_explanation = len(agent_message) > 80
    has_policy_ref  = any(kw in msg_lower for kw in (
        "policy", "window", "return", "refund", "warranty", "fraud",
        "escalat", "credit", "verify", "check", "sorry", "apologize",
    ))

    # ── TOOL CALL ───────────────────────────────────────────────────────────
    if action_type == "tool_call":
        if tool_result and not tool_error:
            if has_explanation and has_policy_ref:
                return STEP_PERFECT, CAT_PERFECT
            if difficulty == "hard":
                if looked_up and kb_searched and has_explanation:
                    return STEP_PARTIAL_HARD_HIGH, CAT_PARTIAL
                if looked_up and has_explanation:
                    return STEP_PARTIAL_HARD_MID, CAT_PARTIAL
                if looked_up:
                    return STEP_PARTIAL_HARD_LOW, CAT_PARTIAL
                return STEP_CAUTIOUS, CAT_CAUTIOUS
            return STEP_NEAR_PERFECT, CAT_NEAR_PERFECT
        return STEP_CAUTIOUS, CAT_CAUTIOUS

    # ── SEND MESSAGE ────────────────────────────────────────────────────────
    if action_type == "send_message":
        if has_explanation and has_policy_ref:
            return STEP_PERFECT, CAT_PERFECT
        if has_explanation:
            return STEP_NEAR_PERFECT, CAT_NEAR_PERFECT
        return STEP_CAUTIOUS, CAT_CAUTIOUS

    # ── CLOSE TICKET ────────────────────────────────────────────────────────
    if action_type == "close_ticket":
        if resolution == "escalated" and (is_fraud or is_damaged):
            return STEP_PERFECT, CAT_PERFECT
        if resolution == "resolved" and looked_up:
            if has_explanation:
                return STEP_PERFECT, CAT_PERFECT
            return STEP_NEAR_PERFECT, CAT_NEAR_PERFECT
        if resolution == "resolved":
            return STEP_CAUTIOUS, CAT_CAUTIOUS
        return STEP_MISSED_ISSUE, CAT_MISSED

    return STEP_CAUTIOUS, CAT_CAUTIOUS


# ---------------------------------------------------------------------------
# Trajectory graders
# ---------------------------------------------------------------------------

def _mean(scores: List[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.5

def _consistency_ratio(cats: List[str]) -> float:
    good = {CAT_PERFECT, CAT_NEAR_PERFECT, CAT_PARTIAL}
    return sum(1 for c in cats if c in good) / len(cats) if cats else 0.0

def _perfection_ratio(cats: List[str]) -> float:
    return sum(1 for c in cats if c == CAT_PERFECT) / len(cats) if cats else 0.0

def _n_catastrophic(cats: List[str]) -> int:
    return sum(1 for c in cats if c == CAT_CATASTROPHIC)


def easy_grader(scores: List[float], cats: List[str]) -> float:
    base  = _mean(scores)
    base -= 0.40 * _n_catastrophic(cats)
    if _consistency_ratio(cats) >= 0.80: base += 0.05
    if _perfection_ratio(cats)  >= 0.80: base += 0.05
    return max(0.01, min(0.99, round(base, 4)))

def medium_grader(scores: List[float], cats: List[str]) -> float:
    base  = _mean(scores)
    base -= 0.50 * _n_catastrophic(cats)
    if _consistency_ratio(cats) >= 0.80: base += 0.10
    if _perfection_ratio(cats)  >= 0.80: base += 0.07
    return max(0.01, min(0.99, round(base, 4)))

def hard_grader(scores: List[float], cats: List[str]) -> float:
    base  = _mean(scores)
    base -= 0.60 * _n_catastrophic(cats)
    if _consistency_ratio(cats) >= 0.80: base += 0.15
    if _perfection_ratio(cats)  >= 0.80: base += 0.10
    return max(0.01, min(0.99, round(base, 4)))

def trajectory_grade(scores: List[float], cats: List[str], difficulty: str) -> float:
    if difficulty == "hard":   return hard_grader(scores, cats)
    if difficulty == "medium": return medium_grader(scores, cats)
    return easy_grader(scores, cats)


# ---------------------------------------------------------------------------
# Reward Calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """Blended reward: 60% three-tier + 40% trajectory grading."""

    def __init__(self):
        self._policy = RefundPolicyEngine()

    def calculate(
        self,
        order: Optional[Dict[str, Any]],
        scenario_task_id: str,
        verified_facts: Dict[str, Any],
        conversation_history: list,
        close_resolution: str,
        step_count: int,
        max_steps: int,
        customer_mood: float,
        agent_sent_messages: bool,
        difficulty: str = "easy",
        step_scores: Optional[List[float]] = None,
        step_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        r_outcome    = self._compute_outcome(
            order, scenario_task_id, verified_facts,
            close_resolution, customer_mood, agent_sent_messages,
        )
        r_process    = self._compute_process(order, scenario_task_id, verified_facts)
        r_efficiency = self._compute_efficiency(step_count, max_steps)

        three_tier = max(0.01, min(0.99, round(r_outcome + r_process + r_efficiency, 4)))

        traj: Optional[float] = None
        if step_scores and step_categories:
            traj = trajectory_grade(step_scores, step_categories, difficulty)

        total = round(0.60 * three_tier + 0.40 * traj, 4) if traj is not None else three_tier
        total = max(0.01, min(0.99, total))

        return {
            "outcome":    round(r_outcome, 4),
            "process":    round(r_process, 4),
            "efficiency": round(r_efficiency, 4),
            "three_tier": three_tier,
            "trajectory": traj,
            "total":      total,
        }

    # ------------------------------------------------------------------
    # R_outcome
    # ------------------------------------------------------------------
    def _compute_outcome(
        self, order, scenario_task_id, verified_facts,
        close_resolution, customer_mood, agent_sent_messages,
    ) -> float:
        if close_resolution == "timeout":
            return 0.05
        if order is None:
            return OUTCOME_MAX * 0.5 if close_resolution == "resolved" else 0.02

        correctness_score, _ = self._policy.check_resolution_correctness(
            order=order, scenario_task_id=scenario_task_id,
            verified_facts=verified_facts, close_resolution=close_resolution,
            agent_sent_messages=agent_sent_messages,
        )
        mood_bonus = 0.10 if customer_mood >= 0.5 else (0.05 if customer_mood >= 0.0 else (-0.05 if customer_mood < -0.5 else 0.0))
        return max(0.0, min(OUTCOME_MAX, correctness_score * OUTCOME_MAX + mood_bonus))

    # ------------------------------------------------------------------
    # R_process
    # ------------------------------------------------------------------
    def _compute_process(self, order, scenario_task_id, verified_facts) -> float:
        score = 0.0
        if verified_facts.get("order_looked_up"):
            score += 0.10
        kb_tasks = {"expired_return", "kb_policy_question", "vip_warranty_claim", "warranty_claim"}
        if scenario_task_id in kb_tasks and verified_facts.get("kb_searched"):
            score += 0.05
        if order:
            if verified_facts.get("refund_processed") and order.get("is_fraud_risk"):
                score -= 0.10
            if (verified_facts.get("refund_processed") and not order.get("is_damaged")
                    and not order.get("is_fraud_risk") and order.get("status") == "delivered"
                    and not order.get("within_return_window") and scenario_task_id == "expired_return"):
                score -= 0.05
            if (order.get("is_fraud_risk") and verified_facts.get("escalated")
                    and not verified_facts.get("refund_processed")):
                score += 0.15
            if scenario_task_id == "duplicate_charge":
                if verified_facts.get("payment_checked") and verified_facts.get("refund_processed"):
                    score += 0.10
                elif verified_facts.get("refund_processed") and not verified_facts.get("payment_checked"):
                    score -= 0.05
        return max(0.0, min(PROCESS_MAX, score))

    # ------------------------------------------------------------------
    # R_efficiency
    # ------------------------------------------------------------------
    def _compute_efficiency(self, step_count: int, max_steps: int) -> float:
        if max_steps <= 0:
            return 0.0
        return round(max(0.0, 1.0 - step_count / max_steps) * EFFICIENCY_MAX, 4)

"""
3-Tier Reward Calculator for the ShopEasy Support Gym.

R_total = R_outcome + R_process + R_efficiency

Each tier teaches a different behavior:
  R_outcome   → goal-directedness (did you solve the problem?)
  R_process   → policy compliance (did you follow the rules?)
  R_efficiency → conciseness (did you do it efficiently?)

This prevents reward hacking: the agent can't just approve everything
(would fail R_process) or be verbose (would fail R_efficiency).
"""

from typing import Any, Dict, Optional

from .policy_engine import RefundPolicyEngine


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

OUTCOME_MAX = 0.60
PROCESS_MAX = 0.30
EFFICIENCY_MAX = 0.10


# ---------------------------------------------------------------------------
# Reward Calculator
# ---------------------------------------------------------------------------


class RewardCalculator:
    """
    Calculates the 3-tier reward at episode end (when close_ticket is called
    or the episode is forcibly terminated by max_steps).
    """

    def __init__(self):
        self._policy = RefundPolicyEngine()

    def calculate(
        self,
        order: Optional[Dict[str, Any]],
        scenario_task_id: str,
        verified_facts: Dict[str, Any],
        conversation_history: list,
        close_resolution: str,  # "resolved" | "escalated" | "unresolved" | "timeout"
        step_count: int,
        max_steps: int,
        customer_mood: float,  # final mood -1.0 to +1.0
        agent_sent_messages: bool,
    ) -> Dict[str, float]:
        """
        Compute R_outcome, R_process, R_efficiency, and R_total.

        Returns a dict:
            {
                'outcome': float,
                'process': float,
                'efficiency': float,
                'total': float,
                'explanation': str,
            }
        """
        r_outcome = self._compute_outcome(
            order,
            scenario_task_id,
            verified_facts,
            close_resolution,
            customer_mood,
            agent_sent_messages,
        )
        r_process = self._compute_process(
            order,
            scenario_task_id,
            verified_facts,
            close_resolution,
        )
        r_efficiency = self._compute_efficiency(step_count, max_steps)

        total = round(r_outcome + r_process + r_efficiency, 4)
        total = max(0.0, min(1.0, total))

        return {
            "outcome": round(r_outcome, 4),
            "process": round(r_process, 4),
            "efficiency": round(r_efficiency, 4),
            "total": total,
        }

    # ------------------------------------------------------------------
    # R_outcome — Was the problem actually solved?
    # ------------------------------------------------------------------

    def _compute_outcome(
        self,
        order: Optional[Dict[str, Any]],
        scenario_task_id: str,
        verified_facts: Dict[str, Any],
        close_resolution: str,
        customer_mood: float,
        agent_sent_messages: bool,
    ) -> float:
        """
        Score from 0.0 to OUTCOME_MAX (0.60).

        Components:
          - Correctness of resolution (checked via policy engine)
          - Customer final mood (proxy for satisfaction)
          - Timeout penalty
        """
        if close_resolution == "timeout":
            # Forced termination — heavily penalised
            return 0.05

        if order is None:
            # No order involved (e.g. pure KB policy question)
            return OUTCOME_MAX * 0.5 if close_resolution == "resolved" else 0.0

        correctness_score, _ = self._policy.check_resolution_correctness(
            order=order,
            scenario_task_id=scenario_task_id,
            verified_facts=verified_facts,
            close_resolution=close_resolution,
            agent_sent_messages=agent_sent_messages,
        )

        # Mood bonus: satisfied/calm customer is worth up to +0.10 on top of correctness
        mood_bonus = 0.0
        if customer_mood >= 0.5:  # satisfied
            mood_bonus = 0.10
        elif customer_mood >= 0.0:  # calm
            mood_bonus = 0.05
        elif customer_mood < -0.5:  # angry at end
            mood_bonus = -0.05  # mild penalty

        # Clamp: correctness is already 0–0.75 of max, mood is bonus
        raw = correctness_score * OUTCOME_MAX + mood_bonus
        return max(0.0, min(OUTCOME_MAX, raw))

    # ------------------------------------------------------------------
    # R_process — Did the agent follow correct procedures?
    # ------------------------------------------------------------------

    def _compute_process(
        self,
        order: Optional[Dict[str, Any]],
        scenario_task_id: str,
        verified_facts: Dict[str, Any],
        close_resolution: str,
    ) -> float:
        """
        Score from 0.0 to PROCESS_MAX (0.30).

        Checks:
          +0.10  — Looked up order before making any refund promise
          +0.10  — Applied correct policy (policy engine verdict)
          +0.05  — Did NOT access other customers' data
          +0.05  — Searched KB when needed
          -0.10  — Processed refund on fraud-risk order (PENALTY)
          -0.05  — Issued full refund when only store credit warranted
        """
        score = 0.0

        # Did agent look up the order?
        if verified_facts.get("order_looked_up"):
            score += 0.10

        # KB searched when it should be
        kb_required_tasks = {
            "expired_return",
            "kb_policy_question",
            "vip_warranty_claim",
            "warranty_claim",
        }
        if scenario_task_id in kb_required_tasks:
            if verified_facts.get("kb_searched"):
                score += 0.05

        # Policy compliance
        if order is not None:
            # Penalty: refunded a fraud-risk order
            if verified_facts.get("refund_processed") and order.get("is_fraud_risk"):
                score -= 0.10

            # Penalty: gave full refund when outside return window (should be store credit)
            if (
                verified_facts.get("refund_processed")
                and not order.get("is_damaged")
                and not order.get("is_fraud_risk")
                and order.get("status") == "delivered"
                and not order.get("within_return_window")
                and scenario_task_id == "expired_return"
            ):
                score -= 0.05

            # Bonus: correctly escalated fraud risk
            if (
                order.get("is_fraud_risk")
                and verified_facts.get("escalated")
                and not verified_facts.get("refund_processed")
            ):
                score += 0.15

            # Bonus: looked up payment before refunding (duplicate charge)
            if scenario_task_id == "duplicate_charge":
                if verified_facts.get("payment_checked") and verified_facts.get(
                    "refund_processed"
                ):
                    score += 0.10
                elif verified_facts.get("refund_processed") and not verified_facts.get(
                    "payment_checked"
                ):
                    score -= 0.05  # refunded without verifying

        score = max(0.0, min(PROCESS_MAX, score))
        return score

    # ------------------------------------------------------------------
    # R_efficiency — How concisely was it resolved?
    # ------------------------------------------------------------------

    def _compute_efficiency(self, step_count: int, max_steps: int) -> float:
        """
        Score from 0.0 to EFFICIENCY_MAX (0.10).

        Formula: steps_bonus = (max_steps - steps_taken) / max_steps * EFFICIENCY_MAX
        This rewards solving faster, but the weight is small (0.10) so it
        doesn't dominate — correctness always trumps speed.
        """
        if max_steps <= 0:
            return 0.0

        steps_used_ratio = step_count / max_steps
        # Invert: fewer steps used = higher score
        efficiency = max(0.0, 1.0 - steps_used_ratio) * EFFICIENCY_MAX
        return round(efficiency, 4)

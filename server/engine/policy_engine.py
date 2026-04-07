"""
Refund Policy Engine for the ShopEasy Support Gym.

Encodes the real business rules that determine whether a refund,
store credit, replacement, or denial is the correct resolution.
Used by the reward engine to evaluate whether the agent applied
the correct policy.
"""

from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Policy result type
# ---------------------------------------------------------------------------

class PolicyResult:
    """
    The verdict from evaluating a refund/resolution request against policy.

    Attributes:
        allowed        : True if the action is permitted by policy
        correct_action : what the agent SHOULD do: 'full_refund' | 'store_credit' |
                         'replacement' | 'escalate' | 'deny' | 'warranty_claim'
        reason         : human-readable explanation of the policy ruling
        penalty        : True if the agent's action VIOLATES policy (e.g., refunding fraud)
    """

    def __init__(
        self,
        allowed: bool,
        correct_action: str,
        reason: str,
        penalty: bool = False,
    ):
        self.allowed = allowed
        self.correct_action = correct_action
        self.reason = reason
        self.penalty = penalty


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------

class RefundPolicyEngine:
    """
    Evaluates whether a proposed resolution is correct per ShopEasy policy.

    Used by the reward calculator to determine partial/full credit for
    R_process (did the agent follow the right rules?).
    """

    def evaluate_refund(
        self,
        order: Dict[str, Any],
        reason: str,
    ) -> PolicyResult:
        """
        Given order details and the stated reason, determine the correct refund action.

        This is the ground-truth policy against which agent behavior is graded.
        """
        # Rule 0: Already refunded → deny
        if order.get("refund_issued"):
            return PolicyResult(
                allowed=False,
                correct_action="deny",
                reason="A refund has already been issued for this order.",
                penalty=False,
            )

        # Rule 1: Fraud risk → must escalate, NEVER refund directly
        if order.get("is_fraud_risk"):
            return PolicyResult(
                allowed=False,
                correct_action="escalate",
                reason=(
                    "This order is flagged for fraud risk review. "
                    "Refunds must not be processed directly; escalate to the fraud team."
                ),
                penalty=True,  # Processing refund here is a policy violation
            )

        # Rule 2: Damaged item → instant full refund (no window check)
        if order.get("is_damaged"):
            return PolicyResult(
                allowed=True,
                correct_action="full_refund",
                reason="Item arrived damaged. Instant full refund applies with no return window restriction.",
            )

        # Rule 3: Wrong item scenario (handled by wrong_item flag in reason)
        reason_lower = reason.lower()
        if any(kw in reason_lower for kw in ("wrong item", "incorrect item", "wrong product")):
            return PolicyResult(
                allowed=True,
                correct_action="full_refund",
                reason="Wrong item received. Full refund applies regardless of return window.",
            )

        # Rule 4: Status = lost_in_transit → full refund
        if order.get("status") == "lost_in_transit":
            return PolicyResult(
                allowed=True,
                correct_action="full_refund",
                reason="Order lost in transit. Automatic full refund.",
            )

        # Rule 5: Status = pending → cancel, not refund
        if order.get("status") == "pending":
            return PolicyResult(
                allowed=True,
                correct_action="cancellation",
                reason="Order is still pending. Should be cancelled, not refunded.",
            )

        # Rule 6: Status = shipped → cannot refund until delivered
        if order.get("status") == "shipped":
            return PolicyResult(
                allowed=False,
                correct_action="deny",
                reason="Order is in transit. Cannot process refund until delivered.",
            )

        # Rule 7: Status = cancelled → deny (already cancelled)
        if order.get("status") == "cancelled":
            return PolicyResult(
                allowed=False,
                correct_action="deny",
                reason="Order was already cancelled.",
            )

        # Rule 8: Delivered + within return window → full refund
        if order.get("status") == "delivered" and order.get("within_return_window"):
            return PolicyResult(
                allowed=True,
                correct_action="full_refund",
                reason="Order delivered and within return window. Full refund permitted.",
            )

        # Rule 9: Delivered + outside return window → store credit only
        if order.get("status") == "delivered" and not order.get("within_return_window"):
            return PolicyResult(
                allowed=True,
                correct_action="store_credit",
                reason=(
                    f"Return window of {order.get('return_window_days', '?')} days has expired "
                    f"({order.get('days_since_delivery', '?')} days since delivery). "
                    "Only store credit can be offered."
                ),
            )

        # Default: needs more information
        return PolicyResult(
            allowed=False,
            correct_action="lookup_required",
            reason="Insufficient order information to determine policy. Use lookup_order first.",
        )

    def evaluate_escalation(
        self,
        order: Dict[str, Any],
        issue_type: str,
        agent_tried_resolution: bool,
    ) -> PolicyResult:
        """
        Determine if escalating to a human is the correct action.

        Escalation is REQUIRED for: fraud risk, threats, refunds > ₹50k.
        Escalation is PENALIZED for: issues that could be self-resolved.
        """
        if order.get("is_fraud_risk"):
            return PolicyResult(
                allowed=True,
                correct_action="escalate",
                reason="Fraud-risk order requires human specialist review.",
            )

        if order.get("total", 0) > 50000:
            return PolicyResult(
                allowed=True,
                correct_action="escalate",
                reason="Refund amount exceeds ₹50,000 threshold; requires human authorization.",
            )

        # Escalating a simple refunable issue without trying = penalty
        if not agent_tried_resolution and issue_type in (
            "refund_request", "delivery_inquiry", "policy_inquiry", "cancellation"
        ):
            return PolicyResult(
                allowed=False,
                correct_action="resolve_directly",
                reason=(
                    "This issue can be resolved under standard policy. "
                    "Escalating without attempting resolution increases customer wait time."
                ),
                penalty=True,
            )

        return PolicyResult(
            allowed=True,
            correct_action="escalate",
            reason="Escalation permitted for this issue type.",
        )

    def check_resolution_correctness(
        self,
        order: Dict[str, Any],
        scenario_task_id: str,
        verified_facts: Dict[str, Any],
        close_resolution: str,
        agent_sent_messages: bool,
    ) -> Tuple[float, str]:
        """
        Evaluate the overall resolution quality.

        Returns:
            (correctness_score 0.0–1.0, explanation str)
        """
        score = 0.0
        reasons = []

        # Did agent lookup order before acting?
        if verified_facts.get("order_looked_up"):
            score += 0.15
            reasons.append("✓ Order verified before resolution")
        else:
            reasons.append("✗ Agent never looked up order (policy violation)")

        # Did agent send at least one message to the customer?
        if agent_sent_messages:
            score += 0.10
            reasons.append("✓ Agent communicated with customer")
        else:
            reasons.append("✗ Agent closed ticket without messaging customer")

        # Scenario-specific checks
        task_checks = {
            "simple_refund": self._check_simple_refund,
            "delivery_tracking": self._check_delivery_tracking,
            "expired_return": self._check_expired_return,
            "wrong_item_sent": self._check_wrong_item,
            "duplicate_charge": self._check_duplicate_charge,
            "damaged_item": self._check_damaged_item,
            "fraud_risk": self._check_fraud,
            "angry_customer": self._check_angry_customer,
            "vip_warranty_claim": self._check_vip_warranty,
        }

        checker = task_checks.get(scenario_task_id)
        if checker:
            task_score, task_reason = checker(order, verified_facts, close_resolution)
            score += task_score
            reasons.append(task_reason)
        else:
            # Generic check: was it resolved?
            if close_resolution == "resolved":
                score += 0.50
                reasons.append("✓ Ticket closed as resolved")

        return min(1.0, score), " | ".join(reasons)

    # ------------------------------------------------------------------
    # Scenario-specific resolution checkers
    # ------------------------------------------------------------------

    def _check_simple_refund(self, order, facts, resolution) -> Tuple[float, str]:
        if facts.get("refund_processed") and resolution == "resolved":
            return 0.60, "✓ Correct: refund processed and ticket resolved"
        if not facts.get("refund_processed"):
            return 0.10, "✗ Refund not processed"
        return 0.30, "~ Partially correct"

    def _check_delivery_tracking(self, order, facts, resolution) -> Tuple[float, str]:
        if facts.get("order_looked_up") and resolution == "resolved":
            return 0.60, "✓ Order status shared and ticket closed"
        if facts.get("refund_processed"):
            return 0.0, "✗ Penalty: processed refund for a tracking inquiry (unnecessary)"
        return 0.20, "~ Order looked up but ticket not properly closed"

    def _check_expired_return(self, order, facts, resolution) -> Tuple[float, str]:
        if facts.get("refund_processed") and not order.get("within_return_window"):
            return -0.20, "✗ Penalty: processed full refund outside return window (should offer store credit)"
        if facts.get("kb_searched") and not facts.get("refund_processed"):
            return 0.55, "✓ Correctly offered store credit / denied full refund per policy"
        return 0.20, "~ Partial credit: checked some facts but resolution unclear"

    def _check_wrong_item(self, order, facts, resolution) -> Tuple[float, str]:
        if facts.get("refund_processed") and resolution == "resolved":
            return 0.60, "✓ Correct: full refund for wrong item"
        return 0.10, "✗ Wrong item case not fully resolved"

    def _check_duplicate_charge(self, order, facts, resolution) -> Tuple[float, str]:
        if facts.get("payment_checked") and facts.get("refund_processed"):
            return 0.60, "✓ Correct: payment verified then refund processed"
        if facts.get("refund_processed") and not facts.get("payment_checked"):
            return 0.25, "~ Refund issued but payment not verified first (should check_payment first)"
        return 0.10, "✗ Duplicate charge case not resolved"

    def _check_damaged_item(self, order, facts, resolution) -> Tuple[float, str]:
        if facts.get("refund_processed") and order.get("is_damaged") and resolution == "resolved":
            return 0.65, "✓ Correct: instant refund for damaged item"
        if facts.get("escalated") and not facts.get("refund_processed"):
            return -0.10, "✗ Penalty: escalated instead of resolving directly (damaged items are agent-resolvable)"
        return 0.10, "✗ Damaged item not correctly resolved"

    def _check_fraud(self, order, facts, resolution) -> Tuple[float, str]:
        if facts.get("escalated") and not facts.get("refund_processed"):
            return 0.65, "✓ Correct: escalated fraud risk without issuing refund"
        if facts.get("refund_processed") and order.get("is_fraud_risk"):
            return -0.30, "✗ CRITICAL PENALTY: issued refund on fraud-risk order"
        return 0.0, "✗ Fraud case not correctly handled"

    def _check_angry_customer(self, order, facts, resolution) -> Tuple[float, str]:
        if facts.get("order_looked_up") and resolution == "resolved" and not facts.get("escalated"):
            return 0.60, "✓ De-escalated and resolved without unnecessary escalation"
        if facts.get("escalated") and not order.get("is_fraud_risk"):
            return 0.20, "~ Resolved but chose to escalate (slightly penalized for unnecessary escalation)"
        return 0.10, "✗ Angry customer case not resolved"

    def _check_vip_warranty(self, order, facts, resolution) -> Tuple[float, str]:
        score = 0.0
        parts = []
        if facts.get("order_looked_up") and facts.get("is_vip"):
            score += 0.10
            parts.append("✓ VIP status identified")
        if facts.get("kb_searched"):
            score += 0.20
            parts.append("✓ KB searched for warranty terms")
        if facts.get("order_looked_up") and order.get("has_warranty") and resolution == "resolved":
            score += 0.30
            parts.append("✓ Warranty claim resolved")
        return score, " | ".join(parts) if parts else "✗ VIP warranty case not handled"

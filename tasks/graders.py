# tasks/graders.py
# Reward functions for each task. All rewards are strictly in (0.0, 1.0).
#
# IMPORTANT: This module must be fully self-contained — NO imports from
# reward/, server/, or any project-specific modules. The OpenEnv validator
# imports this file in isolation.


def grade_action(task_id: str, action: str, signals: dict) -> float:
    """
    Score a single action for a given task and signal state.
    Returns a float strictly in (0.0, 1.0).
    """
    action = action.lower().strip()

    # Normalise common action variants the LLM might emit
    ACTION_ALIASES = {
        "buy": "refund", "approve": "resolve", "cancel": "resolve",
        "search": "search_kb", "kb": "search_kb", "lookup": "lookup_order",
        "look_up": "lookup_order", "check": "lookup_order",
        "wait": "hold", "defer": "hold", "transfer": "escalate",
    }
    if action not in ("resolve", "escalate", "refund", "search_kb", "lookup_order", "hold"):
        action = ACTION_ALIASES.get(action, action)
        # Try partial match as last resort
        if action not in ("resolve", "escalate", "refund", "search_kb", "lookup_order", "hold"):
            for a in ("resolve", "escalate", "refund", "search_kb", "lookup_order", "hold"):
                if a in action:
                    action = a
                    break
            else:
                return 0.10  # unrecognised action — small non-zero

    raw_score = 0.5  # default for unknown tasks

    if task_id == "simple_refund":
        raw_score = _grade_simple_refund(action, signals)
    elif task_id == "delivery_tracking":
        raw_score = _grade_delivery_tracking(action, signals)
    elif task_id == "kb_policy_question":
        raw_score = _grade_kb_policy(action, signals)
    elif task_id == "cancellation_request":
        raw_score = _grade_cancellation(action, signals)
    elif task_id == "expired_return":
        raw_score = _grade_expired_return(action, signals)
    elif task_id == "wrong_item_sent":
        raw_score = _grade_wrong_item(action, signals)
    elif task_id == "duplicate_charge":
        raw_score = _grade_duplicate_charge(action, signals)
    elif task_id == "partial_order":
        raw_score = _grade_partial_order(action, signals)
    elif task_id == "damaged_item":
        raw_score = _grade_damaged_item(action, signals)
    elif task_id == "angry_customer":
        raw_score = _grade_angry_customer(action, signals)
    elif task_id == "fraud_risk":
        raw_score = _grade_fraud_risk(action, signals)
    elif task_id == "vip_warranty_claim":
        raw_score = _grade_vip_warranty(action, signals)

    # OpenEnv requirement: scores must be strictly in (0.0, 1.0)
    return round(min(max(raw_score, 0.01), 0.99), 3)


# ---------------------------------------------------------------------------
# Task-specific graders
# ---------------------------------------------------------------------------

def _grade_simple_refund(action: str, signals: dict) -> float:
    """Within-window refund. Correct: refund."""
    if action == "refund":
        return 0.95
    elif action == "resolve":
        return 0.60  # acceptable but not ideal
    elif action == "lookup_order":
        return 0.40  # partial — looked up but didn't refund
    elif action == "escalate":
        return 0.20  # unnecessary escalation
    return 0.10


def _grade_delivery_tracking(action: str, signals: dict) -> float:
    """Customer asking where order is. Correct: lookup_order."""
    if action == "lookup_order":
        return 0.95
    elif action == "resolve":
        return 0.55  # resolved without looking up tracking
    elif action == "refund":
        return 0.10  # refunding a tracking question is wrong
    elif action == "search_kb":
        return 0.30
    return 0.15


def _grade_kb_policy(action: str, signals: dict) -> float:
    """Policy question. Correct: search_kb."""
    if action == "search_kb":
        return 0.95
    elif action == "resolve":
        return 0.55  # resolved but didn't cite policy
    elif action == "hold":
        return 0.20
    return 0.10


def _grade_cancellation(action: str, signals: dict) -> float:
    """Cancel a pending order. Correct: resolve."""
    cancellable = signals.get("cancellation_eligible", True)
    if action == "resolve" and cancellable:
        return 0.95
    elif action == "resolve":
        return 0.70
    elif action == "refund":
        return 0.50  # refund instead of cancel — partial
    elif action == "lookup_order":
        return 0.35
    elif action == "escalate":
        return 0.20
    return 0.10


def _grade_expired_return(action: str, signals: dict) -> float:
    """Expired return — must check policy, not blindly refund. Correct: search_kb."""
    if action == "search_kb":
        return 0.90  # checked policy — correct approach
    elif action == "resolve":
        return 0.60
    elif action == "refund":
        # Refunding outside the window is a policy violation
        return 0.15
    elif action == "hold":
        return 0.30
    elif action == "escalate":
        return 0.45  # reasonable to escalate edge case
    return 0.10


def _grade_wrong_item(action: str, signals: dict) -> float:
    """Wrong item sent. Correct: refund."""
    if action == "refund":
        return 0.95
    elif action == "resolve":
        return 0.55
    elif action == "escalate":
        return 0.40
    elif action == "lookup_order":
        return 0.35
    return 0.10


def _grade_duplicate_charge(action: str, signals: dict) -> float:
    """Double charged. Correct: refund (the duplicate)."""
    payment_verified = signals.get("payment_verified", False)
    if action == "refund" and payment_verified:
        return 0.95
    elif action == "refund" and not payment_verified:
        return 0.70  # refunded without verifying — risky but correct intent
    elif action == "lookup_order":
        return 0.45  # investigating is reasonable
    elif action == "resolve":
        return 0.40
    elif action == "escalate":
        return 0.35
    return 0.10


def _grade_partial_order(action: str, signals: dict) -> float:
    """Partial delivery. Correct: refund (missing items)."""
    if action == "refund":
        return 0.90
    elif action == "resolve":
        return 0.55
    elif action == "lookup_order":
        return 0.40
    elif action == "escalate":
        return 0.35
    return 0.10


def _grade_damaged_item(action: str, signals: dict) -> float:
    """Damaged item — instant refund policy. Correct: refund."""
    is_damaged = signals.get("is_damaged", True)
    if action == "refund" and is_damaged:
        return 0.95
    elif action == "refund":
        return 0.75
    elif action == "escalate":
        return 0.30  # unnecessary escalation for clear damage case
    elif action == "resolve":
        return 0.45
    elif action == "lookup_order":
        return 0.30
    return 0.10


def _grade_angry_customer(action: str, signals: dict) -> float:
    """Angry customer — de-escalate and resolve. Correct: resolve."""
    sentiment = signals.get("customer_sentiment", -0.5)
    if action == "resolve":
        # Better score if the customer was very angry (harder case handled well)
        anger_bonus = max(0.0, -sentiment) * 0.15
        return min(0.90 + anger_bonus, 0.98)
    elif action == "escalate":
        return 0.35  # escalating an angry customer = passing the buck
    elif action == "refund":
        return 0.50  # might help but doesn't address the anger
    elif action == "hold":
        return 0.15  # worst — making angry customer wait
    return 0.10


def _grade_fraud_risk(action: str, signals: dict) -> float:
    """Fraud risk — must escalate, NOT refund. Correct: escalate."""
    fraud_score = signals.get("fraud_score", 0.8)
    if action == "escalate":
        return min(0.70 + 0.25 * fraud_score, 0.98)
    elif action == "hold":
        return 0.40  # at least didn't refund
    elif action == "lookup_order":
        return 0.35  # investigating is OK-ish
    elif action == "resolve":
        return 0.20  # resolved without escalating fraud
    elif action == "refund":
        return 0.02  # WORST: refunding a fraudster
    return 0.10


def _grade_vip_warranty(action: str, signals: dict) -> float:
    """VIP warranty claim. Correct: resolve (with warranty coverage)."""
    is_vip = signals.get("is_vip", True)
    has_warranty = signals.get("has_warranty", True)
    score = 0.3

    if action == "resolve":
        score = 0.70
        if is_vip:
            score += 0.10
        if has_warranty:
            score += 0.15
    elif action == "refund":
        score = 0.50  # refund works but warranty replacement is better
    elif action == "search_kb":
        score = 0.45  # checking warranty terms is reasonable
    elif action == "escalate":
        score = 0.35
    elif action == "lookup_order":
        score = 0.30

    return min(score, 0.98)

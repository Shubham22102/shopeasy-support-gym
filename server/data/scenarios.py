"""
12 canonical episode scenarios for the ShopEasy Customer Support Gym.

Each scenario is a template that seeds an episode with a specific
customer issue type, difficulty, and required agent behaviors.
The environment samples one scenario per reset() call.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Scenario:
    """
    Defines an episode type for the support gym.

    Attributes:
        task_id         : unique identifier used in openenv.yaml tasks list
        title           : human-readable name
        difficulty      : easy | medium | hard
        issue_type      : maps to observation.issue_type
        description     : what the customer is calling about (for README)
        opening_message : the customer's first message (template, {name} filled in)
        required_tools  : tools the agent SHOULD call (used for R_process grading)
        forbidden_actions: actions that result in a penalty
        order_filters   : kwargs passed to OrderDatabase.get_random_order()
        max_steps       : episode step limit (overrides default 20)
        resolution_hint : what a correct resolution looks like (for grader)
        reward_criteria : dict of grading criteria and their weights
    """

    task_id: str
    title: str
    difficulty: str  # "easy" | "medium" | "hard"
    issue_type: str
    description: str
    opening_message: str  # may use {name} placeholder
    required_tools: List[str]
    forbidden_actions: List[str]
    order_filters: Dict[str, Any]
    max_steps: int
    resolution_hint: str
    reward_criteria: Dict[str, float]


# ---------------------------------------------------------------------------
# The 12 Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: List[Scenario] = [

    # ── EASY (1–4) ──────────────────────────────────────────────────────────

    Scenario(
        task_id="simple_refund",
        title="Simple Refund Request",
        difficulty="easy",
        issue_type="refund_request",
        description="Customer wants a refund for a recently delivered item within the return window.",
        opening_message=(
            "Hi, I'd like to return my recent order and get a refund please. "
            "I just changed my mind about it."
        ),
        required_tools=["lookup_order", "process_refund"],
        forbidden_actions=[],
        order_filters={"status": "delivered", "within_return_window": True, "is_damaged": False, "is_fraud_risk": False},
        max_steps=10,
        resolution_hint="Verify order is within return window → process full refund → close resolved.",
        reward_criteria={"correct_refund_issued": 0.5, "policy_checked": 0.3, "efficiency": 0.2},
    ),

    Scenario(
        task_id="delivery_tracking",
        title="Order Status / Tracking Inquiry",
        difficulty="easy",
        issue_type="delivery_inquiry",
        description="Customer wants to know where their shipped order is.",
        opening_message=(
            "Hi there! I placed an order a few days ago and haven't received it yet. "
            "Can you check the status for me?"
        ),
        required_tools=["lookup_order"],
        forbidden_actions=["process_refund"],
        order_filters={"status": "shipped"},
        max_steps=8,
        resolution_hint="Lookup order → share shipping status and estimated delivery date → close resolved.",
        reward_criteria={"accurate_status_shared": 0.6, "no_unnecessary_refund": 0.2, "efficiency": 0.2},
    ),

    Scenario(
        task_id="kb_policy_question",
        title="Return Policy Question",
        difficulty="easy",
        issue_type="policy_inquiry",
        description="Customer asks about the return policy for electronics.",
        opening_message=(
            "What's your return policy for electronics? I might want to return something."
        ),
        required_tools=["search_kb"],
        forbidden_actions=["process_refund"],
        order_filters={},
        max_steps=6,
        resolution_hint="Search KB for electronics return policy → share accurate policy → close resolved.",
        reward_criteria={"kb_searched": 0.4, "accurate_policy_shared": 0.4, "efficiency": 0.2},
    ),

    Scenario(
        task_id="cancellation_request",
        title="Order Cancellation",
        difficulty="easy",
        issue_type="cancellation",
        description="Customer wants to cancel an order that is still pending.",
        opening_message=(
            "I'd like to cancel my order please, I accidentally ordered the wrong thing."
        ),
        required_tools=["lookup_order"],
        forbidden_actions=[],
        order_filters={"status": "pending"},
        max_steps=8,
        resolution_hint="Lookup order → confirm it's cancellable (pending status) → cancel → close resolved.",
        reward_criteria={"order_verified_before_cancel": 0.4, "correct_cancellation": 0.4, "efficiency": 0.2},
    ),

    # ── MEDIUM (5–8) ─────────────────────────────────────────────────────────

    Scenario(
        task_id="expired_return",
        title="Return Request Past Policy Window",
        difficulty="medium",
        issue_type="refund_request",
        description=(
            "Customer wants a refund but the return window has expired. "
            "Agent must apply policy correctly (no full refund, can offer store credit)."
        ),
        opening_message=(
            "I want to return my order. I've been meaning to do it but just got around to it now."
        ),
        required_tools=["lookup_order", "search_kb"],
        forbidden_actions=[],
        order_filters={"status": "delivered", "within_return_window": False, "is_damaged": False},
        max_steps=12,
        resolution_hint=(
            "Lookup order → check days since delivery → search KB for expired return policy → "
            "explain store credit option (not full refund) → close appropriately."
        ),
        reward_criteria={
            "correctly_denied_full_refund": 0.4,
            "offered_store_credit_alternative": 0.2,
            "policy_verified_via_kb": 0.2,
            "efficiency": 0.2,
        },
    ),

    Scenario(
        task_id="wrong_item_sent",
        title="Wrong Item Received",
        difficulty="medium",
        issue_type="wrong_item",
        description="Customer received a different product than what they ordered.",
        opening_message=(
            "I received a completely different item than what I ordered! This is unacceptable. "
            "I ordered {item_name} but got something totally different."
        ),
        required_tools=["lookup_order", "process_refund"],
        forbidden_actions=[],
        order_filters={"status": "delivered"},
        max_steps=14,
        resolution_hint=(
            "Lookup order → verify item mismatch → apologize → process full refund (wrong item = instant refund) "
            "→ advise on return shipping → close resolved."
        ),
        reward_criteria={
            "apologized_appropriately": 0.2,
            "full_refund_processed": 0.4,
            "order_details_verified": 0.2,
            "efficiency": 0.2,
        },
    ),

    Scenario(
        task_id="duplicate_charge",
        title="Duplicate Payment Charge",
        difficulty="medium",
        issue_type="payment_dispute",
        description="Customer was charged twice for the same order.",
        opening_message=(
            "I've been charged twice for my order! I can see two transactions on my bank statement. "
            "Please fix this immediately."
        ),
        required_tools=["lookup_order", "check_payment", "process_refund"],
        forbidden_actions=[],
        order_filters={"is_duplicate_charge": True},
        max_steps=14,
        resolution_hint=(
            "Lookup order → check_payment to confirm duplicate → process refund for duplicate amount → "
            "apologize → close resolved."
        ),
        reward_criteria={
            "payment_verified_before_refund": 0.3,
            "correct_refund_amount": 0.3,
            "customer_reassured": 0.2,
            "efficiency": 0.2,
        },
    ),

    Scenario(
        task_id="partial_order",
        title="Partial Order Received",
        difficulty="medium",
        issue_type="delivery_issue",
        description="Customer ordered multiple items but only received some.",
        opening_message=(
            "I received my package today but it's missing items! I ordered 3 things "
            "but only 2 arrived. Where's the rest?"
        ),
        required_tools=["lookup_order"],
        forbidden_actions=[],
        order_filters={"status": "delivered"},
        max_steps=14,
        resolution_hint=(
            "Lookup order → verify item count → acknowledge missing items → "
            "offer to re-ship missing items or partial refund → close resolved or escalate."
        ),
        reward_criteria={
            "order_items_verified": 0.3,
            "partial_resolution_appropriate": 0.3,
            "customer_given_clear_next_steps": 0.2,
            "efficiency": 0.2,
        },
    ),

    # ── HARD (9–12) ──────────────────────────────────────────────────────────

    Scenario(
        task_id="damaged_item",
        title="Damaged Item Received",
        difficulty="hard",
        issue_type="damaged_goods",
        description=(
            "Customer received a damaged product. Policy: instant full refund regardless of return window. "
            "Agent must identify the damage flag and bypass normal window checks."
        ),
        opening_message=(
            "The item I received is completely broken! It looks like it was damaged during shipping. "
            "I want a full refund RIGHT NOW. This is ridiculous."
        ),
        required_tools=["lookup_order", "process_refund"],
        forbidden_actions=["escalate_to_human"],  # should handle it directly
        order_filters={"status": "delivered", "is_damaged": True},
        max_steps=15,
        resolution_hint=(
            "Lookup order → confirm is_damaged=True → apply damage policy (instant full refund, no window check) "
            "→ apologize sincerely → process refund → close resolved."
        ),
        reward_criteria={
            "damage_flag_identified": 0.2,
            "full_refund_issued_correctly": 0.3,
            "did_not_incorrectly_deny": 0.2,
            "customer_tone_managed": 0.1,
            "efficiency": 0.2,
        },
    ),

    Scenario(
        task_id="angry_customer",
        title="De-escalation: Extremely Angry Customer",
        difficulty="hard",
        issue_type="complaint",
        description=(
            "Customer opens with aggressive language and demands a manager immediately. "
            "Agent must de-escalate, solve the issue, and avoid unnecessary human escalation."
        ),
        opening_message=(
            "This is absolutely RIDICULOUS. Your company is a scam. I've been waiting 2 weeks "
            "and nobody has helped me. Get me your manager RIGHT NOW or I'm disputing the charge!"
        ),
        required_tools=["lookup_order"],
        forbidden_actions=[],  # escalating is allowed but penalised if issue could be solved
        order_filters={"status": "shipped"},
        max_steps=20,
        resolution_hint=(
            "Stay calm → acknowledge frustration → lookup order → explain status → "
            "offer concrete resolution → do NOT escalate unnecessarily → close resolved."
        ),
        reward_criteria={
            "de_escalated_tone": 0.2,
            "issue_resolved_without_escalation": 0.3,
            "order_verified": 0.2,
            "concrete_resolution_given": 0.1,
            "efficiency": 0.2,
        },
    ),

    Scenario(
        task_id="fraud_risk",
        title="Potential Fraud: Suspicious Request",
        difficulty="hard",
        issue_type="fraud_report",
        description=(
            "Customer request matches a known fraud pattern (high-value refund, unusual circumstances). "
            "Agent must escalate correctly WITHOUT accusing the customer directly."
        ),
        opening_message=(
            "Hi, I want to report that I never received my order but my account shows it as delivered. "
            "I also never made this purchase — I think my account was hacked."
        ),
        required_tools=["lookup_order", "escalate_to_human"],
        forbidden_actions=["process_refund"],  # must NOT issue refund on fraud risk, must escalate
        order_filters={"is_fraud_risk": True},
        max_steps=15,
        resolution_hint=(
            "Lookup order → identify fraud_risk=True → DO NOT process refund → "
            "politely explain it needs specialist review → escalate to human with 'high' priority → close escalated."
        ),
        reward_criteria={
            "correctly_escalated_not_refunded": 0.4,
            "fraud_risk_identified_in_tool": 0.2,
            "customer_not_accused": 0.2,
            "correct_close_code": 0.1,
            "efficiency": 0.1,
        },
    ),

    Scenario(
        task_id="vip_warranty_claim",
        title="VIP Customer: Warranty Claim",
        difficulty="hard",
        issue_type="warranty_claim",
        description=(
            "A VIP customer's electronics product failed after several months. "
            "Agent must check warranty terms from KB, verify VIP status, and prioritize resolution."
        ),
        opening_message=(
            "I've been a loyal ShopEasy customer for years. My {item_name} stopped working "
            "after 8 months. I know there's supposed to be a warranty. Please help me urgently."
        ),
        required_tools=["lookup_order", "search_kb"],
        forbidden_actions=[],
        order_filters={"status": "delivered", "is_vip": True, "has_warranty": True},
        max_steps=18,
        resolution_hint=(
            "Lookup order → verify VIP status and has_warranty → search KB for warranty terms → "
            "confirm 12-month warranty coverage → initiate warranty claim → close resolved with priority."
        ),
        reward_criteria={
            "vip_status_acknowledged": 0.1,
            "warranty_terms_kb_searched": 0.2,
            "correct_warranty_resolution": 0.3,
            "customer_treated_with_priority": 0.2,
            "efficiency": 0.2,
        },
    ),
]

# Quick lookup by task_id
SCENARIO_BY_ID: Dict[str, Scenario] = {s.task_id: s for s in SCENARIOS}

# Grouped by difficulty for curriculum sampling
EASY_SCENARIOS = [s for s in SCENARIOS if s.difficulty == "easy"]
MEDIUM_SCENARIOS = [s for s in SCENARIOS if s.difficulty == "medium"]
HARD_SCENARIOS = [s for s in SCENARIOS if s.difficulty == "hard"]


def get_scenario(task_id: Optional[str] = None, difficulty: Optional[str] = None) -> Scenario:
    """
    Return a scenario by task_id, or randomly pick one matching the given difficulty.
    Falls back to uniform random if neither is specified.
    """
    import random as _random

    if task_id is not None:
        if task_id not in SCENARIO_BY_ID:
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(SCENARIO_BY_ID.keys())}")
        return SCENARIO_BY_ID[task_id]

    if difficulty == "easy":
        return _random.choice(EASY_SCENARIOS)
    elif difficulty == "medium":
        return _random.choice(MEDIUM_SCENARIOS)
    elif difficulty == "hard":
        return _random.choice(HARD_SCENARIOS)
    else:
        return _random.choice(SCENARIOS)

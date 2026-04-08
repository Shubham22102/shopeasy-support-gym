"""
grader.py — Explicit Standalone Grader for the ShopEasy Support Gym
=====================================================================
Exposes precise, deterministic (binary / numeric) checks drawn directly
from RefundPolicyEngine and RewardCalculator — the same logic the
environment uses internally for reward computation.

Binary checks answer YES / NO questions:
  • Did the agent look up the order before promising a refund?
  • Did the agent correctly escalate a fraud-risk order?
  • Was the refund amount correct?

Aggregate scoring applies the full 3-tier reward formula
(outcome + process + efficiency) to a completed episode's evidence dict.

PUBLIC API
----------
  check_order_lookup_before_refund(verified_facts)         → BinaryResult
  check_fraud_escalated_correctly(order, verified_facts)   → BinaryResult
  check_refund_amount_correct(order, verified_facts)       → BinaryResult
  check_no_refund_on_fraud(order, verified_facts)          → BinaryResult
  check_kb_searched_when_required(task_id, verified_facts) → BinaryResult
  check_payment_verified_before_refund(task_id, facts)     → BinaryResult
  check_correct_close_code(task_id, order, facts, code)    → BinaryResult

  grade_episode(episode_evidence)                          → GradeReport
  grade_trajectory(trajectory)                             → GradeReport (from raw action list)
  print_report(report)                                     → None  (pretty-prints to stdout)

USAGE
-----
  from grader import grade_episode, print_report

  evidence = {
      "task_id": "fraud_risk",
      "order": { ... },          # dict returned by lookup_order tool
      "verified_facts": {        # dict maintained by environment state machine
          "order_looked_up": True,
          "refund_processed": False,
          "escalated": True,
          "kb_searched": False,
          "payment_checked": False,
          "refund_amount": None,
      },
      "close_resolution": "escalated",   # "resolved"|"escalated"|"unresolved"|"timeout"
      "step_count": 7,
      "max_steps": 15,
      "customer_mood": 0.1,             # float -1.0 to +1.0
      "agent_sent_messages": True,
      # optional: include conversation turns for LLM-backed checks
      "conversation_history": [ {"role": "agent", "content": "..."}, ... ],
  }

  report = grade_episode(evidence)
  print_report(report)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from server.engine.policy_engine import RefundPolicyEngine
from server.engine.reward import RewardCalculator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BinaryResult:
    """Result of one binary / deterministic check."""

    check_name: str
    passed: bool
    score: float  # 1.0 | 0.0  (or a partial weight for weighted checks)
    explanation: str
    severity: str = "info"  # "info" | "warning" | "critical"

    def __str__(self) -> str:
        icon = "✔️ " if self.passed else "❌"
        return f"{icon} [{self.severity.upper()}] {self.check_name}: {self.explanation}"


@dataclass
class GradeReport:
    """Full grading report for one completed episode."""

    task_id: str
    binary_checks: List[BinaryResult] = field(default_factory=list)

    # 3-tier reward components (0.0–1.0)
    r_outcome: float = 0.0
    r_process: float = 0.0
    r_efficiency: float = 0.0
    r_total: float = 0.0

    # Summary
    n_passed: int = 0
    n_failed: int = 0
    critical_failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "binary_checks": [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "score": r.score,
                    "explanation": r.explanation,
                    "severity": r.severity,
                }
                for r in self.binary_checks
            ],
            "reward": {
                "outcome": self.r_outcome,
                "process": self.r_process,
                "efficiency": self.r_efficiency,
                "total": self.r_total,
            },
            "summary": {
                "n_passed": self.n_passed,
                "n_failed": self.n_failed,
                "critical_failures": self.critical_failures,
            },
        }


# ---------------------------------------------------------------------------
# Internal singletons
# ---------------------------------------------------------------------------

_policy = RefundPolicyEngine()
_reward_calc = RewardCalculator()
STRICT_SCORE_MIN = 0.001
STRICT_SCORE_MAX = 0.999


def _strict_unit_score(score: float) -> float:
    """Clamp scores to the open interval (0, 1) for hackathon validation."""
    return min(STRICT_SCORE_MAX, max(STRICT_SCORE_MIN, float(score)))


# ---------------------------------------------------------------------------
# Individual binary checks
# ---------------------------------------------------------------------------


def check_order_lookup_before_refund(verified_facts: Dict[str, Any]) -> BinaryResult:
    """
    Binary check: Did the agent call lookup_order BEFORE processing a refund?

    This is the most fundamental procedural check — an agent that issues a
    refund without first verifying the order may refund fraudulent, cancelled
    or already-refunded orders.
    """
    refund_attempted = bool(verified_facts.get("refund_processed"))
    order_looked_up = bool(verified_facts.get("order_looked_up"))

    if not refund_attempted:
        # No refund was made — check is N/A but we count it as passing
        # (agent can't violate the rule if it didn't trigger it)
        return BinaryResult(
            check_name="order_lookup_before_refund",
            passed=True,
            score=1.0,
            explanation="No refund was processed — check not applicable; counting as pass.",
            severity="info",
        )

    passed = order_looked_up
    return BinaryResult(
        check_name="order_lookup_before_refund",
        passed=passed,
        score=1.0 if passed else 0.0,
        explanation=(
            "✓ Agent looked up order before processing refund."
            if passed
            else "✗ Agent processed refund WITHOUT calling lookup_order first (policy violation)."
        ),
        severity="warning" if not passed else "info",
    )


def check_fraud_escalated_correctly(
    order: Dict[str, Any],
    verified_facts: Dict[str, Any],
) -> BinaryResult:
    """
    Binary check: On a fraud-risk order, did the agent escalate (not refund)?

    Correct behaviour = escalated=True AND refund_processed=False.
    Processing a refund on a fraud-risk order is a CRITICAL policy violation.
    """
    is_fraud = bool(order.get("is_fraud_risk"))

    if not is_fraud:
        return BinaryResult(
            check_name="fraud_escalated_correctly",
            passed=True,
            score=1.0,
            explanation="Order is not flagged as fraud risk — check not applicable.",
            severity="info",
        )

    escalated = bool(verified_facts.get("escalated"))
    refund_issued = bool(verified_facts.get("refund_processed"))

    passed = escalated and not refund_issued

    if refund_issued and is_fraud:
        explanation = (
            "✗ CRITICAL: Agent issued a refund on a FRAUD-RISK order. "
            "This is a serious policy violation — must escalate to specialist."
        )
        severity = "critical"
    elif not escalated:
        explanation = (
            "✗ Agent did not escalate the fraud-risk order "
            "(should call escalate_to_human with priority='high')."
        )
        severity = "warning"
    else:
        explanation = (
            "✓ Agent correctly escalated fraud-risk order without issuing refund."
        )
        severity = "info"

    return BinaryResult(
        check_name="fraud_escalated_correctly",
        passed=passed,
        score=1.0 if passed else 0.0,
        explanation=explanation,
        severity=severity,
    )


def check_refund_amount_correct(
    order: Dict[str, Any],
    verified_facts: Dict[str, Any],
) -> BinaryResult:
    """
    Binary/numeric check: Was the refund amount correct?

    Rules:
      • Duplicate charge  → refund should equal exactly one payment amount (order["total"])
      • Damaged item / wrong item / within-window → full refund (order["total"])
      • Expired return    → store credit, NOT full refund (any amount is wrong)
      • Fraud risk        → no refund permitted (amount irrelevant)

    verified_facts["refund_amount"] is set by the tools engine when
    process_refund is called.
    """
    refund_processed = bool(verified_facts.get("refund_processed"))

    if not refund_processed:
        return BinaryResult(
            check_name="refund_amount_correct",
            passed=True,
            score=1.0,
            explanation="No refund was processed — amount check not applicable.",
            severity="info",
        )

    agent_amount = verified_facts.get("refund_amount")  # float or None
    order_total = order.get("total", 0.0)  # correct amount

    # Fraud-risk: any refund amount is wrong
    if order.get("is_fraud_risk"):
        return BinaryResult(
            check_name="refund_amount_correct",
            passed=False,
            score=0.0,
            explanation=(
                f"✗ CRITICAL: Refund of {agent_amount} issued on a fraud-risk order. "
                "No refund should have been made at all."
            ),
            severity="critical",
        )

    # Expired return: any monetary refund is wrong (should be store credit)
    if (
        order.get("status") == "delivered"
        and not order.get("within_return_window")
        and not order.get("is_damaged")
    ):
        return BinaryResult(
            check_name="refund_amount_correct",
            passed=False,
            score=0.0,
            explanation=(
                f"✗ Agent processed a cash refund ({agent_amount}) outside the return window. "
                "Policy requires store credit, not a cash refund."
            ),
            severity="warning",
        )

    # Duplicate charge: should refund exactly the order total (one charge)
    if order.get("is_duplicate_charge"):
        tolerance = 0.01
        passed = (
            agent_amount is not None
            and abs(float(agent_amount) - order_total) <= tolerance
        )
        return BinaryResult(
            check_name="refund_amount_correct",
            passed=passed,
            score=1.0 if passed else 0.0,
            explanation=(
                f"✓ Refund amount {agent_amount} matches duplicate charge amount {order_total}."
                if passed
                else f"✗ Refund amount {agent_amount} does not match expected {order_total} "
                f"(duplicate charge should be exactly one payment)."
            ),
            severity="info" if passed else "warning",
        )

    # General cases: should be exact total
    tolerance = 0.01
    if agent_amount is None:
        passed = False
        explanation = (
            "✗ Refund amount not recorded in verified_facts (refund_amount is None)."
        )
        severity = "warning"
    else:
        # Allow slight rounding
        passed = abs(float(agent_amount) - order_total) <= tolerance
        explanation = (
            f"✓ Refund amount {agent_amount} is correct (order total: {order_total})."
            if passed
            else (
                f"✗ Refund amount {agent_amount} differs from expected order total "
                f"{order_total} (difference: {abs(float(agent_amount) - order_total):.2f})."
            )
        )
        severity = "info" if passed else "warning"

    return BinaryResult(
        check_name="refund_amount_correct",
        passed=passed,
        score=1.0 if passed else 0.0,
        explanation=explanation,
        severity=severity,
    )


def check_no_refund_on_fraud(
    order: Dict[str, Any],
    verified_facts: Dict[str, Any],
) -> BinaryResult:
    """
    Binary check: Confirm agent did NOT process a refund on a fraud-risk order.

    This duplicates the core of check_fraud_escalated_correctly but focuses
    purely on the refund-prohibition rule (useful as a separate signal).
    """
    is_fraud = bool(order.get("is_fraud_risk"))
    refund_issued = bool(verified_facts.get("refund_processed"))

    if not is_fraud:
        return BinaryResult(
            check_name="no_refund_on_fraud",
            passed=True,
            score=1.0,
            explanation="Order is not fraud-flagged — check not applicable.",
            severity="info",
        )

    passed = not refund_issued
    return BinaryResult(
        check_name="no_refund_on_fraud",
        passed=passed,
        score=1.0 if passed else 0.0,
        explanation=(
            "✓ No refund issued on fraud-risk order (correct)."
            if passed
            else "✗ CRITICAL: Refund issued on fraud-risk order — must NEVER happen."
        ),
        severity="info" if passed else "critical",
    )


def check_kb_searched_when_required(
    task_id: str,
    verified_facts: Dict[str, Any],
) -> BinaryResult:
    """
    Binary check: Did the agent search the knowledge base on tasks that require it?

    Required tasks: expired_return, kb_policy_question, vip_warranty_claim.
    """
    KB_REQUIRED = {
        "expired_return",
        "kb_policy_question",
        "vip_warranty_claim",
        "warranty_claim",
    }

    if task_id not in KB_REQUIRED:
        return BinaryResult(
            check_name="kb_searched_when_required",
            passed=True,
            score=1.0,
            explanation=f"Task '{task_id}' does not require a KB search.",
            severity="info",
        )

    passed = bool(verified_facts.get("kb_searched"))
    return BinaryResult(
        check_name="kb_searched_when_required",
        passed=passed,
        score=1.0 if passed else 0.0,
        explanation=(
            f"✓ Agent searched knowledge base (required for '{task_id}')."
            if passed
            else f"✗ Agent never searched KB for task '{task_id}' — policy accuracy may be wrong."
        ),
        severity="info" if passed else "warning",
    )


def check_payment_verified_before_refund(
    task_id: str,
    verified_facts: Dict[str, Any],
) -> BinaryResult:
    """
    Binary check: For duplicate_charge, did agent call check_payment first?
    """
    if task_id != "duplicate_charge":
        return BinaryResult(
            check_name="payment_verified_before_refund",
            passed=True,
            score=1.0,
            explanation=f"Task '{task_id}' does not require payment verification.",
            severity="info",
        )

    refund_processed = bool(verified_facts.get("refund_processed"))
    payment_checked = bool(verified_facts.get("payment_checked"))

    if not refund_processed:
        return BinaryResult(
            check_name="payment_verified_before_refund",
            passed=True,
            score=1.0,
            explanation="No refund was processed in this duplicate_charge episode.",
            severity="info",
        )

    passed = payment_checked
    return BinaryResult(
        check_name="payment_verified_before_refund",
        passed=passed,
        score=1.0 if passed else 0.0,
        explanation=(
            "✓ Agent verified payment with check_payment before processing refund."
            if passed
            else "✗ Agent processed refund for duplicate_charge WITHOUT calling check_payment first."
        ),
        severity="info" if passed else "warning",
    )


def check_correct_close_code(
    task_id: str,
    order: Dict[str, Any],
    verified_facts: Dict[str, Any],
    close_resolution: str,
) -> BinaryResult:
    """
    Binary check: Was the ticket closed with the semantically correct resolution code?

    Expected close codes:
      fraud_risk            → "escalated"
      angry_customer        → "resolved" (should not need escalation)
      delivery_tracking     → "resolved"
      kb_policy_question    → "resolved"
      everything else       → "resolved" (unless fraud path → "escalated")
    """
    expected = "resolved"

    if order.get("is_fraud_risk"):
        expected = "escalated"
    elif task_id == "angry_customer" and not order.get("is_fraud_risk"):
        expected = "resolved"

    passed = close_resolution == expected
    return BinaryResult(
        check_name="correct_close_code",
        passed=passed,
        score=1.0 if passed else 0.0,
        explanation=(
            f"✓ Ticket closed with correct code '{close_resolution}'."
            if passed
            else f"✗ Ticket closed as '{close_resolution}' but expected '{expected}' for task '{task_id}'."
        ),
        severity="info" if passed else "warning",
    )


# ---------------------------------------------------------------------------
# Full episode grader  (runs all checks + 3-tier reward)
# ---------------------------------------------------------------------------


def grade_episode(episode_evidence: Dict[str, Any]) -> GradeReport:
    """
    Grade a completed episode from its evidence dict.

    episode_evidence keys (all required unless marked optional):
        task_id             str
        order               Dict | None   — from lookup_order tool result
        verified_facts      Dict          — state machine bookkeeping dict
        close_resolution    str           — "resolved"|"escalated"|"unresolved"|"timeout"
        step_count          int
        max_steps           int
        customer_mood       float         — -1.0 to +1.0
        agent_sent_messages bool
        conversation_history list[dict]  (optional)

    Returns GradeReport with binary results + aggregate score.
    """
    task_id = episode_evidence.get("task_id", "unknown")
    order = episode_evidence.get("order") or {}
    facts = episode_evidence.get("verified_facts", {})
    close_resolution = episode_evidence.get("close_resolution", "unresolved")
    step_count = int(episode_evidence.get("step_count", 0))
    max_steps = int(episode_evidence.get("max_steps", 20))
    customer_mood = float(episode_evidence.get("customer_mood", 0.0))
    agent_sent_msgs = bool(episode_evidence.get("agent_sent_messages", True))

    report = GradeReport(task_id=task_id)

    # ── Run binary checks ──────────────────────────────────────────────────
    checks: List[BinaryResult] = [
        check_order_lookup_before_refund(facts),
        check_fraud_escalated_correctly(order, facts),
        check_refund_amount_correct(order, facts),
        check_no_refund_on_fraud(order, facts),
        check_kb_searched_when_required(task_id, facts),
        check_payment_verified_before_refund(task_id, facts),
        check_correct_close_code(task_id, order, facts, close_resolution),
    ]

    report.binary_checks = checks

    # ── Aggregate binary summary ───────────────────────────────────────────
    for c in checks:
        if c.passed:
            report.n_passed += 1
        else:
            report.n_failed += 1
            if c.severity == "critical":
                report.critical_failures.append(c.check_name)

    # ── 3-tier reward (same formula as environment) ────────────────────────
    reward_breakdown = _reward_calc.calculate(
        order=order or None,
        scenario_task_id=task_id,
        verified_facts=facts,
        conversation_history=episode_evidence.get("conversation_history", []),
        close_resolution=close_resolution,
        step_count=step_count,
        max_steps=max_steps,
        customer_mood=customer_mood,
        agent_sent_messages=agent_sent_msgs,
    )

    report.r_outcome = reward_breakdown["outcome"]
    report.r_process = reward_breakdown["process"]
    report.r_efficiency = reward_breakdown["efficiency"]
    report.r_total = reward_breakdown["total"]

    return report


# ---------------------------------------------------------------------------
# Trajectory grader  (parses raw action list → derives verified_facts)
# ---------------------------------------------------------------------------


def grade_trajectory(
    task_id: str,
    order: Optional[Dict[str, Any]],
    action_history: List[Dict[str, Any]],
    close_resolution: str = "unresolved",
    max_steps: int = 20,
    customer_mood: float = 0.0,
) -> GradeReport:
    """
    Grade an episode from the raw list of actions taken by the agent.

    Derives `verified_facts` by replaying the action sequence and then
    calls grade_episode.

    action_history items should look like one of:
        {"action_type": "tool_call", "tool_name": "lookup_order", "tool_args": {...}}
        {"action_type": "tool_call", "tool_name": "process_refund", "tool_args": {"amount": 599.0}}
        {"action_type": "send_message", "message": "..."}
        {"action_type": "close_ticket", "resolution": "resolved"}

    This is the grader entry-point when you only have the raw trajectory,
    not the environment's internal verified_facts dict.
    """
    facts: Dict[str, Any] = {
        "order_looked_up": False,
        "refund_processed": False,
        "refund_amount": None,
        "escalated": False,
        "kb_searched": False,
        "payment_checked": False,
        "is_vip": order.get("is_vip", False) if order else False,
    }

    agent_sent_messages = False
    derived_close = close_resolution

    for action in action_history:
        atype = action.get("action_type", "")

        if atype == "tool_call":
            tname = action.get("tool_name", "")
            targs = action.get("tool_args", {})

            if tname == "lookup_order":
                facts["order_looked_up"] = True

            elif tname == "process_refund":
                facts["refund_processed"] = True
                facts["refund_amount"] = targs.get("amount")

            elif tname == "escalate_to_human":
                facts["escalated"] = True

            elif tname == "search_kb":
                facts["kb_searched"] = True

            elif tname == "check_payment":
                facts["payment_checked"] = True

        elif atype == "send_message":
            agent_sent_messages = True

        elif atype == "close_ticket":
            derived_close = action.get("resolution", close_resolution)

    evidence = {
        "task_id": task_id,
        "order": order,
        "verified_facts": facts,
        "close_resolution": derived_close,
        "step_count": len(action_history),
        "max_steps": max_steps,
        "customer_mood": customer_mood,
        "agent_sent_messages": agent_sent_messages,
    }

    return grade_episode(evidence)


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------


def print_report(report: GradeReport) -> None:
    """Print a well-formatted grading report to stdout."""
    divider = "─" * 60
    print(f"\n{divider}")
    print(f"  GRADER REPORT  ·  task_id = {report.task_id}")
    print(divider)

    print("\n  Binary Checks")
    print("  " + "─" * 56)
    for c in report.binary_checks:
        print(f"  {c}")

    print(f"\n  Passed: {report.n_passed}  |  Failed: {report.n_failed}", end="")
    if report.critical_failures:
        print(
            f"  |  ⚠️  Critical failures: {', '.join(report.critical_failures)}", end=""
        )
    print()

    print("\n  Reward Breakdown")
    print("  " + "─" * 56)
    print(f"    R_outcome   = {report.r_outcome:.4f}  (max 0.60)")
    print(f"    R_process   = {report.r_process:.4f}  (max 0.30)")
    print(f"    R_efficiency= {report.r_efficiency:.4f}  (max 0.10)")
    print("    ─────────────────────────────")
    print(
        f"    R_total     = {report.r_total:.4f}  {'✔ SUCCESS' if report.r_total >= 0.4 else '✘ FAIL'}"
    )
    print(divider + "\n")


# ---------------------------------------------------------------------------
# CLI entry point — grade a JSON episode file from disk
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SupportTaskGrader — Explicit standalone grader class (Scaler compliance)
# ---------------------------------------------------------------------------


@dataclass
class GraderResult:
    """
    Minimal result object returned by SupportTaskGrader.grade().
    Score is always in [0.0, 1.0]; passed = score >= 0.4 (success threshold).
    """

    score: float  # 0.0 to 1.0
    passed: bool  # score >= 0.4
    feedback: str  # human-readable explanation of the score

    def __post_init__(self) -> None:
        self.score = _strict_unit_score(self.score)


class SupportTaskGrader:
    """
    Explicit grader for Scaler evaluation compliance.

    Call grade() with task_id + episode evidence to get a GraderResult
    with a deterministic score in [0.0, 1.0].

    This is an importable, self-contained class.  Internally it delegates
    to the existing grade_episode() / binary-check machinery so there is
    no duplication of logic.

    Usage::

        from grader import SupportTaskGrader

        grader = SupportTaskGrader()
        result = grader.grade(
            task_id="fraud_risk",
            verified_facts=env._verified_facts,
            conversation_history=env._conversation_history,
            final_resolution="escalated",
            order=env._order,
            step_count=7,
            max_steps=15,
            customer_mood=0.1,
            agent_sent_messages=True,
        )
        print(result.score, result.passed, result.feedback)
    """

    # Per-task grading dispatch table -----------------------------------------

    _TASK_METHODS = {
        "simple_refund": "_grade_simple_refund",
        "delivery_tracking": "_grade_delivery_tracking",
        "kb_policy_question": "_grade_kb_policy_question",
        "cancellation_request": "_grade_cancellation_request",
        "expired_return": "_grade_expired_return",
        "wrong_item_sent": "_grade_wrong_item_sent",
        "duplicate_charge": "_grade_duplicate_charge",
        "partial_order": "_grade_partial_order",
        "damaged_item": "_grade_damaged_item",
        "angry_customer": "_grade_angry_customer",
        "fraud_risk": "_grade_fraud_risk",
        "vip_warranty_claim": "_grade_vip_warranty_claim",
    }

    def grade(
        self,
        task_id: str,
        verified_facts: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[list] = None,
        final_resolution: Optional[str] = None,
        order: Optional[Dict[str, Any]] = None,
        step_count: int = 0,
        max_steps: int = 20,
        customer_mood: float = 0.0,
        agent_sent_messages: bool = True,
        **kwargs: Any,
    ) -> GraderResult:
        """
        Grade a completed episode.  Returns GraderResult(score, passed, feedback).

        score is the canonical 3-tier reward total (same formula as the env).
        Per-task binary checks are also run and any critical failures zero the score.
        """
        verified_facts = verified_facts or kwargs.get("facts") or {}
        conversation_history = (
            conversation_history or kwargs.get("history") or kwargs.get("trajectory") or []
        )
        final_resolution = (
            final_resolution
            or kwargs.get("close_resolution")
            or kwargs.get("resolution")
            or kwargs.get("final_status")
            or "unresolved"
        )
        order = order or kwargs.get("episode_order") or {}
        step_count = int(kwargs.get("steps", step_count))
        max_steps = int(kwargs.get("episode_max_steps", max_steps))
        customer_mood = float(kwargs.get("mood", customer_mood))
        agent_sent_messages = bool(
            kwargs.get("sent_messages", agent_sent_messages)
        )

        facts = {k: v for k, v in verified_facts.items() if not k.startswith("_ir_")}
        ord_ = order or {}

        evidence: Dict[str, Any] = {
            "task_id": task_id,
            "order": ord_,
            "verified_facts": facts,
            "close_resolution": final_resolution,
            "step_count": step_count,
            "max_steps": max_steps,
            "customer_mood": customer_mood,
            "agent_sent_messages": agent_sent_messages,
            "conversation_history": conversation_history,
        }

        report = grade_episode(evidence)

        # Critical failures immediately zero the score
        if report.critical_failures:
            return GraderResult(
                score=0.0,
                passed=False,
                feedback=(
                    f"CRITICAL FAILURE(s): {', '.join(report.critical_failures)}. "
                    "Score zeroed."
                ),
            )

        score = _strict_unit_score(report.r_total)
        passed = score >= 0.4

        # Build human-readable feedback from binary checks
        parts: List[str] = []
        for c in report.binary_checks:
            if c.severity != "info" or not c.passed:
                parts.append(c.explanation)
        feedback = " | ".join(parts) if parts else "All checks passed."

        # Dispatch per-task overrides (may adjust score for specific edge cases)
        method_name = self._TASK_METHODS.get(task_id)
        if method_name:
            method = getattr(self, method_name, None)
            if method:
                override = method(facts, ord_, final_resolution, score)
                if override is not None:
                    return override

        return GraderResult(score=score, passed=passed, feedback=feedback)

    # ── Per-task grade methods ────────────────────────────────────────────────

    def _grade_simple_refund(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        checks: List[str] = []
        score = 0.0
        if facts.get("order_looked_up"):
            score += 0.30
            checks.append("Order verified ✔")
        else:
            checks.append("FAIL: Did not verify order")
        if facts.get("refund_processed") and resolution == "resolved":
            score += 0.50
            checks.append("Refund processed correctly ✔")
        else:
            checks.append("FAIL: Refund not processed")
        if facts.get("kb_searched"):
            score += 0.20
            checks.append("Policy checked ✔")
        score = min(1.0, score)
        return GraderResult(
            score=score, passed=score >= 0.4, feedback=" | ".join(checks)
        )

    def _grade_fraud_risk(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        if facts.get("refund_processed"):
            return GraderResult(
                score=0.0,
                passed=False,
                feedback="CRITICAL: Processed refund on fraud-risk order — automatic zero.",
            )
        if facts.get("escalated") and resolution == "escalated":
            return GraderResult(
                score=1.0,
                passed=True,
                feedback="Correctly escalated fraud-risk order without issuing refund ✔",
            )
        score = 0.1 if facts.get("order_looked_up") else 0.0
        return GraderResult(
            score=score,
            passed=False,
            feedback="FAIL: Did not escalate fraud-risk order",
        )

    def _grade_delivery_tracking(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        checks: List[str] = []
        score = 0.0
        if facts.get("order_looked_up"):
            score += 0.60
            checks.append("Order status looked up ✔")
        else:
            checks.append("FAIL: Did not look up order")
        if resolution == "resolved":
            score += 0.40
            checks.append("Ticket resolved ✔")
        score = min(1.0, score)
        return GraderResult(
            score=score, passed=score >= 0.4, feedback=" | ".join(checks)
        )

    def _grade_kb_policy_question(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        score = 0.0
        checks: List[str] = []
        if facts.get("kb_searched"):
            score += 0.60
            checks.append("KB searched ✔")
        else:
            checks.append("FAIL: KB not searched")
        if resolution == "resolved":
            score += 0.40
            checks.append("Ticket resolved ✔")
        score = min(1.0, score)
        return GraderResult(
            score=score, passed=score >= 0.4, feedback=" | ".join(checks)
        )

    def _grade_cancellation_request(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        score = 0.0
        checks: List[str] = []
        if facts.get("order_looked_up"):
            score += 0.30
            checks.append("Order checked ✔")
        if facts.get("subscription_cancelled") or resolution == "resolved":
            score += 0.50
            checks.append("Cancellation handled ✔")
        score = min(1.0, score)
        return GraderResult(
            score=score, passed=score >= 0.4, feedback=" | ".join(checks)
        )

    def _grade_expired_return(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        if facts.get("refund_processed") and not facts.get("store_credit_offered"):
            return GraderResult(
                score=0.0,
                passed=False,
                feedback="FAIL: Processed cash refund outside return window (should offer store credit)",
            )
        score = 0.0
        checks: List[str] = []
        if facts.get("order_looked_up"):
            score += 0.20
            checks.append("Order verified ✔")
        if facts.get("kb_searched"):
            score += 0.30
            checks.append("Policy checked ✔")
        if resolution == "resolved":
            score += 0.50
            checks.append("Properly resolved ✔")
        score = min(1.0, score)
        return GraderResult(
            score=score, passed=score >= 0.4, feedback=" | ".join(checks)
        )

    def _grade_wrong_item_sent(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        score = 0.0
        checks: List[str] = []
        if facts.get("order_looked_up"):
            score += 0.30
            checks.append("Order verified ✔")
        if facts.get("refund_processed") and resolution == "resolved":
            score += 0.50
            checks.append("Refund/replacement arranged ✔")
        if facts.get("kb_searched"):
            score += 0.20
            checks.append("Policy checked ✔")
        score = min(1.0, score)
        return GraderResult(
            score=score, passed=score >= 0.4, feedback=" | ".join(checks)
        )

    def _grade_duplicate_charge(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        score = 0.0
        checks: List[str] = []
        if facts.get("order_looked_up"):
            score += 0.20
            checks.append("Order verified ✔")
        if facts.get("payment_checked"):
            score += 0.30
            checks.append("Payment verified ✔")
        if facts.get("refund_processed") and resolution == "resolved":
            score += 0.50
            checks.append("Refund processed ✔")
        score = min(1.0, score)
        return GraderResult(
            score=score, passed=score >= 0.4, feedback=" | ".join(checks)
        )

    def _grade_partial_order(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        score = 0.0
        checks: List[str] = []
        if facts.get("order_looked_up"):
            score += 0.30
            checks.append("Order verified ✔")
        if resolution in ("resolved",):
            score += 0.50
            checks.append("Partial delivery addressed ✔")
        if facts.get("kb_searched"):
            score += 0.20
            checks.append("Policy checked ✔")
        score = min(1.0, score)
        return GraderResult(
            score=score, passed=score >= 0.4, feedback=" | ".join(checks)
        )

    def _grade_damaged_item(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        score = 0.0
        checks: List[str] = []
        if facts.get("order_looked_up"):
            score += 0.20
            checks.append("Order verified ✔")
        if facts.get("refund_processed") and resolution == "resolved":
            score += 0.60
            checks.append("Instant refund issued ✔")
        else:
            checks.append("FAIL: Instant refund not processed for damaged item")
        if facts.get("kb_searched"):
            score += 0.20
            checks.append("Policy checked ✔")
        score = min(1.0, score)
        return GraderResult(
            score=score, passed=score >= 0.4, feedback=" | ".join(checks)
        )

    def _grade_angry_customer(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        # Use base 3-tier score; critical fail only if refund without lookup
        if facts.get("refund_processed") and not facts.get("order_looked_up"):
            return GraderResult(
                score=0.0,
                passed=False,
                feedback="FAIL: Refund without order lookup on angry_customer",
            )
        return None  # use base_score

    def _grade_vip_warranty_claim(
        self, facts: Dict, order: Dict, resolution: str, base_score: float
    ) -> Optional[GraderResult]:
        score = 0.0
        checks: List[str] = []
        if facts.get("order_looked_up"):
            score += 0.20
            checks.append("Order verified ✔")
        if facts.get("kb_searched"):
            score += 0.30
            checks.append("Warranty policy checked ✔")
        if resolution == "resolved":
            score += 0.50
            checks.append("VIP claim resolved ✔")
        else:
            checks.append("FAIL: VIP claim not resolved")
        score = min(1.0, score)
        return GraderResult(
            score=score, passed=score >= 0.4, feedback=" | ".join(checks)
        )


# ---------------------------------------------------------------------------
# Flexible task-level grader entry points
# ---------------------------------------------------------------------------


def grade_task(task_id: str, **kwargs: Any) -> GraderResult:
    """Grade a task with tolerant kwargs for external validators."""
    grader = SupportTaskGrader()
    evidence = kwargs.pop("episode_evidence", None) or kwargs.pop("evidence", None)
    if isinstance(evidence, dict):
        merged = dict(evidence)
        merged.update(kwargs)
        kwargs = merged

    return grader.grade(
        task_id=task_id,
        verified_facts=kwargs.pop("verified_facts", None),
        conversation_history=kwargs.pop("conversation_history", None),
        final_resolution=kwargs.pop("final_resolution", None),
        order=kwargs.pop("order", None),
        step_count=int(kwargs.pop("step_count", 0)),
        max_steps=int(kwargs.pop("max_steps", 20)),
        customer_mood=float(kwargs.pop("customer_mood", 0.0)),
        agent_sent_messages=bool(kwargs.pop("agent_sent_messages", True)),
        **kwargs,
    )


def grade_simple_refund(**kwargs: Any) -> GraderResult:
    return grade_task("simple_refund", **kwargs)


def grade_delivery_tracking(**kwargs: Any) -> GraderResult:
    return grade_task("delivery_tracking", **kwargs)


def grade_kb_policy_question(**kwargs: Any) -> GraderResult:
    return grade_task("kb_policy_question", **kwargs)


def grade_cancellation_request(**kwargs: Any) -> GraderResult:
    return grade_task("cancellation_request", **kwargs)


def grade_expired_return(**kwargs: Any) -> GraderResult:
    return grade_task("expired_return", **kwargs)


def grade_wrong_item_sent(**kwargs: Any) -> GraderResult:
    return grade_task("wrong_item_sent", **kwargs)


def grade_duplicate_charge(**kwargs: Any) -> GraderResult:
    return grade_task("duplicate_charge", **kwargs)


def grade_partial_order(**kwargs: Any) -> GraderResult:
    return grade_task("partial_order", **kwargs)


def grade_damaged_item(**kwargs: Any) -> GraderResult:
    return grade_task("damaged_item", **kwargs)


def grade_angry_customer(**kwargs: Any) -> GraderResult:
    return grade_task("angry_customer", **kwargs)


def grade_fraud_risk(**kwargs: Any) -> GraderResult:
    return grade_task("fraud_risk", **kwargs)


def grade_vip_warranty_claim(**kwargs: Any) -> GraderResult:
    return grade_task("vip_warranty_claim", **kwargs)


TASK_GRADERS = {
    "simple_refund": grade_simple_refund,
    "delivery_tracking": grade_delivery_tracking,
    "kb_policy_question": grade_kb_policy_question,
    "cancellation_request": grade_cancellation_request,
    "expired_return": grade_expired_return,
    "wrong_item_sent": grade_wrong_item_sent,
    "duplicate_charge": grade_duplicate_charge,
    "partial_order": grade_partial_order,
    "damaged_item": grade_damaged_item,
    "angry_customer": grade_angry_customer,
    "fraud_risk": grade_fraud_risk,
    "vip_warranty_claim": grade_vip_warranty_claim,
}

# Common discovery aliases for external validators.
Grader = SupportTaskGrader
grade = grade_task
simple_refund_grader = grade_simple_refund
delivery_tracking_grader = grade_delivery_tracking
kb_policy_question_grader = grade_kb_policy_question
cancellation_request_grader = grade_cancellation_request
expired_return_grader = grade_expired_return
wrong_item_sent_grader = grade_wrong_item_sent
duplicate_charge_grader = grade_duplicate_charge
partial_order_grader = grade_partial_order
damaged_item_grader = grade_damaged_item
angry_customer_grader = grade_angry_customer
fraud_risk_grader = grade_fraud_risk
vip_warranty_claim_grader = grade_vip_warranty_claim


# ---------------------------------------------------------------------------
# CLI entry point — grade a JSON episode file from disk
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python grader.py <episode_evidence.json>")
        print("       python grader.py --demo")
        sys.exit(0)

    if sys.argv[1] == "--demo":
        # Demo: simulate a fraud_risk episode where agent correctly escalated
        demo_evidence = {
            "task_id": "fraud_risk",
            "order": {
                "order_id": "SE-7777",
                "total": 12500.0,
                "status": "delivered",
                "is_fraud_risk": True,
                "is_damaged": False,
                "within_return_window": True,
                "is_duplicate_charge": False,
                "refund_issued": False,
            },
            "verified_facts": {
                "order_looked_up": True,
                "refund_processed": False,
                "refund_amount": None,
                "escalated": True,
                "kb_searched": False,
                "payment_checked": False,
            },
            "close_resolution": "escalated",
            "step_count": 5,
            "max_steps": 15,
            "customer_mood": 0.1,
            "agent_sent_messages": True,
        }
        report = grade_episode(demo_evidence)
        print_report(report)
        print(json.dumps(report.to_dict(), indent=2))
    else:
        with open(sys.argv[1]) as f:
            evidence = json.load(f)
        report = grade_episode(evidence)
        print_report(report)
        print(json.dumps(report.to_dict(), indent=2))

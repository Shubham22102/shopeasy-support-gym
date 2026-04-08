"""
Tool executor for the ShopEasy Support Gym.

Each tool corresponds to a real agent action. The environment calls
execute_tool() to dispatch a tool_call action to the right handler.
Tools return structured dicts that become observation.tool_result.
"""

from typing import Any, Dict

from ..data.knowledge_base import search_kb
from ..data.orders import OrderDatabase


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

VALID_TOOLS = {
    "lookup_order",
    "process_refund",
    "search_kb",
    "escalate_to_human",
    "cancel_subscription",
    "check_payment",
}


def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    db: OrderDatabase,
    verified_facts: Dict[str, Any],
    ticket_id: str,
) -> Dict[str, Any]:
    """
    Dispatch a tool call and return a structured result dict.

    Args:
        tool_name     : name of the tool
        tool_args     : arguments dict from the agent's action
        db            : per-session OrderDatabase instance
        verified_facts: existing verified facts (gets updated in-place)
        ticket_id     : current ticket ID for logging

    Returns:
        result dict with at minimum a 'success' bool field.
        On failure, also includes 'error' field with a human-readable message.
    """
    if tool_name not in VALID_TOOLS:
        return {
            "success": False,
            "error": f"Unknown tool '{tool_name}'. Valid tools: {sorted(VALID_TOOLS)}",
        }

    handlers = {
        "lookup_order": _lookup_order,
        "process_refund": _process_refund,
        "search_kb": _search_kb,
        "escalate_to_human": _escalate_to_human,
        "cancel_subscription": _cancel_subscription,
        "check_payment": _check_payment,
    }

    return handlers[tool_name](tool_args, db, verified_facts, ticket_id)


# ---------------------------------------------------------------------------
# Individual tool handlers
# ---------------------------------------------------------------------------


def _lookup_order(
    args: Dict[str, Any],
    db: OrderDatabase,
    verified_facts: Dict[str, Any],
    ticket_id: str,
) -> Dict[str, Any]:
    """
    Look up an order by order_id.

    Required args: order_id (str)
    Returns: full order details (minus internal fraud fields, shown as risk_flag)
    """
    order_id = args.get("order_id", "").strip()
    if not order_id:
        return {"success": False, "error": "order_id is required for lookup_order."}

    order = db.get_order(order_id)
    if order is None:
        return {
            "success": False,
            "error": f"Order '{order_id}' not found in the system.",
        }

    # Populate verified_facts for reward calculation
    verified_facts["order_id"] = order["order_id"]
    verified_facts["order_status"] = order["status"]
    verified_facts["order_looked_up"] = True
    verified_facts["within_return_window"] = order["within_return_window"]
    verified_facts["is_damaged"] = order["is_damaged"]
    verified_facts["is_vip"] = order["is_vip"]
    verified_facts["is_fraud_risk"] = order["is_fraud_risk"]
    verified_facts["has_warranty"] = order["has_warranty"]
    verified_facts["refund_issued"] = order["refund_issued"]

    # Build a clean response (don't directly expose full fraud risk detail)
    return {
        "success": True,
        "order_id": order["order_id"],
        "customer_name": order["customer_name"],
        "customer_id": order["customer_id"],
        "is_vip": order["is_vip"],
        "items": order["items"],
        "total": order["total"],
        "payment_method": order["payment_method"],
        "status": order["status"],
        "order_date": order["order_date"],
        "delivery_date": order["delivery_date"],
        "days_since_delivery": order["days_since_delivery"],
        "return_window_days": order["return_window_days"],
        "within_return_window": order["within_return_window"],
        "is_damaged": order["is_damaged"],
        "has_warranty": order["has_warranty"],
        "warranty_months": order["warranty_months"],
        "subscription_id": order.get("subscription_id"),
        "refund_already_issued": order["refund_issued"],
        "risk_flag": "requires_specialist_review" if order["is_fraud_risk"] else None,
    }


def _process_refund(
    args: Dict[str, Any],
    db: OrderDatabase,
    verified_facts: Dict[str, Any],
    ticket_id: str,
) -> Dict[str, Any]:
    """
    Process a refund for an order.

    Required args: order_id (str), reason (str)
    Optional args: amount (float) — defaults to full order total
    """
    order_id = args.get("order_id", "").strip()
    reason = args.get("reason", "").strip()

    if not order_id:
        return {"success": False, "error": "order_id is required for process_refund."}
    if not reason:
        return {"success": False, "error": "reason is required for process_refund."}

    order = db.get_order(order_id)
    if order is None:
        return {"success": False, "error": f"Order '{order_id}' not found."}

    if order["refund_issued"]:
        return {
            "success": False,
            "error": f"A refund has already been issued for order '{order_id}'.",
        }

    # Guard: do NOT allow refund on fraud-risk orders
    if order["is_fraud_risk"]:
        return {
            "success": False,
            "error": (
                "This order has been flagged and requires specialist review before any refund can be processed. "
                "Please use escalate_to_human instead."
            ),
        }

    amount = args.get("amount", order["total"])
    try:
        amount = float(amount)
    except (ValueError, TypeError):
        return {"success": False, "error": "amount must be a number."}

    amount = min(amount, order["total"])  # Can't refund more than order total

    success = db.mark_refund_issued(order_id, amount)
    if not success:
        return {
            "success": False,
            "error": "Failed to process refund. Please try again.",
        }

    # Update verified facts
    verified_facts["refund_processed"] = True
    verified_facts["refund_amount"] = amount

    # Determine refund timeline based on payment method
    timeline = (
        "1-2 business days"
        if order["payment_method"] in ("UPI", "wallet")
        else "3-5 business days"
    )

    return {
        "success": True,
        "order_id": order_id,
        "refund_amount": amount,
        "refund_reason": reason,
        "timeline": timeline,
        "message": f"Refund of ₹{amount:.2f} initiated. Customer will receive it in {timeline}.",
    }


def _search_kb(
    args: Dict[str, Any],
    db: OrderDatabase,
    verified_facts: Dict[str, Any],
    ticket_id: str,
) -> Dict[str, Any]:
    """
    Search the knowledge base.

    Required args: query (str)
    Returns: top 3 matching articles with titles and content excerpts.
    """
    query = args.get("query", "").strip()
    if not query:
        return {"success": False, "error": "query is required for search_kb."}

    articles = search_kb(query, top_k=3)
    if not articles:
        return {
            "success": True,
            "results": [],
            "message": "No KB articles matched your query. Try different keywords.",
        }

    verified_facts["kb_searched"] = True
    verified_facts["kb_queries"] = verified_facts.get("kb_queries", []) + [query]

    return {
        "success": True,
        "query": query,
        "results": [
            {
                "id": a["id"],
                "title": a["title"],
                "content": a["content"],
            }
            for a in articles
        ],
    }


def _escalate_to_human(
    args: Dict[str, Any],
    db: OrderDatabase,
    verified_facts: Dict[str, Any],
    ticket_id: str,
) -> Dict[str, Any]:
    """
    Escalate the ticket to a human supervisor.

    Required args: reason (str)
    Optional args: priority (str) — 'normal' | 'high', defaults to 'normal'
    """
    reason = args.get("reason", "").strip()
    priority = args.get("priority", "normal")

    if not reason:
        return {"success": False, "error": "reason is required for escalate_to_human."}

    if priority not in ("normal", "high"):
        priority = "normal"

    verified_facts["escalated"] = True
    verified_facts["escalation_priority"] = priority

    eta = "30-60 minutes" if priority == "high" else "2-4 hours"

    return {
        "success": True,
        "ticket_id": ticket_id,
        "escalated_to": "Senior Support Agent",
        "priority": priority,
        "reason": reason,
        "estimated_response_time": eta,
        "message": f"Ticket escalated to a senior agent (priority: {priority}). Response expected in {eta}.",
    }


def _cancel_subscription(
    args: Dict[str, Any],
    db: OrderDatabase,
    verified_facts: Dict[str, Any],
    ticket_id: str,
) -> Dict[str, Any]:
    """
    Cancel a customer subscription.

    Required args: subscription_id (str), reason (str)
    """
    subscription_id = args.get("subscription_id", "").strip()
    reason = args.get("reason", "").strip()

    if not subscription_id:
        return {
            "success": False,
            "error": "subscription_id is required for cancel_subscription.",
        }
    if not reason:
        return {
            "success": False,
            "error": "reason is required for cancel_subscription.",
        }

    # Verify subscription exists (scan orders for this subscription ID)
    found = any(
        o.get("subscription_id") == subscription_id
        for o in [db.get_order(oid) for oid in db.list_order_ids()]
        if o
    )
    if not found:
        return {
            "success": False,
            "error": f"Subscription '{subscription_id}' not found.",
        }

    verified_facts["subscription_cancelled"] = True
    verified_facts["subscription_id"] = subscription_id

    return {
        "success": True,
        "subscription_id": subscription_id,
        "status": "cancelled",
        "effective_date": "End of current billing cycle",
        "message": f"Subscription {subscription_id} has been cancelled. It remains active until end of billing cycle.",
    }


def _check_payment(
    args: Dict[str, Any],
    db: OrderDatabase,
    verified_facts: Dict[str, Any],
    ticket_id: str,
) -> Dict[str, Any]:
    """
    Check payment details for an order (duplicate charge detection).

    Required args: order_id (str)
    """
    order_id = args.get("order_id", "").strip()
    if not order_id:
        return {"success": False, "error": "order_id is required for check_payment."}

    order = db.get_order(order_id)
    if order is None:
        return {"success": False, "error": f"Order '{order_id}' not found."}

    is_duplicate = order["is_duplicate_charge"]
    verified_facts["payment_checked"] = True
    verified_facts["is_duplicate_charge"] = is_duplicate

    if is_duplicate:
        return {
            "success": True,
            "order_id": order_id,
            "payment_method": order["payment_method"],
            "order_total": order["total"],
            "duplicate_detected": True,
            "transaction_count": 2,
            "total_charged": order["total"] * 2,
            "duplicate_amount": order["total"],
            "message": (
                f"DUPLICATE CHARGE CONFIRMED: Customer was charged ₹{order['total']:.2f} twice "
                f"for order {order_id}. Refund of ₹{order['total']:.2f} should be issued."
            ),
        }
    else:
        return {
            "success": True,
            "order_id": order_id,
            "payment_method": order["payment_method"],
            "order_total": order["total"],
            "duplicate_detected": False,
            "transaction_count": 1,
            "message": "Payment records show only one transaction. No duplicate charge found.",
        }

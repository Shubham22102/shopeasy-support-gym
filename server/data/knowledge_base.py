"""
Knowledge Base for ShopEasy customer support.

Contains 20+ articles indexed by keyword. The search_kb tool queries this
module. Articles cover policies, processes, and common issues.
"""

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Article schema
# ---------------------------------------------------------------------------

KBArticle = Dict[str, Any]  # {id, title, keywords, content}


KB_ARTICLES: List[KBArticle] = [
    {
        "id": "KB-001",
        "title": "Return & Refund Policy — Electronics",
        "keywords": ["return", "refund", "electronics", "policy", "window"],
        "content": (
            "Electronics items (smartphones, laptops, earbuds, speakers, keyboards, etc.) "
            "have a 10-day return window from the date of delivery. "
            "To be eligible, the item must be in original packaging with all accessories. "
            "If the item is defective or damaged on arrival, an instant full refund is issued regardless of the 10-day window. "
            "After 10 days, only store credit is offered for items in resalable condition. "
            "No refund is possible after 30 days for any reason except warranty claims."
        ),
    },
    {
        "id": "KB-002",
        "title": "Return & Refund Policy — Clothing & Apparel",
        "keywords": ["return", "refund", "clothing", "apparel", "fashion", "policy"],
        "content": (
            "Clothing and apparel have a 30-day return window from delivery. "
            "Items must be unworn, unwashed, with tags intact. "
            "Sale items are only eligible for store credit, not cash refunds. "
            "Damaged or incorrectly shipped items are eligible for full refund anytime."
        ),
    },
    {
        "id": "KB-003",
        "title": "Return & Refund Policy — Home Appliances",
        "keywords": ["return", "refund", "home", "appliances", "kitchen", "policy"],
        "content": (
            "Home appliances have a 14-day return window from delivery. "
            "The item must be in original packaging and unused. "
            "Defective appliances qualify for a replacement or full refund under the 12-month warranty. "
            "Installation damage is not covered under the return policy."
        ),
    },
    {
        "id": "KB-004",
        "title": "Return & Refund Policy — Books",
        "keywords": ["return", "refund", "books", "policy"],
        "content": (
            "Books have a 15-day return window from delivery. "
            "The book must be unread and in original condition. "
            "Digital/eBook purchases are non-refundable once accessed. "
            "Damaged books on arrival qualify for a full refund."
        ),
    },
    {
        "id": "KB-005",
        "title": "Return & Refund Policy — Sports & Fitness",
        "keywords": ["return", "refund", "sports", "fitness", "gym", "policy"],
        "content": (
            "Sports and fitness equipment has a 21-day return window. "
            "Items must be unused and in original packaging. "
            "Consumables like protein supplements and resistance bands are non-returnable once opened."
        ),
    },
    {
        "id": "KB-006",
        "title": "Return & Refund Policy — Beauty & Personal Care",
        "keywords": ["return", "refund", "beauty", "skincare", "cosmetics", "policy"],
        "content": (
            "Beauty and personal care products have a 7-day return window. "
            "Products must be unopened and in original packaging for hygiene reasons. "
            "Once opened, beauty products cannot be returned unless defective."
        ),
    },
    {
        "id": "KB-007",
        "title": "Damaged Item Policy",
        "keywords": [
            "damaged",
            "broken",
            "defective",
            "arrived damaged",
            "instant refund",
        ],
        "content": (
            "If a customer receives a damaged or defective item, they are entitled to an instant full refund "
            "OR a replacement, regardless of the return window for their product category. "
            "The customer does NOT need to return the damaged item. "
            "Agent should: (1) Confirm damage report, (2) offer full refund or replacement, (3) process immediately. "
            "No further approval is required for damaged items — agents are authorized to process this directly."
        ),
    },
    {
        "id": "KB-008",
        "title": "Wrong Item Received Policy",
        "keywords": ["wrong item", "incorrect item", "wrong product", "sent wrong"],
        "content": (
            "If a customer receives an incorrect item (not what they ordered), they are entitled to: "
            "(1) A full refund, OR (2) Re-shipment of the correct item. "
            "This applies regardless of return window. The customer is NOT required to return the wrong item. "
            "Agent should: verify order details, apologize, and process the refund or re-shipment."
        ),
    },
    {
        "id": "KB-009",
        "title": "Order Not Received / Lost in Transit",
        "keywords": [
            "not received",
            "lost",
            "missing",
            "never arrived",
            "lost in transit",
        ],
        "content": (
            "If a customer's order shows 'delivered' but was not received: "
            "(1) Wait 2 business days (sometimes packages arrive late after scan). "
            "(2) Check with neighbours or building management. "
            "(3) If still not received after 2 days, a replacement or full refund will be issued. "
            "If the order status is 'lost_in_transit': issue immediate full refund or free re-shipment."
        ),
    },
    {
        "id": "KB-010",
        "title": "Duplicate Charge / Payment Dispute Policy",
        "keywords": [
            "duplicate",
            "charged twice",
            "double charge",
            "payment",
            "billing",
        ],
        "content": (
            "If a customer was charged more than once for the same order: "
            "(1) Use the check_payment tool to verify duplicate transactions. "
            "(2) If confirmed, issue a refund for the duplicate amount immediately. "
            "(3) Apologize and confirm the refund timeline (3-5 business days). "
            "Do NOT process a refund without verifying the duplicate charge first."
        ),
    },
    {
        "id": "KB-011",
        "title": "Warranty Policy — Electronics & Appliances",
        "keywords": [
            "warranty",
            "guarantee",
            "broken",
            "stopped working",
            "malfunction",
        ],
        "content": (
            "Electronics and home appliances come with a 12-month manufacturer warranty from delivery date. "
            "Warranty covers manufacturing defects but NOT: physical damage, water damage, or misuse. "
            "To claim: (1) Verify purchase is within 12 months, (2) describe the defect, (3) initiate warranty claim. "
            "Warranty claims result in free repair, replacement, or full refund (in that order of preference)."
        ),
    },
    {
        "id": "KB-012",
        "title": "Subscription Cancellation Policy",
        "keywords": ["subscription", "cancel", "recurring", "membership", "auto-renew"],
        "content": (
            "ShopEasy Plus subscriptions can be cancelled at any time. "
            "Cancellation takes effect at the end of the current billing cycle. "
            "No partial refunds are issued for unused subscription days. "
            "Exception: if the customer was charged AFTER cancellation, a full refund is due. "
            "Use the cancel_subscription tool with the subscription_id."
        ),
    },
    {
        "id": "KB-013",
        "title": "VIP Customer (ShopEasy Diamond) Benefits",
        "keywords": ["vip", "diamond", "premium", "loyal customer", "priority"],
        "content": (
            "ShopEasy Diamond VIP customers receive priority support. "
            "VIP benefits: extended return window (+7 days for all categories), free returns, "
            "dedicated support queue, and complaints escalated to senior agents automatically. "
            "Identify VIP status via the is_vip=True flag in lookup_order results. "
            "Always acknowledge VIP status and thank the customer for their loyalty."
        ),
    },
    {
        "id": "KB-014",
        "title": "Fraud Prevention Policy",
        "keywords": ["fraud", "suspicious", "scam", "unauthorized", "account hacked"],
        "content": (
            "If an order has is_fraud_risk=True (from lookup_order), the agent MUST: "
            "(1) NOT process any refund directly. "
            "(2) Politely inform the customer that the request requires specialist review. "
            "(3) Use escalate_to_human with priority='high'. "
            "Do NOT accuse the customer of fraud. Use neutral language: 'this requires additional verification.' "
            "Refunds on fraud-risk orders are only approved by the fraud team."
        ),
    },
    {
        "id": "KB-015",
        "title": "Store Credit Policy",
        "keywords": ["store credit", "credit", "voucher", "alternative refund"],
        "content": (
            "Store credit is offered when: "
            "(1) Return window has expired but item is in good condition. "
            "(2) Customer purchased a sale item. "
            "(3) Customer preference. "
            "Store credit is valid for 12 months and can be used on any ShopEasy purchase. "
            "It cannot be converted back to cash. Minimum store credit issued: ₹100."
        ),
    },
    {
        "id": "KB-016",
        "title": "Shipping Delay Compensation Policy",
        "keywords": [
            "delay",
            "late",
            "shipping delay",
            "compensation",
            "late delivery",
        ],
        "content": (
            "If an order is more than 5 days past its estimated delivery date: "
            "(1) Customer is entitled to a ₹100 shipping refund coupon. "
            "If more than 10 days late: (2) Customer is entitled to a 10% order value coupon. "
            "These coupons are applied automatically upon escalation to support. "
            "Agents should acknowledge the delay and share the coupon code."
        ),
    },
    {
        "id": "KB-017",
        "title": "How to Process a Refund",
        "keywords": [
            "how to refund",
            "process refund",
            "initiate refund",
            "refund steps",
        ],
        "content": (
            "To process a refund as an agent: "
            "(1) Always use lookup_order first to verify order details. "
            "(2) Confirm eligibility (return window, damage status, etc.) via KB search. "
            "(3) Use the process_refund tool with order_id, reason, and amount. "
            "(4) Inform customer of refund timeline: 3-5 business days for card payments, "
            "    1-2 days for UPI/wallets. "
            "(5) Never promise a refund before verifying the order."
        ),
    },
    {
        "id": "KB-018",
        "title": "Escalation Guidelines for Human Agents",
        "keywords": ["escalate", "human", "supervisor", "manager", "escalation"],
        "content": (
            "Escalate to a human agent when: "
            "(1) Order has is_fraud_risk=True. "
            "(2) Customer explicitly demands a manager and the issue cannot be resolved by policy. "
            "(3) Refund amount exceeds ₹50,000. "
            "(4) Legal threats are made. "
            "Use escalate_to_human tool with reason and priority ('normal' or 'high'). "
            "Do NOT escalate for issues that can be resolved under standard policy — "
            "this increases wait time for customers unnecessarily."
        ),
    },
    {
        "id": "KB-019",
        "title": "COD (Cash on Delivery) Order Policies",
        "keywords": ["COD", "cash on delivery", "cash payment", "paid on delivery"],
        "content": (
            "For COD orders: "
            "(1) Refunds are issued as store credit or bank transfer (customer provides bank details). "
            "(2) COD orders cannot be cancelled after they leave the warehouse. "
            "(3) COD refunds take 5-7 business days. "
            "(4) Damaged COD items follow the standard damaged item policy."
        ),
    },
    {
        "id": "KB-020",
        "title": "Customer Complaint Handling — Tone Guide",
        "keywords": ["complaint", "angry", "de-escalate", "tone", "apology"],
        "content": (
            "When handling an angry or frustrated customer: "
            "(1) Start with empathy: 'I understand this situation is frustrating.' "
            "(2) Apologize sincerely: 'I'm sorry for the inconvenience.' "
            "(3) Do not argue or be defensive. "
            "(4) Focus on solutions, not explanations. "
            "(5) Set clear expectations: 'I'll have this resolved within X...'. "
            "(6) Avoid: 'That's our policy', 'There's nothing I can do', 'You should have...'. "
            "A calm, solution-focused agent reduces escalations by 60%."
        ),
    },
]


# ---------------------------------------------------------------------------
# Search engine (simple keyword matching)
# ---------------------------------------------------------------------------


def search_kb(query: str, top_k: int = 3) -> List[KBArticle]:
    """
    Search the knowledge base by keyword.

    Returns the top_k most relevant articles based on keyword overlap.
    Uses simple bag-of-words scoring — no embeddings needed.
    """
    query_words = set(query.lower().split())
    scored: List[tuple] = []

    for article in KB_ARTICLES:
        keyword_words = set(" ".join(article["keywords"]).lower().split())
        overlap = len(query_words & keyword_words)
        title_words = set(article["title"].lower().split())
        title_overlap = len(query_words & title_words)
        score = overlap * 1.0 + title_overlap * 0.5
        if score > 0:
            scored.append((score, article))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[:top_k]]


def get_article_by_id(article_id: str) -> Optional[KBArticle]:
    """Return an article by its KB-XXX ID."""
    for article in KB_ARTICLES:
        if article["id"] == article_id:
            return article
    return None

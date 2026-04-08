# tasks/definitions.py
# Task step definitions for the OpenEnv validator.
# Each task has 3 steps with observations (text) and signals (dict).

TASKS = {
    "simple_refund": {
        "description": "Easy: Customer wants a refund within the return window.",
        "ideal_action": "refund",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket — ORDER #10241:\n"
                    "Customer: 'I bought this shirt 3 days ago and it doesn't fit. "
                    "I want my money back.'\n"
                    "Order status: delivered\n"
                    "Within return window: Yes\n"
                    "Item condition: unused\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "within_return_window": True,
                    "order_status": "delivered",
                    "item_condition": "unused",
                    "customer_sentiment": 0.3,
                    "complexity": 0.2,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #10242:\n"
                    "Customer: 'These shoes are the wrong size and I'd like a refund.'\n"
                    "Order status: delivered\n"
                    "Within return window: Yes\n"
                    "Item condition: new in box\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "within_return_window": True,
                    "order_status": "delivered",
                    "item_condition": "new",
                    "customer_sentiment": 0.4,
                    "complexity": 0.2,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #10243:\n"
                    "Customer: 'I changed my mind about this purchase. Can I get a refund?'\n"
                    "Order status: delivered\n"
                    "Within return window: Yes\n"
                    "Days since delivery: 5\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "within_return_window": True,
                    "order_status": "delivered",
                    "item_condition": "unused",
                    "customer_sentiment": 0.5,
                    "complexity": 0.15,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
        ],
    },

    "delivery_tracking": {
        "description": "Easy: Customer asks where their order is.",
        "ideal_action": "lookup_order",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket — ORDER #20301:\n"
                    "Customer: 'Where is my package? It was supposed to arrive yesterday.'\n"
                    "Order status: in_transit\n"
                    "Estimated delivery: overdue by 1 day\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "order_status": "in_transit",
                    "is_overdue": True,
                    "days_overdue": 1,
                    "customer_sentiment": 0.2,
                    "complexity": 0.2,
                    "is_fraud_risk": False,
                    "ideal_action": "lookup_order",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #20302:\n"
                    "Customer: 'Can you check the status of my order? I haven't received it.'\n"
                    "Order status: shipped\n"
                    "Tracking available: Yes\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "order_status": "shipped",
                    "is_overdue": False,
                    "days_overdue": 0,
                    "customer_sentiment": 0.4,
                    "complexity": 0.15,
                    "is_fraud_risk": False,
                    "ideal_action": "lookup_order",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #20303:\n"
                    "Customer: 'My order says delivered but I never got it.'\n"
                    "Order status: delivered\n"
                    "Delivery confirmation: signed\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "order_status": "delivered",
                    "is_overdue": False,
                    "delivery_confirmed": True,
                    "customer_sentiment": 0.1,
                    "complexity": 0.4,
                    "is_fraud_risk": False,
                    "ideal_action": "lookup_order",
                },
            },
        ],
    },

    "kb_policy_question": {
        "description": "Easy: Customer asks about return policy.",
        "ideal_action": "search_kb",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket:\n"
                    "Customer: 'What is your return policy? How many days do I have?'\n"
                    "No active order referenced.\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "question_type": "policy",
                    "has_active_order": False,
                    "customer_sentiment": 0.5,
                    "complexity": 0.1,
                    "is_fraud_risk": False,
                    "ideal_action": "search_kb",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket:\n"
                    "Customer: 'Do you offer free shipping on returns?'\n"
                    "No active order referenced.\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "question_type": "shipping_policy",
                    "has_active_order": False,
                    "customer_sentiment": 0.6,
                    "complexity": 0.1,
                    "is_fraud_risk": False,
                    "ideal_action": "search_kb",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket:\n"
                    "Customer: 'Can I exchange an item instead of getting a refund?'\n"
                    "No active order referenced.\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "question_type": "exchange_policy",
                    "has_active_order": False,
                    "customer_sentiment": 0.5,
                    "complexity": 0.15,
                    "is_fraud_risk": False,
                    "ideal_action": "search_kb",
                },
            },
        ],
    },

    "cancellation_request": {
        "description": "Easy: Customer wants to cancel a pending order.",
        "ideal_action": "resolve",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket — ORDER #30401:\n"
                    "Customer: 'I want to cancel my order. It hasn't shipped yet.'\n"
                    "Order status: processing\n"
                    "Shipped: No\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "order_status": "processing",
                    "is_shipped": False,
                    "cancellation_eligible": True,
                    "customer_sentiment": 0.4,
                    "complexity": 0.15,
                    "is_fraud_risk": False,
                    "ideal_action": "resolve",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #30402:\n"
                    "Customer: 'Please cancel order #30402. I ordered it by mistake.'\n"
                    "Order status: pending\n"
                    "Shipped: No\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "order_status": "pending",
                    "is_shipped": False,
                    "cancellation_eligible": True,
                    "customer_sentiment": 0.5,
                    "complexity": 0.1,
                    "is_fraud_risk": False,
                    "ideal_action": "resolve",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #30403:\n"
                    "Customer: 'I need to cancel. I found it cheaper elsewhere.'\n"
                    "Order status: processing\n"
                    "Shipped: No\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "order_status": "processing",
                    "is_shipped": False,
                    "cancellation_eligible": True,
                    "customer_sentiment": 0.3,
                    "complexity": 0.1,
                    "is_fraud_risk": False,
                    "ideal_action": "resolve",
                },
            },
        ],
    },

    "expired_return": {
        "description": "Medium: Return request outside the policy window.",
        "ideal_action": "search_kb",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket — ORDER #40501:\n"
                    "Customer: 'I want to return this item I bought 45 days ago.'\n"
                    "Order status: delivered\n"
                    "Within return window: No (30-day policy)\n"
                    "Days since delivery: 45\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "within_return_window": False,
                    "order_status": "delivered",
                    "days_since_delivery": 45,
                    "customer_sentiment": 0.2,
                    "complexity": 0.5,
                    "is_fraud_risk": False,
                    "ideal_action": "search_kb",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #40502:\n"
                    "Customer: 'I know it's been a while but the product broke. Can I return it?'\n"
                    "Order status: delivered\n"
                    "Within return window: No\n"
                    "Days since delivery: 60\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "within_return_window": False,
                    "order_status": "delivered",
                    "days_since_delivery": 60,
                    "customer_sentiment": 0.15,
                    "complexity": 0.55,
                    "is_fraud_risk": False,
                    "ideal_action": "search_kb",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #40503:\n"
                    "Customer: 'I missed the return deadline. Is there anything you can do?'\n"
                    "Order status: delivered\n"
                    "Within return window: No\n"
                    "Days since delivery: 35\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "within_return_window": False,
                    "order_status": "delivered",
                    "days_since_delivery": 35,
                    "customer_sentiment": 0.3,
                    "complexity": 0.45,
                    "is_fraud_risk": False,
                    "ideal_action": "search_kb",
                },
            },
        ],
    },

    "wrong_item_sent": {
        "description": "Medium: Customer received incorrect product.",
        "ideal_action": "refund",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket — ORDER #50601:\n"
                    "Customer: 'I ordered a blue jacket but received a red one.'\n"
                    "Order status: delivered\n"
                    "Item match: MISMATCH\n"
                    "Within return window: Yes\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "item_mismatch": True,
                    "within_return_window": True,
                    "order_status": "delivered",
                    "customer_sentiment": 0.15,
                    "complexity": 0.4,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #50602:\n"
                    "Customer: 'This is not what I ordered at all! Completely wrong product.'\n"
                    "Order status: delivered\n"
                    "Item match: MISMATCH\n"
                    "Within return window: Yes\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "item_mismatch": True,
                    "within_return_window": True,
                    "order_status": "delivered",
                    "customer_sentiment": 0.05,
                    "complexity": 0.45,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #50603:\n"
                    "Customer: 'I got size L but ordered size M. Need this fixed.'\n"
                    "Order status: delivered\n"
                    "Item match: SIZE_MISMATCH\n"
                    "Within return window: Yes\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "item_mismatch": True,
                    "within_return_window": True,
                    "order_status": "delivered",
                    "customer_sentiment": 0.2,
                    "complexity": 0.35,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
        ],
    },

    "duplicate_charge": {
        "description": "Medium: Customer was charged twice.",
        "ideal_action": "refund",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket — ORDER #60701:\n"
                    "Customer: 'I was charged twice for the same order! Check my statement.'\n"
                    "Order status: delivered\n"
                    "Payment records: 2 charges found\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "duplicate_charge_confirmed": True,
                    "order_status": "delivered",
                    "payment_verified": True,
                    "customer_sentiment": 0.1,
                    "complexity": 0.5,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #60702:\n"
                    "Customer: 'You took money from my account twice. I want one charge refunded.'\n"
                    "Order status: delivered\n"
                    "Payment records: 2 charges found\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "duplicate_charge_confirmed": True,
                    "order_status": "delivered",
                    "payment_verified": True,
                    "customer_sentiment": 0.05,
                    "complexity": 0.5,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #60703:\n"
                    "Customer: 'My bank shows two transactions for this purchase.'\n"
                    "Order status: delivered\n"
                    "Payment records: investigating\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "duplicate_charge_confirmed": True,
                    "order_status": "delivered",
                    "payment_verified": False,
                    "customer_sentiment": 0.2,
                    "complexity": 0.55,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
        ],
    },

    "partial_order": {
        "description": "Medium: Customer only received part of their order.",
        "ideal_action": "refund",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket — ORDER #70801:\n"
                    "Customer: 'I ordered 3 items but only received 2. Where is the third?'\n"
                    "Order status: delivered\n"
                    "Items ordered: 3, Items delivered: 2\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "partial_delivery": True,
                    "items_missing": 1,
                    "order_status": "delivered",
                    "customer_sentiment": 0.2,
                    "complexity": 0.4,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #70802:\n"
                    "Customer: 'Half my order is missing. I only got the small items.'\n"
                    "Order status: delivered\n"
                    "Items ordered: 4, Items delivered: 2\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "partial_delivery": True,
                    "items_missing": 2,
                    "order_status": "delivered",
                    "customer_sentiment": 0.1,
                    "complexity": 0.45,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #70803:\n"
                    "Customer: 'The box arrived but one item was missing inside.'\n"
                    "Order status: delivered\n"
                    "Items ordered: 2, Items delivered: 1\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "partial_delivery": True,
                    "items_missing": 1,
                    "order_status": "delivered",
                    "customer_sentiment": 0.25,
                    "complexity": 0.35,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
        ],
    },

    "damaged_item": {
        "description": "Hard: Item arrived broken — instant refund policy applies.",
        "ideal_action": "refund",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket — ORDER #80901:\n"
                    "Customer: 'The vase I ordered arrived completely shattered!'\n"
                    "Order status: delivered\n"
                    "Item condition: damaged\n"
                    "Photo provided: Yes\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "is_damaged": True,
                    "photo_evidence": True,
                    "order_status": "delivered",
                    "customer_sentiment": 0.05,
                    "complexity": 0.5,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #80902:\n"
                    "Customer: 'My laptop screen is cracked. It was damaged in shipping.'\n"
                    "Order status: delivered\n"
                    "Item condition: damaged\n"
                    "Item value: high\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "is_damaged": True,
                    "photo_evidence": True,
                    "order_status": "delivered",
                    "item_value": "high",
                    "customer_sentiment": 0.0,
                    "complexity": 0.6,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #80903:\n"
                    "Customer: 'The packaging was crushed and the item inside is broken.'\n"
                    "Order status: delivered\n"
                    "Item condition: damaged\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "is_damaged": True,
                    "photo_evidence": False,
                    "order_status": "delivered",
                    "customer_sentiment": 0.1,
                    "complexity": 0.5,
                    "is_fraud_risk": False,
                    "ideal_action": "refund",
                },
            },
        ],
    },

    "angry_customer": {
        "description": "Hard: Extremely angry customer, agent must de-escalate.",
        "ideal_action": "resolve",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket — ORDER #91001:\n"
                    "Customer: 'THIS IS RIDICULOUS! I've been waiting 2 WEEKS! "
                    "This is the WORST service I've ever experienced!'\n"
                    "Order status: in_transit (delayed)\n"
                    "Customer tier: regular\n"
                    "Previous complaints: 0\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "customer_sentiment": -0.8,
                    "order_status": "in_transit",
                    "is_delayed": True,
                    "customer_tier": "regular",
                    "complexity": 0.7,
                    "is_fraud_risk": False,
                    "ideal_action": "resolve",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #91002:\n"
                    "Customer: 'I want to speak to a MANAGER! Your company is a SCAM!'\n"
                    "Order status: delivered (wrong item)\n"
                    "Customer tier: regular\n"
                    "Previous complaints: 2\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "customer_sentiment": -0.9,
                    "order_status": "delivered",
                    "item_mismatch": True,
                    "customer_tier": "regular",
                    "complexity": 0.75,
                    "is_fraud_risk": False,
                    "ideal_action": "resolve",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #91003:\n"
                    "Customer: 'I'M DONE WITH THIS STORE! Fix this NOW or I'll dispute the charge!'\n"
                    "Order status: delivered\n"
                    "Issue: item not as described\n"
                    "Customer tier: regular\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "customer_sentiment": -0.85,
                    "order_status": "delivered",
                    "customer_tier": "regular",
                    "complexity": 0.7,
                    "is_fraud_risk": False,
                    "ideal_action": "resolve",
                },
            },
        ],
    },

    "fraud_risk": {
        "description": "Hard: Suspicious request must be escalated, not refunded.",
        "ideal_action": "escalate",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket — ORDER #101101:\n"
                    "Customer: 'I need an immediate refund. The item never arrived.'\n"
                    "Order status: delivered (confirmed signature)\n"
                    "Fraud flags: new account, high-value item, address mismatch\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "is_fraud_risk": True,
                    "fraud_score": 0.85,
                    "order_status": "delivered",
                    "delivery_confirmed": True,
                    "customer_sentiment": 0.3,
                    "complexity": 0.8,
                    "ideal_action": "escalate",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #101102:\n"
                    "Customer: 'Refund me now! I already threw out the item.'\n"
                    "Order status: delivered\n"
                    "Fraud flags: multiple refund requests, inconsistent story\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "is_fraud_risk": True,
                    "fraud_score": 0.9,
                    "order_status": "delivered",
                    "delivery_confirmed": True,
                    "customer_sentiment": 0.1,
                    "complexity": 0.85,
                    "ideal_action": "escalate",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #101103:\n"
                    "Customer: 'I want a refund. The quality is terrible.'\n"
                    "Order status: delivered\n"
                    "Fraud flags: 5 refunds in 30 days, different addresses\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "is_fraud_risk": True,
                    "fraud_score": 0.8,
                    "order_status": "delivered",
                    "delivery_confirmed": True,
                    "customer_sentiment": 0.2,
                    "complexity": 0.75,
                    "ideal_action": "escalate",
                },
            },
        ],
    },

    "vip_warranty_claim": {
        "description": "Hard: VIP customer warranty claim for electronics.",
        "ideal_action": "resolve",
        "steps": [
            {
                "observation": (
                    "Customer Support Ticket — ORDER #111201:\n"
                    "Customer: 'My TV stopped working after 8 months. I have the extended warranty.'\n"
                    "Order status: delivered\n"
                    "Customer tier: VIP (Platinum)\n"
                    "Warranty status: active (extended)\n"
                    "Item category: electronics\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "is_vip": True,
                    "has_warranty": True,
                    "warranty_active": True,
                    "customer_tier": "vip",
                    "item_category": "electronics",
                    "customer_sentiment": 0.2,
                    "complexity": 0.7,
                    "is_fraud_risk": False,
                    "ideal_action": "resolve",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #111202:\n"
                    "Customer: 'My headphones broke. I'm a Platinum member and expect this resolved quickly.'\n"
                    "Order status: delivered\n"
                    "Customer tier: VIP (Platinum)\n"
                    "Warranty status: active\n"
                    "Item category: electronics\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "is_vip": True,
                    "has_warranty": True,
                    "warranty_active": True,
                    "customer_tier": "vip",
                    "item_category": "electronics",
                    "customer_sentiment": 0.15,
                    "complexity": 0.65,
                    "is_fraud_risk": False,
                    "ideal_action": "resolve",
                },
            },
            {
                "observation": (
                    "Customer Support Ticket — ORDER #111203:\n"
                    "Customer: 'My laptop keyboard is malfunctioning. I need a replacement ASAP.'\n"
                    "Order status: delivered\n"
                    "Customer tier: VIP (Gold)\n"
                    "Warranty status: active\n"
                    "Item category: electronics\n"
                    "Decide: resolve, escalate, refund, search_kb, lookup_order, or hold."
                ),
                "signals": {
                    "is_vip": True,
                    "has_warranty": True,
                    "warranty_active": True,
                    "customer_tier": "vip",
                    "item_category": "electronics",
                    "customer_sentiment": 0.1,
                    "complexity": 0.7,
                    "is_fraud_risk": False,
                    "ideal_action": "resolve",
                },
            },
        ],
    },
}

TASK_NAMES = list(TASKS.keys())

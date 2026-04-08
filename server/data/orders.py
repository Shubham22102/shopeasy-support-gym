"""
Synthetic order database for ShopEasy.

Contains 100 realistic fake orders with varied statuses, categories,
delivery dates, and edge cases to stress-test agent policies.
"""

import random
from copy import deepcopy
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = ["electronics", "clothing", "books", "home_appliances", "sports", "beauty"]

RETURN_WINDOW_BY_CATEGORY = {
    "electronics": 10,
    "clothing": 30,
    "books": 15,
    "home_appliances": 14,
    "sports": 21,
    "beauty": 7,
}

STATUSES = ["pending", "shipped", "delivered", "cancelled", "lost_in_transit"]

PAYMENT_METHODS = ["UPI", "credit_card", "debit_card", "net_banking", "wallet", "COD"]

PRODUCTS = {
    "electronics": [
        ("Wireless Earbuds", 2499),
        ("Bluetooth Speaker", 3499),
        ("Smartwatch", 8999),
        ("USB-C Hub", 1299),
        ("Mechanical Keyboard", 5999),
        ("Webcam HD", 2999),
        ("Laptop Stand", 1499),
        ("Phone Charger 65W", 999),
    ],
    "clothing": [
        ("Cotton T-Shirt Pack (3)", 799),
        ("Slim Fit Jeans", 1499),
        ("Formal Shirt", 899),
        ("Sports Jacket", 2299),
        ("Winter Hoodie", 1799),
    ],
    "books": [
        ("Deep Learning (Goodfellow)", 1299),
        ("The Lean Startup", 499),
        ("Atomic Habits", 399),
        ("System Design Interview", 899),
    ],
    "home_appliances": [
        ("Air Purifier", 9999),
        ("Electric Kettle", 1299),
        ("Stand Mixer", 4499),
        ("Robot Vacuum", 18999),
    ],
    "sports": [
        ("Yoga Mat Premium", 1299),
        ("Resistance Bands Set", 699),
        ("Adjustable Dumbbells", 3999),
        ("Cycling Helmet", 1799),
    ],
    "beauty": [
        ("Vitamin C Serum", 899),
        ("Hair Dryer Pro", 2499),
        ("Moisturiser SPF50", 699),
        ("Electric Shaver", 3299),
    ],
}

CUSTOMER_NAMES = [
    "Priya Sharma",
    "Rahul Verma",
    "Ananya Patel",
    "Kiran Reddy",
    "Suresh Nair",
    "Deepika Menon",
    "Arjun Gupta",
    "Sneha Iyer",
    "Vikram Joshi",
    "Pooja Agarwal",
    "Aditya Singh",
    "Meera Krishnan",
    "Rohan Mehta",
    "Kavya Pillai",
    "Nikhil Tiwari",
    "Sanya Kapoor",
    "Harsh Malhotra",
    "Riya Bose",
    "Dev Choudhary",
    "Ishaan Bansal",
    "Tanvi Saxena",
    "Ayaan Khan",
    "Diya Nambiar",
    "Parth Shah",
    "Kritika Rao",
    "Aarav Mishra",
    "Shruti Bhatt",
    "Yash Pandey",
    "Naina Desai",
    "Kabir Sood",
]


# ---------------------------------------------------------------------------
# Order generation
# ---------------------------------------------------------------------------


def _random_order(order_num: int, seed_offset: int = 0) -> Dict[str, Any]:
    """Generate one synthetic order deterministically from an index."""
    rng = random.Random(order_num + seed_offset * 1000)

    category = rng.choice(CATEGORIES)
    product_name, unit_price = rng.choice(PRODUCTS[category])
    qty = rng.randint(1, 3)
    total = unit_price * qty

    status = rng.choice(STATUSES)
    order_date = date(2026, 1, 1) + timedelta(days=rng.randint(0, 85))

    if status == "delivered":
        delivery_date = order_date + timedelta(days=rng.randint(3, 10))
    elif status == "shipped":
        delivery_date = order_date + timedelta(days=rng.randint(7, 14))
    else:
        delivery_date = None

    days_since_delivery = (
        (date(2026, 4, 6) - delivery_date).days if delivery_date else None
    )

    customer_idx = rng.randint(0, len(CUSTOMER_NAMES) - 1)
    customer_name = CUSTOMER_NAMES[customer_idx]
    customer_id = f"C-{100 + customer_idx}"

    return_window = RETURN_WINDOW_BY_CATEGORY[category]
    within_return_window = (
        days_since_delivery is not None and days_since_delivery <= return_window
        if status == "delivered"
        else False
    )

    is_damaged = rng.random() < 0.08  # 8% chance damaged
    is_duplicate_charge = rng.random() < 0.05  # 5% duplicate charge
    is_fraud_risk = rng.random() < 0.04  # 4% fraud flag
    has_warranty = category in ("electronics", "home_appliances")
    warranty_months = 12 if has_warranty else 0

    subscription_id = None
    if rng.random() < 0.15:
        subscription_id = f"SUB-{rng.randint(1000, 9999)}"

    is_vip = rng.random() < 0.10  # 10% VIP customers

    return {
        "order_id": f"SE-{1000 + order_num}",
        "customer_id": customer_id,
        "customer_name": customer_name,
        "customer_email": f"{customer_name.split()[0].lower()}@example.com",
        "is_vip": is_vip,
        "items": [
            {
                "name": product_name,
                "category": category,
                "qty": qty,
                "unit_price": unit_price,
                "subtotal": total,
            }
        ],
        "total": total,
        "payment_method": rng.choice(PAYMENT_METHODS),
        "status": status,
        "order_date": order_date.isoformat(),
        "delivery_date": delivery_date.isoformat() if delivery_date else None,
        "days_since_delivery": days_since_delivery,
        "return_window_days": return_window,
        "within_return_window": within_return_window,
        "is_damaged": is_damaged,
        "is_duplicate_charge": is_duplicate_charge,
        "is_fraud_risk": is_fraud_risk,
        "has_warranty": has_warranty,
        "warranty_months": warranty_months,
        "subscription_id": subscription_id,
        "refund_issued": False,
        "refund_amount": 0.0,
        "notes": "",
    }


# ---------------------------------------------------------------------------
# In-memory order store (100 orders, reset-able)
# ---------------------------------------------------------------------------

_MASTER_ORDERS: List[Dict[str, Any]] = [_random_order(i) for i in range(100)]


class OrderDatabase:
    """
    Isolated per-session order store.

    Each environment instance gets its own copy of the master orders so that
    concurrent RL training workers do not interfere with each other.
    """

    __slots__ = ["_orders"]  # Reduce per-instance memory overhead

    def __init__(self):
        # Deep copy so mutations in one session don't affect others
        self._orders: Dict[str, Dict[str, Any]] = {
            o["order_id"]: deepcopy(o) for o in _MASTER_ORDERS
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Return order dict or None if not found."""
        return deepcopy(self._orders.get(order_id))

    def get_orders_by_customer(self, customer_id: str) -> List[Dict[str, Any]]:
        return [
            deepcopy(o)
            for o in self._orders.values()
            if o["customer_id"] == customer_id
        ]

    def list_order_ids(self) -> List[str]:
        return list(self._orders.keys())

    def get_random_order(
        self,
        rng: random.Random,
        status: Optional[str] = None,
        category: Optional[str] = None,
        is_damaged: Optional[bool] = None,
        within_return_window: Optional[bool] = None,
        is_vip: Optional[bool] = None,
        is_fraud_risk: Optional[bool] = None,
        has_subscription: Optional[bool] = None,
        is_duplicate_charge: Optional[bool] = None,
        has_warranty: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a random order matching the given filter criteria."""
        candidates = list(self._orders.values())

        if status is not None:
            candidates = [o for o in candidates if o["status"] == status]
        if category is not None:
            candidates = [
                o
                for o in candidates
                if any(item["category"] == category for item in o["items"])
            ]
        if is_damaged is not None:
            candidates = [o for o in candidates if o["is_damaged"] == is_damaged]
        if within_return_window is not None:
            candidates = [
                o
                for o in candidates
                if o["within_return_window"] == within_return_window
            ]
        if is_vip is not None:
            candidates = [o for o in candidates if o["is_vip"] == is_vip]
        if is_fraud_risk is not None:
            candidates = [o for o in candidates if o["is_fraud_risk"] == is_fraud_risk]
        if has_subscription is not None:
            if has_subscription:
                candidates = [o for o in candidates if o["subscription_id"] is not None]
            else:
                candidates = [o for o in candidates if o["subscription_id"] is None]
        if is_duplicate_charge is not None:
            candidates = [
                o for o in candidates if o["is_duplicate_charge"] == is_duplicate_charge
            ]
        if has_warranty is not None:
            candidates = [o for o in candidates if o["has_warranty"] == has_warranty]

        if not candidates:
            return None
        return deepcopy(rng.choice(candidates))

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def mark_refund_issued(self, order_id: str, amount: float) -> bool:
        if order_id not in self._orders:
            return False
        self._orders[order_id]["refund_issued"] = True
        self._orders[order_id]["refund_amount"] = amount
        return True

    def mark_cancelled(self, order_id: str) -> bool:
        if order_id not in self._orders:
            return False
        self._orders[order_id]["status"] = "cancelled"
        return True

    def update_status(self, order_id: str, new_status: str) -> bool:
        if order_id not in self._orders:
            return False
        self._orders[order_id]["status"] = new_status
        return True

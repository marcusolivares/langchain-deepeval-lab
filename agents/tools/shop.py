"""
shop.py
=======
The ShopEasy demo data and the two tools every support bot shares:
  - get_order_status(order_id)  -> shipping info
  - get_refund_policy(category) -> refund rules

Kept in one place so the agents and the chatbot answer from the same data.
"""

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Fake in-memory data — keeps the demo offline and deterministic.
# ---------------------------------------------------------------------------
ORDERS = {
    "ORD-1042": {"status": "Shipped",    "eta": "2026-05-13", "category": "electronics"},
    "ORD-2099": {"status": "Delivered",  "eta": "2026-05-08", "category": "clothing"},
    "ORD-7777": {"status": "Processing", "eta": "2026-05-15", "category": "food"},
}

REFUND_POLICIES = {
    "electronics": "Electronics can be returned within 15 days, unopened.",
    "clothing":    "Clothing can be returned within 30 days with tags attached.",
    "food":        "Food items are non-returnable for safety reasons.",
    "furniture":   "Furniture can be returned within 30 days if unassembled.",
}


# ---------------------------------------------------------------------------
# Tools — plain Python functions exposed to the LLM via @tool.
# The docstring becomes the tool description the model sees.
# ---------------------------------------------------------------------------
@tool(parse_docstring=True)
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of a customer order by its ID.

    Args:
        order_id: The order ID, e.g. ORD-1042.
    """
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"No order found with ID {order_id}."
    return f"Order {order_id} is {order['status']}. ETA: {order['eta']}."


@tool(parse_docstring=True)
def get_refund_policy(category: str) -> str:
    """Return the refund/return policy for a product category.

    Args:
        category: Product category, e.g. electronics, clothing, food.
    """
    policy = REFUND_POLICIES.get(category.lower())
    if not policy:
        return f"No refund policy on file for '{category}'."
    return policy

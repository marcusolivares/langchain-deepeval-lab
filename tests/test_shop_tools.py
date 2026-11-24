"""The shop tools, called directly: no model involved."""

import pytest

from agents.tools.shop import get_order_status, get_refund_policy


def test_order_status():
    assert (
        get_order_status.invoke({"order_id": "ORD-1042"})
        == "Order ORD-1042 is Shipped. ETA: 2026-05-13."
    )


def test_order_status_ignores_case():
    assert "Shipped" in get_order_status.invoke({"order_id": "ord-1042"})


def test_order_status_unknown_order():
    assert (
        get_order_status.invoke({"order_id": "ORD-0000"})
        == "No order found with ID ORD-0000."
    )


@pytest.mark.xfail(
    reason=(
        "Known agent bug, the same one test_agent_quality.py finds with an LLM "
        "judge: get_order_status does not return the order's category"
    ),
    strict=True,
)
def test_order_status_includes_category():
    assert "clothing" in get_order_status.invoke({"order_id": "ORD-2099"})


@pytest.mark.parametrize("category", ["electronics", "Electronics"])
def test_refund_policy(category):
    assert get_refund_policy.invoke({"category": category}).startswith("Electronics can be returned")


def test_refund_policy_unknown_category():
    assert (
        get_refund_policy.invoke({"category": "toys"})
        == "No refund policy on file for 'toys'."
    )

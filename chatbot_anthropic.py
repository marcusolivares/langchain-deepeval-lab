"""
chatbot_anthropic.py
====================
A multi-turn customer-support chatbot with tool calling, built with Claude.

This is the Anthropic Messages API counterpart to chatbot.py. It keeps the
same tools and chat() return shape so callers can switch between providers.

Usage:
    history = []
    reply, history, tools = chat("Where is order ORD-1042?", history)

How to run standalone:
    python chatbot_anthropic.py
"""

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

client = Anthropic()

# ---------------------------------------------------------------------------
# In-memory data (same as agent_plain.py)
# ---------------------------------------------------------------------------
ORDERS = {
    "ORD-1042": {"status": "Shipped",    "eta": "2026-05-13"},
    "ORD-2099": {"status": "Delivered",  "eta": "2026-05-08"},
    "ORD-7777": {"status": "Processing", "eta": "2026-05-15"},
}

REFUND_POLICIES = {
    "electronics": "Electronics can be returned within 15 days, unopened.",
    "clothing":    "Clothing can be returned within 30 days with tags attached.",
    "food":        "Food items are non-returnable for safety reasons.",
    "furniture":   "Furniture can be returned within 30 days if unassembled.",
}

# ---------------------------------------------------------------------------
# Tool definitions — passed to the Anthropic API
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_order_status",
        "description": "Look up the shipping status of a customer order by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID, e.g. ORD-1042",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "get_refund_policy",
        "description": "Return the refund/return policy for a product category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Product category, e.g. electronics, clothing, food",
                }
            },
            "required": ["category"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------
def _execute_tool(name: str, args: dict) -> str:
    if name == "get_order_status":
        order_id = args.get("order_id", "").upper()
        order = ORDERS.get(order_id)
        if not order:
            return f"No order found with ID {order_id}."
        return f"Order {order_id} is {order['status']}. ETA: {order['eta']}."

    if name == "get_refund_policy":
        category = args.get("category", "").lower()
        policy = REFUND_POLICIES.get(category)
        if not policy:
            return f"No refund policy on file for '{category}'."
        return policy

    return f"Unknown tool: {name}"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a friendly and professional customer-support chatbot \
for ShopEasy, an online retail store.

You have access to two tools:
  - get_order_status : use whenever a customer asks about a specific order
  - get_refund_policy: use whenever a customer asks about returns or refunds

Rules:
- Always use the tools when you have the information needed to call them.
- Be concise and polite.
- Never discuss topics outside of ShopEasy customer support.
- Remember everything the customer tells you in the current conversation."""


# ---------------------------------------------------------------------------
# chat() — one user turn, returns reply + updated history + tools used
# ---------------------------------------------------------------------------
def chat(
    user_message: str,
    history: list[dict],
) -> tuple[str, list[dict], list[dict]]:
    """
    Send one user message, handle any tool calls, return the final reply.

    Returns:
        reply        : final assistant text
        history      : updated message list (include in next call)
        tools_called : list of {"name": ..., "args": ..., "result": ...}
                       for all tools the model called this turn
    """
    history = history + [{"role": "user", "content": user_message}]
    tools_called = []

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=history,
            tools=TOOLS,
        )

        tool_uses = [
            block for block in response.content
            if block.type == "tool_use"
        ]

        if not tool_uses:
            reply = "".join(
                block.text for block in response.content
                if block.type == "text"
            )
            history = history + [{"role": "assistant", "content": response.content}]
            return reply, history, tools_called

        history = history + [{"role": "assistant", "content": response.content}]
        tool_results = []
        for tool_use in tool_uses:
            args = tool_use.input
            result = _execute_tool(tool_use.name, args)
            tools_called.append({
                "name": tool_use.name,
                "args": args,
                "result": result,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })

        history = history + [{"role": "user", "content": tool_results}]


# ---------------------------------------------------------------------------
# Standalone interactive demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("ShopEasy Support Chatbot (Claude with tools) — type 'quit' to exit\n")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        reply, history, tools = chat(user_input, history)
        if tools:
            for tool in tools:
                print(f"  [tool] {tool['name']}({tool['args']}) → {tool['result']}")
        print(f"Bot: {reply}\n")

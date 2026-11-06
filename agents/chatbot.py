"""
chatbot.py
==========
A multi-turn customer-support chatbot with tool calling.

The LLM comes from config.py (Claude by default, GPT-4o with
LLM_PROVIDER=openai). LangChain's bind_tools() translates the tools into
each vendor's own tool-calling format, so one loop works for both.

Tools available (shared with the agents via tools/shop.py):
  - get_order_status(order_id)   → real-time shipping status from in-memory DB
  - get_refund_policy(category)  → refund rules per product category

Unlike plain.py (which delegates the tool-call loop to LangGraph),
this chatbot manages the loop manually so the full message history — including
tool calls and tool results — is preserved across turns. This is what makes
multi-turn evaluation possible with DeepEval.

chat() returns:
  - reply        : the final assistant text shown to the user
  - history      : updated message history to pass into the next turn
  - tools_called : list of {"name": ..., "args": ..., "result": ...} for this turn

Usage:
    history = []
    reply, history, tools = chat("Where is order ORD-1042?", history)
    reply, history, tools = chat("What about the refund policy?", history)

How to run standalone:
    python -m agents.chatbot
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from config import get_chat_model
from agents.tools.shop import get_order_status, get_refund_policy

TOOLS = [get_order_status, get_refund_policy]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

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
# Model — the default from config.py with the shop tools bound to it
# ---------------------------------------------------------------------------
def build_model(llm=None):
    """Bind the tools to a chat model. Uses the default from config.py unless llm is given."""
    return (llm or get_chat_model()).bind_tools(TOOLS)


model = build_model()


# ---------------------------------------------------------------------------
# chat() — one user turn, returns reply + updated history + tools used
# ---------------------------------------------------------------------------
def chat(
    user_message: str,
    history: list[BaseMessage],
) -> tuple[str, list[BaseMessage], list[dict]]:
    """
    Send one user message, handle any tool calls, return the final reply.

    Returns:
        reply        : final assistant text
        history      : updated message list (include in next call)
        tools_called : list of {"name": ..., "args": ..., "result": ...}
                       for all tools the model called this turn
    """
    history = history + [HumanMessage(user_message)]
    tools_called = []

    while True:
        response = model.invoke([SystemMessage(SYSTEM_PROMPT)] + history)
        history = history + [response]

        # No tool call — final answer
        if not response.tool_calls:
            return response.text, history, tools_called

        # Execute every tool the model requested, then loop back
        for tc in response.tool_calls:
            result = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])
            tools_called.append({
                "name":   tc["name"],
                "args":   tc["args"],
                "result": result,
            })
            history = history + [ToolMessage(result, tool_call_id=tc["id"])]


# ---------------------------------------------------------------------------
# Standalone interactive demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("ShopEasy Support Chatbot (with tools) — type 'quit' to exit\n")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        reply, history, tools = chat(user_input, history)
        if tools:
            for t in tools:
                print(f"  [tool] {t['name']}({t['args']}) → {t['result']}")
        print(f"Bot: {reply}\n")

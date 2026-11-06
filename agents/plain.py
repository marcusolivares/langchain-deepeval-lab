"""
plain.py
========
A simple customer-support agent built with LangChain.
No evaluation code anywhere. This is what a developer would write
if they'd never heard of DeepEval.

The agent has two tools (defined in tools/shop.py):
  - get_order_status(order_id)  -> shipping info
  - get_refund_policy(category) -> refund rules

It uses an LLM (Claude by default, see config.py) to decide which tool
to call and to compose the answer.
"""

from langchain.agents import create_agent

from config import get_chat_model
from agents.tools.shop import get_order_status, get_refund_policy


# ---------------------------------------------------------------------------
# The agent — a LangGraph ReAct agent powered by the model from config.py.
# ---------------------------------------------------------------------------
def build_agent(llm=None):
    """Build the agent. Uses the default model from config.py unless llm is given."""
    return create_agent(
        model=llm or get_chat_model(),
        tools=[get_order_status, get_refund_policy],
        system_prompt=(
            "You are a friendly customer-support agent. "
            "Use the available tools to answer order and refund questions. "
            "Keep replies short and helpful."
        ),
    )


agent = build_agent()


# ---------------------------------------------------------------------------
# Single entry point — this is what test code will import.
# ---------------------------------------------------------------------------
def support_agent(user_input: str) -> str:
    """Run the agent on a user message and return the final reply."""
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    return result["messages"][-1].text


if __name__ == "__main__":
    print(support_agent("Where is my order ORD-1099?"))

"""
agent_plain.py
==============
A simple customer-support agent built with LangChain + Claude.
No evaluation code anywhere. This is what a developer would write
if they'd never heard of DeepEval.

The agent has two tools (defined in shop_tools.py):
  - get_order_status(order_id)  -> shipping info
  - get_refund_policy(category) -> refund rules

It uses Claude to decide which tool to call and to compose the answer.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()
from langchain.agents import create_agent

from shop_tools import get_order_status, get_refund_policy


# ---------------------------------------------------------------------------
# The agent — a LangGraph ReAct agent powered by Claude.
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

agent = create_agent(
    model=llm,
    tools=[get_order_status, get_refund_policy],
    system_prompt=(
        "You are a friendly customer-support agent. "
        "Use the available tools to answer order and refund questions. "
        "Keep replies short and helpful."
    ),
)


# ---------------------------------------------------------------------------
# Single entry point — this is what test code will import.
# ---------------------------------------------------------------------------
def support_agent(user_input: str) -> str:
    """Run the agent on a user message and return the final reply."""
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(support_agent("Where is my order ORD-1099?"))

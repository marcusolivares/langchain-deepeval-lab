"""
agent_instrumented.py
=====================
SAME agent as agent_plain.py, with DeepEval observability added.

What changed vs agent_plain.py? Only 4 things — all marked with "NEW":

  (1) Import CallbackHandler from deepeval.integrations.langchain
  (2) Build the callback handler once (no metrics baked in — metrics are
      attached by the test file at run time)
  (3) Pass it via config={"callbacks": [...]} on .invoke()
  (4) (Optional) Add @observe on the outer Python wrapper so any
      non-LangChain code we add later (guardrails, post-processing)
      also shows up as spans

That's it. Four lines, and now QA can run component-level evals
without ever touching this file again.
"""

import asyncio
import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from config import get_chat_model
from shop_tools import get_order_status, get_refund_policy

# --- (1) NEW: import DeepEval tracing primitives --------------------------
from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace


# ---------------------------------------------------------------------------
# Same agent as agent_plain.py.
# ---------------------------------------------------------------------------
# Load one extra tool from the MCP server at startup and merge it in.
_mcp_tools = asyncio.run(
    MultiServerMCPClient({
        "shopease": {
            "command": sys.executable,
            "args": [os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")],
            "transport": "stdio",
        }
    }).get_tools()
)

def build_agent(llm=None):
    """Build the agent. Uses the default model from config.py unless llm is given."""
    return create_agent(
        model=llm or get_chat_model(),
        tools=[get_order_status, get_refund_policy] + _mcp_tools,
        system_prompt=(
            "You are a friendly customer-support agent. "
            "Use the available tools to answer order and refund questions. "
            "Keep replies short and helpful."
        ),
    )


agent = build_agent()


# --- (2) NEW: a single shared callback handler ----------------------------
# This handler captures every LangChain LLM call, tool call, and chain step
# as a DeepEval span automatically. No per-function decoration needed for
# anything LangChain owns.
deepeval_callback = CallbackHandler()


def support_agent(user_input: str) -> str:
    """Run the agent on a user message and return the final reply."""
    # --- (3) NEW: pass the callback handler in the .invoke() config ------
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": [deepeval_callback]},
    )
    reply = result["messages"][-1].text
    update_current_trace(output=reply)
    return reply


if __name__ == "__main__":
    print(support_agent("Where is my order ORD-1042?"))

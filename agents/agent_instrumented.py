"""
agent_instrumented.py
=====================
SAME agent as agent_plain.py, with DeepEval observability added.

What changed vs agent_plain.py? Only 4 things — all marked with "NEW":

  (1) Import CallbackHandler from deepeval.integrations.langchain
  (2) Build the callback handler once (no metrics baked in — metrics are
      attached by the test file at run time)
  (3) Pass it via config={"callbacks": [...]} on .invoke()
  (4) Call update_current_trace(output=reply) so the trace carries the
      clean reply string. The @observe wrapper that opens the trace lives
      on the eval side (traced() in evals/common.py)

That's it. Four lines, and now QA can run component-level evals
without ever touching this file again.
"""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from config import get_chat_model
from agents.shop_tools import get_order_status, get_refund_policy

# --- (1) NEW: import DeepEval tracing primitives --------------------------
from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace


# ---------------------------------------------------------------------------
# Same agent as agent_plain.py.
# ---------------------------------------------------------------------------
@cache
def load_mcp_tools() -> list:
    """Start mcp_server.py over stdio and return its tools as LangChain tools.

    Runs on the first build_agent() call, not at import. The async client runs
    in a worker thread so it never touches the caller's event loop (DeepEval's
    evals_iterator keeps one open).
    """
    client = MultiServerMCPClient({
        "shopease": {
            "command": sys.executable,
            "args": [str(Path(__file__).resolve().parent / "mcp_server.py")],
            "transport": "stdio",
        }
    })
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, client.get_tools()).result()


def build_agent(llm=None):
    """Build the agent. Uses the default model from config.py unless llm is given."""
    return create_agent(
        model=llm or get_chat_model(),
        tools=[get_order_status, get_refund_policy] + load_mcp_tools(),
        system_prompt=(
            "You are a friendly customer-support agent. "
            "Use the available tools to answer order and refund questions. "
            "Keep replies short and helpful."
        ),
    )


@cache
def default_agent():
    """The agent support_agent() runs, built on first use."""
    return build_agent()


# --- (2) NEW: a single shared callback handler ----------------------------
# This handler captures every LangChain LLM call, tool call, and chain step
# as a DeepEval span automatically. No per-function decoration needed for
# anything LangChain owns.
deepeval_callback = CallbackHandler()


def support_agent(user_input: str) -> str:
    """Run the agent on a user message and return the final reply."""
    # --- (3) NEW: pass the callback handler in the .invoke() config ------
    result = default_agent().invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": [deepeval_callback]},
    )
    reply = result["messages"][-1].text
    update_current_trace(output=reply)
    return reply


if __name__ == "__main__":
    print(support_agent("Where is my order ORD-1042?"))

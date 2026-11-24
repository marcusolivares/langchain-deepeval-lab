"""
instrumented.py
===============
The plain.py agent plus one tool from an MCP server, with DeepEval
observability added.

build_agent() calls plain.build_agent(), so the prompt and the shop tools
are plain.py's own. The DeepEval additions are 4 things, all marked with "NEW":

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
from functools import cache, wraps
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents import plain

# --- (1) NEW: import DeepEval tracing primitives --------------------------
from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace


# ---------------------------------------------------------------------------
# The plain.py agent, plus the tools the MCP server exposes.
# ---------------------------------------------------------------------------
def run_in_worker_loop(coro):
    """Run a coroutine on its own event loop in a worker thread, so it never
    touches the caller's event loop (DeepEval's eval runner keeps one open)."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def with_sync_call(mcp_tool: StructuredTool) -> StructuredTool:
    """Give an async-only MCP tool a sync implementation.

    langchain-mcp-adapters builds its tools with a coroutine only, so the
    agent's sync invoke() would raise NotImplementedError when the model calls
    one. wraps() copies the coroutine's signature, which LangGraph reads to
    decide what to inject (the tool runtime).
    """

    @wraps(mcp_tool.coroutine)
    def call(*args, **kwargs):
        return run_in_worker_loop(mcp_tool.coroutine(*args, **kwargs))

    return mcp_tool.model_copy(update={"func": call})


@cache
def load_mcp_tools() -> list:
    """Start mcp_server.py over stdio and return its tools as LangChain tools.

    Runs on the first build_agent() call, not at import.
    """
    client = MultiServerMCPClient({
        "shopease": {
            "command": sys.executable,
            "args": [str(Path(__file__).resolve().parent / "tools" / "mcp_server.py")],
            "transport": "stdio",
        }
    })
    return [with_sync_call(tool) for tool in run_in_worker_loop(client.get_tools())]


def build_agent(llm=None):
    """Build the plain.py agent with the MCP tools added. Uses the default
    model from config.py unless llm is given."""
    return plain.build_agent(llm, extra_tools=load_mcp_tools())


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

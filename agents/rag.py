"""
rag.py
======
A customer-support agent that answers questions by retrieving from a small
in-memory policy knowledge base instead of calling fixed lookup functions.

Architecture:
  - Retriever: search_policies (tools/policy_search.py), semantic search over
    9 policy documents embedded with Voyage AI, returns the top-3 chunks
  - LLM: from config.py, Claude by default (same as plain.py, just with retrieval instead of hard-coded data)
  - Instrumentation: DeepEval CallbackHandler + @observe, same 4-line pattern

What's new vs instrumented.py:
  - search_policies tool replaces get_order_status / get_refund_policy
  - The tool returns the retrieved chunks as a ToolMessage artifact; the entry
    point collects them from the run and puts them on the DeepEval trace as
    retrieval_context, so RAG metrics (Faithfulness, Contextual
    Precision/Recall) can run
"""

from langchain.agents import create_agent

from config import get_chat_model
from agents.tools.policy_search import retrieved_chunks, search_policies

from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
def build_agent(llm=None):
    """Build the RAG agent. Uses the default model from config.py unless llm is given."""
    return create_agent(
        model=llm or get_chat_model(),
        tools=[search_policies],
        system_prompt=(
            "You are a friendly customer-support agent. "
            "Use the search_policies tool to look up refund, shipping, and order policies. "
            "Answer only from the retrieved information. Keep replies concise."
        ),
    )


agent = build_agent()

# DeepEval callback handler — captures LangChain spans automatically.
deepeval_callback = CallbackHandler()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def rag_support_agent(user_input: str) -> str:
    """Run the RAG agent and return the final reply."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": [deepeval_callback]},
    )
    reply = result["messages"][-1].text

    # Set clean reply string and retrieved chunks on the trace.
    update_current_trace(
        output=reply,
        retrieval_context=retrieved_chunks(result["messages"]) or None,
    )

    return reply


if __name__ == "__main__":
    print(rag_support_agent("What is the return policy for electronics?"))

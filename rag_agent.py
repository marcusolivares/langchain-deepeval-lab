"""
rag_agent.py
============
A customer-support agent that answers questions by retrieving from a small
in-memory policy knowledge base instead of calling fixed lookup functions.

Architecture:
  - Knowledge base: 9 policy documents stored as embeddings (Voyage AI)
  - Vector store: langchain_core InMemoryVectorStore (no external DB needed)
  - Tool: search_policies(query) — semantic search, returns top-3 chunks
  - LLM: from config.py, Claude by default (same as agent_plain.py, just with retrieval instead of hard-coded data)
  - Instrumentation: DeepEval CallbackHandler + @observe, same 4-line pattern

What's new vs agent_instrumented.py:
  - search_policies tool replaces get_order_status / get_refund_policy
  - The tool returns the retrieved chunks as a ToolMessage artifact; the entry
    point collects them from the run and puts them on the DeepEval trace as
    retrieval_context, so RAG metrics (Faithfulness, Contextual
    Precision/Recall) can run
"""

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from langchain.agents import create_agent

from config import get_chat_model

from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace


# ---------------------------------------------------------------------------
# Knowledge base — product and shipping policies as plain-text documents.
# ---------------------------------------------------------------------------
POLICY_DOCS = [
    # Refund policies
    "Electronics refund policy: Electronics can be returned within 15 days of delivery, "
    "provided the item is unopened and in its original packaging. Opened electronics are "
    "non-returnable unless faulty. Refunds are processed within 5–7 business days.",

    "Clothing refund policy: Clothing items can be returned within 30 days of delivery "
    "with all original tags still attached. Items must be unworn and unwashed. "
    "Refunds are issued to the original payment method within 3–5 business days.",

    "Food refund policy: Food and perishable items are non-returnable for health and "
    "safety reasons. If your food order arrived damaged or spoiled, contact support "
    "within 24 hours and we will issue a full refund or replacement.",

    "Furniture refund policy: Furniture can be returned within 30 days if unassembled "
    "and in original packaging. Assembled furniture cannot be returned unless defective. "
    "Return shipping costs are the customer's responsibility.",

    "Jewellery refund policy: Jewellery and personalised items are non-returnable unless "
    "received in a damaged or incorrect condition. Please inspect items upon delivery.",

    # Shipping policies
    "Standard shipping policy: Standard shipping takes 5–7 business days. "
    "Orders over $50 qualify for free standard shipping. Tracking information is "
    "emailed once the order ships.",

    "Express shipping policy: Express shipping takes 1–2 business days and costs $15. "
    "Express orders placed before 2 PM are dispatched the same day.",

    "International shipping policy: International orders ship via courier and take "
    "10–21 business days. Customs duties and import taxes are the buyer's responsibility. "
    "We ship to over 50 countries.",

    # Order management
    "Order cancellation policy: Orders can be cancelled within 1 hour of placement "
    "for a full refund. After 1 hour, the order enters fulfilment and cannot be "
    "cancelled. Contact support immediately if you need to cancel.",
]


# ---------------------------------------------------------------------------
# Build the vector store once at import time.
# ---------------------------------------------------------------------------
embeddings = VoyageAIEmbeddings(model="voyage-4-lite")
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_texts(POLICY_DOCS)


# ---------------------------------------------------------------------------
# Retrieval tool — returns top-3 policy chunks for a query.
# The model sees the joined text; the chunk list rides along as the
# ToolMessage artifact so the entry point can put it on the trace.
# ---------------------------------------------------------------------------
@tool(response_format="content_and_artifact")
def search_policies(query: str) -> tuple[str, list[str]]:
    """Search the customer-support knowledge base for policy information."""
    docs = vector_store.similarity_search(query, k=3)
    chunks = [doc.page_content for doc in docs]
    return "\n\n".join(chunks), chunks


def retrieved_chunks(messages: list) -> list[str]:
    """Return the chunks every search_policies call in a run retrieved, in order, deduplicated."""
    chunks = [
        chunk
        for msg in messages
        if isinstance(msg, ToolMessage) and msg.name == search_policies.name
        for chunk in msg.artifact or []
    ]
    return list(dict.fromkeys(chunks))


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

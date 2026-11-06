"""
policy_search.py
================
The RAG agent's retriever: a small policy knowledge base, the in-memory vector
store built from it, and the search_policies tool that queries it.

  - Knowledge base: 9 policy documents, one chunk each
  - Embeddings: Voyage AI (voyage-4-lite)
  - Vector store: langchain_core InMemoryVectorStore (no external DB needed)
  - Tool: search_policies(query), semantic search returning the top-3 chunks

The tool returns the chunks as a ToolMessage artifact as well as text.
retrieved_chunks() collects them from a run so the agent can put them on the
DeepEval trace as retrieval_context.
"""

from functools import cache

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_voyageai import VoyageAIEmbeddings


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
# Build the vector store once, on the first search, so importing this module
# makes no embedding call (the Voyage free tier allows 3 requests/minute).
# ---------------------------------------------------------------------------
@cache
def get_vector_store() -> InMemoryVectorStore:
    """Embed POLICY_DOCS into an in-memory vector store and return it."""
    store = InMemoryVectorStore(embedding=VoyageAIEmbeddings(model="voyage-4-lite"))
    store.add_texts(POLICY_DOCS)
    return store


# ---------------------------------------------------------------------------
# Retrieval tool — returns top-3 policy chunks for a query.
# The model sees the joined text; the chunk list rides along as the
# ToolMessage artifact so the entry point can put it on the trace.
# ---------------------------------------------------------------------------
@tool(response_format="content_and_artifact")
def search_policies(query: str) -> tuple[str, list[str]]:
    """Search the customer-support knowledge base for policy information."""
    docs = get_vector_store().similarity_search(query, k=3)
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

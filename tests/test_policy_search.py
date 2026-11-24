"""The RAG retriever, with fake embeddings in place of Voyage AI."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore

from agents.tools import policy_search
from agents.tools.policy_search import POLICY_DOCS, retrieved_chunks, search_policies


@pytest.fixture
def fake_store(monkeypatch):
    """Swap the Voyage-backed vector store for one built with fake embeddings.

    Fake embeddings carry no meaning, so the tests check the shape of what
    search_policies returns, not which chunks rank first.
    """
    store = InMemoryVectorStore(embedding=DeterministicFakeEmbedding(size=16))
    store.add_texts(POLICY_DOCS)
    monkeypatch.setattr(policy_search, "get_vector_store", lambda: store)


def test_search_returns_top_3_chunks_as_text_and_artifact(fake_store):
    # Invoked with a tool call, as the agent does, so the result is the
    # ToolMessage the agent sees, artifact included.
    message = search_policies.invoke(
        {"type": "tool_call", "name": "search_policies", "args": {"query": "refunds"}, "id": "call_1"}
    )
    assert len(message.artifact) == 3
    assert set(message.artifact) <= set(POLICY_DOCS)
    assert message.content == "\n\n".join(message.artifact)


def test_retrieved_chunks_keeps_order_drops_duplicates_and_other_tools():
    messages = [
        ToolMessage("a\n\nb", artifact=["a", "b"], name="search_policies", tool_call_id="1"),
        ToolMessage("Order ORD-1042 is Shipped.", name="get_order_status", tool_call_id="2"),
        ToolMessage("b\n\nc", artifact=["b", "c"], name="search_policies", tool_call_id="3"),
    ]
    assert retrieved_chunks(messages) == ["a", "b", "c"]


def test_retrieved_chunks_without_a_search():
    assert retrieved_chunks([]) == []

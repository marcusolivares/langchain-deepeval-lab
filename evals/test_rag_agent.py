"""RAG agent: contextual precision and recall score the retriever; answer relevancy
and faithfulness score the generated reply.
"""

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)

from agents.rag import rag_support_agent
from evals.common import get_judge, traced

pytestmark = [pytest.mark.rag, pytest.mark.smoke]

run_rag_support_agent = traced(rag_support_agent, name="rag_support_agent")

judge = get_judge()

precision = ContextualPrecisionMetric(threshold=0.7, model=judge)
recall = ContextualRecallMetric(threshold=0.7, model=judge)
relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge)
faithfulness = FaithfulnessMetric(threshold=0.7, model=judge)

GOLDENS = [
    Golden(
        input="What is the return policy for electronics?",
        expected_output=(
            "Electronics can be returned within 15 days of delivery if unopened "
            "and in original packaging. Refunds take 5–7 business days."
        ),
    ),
    Golden(
        input="How long does express shipping take and what does it cost?",
        expected_output=(
            "Express shipping takes 1–2 business days and costs $15. "
            "Orders placed before 2 PM are dispatched the same day."
        ),
    ),
]


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: g.input)
def test_retrieval_and_answer(golden: Golden):
    run_rag_support_agent(golden)
    assert_test(golden=golden, metrics=[precision, recall, relevancy, faithfulness])

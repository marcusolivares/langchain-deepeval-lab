"""RAG agent: contextual precision and recall score the retriever; answer relevancy
and faithfulness score the generated reply.
"""

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)

from evals.common import ASYNC_CONFIG, get_judge, traced
from agents.rag_agent import rag_support_agent as _rag_support_agent

rag_support_agent = traced(_rag_support_agent, name="rag_support_agent")


dataset = EvaluationDataset(
    goldens=[
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
)

judge = get_judge()

precision = ContextualPrecisionMetric(threshold=0.7, model=judge)
recall = ContextualRecallMetric(threshold=0.7, model=judge)
relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge)
faithfulness = FaithfulnessMetric(threshold=0.7, model=judge)

for golden in dataset.evals_iterator(
    metrics=[precision, recall, relevancy, faithfulness],
    async_config=ASYNC_CONFIG,
):
    rag_support_agent(golden.input)

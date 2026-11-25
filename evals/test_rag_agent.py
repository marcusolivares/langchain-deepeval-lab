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

from evals.common import get_judge, load_goldens, run_rag_support_agent

pytestmark = [pytest.mark.rag, pytest.mark.smoke]

judge = get_judge()

precision = ContextualPrecisionMetric(threshold=0.7, model=judge)
recall = ContextualRecallMetric(threshold=0.7, model=judge)
relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge)
faithfulness = FaithfulnessMetric(threshold=0.7, model=judge)


@pytest.mark.parametrize("golden", load_goldens("rag_agent"))
def test_retrieval_and_answer(golden: Golden):
    run_rag_support_agent(golden)
    assert_test(golden=golden, metrics=[precision, recall, relevancy, faithfulness])

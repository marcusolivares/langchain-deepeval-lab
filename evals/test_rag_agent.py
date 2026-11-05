#Contextual Precision -< Groundtruth (Expected output)
#50%
# 5 docs  - LLM(

#Recall - 10 docs-> Top 2 documents = 30% noise
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, AnswerRelevancyMetric, \
    FaithfulnessMetric

from evals.common import ASYNC_CONFIG, get_judge, traced
from rag_agent import rag_support_agent as _rag_support_agent

rag_support_agent = traced(_rag_support_agent, name="rag_support_agent")


dataset = EvaluationDataset(goldens = [

    Golden(
        input = "What is the return policy for electronics?",
        expected_output = (
            "Electronics can be returned within 15 days of delivery if unopened "
            "and in original packaging. Refunds take 5–7 business days."
                 ),),

    Golden(
        input="How long does express shipping take and what does it cost?",
        expected_output=(
            "Express shipping takes 1–2 business days and costs $15. "
            "Orders placed before 2 PM are dispatched the same day."
        ),
    ),])

judge = get_judge()

precisionMetric = ContextualPrecisionMetric(
    threshold=0.7,
    model = judge

)
recallMetric = ContextualRecallMetric(
    threshold=0.7,
    model=judge,
    include_reason=True
)

relevancyMetric = AnswerRelevancyMetric(
    threshold=0.7,
    model=judge,
    include_reason=True
)


faithfulMetric = FaithfulnessMetric(
    threshold=0.7,
    model=judge,
    include_reason=True
)





for golden in dataset.evals_iterator(metrics=[precisionMetric,recallMetric,relevancyMetric,faithfulMetric], async_config=ASYNC_CONFIG):
    rag_support_agent(golden.input)




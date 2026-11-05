import os
import sys

from deepeval.contextvars import get_current_golden
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, AnswerRelevancyMetric, \
    FaithfulnessMetric
from deepeval.models import AnthropicModel
from deepeval.tracing import update_current_trace, observe
sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
from rag_agent import rag_support_agent as _rag_support_agent

@observe(name="rag_support_agent")
def rag_support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return _rag_support_agent(user_input)


dataset = EvaluationDataset(goldens = [

    Golden(
        input= "What is the return policy for electronics?",
        expected_output= (
            "Electronics can be returned within 15 days of delivery if unopened "
            "and in original packaging. Refunds take 5-7 business days."
    ),),

    Golden(
        input= "How long does express shipping take and what does it cost?",
        expected_output= (
            "Express shipping take 1-2 business days and costs $15. "
            "Orders placed before 2 PM are dispatched the same day."
        )
    )])

precisionMetric = ContextualPrecisionMetric(threshold=0.7,model=AnthropicModel(model="claude-sonnet-4-6"),include_reason=True)
recallMetric = ContextualRecallMetric(threshold=0.7,model=AnthropicModel(model="claude-sonnet-4-6"),include_reason=True)
relevancyMetric = AnswerRelevancyMetric(threshold=0.7,model=AnthropicModel(model="claude-sonnet-4-6"),include_reason=True)
faithfulnessMetric = FaithfulnessMetric(threshold=0.7,model=AnthropicModel(model="claude-sonnet-4-6"),include_reason=True)


for golden in dataset.evals_iterator(metrics=[precisionMetric,recallMetric,relevancyMetric,faithfulnessMetric]):
    rag_support_agent(golden.input)
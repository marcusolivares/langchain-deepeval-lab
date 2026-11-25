"""Agent: answer relevancy, system-prompt alignment and step efficiency on traced runs."""

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import (
    AnswerRelevancyMetric,
    PromptAlignmentMetric,
    StepEfficiencyMetric,
)

from evals.common import get_judge, load_goldens, run_support_agent

pytestmark = pytest.mark.agent

judge = get_judge()

answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge)
step_efficiency = StepEfficiencyMetric(threshold=0.5, model=judge)

prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=[
        "You are a friendly customer-support agent. Keep replies short and helpful."
    ],
    threshold=0.7,
    model=judge,
)


@pytest.mark.parametrize("golden", load_goldens("agent_quality"))
def test_reply_quality(golden: Golden):
    run_support_agent(golden)
    assert_test(
        golden=golden, metrics=[prompt_alignment, step_efficiency, answer_relevancy]
    )

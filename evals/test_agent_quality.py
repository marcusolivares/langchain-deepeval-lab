"""Agent: answer relevancy, system-prompt alignment and step efficiency on traced runs."""

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import (
    AnswerRelevancyMetric,
    PromptAlignmentMetric,
    StepEfficiencyMetric,
)

from evals.common import get_judge, run_support_agent

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

GOLDENS = [
    Golden(input="Where is my order ORD-1042?"),
    Golden(input="What's the refund policy for electronics?"),
    pytest.param(
        Golden(input="I want to return order ORD-2099, what should I do?"),
        marks=pytest.mark.xfail(
            reason=(
                "Known agent bug: get_order_status does not return the order's "
                "category, so the agent cannot look up the refund policy"
            ),
            strict=False,
        ),
    ),
]


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: g.input)
def test_reply_quality(golden: Golden):
    run_support_agent(golden)
    assert_test(
        golden=golden, metrics=[prompt_alignment, step_efficiency, answer_relevancy]
    )

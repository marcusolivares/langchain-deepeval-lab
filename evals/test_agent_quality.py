"""Agent: answer relevancy, system-prompt alignment and step efficiency on traced runs."""

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import (
    AnswerRelevancyMetric,
    PromptAlignmentMetric,
    StepEfficiencyMetric,
)

from agents.plain import PERSONA, REPLY_STYLE
from evals.common import get_judge, load_goldens, run_support_agent

pytestmark = pytest.mark.agent

judge = get_judge()

answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge)
step_efficiency = StepEfficiencyMetric(threshold=0.5, model=judge)

# The system prompt's instructions that a reply can show. The judge sees the
# input and the reply, not the tool calls, so it scores plain.TOOL_USE as
# ignored (0.0) even when the agent used the tools. Tool use is checked on the
# trace instead: StepEfficiency here, ToolCorrectness in test_agent_tracing.py.
prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=[f"{PERSONA} {REPLY_STYLE}"],
    threshold=0.7,
    model=judge,
)


@pytest.mark.parametrize("golden", load_goldens("agent_quality"))
def test_reply_quality(golden: Golden):
    run_support_agent(golden)
    assert_test(
        golden=golden, metrics=[prompt_alignment, step_efficiency, answer_relevancy]
    )

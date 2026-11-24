"""Agent: task completion and tool correctness, scored on the full trace."""

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall

from agents.instrumented import support_agent
from evals.common import get_judge, traced

pytestmark = [pytest.mark.agent, pytest.mark.smoke]

run_support_agent = traced(support_agent, name="support_agent")

judge = get_judge()

task_completion = TaskCompletionMetric(threshold=0.7, model=judge)
tool_correctness = ToolCorrectnessMetric(model=judge)

GOLDENS = [
    Golden(
        input="Where is my order ORD-1042?",
        expected_tools=[ToolCall(name="get_order_status")],
    ),
    Golden(
        input="What is refund policy for electronics?",
        expected_tools=[ToolCall(name="get_refund_policy")],
    ),
]


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: g.input)
def test_task_completion_and_tool_use(golden: Golden):
    run_support_agent(golden)
    assert_test(golden=golden, metrics=[task_completion, tool_correctness])

"""Agent: task completion and tool correctness, scored on the full trace."""

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric

from evals.common import get_judge, load_goldens, run_support_agent

pytestmark = [pytest.mark.agent, pytest.mark.smoke]

judge = get_judge()

task_completion = TaskCompletionMetric(threshold=0.7, model=judge)
tool_correctness = ToolCorrectnessMetric(model=judge)


@pytest.mark.parametrize("golden", load_goldens("agent_tracing"))
def test_task_completion_and_tool_use(golden: Golden):
    run_support_agent(golden)
    assert_test(golden=golden, metrics=[task_completion, tool_correctness])

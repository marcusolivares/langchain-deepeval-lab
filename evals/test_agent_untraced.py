"""Agent: the same TaskCompletion check as test_agent_tracing, on the bare reply with
no trace. The judge cannot see the tool call, so this scores low by design.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase

from agents.instrumented import support_agent
from evals.common import RUN_ASYNC, get_judge

pytestmark = pytest.mark.agent


@pytest.mark.xfail(
    reason="By design: without the trace the judge cannot see the tool call",
    strict=False,
)
def test_task_completion_without_trace():
    question = "Where is my order ORD-1042?"
    test_case = LLMTestCase(input=question, actual_output=support_agent(question))
    assert_test(
        test_case,
        [TaskCompletionMetric(threshold=0.7, model=get_judge())],
        run_async=RUN_ASYNC,
    )

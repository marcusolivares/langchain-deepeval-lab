import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace

from agent_instrumented import support_agent as _support_agent

@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace(expected_tools=golden.expected_tools)
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)

    return _support_agent(user_input)

claude_judge = AnthropicModel(model="claude-sonnet-4-6")
task_completion = TaskCompletionMetric(threshold=0.7,model=claude_judge)

tool_correctness = ToolCorrectnessMetric(model=claude_judge)

dataSet = EvaluationDataset(goldens=[
    Golden(input = "Where is my order ORD-1042?",
           expected_tools=[ToolCall(name="get_order_status")]),

    Golden(input = "What is refund policy for electronics?",
           expected_tools=[ToolCall(name="get_refund_policy")])
])

for golden in dataSet.evals_iterator(
    metrics=[task_completion, tool_correctness],
    async_config=AsyncConfig(run_async=False),
):
    support_agent(golden.input)
"""Agent: task completion and tool correctness, scored on the full trace."""

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall

from agents.agent_instrumented import support_agent as _support_agent
from evals.common import ASYNC_CONFIG, get_judge, traced

support_agent = traced(_support_agent, name="support_agent")

judge = get_judge()

task_completion = TaskCompletionMetric(threshold=0.7, model=judge)
tool_correctness = ToolCorrectnessMetric(model=judge)

dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="Where is my order ORD-1042?",
            expected_tools=[ToolCall(name="get_order_status")],
        ),
        Golden(
            input="What is refund policy for electronics?",
            expected_tools=[ToolCall(name="get_refund_policy")],
        ),
    ]
)

for golden in dataset.evals_iterator(
    metrics=[task_completion, tool_correctness], async_config=ASYNC_CONFIG
):
    support_agent(golden.input)

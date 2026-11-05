import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric
from deepeval.models import AnthropicModel
from deepeval.tracing import observe
from agent_instrumented import support_agent as _support_agent

@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    return _support_agent(user_input)

claude_judge = AnthropicModel(model="claude-sonnet-4-6")

step_efficiency = StepEfficiencyMetric(threshold=0.5,model=claude_judge)

prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=[
        "You are a friendly customer-support agent. "
        "Keep replies short and helpful."
    ], threshold=0.7, model=claude_judge
)

dataSet = EvaluationDataset(goldens=[
    Golden(input = "Where is my order ORD-1042?"),
    Golden(input = "What is refund policy for electronics?"),
    Golden(input = "I want to return ORD-2099, what should I do?"),
])

for golden in dataSet.evals_iterator(
    metrics=[prompt_alignment,step_efficiency],
    async_config=AsyncConfig(run_async=False),
):
    support_agent(golden.input)
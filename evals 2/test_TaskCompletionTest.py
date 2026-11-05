import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase
from agent_instrumented import support_agent

actual_output = support_agent("Where is my order ORD-1042?")

test_case = LLMTestCase(
    input = "Where is my order ORD-1042?",
    actual_output = actual_output
)

claude_judge = AnthropicModel(model="claude-sonnet-4-6")

evaluate(test_cases = [test_case],
         metrics=[TaskCompletionMetric(threshold=0.7, model=claude_judge)])

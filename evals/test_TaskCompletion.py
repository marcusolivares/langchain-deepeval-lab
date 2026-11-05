from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase

from agent_instrumented import support_agent
from evals.common import ASYNC_CONFIG, get_judge

actual_output = support_agent("Where is my order ORD-1042?")

test_case = LLMTestCase(
    input = "Where is my order ORD-1042?",
    actual_output = actual_output
)

evaluate(test_cases = [test_case],
         metrics= [TaskCompletionMetric(threshold=0.7,model = get_judge())],
         async_config=ASYNC_CONFIG)
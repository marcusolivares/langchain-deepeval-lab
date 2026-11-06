"""Agent: a custom GEval correctness metric that compares the reply to an expected answer."""

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

from agents.agent_instrumented import support_agent as _support_agent
from evals.common import ASYNC_CONFIG, get_judge, traced

support_agent = traced(_support_agent, name="support_agent")


correctness = GEval(
    name="Correctness",
    criteria=(
        "Determine whether the actual output conveys the same factual information "
        "as the expected output. Minor wording differences are acceptable; "
        "missing or wrong facts are not."
    ),
    model=get_judge(),
    threshold=0.79,
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
)


dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="Where is my order ORD-1042?",
            expected_output="order ORD-1042 is shipped and will arrive by May 13th",
        )
    ]
)

for golden in dataset.evals_iterator(metrics=[correctness], async_config=ASYNC_CONFIG):
    support_agent(golden.input)

"""Agent: a custom GEval correctness metric that compares the reply to an expected answer."""

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

from evals.common import get_judge, run_support_agent

pytestmark = pytest.mark.agent

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

GOLDENS = [
    Golden(
        input="Where is my order ORD-1042?",
        expected_output="order ORD-1042 is shipped and will arrive by May 13th",
    )
]


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: g.input)
def test_correctness(golden: Golden):
    run_support_agent(golden)
    assert_test(golden=golden, metrics=[correctness])

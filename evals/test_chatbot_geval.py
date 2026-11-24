"""
Chatbot: a custom ConversationalGEval correctness metric over a live conversation.
The judge also sees the tools the chatbot called on each turn.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import ConversationalTestCase, MultiTurnParams

from evals.common import RUN_ASYNC, get_judge

pytestmark = pytest.mark.chatbot

correctness = ConversationalGEval(
    name="Correctness",
    criteria=(
        "Did the chatbot fully resolve the customer's issue? "
        "It should use tools when needed and provide accurate answers."
    ),
    model=get_judge(),
    threshold=0.80,
    evaluation_params=[
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
        MultiTurnParams.TOOLS_CALLED,
    ],
)


def test_conversation_correctness(support_conversation):
    # support_conversation (evals/conftest.py) is the live run, shared with test_chatbot.
    assert_test(
        ConversationalTestCase(turns=support_conversation),
        [correctness],
        run_async=RUN_ASYNC,
    )

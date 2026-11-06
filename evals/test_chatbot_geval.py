"""
Chatbot: a custom ConversationalGEval correctness metric over a live conversation.
The judge also sees the tools the chatbot called on each turn.
"""

from deepeval.evaluate import evaluate
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import ConversationalTestCase, MultiTurnParams

from agents.chatbot import chat
from evals.common import ASYNC_CONFIG, SUPPORT_CONVERSATION, get_judge, run_conversation

turns = run_conversation(chat, SUPPORT_CONVERSATION)

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


test_case = ConversationalTestCase(turns=turns)

evaluate(test_cases=[test_case], metrics=[correctness], async_config=ASYNC_CONFIG)

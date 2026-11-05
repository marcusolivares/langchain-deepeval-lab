"""Chatbot: turn relevancy, knowledge retention and completeness over a live
four-turn conversation.
"""

from deepeval.evaluate import evaluate
from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    TurnRelevancyMetric,
)
from deepeval.test_case import ConversationalTestCase, Turn

from chatbot import chat
from evals.common import ASYNC_CONFIG, get_judge

turns = []
history = []
for user_msg in [
    "Hi! I placed an order last week, the order ID is ORD-1042.",
    "Is it going to arrive on time?",
    "What was the ETA you just mentioned?",  # tests memory retention
    "Can I upgrade to express shipping?",
]:
    reply, history, _ = chat(user_msg, history)
    turns.append(Turn(role="user", content=user_msg))
    turns.append(Turn(role="assistant", content=reply))

judge = get_judge()

turn_relevancy = TurnRelevancyMetric(threshold=0.5, model=judge)
retention = KnowledgeRetentionMetric(threshold=0.5, model=judge)
completeness = ConversationCompletenessMetric(threshold=0.5, model=judge)

test_case = ConversationalTestCase(turns=turns)

evaluate(
    test_cases=[test_case],
    metrics=[turn_relevancy, retention, completeness],
    async_config=ASYNC_CONFIG,
)

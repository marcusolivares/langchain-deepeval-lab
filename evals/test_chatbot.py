"""Chatbot: turn relevancy, knowledge retention and completeness over a live
four-turn conversation.
"""

from deepeval.evaluate import evaluate
from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    TurnRelevancyMetric,
)
from deepeval.test_case import ConversationalTestCase

from chatbot import chat
from evals.common import ASYNC_CONFIG, SUPPORT_CONVERSATION, get_judge, run_conversation

turns = run_conversation(chat, SUPPORT_CONVERSATION)

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

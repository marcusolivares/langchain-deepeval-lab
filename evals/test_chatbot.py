"""Chatbot: turn relevancy, knowledge retention and completeness over a live
four-turn conversation.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    TurnRelevancyMetric,
)
from deepeval.test_case import ConversationalTestCase

from evals.common import RUN_ASYNC, get_judge

pytestmark = [pytest.mark.chatbot, pytest.mark.smoke]

judge = get_judge()

turn_relevancy = TurnRelevancyMetric(threshold=0.5, model=judge)
retention = KnowledgeRetentionMetric(threshold=0.5, model=judge)
completeness = ConversationCompletenessMetric(threshold=0.5, model=judge)


def test_conversation_quality(support_conversation):
    # support_conversation (evals/conftest.py) is the live run, shared with test_chatbot_geval.
    assert_test(
        ConversationalTestCase(turns=support_conversation),
        [turn_relevancy, retention, completeness],
        run_async=RUN_ASYNC,
    )

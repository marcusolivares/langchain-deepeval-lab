import os
import sys

from deepeval.evaluate import evaluate

from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric, ConversationCompletenessMetric
from deepeval.models import AnthropicModel

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

from deepeval.test_case import ConversationalTestCase, Turn
from chatbot_anthropic import chat

turns = []
history = []
for user_msg in [
        "Hi! I placed an order last week, the order ID is ORD-1042.",
        "Is it going to arrive on time?",
        "What was the ETA you just mentioned?",   # tests memory retention
        "Can I upgrade to express shipping?",
]:
    reply, history, _ = chat(user_msg,history)
    turns.append(Turn(role="user",content=user_msg))
    turns.append(Turn(role="assistant", content=reply))

turnRelevancyMetric = TurnRelevancyMetric(
    threshold=0.5,
    model=AnthropicModel(model="claude-sonnet-4-6"),
)
retentionMetric = KnowledgeRetentionMetric(threshold=0.5, model=AnthropicModel(model="claude-sonnet-4-6"))
completenessMetric = ConversationCompletenessMetric(threshold=0.5, model=AnthropicModel(model="claude-sonnet-4-6"))

test_case = ConversationalTestCase(
    turns = turns
)

evaluate(test_cases=[test_case], metrics=[turnRelevancyMetric,retentionMetric,completenessMetric])
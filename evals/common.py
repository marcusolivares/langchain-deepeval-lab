"""
Shared helpers for the eval scripts: the judge model, the tracing wrapper and
the conversation driver for the chatbot evals.

Run the eval scripts from the project root as modules, so config.py and
the agents can be imported:

    python -m evals.test_rag_agent
"""

from deepeval.contextvars import get_current_golden
from deepeval.evaluate import AsyncConfig
from deepeval.models import AnthropicModel, OpenAIModel
from deepeval.test_case import ToolCall, Turn
from deepeval.tracing import observe, update_current_trace

from config import JUDGE_PROVIDER, MODELS


def get_judge(provider: str = JUDGE_PROVIDER):
    """Return the DeepEval judge model for the given provider."""
    model = MODELS[provider]
    if provider == "anthropic":
        # Judging is short classification and extraction work, so low effort
        # keeps Claude's thinking (and its tokens) small.
        return AnthropicModel(model=model, generation_kwargs={"output_config": {"effort": "low"}})
    return OpenAIModel(model=model)


# The Claude-judged evals ran with run_async=False; the GPT-4o-judged ones
# ran concurrently (DeepEval's default). Keep both behaviours.
ASYNC_CONFIG = AsyncConfig(run_async=JUDGE_PROVIDER != "anthropic")


def traced(agent_fn, name: str):
    """
    Wrap an agent function in a DeepEval trace.

    DeepEval metrics read expected_output and expected_tools from the trace,
    not from the golden, so copy them over from the golden being evaluated.
    """

    @observe(name=name)
    def wrapper(user_input: str) -> str:
        golden = get_current_golden()
        if golden:
            if golden.expected_output:
                update_current_trace(expected_output=golden.expected_output)
            if golden.expected_tools:
                update_current_trace(expected_tools=golden.expected_tools)
        return agent_fn(user_input)

    return wrapper


# The scripted customer side of the conversation the chatbot evals replay.
SUPPORT_CONVERSATION = [
    "Hi! I placed an order last week, the order ID is ORD-1042.",
    "Is it going to arrive on time?",
    "What was the ETA you just mentioned?",  # tests memory retention
    "Can I upgrade to express shipping?",
]


def run_conversation(chat_fn, user_messages: list[str]) -> list[Turn]:
    """
    Drive a chatbot through user_messages and return the conversation as Turns.

    chat_fn has the signature of chatbot.chat. The tools it called on each
    turn are attached to the assistant Turn, so conversational metrics can
    judge tool use (MultiTurnParams.TOOLS_CALLED).
    """
    turns = []
    history = []
    for user_msg in user_messages:
        reply, history, tools_called = chat_fn(user_msg, history)
        turns.append(Turn(role="user", content=user_msg))
        turns.append(
            Turn(
                role="assistant",
                content=reply,
                tools_called=[
                    ToolCall(
                        name=t["name"], input_parameters=t["args"], output=t["result"]
                    )
                    for t in tools_called
                ]
                or None,
            )
        )
    return turns

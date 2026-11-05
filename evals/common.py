"""
Shared helpers for the eval scripts: the judge model and the tracing wrapper.

Run the eval scripts from the project root as modules, so config.py and
the agents can be imported:

    python -m evals.test_rag_agent
"""

from deepeval.contextvars import get_current_golden
from deepeval.evaluate import AsyncConfig
from deepeval.models import AnthropicModel, OpenAIModel
from deepeval.tracing import observe, update_current_trace

from config import JUDGE_PROVIDER, MODELS


def get_judge(provider: str = JUDGE_PROVIDER):
    """Return the DeepEval judge model for the given provider."""
    model = MODELS[provider]
    if provider == "anthropic":
        return AnthropicModel(model=model)
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

"""
config.py
=========
The one place that decides which LLM vendor the project talks to.

The agents, the chatbot and the eval judges ask this module for a model
instead of hard-coding one, so switching vendor is a .env change, not a
code change:

    LLM_PROVIDER=anthropic   # default: Claude
    LLM_PROVIDER=openai      # GPT-4o
    JUDGE_PROVIDER=openai    # optional: judge evals with a different vendor
                             # (defaults to LLM_PROVIDER)

To use a specific model in code instead, pass it in explicitly,
e.g. build_agent(get_chat_model("openai")).
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai":    "gpt-4o",
}

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", PROVIDER)


def get_chat_model(provider: str = PROVIDER, effort: str = "low"):
    """Return a LangChain chat model for the given provider.

    Claude Sonnet 5.5 always thinks before answering and rejects a non-default
    temperature, so its token spend is tuned with `effort` ("low" to "max")
    instead. The bots only do short lookups, where "low" mostly skips thinking;
    raise it per call if quality drops. GPT-4o has no effort setting and keeps
    temperature=0.
    """
    if provider == "anthropic":
        return init_chat_model(MODELS[provider], model_provider=provider, effort=effort)
    return init_chat_model(MODELS[provider], model_provider=provider, temperature=0)

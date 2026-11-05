"""
config.py
=========
The one place that decides which LLM vendor the project talks to.

The agents and the chatbot ask this module for a model instead of
hard-coding one, so switching vendor is a .env change, not a code change:

    LLM_PROVIDER=anthropic   # default: Claude
    LLM_PROVIDER=openai      # GPT-4o

To use a specific model in code instead, pass it in explicitly,
e.g. build_agent(get_chat_model("openai")).
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
}

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")


def get_chat_model(provider: str = PROVIDER):
    """Return a LangChain chat model for the given provider."""
    return init_chat_model(MODELS[provider], model_provider=provider, temperature=0)

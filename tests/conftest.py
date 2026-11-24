"""
Pytest setup for the offline tests.

These tests swap every chat model and embedding for a fake, so they are free,
fast and need no API keys:

    pytest                      # runs tests/ only (testpaths in pytest.ini)

The paid, LLM-judged evals live in evals/ and run with `deepeval test run evals/`.
"""

import os

# Fake API keys, set before any project module calls load_dotenv(). load_dotenv()
# never overrides a variable that is already set, so a test that reaches a real
# API by mistake fails with an authentication error instead of spending money.
for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY"):
    os.environ[key] = "fake-key-for-offline-tests"

# Importing DeepEval would otherwise send usage telemetry.
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"

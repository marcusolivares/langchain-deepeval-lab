"""
Pytest setup shared by every eval.

Every test here makes paid API calls: the bots run live and an LLM judge
scores them.
"""

import pytest
from deepeval.utils import get_is_running_deepeval

from agents.chatbot import chat
from evals.common import SUPPORT_CONVERSATION, run_conversation


def pytest_runtestloop(session: pytest.Session):
    """Stop a plain `pytest` run before any test, and so any paid call, runs.

    The traced tests call assert_test(golden=...), which only works under
    `deepeval test run`: its pytest plugin keeps each test's trace open until
    the assertion reads it. Under plain pytest those tests would pay for an
    agent run and then error, so require the DeepEval runner for every eval.
    Collecting (`pytest --collect-only`) makes no API calls and is allowed.
    """
    if not session.config.option.collectonly and not get_is_running_deepeval():
        pytest.exit(
            "Run the evals with `deepeval test run evals/`, not plain pytest.",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )


@pytest.fixture(scope="session")
def support_conversation():
    """The chatbot's live run of SUPPORT_CONVERSATION, as Turns.

    Session-scoped so the chatbot evals share one conversation and pay for it once.
    """
    return run_conversation(chat, SUPPORT_CONVERSATION)

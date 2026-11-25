"""The helpers in evals/common.py that need no judge."""

import json

import pytest
from deepeval.dataset import Golden

from evals import common
from evals.common import DATASETS, load_goldens, run_conversation


@pytest.mark.parametrize("path", sorted(DATASETS.glob("*.json")), ids=lambda p: p.name)
def test_every_saved_dataset_loads(path):
    params = load_goldens(path.stem)

    assert params
    for param in params:
        [golden] = param.values
        assert isinstance(golden, Golden)
        assert golden.input


def test_load_goldens_marks_known_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATASETS", tmp_path)
    (tmp_path / "cases.json").write_text(
        json.dumps([
            {"input": "works", "expected_tools": [{"name": "get_order_status"}]},
            {"input": "broken", "additional_metadata": {"xfail": "known bug"}},
        ]),
        encoding="utf-8",
    )

    works, broken = load_goldens("cases")

    assert (works.id, list(works.marks)) == ("works", [])
    assert works.values[0].expected_tools[0].name == "get_order_status"
    [mark] = broken.marks
    assert (broken.id, mark.name, mark.kwargs) == (
        "broken",
        "xfail",
        {"reason": "known bug", "strict": False},
    )


def test_load_goldens_skips_a_missing_dataset_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATASETS", tmp_path)

    [param] = load_goldens("missing", skip_if_missing="not generated")

    assert param.values == (None,)
    [mark] = param.marks
    assert (mark.name, mark.kwargs) == ("skip", {"reason": "not generated"})
    with pytest.raises(FileNotFoundError):
        load_goldens("missing")


def test_run_conversation_builds_turns_and_threads_history():
    histories_seen = []

    def chat(message, history):
        # Stands in for agents.chatbot.chat: calls a tool only for order questions.
        histories_seen.append(history)
        tools = (
            [{"name": "get_order_status", "args": {"order_id": "ORD-1042"}, "result": "Shipped"}]
            if "ORD-" in message
            else []
        )
        return f"reply to {message}", history + [message], tools

    turns = run_conversation(chat, ["Where is ORD-1042?", "Thanks"])

    assert histories_seen == [[], ["Where is ORD-1042?"]]
    assert [(t.role, t.content) for t in turns] == [
        ("user", "Where is ORD-1042?"),
        ("assistant", "reply to Where is ORD-1042?"),
        ("user", "Thanks"),
        ("assistant", "reply to Thanks"),
    ]
    [called] = turns[1].tools_called
    assert (called.name, called.input_parameters, called.output) == (
        "get_order_status",
        {"order_id": "ORD-1042"},
        "Shipped",
    )
    assert turns[3].tools_called is None

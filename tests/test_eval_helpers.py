"""The helpers in evals/common.py that need no judge."""

from evals.common import run_conversation


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

"""The agents and the chatbot, driven by a fake model that scripts their tool calls.

The model is fake but everything around it is real: the agent loop, the tools,
and the MCP server, which runs as a local subprocess.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents import chatbot, instrumented, plain
from tests.fakes import fake_model, tool_call

QUESTION = {"messages": [{"role": "user", "content": "Where is my order ORD-1042?"}]}


def tool_messages(result: dict) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


def test_agent_runs_the_tool_the_model_asks_for():
    agent = plain.build_agent(
        fake_model(
            tool_call("get_order_status", order_id="ORD-1042"),
            "Your order has shipped and arrives on 2026-05-13.",
        )
    )
    result = agent.invoke(QUESTION)

    [message] = tool_messages(result)
    assert message.name == "get_order_status"
    assert message.content == "Order ORD-1042 is Shipped. ETA: 2026-05-13."
    assert result["messages"][-1].text == "Your order has shipped and arrives on 2026-05-13."


def test_instrumented_agent_calls_the_mcp_tool():
    # invoke(), not ainvoke(): support_agent() runs the agent synchronously, so
    # this checks the sync implementation with_sync_call() gives the MCP tool.
    agent = instrumented.build_agent(
        fake_model(tool_call("get_shipping_options"), "We offer three shipping options.")
    )
    result = agent.invoke(QUESTION)

    [message] = tool_messages(result)
    assert message.name == "get_shipping_options"
    assert "Express" in message.text


def test_chatbot_turn_returns_reply_history_and_tools_called():
    model = chatbot.build_model(
        fake_model(tool_call("get_order_status", order_id="ORD-1042"), "It has shipped.")
    )
    reply, history, tools_called = chatbot.chat("Where is ORD-1042?", [], model=model)

    assert reply == "It has shipped."
    assert tools_called == [
        {
            "name": "get_order_status",
            "args": {"order_id": "ORD-1042"},
            "result": "Order ORD-1042 is Shipped. ETA: 2026-05-13.",
        }
    ]
    assert [type(m) for m in history] == [HumanMessage, AIMessage, ToolMessage, AIMessage]


def test_chatbot_carries_history_into_the_next_turn():
    model = chatbot.build_model(fake_model("Hi!", "You're welcome."))
    _, history, _ = chatbot.chat("Hello", [], model=model)
    reply, history, tools_called = chatbot.chat("Thanks", history, model=model)

    assert reply == "You're welcome."
    assert tools_called == []
    assert [m.text for m in history] == ["Hello", "Hi!", "Thanks", "You're welcome."]

"""A scripted chat model for testing the bots without calling a real one."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeToolCallingModel(GenericFakeChatModel):
    """Replays scripted replies, one per call, tool calls included.

    GenericFakeChatModel has no bind_tools(), which create_agent() and the
    chatbot call, so this one accepts the tools and ignores them: the script
    decides which tools get called.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def fake_model(*replies: AIMessage | str) -> FakeToolCallingModel:
    """A fake model that returns replies in order. Build a new one per test:
    the replies are an iterator, used up as the model is called."""
    return FakeToolCallingModel(messages=iter(replies))


def tool_call(name: str, **args) -> AIMessage:
    """The reply a model sends to ask for one tool call."""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])

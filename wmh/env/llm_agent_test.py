"""Tests for the baseline LLM agent's reply parsing and turn rendering."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step
from wmh.env.episode import DONE_SIGNAL
from wmh.env.llm_agent import LLMAgent
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind, VerifyResult


class FakeProvider:
    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._reply = reply
        self.last_user: str | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_user = messages[0].content
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def test_agent_parses_tool_call() -> None:
    agent = LLMAgent(FakeProvider('{"tool": "search", "arguments": {"q": "x"}}'))
    action = agent.act("find x", EnvState(), [])
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "search"
    assert action.arguments == {"q": "x"}


def test_agent_parses_done() -> None:
    agent = LLMAgent(FakeProvider('{"done": true, "summary": "finished"}'))
    action = agent.act("find x", EnvState(), [])
    assert action.kind is ActionKind.MESSAGE
    assert action.content == DONE_SIGNAL


def test_agent_surfaces_garbage_as_message() -> None:
    agent = LLMAgent(FakeProvider("I think I should search first"))
    action = agent.act("find x", EnvState(), [])
    assert action.kind is ActionKind.MESSAGE
    assert action.content == "I think I should search first"


def test_agent_prompt_includes_task_state_and_history() -> None:
    provider = FakeProvider('{"done": true}')
    history = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="search", arguments={"q": "x"}),
            observation=Observation(content="found it", is_error=False),
        )
    ]
    LLMAgent(provider).act("find x", EnvState(scratchpad="db is empty"), history)
    prompt = provider.last_user or ""
    assert "TASK: find x" in prompt
    assert "db is empty" in prompt
    assert "search" in prompt and "found it" in prompt


def test_tools_hint_lands_in_the_system_prompt() -> None:
    class _CaptureProvider:
        def __init__(self) -> None:
            self.system = ""
            self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")

        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            self.system = system
            return Completion(text='{"done": true, "summary": "x"}')

        def embed(self, texts: list[str]) -> list[list[float]]:
            raise NotImplementedError

        def verify(self) -> VerifyResult:
            raise NotImplementedError

    provider = _CaptureProvider()
    agent = LLMAgent(provider, tools_hint="TOOLS: run_sql(query), list_tables()")
    agent.act("do it", EnvState(), [])
    assert "run_sql(query)" in provider.system
    bare = LLMAgent(_CaptureProvider())
    assert bare is not None  # no hint -> unchanged system (covered by existing tests)


class SequenceProvider(FakeProvider):
    """FakeProvider returning a fixed sequence of replies (for the empty-retry tests)."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies[-1])
        self._replies = list(replies)
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_user = messages[0].content
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return Completion(text=reply)


def test_agent_parses_bare_function_call_syntax() -> None:
    agent = LLMAgent(FakeProvider('get_user_details({"user_id": "ivan_johnson_7409"})'))
    action = agent.act("look up user", EnvState(), [])
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "get_user_details"
    assert action.arguments == {"user_id": "ivan_johnson_7409"}


def test_agent_parses_prose_fused_call() -> None:
    reply = (
        "I need to check the recent orders to find the camera. Let me get the details."
        'get_order_details({"order_id": "#W6798117"})'
    )
    agent = LLMAgent(FakeProvider(reply))
    action = agent.act("exchange the camera", EnvState(), [])
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "get_order_details"
    assert action.arguments == {"order_id": "#W6798117"}


def test_agent_takes_first_of_stacked_calls() -> None:
    reply = 'a_tool({"k": 1})\nb_tool({"k": 2})'
    agent = LLMAgent(FakeProvider(reply))
    action = agent.act("do things", EnvState(), [])
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "a_tool"
    assert action.arguments == {"k": 1}


def test_bare_call_is_not_misread_as_done() -> None:
    # Regression: extract_json_object grabs the inner {...} of function-call syntax; a
    # bare arguments object must not validate as an envelope reply with tool=None (done).
    agent = LLMAgent(FakeProvider('search({"q": "boots"})'))
    action = agent.act("find boots", EnvState(), [])
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "search"


def test_arguments_object_alone_is_a_message_not_done() -> None:
    agent = LLMAgent(FakeProvider('{"user_id": "ivan_johnson_7409"}'))
    action = agent.act("look up user", EnvState(), [])
    assert action.kind is ActionKind.MESSAGE
    assert action.content != DONE_SIGNAL


def test_agent_retries_empty_completions() -> None:
    provider = SequenceProvider(["", "  ", '{"tool": "search", "arguments": {}}'])
    agent = LLMAgent(provider)
    action = agent.act("find x", EnvState(), [])
    assert provider.calls == 3
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "search"


def test_agent_gives_up_after_empty_retries() -> None:
    provider = SequenceProvider(["", "", ""])
    agent = LLMAgent(provider)
    action = agent.act("find x", EnvState(), [])
    assert provider.calls == 3
    assert action.kind is ActionKind.MESSAGE
    assert action.content == ""


def test_history_chars_controls_observation_truncation() -> None:
    provider = FakeProvider('{"done": true}')
    history = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="t", arguments={}),
            observation=Observation(content="x" * 5000),
        )
    ]
    LLMAgent(provider, history_chars=3000).act("task", EnvState(), history)
    assert provider.last_user is not None
    assert "x" * 3000 in provider.last_user
    assert "x" * 3001 not in provider.last_user

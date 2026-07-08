"""Tests for the baseline LLM agent's reply parsing and turn rendering."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step
from wmh.env.episode import DONE_SIGNAL
from wmh.env.llm_agent import LLMAgent
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


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

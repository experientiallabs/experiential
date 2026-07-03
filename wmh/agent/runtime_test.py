"""Tests for the pi-style agent runtime, using a scripted provider and a fake environment."""

from __future__ import annotations

from wmh.agent.runtime import AgentRuntime, StopReason
from wmh.agent.skills import SkillLibrary
from wmh.agent.spec import HarnessSpec
from wmh.core.types import Action, Observation
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


class ScriptedProvider:
    """Replies with a fixed list of texts, one per `complete` call (the agent's turns)."""

    def __init__(self, replies: list[str]) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._replies = replies
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        text = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return Completion(text=text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class RecordingEnv:
    """Fake environment: records executed actions and echoes a canned observation."""

    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, action: Action) -> Observation:
        self.actions.append(action)
        return Observation(content=f"ran {action.name}")

    def close(self) -> None:
        pass


def test_runtime_runs_until_submit() -> None:
    provider = ScriptedProvider(
        [
            '{"tool": "bash", "arguments": {"command": "ls"}}',
            '{"tool": "submit", "arguments": {"answer": "done"}}',
        ]
    )
    env = RecordingEnv()
    result = AgentRuntime(HarnessSpec(), provider).run("t1", "list files", env)
    assert result.stop_reason == StopReason.SUBMITTED
    assert result.answer == "done"
    assert result.turns == 2
    assert [a.name for a in env.actions] == ["bash"]  # only the env action reached the env


def test_runtime_hits_turn_cap() -> None:
    provider = ScriptedProvider(['{"tool": "bash", "arguments": {"command": "true"}}'])
    result = AgentRuntime(HarnessSpec(max_turns=3), provider).run("t", "loop", RecordingEnv())
    assert result.stop_reason == StopReason.MAX_TURNS
    assert result.turns == 3


def test_runtime_stops_on_unparseable_reply() -> None:
    result = AgentRuntime(HarnessSpec(), ScriptedProvider(["i refuse to json"])).run(
        "t", "x", RecordingEnv()
    )
    assert result.stop_reason == StopReason.NO_ACTION


def test_save_skill_writes_through_to_library() -> None:
    provider = ScriptedProvider(
        [
            '{"tool": "save_skill", "arguments": {"name": "ls-trick", '
            '"description": "list", "body": "ls -la"}}',
            '{"tool": "submit", "arguments": {"answer": "ok"}}',
        ]
    )
    library = SkillLibrary()
    result = AgentRuntime(HarnessSpec(), provider, library=library).run("t", "x", RecordingEnv())
    assert result.saved_skills == ["ls-trick"]
    assert library.get("ls-trick") is not None


def test_read_skill_returns_body_from_seeded_library() -> None:
    library = SkillLibrary()
    library.save("known", "a known skill", "the body")
    provider = ScriptedProvider(
        [
            '{"tool": "read_skill", "arguments": {"name": "known"}}',
            '{"tool": "submit", "arguments": {"answer": "ok"}}',
        ]
    )
    result = AgentRuntime(HarnessSpec(), provider, library=library).run("t", "x", RecordingEnv())
    # The read_skill observation carries the body back to the agent.
    read_step = result.steps[0]
    assert read_step.observation.content == "the body"

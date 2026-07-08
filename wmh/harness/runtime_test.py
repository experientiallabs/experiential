"""Tests for the rollout loop, using a scripted provider and a recording environment."""

from __future__ import annotations

from wmh.core.types import Action, Observation
from wmh.harness.runtime import AgentRuntime, StopReason
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
        self.closed = False

    def execute(self, action: Action) -> Observation:
        self.actions.append(action)
        return Observation(content=f"ran {action.name}")

    def close(self) -> None:
        self.closed = True


def test_runtime_runs_until_submit() -> None:
    provider = ScriptedProvider(
        [
            '{"tool": "bash", "arguments": {"command": "ls"}}',
            '{"tool": "submit", "arguments": {"answer": "done"}}',
        ]
    )
    env = RecordingEnv()
    result = AgentRuntime(provider).run("t1", "list files", env)
    assert result.stop_reason == StopReason.SUBMITTED
    assert result.answer == "done"
    assert result.turns == 2
    assert [a.name for a in env.actions] == ["bash"]  # submit never reaches the env


def test_runtime_hits_turn_cap() -> None:
    provider = ScriptedProvider(['{"tool": "bash", "arguments": {"command": "true"}}'])
    result = AgentRuntime(provider, max_turns=3).run("t", "loop", RecordingEnv())
    assert result.stop_reason == StopReason.MAX_TURNS
    assert result.turns == 3


def test_runtime_recovers_from_one_unparseable_reply() -> None:
    # One malformed reply is agent noise: a nudge goes back and the run continues.
    provider = ScriptedProvider(
        ["i refuse to json", '{"tool": "submit", "arguments": {"answer": "recovered"}}']
    )
    result = AgentRuntime(provider).run("t", "x", RecordingEnv())
    assert result.stop_reason == StopReason.SUBMITTED
    assert result.answer == "recovered"


def test_runtime_stops_after_two_unparseable_replies() -> None:
    result = AgentRuntime(ScriptedProvider(["nope", "still nope"])).run("t", "x", RecordingEnv())
    assert result.stop_reason == StopReason.NO_ACTION
    assert result.steps == []


def test_unavailable_tool_is_an_error_observation_not_a_crash() -> None:
    provider = ScriptedProvider(
        [
            '{"tool": "teleport", "arguments": {}}',
            '{"tool": "submit", "arguments": {"answer": "ok"}}',
        ]
    )
    env = RecordingEnv()
    result = AgentRuntime(provider).run("t", "x", env)
    assert result.stop_reason == StopReason.SUBMITTED
    assert result.steps[0].observation.is_error  # the bogus call became an error observation
    assert env.actions == []  # and never reached the environment


def test_read_skill_returns_body_and_index_is_in_prompt() -> None:
    from wmh.harness.skills import Skill, SkillLibrary

    provider = ScriptedProvider(
        [
            '{"tool": "read_skill", "arguments": {"name": "count-words"}}',
            '{"tool": "submit", "arguments": {"answer": "ok"}}',
        ]
    )
    library = SkillLibrary(
        [Skill(name="count-words", description="count words", body="wc -w <path>")]
    )
    env = RecordingEnv()
    result = AgentRuntime(provider, skills=library).run("t", "x", env)
    # read_skill is handled by the runtime (never reaches the env) and returns the body.
    assert result.steps[0].observation.content == "wc -w <path>"
    assert env.actions == []
    # Unknown skill -> error observation.
    provider2 = ScriptedProvider(
        [
            '{"tool": "read_skill", "arguments": {"name": "ghost"}}',
            '{"tool": "submit", "arguments": {"answer": "ok"}}',
        ]
    )
    result2 = AgentRuntime(provider2, skills=library).run("t", "x", RecordingEnv())
    assert result2.steps[0].observation.is_error


def test_transcript_shows_actions_and_observations() -> None:
    provider = ScriptedProvider(
        [
            '{"tool": "bash", "arguments": {"command": "wc -w f"}}',
            '{"tool": "submit", "arguments": {"answer": "4"}}',
        ]
    )
    result = AgentRuntime(provider).run("t", "count words", RecordingEnv())
    transcript = result.transcript()
    assert "bash" in transcript
    assert "ran bash" in transcript

"""Tests for run_episode: stop reasons, history threading, env lifecycle."""

from __future__ import annotations

import pytest

from wmo.common.core.types import Action, ActionKind, EnvState, Observation, Step
from wmo.runtime import DONE_SIGNAL, EpisodeResult, StopReason, run_episode


class ScriptedEnv:
    """Deterministic Env: replies `obs {n}` and mutates its live state; records lifecycle calls."""

    def __init__(self, fail_on_step: int | None = None, fail_on_reset: bool = False) -> None:
        self.reset_calls = 0
        self.closed = False
        self._n = 0
        self._fail_on_step = fail_on_step
        self._fail_on_reset = fail_on_reset
        self._state = EnvState()

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        self.reset_calls += 1
        if self._fail_on_reset:
            raise RuntimeError("standup failed mid-allocation")
        self._state = seed_state or EnvState(scratchpad="start")
        return self._state  # the LIVE state view, per the Env.reset contract

    def step(self, action: Action) -> Observation:
        self._n += 1
        if self._fail_on_step is not None and self._n >= self._fail_on_step:
            raise ConnectionError("backend went away")
        self._state.scratchpad += f"|after {self._n}"
        return Observation(content=f"obs {self._n}")

    def close(self) -> None:
        self.closed = True


class ScriptedAgent:
    """Emits tool calls until `stop_after` steps have been taken, then signals done."""

    def __init__(self, stop_after: int | None = None) -> None:
        self._stop_after = stop_after
        self.seen_history_lengths: list[int] = []

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
        self.seen_history_lengths.append(len(history))
        if self._stop_after is not None and len(history) >= self._stop_after:
            return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
        return Action(kind=ActionKind.TOOL_CALL, name="poke", arguments={"turn": len(history)})


def test_episode_stops_when_agent_is_done() -> None:
    env = ScriptedEnv()
    agent = ScriptedAgent(stop_after=2)
    result = run_episode(env, agent, task="t", max_steps=10)

    assert isinstance(result, EpisodeResult)
    assert result.stop_reason == StopReason.AGENT_DONE
    assert [s.observation.content for s in result.steps] == ["obs 1", "obs 2"]
    # The agent saw the growing history each turn: 0, 1, then 2 steps.
    assert agent.seen_history_lengths == [0, 1, 2]
    assert env.closed


def test_episode_hits_max_steps() -> None:
    result = run_episode(ScriptedEnv(), ScriptedAgent(), task="t", max_steps=3)
    assert result.stop_reason == StopReason.MAX_STEPS
    assert len(result.steps) == 3


def test_episode_records_env_error_instead_of_raising() -> None:
    env = ScriptedEnv(fail_on_step=2)
    result = run_episode(env, ScriptedAgent(), max_steps=5)
    assert result.stop_reason == StopReason.ENV_ERROR
    assert result.error is not None and "ConnectionError" in result.error
    assert len(result.steps) == 1  # everything before the failure is kept
    assert env.closed


def test_episode_threads_task_and_state_into_steps() -> None:
    result = run_episode(
        ScriptedEnv(),
        ScriptedAgent(stop_after=1),
        task="find alice",
        seed_state=EnvState(scratchpad="seeded"),
        max_steps=5,
    )
    step = result.steps[0]
    assert step.task == "find alice"
    assert step.state_before.scratchpad == "seeded"


def test_episode_rejects_nonpositive_budget() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        run_episode(ScriptedEnv(), ScriptedAgent(), max_steps=0)


def test_recorded_state_before_snapshots_the_evolving_state() -> None:
    # The env mutates its live state each step; recorded steps must hold per-step copies,
    # not N aliases of one final state.
    result = run_episode(ScriptedEnv(), ScriptedAgent(stop_after=3), max_steps=10)
    scratchpads = [s.state_before.scratchpad for s in result.steps]
    assert scratchpads == ["start", "start|after 1", "start|after 1|after 2"]
    assert result.steps[0].state_before is not result.steps[1].state_before


def test_done_signal_requires_a_message_action() -> None:
    class ToolCallWithSentinelContent:
        def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
            if len(history) >= 2:
                return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
            # A TOOL_CALL that happens to carry the sentinel in content must NOT end the episode.
            return Action(kind=ActionKind.TOOL_CALL, name="poke", content=DONE_SIGNAL)

    result = run_episode(ScriptedEnv(), ToolCallWithSentinelContent(), max_steps=10)
    assert result.stop_reason == StopReason.AGENT_DONE
    assert len(result.steps) == 2  # both tool calls executed


def test_close_called_even_when_reset_raises() -> None:
    env = ScriptedEnv(fail_on_reset=True)
    with pytest.raises(RuntimeError, match="standup failed"):
        run_episode(env, ScriptedAgent(), max_steps=3)
    assert env.closed  # partial allocations get torn down


def test_close_failure_does_not_mask_the_result() -> None:
    class ExplodingCloseEnv(ScriptedEnv):
        def close(self) -> None:
            super().close()
            raise OSError("teardown hiccup")

    result = run_episode(ExplodingCloseEnv(), ScriptedAgent(stop_after=1), max_steps=5)
    assert result.stop_reason == StopReason.AGENT_DONE  # not replaced by the close() error


def test_env_error_names_the_failing_action() -> None:
    result = run_episode(ScriptedEnv(fail_on_step=1), ScriptedAgent(), max_steps=5)
    assert result.stop_reason == StopReason.ENV_ERROR
    assert result.error is not None
    assert "poke" in result.error  # the action that triggered the failure is recorded


def test_agent_exception_is_recorded_not_raised() -> None:
    class ExplodingAgent:
        def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
            raise KeyError("agent bug")

    env = ScriptedEnv()
    result = run_episode(env, ExplodingAgent(), max_steps=5)
    assert result.stop_reason == StopReason.AGENT_ERROR
    assert result.error is not None and "agent.act" in result.error
    assert result.error_traceback is not None and "KeyError" in result.error_traceback
    assert env.closed


def test_env_error_carries_traceback() -> None:
    result = run_episode(ScriptedEnv(fail_on_step=1), ScriptedAgent(), max_steps=5)
    assert result.error_traceback is not None
    assert "ConnectionError" in result.error_traceback

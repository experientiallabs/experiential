"""Tests for the Env protocol and the WorldModelEnv backend."""

from __future__ import annotations

import pytest

from wmo.common.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmo.common.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmo.runtime import Env
from wmo.simulation import WorldModelEnv
from wmo.simulation.model.world_model import WorldModel
from wmo.simulation.retrieval import EmbeddingRetriever, HashingEmbedder


class FakeProvider:
    """Returns a canned world-model JSON completion."""

    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._reply = reply

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _world_model(reply: str) -> WorldModel:
    demo = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "bob"}),
        observation=Observation(content="user found: bob"),
    )
    retriever = EmbeddingRetriever(HashingEmbedder(dim=64))
    retriever.index([Trace(trace_id="t", steps=[demo])])
    return WorldModel(FakeProvider(reply), retriever, top_k=1)


def test_world_model_env_satisfies_protocol() -> None:
    env = WorldModelEnv(_world_model('{"output": "ok", "is_error": false}'))
    assert isinstance(env, Env)


def test_world_model_env_episode_lifecycle() -> None:
    wm = _world_model('{"output": "user found: alice", "is_error": false}')
    env = WorldModelEnv(wm)

    state = env.reset(task="look up alice", seed_state=EnvState(scratchpad="fresh"))
    assert state.scratchpad == "fresh"
    assert wm.get_session(env.session_id).task == "look up alice"

    obs = env.step(Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "alice"}))
    assert obs.content == "user found: alice"
    assert len(wm.get_session(env.session_id).history) == 1

    env.close()
    with pytest.raises(RuntimeError, match="call reset"):
        _ = env.session_id


def test_world_model_env_reset_starts_fresh_session() -> None:
    env = WorldModelEnv(_world_model('{"output": "ok", "is_error": false}'))
    env.reset(task="a")
    first = env.session_id
    env.reset(task="b")
    assert env.session_id != first


def test_close_ends_session_in_world_model_and_keeps_usage() -> None:
    wm = _world_model('{"output": "ok", "is_error": false}')
    env = WorldModelEnv(wm)
    env.reset(task="a")
    session_id = env.session_id
    env.step(Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={}))

    env.close()

    # The session and its tracker are gone from the world model (no batch-rollout leak)...
    with pytest.raises(KeyError):
        wm.get_session(session_id)
    with pytest.raises(KeyError):
        wm.session_usage(session_id)
    # ...but the final usage record survives on the env.
    assert env.usage is not None and env.usage.run_id == session_id
    env.close()  # idempotent


def test_reset_releases_previous_session() -> None:
    wm = _world_model('{"output": "ok", "is_error": false}')
    env = WorldModelEnv(wm)
    env.reset(task="a")
    first = env.session_id
    env.reset(task="b")
    with pytest.raises(KeyError):  # first session must not linger in the world model
        wm.get_session(first)


def test_recorded_history_snapshots_state_per_step() -> None:
    # WorldModel._update_state mutates session.state in place; recorded steps must not alias it.
    wm = _world_model('{"output": "ok", "is_error": false, "state_note": "did a thing"}')
    env = WorldModelEnv(wm)
    env.reset(task="a")
    env.step(Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={}))
    env.step(Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={}))

    history = wm.get_session(env.session_id).history
    assert history[0].state_before.scratchpad == ""  # state BEFORE the first action
    assert history[1].state_before.scratchpad == "- did a thing"
    assert history[0].state_before is not history[1].state_before


def test_score_on_close_captures_episode_score_before_session_ends() -> None:
    """RL rollouts: run_episode closes the env, so scoring must happen inside close()."""
    from wmo.optimize.reward import EpisodeScore

    env_reply = '{"output": "found u1", "is_error": false}'
    judge_reply = '{"success": true, "reward": 0.7, "step_rewards": [0.7], "critique": "nice"}'
    retriever = EmbeddingRetriever(HashingEmbedder(dim=64))
    wm = WorldModel(
        FakeProvider(env_reply), retriever, top_k=1, reward_provider=FakeProvider(judge_reply)
    )
    env = WorldModelEnv(wm, score_on_close=True)
    env.reset(task="find u1")
    session_id = env.session_id
    env.step(Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "u1"}))
    env.close()
    assert isinstance(env.last_score, EpisodeScore)
    assert env.last_score.reward == 0.7
    assert env.last_score.critique == "nice"
    # the session is gone (memory freed) but the score survived
    with pytest.raises(KeyError):
        wm.get_session(session_id)


def test_last_score_raises_before_any_scored_episode() -> None:
    env = WorldModelEnv(_world_model("{}"), score_on_close=True)
    with pytest.raises(RuntimeError, match="no scored episode"):
        _ = env.last_score


def test_close_without_score_on_close_does_not_judge() -> None:
    env = WorldModelEnv(_world_model("{}"))
    env.reset(task="t")
    env.close()
    with pytest.raises(RuntimeError):
        _ = env.last_score


def test_judge_failure_during_scoring_close_ends_session_and_surfaces_cause() -> None:
    """A throttled judge must not leak the session or masquerade as 'no scored episode'."""

    class ExplodingJudge(FakeProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            raise ConnectionError("throttled")

    retriever = EmbeddingRetriever(HashingEmbedder(dim=64))
    wm = WorldModel(
        FakeProvider('{"output": "ok", "is_error": false}'),
        retriever,
        top_k=1,
        reward_provider=ExplodingJudge("unused"),
    )
    env = WorldModelEnv(wm, score_on_close=True)
    env.reset(task="t")
    session_id = env.session_id
    env.step(Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={}))
    env.close()  # must not raise (run_episode swallows teardown), must still end the session
    with pytest.raises(KeyError):
        wm.get_session(session_id)  # session freed despite the judge failure
    with pytest.raises(RuntimeError, match="scoring failed") as excinfo:
        _ = env.last_score
    assert isinstance(excinfo.value.__cause__, ConnectionError)

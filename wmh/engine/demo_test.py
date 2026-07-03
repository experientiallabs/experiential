"""Tests for the demo scenario replay, with a fake world-model provider (no network)."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.engine.demo import run_demo
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class ScriptedProvider:
    """World-model provider that always predicts the same observation."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(text='{"output": "found u1", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _step(name: str, observed: str) -> Step:
    return Step(
        action=Action(kind=ActionKind.TOOL_CALL, name=name, arguments={"id": "u1"}),
        observation=Observation(content=observed),
        task="look up users",
    )


def _world_model(provider: ScriptedProvider) -> WorldModel:
    retriever = EmbeddingRetriever(HashingEmbedder(dim=32))
    retriever.index([Trace(trace_id="corpus", steps=[_step("get_user", "found u0")])])
    return WorldModel(provider, retriever, top_k=3)


def test_run_demo_replays_open_loop_with_predicted_vs_actual() -> None:
    wm = _world_model(ScriptedProvider())
    trace = Trace(
        trace_id="scenario",
        steps=[_step("get_user", "found u1"), _step("list_users", "u1, u2")],
    )

    result = run_demo(wm, trace, max_steps=5)

    assert result.trace_id == "scenario"
    assert result.task == "look up users"
    assert len(result.steps) == 2
    assert "get_user" in result.first_env_prompt  # retrieved demo appears in the first prompt
    # Prediction always "found u1": exact match on step 1, differs on step 2.
    assert result.steps[0].exact_match is True
    assert result.steps[1].exact_match is False
    assert result.steps[1].actual.content == "u1, u2"


def test_run_demo_open_loop_pins_history_to_recorded_observations() -> None:
    wm = _world_model(ScriptedProvider())
    trace = Trace(
        trace_id="scenario",
        steps=[_step("get_user", "RECORDED-1"), _step("list_users", "RECORDED-2")],
    )
    run_demo(wm, trace, max_steps=5)
    # Open loop: after replay, the session history holds the RECORDED observations, not the
    # model's predictions — later steps were conditioned on ground truth.
    session = next(iter(wm._sessions.values()))
    assert [s.observation.content for s in session.history] == ["RECORDED-1", "RECORDED-2"]


def test_run_demo_caps_steps() -> None:
    wm = _world_model(ScriptedProvider())
    trace = Trace(trace_id="long", steps=[_step(f"t{i}", "x") for i in range(9)])
    result = run_demo(wm, trace, max_steps=3)
    assert len(result.steps) == 3

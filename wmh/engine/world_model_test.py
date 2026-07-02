"""Tests for the WorldModel session lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import wmh.telemetry as telemetry
from wmh.config import ArtifactPaths, HarnessConfig, save_config
from wmh.config.settings import set_telemetry_enabled
from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.engine.world_model import WorldModel
from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


def test_world_model_new_session_works() -> None:
    wm = WorldModel.__new__(WorldModel)
    wm._telemetry_root = Path(".wmh")
    wm._sessions = {}
    wm._trackers = {}
    session = WorldModel.new_session(wm, task="hi")
    assert session.id
    assert WorldModel.get_session(wm, session.id) is session


class FakeProvider:
    """Returns a canned world-model JSON completion; captures the last prompt for assertions."""

    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._reply = reply
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_system = system
        self.last_user = messages[0].content
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _retriever_with(steps: list[Step]) -> EmbeddingRetriever:
    r = EmbeddingRetriever(HashingEmbedder(dim=64))
    r.index([Trace(trace_id="t", steps=steps)])
    return r


def test_step_predicts_parses_and_advances_session() -> None:
    provider = FakeProvider(
        '{"output": "user found: alice", "is_error": false, "state_note": "looked up alice"}'
    )
    demo = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "bob"}),
        observation=Observation(content="user found: bob"),
    )
    wm = WorldModel(provider, _retriever_with([demo]), top_k=3)
    session = wm.new_session(task="look up alice")

    obs = wm.step(
        session.id, Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "alice"})
    )

    assert obs.content == "user found: alice"
    assert obs.is_error is False
    # The retrieved demo made it into the prompt.
    assert provider.last_user is not None and "get_user" in provider.last_user
    # Session advanced: history grew and the scratchpad recorded the state note.
    assert len(session.history) == 1
    assert "looked up alice" in session.state.scratchpad


def test_step_marks_errors_and_enriches_buffer() -> None:
    provider = FakeProvider('{"output": "no such reservation", "is_error": true}')
    retriever = _retriever_with([])
    wm = WorldModel(provider, retriever, top_k=3)
    session = wm.new_session(task="check r_999")

    obs = wm.step(
        session.id,
        Action(kind=ActionKind.TOOL_CALL, name="get_reservation", arguments={"id": "r_999"}),
    )
    assert obs.is_error is True
    # The freshly produced step was added to the buffer (online enrichment).
    assert len(retriever._steps) == 1


class _UsageProvider(FakeProvider):
    """FakeProvider that also reports token usage, for serve-metering assertions."""

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_system = system
        self.last_user = messages[0].content
        return Completion(text=self._reply, usage=TokenUsage(input_tokens=120, output_tokens=30))


def test_step_meters_usage_per_session() -> None:
    provider = _UsageProvider('{"output": "ok", "is_error": false}')
    provider.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="claude-opus-4-8")
    wm = WorldModel(provider, _retriever_with([]), top_k=1)
    session = wm.new_session(task="t")

    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))

    usage = wm.session_usage(session.id)
    assert usage.kind == "serve"
    assert usage.total.calls == 2
    assert usage.total.input_tokens == 240
    assert usage.total.output_tokens == 60
    # 240*5/1e6 + 60*25/1e6 = 0.0012 + 0.0015 = 0.0027 (float division → approx)
    assert usage.total.cost_usd == pytest.approx(0.0027)


class _FailedOverProvider(_UsageProvider):
    """Mimics a failover chain: config reports the primary, Completion reports who served."""

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        completion = super().complete(
            system, messages, temperature=temperature, max_tokens=max_tokens
        )
        return completion.model_copy(update={"model": "claude-haiku-4-5"})


def test_step_meters_usage_at_serving_model_rate() -> None:
    # Regression: serve-path metering must price a failed-over call at the SERVING model's
    # rate (haiku 1/5 per Mtok), not the configured primary's (opus 5/25) — a 5x over-report.
    provider = _FailedOverProvider('{"output": "ok", "is_error": false}')
    provider.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="claude-opus-4-8")
    wm = WorldModel(provider, _retriever_with([]), top_k=1)
    session = wm.new_session(task="t")

    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))

    usage = wm.session_usage(session.id)
    assert usage.total.cost_usd == pytest.approx((120 * 1.0 + 30 * 5.0) / 1_000_000)


def test_step_telemetry_counts_steps_without_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[object] = []

    class FakePosthog:
        def __init__(self, project_api_key: str, **kwargs: object) -> None:
            pass

        def capture(self, event: str, **kwargs: object) -> str:
            calls.append({"event": event, **kwargs})
            return "message-id"

        def shutdown(self) -> None:
            pass

    telemetry._CLIENTS.clear()
    monkeypatch.setattr(telemetry, "Posthog", FakePosthog)
    monkeypatch.setenv("WMH_TELEMETRY", "1")
    monkeypatch.setenv("WMH_POSTHOG_PROJECT_API_KEY", "phc_test")

    provider = _UsageProvider('{"output": "secret observation", "is_error": false}')
    provider.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="claude-opus-4-8")
    wm = WorldModel(provider, _retriever_with([]), top_k=1, telemetry_root=tmp_path / ".wmh")
    session = wm.new_session(task="secret task")
    wm.step(
        session.id,
        Action(kind=ActionKind.TOOL_CALL, name="secret_tool", arguments={"secret": "value"}),
    )

    serialized = json.dumps(calls)
    assert "wmh generated trace started" in serialized
    assert "wmh generated step completed" in serialized
    assert "generated_trace_count" in serialized
    assert "generated_step_count" in serialized
    assert "secret task" not in serialized
    assert "secret_tool" not in serialized
    assert "secret observation" not in serialized
    assert "value" not in serialized


def test_load_reads_artifact(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    root = tmp_path / ".wmh"
    # embed_dim must match the embedder the index was built with (64 here), or load() rebuilds a
    # mismatched query embedder. This is the contract WorldModel.load relies on.
    save_config(HarnessConfig(top_k=2, embed_dim=64), root)
    paths = ArtifactPaths(root)
    paths.optimized_prompt.parent.mkdir(parents=True, exist_ok=True)
    paths.optimized_prompt.write_text("OPTIMIZED ENV PROMPT", encoding="utf-8")
    r = _retriever_with(
        [
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "x"}),
                observation=Observation(content="ok"),
            )
        ]
    )
    r.save(paths.index)

    wm = WorldModel.load(str(root), FakeProvider("{}"))
    assert wm._env_prompt == "OPTIMIZED ENV PROMPT"
    assert wm._top_k == 2
    # The persisted index was reloaded: the stored step is retrievable.
    restored = wm._retriever.topk(
        EnvState(), Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "x"}), k=1
    )
    assert len(restored) == 1 and restored[0].observation.content == "ok"


def test_load_named_model_uses_project_root_for_telemetry_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[object] = []

    class FakePosthog:
        def __init__(self, project_api_key: str, **kwargs: object) -> None:
            pass

        def capture(self, event: str, **kwargs: object) -> str:
            calls.append({"event": event, **kwargs})
            return "message-id"

        def shutdown(self) -> None:
            pass

    project_root = tmp_path / ".wmh"
    model_dir = project_root / "models" / "demo"
    save_config(HarnessConfig(top_k=2, embed_dim=64), model_dir)
    set_telemetry_enabled(False, project_root)
    telemetry._CLIENTS.clear()
    monkeypatch.setattr(telemetry, "Posthog", FakePosthog)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("WMH_POSTHOG_PROJECT_API_KEY", "phc_test")

    wm = WorldModel.load(str(model_dir), FakeProvider("{}"))
    wm.new_session(task="should respect project opt-out")

    assert wm._telemetry_root == project_root
    assert calls == []


def test_score_session_meters_judge_separately_from_serve() -> None:
    """Reward-judge tokens land under Phase.JUDGE on the session tracker, not SERVE (D12 split)."""
    from wmh.optimize.reward import EpisodeScore
    from wmh.tracking import Phase

    class JudgeReply:
        def __init__(self) -> None:
            self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="judge-m")

        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            return Completion(
                text='{"success": false, "reward": 0.2, "step_rewards": [0.2], "critique": "c"}',
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        def verify(self) -> VerifyResult:
            raise NotImplementedError

    env_provider = FakeProvider('{"output": "found u1", "is_error": false}')
    retriever = _retriever_with(
        [
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={}),
                observation=Observation(content="found u1"),
            )
        ]
    )
    wm = WorldModel(env_provider, retriever, top_k=1, reward_provider=JudgeReply())
    session = wm.new_session(task="do the thing")
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={}))
    score = wm.score_session(session.id)
    assert isinstance(score, EpisodeScore)
    assert score.reward == 0.2
    assert session.history[-1].observation.reward == 0.2
    usage = wm.session_usage(session.id)
    assert usage.by_phase[Phase.JUDGE].input_tokens == 10  # judge cost split out (D12)
    assert Phase.SERVE in usage.by_phase  # the step call stays attributed to SERVE

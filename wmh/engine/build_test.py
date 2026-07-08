"""Tests for the build pipeline (ingest -> split -> index -> GEPA -> persist), no network."""

from __future__ import annotations

import json

import pytest

from wmh.config import ArtifactPaths, HarnessConfig
from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.engine.build import build, split_traces, split_traces_3way
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import HashingEmbedder


class FakeProvider:
    """Canned world-model JSON for rollouts; a fixed 'improved' prompt for GEPA reflection."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self.systems: list[str] = []  # system prompt of every complete() call, for assertions

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.systems.append(system)
        if "improve the system prompt" in system:
            return Completion(text="IMPROVED ENV PROMPT")
        if "grade a world model" in system:  # the judge
            return Completion(
                text=(
                    '{"format": 0.5, "factuality": 0.5, "consistency": 0.5, '
                    '"realism": 0.5, "quality": 0.5, "critique": "be more specific"}'
                )
            )
        return Completion(text='{"output": "ok", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def test_split_traces_is_deterministic_and_partitions() -> None:
    traces = [Trace(trace_id=f"t{i}") for i in range(50)]
    a_train, a_test = split_traces(traces, 0.8)
    b_train, b_test = split_traces(list(reversed(traces)), 0.8)
    # Same assignment regardless of order; every trace lands in exactly one side.
    assert {t.trace_id for t in a_train} == {t.trace_id for t in b_train}
    assert len(a_train) + len(a_test) == 50
    assert 0 < len(a_train) < 50  # roughly an 80/20 split, both sides non-empty


def test_split_traces_3way_partitions_disjointly_and_is_stable() -> None:
    traces = [Trace(trace_id=f"t{i}") for i in range(60)]
    train, val, test = split_traces_3way(traces, 0.6, 0.2)
    ids = lambda ts: {t.trace_id for t in ts}  # noqa: E731
    # Disjoint and exhaustive.
    assert ids(train) | ids(val) | ids(test) == ids(traces)
    assert (
        not (ids(train) & ids(val)) and not (ids(val) & ids(test)) and not (ids(train) & ids(test))
    )
    assert len(train) + len(val) + len(test) == 60
    assert all(len(s) > 0 for s in (train, val, test))
    # Order-independent (same stable hash).
    r_train, r_val, r_test = split_traces_3way(list(reversed(traces)), 0.6, 0.2)
    assert ids(train) == ids(r_train) and ids(val) == ids(r_val) and ids(test) == ids(r_test)


def test_split_traces_3way_train_is_prefix_compatible_with_2way() -> None:
    # The 3-way train set is exactly the 2-way train set at the same cut, so switching to a 3-way
    # split never reshuffles which traces GEPA trains on — it only carves val out of the old test.
    traces = [Trace(trace_id=f"t{i}") for i in range(60)]
    two_train, two_test = split_traces(traces, 0.6)
    three_train, three_val, three_test = split_traces_3way(traces, 0.6, 0.2)
    assert {t.trace_id for t in two_train} == {t.trace_id for t in three_train}
    assert {t.trace_id for t in two_test} == {t.trace_id for t in three_val} | {
        t.trace_id for t in three_test
    }


def test_split_traces_3way_rejects_degenerate_fractions() -> None:
    traces = [Trace(trace_id=f"t{i}") for i in range(10)]
    with pytest.raises(ValueError, match="train_frac"):
        split_traces_3way(traces, 0.7, 0.4)  # sums to > 1 -> empty test
    with pytest.raises(ValueError, match="train_frac"):
        split_traces_3way(traces, 0.0, 0.5)


def test_cap_gepa_valset_bounds_steps_and_keeps_at_least_one_trace() -> None:
    from wmh.engine.build import _GEPA_VAL_STEP_CAP, _cap_gepa_valset

    def trace_with_steps(tid: str, n: int) -> Trace:
        step = Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="get", arguments={}),
            observation=Observation(content="ok"),
        )
        return Trace(trace_id=tid, steps=[step.model_copy() for _ in range(n)])

    many = [trace_with_steps(f"t{i}", 2) for i in range(200)]  # 400 steps uncapped
    capped = _cap_gepa_valset(many)
    assert sum(len(t.steps) for t in capped) <= _GEPA_VAL_STEP_CAP
    assert capped == many[: len(capped)]  # stable prefix of the (already shuffled) split

    # A single over-cap trace still passes through: never starve GEPA of a valset entirely.
    huge = trace_with_steps("huge", _GEPA_VAL_STEP_CAP + 10)
    assert _cap_gepa_valset([huge]) == [huge]


def test_build_writes_a_loadable_artifact(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    # A tiny OTel JSONL with one tool-call step.
    span_llm = {
        "traceId": "a" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
            {"key": "gen_ai.prompt", "value": {"stringValue": "look up u1"}},
        ],
    }
    span_tool = {
        "traceId": "a" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
        ],
    }
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text(
        json.dumps(span_llm) + "\n" + json.dumps(span_tool) + "\n", encoding="utf-8"
    )

    root = tmp_path / ".wmh"
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=4,
        train_split=0.5,
    )
    result = build(
        config,
        file=str(traces_file),
        root=str(root),
        serve_provider=FakeProvider(),
        embedder=HashingEmbedder(dim=64),
    )

    paths = ArtifactPaths(root)
    assert paths.config.exists()
    assert paths.optimized_prompt.read_text(encoding="utf-8")  # a non-empty winning prompt
    assert json.loads(paths.frontier.read_text(encoding="utf-8"))  # frontier persisted
    assert result.prompt
    # The index round-trips: a freshly loaded WorldModel can retrieve the indexed step.
    from wmh.engine.world_model import WorldModel

    wm = WorldModel.load(str(root), FakeProvider())
    assert wm.sample_steps(5)


def test_build_judge_stays_on_the_judge_provider(tmp_path) -> None:  # noqa: ANN001 - fixture
    # The serve provider may be a failover chain; the judge (GEPA's fitness metric) must run on
    # the separately supplied pinned provider so optimization is scored by one model throughout.
    span_llm = {
        "traceId": "b" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
        ],
    }
    span_tool = {
        "traceId": "b" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
        ],
    }
    # Four copies of the trace under distinct trace ids: the 3-way split (train/val/test) must
    # leave a non-empty val set, or GEPA never scores a candidate and the judge is never called.
    lines: list[str] = []
    for i in range(4):
        trace_id = f"{i:x}" * 32
        lines.append(json.dumps({**span_llm, "traceId": trace_id, "spanId": f"s1-{i}"}))
        lines.append(json.dumps({**span_tool, "traceId": trace_id, "spanId": f"s2-{i}"}))
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    serve, judge = FakeProvider(), FakeProvider()
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=4,
        train_split=0.5,
    )
    result = build(
        config,
        file=str(traces_file),
        root=str(tmp_path / ".wmh"),
        serve_provider=serve,
        judge_provider=judge,
        embedder=HashingEmbedder(dim=64),
    )
    assert result.prompt
    assert all("grade a world model" not in s for s in serve.systems)
    assert any("grade a world model" in s for s in judge.systems)
    assert all("grade a world model" in s for s in judge.systems)  # judge does ONLY judging

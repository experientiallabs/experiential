"""Tests for the improvement report (D-REPORT): aggregation over an outcome matrix."""

from __future__ import annotations

import pytest

from wmo.optimize.routing.compression import (
    CompressionConfig,
    CompressionResult,
    estimate_tokens,
    register_compressor,
)
from wmo.optimize.routing.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.routing.policy import ClusterRanking, EmbedderSpec, RoutingPolicy
from wmo.optimize.routing.report import build_report
from wmo.providers.base import Embedder, ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry
from wmo.simulation.retrieval.embedders import HashingEmbedder


def _entries() -> list[PoolEntry]:
    return [
        PoolEntry(name="fable-5", kind=ProviderKind.ANTHROPIC, model="claude-fable-5"),
        PoolEntry(
            name="cheap",
            kind=ProviderKind.OPENAI,
            model="custom-cheap",
            tier="open",
            input_per_mtok=1.0,
            output_per_mtok=2.0,
        ),
    ]


def _outcome(sid: str, task: str, model: str, reward: float, cost: float) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=sid,
        task=task,
        model=model,
        reward=reward,
        success=reward >= 0.5,
        steps=2,
        stop_reason="agent_done",
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        cost_usd=cost,
        call_seconds=[0.2, 0.4],
        replies=["a", "b"],
    )


_SQL_TASK = "SELECT count(*) FROM superheroes"
_PROSE_TASK = "write a friendly email"


def _matrix() -> OutcomeMatrix:
    return OutcomeMatrix(
        pool=_entries(),
        outcomes=[
            _outcome("s1", _SQL_TASK, "fable-5", reward=0.9, cost=0.010),
            _outcome("s2", _PROSE_TASK, "fable-5", reward=0.8, cost=0.012),
            _outcome("s1", _SQL_TASK, "cheap", reward=0.7, cost=0.001),
            _outcome("s2", _PROSE_TASK, "cheap", reward=0.8, cost=0.001),
        ],
    )


def _cluster_policy() -> RoutingPolicy:
    embedder = HashingEmbedder(dim=64)
    sql, prose = embedder.embed([_SQL_TASK, _PROSE_TASK])
    return RoutingPolicy(
        kind="rank",
        default_model="cheap",
        pool=_entries(),
        # This policy represents one fitted on a separate matrix. None of these report rows
        # overlap its recorded fit evidence, so all are legitimately held out.
        fit_scenario_ids=["external-fit-scenario"],
        embedder=EmbedderSpec(dim=64),
        top_k_clusters=1,
        clusters=[
            ClusterRanking(cluster_id=0, label="sql", centroid=sql, ranking=["fable-5", "cheap"]),
            ClusterRanking(
                cluster_id=1, label="prose", centroid=prose, ranking=["cheap", "fable-5"]
            ),
        ],
    )


def test_report_headline_vs_baseline() -> None:
    report = build_report(
        _matrix(),
        _cluster_policy(),
        baseline="fable-5",
        endpoint="tau-bench",
        generated_at="2026-07-24T00:00:00Z",
    )
    # Routed: s1 -> fable-5 (0.9, $0.010), s2 -> cheap (0.8, $0.001).
    assert report.headline.accuracy == pytest.approx(0.85)
    assert report.headline.cost_per_run_usd == pytest.approx(0.0055)
    # Baseline: fable-5 on everything.
    assert report.headline.baseline_accuracy == pytest.approx(0.85)
    assert report.headline.baseline_cost_per_run_usd == pytest.approx(0.011)
    assert report.scenario_count == 2
    assert "2 held-out scenarios" in report.scenario_label
    assert report.baseline.model_id == "fable-5"
    assert report.baseline.tier == "frontier"
    assert report.cost_assumptions  # the honesty string is mandatory


def test_report_model_mix_and_candidates() -> None:
    report = build_report(
        _matrix(),
        _cluster_policy(),
        baseline="fable-5",
        endpoint="tau-bench",
        generated_at="2026-07-24T00:00:00Z",
    )
    mix = {m.model_id: m.share for m in report.model_mix}
    assert mix == {"fable-5": 0.5, "cheap": 0.5}
    by_name = {c.model.model_id: c for c in report.candidates}
    assert by_name["cheap"].accuracy == pytest.approx(0.75)
    assert by_name["cheap"].cost_per_run_usd == pytest.approx(0.001)
    assert by_name["cheap"].model.tier == "open"
    assert by_name["fable-5"].latency_p50_ms == pytest.approx(300.0)


def test_report_excludes_rows_recorded_as_router_fit_evidence() -> None:
    policy = _cluster_policy().model_copy(update={"fit_scenario_ids": ["s1"]})

    report = build_report(
        _matrix(),
        policy,
        baseline="fable-5",
        endpoint="tau-bench",
        generated_at="2026-07-24T00:00:00Z",
    )

    assert report.scenario_ids == ["s2"]
    assert report.scenario_count == report.headline.scenarios_compared == 1
    assert report.headline.accuracy == pytest.approx(0.8)
    assert all(candidate.scored_episodes == 1 for candidate in report.candidates)


def test_report_refuses_when_every_scenario_was_used_for_router_fit() -> None:
    policy = _cluster_policy().model_copy(update={"fit_scenario_ids": ["s1", "s2"]})

    with pytest.raises(ValueError, match="no scenario outside the router fit set"):
        build_report(
            _matrix(),
            policy,
            baseline="fable-5",
            endpoint="tau-bench",
            generated_at="2026-07-24T00:00:00Z",
        )


def test_report_static_policy_and_unscored_rows() -> None:
    matrix = _matrix()
    matrix.outcomes.append(
        ScenarioOutcome(scenario_id="s3", task="broken run", model="cheap", error="env exploded")
    )
    policy = RoutingPolicy(kind="static", default_model="cheap", pool=_entries())
    report = build_report(
        matrix, policy, baseline="fable-5", endpoint="e", generated_at="2026-07-24T00:00:00Z"
    )
    by_name = {c.model.model_id: c for c in report.candidates}
    assert by_name["cheap"].unscored_episodes == 1  # surfaced, never averaged as 0
    assert by_name["cheap"].accuracy == pytest.approx(0.75)


def test_headline_compares_only_commonly_scored_scenarios() -> None:
    # The cheap model CRASHED on the two hard scenarios and scored well on the two easy ones.
    # Averaging each side over whatever it happened to score would hand the cheap model a
    # winning headline (0.85 vs 0.72) on the strength of the scenarios it skipped.
    pool = _entries()
    outcomes: list[ScenarioOutcome] = []
    for index in range(4):
        hard = index < 2
        task = f"task {index}"
        outcomes.append(
            _outcome(f"s{index}", task, "fable-5", reward=0.3 if hard else 0.9, cost=0.02)
        )
        if hard:
            outcomes.append(
                ScenarioOutcome(scenario_id=f"s{index}", task=task, model="cheap", error="timeout")
            )
        else:
            outcomes.append(_outcome(f"s{index}", task, "cheap", reward=0.85, cost=0.001))
    matrix = OutcomeMatrix(pool=pool, outcomes=outcomes)
    report = build_report(
        matrix,
        RoutingPolicy(kind="static", default_model="cheap", pool=pool),
        baseline="fable-5",
        endpoint="e",
        generated_at="2026-07-24T00:00:00Z",
    )
    # Both sides over the two commonly-scored scenarios: the baseline wins, as it should.
    assert report.headline.accuracy == pytest.approx(0.85)
    assert report.headline.baseline_accuracy == pytest.approx(0.9)
    assert report.headline.scenarios_compared == 2
    assert report.headline.scenarios_excluded == 2
    assert report.scenario_count == 4  # the excluded scenarios are still counted, not hidden


def test_report_over_a_fully_unscored_matrix_raises() -> None:
    pool = _entries()
    matrix = OutcomeMatrix(
        pool=pool,
        outcomes=[
            ScenarioOutcome(scenario_id=f"s{i}", task=f"t{i}", model=name, error="provider 429")
            for i in range(3)
            for name in ("fable-5", "cheap")
        ],
    )
    with pytest.raises(ValueError, match="nothing to compare"):
        build_report(
            matrix,
            RoutingPolicy(kind="static", default_model="cheap", pool=pool),
            baseline="fable-5",
            endpoint="e",
            generated_at="2026-07-24T00:00:00Z",
        )


def test_report_builds_the_embedder_once(monkeypatch: pytest.MonkeyPatch) -> None:
    builds = 0
    original = EmbedderSpec.build

    def counted(self: EmbedderSpec) -> Embedder:
        nonlocal builds
        builds += 1
        return original(self)

    monkeypatch.setattr(EmbedderSpec, "build", counted)
    build_report(
        _matrix(),
        _cluster_policy(),
        baseline="fable-5",
        endpoint="e",
        generated_at="2026-07-24T00:00:00Z",
    )
    assert builds == 1  # not one per routed scenario


def test_report_requires_baseline_in_matrix() -> None:
    with pytest.raises(KeyError, match="nope"):
        build_report(
            _matrix(),
            _cluster_policy(),
            baseline="nope",
            endpoint="e",
            generated_at="2026-07-24T00:00:00Z",
        )


def test_blank_retry_calls_do_not_deflate_latency() -> None:
    # `LLMAgent` retries a blank completion, and the metering wrapper records each attempt as
    # its own fast call. Counting those would make a model that blanks often look FASTER.
    from wmo.optimize.routing.report import _productive_call_seconds

    outcome = ScenarioOutcome(
        scenario_id="s",
        task="t",
        model="m",
        reward=1.0,
        call_seconds=[0.01, 0.01, 2.0],
        replies=["", "   ", '{"done": true}'],
    )
    assert _productive_call_seconds([outcome]) == [2.0]


def test_latency_counts_every_call_when_replies_are_not_call_aligned() -> None:
    # The real-episode runner records one reply per message that HAS content but one duration
    # per timed call, so durations must not be attributed to replies by index there.
    from wmo.optimize.routing.report import _productive_call_seconds

    outcome = ScenarioOutcome(
        scenario_id="s",
        task="t",
        model="m",
        reward=1.0,
        call_seconds=[1.0, 2.0, 3.0],
        replies=["only one reply"],
    )
    assert _productive_call_seconds([outcome]) == [1.0, 2.0, 3.0]


class _RecordingCompressor:
    """Records every segment it is handed and returns it unchanged (a pass-through spy)."""

    id = "report-test-recorder"
    version = "1"
    append_stable = True

    def __init__(self) -> None:
        self.seen: list[str] = []

    def compress(self, segments: list[str], config: CompressionConfig) -> CompressionResult:
        del config
        self.seen.extend(segments)
        tokens = sum(estimate_tokens(segment) for segment in segments)
        return CompressionResult(
            segments=list(segments),
            tokens_in_raw=tokens,
            tokens_in_compressed=tokens,
            latency_s=0.0,
        )


_RECORDER = _RecordingCompressor()
register_compressor(_RECORDER)


def test_a_compressed_policys_report_replays_in_the_geometry_it_was_fitted_in() -> None:
    """The reporting half of representation consistency (C2's Q2 rule).

    A compressed endpoint's bank lives in the geometry of compressed text, and serving compresses
    each request before the router embeds it. A report that embedded RAW task text against that
    bank would measure a policy nobody serves: the queries land farther from every row, the novelty
    floor trips, and routing reads as collapsed to the fallback.
    """
    _RECORDER.seen.clear()
    arm = CompressionConfig(compressor_id=_RecordingCompressor.id, aggressiveness=0.5)
    policy = _cluster_policy().model_copy(update={"compression": arm, "fit_compression": arm})
    report = build_report(
        _matrix(),
        policy,
        baseline="fable-5",
        endpoint="tau-bench",
        generated_at="2026-07-24T00:00:00Z",
    )
    assert report.scenario_count == 2
    # Every routed scenario's task text went through the compressor on its way to the embedder.
    assert sorted(_RECORDER.seen) == sorted([_SQL_TASK, _PROSE_TASK])


def test_an_uncompressed_policys_report_embeds_the_task_text_as_it_is() -> None:
    _RECORDER.seen.clear()
    build_report(
        _matrix(),
        _cluster_policy(),
        baseline="fable-5",
        endpoint="tau-bench",
        generated_at="2026-07-24T00:00:00Z",
    )
    assert _RECORDER.seen == []

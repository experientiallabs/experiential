"""Tests for the improvement report (D-REPORT): aggregation over an outcome matrix."""

from __future__ import annotations

import pytest

from wmh.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmh.optimize.policy import ClusterRanking, EmbedderSpec, RoutingPolicy
from wmh.optimize.report import build_report
from wmh.providers.base import ProviderKind, TokenUsage
from wmh.providers.pool import PoolEntry
from wmh.retrieval.embedders import HashingEmbedder


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


def test_report_requires_baseline_in_matrix() -> None:
    with pytest.raises(KeyError, match="nope"):
        build_report(
            _matrix(),
            _cluster_policy(),
            baseline="nope",
            endpoint="e",
            generated_at="2026-07-24T00:00:00Z",
        )

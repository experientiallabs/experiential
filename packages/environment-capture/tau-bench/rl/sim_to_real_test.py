"""Tests for the sim-to-real rank-agreement report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from real_episodes import RealEpisodeRow
from sim_to_real import (
    RankAgreement,
    has_inline_call,
    main,
    paired_scores,
    report,
    scenario_clustered_stats,
)

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry

_MODELS = ("alpha", "beta", "gamma", "delta")
_POOL = [
    PoolEntry(
        name=name,
        kind=ProviderKind.ANTHROPIC,
        model=f"claude-{name}",
        input_per_mtok=1.0,
        output_per_mtok=2.0,
    )
    for name in _MODELS
]


def _real_row(model: str, scenario: str, reward: float | None, episode: int = 0) -> RealEpisodeRow:
    return RealEpisodeRow(
        scenario_id=scenario,
        domain=scenario.split(":")[0],
        task_id=scenario.split(":")[1],
        task=json.dumps({"reason_for_call": f"do {scenario}"}),
        provenance=["p"],
        model=model,
        route=f"anthropic/{model}",
        episode=episode,
        reward=reward,
        nl_assertion_reward=False,
        termination_reason="user_stop",
        duration_s=1.0,
        agent_input_tokens=10,
        agent_output_tokens=1,
        user_input_tokens=5,
        user_output_tokens=1,
        cost_usd_pool=0.01,
        cost_usd_tau2_agent=0.01,
        cost_usd_tau2_user=0.001,
        steps=1,
        call_seconds=[0.5],
        replies=[],
        user_sim="gpt-5.4-mini",
    )


def _wm_outcome(
    model: str, scenario: str, reward: float, replies: list[str] | None = None, episode: int = 0
) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=scenario,
        task=json.dumps({"reason_for_call": f"do {scenario}"}),
        model=model,
        episode=episode,
        reward=reward,
        success=reward >= 1.0,
        replies=replies or [],
    )


def _agreeing_corpus(
    flip: bool = False,
) -> tuple[list[RealEpisodeRow], OutcomeMatrix]:
    """Four models with a clean quality ordering, optionally reversed in the world model."""
    scenarios = ["airline:0", "retail:3"]
    scores = {"alpha": 0.2, "beta": 0.4, "gamma": 0.6, "delta": 0.8}
    rows = [
        _real_row(model, scenario, scores[model]) for model in _MODELS for scenario in scenarios
    ]
    outcomes = [
        _wm_outcome(model, scenario, 1.0 - scores[model] if flip else scores[model])
        for model in _MODELS
        for scenario in scenarios
    ]
    return rows, OutcomeMatrix(pool=_POOL, outcomes=outcomes)


def test_scenario_clustered_stats_averages_scenarios_not_episodes() -> None:
    # Two episodes of one scenario are not two independent draws: the scenario with two
    # episodes must not outweigh the one with a single episode.
    mean, se, count = scenario_clustered_stats({"a": [0.0, 0.0], "b": [1.0]})
    assert mean == pytest.approx(0.5)
    assert count == 2
    assert se == pytest.approx(0.5)


def test_rank_agreement_detects_a_matching_ordering() -> None:
    real = {"alpha": 0.2, "beta": 0.4, "gamma": 0.6}
    agreement = RankAgreement(real, {"alpha": 0.1, "beta": 0.5, "gamma": 0.9})
    assert agreement.spearman.statistic == pytest.approx(1.0)
    assert agreement.best_real == "gamma"
    assert agreement.best_wm == "gamma"


def test_rank_agreement_detects_an_inverted_ordering() -> None:
    real = {"alpha": 0.2, "beta": 0.4, "gamma": 0.6}
    agreement = RankAgreement(real, {"alpha": 0.9, "beta": 0.5, "gamma": 0.1})
    assert agreement.spearman.statistic == pytest.approx(-1.0)
    assert agreement.top3_overlap == 3  # the same three models, ranked backwards


def test_rank_agreement_uses_only_shared_models() -> None:
    agreement = RankAgreement({"a": 1.0, "b": 2.0, "ghost": 3.0}, {"a": 1.0, "b": 2.0})
    assert agreement.models == ["a", "b"]


def test_inline_call_detection() -> None:
    assert has_inline_call(_wm_outcome("a", "airline:0", 1.0, ['get_user_details({"id": 1})']))
    assert not has_inline_call(
        _wm_outcome("a", "airline:0", 1.0, ['{"tool": "get_user_details", "arguments": {}}'])
    )


def test_paired_scores_prefer_exact_scenario_ids() -> None:
    rows, matrix = _agreeing_corpus()
    real, wm, shared = paired_scores(rows, matrix.outcomes)
    assert shared == 2
    assert sorted(real) == sorted(_MODELS)


def test_paired_scores_fall_back_to_reason_matching() -> None:
    # An older world-model matrix hashes the task instead of using "<domain>:<task_id>".
    rows, matrix = _agreeing_corpus()
    rehashed = OutcomeMatrix(
        pool=_POOL,
        outcomes=[
            outcome.model_copy(update={"scenario_id": f"hash-{outcome.scenario_id}"})
            for outcome in matrix.outcomes
        ],
    )
    real, wm, shared = paired_scores(rows, rehashed.outcomes)
    assert shared == 2
    assert real["delta"] == pytest.approx(0.8)


def test_report_headline_agreement() -> None:
    rows, matrix = _agreeing_corpus()
    text = "\n".join(report(rows, matrix, glm_clean=False))
    assert "spearman  +1.000" in text
    assert "8 real rows" in text
    assert "paired overlap (2 scenarios" in text


def test_report_flags_disagreement() -> None:
    rows, matrix = _agreeing_corpus(flip=True)
    text = "\n".join(report(rows, matrix, glm_clean=False))
    assert "spearman  -1.000" in text


def test_report_skips_unscored_real_rows() -> None:
    rows, matrix = _agreeing_corpus()
    rows.append(_real_row("alpha", "airline:9", None))
    text = "\n".join(report(rows, matrix, glm_clean=False))
    assert "8 real rows" in text  # the unscored row is not counted, and never scored as 0


def test_glm_clean_arm_scores_broken_scenarios_on_their_clean_episode() -> None:
    scenarios = ["airline:0", "retail:3"]
    real_scores = {"glm-5.2": 0.5, "alpha": 0.2, "beta": 0.8}
    rows = [_real_row(m, s, real_scores[m]) for m in real_scores for s in scenarios]
    pool = [
        PoolEntry(
            name=name,
            kind=ProviderKind.ANTHROPIC,
            model=f"claude-{name}",
            input_per_mtok=1.0,
            output_per_mtok=2.0,
        )
        for name in ("glm-5.2", "alpha", "beta")
    ]
    outcomes = [_wm_outcome(m, s, real_scores[m]) for m in ("alpha", "beta") for s in scenarios]
    for scenario in scenarios:
        outcomes.append(_wm_outcome("glm-5.2", scenario, 0.0, ['get_user({"id": 1})'], episode=0))
        outcomes.append(_wm_outcome("glm-5.2", scenario, 0.9, episode=1))
    text = "\n".join(report(rows, OutcomeMatrix(pool=pool, outcomes=outcomes), glm_clean=True))
    assert "glm-format-corrected" in text
    # headline averages the broken and clean episodes (0.45); corrected keeps only the clean one.
    assert "glm-5.2          0.500 +/- 0.000     2 0.450" in text


def test_main_writes_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows, matrix = _agreeing_corpus()
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("\n".join(row.model_dump_json() for row in rows) + "\n", encoding="utf-8")
    matrix_path = tmp_path / "matrix.json"
    matrix.save(matrix_path)
    assert main(["--real", str(rows_path), "--wm", str(matrix_path)]) == 0
    assert "spearman  +1.000" in capsys.readouterr().out


def test_main_says_what_to_run_when_there_are_no_rows(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _agreeing_corpus()[1].save(matrix_path)
    with pytest.raises(SystemExit):
        main(["--real", str(tmp_path / "missing.jsonl"), "--wm", str(matrix_path)])

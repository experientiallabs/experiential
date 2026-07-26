"""CLI tests for `wmh optimize route` (fit + report), driven via CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from wmh.cli.app import app
from wmh.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmh.optimize.policy import KNN_BANK_FILENAME, POLICY_FILENAME, RoutingPolicy
from wmh.optimize.routing import evaluate_policy
from wmh.providers.base import ProviderKind
from wmh.providers.pool import PoolEntry

runner = CliRunner()


def _matrix_file(tmp_path: Path) -> Path:
    pool = [
        PoolEntry(
            name="a", kind=ProviderKind.OPENAI, model="a", input_per_mtok=1.0, output_per_mtok=1.0
        ),
        PoolEntry(
            name="b", kind=ProviderKind.OPENAI, model="b", input_per_mtok=1.0, output_per_mtok=1.0
        ),
    ]
    outcomes = []
    tasks = {
        "s1": "SELECT count(*) FROM t",
        "s2": "SELECT name FROM users WHERE id = 4",
        "s3": "write a poem about rivers",
        "s4": "draft a thank-you note",
    }
    for sid, task in tasks.items():
        sql = sid in ("s1", "s2")
        for model in ("a", "b"):
            wins = (model == "a") == sql
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=sid,
                    task=task,
                    model=model,
                    reward=1.0 if wins else 0.0,
                    success=wins,
                    cost_usd=0.001,
                )
            )
    path = tmp_path / "matrix.json"
    OutcomeMatrix(pool=pool, outcomes=outcomes).save(path)
    return path


def test_route_fit_and_report(tmp_path: Path) -> None:
    matrix_file = _matrix_file(tmp_path)
    policy_file = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--out",
            str(policy_file),
            "--clusters",
            "2",
            "--top-k-clusters",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.kind == "rank"
    assert len(policy.clusters) == 2

    report_file = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "report",
            str(matrix_file),
            str(policy_file),
            "--baseline",
            "a",
            "--out",
            str(report_file),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(report_file.read_text())
    assert report["baseline"]["model_id"] == "a"
    assert report["headline"]["baseline_accuracy"] == 0.5
    assert report["cost_assumptions"]


def test_route_fit_rejects_unknown_embedder(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["optimize", "route", "fit", str(_matrix_file(tmp_path)), "--embedder", "vibes"],
    )
    assert result.exit_code != 0
    assert "hashing or azure" in result.output


def _knn_matrix_file(tmp_path: Path) -> Path:
    """Twelve scenarios: enough neighbors per query for a guarded fit to route at all."""
    pool = [
        PoolEntry(
            name="a", kind=ProviderKind.OPENAI, model="a", input_per_mtok=1.0, output_per_mtok=1.0
        ),
        PoolEntry(
            name="b", kind=ProviderKind.OPENAI, model="b", input_per_mtok=1.0, output_per_mtok=1.0
        ),
    ]
    sql = [
        "SELECT count(*) FROM orders WHERE total > 100",
        "SELECT name FROM users WHERE id = 4",
        "SELECT avg(price) FROM products GROUP BY category",
        "SELECT id FROM events WHERE kind = 'click'",
        "SELECT max(score) FROM matches WHERE season = 2025",
        "SELECT city FROM stores WHERE stock > 0",
    ]
    prose = [
        "write a friendly email to the team about the offsite",
        "write a warm welcome note for new employees",
        "write a short thank-you message for the organizers",
        "write a cheerful newsletter intro about spring",
        "write a gentle reminder about the expense deadline",
        "write a farewell note for a departing teammate",
    ]
    outcomes = []
    for group, tasks in (("sql", sql), ("prose", prose)):
        for index, task in enumerate(tasks):
            for model in ("a", "b"):
                wins = (model == "a") == (group == "sql")
                outcomes.append(
                    ScenarioOutcome(
                        scenario_id=f"{group}:{index}",
                        task=task,
                        model=model,
                        reward=1.0 if wins else 0.0,
                        success=wins,
                        cost_usd=0.001,
                    )
                )
    path = tmp_path / "knn_matrix.json"
    OutcomeMatrix(pool=pool, outcomes=outcomes).save(path)
    return path


def test_route_fit_knn_writes_policy_and_sidecar(tmp_path: Path) -> None:
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--kind",
            "knn",
            "--fallback",
            "a",
            "--z",
            "0.5",
            "--rag-num",
            "3",
            "--min-pairs",
            "2",
            "--out",
            str(policy_file),
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.kind == "knn"
    assert policy.default_model == "a" == policy.guard_model  # the pinned fallback
    assert (tmp_path / KNN_BANK_FILENAME).is_file()  # sidecar beside the policy
    assert len(policy.knn_bank().scenario_ids) == 12
    assert "routed away from the fallback" in result.output
    # The prose neighborhoods carry unanimous evidence for b, so that traffic leaves the
    # fallback while the SQL half stays on it.
    matrix = OutcomeMatrix.load(matrix_file)
    prose_ids = [sid for sid in matrix.scenario_ids() if sid.startswith("prose:")]
    assert evaluate_policy(policy, matrix, prose_ids).model_mix == {"b": 1.0}


def test_route_fit_knn_rejects_the_rank_only_cost_knob(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path)),
            "--kind",
            "knn",
            "--cost-weight",
            "0.5",
        ],
    )
    assert result.exit_code != 0
    # The message points at the knn cost control that does exist, not just at what is wrong.
    assert "--cost-quality" in result.output


def test_route_fit_rejects_unknown_kind(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["optimize", "route", "fit", str(_matrix_file(tmp_path)), "--kind", "vibes"]
    )
    assert result.exit_code != 0
    assert "knn or rank" in result.output


def _fitted_knn_policy(tmp_path: Path) -> Path:
    policy_file = tmp_path / POLICY_FILENAME
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_knn_matrix_file(tmp_path)),
            "--kind",
            "knn",
            "--fallback",
            "a",
            "--rag-num",
            "3",
            "--min-pairs",
            "2",
            "--out",
            str(policy_file),
        ],
    )
    assert result.exit_code == 0, result.output
    return policy_file


def test_route_tune_sets_the_dial_and_keeps_the_policy_as_fitted(tmp_path: Path) -> None:
    policy_file = _fitted_knn_policy(tmp_path)
    fitted = RoutingPolicy.load(policy_file)
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.6"]
    )
    assert result.exit_code == 0, result.output
    tuned = RoutingPolicy.load(policy_file)
    assert tuned.cost_quality == 0.6
    assert tuned.pick_lam > 0.0
    assert tuned.guard_mode == "asymmetric"
    # The un-tuned artifact is preserved, so the dial is always re-appliable from the fit.
    base = RoutingPolicy.load(tmp_path / "policy.base.json")
    assert base.model_dump() == fitted.model_dump()
    # The printed anchor table is how an operator learns what the position measured.
    assert "cost_quality=0.6" in result.output
    assert "-46.2%" in result.output


def test_route_tune_twice_equals_tuning_once_from_the_base(tmp_path: Path) -> None:
    policy_file = _fitted_knn_policy(tmp_path)
    runner.invoke(app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "1.0"])
    once = RoutingPolicy.load(policy_file).model_dump()
    for _ in range(2):
        result = runner.invoke(
            app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "1.0"]
        )
        assert result.exit_code == 0, result.output
    assert RoutingPolicy.load(policy_file).model_dump() == once
    # Sliding back down lands exactly where a first-time slide to that position would.
    runner.invoke(app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.25"])
    balanced = RoutingPolicy.load(policy_file)
    assert balanced.cost_quality == 0.25
    assert (balanced.pick_lam, balanced.guard_mode) == (0.0, "symmetric")


def test_route_tune_still_routes_after_the_dial_moves(tmp_path: Path) -> None:
    # The dial must leave a servable policy: same bank, same baseline, still routing.
    matrix_file = _knn_matrix_file(tmp_path)
    policy_file = _fitted_knn_policy(tmp_path)
    runner.invoke(app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "1.0"])
    tuned = RoutingPolicy.load(policy_file)
    matrix = OutcomeMatrix.load(matrix_file)
    prose_ids = [sid for sid in matrix.scenario_ids() if sid.startswith("prose:")]
    assert evaluate_policy(tuned, matrix, prose_ids).model_mix == {"b": 1.0}


def test_route_tune_rejects_a_policy_kind_without_a_dial(tmp_path: Path) -> None:
    policy_file = tmp_path / POLICY_FILENAME
    fit = runner.invoke(
        app,
        ["optimize", "route", "fit", str(_matrix_file(tmp_path)), "--out", str(policy_file)],
    )
    assert fit.exit_code == 0, fit.output
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "0.5"]
    )
    assert result.exit_code != 0
    assert "kind='rank'" in result.output


def test_route_tune_rejects_a_missing_policy_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(tmp_path / "nope.json"), "--cost-quality", "0.5"]
    )
    assert result.exit_code != 0
    assert "no policy file" in result.output


def test_route_tune_rejects_a_dial_outside_the_range(tmp_path: Path) -> None:
    policy_file = _fitted_knn_policy(tmp_path)
    result = runner.invoke(
        app, ["optimize", "route", "tune", str(policy_file), "--cost-quality", "2"]
    )
    assert result.exit_code != 0

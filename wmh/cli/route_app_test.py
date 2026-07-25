"""CLI tests for `wmh optimize route` (fit + report), driven via CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from wmh.cli.app import app
from wmh.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmh.optimize.policy import RoutingPolicy
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

"""CLI tests for `wmo optimize route` (fit + report), driven via CliRunner."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from wmo.cli.app import app
from wmo.distill.store import DistillModelCard
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import (
    KNN_BANK_FILENAME,
    POLICY_FILENAME,
    RoutingPolicy,
    select_model,
)
from wmo.optimize.routing import evaluate_policy
from wmo.providers import pool as pool_module
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry, load_pool

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


def _run_dir(tmp_path: Path, sampler: str = "tinker://fake/sampler/final/0") -> Path:
    """A distillation run dir with just the artifact `route student` reads."""
    run_dir = tmp_path / "distill" / "support"
    run_dir.mkdir(parents=True, exist_ok=True)
    card = DistillModelCard(
        base_model="Qwen/Qwen3-8B",
        lora_rank=32,
        teacher_model="glm-5.2",
        sampler_path=sampler,
        steps_completed=200,
    )
    (run_dir / "model_card.json").write_text(card.model_dump_json(indent=2), encoding="utf-8")
    return run_dir


def _built_model(tmp_path: Path, name: str = "support") -> Path:
    """A world model dir as `WorldModelStore` recognizes one (a dir carrying config.toml)."""
    model_dir = tmp_path / "models" / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("", encoding="utf-8")
    return model_dir


def _add_student(tmp_path: Path, pool_file: Path, *, name: str = "student") -> Result:
    return runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--name",
            name,
            "--pool",
            str(pool_file),
        ],
    )


def test_route_student_makes_a_trained_adapter_routable(tmp_path: Path) -> None:
    """The keystone: a run dir becomes a loadable pool candidate with no hand-edited TOML."""
    pool_file = tmp_path / "pool.toml"

    result = _add_student(tmp_path, pool_file)

    assert result.exit_code == 0, result.output
    entry = load_pool(pool_file).entry("student")
    assert entry.kind is ProviderKind.OPENAI
    assert entry.model == "tinker://fake/sampler/final/0"
    assert entry.model_type == "Qwen/Qwen3-8B"
    assert entry.chat_max_tokens_field == "max_tokens"
    assert entry.api_key_env == "TINKER_API_KEY"
    assert entry.price().input_per_mtok == 0.1


def test_route_student_requires_prices(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["optimize", "route", "student", str(_run_dir(tmp_path)), "--pool", str(tmp_path / "p")],
    )
    assert result.exit_code != 0
    assert "--input-per-mtok" in result.output


def test_route_student_names_the_missing_model_card(tmp_path: Path) -> None:
    empty = tmp_path / "not-a-run"
    empty.mkdir()
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(empty),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
        ],
    )
    assert result.exit_code != 0
    assert "model_card.json" in result.output
    assert "adapter version directory" in result.output  # says what to pass instead


def test_route_student_rejects_a_run_with_no_trained_weights(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, sampler="Qwen/Qwen3-8B")
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(run_dir),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
        ],
    )
    assert result.exit_code != 0
    assert "tinker://" in result.output


def test_route_student_declining_the_replacement_leaves_the_pool_alone(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    before = pool_file.read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "9.9",
            "--output-per-mtok",
            "9.9",
            "--pool",
            str(pool_file),
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert pool_file.read_text(encoding="utf-8") == before  # the 9.9 price never landed


def test_route_student_replaces_the_same_name_under_yes(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.2",
            "--output-per-mtok",
            "0.8",
            "--pool",
            str(pool_file),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "replaced" in result.output
    models = load_pool(pool_file).models
    assert len(models) == 1  # replaced, not duplicated
    assert models[0].price().output_per_mtok == 0.8


def test_route_pin_writes_a_serveable_static_policy(tmp_path: Path) -> None:
    """One step from pool candidate to endpoint: the policy lands where `wmo serve` reads it."""
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    model_dir = _built_model(tmp_path)

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(model_dir / POLICY_FILENAME)
    assert policy.kind == "static"
    assert policy.default_model == "student"
    assert [entry.name for entry in policy.pool] == ["student"]
    assert policy.fitted_from is not None
    assert "no outcome matrix" in policy.fitted_from  # provenance says it measured nothing


def test_route_pin_serves_through_the_endpoint_it_installed(tmp_path: Path) -> None:
    """The pinned policy is not just well formed: `select_model` actually routes on it."""
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    model_dir = _built_model(tmp_path)
    runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    policy = RoutingPolicy.load(model_dir / POLICY_FILENAME)
    decision = select_model(policy, "anything at all")

    assert decision.model == "student"


def test_route_pin_rejects_a_model_outside_the_pool(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    _built_model(tmp_path)

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "ghost",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "no pool model named 'ghost'" in result.output
    assert "student" in result.output  # lists what IS available


def test_route_pin_names_the_missing_world_model(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "nope",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "no world model named 'nope'" in result.output


def test_route_pin_declining_keeps_a_fitted_policy(tmp_path: Path) -> None:
    """Pinning over a fitted knn policy would orphan its evidence bank, so it must ask first."""
    pool_file = tmp_path / "pool.toml"
    assert _add_student(tmp_path, pool_file).exit_code == 0
    model_dir = _built_model(tmp_path)
    installed = model_dir / POLICY_FILENAME
    fitted = _fitted_knn_policy(tmp_path)
    installed.write_text(fitted.read_text(encoding="utf-8"), encoding="utf-8")
    before = installed.read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "pin",
            "support",
            "--model",
            "student",
            "--pool",
            str(pool_file),
            "--root",
            str(tmp_path),
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert installed.read_text(encoding="utf-8") == before


def test_route_student_rejects_an_empty_endpoint(tmp_path: Path) -> None:
    """`--endpoint "$UNSET_VAR"` must not silently fall back to a different host."""
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--endpoint",
            "",
            "--pool",
            str(tmp_path / "pool.toml"),
        ],
    )

    assert result.exit_code != 0
    assert "--endpoint is empty" in result.output
    assert not (tmp_path / "pool.toml").exists()  # nothing was written


def test_route_student_summary_does_not_claim_a_key_it_will_not_send(tmp_path: Path) -> None:
    """A custom endpoint authenticates via WMO_ENDPOINT_API_KEY, and the summary must say so."""
    pool_file = tmp_path / "pool.toml"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--endpoint",
            "https://my-vllm.example/v1",
            "--pool",
            str(pool_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "WMO_ENDPOINT_API_KEY" in result.output
    assert "TINKER_API_KEY" not in result.output
    assert load_pool(pool_file).entry("student").api_key_env is None


def test_route_student_reports_a_busy_pool_without_claiming_it_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held roster lock is a retryable busy state, so it must exit non-zero and say to retry.

    The lock is real here (taken with flock, from this process, on the file the command writes);
    only the wait is shortened, so the CLI runs the same path an operator hits when a second
    registration is in flight. What matters is that it never prints its "added pool candidate"
    line for a write that did not happen, and never reports a lock holder as a bad flag.
    """
    monkeypatch.setattr(pool_module, "POOL_LOCK_TIMEOUT_S", 0.05)
    pool_file = tmp_path / "pool.toml"
    lock_path = pool_file.with_name(f"{pool_file.name}.lock")
    pool_file.parent.mkdir(parents=True, exist_ok=True)
    holder = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        result = runner.invoke(
            app,
            [
                "optimize",
                "route",
                "student",
                str(_run_dir(tmp_path)),
                "--input-per-mtok",
                "0.1",
                "--output-per-mtok",
                "0.4",
                "--pool",
                str(pool_file),
            ],
        )
    finally:
        os.close(holder)

    assert result.exit_code == 1, result.output
    assert "pool busy" in result.output
    assert "retry" in result.output
    assert "added pool candidate" not in result.output
    assert not pool_file.exists()  # nothing was written


def test_route_student_rejects_an_unknown_output_budget_field(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "student",
            str(_run_dir(tmp_path)),
            "--input-per-mtok",
            "0.1",
            "--output-per-mtok",
            "0.4",
            "--chat-max-tokens-field",
            "max_output_tokens",
            "--pool",
            str(tmp_path / "pool.toml"),
        ],
    )

    assert result.exit_code != 0
    assert "max_tokens or max_completion_tokens" in result.output

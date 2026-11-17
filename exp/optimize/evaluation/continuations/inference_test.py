"""Lineage-level uncertainty and baselines stay separate from repeats of one checkpoint."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.evaluations.uncertainty import wilson_interval
from exp.optimize.evaluation.continuations.campaign import run_continuation_campaign
from exp.optimize.evaluation.continuations.campaign_test import campaign_fixture
from exp.optimize.evaluation.continuations.experiment import (
    CheckpointExperimentSpec,
    run_checkpoint_experiment,
)
from exp.optimize.evaluation.continuations.inference import (
    BaseRate,
    CellOutcomes,
    campaign_inference,
    cell_inference,
    experiment_inference,
    experiment_summary,
)
from exp.optimize.evaluation.continuations.selection import persist_selection_plan
from exp.optimize.evaluation.continuations.selection_test import selection_fixture

BASE = BaseRate(rate=0.5, successes=8, assessable=16, unknown=0, source="initial attempts")


def cells() -> tuple[CellOutcomes, ...]:
    """Two lineages, one matched pair each; one random cell lost two outcomes."""
    return (
        CellOutcomes(
            arm="proposed",
            pair=0,
            checkpoint="cp-a1",
            lineage="task-a",
            planned=8,
            outcomes=(1, 1, 1, 1, 1, 1, 1, 0),
            forecast=0.9,
        ),
        CellOutcomes(
            arm="random",
            pair=0,
            checkpoint="cp-a2",
            lineage="task-a",
            planned=8,
            outcomes=(1, 0, 0, 0, 1, 0),
            actor_budget_exhausted=1,
            forecast=0.2,
        ),
        CellOutcomes(
            arm="proposed",
            pair=1,
            checkpoint="cp-b1",
            lineage="task-b",
            planned=8,
            outcomes=(0, 0, 0, 0, 0, 0, 0, 0),
            forecast=0.8,
        ),
        CellOutcomes(
            arm="random",
            pair=1,
            checkpoint="cp-b2",
            lineage="task-b",
            planned=8,
            outcomes=(0, 0, 0, 0, 0, 0, 0, 0),
        ),
    )


def field(value: JsonValue, *path: str | int) -> JsonValue:
    """Walk a JSON report."""
    for key in path:
        if isinstance(key, int):
            assert isinstance(value, list)
            value = value[key]
        else:
            assert isinstance(value, dict)
            value = value[key]
    return value


def test_repeats_estimate_one_checkpoint_and_lineages_estimate_tasks() -> None:
    """Eight continuations of a checkpoint are not eight tasks; lineages are the unit."""
    report = cell_inference(cells(), base_rate=BASE)
    assert report["independent_lineages"] == 2
    checkpoint = field(report, "arms", "proposed", "checkpoints", 0, "repeated_attempts")
    assert isinstance(checkpoint, dict)
    assert checkpoint["trials"] == 8 and checkpoint["independent_units"] == 8
    across = field(report, "arms", "proposed", "across_lineages")
    assert isinstance(across, dict)
    assert across["independent_units"] == 2
    assert across["estimate"] == pytest.approx((7 / 8 + 0) / 2)
    assert across["wilson_95"] == list(wilson_interval((7 / 8) / 2, 2))
    random = field(report, "arms", "random")
    assert isinstance(random, dict)
    assert random["missing_outcome_bounds"] == [2 / 16, 4 / 16]
    assert field(random, "checkpoints", 0, "unknown") == 2
    assert field(random, "checkpoints", 0, "actor_budget_exhausted") == 1
    assert (random["actor_budget_exhausted"], random["actor_budget_exhausted_share"]) == (1, 1 / 16)
    assert field(report, "arms", "proposed", "actor_budget_exhausted") == 0


def test_critic_brier_is_outcome_level_and_compared_with_a_disjoint_base_rate() -> None:
    """Brier stays mean((q - y)^2) over realized outcomes, beside the base rate on the same ones."""
    report = cell_inference(cells(), base_rate=BASE)
    critic = field(report, "arms", "proposed", "critic")
    assert isinstance(critic, dict) and critic["forecast_checkpoints"] == 2
    first = (7 * (0.9 - 1) ** 2 + (0.9 - 0) ** 2) / 8
    second = (0.8 - 0) ** 2
    assert critic["checkpoint_brier"] == pytest.approx([first, second])
    comparison = critic["base_rate_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["mean_checkpoint_brier"] == pytest.approx(0.25)
    assert comparison["brier_skill"] == pytest.approx(1 - ((first + second) / 2) / 0.25)
    assert comparison["critic_better_checkpoints"] == 1
    unregistered = cell_inference(cells(), base_rate=None)
    status = field(unregistered, "arms", "proposed", "critic", "base_rate_comparison", "status")
    assert "no disjoint development base rate" in str(status)
    missing = field(report, "arms", "random", "critic")
    assert isinstance(missing, dict) and missing["forecast_checkpoints"] == 1


def test_selector_effects_are_paired_against_random_controls_per_lineage() -> None:
    """The matched random control is the selector's explicit baseline."""
    comparison = cell_inference(cells(), base_rate=BASE)["random_control_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["lineage_effects"] == pytest.approx({"task-a": 7 / 8 - 2 / 6, "task-b": 0.0})
    assert comparison["mean_lineage_effect"] == pytest.approx((7 / 8 - 2 / 6) / 2)
    one_lineage = cell_inference(cells()[:2], base_rate=BASE)
    assert one_lineage["independent_lineages"] == 1
    assert "says nothing reliable about unseen tasks" in str(one_lineage["interpretation"])
    lost = cell_inference(
        (cells()[0], cells()[1].model_copy(update={"outcomes": (), "failed": True})),
        base_rate=None,
    )
    assert field(lost, "random_control_comparison", "pairs_with_unknown_outcomes") == 1


def test_saved_experiment_inference_matches_its_saved_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inference reads saved studies without changing the experiment's own measurements."""
    project, selection = selection_fixture(tmp_path, monkeypatch)
    reference = persist_selection_plan(
        project, selection, created_at=datetime.now(UTC), code_revision="test"
    )
    report = run_checkpoint_experiment(
        project, CheckpointExperimentSpec(selection=reference), code_revision="test"
    )
    inference = experiment_inference(project, report)
    assert inference["independent_lineages"] == 1
    for arm in ("proposed", "random"):
        metrics = report.metrics[arm]
        assert isinstance(metrics, dict)
        bounds = field(inference, "arms", arm, "missing_outcome_bounds")
        assert bounds == metrics["success_bounds"]
    summary = experiment_summary(project, report)
    assert summary["unmatched"] == report.unmatched and summary["inference"] == inference
    assert summary["proposed"] == report.metrics["proposed"], "saved metrics stay unchanged"


def test_campaign_inference_uses_source_attempts_as_the_base_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The source attempts are disjoint from continuations; one source task is one lineage."""
    project, spec, *_ = campaign_fixture(tmp_path, monkeypatch)
    report = run_continuation_campaign(project, spec, code_revision="test")
    inference = campaign_inference(project, report)
    assert inference["independent_lineages"] == 1
    source = inference["source_attempts"]
    initial = report.metrics["initial_attempts"]
    assert isinstance(source, dict) and isinstance(initial, dict)
    assert source["planned"] == initial["planned"] == 3
    assert source["unknown"] == initial["unassessable"]
    experiments = inference["experiments"]
    assert isinstance(experiments, list) and len(experiments) == 1
    assert "comparison arms, not replications" in str(inference["interpretation"])
    base_rate = inference["base_rate"]
    assert isinstance(base_rate, dict) and "initial source attempts" in str(base_rate["source"])
    assert (base_rate["successes"], base_rate["assessable"]) == (3, 3)
    assert field(source, "repeated_attempts", "independent_units") == 3
    assert field(experiments, 0, "arms", "proposed", "critic", "base_rate_comparison") is not None


def test_campaign_cli_reports_inference_beside_saved_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command prints saved measurements and the derived inference together."""
    _, spec, *_ = campaign_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    path = tmp_path / "campaign.json"
    path.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="run-continuation-campaign", campaign=spec
        ).model_dump_json()
    )
    args = ["eval", "case", "--root", str(tmp_path), "--continuations", str(path)]
    result = CliRunner().invoke(app, [*args, "--yes", "--non-interactive"])
    assert result.exit_code == 0, result.output
    printed: JsonObject = json.loads(result.output[result.output.index("{") :])
    assert printed["completed_experiments"] == 1
    inference = printed["inference"]
    assert isinstance(inference, dict) and inference["independent_lineages"] == 1

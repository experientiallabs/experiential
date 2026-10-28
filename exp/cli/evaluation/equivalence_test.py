"""The public CLI executes and replays paired measurements and trace-council analysis."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import ArtifactInput
from exp.common.project import artifact_input
from exp.optimize.evaluation.continuations.traces import trajectory_studies
from exp.optimize.evaluation.equivalence import rubric as rubric_owner
from exp.optimize.evaluation.equivalence.rubric import RubricReview
from exp.optimize.evaluation.equivalence.rubric_test import derivation_fixture
from exp.optimize.evaluation.equivalence.runner_test import equivalence_fixture
from exp.optimize.evaluation.spending import SpendPolicy


def test_public_cli_compares_environments_and_runs_its_council(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry-run is inert, execution saves complete evidence, and replay needs no provider setup.

    The printed reference names the saved report, the generator export-traces consumes.
    """
    project, spec, actors, reviews = equivalence_fixture(tmp_path, monkeypatch)
    request = tmp_path / "equivalence.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="compare-environments", equivalence=spec
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not actors.requests and not reviews.requests
    assert not project.artifacts.list_ids(artifact_type="actor-equivalence-registration")
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert '"material_matches": 1' in result.output and '"reference_control"' in result.output
    assert '"analyses"' in result.output and "No training or promotion" in " ".join(
        result.output.split()
    )
    assert len(actors.requests) == 6 and len(reviews.requests) == 3
    # The printed reference is the saved report itself, so export-traces can consume it.
    printed = json.loads(result.output[result.output.index("{") : result.output.rindex("}") + 1])
    reference = ArtifactInput.model_validate(printed["reference"])
    assert reference == artifact_input(project.artifacts.read(reference.artifact_id).manifest)
    assert len(trajectory_studies(project.artifacts, reference)) == 3
    spec.cases[0].reference.runtime_file.unlink()
    project.model_catalog_path.unlink()
    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0, replay.output
    assert replay.output[replay.output.index("{") :] == result.output[result.output.index("{") :]
    assert len(actors.requests) == 6 and len(reviews.requests) == 3


def test_public_cli_derives_reviews_freezes_and_replays_rubric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Separate derivation and review; keep dry runs inert and replay without a model."""
    project, spec, proposal = derivation_fixture(tmp_path, monkeypatch)
    complete = Mock(return_value=proposal.model_dump(mode="json"))
    monkeypatch.setattr(rubric_owner, "review_call", complete)
    request = tmp_path / "rubric.json"
    request.write_text(
        ContinuationCommand(
            action="derive-equivalence-rubric",
            rubric_derivation=spec,
            spend=SpendPolicy(unpriced_attempts=spec.maximum_attempts),
        ).model_dump_json()
    )
    command = [
        "eval",
        "case",
        "--root",
        str(tmp_path),
        "--continuations",
        str(request),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not complete.called and not project.artifacts.list_ids(
        artifact_type="actor-rubric-draft"
    )
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert complete.call_count == 1 and '"calibration": "provisional"' in result.output
    draft = project.artifacts.list_ids(artifact_type="actor-rubric-draft")[0]
    request.write_text(
        ContinuationCommand(
            action="freeze-equivalence-rubric",
            rubric_review=RubricReview(
                draft=artifact_input(project.artifacts.read(draft).manifest),
                reviewer="fixture author",
                rationale=(
                    "Recovery criteria and alternatives reviewed; calibration remains pending."
                ),
            ),
        ).model_dump_json()
    )
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert not project.artifacts.list_ids(artifact_type="actor-equivalence-rubric")
    freeze = runner.invoke(app, command)
    assert freeze.exit_code == 0, freeze.output
    assert len(project.artifacts.list_ids(artifact_type="actor-equivalence-rubric")) == 1
    project.model_catalog_path.unlink()
    monkeypatch.delenv("HELD_OUT_TEST_KEY", raising=False)
    replay = runner.invoke(app, command)
    assert replay.exit_code == 0 and replay.output == freeze.output
    assert complete.call_count == 1

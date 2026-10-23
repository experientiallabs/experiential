"""The induce-transitions and validate-transitions continuation actions, end to end offline."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.optimize.evaluation.induction.service import TransitionValidationSpec
from exp.optimize.evaluation.induction.service_test import PROGRAMS, induction_fixture


def test_induction_previews_on_dry_run_asks_consent_then_replays_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry run makes no request; the paid run draws once per tool; a replay draws nothing."""
    project, spec, transport = induction_fixture(tmp_path, monkeypatch)
    operation = tmp_path / "induce.json"
    operation.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="induce-transitions", induction=spec
        ).model_dump_json()
    )
    command = [
        "eval",
        project.paths.project_id,
        "--root",
        str(project.paths.root),
        "--continuations",
        str(operation),
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*command, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "inducible tools" in dry.output and not transport.requests
    refused = runner.invoke(app, command)
    assert refused.exit_code != 0 and not transport.requests
    paid = runner.invoke(app, [*command, "--yes"])
    assert paid.exit_code == 0, paid.output
    assert len(transport.requests) == len(PROGRAMS)
    summary = json.loads(paid.output[paid.output.index("{") :])
    assert summary["induced_tools"] == ["find_user_id_by_email", "get_user_details"]
    replay = runner.invoke(app, [*command, "--yes"])
    assert replay.exit_code == 0, replay.output
    assert len(transport.requests) == len(PROGRAMS)

    validation = tmp_path / "validate.json"
    reference = summary["reference"]
    validation.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND,
            action="validate-transitions",
            transition_validation=TransitionValidationSpec.model_validate({"induction": reference}),
        ).model_dump_json()
    )
    validate = [*command[:5], str(validation), "--non-interactive"]
    checked = runner.invoke(app, [*validate, "--dry-run"])
    assert checked.exit_code == 0 and "nothing replayed" in checked.output
    validated = runner.invoke(app, validate)
    assert validated.exit_code == 0, validated.output
    assert '"unchanged_evidence": true' in validated.output
    assert len(transport.requests) == len(PROGRAMS)

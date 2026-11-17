"""The hill-climb continuation actions dry-run, run, replay and save their full reports."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import sha256_json
from exp.common.models import load_model_catalog
from exp.optimize.evaluation.hillclimb.prompts import PromptEdit
from exp.optimize.evaluation.hillclimb.proposals import EditProposalSpec
from exp.optimize.evaluation.hillclimb.report import ClimbReportSpec
from exp.optimize.evaluation.hillclimb.rounds import ClimbRound
from exp.optimize.evaluation.hillclimb.tests.climb_test import HELPFUL, climb_fixture
from exp.runtime.models.registry import RuntimeModelCatalog


def _run(tmp_path: Path, name: str, command: ContinuationCommand, *flags: str) -> Result:
    """Run one continuation action on the records project."""
    request = tmp_path / f"{name}.json"
    request.write_text(command.model_dump_json())
    return CliRunner().invoke(
        app,
        [
            "eval",
            "records",
            "--root",
            str(tmp_path),
            "--continuations",
            str(request),
            "--non-interactive",
            *flags,
        ],
    )


def _invoke(tmp_path: Path, name: str, command: ContinuationCommand, *flags: str) -> str:
    """Run one continuation action that must succeed and return its output."""
    result = _run(tmp_path, name, command, *flags)
    assert result.exit_code == 0, result.output
    return result.output


def _report_path(output: str) -> Path:
    """The saved report path printed on the ``Report:`` line."""
    line = next(row for row in output.splitlines() if row.startswith("Report: "))
    return Path(line.removeprefix("Report: ").strip())


def test_cli_climbs_one_round_proposes_dry_and_reports_the_final_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry runs make no calls; a kept round and its report replay from saved evidence."""
    project, spec, actor = climb_fixture(tmp_path, monkeypatch, "records")
    deadline = datetime.now(UTC) + timedelta(hours=1)
    edit = PromptEdit(kind="append", text=HELPFUL, change_note="Re-read before editing.")
    climb_round = ContinuationCommand(
        spend=UNPRICED_SPEND,
        action="climb-round",
        climb_round=ClimbRound(climb=spec, round=1, edit=edit, deadline=deadline),
    )
    dry = _invoke(tmp_path, "round", climb_round, "--dry-run")
    assert "2 train and 2 test lineages" in dry and "Validated without saving" in dry
    assert not actor.requests
    run = _invoke(tmp_path, "round", climb_round, "--yes")
    assert "Round 1: kept" in run and "excludes zero" in run
    saved = json.loads(_report_path(run).read_text())
    assert saved["kept"] is True and saved["confirm"]["gain"]["estimate"] == 0.5
    calls = len(actor.requests)
    assert "Round 1: kept" in _invoke(tmp_path, "round", climb_round, "--yes")
    assert len(actor.requests) == calls
    catalog = load_model_catalog(project.model_catalog_path)
    proposals = ContinuationCommand(
        spend=UNPRICED_SPEND,
        action="propose-edits",
        edit_proposals=EditProposalSpec(
            climb=spec,
            after_round=1,
            analyst_alias="judge",
            analyst_model=RuntimeModelCatalog(catalog, environment={}).snapshot("judge")[0],
            catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
            deadline=deadline,
        ),
    )
    refused = _run(tmp_path, "proposals", proposals, "--dry-run")
    assert refused.exit_code == 2 and "no failed train branch" in refused.output
    report = ContinuationCommand(
        spend=UNPRICED_SPEND,
        action="climb-report",
        climb_report=ClimbReportSpec(climb=spec, rounds=1, deadline=deadline),
    )
    assert "Final comparison" in _invoke(tmp_path, "report", report, "--dry-run")
    final = _invoke(tmp_path, "report", report, "--yes")
    assert "Climb after 1 rounds (1 kept): merge" in final
    assert "not an independent holdout estimate" in final
    assert json.loads(_report_path(final).read_text())["verdict"] == "merge"

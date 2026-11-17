"""Analysis continuation handlers run through the CLI here and in the analysis owner tests.

See ``continuations_test.py`` and ``exp/optimize/evaluation/analysis/``.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.optimize.evaluation.analysis import decision_cascade
from exp.optimize.evaluation.analysis.decision_cascade_test import (
    DEVELOPMENT,
    fit_spec,
    systemone,
)
from exp.optimize.evaluation.analysis.decision_points_test import (
    ScriptedProviders,
    main_study,
    project_fixture,
)


def test_cli_fits_a_decision_cascade_after_a_free_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dry run makes no decider call; the paid run saves the fit and its report path."""
    project = project_fixture(tmp_path, monkeypatch)
    studies = tuple(main_study(project, "tickets", lineage) for lineage in DEVELOPMENT)
    transport = ScriptedProviders("tickets")
    monkeypatch.setattr(decision_cascade, "HttpxJsonTransport", lambda: transport)
    request = tmp_path / "fit.json"
    request.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND,
            action="fit-decision-cascade",
            cascade_fit=fit_spec(project, studies, systemone(project)),
        ).model_dump_json()
    )
    command = ["eval", "case", "--root", str(tmp_path), "--continuations", str(request)]
    dry = CliRunner().invoke(app, [*command, "--dry-run", "--non-interactive"])
    assert dry.exit_code == 0, dry.output
    assert "12 reached states over 4 development lineages" in " ".join(dry.output.split())
    assert "no provider calls" in dry.output and not transport.requests
    paid = CliRunner().invoke(app, [*command, "--yes", "--non-interactive"])
    assert paid.exit_code == 0, paid.output
    assert "Status: selected; selected threshold: 0.7" in " ".join(paid.output.split())
    (saved,) = (project.paths.runtime_directory / "decision-cascade").iterdir()
    assert saved.name.startswith("decision-cascade-fit-")
    assert f"Report: {saved / 'report.txt'}" in paid.output
    assert '"status": "selected"' in (saved / "report.txt").read_text()
    assert len(transport.requests) == 12

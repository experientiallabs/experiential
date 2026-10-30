"""Public task-refinement routing and qualification display."""

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.task_refinement import render_task_refinement


@pytest.mark.parametrize(
    "extra",
    [
        ["--models", "actor"],
        ["--concurrency", "2"],
        ["--trace-analysis", "auto"],
        ["--fidelity", "fidelity.json"],
        ["--audit", "1"],
        ["--resume", "saved"],
        ["--report", "saved"],
        ["--repeats", "2"],
        ["--max-steps", "2"],
        ["--max-output-tokens", "20"],
        ["--verbose"],
    ],
)
def test_refinement_rejects_other_modes_before_creating_project(
    tmp_path: Path, extra: list[str]
) -> None:
    """Overrides cannot silently change the frozen refinement specification."""
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "example",
            "--root",
            str(tmp_path),
            "--refine-task",
            "task.json",
            "--dry-run",
            *extra,
        ],
    )
    assert result.exit_code != 0
    assert "cannot combine" in result.output
    assert not list(tmp_path.iterdir())


def test_refinement_requires_named_project(tmp_path: Path) -> None:
    """Refinement never opens an interactive picker or creates an implicit project."""
    result = CliRunner().invoke(
        app, ["eval", "--root", str(tmp_path), "--refine-task", "task.json", "--dry-run"]
    )
    assert result.exit_code != 0
    assert "requires PROJECT" in result.output
    assert not list(tmp_path.iterdir())


def test_refinement_routes_dry_run_without_entering_other_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public mode forwards explicit dry-run intent to its one lifecycle owner."""
    calls: list[tuple[str, Path, Path, bool, bool, bool]] = []

    def run(
        project: str,
        spec_path: Path,
        *,
        root: Path,
        yes: bool,
        non_interactive: bool,
        dry_run: bool,
    ) -> None:
        """Record public arguments without opening a provider."""
        calls.append((project, spec_path, root, yes, non_interactive, dry_run))

    monkeypatch.setattr("exp.cli.evaluation.task_refinement.run_task_refinement", run)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "example",
            "--root",
            str(tmp_path),
            "--refine-task",
            "task.json",
            "--dry-run",
            "--yes",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [("example", Path("task.json"), tmp_path, True, True, True)]
    assert not list(tmp_path.iterdir())


def test_refinement_result_keeps_development_confirmation_and_truth_distinct() -> None:
    """A development gain cannot be rendered as native accuracy or release approval."""
    output = StringIO()
    render_task_refinement(
        Console(file=output, width=160),
        {
            "parent_artifact_id": "parent",
            "child_artifact_id": "child",
            "development_status": "accepted_development",
            "confirmation_status": "rejected",
            "gains": ["correct-positive"],
            "losses": ["protected-negative"],
            "physical_attempts": 3,
        },
    )
    text = output.getvalue()
    assert "Parent artifact: parent" in text
    assert "Child artifact: child" in text
    assert "Development: accepted_development" in text
    assert "Confirmation: rejected" in text
    assert "correct-positive" in text
    assert "protected-negative" in text
    assert "Physical attempts: 3" in text
    assert "does not establish native environment fidelity or RL release approval" in text

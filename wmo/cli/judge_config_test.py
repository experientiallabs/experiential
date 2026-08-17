"""CLI surface tests for explicit manual judge configuration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.cli.judge_config import _label_value
from wmo.cli.judge_rubric import render_setup_contract
from wmo.common.judging import default_task_success_axis
from wmo.common.models import ModelSnapshot
from wmo.optimize.router.judging.service import DEFAULT_JUDGE_TEMPLATE, ManualJudgeSetupPlan


@pytest.mark.parametrize(
    "arguments",
    [
        ["config", "judge", "setup", "support"],
        [
            "config",
            "judge",
            "calibrate",
            "support",
            "--input-usd-per-million",
            "0",
            "--output-usd-per-million",
            "0",
        ],
    ],
)
def test_malformed_release_revision_fails_before_judge_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    """Reject malformed producer identity before judge project or artifact writes.

    Args:
        tmp_path: Pytest-owned state directory.
        monkeypatch: Scoped invalid release revision override.
        arguments: Exact nested judge command invocation.
    """
    root = tmp_path / ".wmo"
    monkeypatch.setenv("WMO_RELEASE_REVISION", "HEAD")

    result = CliRunner().invoke(app, [*arguments, "--root", str(root)])

    assert result.exit_code == 2
    assert "full lowercase 40-hex" in result.output
    assert not root.exists()


def test_judge_commands_render_as_nested_config_commands() -> None:
    """Both manual judge stages are discoverable without expanding the root surface."""
    runner = CliRunner()

    setup = runner.invoke(app, ["config", "judge", "setup", "--help"])
    calibrate = runner.invoke(app, ["config", "judge", "calibrate", "--help"])

    assert setup.exit_code == 0, setup.output
    assert calibrate.exit_code == 0, calibrate.output
    setup_output = unstyle(setup.output)
    calibrate_output = unstyle(calibrate.output)
    assert "--approve" in setup_output
    assert "zero-to-five" not in setup_output
    assert "rubric axes" in setup_output
    assert "--yes" in calibrate_output
    assert "--approve" in calibrate_output


def test_setup_contract_renders_rubric_table_instead_of_schema_or_traces() -> None:
    """Setup shows alias, model, prompt, and a human-readable axis table."""
    plan = SimpleNamespace(
        judge_alias="judge-main",
        judge_model=ModelSnapshot(
            provider="fake",
            model_id="judge-model",
            capabilities_sha256="a" * 64,
            connection_sha256="a" * 64,
        ),
        prompt_template=DEFAULT_JUDGE_TEMPLATE,
        dimensions=(default_task_success_axis(),),
    )

    rendered = render_setup_contract(cast(ManualJudgeSetupPlan, plan), width=120)

    assert "Judge alias: judge-main" in rendered
    assert "1. task-success  Task success" in rendered
    assert "Range: 0-1" in rendered
    assert (
        "The agent successfully completed the task requested in the original user prompt"
        in rendered
    )
    assert "RESPONSE_SCHEMA" not in rendered
    assert "Variable mapping" not in rendered
    assert "Real trace preview" not in rendered
    assert "Structured response schema" not in rendered


def test_scalar_label_values_follow_the_axis_range() -> None:
    """CLI label parsing accepts 0-1 default scores and rejects out-of-range values."""
    axis = default_task_success_axis()

    assert _label_value("trace-1:task-success=1", pairwise=False, axis=axis) == 1
    with pytest.raises(ValueError, match="from 0 through 1"):
        _label_value("trace-1:task-success=4", pairwise=False, axis=axis)

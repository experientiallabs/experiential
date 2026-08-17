"""CLI surface tests for explicit manual judge configuration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click import unstyle
from typer.testing import CliRunner

from wmo.cli import judge_config as judge_config_module
from wmo.cli.app import app
from wmo.optimize.router.judging.contracts import JudgeCalibrationBudget, ManualJudgeLabel
from wmo.runtime.models.providers.errors import ProviderError


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
    assert "--yes" in calibrate_output
    assert "--approve" in calibrate_output
    assert "--debug" in calibrate_output


def test_calibrate_renders_provider_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A typed provider rejection exits with saved-label progress and no traceback.

    Args:
        monkeypatch: Scoped replacements for the calibration command's project work.
        tmp_path: Isolated local project root passed through the retry command.
    """
    secret = "sk-secret-live-key-1234567890"
    labels = (ManualJudgeLabel(trace_id="trace-1", dimension_id="task-success", score=4),)
    budget = JudgeCalibrationBudget(
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4096,
        maximum_attempts_per_call=3,
        call_count=1,
        estimated_cost_usd=0.1,
        maximum_cost_usd=5.0,
    )
    error = ProviderError(
        f"Unsupported parameter: 'temperature' is not supported. Authorization: Bearer {secret}",
        provider="openai",
        endpoint_class="responses",
        status_code=400,
        error_code="unsupported_parameter",
        error_type="invalid_request_error",
        rejected_parameter="temperature",
        request_id="req_safe_1",
    )

    monkeypatch.setattr(judge_config_module, "installed_release_revision", lambda: "a" * 40)
    monkeypatch.setattr(
        judge_config_module,
        "ProjectStore",
        lambda root, project: SimpleNamespace(model_catalog_path=tmp_path / "models.toml"),
    )
    monkeypatch.setattr(
        judge_config_module,
        "prepare_manual_judge_calibration",
        lambda store, sample_size: SimpleNamespace(setup=object(), previews=()),
    )
    monkeypatch.setattr(judge_config_module, "_load_setup_rubric", lambda store, setup: object())
    monkeypatch.setattr(judge_config_module, "_render_calibration_previews", lambda previews: None)
    monkeypatch.setattr(judge_config_module, "calibration_sample", lambda plan: ())
    monkeypatch.setattr(
        judge_config_module, "calibration_sample_digest", lambda setup, sample: "b" * 64
    )
    monkeypatch.setattr(judge_config_module, "read_label_draft", lambda *args: labels)
    monkeypatch.setattr(
        judge_config_module, "manual_judge_calibration_is_complete", lambda store: False
    )
    monkeypatch.setattr(
        judge_config_module,
        "_collect_labels",
        lambda *args, **kwargs: labels,
    )
    monkeypatch.setattr(
        judge_config_module,
        "estimate_manual_judge_budget",
        lambda *args, **kwargs: budget,
    )
    monkeypatch.setattr(judge_config_module, "require_spend_consent", lambda *args, **kwargs: True)
    monkeypatch.setattr(judge_config_module, "load_model_catalog", lambda path: object())
    monkeypatch.setattr(judge_config_module, "RuntimeModelCatalog", lambda catalog: object())

    def fail_calibrate(*args: object, **kwargs: object) -> None:
        """Raise the typed provider rejection after labels are already durable."""
        raise error

    monkeypatch.setattr(judge_config_module, "calibrate_manual_judge", fail_calibrate)

    root = tmp_path / ".wmo"
    result = CliRunner().invoke(
        app,
        [
            "config",
            "judge",
            "calibrate",
            "support",
            "--root",
            str(root),
            "--input-usd-per-million",
            "1.0",
            "--output-usd-per-million",
            "2.0",
            "--yes",
        ],
    )
    printed = unstyle(result.output)

    assert result.exit_code == 1
    assert "Provider call failed" in printed
    assert "openai responses HTTP 400" in printed
    assert "unsupported_parameter" in printed
    assert "rejected parameter: temperature" in printed
    assert "1 human labels saved" in printed
    assert "the failed provider attempt was not recorded" in printed
    assert f"wmo config judge calibrate support --root {root}" in printed
    assert "Traceback" not in printed
    assert secret not in printed
    assert result.exception is None or result.exception.__class__.__name__ == "Exit"

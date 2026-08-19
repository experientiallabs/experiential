"""CLI surface tests for explicit manual judge configuration."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from click import unstyle
from rich.console import Console
from typer.testing import CliRunner

from wmo.cli import judge_config as judge_config_module
from wmo.cli.app import app
from wmo.common.models import (
    BillingSource,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelSnapshot,
    write_model_catalog,
)
from wmo.optimize.router.judging.service import (
    ManualJudgeSetupPlan,
    calibrate_manual_judge,
    default_judge_dimensions,
    estimate_manual_judge_budget,
    prepare_manual_judge_calibration,
)
from wmo.optimize.router.judging.service_test import (
    _built_store,
    _catalog,
    _JudgeClient,
    _labels,
    _RuntimeCatalog,
    _setup,
)
from wmo.runtime.models import ResolvedModel
from wmo.runtime.models.registry import RuntimeModelCatalog


@pytest.mark.parametrize(
    "arguments",
    [
        ["config", "judge", "setup", "support"],
        ["config", "judge", "calibrate", "support"],
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
    """Setup and judge-first review controls remain discoverable under config."""
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
    assert "Advanced" in calibrate_output
    assert "--input-usd-per-million" in calibrate_output
    assert "--output-usd-per-million" in calibrate_output
    assert "--maximum-cost-usd" in calibrate_output
    assert "command-budget" in calibrate_output
    assert "--judgment" in calibrate_output
    assert "--sample-size" in calibrate_output
    assert "[default: 5]" in calibrate_output
    assert "five." in calibrate_output


def _write_catalog(root: Path, catalog: ModelCatalog) -> None:
    """Persist one secret-free catalog at the project root."""
    write_model_catalog(root / "models.toml", catalog)


def _priced_catalog() -> ModelCatalog:
    """Return the fixture catalog with explicit persisted judge prices."""
    catalog = _catalog()
    record = catalog.models["judge-main"]
    return catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "judge-main": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection=record.connection,
                    model=record.model,
                    capabilities=ModelCapabilities(
                        input_cost_per_million_tokens_usd=1.0,
                        output_cost_per_million_tokens_usd=2.0,
                    ),
                ),
            }
        }
    )


def test_interactive_judge_setup_accepts_enter_as_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank response accepts the displayed judge setup and persists it."""
    store = _built_store(tmp_path)
    _write_catalog(store.paths.root, _catalog())
    monkeypatch.setenv("WMO_RELEASE_REVISION", "a" * 40)
    monkeypatch.setattr(judge_config_module, "can_prompt", lambda _console: True)
    monkeypatch.setattr(
        judge_config_module,
        "maybe_edit_setup_plan",
        lambda plan, *, console: plan,
    )

    result = CliRunner().invoke(
        app,
        ["config", "judge", "setup", "support", "--root", str(store.paths.root)],
        input="\n",
    )

    assert result.exit_code == 0, result.output
    assert "Save this judge setup and finalize its rubric? [y/n] (y):" in result.output
    assert "Saved judge setup" in result.output
    review = store.read_review()
    assert isinstance(review, dict)
    assert "manual_judge" in review


def test_interactive_judge_setup_preserves_explicit_n_decline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit ``n`` still declines the displayed judge setup."""
    store = _built_store(tmp_path)
    _write_catalog(store.paths.root, _catalog())
    monkeypatch.setenv("WMO_RELEASE_REVISION", "a" * 40)
    monkeypatch.setattr(judge_config_module, "can_prompt", lambda _console: True)
    monkeypatch.setattr(
        judge_config_module,
        "maybe_edit_setup_plan",
        lambda plan, *, console: plan,
    )

    result = CliRunner().invoke(
        app,
        ["config", "judge", "setup", "support", "--root", str(store.paths.root)],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert "Save this judge setup and finalize its rubric? [y/n] (y):" in result.output
    assert "Judge setup was not saved." in result.output
    review = store.read_review()
    assert isinstance(review, dict)
    assert "manual_judge" not in review


def test_interactive_completed_calibration_accepts_enter_as_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank response approves completed calibration evidence without new calls."""
    store = _built_store(tmp_path)
    _setup(store)
    _write_catalog(store.paths.root, _catalog())
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    labels = _labels(store)
    client = _JudgeClient(plan.setup.judge_model)
    runtime = _RuntimeCatalog(
        ResolvedModel(
            alias="judge-main",
            snapshot=plan.setup.judge_model,
            capabilities=ModelCapabilities(),
            client=client,
            embedding_client=None,
        )
    )
    budget = estimate_manual_judge_budget(
        plan,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
        maximum_input_tokens_per_call=4_096,
        maximum_cost_usd=1.0,
    )
    reviewed = calibrate_manual_judge(
        store,
        cast(RuntimeModelCatalog, runtime),
        plan,
        labels,
        budget,
        spend_consented=True,
        approve=False,
        accept_insufficient_labels=True,
        created_at=datetime.now(UTC),
        code_revision="test-revision",
    )
    assert reviewed.approved_calibration is None
    assert len(client.requests) == 3

    monkeypatch.setenv("WMO_RELEASE_REVISION", "a" * 40)
    monkeypatch.setattr(judge_config_module, "can_prompt", lambda _console: True)
    monkeypatch.setattr(
        judge_config_module,
        "RuntimeModelCatalog",
        lambda _catalog: runtime,
    )
    result = CliRunner().invoke(
        app,
        [
            "config",
            "judge",
            "calibrate",
            "support",
            "--root",
            str(store.paths.root),
            "--sample-size",
            "3",
            "--accept-insufficient-labels",
        ],
        input="\n",
    )

    assert result.exit_code == 0, result.output
    assert "Approve this immutable judge calibration? [y/n] (y):" in result.output
    assert "Approved judge calibration" in result.output
    assert len(client.requests) == 3


def test_calibrate_prints_catalog_pricing_breakdown_before_labels(tmp_path: Path) -> None:
    """Known catalog prices produce a spend preflight before consent or labels."""
    store = _built_store(tmp_path)
    _setup(store)
    _write_catalog(store.paths.root, _priced_catalog())
    (store.paths.root / "settings.toml").write_text(
        "[commands]\nmaximum_cost_usd = 1.0\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "config",
            "judge",
            "calibrate",
            "support",
            "--root",
            str(store.paths.root),
            "--sample-size",
            "3",
            "--non-interactive",
        ],
    )

    output = " ".join(unstyle(result.output).replace("│", " ").split())
    assert result.exit_code == 2
    assert "Cost preflight wmo config judge calibrate support" in output
    assert "estimated cost $0.589824 (conservative maximum)" in output
    assert "of the $1.00 per-command budget" in output
    assert "judge judge-main: openai/judge-model" in output
    assert "pricing source: configured" in output
    assert "at most 3 remaining judge calls with up to 3 attempts each" in output
    assert "32768 input and 16384 output tokens per attempt" in output
    assert "$1.000000 input and $2.000000 output per million tokens" in output
    assert "re-run with --yes" in output
    assert "missing labels" not in output


def test_calibrate_fails_closed_when_catalog_pricing_is_missing(tmp_path: Path) -> None:
    """Missing catalog and known-model prices fail before labels or credentials."""
    store = _built_store(tmp_path)
    _setup(store)
    _write_catalog(store.paths.root, _catalog())

    result = CliRunner().invoke(
        app,
        [
            "config",
            "judge",
            "calibrate",
            "support",
            "--root",
            str(store.paths.root),
            "--non-interactive",
        ],
    )

    output = " ".join(unstyle(result.output).replace("│", " ").split())
    assert result.exit_code == 2
    assert "no trustworthy input/output" in output
    assert "missing labels" not in output


def test_calibrate_uses_shared_command_budget_when_flag_is_omitted(tmp_path: Path) -> None:
    """The shared command-budget setting becomes the calibration ceiling."""
    store = _built_store(tmp_path)
    _setup(store)
    root = store.paths.root
    _write_catalog(root, _priced_catalog())
    (root / "settings.toml").write_text(
        "[commands]\nmaximum_cost_usd = 0.000001\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "config",
            "judge",
            "calibrate",
            "support",
            "--root",
            str(root),
            "--sample-size",
            "3",
            "--non-interactive",
        ],
    )

    output = " ".join(unstyle(result.output).replace("│", " ").split())
    assert result.exit_code == 2
    assert "exceeds the configured per-command" in output
    assert "wmo config budget 0.589824" in output
    assert "--yes cannot override" in output
    assert "missing labels" not in output


def test_setup_output_is_plain_language_and_hides_execution_internals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default setup shows identity, full anchors, and tasks without prompt machinery."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        judge_config_module,
        "_console",
        Console(file=buffer, width=42, color_system=None),
    )
    plan = cast(
        ManualJudgeSetupPlan,
        SimpleNamespace(
            judge_alias="review-judge",
            judge_model=_model(),
            prompt_template=SimpleNamespace(response_shape="scalar"),
            dimensions=default_judge_dimensions(),
            previews=(
                SimpleNamespace(task="Summarize the customer issue.", outcome="success"),
                SimpleNamespace(task="Find the failed command.", outcome="failure"),
            ),
        ),
    )

    judge_config_module._render_setup(plan)

    output = buffer.getvalue()
    compact = " ".join(output.split())
    assert "Judge name: review-judge" in output
    assert "Exact model: openai/judge-model" in output
    assert "Integer scoring from 0 to 1" in compact
    assert "Task success" in output
    assert "Range: 0-1" in output
    assert "did not complete the requested task" in compact
    assert "Summarize the customer issue." in output
    assert "Find the failed command." in output
    assert "Prompt:" not in output
    assert "Variable mapping" not in output
    assert "response schema" not in output.lower()
    assert "Score projection" not in output
    assert "0000000000000000" not in output


def _model() -> ModelSnapshot:
    """Return one exact secret-free judge identity."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="judge-model",
        revision=None,
        capabilities_sha256="0" * 64,
        connection_sha256="1" * 64,
    )

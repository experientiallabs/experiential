"""CLI surface tests for explicit manual judge configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.models import (
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    write_model_catalog,
)
from wmo.optimize.router.judging.service_test import _built_store, _catalog, _setup


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
    assert "Advanced" in calibrate_output
    assert "--input-usd-per-million" in calibrate_output
    assert "--output-usd-per-million" in calibrate_output
    assert "--maximum-cost-usd" in calibrate_output
    assert "command-budget" in calibrate_output


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


def test_calibrate_prints_catalog_pricing_breakdown_before_labels(tmp_path: Path) -> None:
    """Known catalog prices produce a consent breakdown before human labels."""
    store = _built_store(tmp_path)
    _setup(store)
    _write_catalog(store.paths.root, _priced_catalog())

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

    output = unstyle(result.output)
    assert result.exit_code == 2
    assert "Judge: judge-main (openai/judge-model)" in output
    assert "Pricing: configured" in output
    assert "Calls: 3" in output
    assert "Tokens: up to 32768 input and 4096 output per attempt" in output
    assert "Estimated maximum:" in output
    assert "Budget: $10.0000" in output
    assert "missing labels" in output


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

    output = unstyle(result.output)
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

    output = unstyle(result.output)
    assert result.exit_code == 2
    assert "exceeds --maximum-cost-usd" in output
    assert "missing labels" not in output

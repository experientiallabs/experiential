"""Provider setup CLI tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.models import load_model_catalog


def test_noninteractive_provider_setup_writes_exact_roles(tmp_path: Path) -> None:
    """Automation can configure all build-time roles without a prompt."""
    root = tmp_path / ".wmo"

    result = CliRunner().invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(root),
            "--non-interactive",
            "--provider",
            "openai",
            "--connection",
            "openai-main",
            "--api-key-env",
            "OPENAI_API_KEY",
            "--world-model",
            "world-id",
            "--judge",
            "judge-id",
            "--embedder",
            "embed-id",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "configured providers" in result.output
    catalog = load_model_catalog(root / "models.toml")
    assert catalog.models["world-model"].model == "world-id"
    assert catalog.models["judge"].model == "judge-id"
    assert catalog.models["embedder"].model == "embed-id"
    assert catalog.roles.candidates == ()


def test_noninteractive_setup_reports_all_missing_required_flags(tmp_path: Path) -> None:
    """Automation failures explain how to complete setup."""
    result = CliRunner().invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(tmp_path / ".wmo"),
            "--non-interactive",
            "--provider",
            "openai",
        ],
    )

    assert result.exit_code == 2
    for flag in ("--connection", "--api-key-env", "--world-model", "--judge", "--embedder"):
        assert flag in result.output
    assert not (tmp_path / ".wmo" / "models.toml").exists()


def test_interactive_judge_rejection_cancels_without_writing(tmp_path: Path) -> None:
    """The interactive collector finishes before the atomic service is invoked."""
    root = tmp_path / ".wmo"
    result = CliRunner().invoke(
        app,
        ["config", "providers", "--root", str(root)],
        input="openai\n\n\ny\nworld-id\njudge-id\nn\n",
    )

    assert result.exit_code == 1
    assert not (root / "models.toml").exists()

"""Command-level tests for role-free gateway provider connection management."""

from __future__ import annotations

import json
from pathlib import Path

from click import unstyle
from typer.testing import CliRunner

from exp.cli.app import app
from exp.runtime.gateway.management import GatewayManagement

_runner = CliRunner()


def _initialized_root(tmp_path: Path) -> Path:
    """Initialize one gateway root for provider commands."""
    GatewayManagement(tmp_path).initialize()
    return tmp_path


def _plain_output(output: str) -> str:
    """Flatten styled panel output into one space-separated searchable line."""
    return " ".join(unstyle(output).replace("\u2502", " ").split())


def test_provider_add_rejects_an_unsupported_provider_identifier(tmp_path: Path) -> None:
    """A mistyped provider id fails closed with the supported identifiers listed."""
    root = _initialized_root(tmp_path)

    result = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "mistyped",
            "--provider",
            "openai_compatible",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )

    assert result.exit_code != 0
    plain = _plain_output(result.output)
    assert "unsupported provider 'openai_compatible'" in plain
    assert "openai-compatible" in plain
    assert GatewayManagement(root).provider_connections() == ()


def test_provider_update_rejects_an_unsupported_provider_identifier(tmp_path: Path) -> None:
    """Updates pass through the same fail-closed provider identifier validation."""
    root = _initialized_root(tmp_path)
    add = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "main",
            "--provider",
            "openai-compatible",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )
    assert add.exit_code == 0

    update = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "update",
            "main",
            "--provider",
            "open-ai",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )

    assert update.exit_code != 0
    assert "unsupported provider 'open-ai'" in _plain_output(update.output)
    (authority,) = GatewayManagement(root).provider_connections()
    assert authority.config.provider == "openai-compatible"


def test_provider_add_accepts_a_supported_provider_identifier(tmp_path: Path) -> None:
    """A registry-supported provider id is stored and listed without secret values."""
    root = _initialized_root(tmp_path)

    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "main",
            "--provider",
            "openai-compatible",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )
    listed = _runner.invoke(
        app,
        ["config", "gateway", "provider", "list", "--json", "--root", str(root)],
    )

    assert added.exit_code == 0
    receipt = json.loads(added.output)
    assert receipt["resource_id"] == "main"
    assert receipt["changed"] is True
    assert listed.exit_code == 0
    listing = json.loads(listed.output)
    assert listing["resource_kind"] == "providers"
    (item,) = listing["items"]
    assert item["provider"] == "openai-compatible"


def test_azure_update_preserves_model_inference_surface_when_omitted(tmp_path: Path) -> None:
    """An endpoint-only update cannot silently switch Foundry back to classic Azure."""
    root = _initialized_root(tmp_path)
    common = [
        "--provider",
        "azure",
        "--base-url",
        "https://resource.services.ai.azure.com/models",
        "--credential-env",
        "AZURE_FOUNDRY_API_KEY",
        "--api-version",
        "2024-05-01-preview",
        "--non-interactive",
        "--json",
        "--root",
        str(root),
    ]
    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "foundry",
            *common,
            "--azure-api-surface",
            "model_inference",
        ],
    )
    updated = _runner.invoke(
        app,
        ["config", "gateway", "provider", "update", "foundry", *common],
    )
    listed = _runner.invoke(
        app,
        ["config", "gateway", "provider", "list", "--json", "--root", str(root)],
    )

    assert added.exit_code == 0
    assert updated.exit_code == 0
    (authority,) = GatewayManagement(root).provider_connections()
    assert authority.config.azure_api_surface == "model_inference"
    assert json.loads(listed.output)["items"][0]["azure_api_surface"] == "model_inference"


def test_provider_update_drops_azure_surface_when_changing_provider(tmp_path: Path) -> None:
    """A cross-provider update cannot carry an Azure-only discriminator forward."""
    root = _initialized_root(tmp_path)
    added = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "add",
            "foundry",
            "--provider",
            "azure",
            "--base-url",
            "https://resource.services.ai.azure.com",
            "--credential-env",
            "AZURE_FOUNDRY_API_KEY",
            "--api-version",
            "2024-05-01-preview",
            "--azure-api-surface",
            "model_inference",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )
    updated = _runner.invoke(
        app,
        [
            "config",
            "gateway",
            "provider",
            "update",
            "foundry",
            "--provider",
            "openai-compatible",
            "--base-url",
            "https://gateway.example.test/v1",
            "--credential-env",
            "CUSTOM_API_KEY",
            "--non-interactive",
            "--json",
            "--root",
            str(root),
        ],
    )

    assert added.exit_code == 0, added.output
    assert updated.exit_code == 0, updated.output
    (authority,) = GatewayManagement(root).provider_connections()
    assert authority.config.provider == "openai-compatible"
    assert authority.config.azure_api_surface is None

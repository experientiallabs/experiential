"""Tests for the exp auth list, login, and logout commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.auth import ProviderAuthStore, default_auth_path
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
)
from exp.common.models.catalog import ModelRoles, write_model_catalog

_SECRET = "sk-auth-cli-secret"
_OTHER = "sk-auth-cli-other"


def _write_catalog(root: Path) -> None:
    """Write one OpenAI and one OpenAI-compatible connection into models.toml.

    Args:
        root: Temporary EXP root.
    """
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                "openai": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY"),
                "acme": ConnectionConfig(
                    provider="openai-compatible",
                    base_url="https://acme.example/v1",
                    api_key_env="ACME_API_KEY",
                ),
            },
            models={
                "luna": ModelRecord(
                    connection="openai",
                    model="gpt-5.6-luna",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities=ModelCapabilities(
                        supports_completions=True,
                        supports_embeddings=True,
                        input_cost_per_million_tokens_usd=0,
                        output_cost_per_million_tokens_usd=0,
                        cached_input_cost_per_million_tokens_usd=0,
                        cache_write_cost_per_million_tokens_usd=0,
                    ),
                )
            },
            roles=ModelRoles(world_model="luna", judge="luna", embedder="luna"),
        ),
    )


def test_list_shows_source_without_secret_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """List reports environment and stored sources and never prints a key."""
    _write_catalog(tmp_path)
    ProviderAuthStore(default_auth_path()).put("acme", _SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.delenv("ACME_API_KEY", raising=False)
    runner = CliRunner()

    result = runner.invoke(app, ["auth", "list", "--root", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    by_id = {row["connection_id"]: row for row in payload["connections"]}
    assert by_id["openai"]["source"] == "environment"
    assert by_id["acme"]["source"] == "stored"
    assert _SECRET not in result.output
    assert "from-env" not in result.output


def test_login_persists_a_pasted_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Login writes the hidden paste for the selected connection."""
    _write_catalog(tmp_path)
    monkeypatch.setattr("exp.cli.auth.app.getpass", lambda prompt="": _SECRET)
    runner = CliRunner()

    result = runner.invoke(app, ["auth", "login", "acme", "--root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "stored credential for connection acme" in result.output
    assert _SECRET not in result.output
    assert ProviderAuthStore(default_auth_path()).get("acme") == _SECRET


def test_logout_removes_only_the_selected_stored_credential(tmp_path: Path) -> None:
    """Logout deletes one stored key and leaves the other connection intact."""
    _write_catalog(tmp_path)
    store = ProviderAuthStore(default_auth_path())
    store.put("openai", _SECRET)
    store.put("acme", _OTHER)
    runner = CliRunner()

    result = runner.invoke(app, ["auth", "logout", "acme", "--root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "removed stored credential for connection acme" in result.output
    assert _OTHER not in result.output
    assert store.get("acme") is None
    assert store.get("openai") == _SECRET


def test_noninteractive_login_without_connection_fails(tmp_path: Path) -> None:
    """Automation must name the connection instead of receiving a prompt."""
    _write_catalog(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["auth", "login", "--root", str(tmp_path)])

    assert result.exit_code != 0
    assert "noninteractive auth login requires a connection ID" in result.output
    assert ProviderAuthStore(default_auth_path()).get("openai") is None


def test_login_rejects_bedrock(tmp_path: Path) -> None:
    """Bedrock stays on the AWS chain and cannot store an API key."""
    write_model_catalog(
        tmp_path / "models.toml",
        ModelCatalog(
            connections={"bedrock": ConnectionConfig(provider="bedrock", region="us-east-1")},
            models={
                "claude": ModelRecord(
                    connection="bedrock",
                    model="anthropic.claude",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities=ModelCapabilities(
                        supports_completions=True,
                        supports_embeddings=True,
                        input_cost_per_million_tokens_usd=0,
                        output_cost_per_million_tokens_usd=0,
                        cached_input_cost_per_million_tokens_usd=0,
                        cache_write_cost_per_million_tokens_usd=0,
                    ),
                )
            },
            roles=ModelRoles(world_model="claude", judge="claude", embedder="claude"),
        ),
    )
    runner = CliRunner()

    result = runner.invoke(app, ["auth", "login", "bedrock", "--root", str(tmp_path)])

    assert result.exit_code != 0
    assert "AWS credential chain" in result.output

"""Provider setup CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.models import (
    ConnectionConfig,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    load_model_catalog,
    write_model_catalog,
)

_RUNNER = CliRunner()


def _connection_json(name: str, provider: str, env: str) -> str:
    """Return one concise structured connection flag value.

    Args:
        name: Local connection name.
        provider: Supported provider kind.
        env: Credential environment-variable name.

    Returns:
        JSON value accepted by ``--connection-json``.
    """
    return json.dumps({"name": name, "provider": provider, "api_key_env": env})


def _model_json(
    alias: str,
    connection: str,
    model: str,
    *,
    embeddings: bool = False,
) -> str:
    """Return one concise structured model flag value.

    Args:
        alias: Stable local model alias.
        connection: Referenced local provider connection.
        model: Exact provider-side model ID.
        embeddings: Whether this model supports embeddings.

    Returns:
        JSON value accepted by ``--model-json``.
    """
    return json.dumps(
        {
            "alias": alias,
            "connection": connection,
            "model": model,
            "supports_embeddings": embeddings,
            "input_cost_per_million_tokens_usd": 0.1 if embeddings else None,
        }
    )


def test_noninteractive_setup_collects_many_connections_models_and_roles(tmp_path: Path) -> None:
    """Automation supplies repeatable collections before independent role aliases.

    Args:
        tmp_path: Temporary WMO root receiving the model catalog.
    """
    root = tmp_path / ".wmo"
    result = _RUNNER.invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(root),
            "--non-interactive",
            "--connection-json",
            _connection_json("openai", "openai", "OPENAI_API_KEY"),
            "--connection-json",
            _connection_json("gemini", "gemini", "GEMINI_API_KEY"),
            "--model-json",
            _model_json("world", "openai", "world-id"),
            "--model-json",
            _model_json("judge", "openai", "judge-id"),
            "--model-json",
            _model_json("embed", "gemini", "embed-id", embeddings=True),
            "--world-model",
            "world",
            "--judge",
            "judge",
            "--embedder",
            "embed",
        ],
    )

    assert result.exit_code == 0, result.output
    catalog = load_model_catalog(root / "models.toml")
    assert tuple(sorted(catalog.connections)) == ("gemini", "openai")
    assert tuple(sorted(catalog.models)) == ("embed", "judge", "world")
    assert catalog.roles == ModelRoles(world_model="world", judge="judge", embedder="embed")


def test_noninteractive_setup_reports_every_missing_collection_and_role(tmp_path: Path) -> None:
    """One failure lists the complete remediation instead of serial missing prompts.

    Args:
        tmp_path: Temporary WMO root without provider configuration.
    """
    result = _RUNNER.invoke(
        app,
        ["config", "providers", "--root", str(tmp_path / ".wmo"), "--non-interactive"],
        color=True,
    )

    assert result.exit_code == 2
    output = " ".join(unstyle(result.output).replace("│", " ").split())
    for value in (
        "at least one --connection-json",
        "at least one --model-json",
        "--world-model",
        "--judge",
        "--embedder",
    ):
        assert value in output
    assert not (tmp_path / ".wmo" / "models.toml").exists()


def test_setup_preserves_router_candidates_and_unrelated_entries(tmp_path: Path) -> None:
    """Editing build roles does not consume or mutate router candidate selection.

    Args:
        tmp_path: Temporary WMO root containing preserved catalog state.
    """
    root = tmp_path / ".wmo"
    root.mkdir()
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                "router": ConnectionConfig(provider="openrouter", api_key_env="OPENROUTER_API_KEY")
            },
            models={"candidate": ModelRecord(connection="router", model="vendor/candidate")},
            roles=ModelRoles(candidates=("candidate",), incumbent="candidate"),
        ),
    )

    result = _RUNNER.invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(root),
            "--non-interactive",
            "--connection-json",
            _connection_json("openai", "openai", "OPENAI_API_KEY"),
            "--model-json",
            _model_json("all", "openai", "all-id", embeddings=True),
            "--world-model",
            "all",
            "--judge",
            "all",
            "--embedder",
            "all",
        ],
    )

    assert result.exit_code == 0, result.output
    catalog = load_model_catalog(root / "models.toml")
    assert catalog.roles.candidates == ("candidate",)
    assert catalog.roles.incumbent == "candidate"
    assert catalog.models["candidate"].model == "vendor/candidate"


def test_structured_input_rejects_openai_compatible_without_capabilities(tmp_path: Path) -> None:
    """Private compatible endpoints cannot acquire provider-wide capability guesses.

    Args:
        tmp_path: Temporary WMO root receiving rejected structured input.
    """
    connection = json.dumps(
        {
            "name": "private",
            "provider": "openai-compatible",
            "api_key_env": "PRIVATE_API_KEY",
            "base_url": "https://models.example.test/v1",
        }
    )
    model = json.dumps({"alias": "private", "connection": "private", "model": "private-model"})

    result = _RUNNER.invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(tmp_path / ".wmo"),
            "--non-interactive",
            "--connection-json",
            connection,
            "--model-json",
            model,
            "--world-model",
            "private",
            "--judge",
            "private",
            "--embedder",
            "private",
        ],
    )

    assert result.exit_code == 2
    assert "must declare embedding support" in result.output
    assert not (tmp_path / ".wmo" / "models.toml").exists()


def test_noninteractive_setup_accepts_azure_and_bedrock_connections(tmp_path: Path) -> None:
    """Azure and Bedrock connections persist secret-free catalog fields only."""
    root = tmp_path / ".wmo"
    azure = json.dumps(
        {
            "name": "azure",
            "provider": "azure",
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "base_url": "https://resource.openai.azure.com",
            "api_version": "v1",
        }
    )
    bedrock = json.dumps({"name": "bedrock", "provider": "bedrock", "region": "us-east-1"})
    completion = json.dumps(
        {
            "alias": "gpt",
            "connection": "azure",
            "model": "gpt-deployment",
            "supports_completions": True,
            "input_cost_per_million_tokens_usd": 0,
            "output_cost_per_million_tokens_usd": 0,
            "cached_input_cost_per_million_tokens_usd": 0,
            "cache_write_cost_per_million_tokens_usd": 0,
        }
    )
    embed = json.dumps(
        {
            "alias": "titan",
            "connection": "bedrock",
            "model": "amazon.titan-embed-text-v2:0",
            "supports_embeddings": True,
            "input_cost_per_million_tokens_usd": 0,
        }
    )

    result = _RUNNER.invoke(
        app,
        [
            "config",
            "providers",
            "--root",
            str(root),
            "--non-interactive",
            "--connection-json",
            azure,
            "--connection-json",
            bedrock,
            "--model-json",
            completion,
            "--model-json",
            embed,
            "--world-model",
            "gpt",
            "--judge",
            "gpt",
            "--embedder",
            "titan",
        ],
    )

    assert result.exit_code == 0, result.output
    catalog = load_model_catalog(root / "models.toml")
    assert catalog.connections["azure"].api_version == "v1"
    assert catalog.connections["bedrock"].region == "us-east-1"
    assert catalog.connections["bedrock"].api_key_env is None
    assert catalog.models["gpt"].model == "gpt-deployment"
    assert catalog.models["titan"].model == "amazon.titan-embed-text-v2:0"
    text = (root / "models.toml").read_text(encoding="utf-8")
    assert "AZURE_OPENAI_API_KEY" in text
    assert "sk-" not in text


def test_interactive_final_rejection_writes_no_catalog(tmp_path: Path) -> None:
    """All answers remain in memory until the user confirms the complete summary.

    Args:
        tmp_path: Temporary WMO root used to prove no catalog write occurs.
    """
    root = tmp_path / ".wmo"
    result = _RUNNER.invoke(
        app,
        ["config", "providers", "--root", str(root)],
        input=(
            "y\n\n\nn\nn\nn\nn\nn\nn\nn\nopenai\nall\nmodel-id\nn\ny\ny\nn\nn\nn\n0\nn\nall\ny\nall\nn\n"
        ),
    )

    assert result.exit_code == 1
    assert "Configuration summary" in result.output
    assert not (root / "models.toml").exists()

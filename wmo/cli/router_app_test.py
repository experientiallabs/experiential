"""Configless automatic router CLI boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.models import (
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    write_model_catalog,
)

_RUNNER = CliRunner()


def test_router_help_requires_only_project_and_never_exposes_config() -> None:
    """The supported happy path is configless and project-addressed."""
    result = _RUNNER.invoke(app, ["optimize", "router", "--help"])

    assert result.exit_code == 0, result.output
    output = unstyle(result.output)
    assert "PROJECT" in output
    assert "--config" not in output
    assert "--candidate" in output
    assert "--maximum-provider-co" in output
    assert "--approve-fidelity" not in output
    assert "--preferred-fidelity" not in output
    assert "fidelity" not in output.lower()


def test_missing_project_aggregates_before_catalog_write_or_credential_read(
    tmp_path: Path,
) -> None:
    """A normal configless invocation fails read-only when build and review state are absent.

    Args:
        tmp_path: Temporary local WMO root containing only a secret-free catalog.
    """
    root = tmp_path / ".wmo"
    root.mkdir()
    write_model_catalog(root / "models.toml", _eligible_catalog())
    original = (root / "models.toml").read_bytes()

    result = _RUNNER.invoke(
        app,
        [
            "optimize",
            "router",
            "missing-project",
            "--root",
            str(root),
            "--candidate",
            "candidate-a",
            "--candidate",
            "candidate-b",
            "--incumbent",
            "candidate-a",
            "--non-interactive",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    output = " ".join(unstyle(result.output).replace("│", " ").split())
    assert "router optimization prerequisites are incomplete" in output
    assert "project:" in output
    assert (root / "models.toml").read_bytes() == original
    assert not (root / "projects" / "missing-project").exists()


def test_missing_package_revision_fails_before_spend_consent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject unavailable producer provenance before rendering or approving provider spend.

    Args:
        monkeypatch: CLI boundary overrides.
        tmp_path: Temporary local WMO root.
    """
    root = tmp_path / ".wmo"
    root.mkdir()
    write_model_catalog(root / "models.toml", _eligible_catalog())
    consent_calls = 0

    def missing_revision() -> str:
        """Model unavailable installed-package producer provenance."""
        raise ValueError("package producer revision is unavailable")

    def unexpected_consent(*_args: object, **_kwargs: object) -> bool:
        """Fail if revision preflight allows the spend-consent boundary to open."""
        nonlocal consent_calls
        consent_calls += 1
        return True

    monkeypatch.setattr("wmo.cli.router_app.installed_release_revision", missing_revision)
    monkeypatch.setattr("wmo.cli.router_app.require_spend_consent", unexpected_consent)

    result = _RUNNER.invoke(
        app,
        [
            "optimize",
            "router",
            "missing-project",
            "--root",
            str(root),
            "--candidate",
            "candidate-a",
            "--candidate",
            "candidate-b",
            "--incumbent",
            "candidate-a",
            "--non-interactive",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "package producer revision is unavailable" in unstyle(result.output)
    assert consent_calls == 0


def _eligible_catalog() -> ModelCatalog:
    """Return two candidates plus complete build roles without real credentials.

    Returns:
        Secret-free catalog whose environment references deliberately do not exist.
    """
    connection = ConnectionConfig(provider="openai", api_key_env="P10_MISSING_API_KEY")
    completion = ModelCapabilities(
        supports_completions=True,
        context_window_tokens=32_000,
        maximum_output_tokens=16_000,
        input_cost_per_million_tokens_usd=1.0,
        output_cost_per_million_tokens_usd=2.0,
        cached_input_cost_per_million_tokens_usd=0.5,
        cache_write_cost_per_million_tokens_usd=1.5,
    )
    embedding = ModelCapabilities(
        supports_embeddings=True,
        input_cost_per_million_tokens_usd=0.1,
    )
    return ModelCatalog(
        connections={"provider": connection},
        models={
            "candidate-a": ModelRecord(
                connection="provider", model="candidate-a", capabilities=completion
            ),
            "candidate-b": ModelRecord(
                connection="provider", model="candidate-b", capabilities=completion
            ),
            "world": ModelRecord(connection="provider", model="world", capabilities=completion),
            "judge": ModelRecord(connection="provider", model="judge", capabilities=completion),
            "embedder": ModelRecord(
                connection="provider", model="embedder", capabilities=embedding
            ),
        },
        roles=ModelRoles(
            world_model="world",
            judge="judge",
            embedder="embedder",
        ),
    )

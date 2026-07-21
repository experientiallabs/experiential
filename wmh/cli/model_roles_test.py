"""Tests for shared CLI resolution of opt-in model providers."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

import wmh.cli.model_roles as model_roles_module
from wmh.cli.model_roles import OptInModelRole, resolve_opt_in_model_provider
from wmh.config.settings import ModelRole, ModelsSettings, ProjectSettings, save_settings
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
)


class _Provider:
    """A provider identity used without making model calls."""

    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="fallback")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self) -> VerifyResult:
        raise NotImplementedError


def _settings_for_role(role: OptInModelRole, configured: ModelRole) -> ProjectSettings:
    """Build settings with exactly one opt-in role configured."""
    models = (
        ModelsSettings(agent=configured) if role == "agent" else ModelsSettings(meta=configured)
    )
    return ProjectSettings(models=models)


@pytest.mark.parametrize("role", ["agent", "meta"])
def test_unset_opt_in_role_keeps_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: OptInModelRole
) -> None:
    fallback = _Provider()

    def unexpected_provider(_config: ProviderConfig) -> _Provider:
        raise AssertionError("unset roles must not construct a provider")

    monkeypatch.setattr(model_roles_module, "get_provider", unexpected_provider)

    provider, model = resolve_opt_in_model_provider(str(tmp_path / ".wmh"), role, fallback)

    assert provider is fallback
    assert model is None


def test_configured_role_forwards_fields_and_defaults_the_azure_api_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".wmh"
    save_settings(
        ProjectSettings(
            models=ModelsSettings(
                agent=ModelRole(
                    provider="azure",
                    model="gpt-5.5",
                    region="us-east-2",
                    endpoint="https://x.example",
                    deployment="gpt-5-5",
                    reasoning_effort="high",
                )
            )
        ),
        root,
    )
    sentinel = _Provider()
    configs: list[ProviderConfig] = []

    def fake_get_provider(config: ProviderConfig) -> _Provider:
        configs.append(config)
        return sentinel

    monkeypatch.setattr(model_roles_module, "get_provider", fake_get_provider)

    provider, model = resolve_opt_in_model_provider(str(root), "agent", _Provider())

    assert provider is sentinel
    assert model == "gpt-5.5"
    [config] = configs
    assert config.kind is ProviderKind.AZURE_OPENAI
    assert config.model == "gpt-5.5"
    assert config.region == "us-east-2"
    assert config.endpoint == "https://x.example"
    assert config.deployment == "gpt-5-5"
    assert config.api_version == "2024-05-01-preview"
    assert config.reasoning_effort == "high"


def test_explicit_api_version_overrides_the_azure_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".wmh"
    save_settings(
        ProjectSettings(
            models=ModelsSettings(
                meta=ModelRole(
                    provider="azure",
                    model="gpt-5.5",
                    deployment="gpt-5-5",
                    api_version="2025-01-01",
                )
            )
        ),
        root,
    )
    configs: list[ProviderConfig] = []

    def fake_get_provider(config: ProviderConfig) -> _Provider:
        configs.append(config)
        return _Provider()

    monkeypatch.setattr(model_roles_module, "get_provider", fake_get_provider)

    resolve_opt_in_model_provider(str(root), "meta", _Provider())

    [config] = configs
    assert config.api_version == "2025-01-01"


@pytest.mark.parametrize("role", ["agent", "meta"])
def test_unknown_provider_names_the_configured_role(tmp_path: Path, role: OptInModelRole) -> None:
    root = tmp_path / ".wmh"
    save_settings(
        _settings_for_role(role, ModelRole(provider="bogus", model="m")),
        root,
    )

    with pytest.raises(
        typer.BadParameter,
        match=rf"settings \[models\.{role}\] has unknown provider 'bogus'",
    ):
        resolve_opt_in_model_provider(str(root), role, _Provider())

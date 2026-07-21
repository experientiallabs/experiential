"""Provider resolution for opt-in model roles shared by CLI workflows."""

from __future__ import annotations

from typing import Literal

import typer

from wmh.config.settings import ModelRole, load_settings
from wmh.providers.base import Provider, ProviderConfig, ProviderKind
from wmh.providers.registry import get_provider

OptInModelRole = Literal["agent", "meta"]

# Azure OpenAI chat completions need an API version on every call. When an opt-in role does not
# pin one in settings, this shared default API version applies.
_DEFAULT_AZURE_API_VERSION = "2024-05-01-preview"


def resolve_opt_in_model_provider(
    root: str,
    role: OptInModelRole,
    fallback: Provider,
) -> tuple[Provider, str | None]:
    """Resolve one opt-in model role, or return the caller's fallback provider.

    Args:
        root: Project artifact root containing ``settings.toml``.
        role: The opt-in role to resolve.
        fallback: Provider retained when the role is not configured.

    Returns:
        The resolved provider and configured model name, or the fallback and ``None``.

    Raises:
        typer.BadParameter: The configured provider kind is unknown.
    """
    configured = load_settings(root).models.resolve(role)
    if configured is None:
        return fallback, None
    config = _model_config(configured, role=role)
    return get_provider(config), configured.model


def resolve_required_model_config(root: str, role: OptInModelRole) -> ProviderConfig:
    """Resolve one opt-in role a workflow requires (no fallback provider exists for it).

    The harbor optimize flow has no world model whose provider could stand in, so its
    ``agent`` (worker) and ``meta`` (proposer) roles must be configured explicitly.
    """
    configured = load_settings(root).models.resolve(role)
    if configured is None:
        raise typer.BadParameter(
            f"settings [models.{role}] must be configured in <root>/settings.toml for this "
            f"workflow; add a [models.{role}] table with provider and model"
        )
    return _model_config(configured, role=role)


def _model_config(configured: ModelRole, *, role: OptInModelRole) -> ProviderConfig:
    """Turn one configured role into provider-neutral config with the Azure default."""
    try:
        kind = ProviderKind(configured.provider)
    except ValueError:
        kinds = ", ".join(kind.value for kind in ProviderKind)
        raise typer.BadParameter(
            f"settings [models.{role}] has unknown provider {configured.provider!r}; "
            f"choose one of: {kinds}"
        ) from None
    api_version = configured.api_version
    if api_version is None and kind is ProviderKind.AZURE_OPENAI:
        api_version = _DEFAULT_AZURE_API_VERSION
    return ProviderConfig(
        kind=kind,
        model=configured.model,
        region=configured.region,
        endpoint=configured.endpoint,
        deployment=configured.deployment,
        api_version=api_version,
        reasoning_effort=configured.reasoning_effort,
    )

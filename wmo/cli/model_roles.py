"""Reading project settings, and resolving the opt-in model roles, for CLI workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from wmo.common.config.config import ARTIFACT_DIR
from wmo.common.config.settings import ModelRole, ProjectSettings, load_settings
from wmo.common.providers.base import Provider, ProviderConfig, ProviderKind
from wmo.common.providers.models import resolve_provider_model
from wmo.common.providers.registry import get_provider

OptInModelRole = Literal["agent", "meta"]
ModelRoleName = Literal["worker", "judge", "summary", "meta", "agent"]

MODEL_ROLE_NAMES: tuple[ModelRoleName, ...] = ("worker", "judge", "summary", "meta", "agent")
"""Every role `ModelsSettings` carries, in report order. Pinned to that model by a test."""

# Azure OpenAI chat completions need an API version on every call. When an opt-in role (or a
# pool entry, or the local worker role) does not pin one, this shared default applies. Imported
# by `wmo.cli.app` and `wmo.cli.pool_registry` so the three surfaces cannot drift apart.
DEFAULT_AZURE_API_VERSION = "2024-05-01-preview"


def load_settings_or_abort(root: str | Path = ARTIFACT_DIR) -> ProjectSettings:
    """`load_settings`, with an unreadable settings file as a usage error, not a traceback.

    The one path every CLI command uses to read `<root>/settings.toml`. Hand-editing that file is
    a documented workflow, and a file written by an older CLI outlives an upgrade, so "the file is
    broken" is an ordinary user state: it has to print as an error naming the file and the repair,
    the way an unknown provider inside the file already does.

    Raises:
        typer.BadParameter: The settings file is unreadable, is not valid TOML, or does not match
            the current schema.
    """
    try:
        return load_settings(root)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


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
        typer.BadParameter: The settings file is unreadable, or the configured provider kind is
            unknown.
    """
    configured = load_settings_or_abort(root).models.resolve(role)
    if configured is None:
        return fallback, None
    config = _model_config(configured, role=role)
    return get_provider(config), configured.model


def resolve_required_model_config(root: str, role: OptInModelRole) -> ProviderConfig:
    """Resolve one opt-in role a workflow requires (no fallback provider exists for it).

    The harbor optimize flow has no world model whose provider could stand in, so its
    ``agent`` (worker) and ``meta`` (proposer) roles must be configured explicitly.
    """
    configured = load_settings_or_abort(root).models.resolve(role)
    if configured is None:
        raise typer.BadParameter(
            f"settings [models.{role}] must be configured in <root>/settings.toml for this "
            f"workflow; add a [models.{role}] table with provider and model"
        )
    return _model_config(configured, role=role)


def configured_role_configs(root: str) -> list[tuple[ModelRoleName, ProviderConfig]]:
    """Resolve every model role a project has actually written down, in `MODEL_ROLE_NAMES` order.

    This is the "what has this project configured?" view, as opposed to the "what serves this
    call?" view of `ModelsSettings.resolve`: unset `judge`/`summary` fall back to `worker` at USE
    time, and resolving that fallback here would report one backend three times under three role
    names. So this reads the raw fields and returns only the roles the settings file sets.

    The model is canonicalized through the built-in catalog, because that is what a role holds:
    `wmo providers set` stores the canonical TYPE (`claude-opus-4-8`), not the backend's runtime
    id (`us.anthropic.claude-opus-4-8`), so a caller that wants to reach the backend has to
    resolve it before invoking a provider. Unknown ids pass through unchanged, which is
    what a self-hosted model or a `tinker://` weights path needs.

    Args:
        root: Project artifact root containing ``settings.toml``.

    Returns:
        One ``(role, config)`` pair per configured role; empty when nothing is configured.

    Raises:
        typer.BadParameter: The settings file is unreadable, or a configured role names an
            unknown provider kind.
    """
    models = load_settings_or_abort(root).models
    resolved: list[tuple[ModelRoleName, ProviderConfig]] = []
    for role in MODEL_ROLE_NAMES:
        configured: ModelRole | None = getattr(models, role)
        if configured is None:
            continue
        config = _model_config(configured, role=role)
        spec = resolve_provider_model(config.kind, config.model)
        resolved.append(
            (
                role,
                config.model_copy(
                    update={
                        "model": spec.model_id,
                        "model_type": config.model_type or spec.model_type,
                    }
                ),
            )
        )
    return resolved


def _model_config(configured: ModelRole, *, role: ModelRoleName) -> ProviderConfig:
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
        api_version = DEFAULT_AZURE_API_VERSION
    config = ProviderConfig(
        kind=kind,
        model=configured.model,
        model_type=configured.model_type,
        region=configured.region,
        endpoint=configured.endpoint,
        deployment=configured.deployment,
        api_version=api_version,
        reasoning_effort=configured.reasoning_effort,
    )
    if configured.chat_max_tokens_field is None:
        # Unset keeps meaning "resolve from the built-in catalog", which is right for every named
        # model; forcing the field here would override the catalog for all of them.
        return config
    return config.model_copy(update={"chat_max_tokens_field": configured.chat_max_tokens_field})

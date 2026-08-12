"""Shared provider and scenario configuration for CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from wmo.cli.model_roles import load_settings_or_abort
from wmo.common.config import (
    PROVIDER_ENV_VARS,
)

_PROVIDER_EXTRAS: dict[str, str] = {"tinker": "sft"}
"""Providers whose SDK ships in an optional extra, keyed by kind value for the hint.

Every other provider SDK is a core dependency. A missing module for one of those means the
environment is stale or hand-rolled, rather than that an optional install step was skipped.
"""

if TYPE_CHECKING:
    from wmo.common.providers import ProviderConfig, ProviderKind
    from wmo.common.providers.models import ProviderModel


def _worker_provider_config(
    provider: str,
    model: str,
    region: str | None,
    *,
    endpoint: str | None = None,
    deployment: str | None = None,
    api_version: str | None = None,
) -> ProviderConfig:
    """Resolve the provider settings used by the built-in worker agent."""
    from wmo.common.providers import ProviderKind

    config = _provider_config(provider, model, region)
    if endpoint is not None:
        config = config.model_copy(update={"endpoint": endpoint})
    if config.kind is ProviderKind.AZURE_OPENAI:
        config = config.model_copy(
            update={
                "deployment": deployment or config.model_type or config.model,
                "api_version": api_version or "2024-05-01-preview",
            }
        )
    return config


def _missing_sdk(detail: str) -> bool:
    """Does a failed ping's detail mean "the SDK is absent" rather than "the creds are wrong"?

    Two shapes reach here: the raw ImportError text of a core SDK ("No module named 'boto3'"), and
    an optional extra's own message, which replaces that text with its install hint (see
    `wmo.common.providers.tinker.check_tinker_prerequisites`) and therefore never contains the
    module wording.
    """
    return "No module named" in detail or "SDK is not installed" in detail


def _credential_hint(kind: ProviderKind, detail: str) -> str:
    """The next step for a failed provider ping: install the SDKs, or fix creds/model id.

    Shared by the pre-build guard and `wmo providers verify` so both name the same env vars.
    """
    if _missing_sdk(detail):
        extra = _PROVIDER_EXTRAS.get(kind.value)
        if extra is not None:
            return (
                f"run `pip install 'world-model-optimizer[{extra}]'` (or `uv sync --extra {extra}` "
                "in a checkout), then re-run `wmo providers verify`"
            )
        # The rest are core deps; a missing module means the env is stale or hand-rolled.
        return "run `uv sync` to install the provider SDKs"
    envs = ", ".join(PROVIDER_ENV_VARS.get(kind, []))
    hint = f" ({envs})" if envs else ""
    return f"check the model id and that your credentials are set{hint}"


def _provider_kind(provider: str) -> ProviderKind:
    """The `ProviderKind` a `--provider` flag names, as a usage error when it names none."""
    from wmo.common.providers import ProviderKind

    try:
        return ProviderKind(provider)
    except ValueError:
        kinds = ", ".join(k.value for k in ProviderKind)
        raise typer.BadParameter(f"unknown provider {provider!r}; choose one of: {kinds}") from None


def _provider_config(provider: str, model: str, region: str | None) -> ProviderConfig:
    from wmo.common.providers import ProviderConfig
    from wmo.common.providers.models import resolve_provider_model

    kind = _provider_kind(provider)
    spec = resolve_provider_model(kind, model)
    return ProviderConfig(
        kind=kind,
        model_type=spec.model_type,
        model=spec.model_id,
        region=region,
    )


# The backend a worker-role command falls back to when the project configured no
# `[models.worker]` role at all. Never a substitute for a configured role.
_DEFAULT_WORKER_PROVIDER = "bedrock"
_DEFAULT_WORKER_MODEL = "claude-opus-4-8"


def _default_model_for_provider(kind: ProviderKind) -> str:
    """`kind`'s flagship: the model to run when neither a flag nor a role named one.

    A default model belongs to ONE backend - pairing `--provider openai` with bedrock's
    `claude-opus-4-8` sends a model OpenAI has never heard of. `openrouter` and `tinker` publish
    no built-in rows (nothing can derive an operator's route or weights path), so they must be
    told which model to run.
    """
    from wmo.common.providers.models import model_types_for_provider

    catalog = model_types_for_provider(kind)
    if not catalog:
        raise typer.BadParameter(
            f"provider {kind.value!r} has no default model; pass --model <model>, or run "
            f"`wmo providers set --provider {kind.value} --model <model>` to configure the "
            f"worker role"
        )
    return catalog[0]


def _role_provider_config(role: str, region: str | None) -> ProviderConfig | None:
    """ProviderConfig for a settings-defined model role, or None when the role isn't configured.

    Roles live in `.wmo/settings.toml` under `[models.worker|judge|summary]`; unset judge/summary
    fall back to worker (see `ModelsSettings.resolve`). A role's stored region wins over the
    generic `--region` flag - the flag also feeds the embedder, and e.g. a judge pinned to the
    one region where its model is enabled must not follow it.
    """
    configured = load_settings_or_abort().models.resolve(role)
    if configured is None:
        return None
    config = _provider_config(configured.provider, configured.model, configured.region or region)
    return config.model_copy(
        update={"endpoint": configured.endpoint, "deployment": configured.deployment}
    )


def _azure_deployment_for_model(
    configured: ProviderConfig, spec: ProviderModel, deployment: str | None
) -> str:
    """The Azure deployment to invoke after `--model` moved the role off its configured one.

    On Azure the wire `model` IS the deployment name (`AzureOpenAIProvider._deployment`), so a
    role's deployment names the model being replaced. Keeping it would call the old model and
    report the new one. Guessing the new one is no better: an operator's deployment name is not
    derivable from a model id, and a wrong guess 404s on every prediction, which `wmo eval`
    reports as a silent `fidelity=0.000` at exit 0 - the defect this whole path exists to stop.
    So a model swap on Azure has to be told which deployment serves it.
    """
    if deployment is not None:
        return deployment
    if configured.deployment is None:
        # Nothing configured to contradict, so derive it from the model as
        # `_worker_provider_config` does; the role could not have been called without one anyway.
        return spec.model_type
    if configured.deployment in (spec.model_type, spec.model_id):
        return configured.deployment
    raise typer.BadParameter(
        f"the configured azure worker serves {configured.model} from deployment "
        f"{configured.deployment!r}, and on Azure the deployment name is what is actually "
        f"invoked, so --model {spec.model_type} needs the deployment that serves it. Run "
        f"`wmo providers set --provider azure --model {spec.model_type} "
        f"--deployment <deployment>` to point the worker role at it."
    )


def _worker_role_provider_config(
    provider: str | None,
    model: str | None,
    region: str | None,
    *,
    deployment: str | None = None,
) -> ProviderConfig:
    """The backend for a worker-role call: explicit flags, then the worker role, then the default.

    `wmo providers set` writes `[models.worker]` and is step 1 of the documented getting-started
    path, so a command that ignored it would run against a provider the user never configured.
    Each field falls back independently, and a `--provider` naming a DIFFERENT backend than the
    configured role drops that role's model and connection fields, which belong to the backend it
    replaced - the model then comes from the NEW backend's catalog, never from bedrock's.
    """
    from wmo.common.providers import ProviderKind
    from wmo.common.providers.models import resolve_provider_model

    configured = _role_provider_config("worker", region)
    if configured is None or (provider is not None and provider != configured.kind.value):
        kind = _provider_kind(provider or _DEFAULT_WORKER_PROVIDER)
        config = _provider_config(kind.value, model or _default_model_for_provider(kind), region)
    elif model is None:
        config = configured
    else:
        spec = resolve_provider_model(configured.kind, model)
        if spec.model_id == configured.model:
            # Re-stating the role's own model is not a model change: leave its connection alone.
            config = configured
        else:
            update: dict[str, object] = {"model_type": spec.model_type, "model": spec.model_id}
            if configured.kind is ProviderKind.AZURE_OPENAI:
                update["deployment"] = _azure_deployment_for_model(configured, spec, deployment)
            config = configured.model_copy(update=update)
    if deployment is not None:
        config = config.model_copy(update={"deployment": deployment})
    return config

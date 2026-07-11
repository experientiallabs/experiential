"""Canonical model types and provider-specific runtime identifiers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from wmh.providers.base import ProviderKind


class ProviderModel(BaseModel):
    """One canonical model type as exposed by a concrete provider.

    ``model_type`` is the provider-independent identity used in product and
    configuration surfaces. ``model_id`` is the provider-specific value sent
    over the wire. Keeping both prevents Bedrock inference-profile ids and
    Azure deployment names from leaking into model identity.
    """

    model_config = ConfigDict(frozen=True)

    provider: ProviderKind
    model_type: str
    model_id: str


_MODELS: tuple[ProviderModel, ...] = (
    ProviderModel(provider=ProviderKind.OPENAI, model_type="gpt-5.5", model_id="gpt-5.5"),
    ProviderModel(provider=ProviderKind.OPENAI, model_type="gpt-5.5-pro", model_id="gpt-5.5-pro"),
    ProviderModel(provider=ProviderKind.OPENAI, model_type="gpt-5.4", model_id="gpt-5.4"),
    ProviderModel(provider=ProviderKind.OPENAI, model_type="gpt-5.4-mini", model_id="gpt-5.4-mini"),
    ProviderModel(provider=ProviderKind.OPENAI_RESPONSES, model_type="gpt-5.5", model_id="gpt-5.5"),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.5-pro",
        model_id="gpt-5.5-pro",
    ),
    ProviderModel(provider=ProviderKind.OPENAI_RESPONSES, model_type="gpt-5.4", model_id="gpt-5.4"),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.4-mini",
        model_id="gpt-5.4-mini",
    ),
    ProviderModel(
        provider=ProviderKind.ANTHROPIC,
        model_type="claude-opus-4-8",
        model_id="claude-opus-4-8",
    ),
    ProviderModel(
        provider=ProviderKind.ANTHROPIC,
        model_type="claude-opus-4-7",
        model_id="claude-opus-4-7",
    ),
    ProviderModel(
        provider=ProviderKind.ANTHROPIC,
        model_type="claude-sonnet-4-6",
        model_id="claude-sonnet-4-6",
    ),
    ProviderModel(
        provider=ProviderKind.ANTHROPIC,
        model_type="claude-haiku-4-5",
        model_id="claude-haiku-4-5",
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-opus-4-8",
        model_id="us.anthropic.claude-opus-4-8",
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-opus-4-7",
        model_id="us.anthropic.claude-opus-4-7",
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-sonnet-4-6",
        model_id="us.anthropic.claude-sonnet-4-6",
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-haiku-4-5",
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    ),
    ProviderModel(provider=ProviderKind.BEDROCK, model_type="glm-5", model_id="zai.glm-5"),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="qwen3-vl-235b-a22b",
        model_id="qwen.qwen3-vl-235b-a22b",
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="gpt-oss-120b",
        model_id="openai.gpt-oss-120b-1:0",
    ),
    # Azure uses deployment names at runtime. These defaults deliberately
    # match the canonical type; callers with custom deployment names override
    # ProviderConfig.deployment without changing model identity.
    ProviderModel(provider=ProviderKind.AZURE_OPENAI, model_type="gpt-5.5", model_id="gpt-5.5"),
    ProviderModel(provider=ProviderKind.AZURE_OPENAI, model_type="gpt-5.4", model_id="gpt-5.4"),
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.4-mini",
        model_id="gpt-5.4-mini",
    ),
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="deepseek-v4-pro",
        model_id="deepseek-v4-pro",
    ),
    ProviderModel(provider=ProviderKind.AZURE_OPENAI, model_type="kimi-k2.6", model_id="kimi-k2.6"),
)


def model_types_for_provider(provider: ProviderKind) -> tuple[str, ...]:
    """Return canonical model types offered by ``provider`` in catalog order."""
    return tuple(spec.model_type for spec in _MODELS if spec.provider is provider)


def resolve_provider_model(provider: ProviderKind, model: str) -> ProviderModel:
    """Resolve a canonical model type or known runtime id for ``provider``.

    Unknown values remain valid as custom/self-hosted model types whose wire id
    is identical. This preserves WMH's open-ended provider contract while
    canonicalizing every model in the built-in catalog.
    """
    for spec in _MODELS:
        if spec.provider is provider and model in (spec.model_type, spec.model_id):
            return spec
    return ProviderModel(provider=provider, model_type=model, model_id=model)

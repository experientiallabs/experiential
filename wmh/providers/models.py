"""Canonical model types and provider-specific runtime identifiers."""

from __future__ import annotations

import re

from llm_waterfall import ChatMaxTokensField, ReasoningEffort
from llm_waterfall.reasoning import bedrock_base_model_id
from pydantic import BaseModel, ConfigDict

from wmh.providers.base import ProviderKind


class ProviderModel(BaseModel):
    """One canonical model type as exposed by a concrete provider.

    ``model_type`` is the provider-independent identity used in product and
    configuration surfaces. ``model_id`` is the provider-specific value sent
    over the wire. ``chat_max_tokens_field`` records which output-token field
    the model accepts on OpenAI-compatible chat requests. ``forward_temperature``
    records whether structured chat requests may send the sampling parameter at
    all. Keeping these together prevents provider details from leaking into
    product catalogs.
    """

    model_config = ConfigDict(frozen=True)

    provider: ProviderKind
    model_type: str
    model_id: str
    chat_max_tokens_field: ChatMaxTokensField = "max_completion_tokens"
    forward_temperature: bool = True
    reasoning_efforts: tuple[ReasoningEffort, ...] = ()


_OPENAI_GPT_54_55_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
)
_OPENAI_PRO_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = ("medium", "high", "xhigh")
_BEDROCK_ADAPTIVE_EFFORTS: tuple[ReasoningEffort, ...] = ("low", "medium", "high")
_BEDROCK_OPUS_46_EFFORTS: tuple[ReasoningEffort, ...] = (
    *_BEDROCK_ADAPTIVE_EFFORTS,
    "max",
)


_MODELS: tuple[ProviderModel, ...] = (
    ProviderModel(
        provider=ProviderKind.OPENAI,
        model_type="gpt-5.5",
        model_id="gpt-5.5",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI,
        model_type="gpt-5.5-pro",
        model_id="gpt-5.5-pro",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI,
        model_type="gpt-5.4",
        model_id="gpt-5.4",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI,
        model_type="gpt-5.4-mini",
        model_id="gpt-5.4-mini",
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.5",
        model_id="gpt-5.5",
        forward_temperature=False,
        reasoning_efforts=_OPENAI_GPT_54_55_REASONING_EFFORTS,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.5-pro",
        model_id="gpt-5.5-pro",
        forward_temperature=False,
        reasoning_efforts=_OPENAI_PRO_REASONING_EFFORTS,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.4",
        model_id="gpt-5.4",
        forward_temperature=False,
        reasoning_efforts=_OPENAI_GPT_54_55_REASONING_EFFORTS,
    ),
    ProviderModel(
        provider=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.4-mini",
        model_id="gpt-5.4-mini",
        forward_temperature=False,
        reasoning_efforts=_OPENAI_GPT_54_55_REASONING_EFFORTS,
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
        model_type="claude-opus-4-6",
        model_id="claude-opus-4-6",
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
        # Opus 4.8 dropped sampling parameters. Bedrock rejects a forwarded
        # temperature with a ValidationException instead of ignoring it.
        forward_temperature=False,
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-opus-4-7",
        model_id="us.anthropic.claude-opus-4-7",
        reasoning_efforts=_BEDROCK_ADAPTIVE_EFFORTS,
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-opus-4-6",
        model_id="us.anthropic.claude-opus-4-6-v1",
        reasoning_efforts=_BEDROCK_OPUS_46_EFFORTS,
    ),
    ProviderModel(
        provider=ProviderKind.BEDROCK,
        model_type="claude-sonnet-4-6",
        model_id="us.anthropic.claude-sonnet-4-6",
        reasoning_efforts=_BEDROCK_ADAPTIVE_EFFORTS,
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
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model_id="gpt-5.5",
        forward_temperature=False,
        reasoning_efforts=_OPENAI_GPT_54_55_REASONING_EFFORTS,
    ),
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.4",
        model_id="gpt-5.4",
        forward_temperature=False,
        reasoning_efforts=_OPENAI_GPT_54_55_REASONING_EFFORTS,
    ),
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.4-mini",
        model_id="gpt-5.4-mini",
        forward_temperature=False,
        reasoning_efforts=_OPENAI_GPT_54_55_REASONING_EFFORTS,
    ),
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="deepseek-v4-pro",
        model_id="deepseek-v4-pro",
        chat_max_tokens_field="max_tokens",
    ),
    ProviderModel(
        provider=ProviderKind.AZURE_OPENAI,
        model_type="kimi-k2.6",
        model_id="kimi-k2.6",
        chat_max_tokens_field="max_tokens",
    ),
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
        if (
            provider is ProviderKind.OPENAI_RESPONSES
            and spec.provider is provider
            and _is_openai_snapshot(model, spec.model_id)
        ):
            return spec
        if (
            provider is ProviderKind.BEDROCK
            and spec.provider is provider
            and bedrock_base_model_id(model) == bedrock_base_model_id(spec.model_id)
        ):
            return spec
    return ProviderModel(provider=provider, model_type=model, model_id=model)


def _is_openai_snapshot(model: str, alias: str) -> bool:
    """Match only the documented ``alias-YYYY-MM-DD`` snapshot shape.

    A raw prefix check would misclassify ``gpt-5.5-pro`` as a snapshot of ``gpt-5.5`` and grant
    it the wrong effort capability.  Requiring the date suffix keeps model-family resolution
    exact while supporting pinned experiment models.
    """
    return re.fullmatch(rf"{re.escape(alias)}-\d{{4}}-\d{{2}}-\d{{2}}", model) is not None


def resolve_chat_max_tokens_field(
    provider: ProviderKind,
    model: str,
    *,
    fallback: ChatMaxTokensField = "max_completion_tokens",
) -> ChatMaxTokensField:
    """Resolve a known model contract, or preserve a custom endpoint's fallback."""
    for spec in _MODELS:
        if spec.provider is provider and model in (spec.model_type, spec.model_id):
            return spec.chat_max_tokens_field
    return fallback

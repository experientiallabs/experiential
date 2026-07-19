"""Tests for canonical model types and provider runtime ids."""

import pytest
from llm_waterfall import ReasoningEffort
from pydantic import ValidationError

from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.models import model_types_for_provider, resolve_provider_model


def test_same_model_type_resolves_to_provider_specific_ids() -> None:
    """Claude keeps one identity while direct and Bedrock wire ids differ."""
    direct = resolve_provider_model(ProviderKind.ANTHROPIC, "claude-opus-4-8")
    bedrock = resolve_provider_model(ProviderKind.BEDROCK, "claude-opus-4-8")

    assert direct.model_type == bedrock.model_type == "claude-opus-4-8"
    assert direct.model_id == "claude-opus-4-8"
    assert bedrock.model_id == "us.anthropic.claude-opus-4-8"


def test_opus_4_6_resolves_to_exact_bedrock_inference_profile() -> None:
    """Opus 4.6 keeps one identity while Bedrock requires its versioned profile id."""
    direct = resolve_provider_model(ProviderKind.ANTHROPIC, "claude-opus-4-6")
    bedrock = resolve_provider_model(ProviderKind.BEDROCK, "claude-opus-4-6")
    bedrock_runtime_id = resolve_provider_model(
        ProviderKind.BEDROCK, "us.anthropic.claude-opus-4-6-v1"
    )

    assert direct.model_type == bedrock.model_type == "claude-opus-4-6"
    assert direct.model_id == "claude-opus-4-6"
    assert bedrock.model_id == "us.anthropic.claude-opus-4-6-v1"
    assert bedrock_runtime_id == bedrock


def test_bedrock_cross_region_profile_resolves_to_same_model_contract() -> None:
    us = resolve_provider_model(ProviderKind.BEDROCK, "us.anthropic.claude-opus-4-6-v1")
    global_profile = resolve_provider_model(
        ProviderKind.BEDROCK, "global.anthropic.claude-opus-4-6-v1"
    )

    assert global_profile == us


@pytest.mark.parametrize(
    "runtime_id",
    [
        "au.anthropic.claude-opus-4-6-v1",
        (
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
            "us.anthropic.claude-opus-4-6-v1"
        ),
        ("arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-4-6-v1"),
    ],
)
def test_bedrock_profile_and_foundation_arns_resolve_to_model_contract(
    runtime_id: str,
) -> None:
    expected = resolve_provider_model(ProviderKind.BEDROCK, "us.anthropic.claude-opus-4-6-v1")

    assert resolve_provider_model(ProviderKind.BEDROCK, runtime_id) == expected


def test_bedrock_opus_4_6_accepts_and_serializes_max_reasoning_effort() -> None:
    config = ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model_type="claude-opus-4-6",
        model="us.anthropic.claude-opus-4-6-v1",
        reasoning_effort="max",
    )

    assert config.reasoning_effort == "max"
    assert config.model_dump(mode="json")["reasoning_effort"] == "max"
    with pytest.raises(ValidationError, match="frozen"):
        config.reasoning_effort = "low"


@pytest.mark.parametrize(
    ("model", "accepted", "rejected"),
    [
        ("gpt-5.5", ("none", "high", "xhigh"), ("minimal", "max")),
        ("gpt-5.5-pro", ("medium", "high", "xhigh"), ("none", "low", "max")),
        ("gpt-5.4", ("none", "high", "xhigh"), ("minimal", "max")),
        ("gpt-5.4-mini", ("none", "high", "xhigh"), ("minimal", "max")),
    ],
)
def test_openai_responses_effort_capabilities_match_model_contract(
    model: str,
    accepted: tuple[ReasoningEffort, ...],
    rejected: tuple[ReasoningEffort, ...],
) -> None:
    for effort in accepted:
        assert (
            ProviderConfig(
                kind=ProviderKind.OPENAI_RESPONSES,
                model=model,
                reasoning_effort=effort,
            ).reasoning_effort
            == effort
        )
    for effort in rejected:
        with pytest.raises(ValidationError, match="does not support reasoning effort|Opus 4.6"):
            ProviderConfig(
                kind=ProviderKind.OPENAI_RESPONSES,
                model=model,
                reasoning_effort=effort,
            )


def test_openai_responses_max_error_names_the_actual_model_contract() -> None:
    with pytest.raises(ValidationError, match="openai_responses/gpt-5.5") as error:
        ProviderConfig(
            kind=ProviderKind.OPENAI_RESPONSES,
            model="gpt-5.5",
            reasoning_effort="max",
        )

    assert "Claude Opus" not in str(error.value)


def test_openai_responses_pinned_snapshot_keeps_alias_capabilities() -> None:
    config = ProviderConfig(
        kind=ProviderKind.OPENAI_RESPONSES,
        model_type="gpt-5.5",
        model="gpt-5.5-2026-04-23",
        reasoning_effort="high",
    )

    assert config.reasoning_effort == "high"
    assert (
        resolve_provider_model(ProviderKind.OPENAI_RESPONSES, "gpt-5.5-pro-2026-04-23").model_type
        == "gpt-5.5-pro"
    )


@pytest.mark.parametrize(
    ("kind", "model_type", "model", "effort", "message"),
    [
        (
            ProviderKind.BEDROCK,
            "claude-opus-4-7",
            "us.anthropic.claude-opus-4-7",
            "max",
            "Claude Opus 4.6",
        ),
        (
            ProviderKind.BEDROCK,
            "claude-haiku-4-5",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "high",
            "does not support reasoning effort",
        ),
        (
            ProviderKind.OPENAI,
            "gpt-5.5",
            "gpt-5.5",
            "high",
            "does not support reasoning effort",
        ),
    ],
)
def test_provider_config_rejects_unsupported_reasoning_effort(
    kind: ProviderKind,
    model_type: str,
    model: str,
    effort: ReasoningEffort,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ProviderConfig(
            kind=kind,
            model_type=model_type,
            model=model,
            reasoning_effort=effort,
        )


def test_provider_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="reasoning_mode"):
        ProviderConfig.model_validate(
            {
                "kind": "bedrock",
                "model": "us.anthropic.claude-opus-4-6-v1",
                "reasoning_mode": "adaptive",
            }
        )


@pytest.mark.parametrize(
    ("model_type", "model"),
    [
        ("claude-opus-4-6", "us.anthropic.claude-sonnet-4-6"),
        ("claude-sonnet-4-6", "us.anthropic.claude-opus-4-6-v1"),
    ],
)
def test_reasoning_config_rejects_mismatched_model_identity_and_runtime(
    model_type: str,
    model: str,
) -> None:
    with pytest.raises(ValidationError, match="does not match runtime model"):
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model_type=model_type,
            model=model,
            reasoning_effort="high",
        )


def test_model_catalog_owns_temperature_compatibility() -> None:
    """Sampling compatibility follows the canonical model across callers."""
    opus = resolve_provider_model(ProviderKind.BEDROCK, "claude-opus-4-8")
    opus_runtime_id = resolve_provider_model(ProviderKind.BEDROCK, "us.anthropic.claude-opus-4-8")
    sonnet = resolve_provider_model(ProviderKind.BEDROCK, "claude-sonnet-4-6")

    assert opus.forward_temperature is False
    assert opus_runtime_id.forward_temperature is False
    assert sonnet.forward_temperature is True
    assert resolve_provider_model(ProviderKind.OPENAI, "gpt-5.5").forward_temperature is False
    assert (
        resolve_provider_model(ProviderKind.AZURE_OPENAI, "gpt-5.4-mini").forward_temperature
        is False
    )
    assert (
        resolve_provider_model(ProviderKind.AZURE_OPENAI, "deepseek-v4-pro").forward_temperature
        is True
    )


def test_runtime_id_resolves_back_to_canonical_model_type() -> None:
    """Known provider ids never become a second public model identity."""
    resolved = resolve_provider_model(
        ProviderKind.BEDROCK, "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )

    assert resolved.model_type == "claude-haiku-4-5"
    assert resolved.model_id == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_provider_catalog_exposes_only_canonical_model_types() -> None:
    assert model_types_for_provider(ProviderKind.BEDROCK)[:5] == (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    )


def test_azure_models_declare_their_chat_token_parameter() -> None:
    """Each built-in Azure model owns its compatible output-token field."""
    expected = {
        "gpt-5.5": "max_completion_tokens",
        "gpt-5.4": "max_completion_tokens",
        "gpt-5.4-mini": "max_completion_tokens",
        "deepseek-v4-pro": "max_tokens",
        "kimi-k2.6": "max_tokens",
    }

    actual = {
        model_type: resolve_provider_model(
            ProviderKind.AZURE_OPENAI, model_type
        ).chat_max_tokens_field
        for model_type in expected
    }

    assert actual == expected


def test_unknown_custom_model_round_trips() -> None:
    resolved = resolve_provider_model(ProviderKind.OPENAI, "my-fine-tune")
    assert resolved.model_type == "my-fine-tune"
    assert resolved.model_id == "my-fine-tune"
    assert resolved.chat_max_tokens_field == "max_completion_tokens"


def test_explicit_openai_compatible_endpoint_keeps_sampling_capability() -> None:
    config = ProviderConfig(
        kind=ProviderKind.OPENAI,
        model="gpt-5.5",
        endpoint="http://localhost:8001/v1",
    )

    assert config.resolved_chat_forward_temperature() is True


def test_provider_config_resolves_model_contract_before_custom_deployment() -> None:
    """Canonical model type, not an opaque Azure deployment name, selects parameters."""
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="customer-gpt-deployment",
        deployment="customer-gpt-deployment",
    )

    assert config.chat_max_tokens_field == "max_completion_tokens"
    assert "chat_max_tokens_field" not in config.model_fields_set
    assert config.resolved_chat_max_tokens_field() == "max_completion_tokens"


def test_provider_config_allows_an_explicit_custom_endpoint_override() -> None:
    """Unknown OpenAI-compatible servers can override the catalog default."""
    config = ProviderConfig(
        kind=ProviderKind.OPENAI,
        model="legacy-compatible-model",
        chat_max_tokens_field="max_tokens",
    )

    assert "chat_max_tokens_field" in config.model_fields_set
    assert config.resolved_chat_max_tokens_field() == "max_tokens"


def test_persisted_default_does_not_override_a_known_model_contract() -> None:
    """Serialized defaults remain fallbacks; known catalog metadata still wins."""
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="kimi-k2.6",
        model="customer-kimi-deployment",
    )

    loaded = ProviderConfig.model_validate(config.model_dump(mode="json"))

    assert loaded.chat_max_tokens_field == "max_completion_tokens"
    assert loaded.resolved_chat_max_tokens_field() == "max_tokens"

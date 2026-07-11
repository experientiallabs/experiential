"""Tests for canonical model types and provider runtime ids."""

from wmh.providers.base import ProviderKind
from wmh.providers.models import model_types_for_provider, resolve_provider_model


def test_same_model_type_resolves_to_provider_specific_ids() -> None:
    """Claude keeps one identity while direct and Bedrock wire ids differ."""
    direct = resolve_provider_model(ProviderKind.ANTHROPIC, "claude-opus-4-8")
    bedrock = resolve_provider_model(ProviderKind.BEDROCK, "claude-opus-4-8")

    assert direct.model_type == bedrock.model_type == "claude-opus-4-8"
    assert direct.model_id == "claude-opus-4-8"
    assert bedrock.model_id == "us.anthropic.claude-opus-4-8"


def test_runtime_id_resolves_back_to_canonical_model_type() -> None:
    """Known provider ids never become a second public model identity."""
    resolved = resolve_provider_model(
        ProviderKind.BEDROCK, "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )

    assert resolved.model_type == "claude-haiku-4-5"
    assert resolved.model_id == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_provider_catalog_exposes_only_canonical_model_types() -> None:
    assert model_types_for_provider(ProviderKind.BEDROCK)[:4] == (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    )


def test_unknown_custom_model_round_trips() -> None:
    resolved = resolve_provider_model(ProviderKind.OPENAI, "my-fine-tune")
    assert resolved.model_type == "my-fine-tune"
    assert resolved.model_id == "my-fine-tune"

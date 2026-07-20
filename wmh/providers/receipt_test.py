"""Tests for sanitized provider receipt construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import wmh.providers.receipt as mod
from wmh.core.types import JsonObject
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.receipt import build_chat_provider_receipt


def test_response_identity_contract_distinguishes_bedrock_and_openai_routes() -> None:
    assert mod.ProviderResponseIdentity(provider=ProviderKind.BEDROCK) == (
        mod.ProviderResponseIdentity(
            provider=ProviderKind.BEDROCK,
            response_model=None,
            system_fingerprint=None,
        )
    )
    with pytest.raises(ValidationError, match="does not return"):
        mod.ProviderResponseIdentity(
            provider=ProviderKind.BEDROCK,
            response_model="fabricated-model",
        )
    with pytest.raises(ValidationError, match="require an expected response model"):
        mod.ProviderResponseIdentity(provider=ProviderKind.AZURE_OPENAI)

    identity = mod.ProviderResponseIdentity(
        provider=ProviderKind.AZURE_OPENAI,
        response_model="glm-served-model",
        system_fingerprint="fp-123",
    )

    assert identity.response_model == "glm-served-model"
    assert identity.system_fingerprint == "fp-123"

    for field, value in (
        ("response_model", "m" * 2_049),
        ("system_fingerprint", "f" * 513),
    ):
        payload = {
            "provider": ProviderKind.AZURE_OPENAI,
            "response_model": "served-model",
            "system_fingerprint": None,
        }
        payload[field] = value
        with pytest.raises(ValidationError, match="at most"):
            mod.ProviderResponseIdentity.model_validate(payload)


def test_receipt_validation_requires_frozen_served_model_and_fingerprint() -> None:
    provider = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.4-mini",
        deployment="worker-deployment",
        endpoint="https://example.openai.azure.com",
        api_version="2026-01-01",
    )
    identity = mod.ProviderResponseIdentity(
        provider=ProviderKind.AZURE_OPENAI,
        response_model="gpt-5.4-mini-2026-06-01",
        system_fingerprint="fp-exact",
    )
    common = {
        "provider": ProviderKind.AZURE_OPENAI.value,
        "provider_request_id": "request-1",
        "response_id": "response-1",
        "requested_model": "worker-deployment",
        "request_payload": {},
        "temperature": None,
        "max_tokens": 4_096,
        "max_tokens_field": provider.resolved_chat_max_tokens_field(),
        "started_at_unix_s": 10.0,
        "finished_at_unix_s": 11.0,
    }
    valid = build_chat_provider_receipt(
        **common,
        response_model=identity.response_model,
        system_fingerprint=identity.system_fingerprint,
    )
    mod.validate_chat_provider_receipt(
        valid,
        provider_config=provider,
        requested_temperature=0.7,
        max_tokens=4_096,
        response_identity=identity,
    )

    for field, value in (
        ("response_model", "retargeted-model"),
        ("system_fingerprint", "fp-drifted"),
    ):
        receipt = valid.model_copy(update={field: value})
        with pytest.raises(ValueError, match="frozen response identity"):
            mod.validate_chat_provider_receipt(
                receipt,
                provider_config=provider,
                requested_temperature=0.7,
                max_tokens=4_096,
                response_identity=identity,
            )

    omitted_fingerprint = identity.model_copy(update={"system_fingerprint": None})
    with pytest.raises(ValueError, match="frozen response identity"):
        mod.validate_chat_provider_receipt(
            valid,
            provider_config=provider,
            requested_temperature=0.7,
            max_tokens=4_096,
            response_identity=omitted_fingerprint,
        )


def test_receipt_hashes_full_wire_payload_and_detects_sampling_controls() -> None:
    payload: JsonObject = {
        "model": "deployment",
        "messages": [{"role": "user", "content": "secret prompt"}],
        "temperature": 0.7,
        "max_completion_tokens": 4_096,
        "seed": 7,
        "cache_control": {"type": "ephemeral"},
    }

    receipt = build_chat_provider_receipt(
        provider="azure",
        provider_request_id="provider-request-1",
        response_id="completion-1",
        requested_model="deployment",
        response_model="gpt-5.5-2026-06-01",
        system_fingerprint="fp-1",
        request_payload=payload,
        temperature=0.7,
        max_tokens=4_096,
        max_tokens_field="max_completion_tokens",
        started_at_unix_s=10.0,
        finished_at_unix_s=11.0,
    )

    assert receipt.request_digest.startswith("sha256:")
    assert "secret prompt" not in receipt.model_dump_json()
    assert receipt.provider_request_id == "provider-request-1"
    assert receipt.response_id == "completion-1"
    assert receipt.seed_supplied is True
    assert receipt.cache_config_supplied is True


def test_receipt_digest_changes_with_nested_request_content() -> None:
    kwargs = {
        "provider": "bedrock",
        "provider_request_id": "request-1",
        "response_id": None,
        "requested_model": "us.example.model",
        "response_model": None,
        "system_fingerprint": None,
        "temperature": 0.7,
        "max_tokens": 4_096,
        "max_tokens_field": "inferenceConfig.maxTokens",
        "started_at_unix_s": 10.0,
        "finished_at_unix_s": 11.0,
    }
    first = build_chat_provider_receipt(
        **kwargs,
        request_payload={"messages": [{"role": "user", "content": "first"}]},
    )
    second = build_chat_provider_receipt(
        **kwargs,
        request_payload={"messages": [{"role": "user", "content": "second"}]},
    )

    assert first.request_digest != second.request_digest
    assert first.seed_supplied is False
    assert first.cache_config_supplied is False


def test_receipt_digest_supports_signed_bedrock_reasoning_bytes_without_collisions() -> None:
    kwargs = {
        "provider": "bedrock",
        "provider_request_id": "request-1",
        "response_id": None,
        "requested_model": "us.anthropic.claude-opus-4-6-v1",
        "response_model": None,
        "system_fingerprint": None,
        "temperature": None,
        "max_tokens": 4_096,
        "max_tokens_field": "inferenceConfig.maxTokens",
        "started_at_unix_s": 10.0,
        "finished_at_unix_s": 11.0,
    }
    signed = build_chat_provider_receipt(
        **kwargs,
        request_payload={
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"reasoningContent": {"redactedContent": b"\x00signed\xffreasoning"}}
                    ],
                }
            ]
        },
    )
    repeated = build_chat_provider_receipt(
        **kwargs,
        request_payload={
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"reasoningContent": {"redactedContent": b"\x00signed\xffreasoning"}}
                    ],
                }
            ]
        },
    )
    json_lookalike = build_chat_provider_receipt(
        **kwargs,
        request_payload={
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "reasoningContent": {
                                "redactedContent": ["bytes", "AHNpZ25lZP9yZWFzb25pbmc="]
                            }
                        }
                    ],
                }
            ]
        },
    )

    assert signed.request_digest == repeated.request_digest
    assert signed.request_digest != json_lookalike.request_digest


def test_ordinary_tool_schema_cache_names_are_not_provider_cache_controls() -> None:
    receipt = build_chat_provider_receipt(
        provider="bedrock",
        provider_request_id="request-1",
        response_id=None,
        requested_model="model",
        response_model=None,
        system_fingerprint=None,
        request_payload={
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "inputSchema": {
                                "json": {
                                    "properties": {
                                        "cache_control": {},
                                        "cachePoint": {},
                                        "prompt_cache_key": {},
                                        "seed": {},
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        },
        temperature=0.7,
        max_tokens=4_096,
        max_tokens_field="inferenceConfig.maxTokens",
        started_at_unix_s=10.0,
        finished_at_unix_s=11.0,
    )

    assert receipt.cache_config_supplied is False
    assert receipt.seed_supplied is False


def test_openai_tool_schema_cache_names_are_not_request_cache_controls() -> None:
    receipt = build_chat_provider_receipt(
        provider="openai",
        provider_request_id="request-1",
        response_id="completion-1",
        requested_model="model",
        response_model="served-model",
        system_fingerprint=None,
        request_payload={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "inspect",
                        "parameters": {
                            "properties": {
                                "cache_control": {},
                                "prompt_cache_key": {},
                                "prompt_cache_retention": {},
                            }
                        },
                    },
                }
            ]
        },
        temperature=None,
        max_tokens=4_096,
        max_tokens_field="max_completion_tokens",
        started_at_unix_s=10.0,
        finished_at_unix_s=11.0,
    )

    assert receipt.cache_config_supplied is False


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_openai_compatible_message_content_cache_control_is_detected(provider: str) -> None:
    receipt = build_chat_provider_receipt(
        provider=provider,
        provider_request_id="request-1",
        response_id="completion-1",
        requested_model="model",
        response_model="served-model",
        system_fingerprint=None,
        request_payload={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        },
        temperature=None,
        max_tokens=4_096,
        max_tokens_field="max_completion_tokens",
        started_at_unix_s=10.0,
        finished_at_unix_s=11.0,
    )

    assert receipt.cache_config_supplied is True


def test_provider_cache_controls_are_detected_only_at_structural_wire_locations() -> None:
    common = {
        "provider_request_id": "request-1",
        "response_id": None,
        "requested_model": "model",
        "response_model": None,
        "system_fingerprint": None,
        "temperature": None,
        "max_tokens": 4_096,
        "max_tokens_field": "inferenceConfig.maxTokens",
        "started_at_unix_s": 10.0,
        "finished_at_unix_s": 11.0,
    }
    bedrock = build_chat_provider_receipt(
        provider="bedrock",
        request_payload={
            "messages": [{"role": "user", "content": [{"text": "hello"}, {"cachePoint": {}}]}]
        },
        **common,
    )
    openai = build_chat_provider_receipt(
        provider="openai",
        request_payload={"messages": [], "prompt_cache_key": "stable-prefix"},
        **common,
    )

    assert bedrock.cache_config_supplied is True
    assert openai.cache_config_supplied is True


def test_bedrock_direct_tool_cache_point_is_detected() -> None:
    receipt = build_chat_provider_receipt(
        provider="bedrock",
        provider_request_id="request-1",
        response_id=None,
        requested_model="model",
        response_model=None,
        system_fingerprint=None,
        request_payload={"toolConfig": {"tools": [{"cachePoint": {"type": "default"}}]}},
        temperature=0.7,
        max_tokens=4_096,
        max_tokens_field="inferenceConfig.maxTokens",
        started_at_unix_s=10.0,
        finished_at_unix_s=11.0,
    )

    assert receipt.cache_config_supplied is True


def test_receipt_rejects_missing_request_identity_and_reversed_time() -> None:
    with pytest.raises(ValidationError, match="provider_request_id"):
        build_chat_provider_receipt(
            provider="bedrock",
            provider_request_id="",
            response_id=None,
            requested_model="model",
            response_model=None,
            system_fingerprint=None,
            request_payload={},
            temperature=None,
            max_tokens=1,
            max_tokens_field="inferenceConfig.maxTokens",
            started_at_unix_s=10.0,
            finished_at_unix_s=11.0,
        )
    with pytest.raises(ValidationError, match="finish before"):
        build_chat_provider_receipt(
            provider="bedrock",
            provider_request_id="request-1",
            response_id=None,
            requested_model="model",
            response_model=None,
            system_fingerprint=None,
            request_payload={},
            temperature=None,
            max_tokens=1,
            max_tokens_field="inferenceConfig.maxTokens",
            started_at_unix_s=11.0,
            finished_at_unix_s=10.0,
        )


def test_receipt_rejects_conflated_transport_and_response_identity() -> None:
    with pytest.raises(ValidationError, match="distinct from the response id"):
        build_chat_provider_receipt(
            provider="openai",
            provider_request_id="same-id",
            response_id="same-id",
            requested_model="model",
            response_model="served-model",
            system_fingerprint=None,
            request_payload={},
            temperature=None,
            max_tokens=1,
            max_tokens_field="max_completion_tokens",
            started_at_unix_s=10.0,
            finished_at_unix_s=11.0,
        )

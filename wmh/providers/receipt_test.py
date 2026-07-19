"""Tests for sanitized provider receipt construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmh.core.types import JsonObject
from wmh.providers.receipt import build_chat_provider_receipt


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

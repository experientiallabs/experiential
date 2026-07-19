"""Sanitized, content-addressed provider call receipts for evaluation traces."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from typing import Self, TypeAlias

from llm_waterfall import ChatProviderReceipt
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from wmh.core.text import validate_durable_text
from wmh.providers.base import ProviderConfig, ProviderKind

ProviderRequestScalar: TypeAlias = None | bool | int | float | str | bytes
ProviderRequestValue: TypeAlias = (
    ProviderRequestScalar | Sequence["ProviderRequestValue"] | Mapping[str, "ProviderRequestValue"]
)
ProviderRequestPayload: TypeAlias = "Mapping[str, ProviderRequestValue]"


class ProviderResponseIdentity(BaseModel):
    """Frozen provider-reported model identity required from every scored receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderKind
    response_model: str | None = Field(default=None, min_length=1)
    system_fingerprint: str | None = Field(default=None, min_length=1)

    @field_validator("response_model", "system_fingerprint")
    @classmethod
    def _require_canonical_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value != value.strip():
            raise ValueError("provider response identity cannot have surrounding whitespace")
        validate_durable_text(value, field="provider response identity")
        return value

    @model_validator(mode="after")
    def _require_provider_evidence_shape(self) -> Self:
        if self.provider is ProviderKind.BEDROCK:
            if self.response_model is not None or self.system_fingerprint is not None:
                raise ValueError(
                    "Bedrock Converse does not return a served model or system fingerprint"
                )
        elif self.provider in {ProviderKind.AZURE_OPENAI, ProviderKind.OPENAI}:
            if self.response_model is None:
                raise ValueError("OpenAI-shaped scored routes require an expected response model")
        else:
            raise ValueError(
                "scored provider receipts currently support Bedrock and OpenAI-shaped routes"
            )
        return self


def build_chat_provider_receipt(
    *,
    provider: str,
    provider_request_id: str,
    response_id: str | None,
    requested_model: str,
    response_model: str | None,
    system_fingerprint: str | None,
    request_payload: ProviderRequestPayload,
    temperature: float | None,
    max_tokens: int,
    max_tokens_field: str,
    started_at_unix_s: float,
    finished_at_unix_s: float | None = None,
) -> ChatProviderReceipt:
    """Build a strict receipt without retaining prompt, tool, or credential material."""
    canonical = json.dumps(
        _canonical_receipt_value(request_payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return ChatProviderReceipt(
        provider=provider,
        provider_request_id=provider_request_id,
        response_id=response_id,
        requested_model=requested_model,
        response_model=response_model,
        system_fingerprint=system_fingerprint,
        request_digest="sha256:" + hashlib.sha256(canonical).hexdigest(),
        temperature=temperature,
        max_tokens=max_tokens,
        max_tokens_field=max_tokens_field,
        seed_supplied=_seed_supplied(request_payload),
        cache_config_supplied=_cache_config_supplied(provider, request_payload),
        started_at_unix_s=started_at_unix_s,
        finished_at_unix_s=(time.time() if finished_at_unix_s is None else finished_at_unix_s),
    )


def _canonical_receipt_value(value: ProviderRequestValue) -> JsonValue:
    """Encode one wire value into a collision-safe, deterministic JSON representation."""
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", value]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, Mapping):
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("provider receipt request objects require string keys")
        entries: list[JsonValue] = [
            [key, _canonical_receipt_value(value[key])] for key in sorted(keys)
        ]
        return ["object", entries]
    if isinstance(value, Sequence):
        return ["array", [_canonical_receipt_value(item) for item in value]]
    raise TypeError(f"unsupported provider receipt request value: {type(value).__name__}")


def _seed_supplied(request_payload: ProviderRequestPayload) -> bool:
    if "seed" in request_payload:
        return True
    additional = request_payload.get("additionalModelRequestFields")
    return isinstance(additional, Mapping) and "seed" in additional


def _cache_config_supplied(provider: str, request_payload: ProviderRequestPayload) -> bool:
    """Detect provider cache controls only in their documented wire locations."""
    if provider in {"openai", "azure"}:
        if any(
            field in request_payload
            for field in ("prompt_cache_key", "prompt_cache_retention", "cache_control")
        ):
            return True
        return _openai_message_content_has_cache_control(request_payload.get("messages"))
    if provider == "bedrock":
        if _bedrock_content_has_cache_point(request_payload.get("system")):
            return True
        messages = request_payload.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            for message in messages:
                if isinstance(message, Mapping) and _bedrock_content_has_cache_point(
                    message.get("content")
                ):
                    return True
        tool_config = request_payload.get("toolConfig")
        if isinstance(tool_config, Mapping):
            tools = tool_config.get("tools")
            if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
                if any(isinstance(tool, Mapping) and "cachePoint" in tool for tool in tools):
                    return True
        additional = request_payload.get("additionalModelRequestFields")
        return isinstance(additional, Mapping) and any(
            field in additional
            for field in ("prompt_cache_key", "prompt_cache_retention", "cache_control")
        )
    return False


def _openai_message_content_has_cache_control(value: ProviderRequestValue) -> bool:
    """Detect compatible-endpoint cache controls without scanning tool schemas."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    for message in value:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            if any(isinstance(block, Mapping) and "cache_control" in block for block in content):
                return True
    return False


def _bedrock_content_has_cache_point(value: object) -> bool:
    """Return whether one Bedrock content list contains a cache-point block."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    return any(isinstance(block, Mapping) and "cachePoint" in block for block in value)


def requested_chat_model(config: ProviderConfig) -> str:
    """Return the exact model or deployment placed on one provider request."""
    if config.kind is ProviderKind.AZURE_OPENAI:
        if config.deployment is None:
            raise ValueError("Azure provider receipt validation requires a deployment")
        return config.deployment
    return config.model


def validate_chat_provider_receipt(
    receipt: ChatProviderReceipt,
    *,
    provider_config: ProviderConfig,
    requested_temperature: float,
    max_tokens: int,
    response_identity: ProviderResponseIdentity | None = None,
) -> None:
    """Validate independently observable receipt fields against one frozen request route.

    The adapter-authored request digest is retained as opaque content-addressed evidence: the
    prompt-bearing request payload is deliberately unavailable to downstream validators. Every
    control that *is* independently known is checked here instead of trusting the receipt copy.
    """
    if provider_config.kind not in {
        ProviderKind.BEDROCK,
        ProviderKind.OPENAI,
        ProviderKind.AZURE_OPENAI,
    }:
        raise ValueError("configured provider does not support chat provider receipts")
    frozen_response_identity = (
        None
        if response_identity is None
        else ProviderResponseIdentity.model_validate(response_identity.model_dump())
    )
    if (
        frozen_response_identity is not None
        and frozen_response_identity.provider is not provider_config.kind
    ):
        raise ValueError("provider response identity disagrees with the frozen provider")
    expected_temperature = (
        requested_temperature if provider_config.resolved_chat_forward_temperature() else None
    )
    if provider_config.kind is ProviderKind.BEDROCK:
        expected_max_tokens_field = "inferenceConfig.maxTokens"
    elif (
        provider_config.kind is ProviderKind.AZURE_OPENAI
        and provider_config.reasoning_effort is not None
    ):
        expected_max_tokens_field = "max_output_tokens"
    else:
        expected_max_tokens_field = provider_config.resolved_chat_max_tokens_field()
    if receipt.provider != provider_config.kind.value:
        raise ValueError("provider receipt disagrees with the frozen provider")
    if receipt.requested_model != requested_chat_model(provider_config):
        raise ValueError("provider receipt disagrees with the frozen request model")
    if receipt.temperature != expected_temperature:
        raise ValueError("provider receipt disagrees with the frozen temperature")
    if receipt.max_tokens != max_tokens:
        raise ValueError("provider receipt disagrees with the frozen output-token limit")
    if receipt.max_tokens_field != expected_max_tokens_field:
        raise ValueError("provider receipt disagrees with the frozen output-token field")
    if receipt.seed_supplied:
        raise ValueError("provider receipt reports a forbidden seed control")
    if receipt.cache_config_supplied:
        raise ValueError("provider receipt reports a forbidden cache control")
    if provider_config.kind is ProviderKind.BEDROCK:
        if receipt.response_id is not None:
            raise ValueError("Bedrock provider receipt must not contain a response id")
        if receipt.response_model is not None or receipt.system_fingerprint is not None:
            raise ValueError("Bedrock provider receipt contains unsupported response metadata")
        if frozen_response_identity is not None:
            _validate_provider_response_identity(receipt, frozen_response_identity)
        return
    if receipt.response_id is None or receipt.response_model is None:
        raise ValueError("OpenAI-compatible provider receipt is missing response identity")
    if receipt.provider_request_id == receipt.response_id:
        raise ValueError("provider request identity was conflated with the response identity")
    if frozen_response_identity is not None:
        _validate_provider_response_identity(receipt, frozen_response_identity)


def _validate_provider_response_identity(
    receipt: ChatProviderReceipt,
    response_identity: ProviderResponseIdentity,
) -> None:
    """Require provider-reported routing evidence to match its frozen scored route."""
    if (
        receipt.response_model != response_identity.response_model
        or receipt.system_fingerprint != response_identity.system_fingerprint
    ):
        raise ValueError("provider receipt differs from the frozen response identity")

"""Sanitized, content-addressed provider call receipts for evaluation traces."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence

from llm_waterfall import ChatProviderReceipt

from wmh.core.types import JsonObject


def build_chat_provider_receipt(
    *,
    provider: str,
    provider_request_id: str,
    response_id: str | None,
    requested_model: str,
    response_model: str | None,
    system_fingerprint: str | None,
    request_payload: JsonObject,
    temperature: float | None,
    max_tokens: int,
    max_tokens_field: str,
    started_at_unix_s: float,
    finished_at_unix_s: float | None = None,
) -> ChatProviderReceipt:
    """Build a strict receipt without retaining prompt, tool, or credential material."""
    canonical = json.dumps(
        request_payload,
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


def _seed_supplied(request_payload: JsonObject) -> bool:
    if "seed" in request_payload:
        return True
    additional = request_payload.get("additionalModelRequestFields")
    return isinstance(additional, Mapping) and "seed" in additional


def _cache_config_supplied(provider: str, request_payload: JsonObject) -> bool:
    """Detect provider cache controls only in their documented wire locations."""
    if provider in {"openai", "azure"}:
        return any(
            field in request_payload
            for field in ("prompt_cache_key", "prompt_cache_retention", "cache_control")
        )
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
        additional = request_payload.get("additionalModelRequestFields")
        return isinstance(additional, Mapping) and any(
            field in additional
            for field in ("prompt_cache_key", "prompt_cache_retention", "cache_control")
        )
    return False


def _bedrock_content_has_cache_point(value: object) -> bool:
    """Return whether one Bedrock content list contains a cache-point block."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    return any(isinstance(block, Mapping) and "cachePoint" in block for block in value)

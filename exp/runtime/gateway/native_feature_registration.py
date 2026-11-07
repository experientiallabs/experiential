"""Admission-only exports of sensitive authority for native response features."""

from __future__ import annotations

import base64
from dataclasses import fields

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.native_owner import output_registration
from exp.runtime.gateway.native_cache_registration import cache_registration
from exp.runtime.gateway.native_continuation_registration import continuation_registration
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_search_registration import search_registration
from exp.runtime.gateway.reasoning_carrier import reasoning_history_sha256
from exp.runtime.models.providers.protocol import NativeGatewayDispatchSigner


def feature_registration(entry: InflightRequest) -> JsonObject:
    """Export request-owned authority without exposing it to the HTTP wire or receipts."""
    signing: list[JsonObject | None] = []
    reasoning: list[JsonObject | None] = []
    for depth, _ in enumerate(entry.route.deployments):
        signer = entry.signers[depth] if depth < len(entry.signers) else None
        binding = entry.dispatch_bindings[depth] if depth < len(entry.dispatch_bindings) else None
        if signer is None:
            signing.append(None)
        elif binding is None or not isinstance(signer, NativeGatewayDispatchSigner):
            message = "The admitted body signer does not export native signing authority"
            raise ValueError(message)
        else:
            signing.append(
                {
                    "url": binding.url,
                    "body_sha256": binding.body_sha256,
                    "authority": signer.native_gateway_signing(url=binding.url),
                }
            )
        authority = (
            entry.reasoning_carrier_authorities[depth]
            if depth < len(entry.reasoning_carrier_authorities)
            else None
        )
        if authority is None:
            reasoning.append(None)
            continue
        if not isinstance(entry.request, GatewayRequest):
            message = "Reasoning carriers require a message-bearing request"
            raise ValueError(message)
        claims: JsonObject = {"schema_version": 3}
        for field in fields(authority):
            if field.name not in {"aead_key", "scheme"}:
                claims[field.name] = getattr(authority, field.name)
        reasoning.append(
            {
                "claims": claims,
                "key": base64.b64encode(authority.aead_key).decode("ascii"),
                "prefix": authority.scheme.prefix,
                "history_sha256": reasoning_history_sha256(entry.request.messages),
            }
        )
    return {
        "signing": signing,
        "reasoning": reasoning,
        "continuation": continuation_registration(entry.continuation),
        "output_checks": output_registration(entry.guardrails),
        "search": search_registration(entry),
        "cache": cache_registration(entry),
    }

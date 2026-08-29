"""Authenticated Fireworks reasoning carriers bound to exact gateway authority."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import Field, ValidationError

from exp.common.core.artifacts import (
    ContractModel,
    JsonValue,
    Sha256,
    canonical_json_bytes,
    sha256_json,
)
from exp.common.models import ToolCall
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayMessage,
    OpaqueReasoningContentBlock,
    SealedReasoningContentBlock,
)
from exp.runtime.models.providers.base import GatewayWireProfile

FIREWORKS_REASONING_CONTENT_PREFIX = "x-experiential-fireworks-reasoning-v2:"
"""Public marker for a gateway-authenticated, caller-opaque Fireworks carrier."""

MAXIMUM_REASONING_CONTENT_BYTES = 8 * 1024 * 1024
"""Maximum decrypted provider reasoning retained for one tool continuation."""

_MAXIMUM_CLAIMS_BYTES = MAXIMUM_REASONING_CONTENT_BYTES + 32 * 1024
_MAXIMUM_ENVELOPE_BYTES = 12 + _MAXIMUM_CLAIMS_BYTES + 16
MAXIMUM_REASONING_CARRIER_BYTES = 4 * ((_MAXIMUM_ENVELOPE_BYTES + 2) // 3) + 512
"""Maximum caller-supplied carrier size before any base64 or AEAD work."""

_NONCE_BYTES = 12
_KEY_DERIVATION_DOMAIN = b"experiential/fireworks-reasoning-carrier/aes256gcm/v2\0"
_CREDENTIAL_IDENTITY_DOMAIN = b"experiential/fireworks-reasoning-credential/v2\0"


class ReasoningCarrierClaims(ContractModel):
    """Authenticated plaintext held only while issuing or validating a carrier."""

    schema_version: Literal[2] = 2
    organization_id: str = Field(min_length=1, max_length=256)
    identity_id: str = Field(min_length=1, max_length=256)
    virtual_key_id: str = Field(min_length=1, max_length=256)
    alias: str = Field(min_length=1, max_length=256)
    alias_revision_id: str = Field(min_length=1, max_length=256)
    catalog_sha256: Sha256
    exact_model_id: str = Field(min_length=1, max_length=256)
    pool_id: str = Field(min_length=1, max_length=256)
    deployment_id: str = Field(min_length=1, max_length=256)
    source_alias: str = Field(min_length=1, max_length=256)
    provider: str = Field(min_length=1, max_length=128)
    provider_model: str = Field(min_length=1, max_length=2_048)
    model_revision: str | None = Field(default=None, max_length=256)
    connection_id: str = Field(min_length=1, max_length=256)
    connection_authority_sha256: Sha256
    credential_identity_sha256: Sha256
    reasoning_route_sha256: Sha256
    issuing_request_id: str = Field(min_length=1, max_length=256)
    issuing_route_depth: int = Field(ge=0)
    issuing_history_sha256: Sha256
    issuing_turn_sha256: Sha256
    tool_call_ids: tuple[str, ...] = Field(min_length=1)
    content: str = Field(min_length=1, max_length=MAXIMUM_REASONING_CONTENT_BYTES)


@dataclass(frozen=True)
class ReasoningCarrierAuthority:
    """Exact route authority and ephemeral AEAD key for one Fireworks rung."""

    organization_id: str
    identity_id: str
    virtual_key_id: str
    alias: str
    alias_revision_id: str
    catalog_sha256: str
    exact_model_id: str
    pool_id: str
    deployment_id: str
    source_alias: str
    provider: str
    provider_model: str
    model_revision: str | None
    connection_id: str
    connection_authority_sha256: str
    credential_identity_sha256: str
    reasoning_route_sha256: str
    aead_key: bytes = field(repr=False)


def parse_reasoning_content_carrier(value: str) -> SealedReasoningContentBlock:
    """Validate one carrier's bounded public envelope without decrypting it."""
    raw = _ascii_bytes(value)
    if len(raw) > MAXIMUM_REASONING_CARRIER_BYTES or not value.startswith(
        FIREWORKS_REASONING_CONTENT_PREFIX
    ):
        raise ValueError("reasoning_content is not a bounded gateway carrier")
    encoded = value.removeprefix(FIREWORKS_REASONING_CONTENT_PREFIX)
    deployment_text, separator, envelope_text = encoded.partition(":")
    if not separator or not deployment_text or not envelope_text or ":" in envelope_text:
        raise ValueError("reasoning_content is not a complete gateway carrier")
    deployment_bytes = _decode_urlsafe(deployment_text, maximum_bytes=256)
    try:
        deployment_hint = deployment_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("reasoning_content has an invalid deployment hint") from exc
    if not deployment_hint or len(deployment_hint) > 256:
        raise ValueError("reasoning_content has an invalid deployment hint")
    _decode_urlsafe(envelope_text, maximum_bytes=_MAXIMUM_ENVELOPE_BYTES)
    return SealedReasoningContentBlock(carrier=value, deployment_hint=deployment_hint)


def reasoning_carrier_authority(
    *,
    authorization: AuthorizationSnapshot,
    exact_model_id: str,
    pool_id: str,
    deployment: ExactModelDeployment,
    profile: GatewayWireProfile,
) -> ReasoningCarrierAuthority | None:
    """Derive one replica-stable Fireworks authority from resolved credential material."""
    route_sha256 = profile.fireworks_reasoning_route_sha256
    if route_sha256 is None:
        return None
    credential = _bearer_credential(profile.headers)
    key_context = {
        "schema_version": 2,
        "organization_id": authorization.organization_id,
        "identity_id": authorization.identity_id,
        "virtual_key_id": authorization.virtual_key_id,
        "alias": authorization.alias,
        "alias_revision_id": authorization.alias_revision_id,
        "catalog_sha256": authorization.catalog_sha256,
        "exact_model_id": exact_model_id,
        "pool_id": pool_id,
        "deployment_id": deployment.deployment_id,
        "source_alias": deployment.source_alias,
        "provider": deployment.provider,
        "provider_model": deployment.provider_model,
        "model_revision": deployment.revision,
        "connection_id": deployment.connection,
        "connection_authority_sha256": deployment.connection_sha256,
        "reasoning_route_sha256": route_sha256,
    }
    aead_key = hmac.new(
        credential,
        _KEY_DERIVATION_DOMAIN + canonical_json_bytes(key_context),
        hashlib.sha256,
    ).digest()
    credential_identity = hmac.new(
        aead_key,
        _CREDENTIAL_IDENTITY_DOMAIN,
        hashlib.sha256,
    ).hexdigest()
    return ReasoningCarrierAuthority(
        organization_id=authorization.organization_id,
        identity_id=authorization.identity_id,
        virtual_key_id=authorization.virtual_key_id,
        alias=authorization.alias,
        alias_revision_id=authorization.alias_revision_id,
        catalog_sha256=authorization.catalog_sha256,
        exact_model_id=exact_model_id,
        pool_id=pool_id,
        deployment_id=deployment.deployment_id,
        source_alias=deployment.source_alias,
        provider=deployment.provider,
        provider_model=deployment.provider_model,
        model_revision=deployment.revision,
        connection_id=deployment.connection,
        connection_authority_sha256=deployment.connection_sha256,
        credential_identity_sha256=credential_identity,
        reasoning_route_sha256=route_sha256,
        aead_key=aead_key,
    )


def seal_reasoning_content(
    authority: ReasoningCarrierAuthority,
    *,
    issuing_request_id: str,
    issuing_route_depth: int,
    issuing_history_sha256: Sha256,
    assistant_content: str | None,
    tool_calls: tuple[ToolCall, ...],
    content: str,
) -> str:
    """Seal provider reasoning for one exact issuing turn and waterfall rung."""
    tool_call_ids = _require_tool_calls(tool_calls)
    if issuing_route_depth < 0 or not issuing_request_id:
        raise ValueError("reasoning carrier issuing identity is invalid")
    if not content or len(content.encode("utf-8")) > MAXIMUM_REASONING_CONTENT_BYTES:
        raise ValueError("reasoning content exceeds the authenticated carrier bound")
    claims = ReasoningCarrierClaims.model_validate(
        {
            **_authority_claims(authority),
            "issuing_request_id": issuing_request_id,
            "issuing_route_depth": issuing_route_depth,
            "issuing_history_sha256": issuing_history_sha256,
            "issuing_turn_sha256": _issuing_turn_sha256(assistant_content, tool_calls),
            "tool_call_ids": tool_call_ids,
            "content": content,
        }
    )
    plaintext = canonical_json_bytes(claims)
    if len(plaintext) > _MAXIMUM_CLAIMS_BYTES:
        raise ValueError("reasoning carrier claims exceed the authenticated bound")
    deployment_text = _encode_urlsafe(authority.deployment_id.encode("utf-8"))
    associated_data = _associated_data(deployment_text)
    nonce = secrets.token_bytes(_NONCE_BYTES)
    envelope = nonce + AESGCM(authority.aead_key).encrypt(nonce, plaintext, associated_data)
    carrier = f"{FIREWORKS_REASONING_CONTENT_PREFIX}{deployment_text}:{_encode_urlsafe(envelope)}"
    if len(carrier.encode("ascii")) > MAXIMUM_REASONING_CARRIER_BYTES:
        raise ValueError("reasoning carrier exceeds the public envelope bound")
    return carrier


def unseal_reasoning_content(
    block: SealedReasoningContentBlock,
    authority: ReasoningCarrierAuthority,
    *,
    assistant_content: str | None,
    tool_calls: tuple[ToolCall, ...],
    history_prefix: tuple[GatewayMessage, ...] = (),
) -> tuple[OpaqueReasoningContentBlock, ReasoningCarrierClaims]:
    """Authenticate and decrypt one carrier against current exact authority."""
    tool_call_ids = _require_tool_calls(tool_calls)
    parsed = parse_reasoning_content_carrier(block.carrier)
    if (
        parsed.deployment_hint != block.deployment_hint
        or block.deployment_hint != authority.deployment_id
    ):
        raise ValueError("reasoning carrier deployment differs from current authority")
    encoded = block.carrier.removeprefix(FIREWORKS_REASONING_CONTENT_PREFIX)
    deployment_text, _separator, envelope_text = encoded.partition(":")
    envelope = _decode_urlsafe(envelope_text, maximum_bytes=_MAXIMUM_ENVELOPE_BYTES)
    if len(envelope) <= _NONCE_BYTES + 16:
        raise ValueError("reasoning carrier envelope is incomplete")
    nonce = envelope[:_NONCE_BYTES]
    ciphertext = envelope[_NONCE_BYTES:]
    try:
        plaintext = AESGCM(authority.aead_key).decrypt(
            nonce,
            ciphertext,
            _associated_data(deployment_text),
        )
    except InvalidTag as exc:
        raise ValueError("reasoning carrier authentication failed") from exc
    if len(plaintext) > _MAXIMUM_CLAIMS_BYTES:
        raise ValueError("reasoning carrier claims exceed the authenticated bound")
    try:
        claims = ReasoningCarrierClaims.model_validate(json.loads(plaintext))
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
        raise ValueError("reasoning carrier claims are invalid") from exc
    expected = _authority_claims(authority)
    actual = claims.model_dump(
        mode="json",
        exclude={
            "issuing_request_id",
            "issuing_route_depth",
            "issuing_history_sha256",
            "issuing_turn_sha256",
            "tool_call_ids",
            "content",
        },
    )
    if (
        actual != expected
        or claims.tool_call_ids != tool_call_ids
        or claims.issuing_turn_sha256 != _issuing_turn_sha256(assistant_content, tool_calls)
        or (
            history_prefix
            and claims.issuing_history_sha256 != reasoning_history_sha256(history_prefix)
        )
    ):
        raise ValueError("reasoning carrier authority or tool-call identity changed")
    if len(claims.content.encode("utf-8")) > MAXIMUM_REASONING_CONTENT_BYTES:
        raise ValueError("reasoning content exceeds the authenticated carrier bound")
    return (
        OpaqueReasoningContentBlock(
            route_sha256=claims.reasoning_route_sha256,
            content=claims.content,
            carrier_size_bytes=len(block.carrier.encode("ascii")),
        ),
        claims,
    )


def reasoning_history_sha256(messages: tuple[GatewayMessage, ...]) -> Sha256:
    """Digest visible history plus authenticated provider reasoning state.

    ``GatewayMessage`` excludes provider reasoning from ordinary public request and
    artifact serialization. Carrier chaining is an internal authority boundary, so it
    deliberately adds the normalized blocks back before hashing. A later carrier then
    cannot be transplanted across two visually identical turns with different hidden state.

    The digest covers the conversation a caller can observe, so absent and empty text are
    the same history: a client that echoes the empty string the gateway streamed for a
    tool-only turn continues the same conversation as one that echoes null.
    """
    payload: list[dict[str, object]] = []
    for message in messages:
        item = message.model_dump(mode="json")
        item["content"] = _normalized_content(message.content)
        item["provider_reasoning"] = [
            block.model_dump(mode="json") for block in message.provider_reasoning
        ]
        payload.append(item)
    return sha256_json(payload)


def parse_reasoning_carrier_tool_calls(value: object) -> tuple[ToolCall, ...]:
    """Parse the native sealer's exact completed tool-call identities."""
    if not isinstance(value, list):
        raise ValueError("reasoning carrier tool calls must be an array")
    tool_calls: list[ToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("reasoning carrier tool call must be an object")
        call_id = item.get("call_id")
        name = item.get("name")
        raw_arguments = item.get("raw_arguments")
        if (
            not isinstance(call_id, str)
            or not isinstance(name, str)
            or not isinstance(raw_arguments, str)
        ):
            raise ValueError("reasoning carrier tool call has invalid field types")
        arguments = json.loads(raw_arguments)
        if not isinstance(arguments, dict):
            raise ValueError("reasoning carrier tool arguments must be an object")
        tool_calls.append(
            ToolCall(
                call_id=call_id,
                name=name,
                arguments=arguments,
                raw_arguments=raw_arguments,
            )
        )
    return tuple(tool_calls)


def _authority_claims(authority: ReasoningCarrierAuthority) -> dict[str, object]:
    """Return the canonical claim fields shared by issuance and verification."""
    return {
        "schema_version": 2,
        "organization_id": authority.organization_id,
        "identity_id": authority.identity_id,
        "virtual_key_id": authority.virtual_key_id,
        "alias": authority.alias,
        "alias_revision_id": authority.alias_revision_id,
        "catalog_sha256": authority.catalog_sha256,
        "exact_model_id": authority.exact_model_id,
        "pool_id": authority.pool_id,
        "deployment_id": authority.deployment_id,
        "source_alias": authority.source_alias,
        "provider": authority.provider,
        "provider_model": authority.provider_model,
        "model_revision": authority.model_revision,
        "connection_id": authority.connection_id,
        "connection_authority_sha256": authority.connection_authority_sha256,
        "credential_identity_sha256": authority.credential_identity_sha256,
        "reasoning_route_sha256": authority.reasoning_route_sha256,
    }


def _bearer_credential(headers: Mapping[str, str]) -> bytes:
    """Return one Fireworks bearer value without retaining its public prefix."""
    values = [value for name, value in headers.items() if name.lower() == "authorization"]
    if len(values) != 1 or not isinstance(values[0], str) or not values[0].startswith("Bearer "):
        raise ValueError("Fireworks reasoning authority requires one bearer credential")
    value = values[0].removeprefix("Bearer ").encode("utf-8")
    if not value:
        raise ValueError("Fireworks reasoning authority requires one bearer credential")
    return value


def _require_tool_calls(tool_calls: tuple[ToolCall, ...]) -> tuple[str, ...]:
    """Require one ordered, unique tool-call set and return its IDs."""
    tool_call_ids = tuple(call.call_id for call in tool_calls)
    if not tool_call_ids or len(set(tool_call_ids)) != len(tool_call_ids):
        raise ValueError("reasoning carrier tool-call IDs must be non-empty and unique")
    return tool_call_ids


def _issuing_turn_sha256(
    assistant_content: str | None,
    tool_calls: tuple[ToolCall, ...],
) -> Sha256:
    """Bind visible assistant text and every semantic tool-call field.

    The turn is identified by what the caller can observe, not by one exact byte
    encoding of it. Assistant text is compared with absent text equal to empty text,
    and tool arguments are compared as canonical JSON values, because an
    OpenAI-compatible client parses the streamed arguments and re-encodes them on the
    next turn. Argument values, names, order, and call identity all still bind.
    """
    return sha256_json(
        {
            "assistant_content": _normalized_content(assistant_content),
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments_sha256": sha256_json(_normalized_arguments(call.arguments)),
                }
                for call in tool_calls
            ],
        }
    )


def _normalized_content(content: str | None) -> str | None:
    """Return absent text for the empty string a tool-only turn may carry."""
    return content or None


def _normalized_arguments(value: JsonValue) -> JsonValue:
    """Return one JSON value whose numbers survive a client parse and re-encode.

    A JSON number carries no type, so a client that parses ``1.0`` and serializes the
    same value as ``1`` sent the same arguments. Every integral number therefore digests
    as an integer, leaving strings, booleans, null, and fractional numbers untouched.
    """
    if isinstance(value, dict):
        return {key: _normalized_arguments(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized_arguments(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _associated_data(deployment_text: str) -> bytes:
    """Bind the public routing hint and carrier version against retagging."""
    return f"{FIREWORKS_REASONING_CONTENT_PREFIX}{deployment_text}".encode("ascii")


def _ascii_bytes(value: str) -> bytes:
    """Encode one public envelope without accepting Unicode lookalikes."""
    try:
        return value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("reasoning_content carrier must be ASCII") from exc


def _encode_urlsafe(value: bytes) -> str:
    """Return unpadded URL-safe base64."""
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_urlsafe(value: str, *, maximum_bytes: int) -> bytes:
    """Decode URL-safe base64 only after bounding its maximum output size."""
    if not value or len(value) > 4 * ((maximum_bytes + 2) // 3):
        raise ValueError("reasoning carrier component exceeds its decode bound")
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("reasoning carrier component is not valid base64") from exc
    if len(decoded) > maximum_bytes or _encode_urlsafe(decoded) != value:
        raise ValueError("reasoning carrier component is non-canonical or exceeds its bound")
    return decoded

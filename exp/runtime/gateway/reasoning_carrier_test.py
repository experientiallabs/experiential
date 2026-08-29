"""Authenticated Fireworks reasoning carrier security regressions."""

from __future__ import annotations

import base64
import json
from dataclasses import replace

import pytest

from exp.common.models import ToolCall
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    OpaqueReasoningContentBlock,
)
from exp.runtime.gateway.reasoning_carrier import (
    FIREWORKS_REASONING_CONTENT_PREFIX,
    MAXIMUM_REASONING_CARRIER_BYTES,
    ReasoningCarrierAuthority,
    parse_reasoning_content_carrier,
    reasoning_carrier_authority,
    reasoning_history_sha256,
    seal_reasoning_content,
    unseal_reasoning_content,
)
from exp.runtime.models.providers.base import GatewayWireProfile

_ROUTE_SHA256 = "a" * 64
_HISTORY_SHA256 = "e" * 64


def _tool_calls(
    *call_ids: str,
    name: str = "lookup",
    raw_arguments: str = '{"q":"x"}',
) -> tuple[ToolCall, ...]:
    """Build byte-exact tool identities echoed by one assistant turn."""
    arguments = json.loads(raw_arguments)
    assert isinstance(arguments, dict)
    return tuple(
        ToolCall(
            call_id=call_id,
            name=name,
            arguments=arguments,
            raw_arguments=raw_arguments,
        )
        for call_id in call_ids
    )


def _authorization(**updates: object) -> AuthorizationSnapshot:
    """Build one exact tenant and alias authority."""
    values: dict[str, object] = {
        "request_id": "request-current",
        "organization_id": "organization-one",
        "identity_id": "identity-one",
        "virtual_key_id": "key-one",
        "alias": "deepseek-v4-flash",
        "alias_revision_id": "alias-revision-one",
        "target": DirectTarget(pool_id="pool-one"),
        "surface": GatewayApiSurface.CHAT_COMPLETIONS,
        "catalog_sha256": "b" * 64,
        "canonical_request_sha256": "c" * 64,
        "deadline_monotonic": 1.0,
    }
    values.update(updates)
    return AuthorizationSnapshot.model_validate(values)


def _deployment(**updates: object) -> ExactModelDeployment:
    """Build one exact Fireworks deployment identity."""
    values: dict[str, object] = {
        "deployment_id": "fireworks-rung-one",
        "source_alias": "fireworks-source",
        "exact_model_id": "deepseek-exact",
        "connection": "fireworks-connection",
        "provider": "fireworks",
        "provider_model": "accounts/fireworks/models/deepseek-v4-flash-0731",
        "revision": "provider-revision-one",
        "connection_sha256": "d" * 64,
        "capabilities_sha256": "e" * 64,
    }
    values.update(updates)
    return ExactModelDeployment.model_validate(values)


def _authority(
    *,
    credential: str = "replica-shared-provider-secret",
    authorization: AuthorizationSnapshot | None = None,
    deployment: ExactModelDeployment | None = None,
    route_sha256: str = _ROUTE_SHA256,
) -> ReasoningCarrierAuthority:
    """Derive one carrier authority from route and credential material."""
    authority = reasoning_carrier_authority(
        authorization=authorization or _authorization(),
        exact_model_id="deepseek-exact",
        pool_id="pool-one",
        deployment=deployment or _deployment(),
        profile=GatewayWireProfile(
            dialect="openai_compatible",
            url="https://api.fireworks.ai/inference/v1/chat/completions",
            headers={"Authorization": f"Bearer {credential}"},
            model_id="accounts/fireworks/models/deepseek-v4-flash-0731",
            fireworks_reasoning_route_sha256=route_sha256,
        ),
    )
    assert authority is not None
    return authority


def test_carrier_round_trip_is_opaque_and_cross_replica_stable() -> None:
    """Replicas sharing exact authority decrypt one byte-exact tool turn."""
    issuer = _authority()
    verifier = _authority()
    content = "private provider reasoning with tenant data"
    raw_arguments = '{ "q" : "x" }'
    carrier = seal_reasoning_content(
        issuer,
        issuing_request_id="issuing-request",
        issuing_route_depth=2,
        issuing_history_sha256=_HISTORY_SHA256,
        assistant_content="visible assistant text",
        tool_calls=_tool_calls("call-one", raw_arguments=raw_arguments),
        content=content,
    )

    assert content not in carrier
    assert issuer.reasoning_route_sha256 not in carrier
    assert "replica-shared-provider-secret" not in carrier
    assert issuer.aead_key.hex() not in repr(issuer)
    block, claims = unseal_reasoning_content(
        parse_reasoning_content_carrier(carrier),
        verifier,
        assistant_content="visible assistant text",
        tool_calls=_tool_calls("call-one", raw_arguments=raw_arguments),
    )

    assert block.content == content
    assert block.carrier_size_bytes == len(carrier)
    assert claims.issuing_request_id == "issuing-request"
    assert claims.issuing_route_depth == 2
    assert claims.issuing_history_sha256 == _HISTORY_SHA256


@pytest.mark.parametrize(
    "changed",
    (
        replace(_authority(), organization_id="organization-two"),
        replace(_authority(), identity_id="identity-two"),
        replace(_authority(), virtual_key_id="key-two"),
        replace(_authority(), alias="other-alias"),
        replace(_authority(), alias_revision_id="alias-revision-two"),
        replace(_authority(), deployment_id="fireworks-rung-two"),
        replace(_authority(), connection_authority_sha256="f" * 64),
        _authority(credential="rotated-provider-secret"),
        _authority(route_sha256="f" * 64),
    ),
)
def test_carrier_rejects_cross_route_or_credential_authority(
    changed: ReasoningCarrierAuthority,
) -> None:
    """Tenant, deployment, route, connection, and credential changes fail closed."""
    carrier = seal_reasoning_content(
        _authority(),
        issuing_request_id="issuing-request",
        issuing_route_depth=0,
        issuing_history_sha256=_HISTORY_SHA256,
        assistant_content=None,
        tool_calls=_tool_calls("call-one"),
        content="hidden",
    )
    with pytest.raises(ValueError):
        unseal_reasoning_content(
            parse_reasoning_content_carrier(carrier),
            changed,
            assistant_content=None,
            tool_calls=_tool_calls("call-one"),
        )


def test_carrier_rejects_tampering_retagging_and_turn_changes() -> None:
    """Envelope edits, public retagging, and turn changes cannot authenticate."""
    authority = _authority()
    carrier = seal_reasoning_content(
        authority,
        issuing_request_id="issuing-request",
        issuing_route_depth=0,
        issuing_history_sha256=_HISTORY_SHA256,
        assistant_content="visible assistant text",
        tool_calls=_tool_calls("call-one", raw_arguments='{ "q" : "x" }'),
        content="hidden",
    )
    parsed = parse_reasoning_content_carrier(carrier)
    deployment, envelope = carrier.removeprefix(FIREWORKS_REASONING_CONTENT_PREFIX).split(":", 1)
    tampered_envelope = ("A" if envelope[0] != "A" else "B") + envelope[1:]
    tampered = f"{FIREWORKS_REASONING_CONTENT_PREFIX}{deployment}:{tampered_envelope}"
    retagged_deployment = base64.urlsafe_b64encode(b"fireworks-rung-two").rstrip(b"=").decode()
    retagged = f"{FIREWORKS_REASONING_CONTENT_PREFIX}{retagged_deployment}:{envelope}"

    for candidate in (tampered, retagged):
        with pytest.raises(ValueError):
            unseal_reasoning_content(
                parse_reasoning_content_carrier(candidate),
                authority,
                assistant_content="visible assistant text",
                tool_calls=_tool_calls("call-one", raw_arguments='{ "q" : "x" }'),
            )
    with pytest.raises(ValueError):
        unseal_reasoning_content(
            parsed,
            authority,
            assistant_content="changed assistant text",
            tool_calls=_tool_calls("call-one", raw_arguments='{ "q" : "x" }'),
        )
    with pytest.raises(ValueError):
        unseal_reasoning_content(
            parsed,
            authority,
            assistant_content="visible assistant text",
            tool_calls=_tool_calls("call-one", raw_arguments='{"q":"y"}'),
        )


def test_carrier_authenticates_a_client_reencoding_of_the_same_turn() -> None:
    """A caller that reparses the streamed turn continues it without byte equality."""
    authority = _authority()
    carrier = seal_reasoning_content(
        authority,
        issuing_request_id="issuing-request",
        issuing_route_depth=0,
        issuing_history_sha256=_HISTORY_SHA256,
        assistant_content="",
        tool_calls=_tool_calls("call-one", raw_arguments='{"q": "x", "n": 1}'),
        content="hidden",
    )
    block, _claims = unseal_reasoning_content(
        parse_reasoning_content_carrier(carrier),
        authority,
        assistant_content=None,
        tool_calls=_tool_calls("call-one", raw_arguments='{"n":1,"q":"x"}'),
    )
    assert block.content == "hidden"


def test_carrier_decode_is_bounded_before_base64_or_aead_work() -> None:
    """Malformed and oversized public envelopes fail at the bounded parser."""
    with pytest.raises(ValueError):
        parse_reasoning_content_carrier("raw provider reasoning")
    with pytest.raises(ValueError):
        parse_reasoning_content_carrier("x" * (MAXIMUM_REASONING_CARRIER_BYTES + 1))
    carrier = seal_reasoning_content(
        _authority(),
        issuing_request_id="issuing-request",
        issuing_route_depth=0,
        issuing_history_sha256=_HISTORY_SHA256,
        assistant_content=None,
        tool_calls=_tool_calls("call-one"),
        content="hidden",
    )
    with pytest.raises(ValueError):
        parse_reasoning_content_carrier(f"{carrier}=")


def test_carrier_rejects_a_different_nonempty_conversation_prefix() -> None:
    """A valid turn carrier cannot be transplanted beneath another user prompt."""
    authority = _authority()
    issuing_history = (GatewayMessage(role="user", content="original prompt"),)
    carrier = seal_reasoning_content(
        authority,
        issuing_request_id="issuing-request",
        issuing_route_depth=0,
        issuing_history_sha256=reasoning_history_sha256(issuing_history),
        assistant_content=None,
        tool_calls=_tool_calls("call-one"),
        content="hidden",
    )

    with pytest.raises(ValueError, match="authority or tool-call identity changed"):
        unseal_reasoning_content(
            parse_reasoning_content_carrier(carrier),
            authority,
            assistant_content=None,
            tool_calls=_tool_calls("call-one"),
            history_prefix=(GatewayMessage(role="user", content="different prompt"),),
        )


def test_history_digest_treats_empty_and_absent_text_as_one_conversation() -> None:
    """A caller echoing the empty string of a tool-only turn keeps the same prefix."""
    tool_only = GatewayMessage(
        role="assistant",
        content="",
        tool_calls=_tool_calls("call-one"),
    )
    assert reasoning_history_sha256((tool_only,)) == reasoning_history_sha256(
        (tool_only.model_copy(update={"content": None}),)
    )


def test_later_carrier_binds_earlier_authenticated_reasoning() -> None:
    """Visually identical prefixes with different hidden reasoning cannot be interchanged."""
    authority = _authority()
    visible = GatewayMessage(role="user", content="same prompt")
    first_prefix = (
        visible,
        GatewayMessage(
            role="assistant",
            content="same answer",
            provider_reasoning=(
                OpaqueReasoningContentBlock(
                    route_sha256=_ROUTE_SHA256,
                    content="first hidden state",
                ),
            ),
        ),
    )
    second_prefix = (
        visible,
        first_prefix[1].model_copy(
            update={
                "provider_reasoning": (
                    OpaqueReasoningContentBlock(
                        route_sha256=_ROUTE_SHA256,
                        content="second hidden state",
                    ),
                )
            }
        ),
    )
    assert reasoning_history_sha256(first_prefix) != reasoning_history_sha256(second_prefix)
    carrier = seal_reasoning_content(
        authority,
        issuing_request_id="later-request",
        issuing_route_depth=0,
        issuing_history_sha256=reasoning_history_sha256(first_prefix),
        assistant_content=None,
        tool_calls=_tool_calls("call-later"),
        content="later hidden state",
    )

    with pytest.raises(ValueError, match="authority or tool-call identity changed"):
        unseal_reasoning_content(
            parse_reasoning_content_carrier(carrier),
            authority,
            assistant_content=None,
            tool_calls=_tool_calls("call-later"),
            history_prefix=second_prefix,
        )

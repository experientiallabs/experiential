"""Authenticated Fireworks reasoning carrier security and rotation tests."""

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
)
from exp.runtime.gateway.reasoning_carrier import (
    FIREWORKS_REASONING_CONTENT_PREFIX,
    MAXIMUM_REASONING_CARRIER_BYTES,
    ReasoningCarrierAuthority,
    parse_reasoning_content_carrier,
    reasoning_carrier_authority,
    seal_reasoning_content,
    unseal_reasoning_content,
)
from exp.runtime.models.providers.base import GatewayWireProfile

_ROUTE_SHA256 = "a" * 64


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
) -> ReasoningCarrierAuthority:
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
            fireworks_reasoning_route_sha256=_ROUTE_SHA256,
        ),
    )
    assert authority is not None
    return authority


def test_carrier_is_opaque_and_cross_replica_stable() -> None:
    """Replicas sharing exact credential authority can validate one opaque turn."""
    issuer = _authority()
    verifier = _authority()
    content = "private provider reasoning with tenant data"
    carrier = seal_reasoning_content(
        issuer,
        issuing_request_id="issuing-request",
        issuing_route_depth=2,
        assistant_content="visible assistant text",
        tool_calls=_tool_calls("call-one", "call-two"),
        content=content,
    )

    assert content not in carrier
    assert issuer.reasoning_route_sha256 not in carrier
    assert "replica-shared-provider-secret" not in carrier
    assert issuer.aead_key.hex() not in repr(issuer)
    parsed = parse_reasoning_content_carrier(carrier)
    block, claims = unseal_reasoning_content(
        parsed,
        verifier,
        assistant_content="visible assistant text",
        tool_calls=_tool_calls("call-one", "call-two"),
    )

    assert block.content == content
    assert block.carrier_size_bytes == len(carrier)
    assert claims.issuing_request_id == "issuing-request"
    assert claims.issuing_route_depth == 2


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
    ),
)
def test_carrier_rejects_authority_or_credential_rotation(
    changed: ReasoningCarrierAuthority,
) -> None:
    """Tenant, route, connection, and credential rotation invalidate carriers closed."""
    carrier = seal_reasoning_content(
        _authority(),
        issuing_request_id="issuing-request",
        issuing_route_depth=0,
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


def test_carrier_rejects_tampering_retagging_and_changed_tool_identity() -> None:
    """Caller edits and tool-call replay cannot authenticate."""
    authority = _authority()
    carrier = seal_reasoning_content(
        authority,
        issuing_request_id="issuing-request",
        issuing_route_depth=0,
        assistant_content="visible assistant text",
        tool_calls=_tool_calls("call-one"),
        content="hidden",
    )
    parsed = parse_reasoning_content_carrier(carrier)
    deployment, envelope = carrier.removeprefix(FIREWORKS_REASONING_CONTENT_PREFIX).split(":", 1)
    tampered_envelope = ("A" if envelope[0] != "A" else "B") + envelope[1:]
    tampered = f"{FIREWORKS_REASONING_CONTENT_PREFIX}{deployment}:{tampered_envelope}"
    retagged_deployment = base64.urlsafe_b64encode(b"fireworks-rung-two").rstrip(b"=").decode()
    retagged = f"{FIREWORKS_REASONING_CONTENT_PREFIX}{retagged_deployment}:{envelope}"

    with pytest.raises(ValueError):
        unseal_reasoning_content(
            parse_reasoning_content_carrier(tampered),
            authority,
            assistant_content="visible assistant text",
            tool_calls=_tool_calls("call-one"),
        )
    with pytest.raises(ValueError):
        unseal_reasoning_content(
            parsed,
            authority,
            assistant_content="visible assistant text",
            tool_calls=_tool_calls("call-two"),
        )
    with pytest.raises(ValueError):
        unseal_reasoning_content(
            parsed,
            authority,
            assistant_content="changed assistant text",
            tool_calls=_tool_calls("call-one"),
        )
    with pytest.raises(ValueError):
        unseal_reasoning_content(
            parsed,
            authority,
            assistant_content="visible assistant text",
            tool_calls=_tool_calls("call-one", name="changed-name"),
        )
    with pytest.raises(ValueError):
        unseal_reasoning_content(
            parsed,
            authority,
            assistant_content="visible assistant text",
            tool_calls=_tool_calls("call-one", raw_arguments='{"q":"changed"}'),
        )
    with pytest.raises(ValueError):
        unseal_reasoning_content(
            parse_reasoning_content_carrier(retagged),
            authority,
            assistant_content="visible assistant text",
            tool_calls=_tool_calls("call-one"),
        )
    with pytest.raises(ValueError):
        unseal_reasoning_content(
            parsed.model_copy(update={"deployment_hint": "fireworks-rung-two"}),
            authority,
            assistant_content="visible assistant text",
            tool_calls=_tool_calls("call-one"),
        )


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
        assistant_content=None,
        tool_calls=_tool_calls("call-one"),
        content="hidden",
    )
    with pytest.raises(ValueError):
        parse_reasoning_content_carrier(f"{carrier}=")

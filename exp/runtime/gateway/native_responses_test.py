"""Focused Responses continuation tests below the native bridge boundary."""

from __future__ import annotations

import base64

import pytest

from exp.runtime.gateway.native_responses import ContinuationContext, remember_turn
from exp.runtime.gateway.reasoning_carrier import FIREWORKS_REASONING_CONTENT_PREFIX
from exp.runtime.openai_protocol.state import BoundedContinuationStore, ProtocolNamespace


def _context() -> ContinuationContext:
    """Build one namespaced retention context."""
    return ContinuationContext(
        namespace=ProtocolNamespace(
            organization_id="organization-one",
            identity_id="identity-one",
            alias_revision_id="revision-one",
        ),
        episode_key="e" * 64,
        response_id="response-one",
        messages=(),
    )


def _carrier() -> str:
    """Build one structurally valid carrier retained encrypted by this layer."""
    deployment = base64.urlsafe_b64encode(b"fireworks-rung").rstrip(b"=").decode()
    envelope = base64.urlsafe_b64encode(b"opaque-envelope").rstrip(b"=").decode()
    return f"{FIREWORKS_REASONING_CONTENT_PREFIX}{deployment}:{envelope}"


def test_remember_turn_retains_the_sealed_responses_carrier() -> None:
    """Server-side continuation storage never replaces the carrier with plaintext."""
    store = BoundedContinuationStore()
    context = _context()

    remember_turn(
        store,
        context=context,
        data={
            "text": "",
            "refusal": False,
            "reasoning_content_carrier": _carrier(),
            "tool_calls": [{"call_id": "call-one", "name": "lookup", "arguments": "{}"}],
        },
    )

    state = store.resolve_now(
        namespace=context.namespace,
        previous_response_id=context.response_id,
    )
    message = state.messages[-1]
    assert message.provider_reasoning[0].kind == "sealed_reasoning_content"
    assert message.provider_reasoning[0].carrier == _carrier()


def test_remember_turn_rejects_a_malformed_responses_carrier() -> None:
    """Continuation storage fails before retaining an unauthenticatable envelope."""
    store = BoundedContinuationStore()
    context = _context()

    with pytest.raises(ValueError, match="gateway carrier"):
        remember_turn(
            store,
            context=context,
            data={
                "text": "",
                "refusal": False,
                "reasoning_content_carrier": (f"{FIREWORKS_REASONING_CONTENT_PREFIX}malformed"),
                "tool_calls": [{"call_id": "call-one", "name": "lookup", "arguments": "{}"}],
            },
        )

    with pytest.raises(Exception, match="unavailable"):
        store.resolve_now(
            namespace=context.namespace,
            previous_response_id=context.response_id,
        )

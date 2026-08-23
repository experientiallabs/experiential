"""JSON-typed native-boundary helpers for input and output enforcement."""

from __future__ import annotations

import json
from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.guardrails.client import assert_not_internal_classification
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCompletion,
    GuardrailPolicy,
    GuardrailRejected,
    GuardrailToolCall,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine


def enforce_native_input(
    engine: GuardrailEngine | None,
    *,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    deadline_monotonic: float,
) -> tuple[GatewayRequest, GuardrailPolicy | None]:
    """Apply input enforcement after continuation and before native routing.

    Args:
        engine: Optional composed engine. ``None`` skips all guardrail work.
        authorization: Frozen authenticated identity.
        request: Canonical request after continuation expansion.
        deadline_monotonic: Remaining request-wide deadline.

    Returns:
        The validated or transformed request and the assigned policy, if any.

    Raises:
        GuardrailRejected: The input chain blocked or fail-closed.
        GuardrailRecursionError: A classifier re-entered the public route.
    """
    assert_not_internal_classification()
    if engine is None:
        return request, None
    policy = engine.policy_for(authorization.identity_id)
    if policy is None:
        return request, None
    return (
        engine.enforce_input(
            policy=policy,
            request=request,
            deadline_monotonic=deadline_monotonic,
        ),
        policy,
    )


def parse_output_payload(data: JsonObject) -> GuardrailCompletion:
    """Decode one native output-inspection payload.

    Args:
        data: JSON object with ``text``, ``refusal``, and ``tool_calls``.

    Returns:
        The normalized completion presented to the output chain.
    """
    raw_calls = data.get("tool_calls", [])
    calls: list[GuardrailToolCall] = []
    if isinstance(raw_calls, list):
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            arguments = item.get("arguments") or item.get("raw_arguments") or ""
            if isinstance(call_id, str) and isinstance(name, str) and isinstance(arguments, str):
                calls.append(GuardrailToolCall(call_id=call_id, name=name, arguments=arguments))
    return GuardrailCompletion(
        text=str(data.get("text") or ""),
        refusal=bool(data.get("refusal")),
        tool_calls=tuple(calls),
    )


def encode_output_decision(
    *,
    action: str,
    replacement_text: str | None = None,
    failure: JsonObject | None = None,
) -> str:
    """Encode one native output decision without request content."""
    payload: JsonObject = {"action": action}
    if replacement_text is not None:
        payload["replacement_text"] = replacement_text
    if failure is not None:
        payload["failure"] = failure
    return json.dumps(payload, separators=(",", ":"))


def enforce_native_output(
    engine: GuardrailEngine | None,
    policy: GuardrailPolicy | None,
    argument: str,
    *,
    deadline_monotonic: float,
) -> str:
    """Run the output chain once for a native buffered completion.

    Args:
        engine: Optional composed engine.
        policy: Policy captured at admission. ``None`` means unguarded.
        argument: JSON object with the winning completion fields.
        deadline_monotonic: Remaining request-wide deadline.

    Returns:
        JSON decision with ``action``, optional ``replacement_text``, and
        optional sanitized ``failure``.
    """
    if engine is None or policy is None:
        return encode_output_decision(
            action=GuardrailAction.ERROR.value,
            failure={
                "failure_class": "guardrail",
                "safe_message": "A gateway guardrail could not complete this request.",
            },
        )
    if not policy.output_checks:
        return encode_output_decision(action=GuardrailAction.ALLOW.value)
    data = cast(JsonObject, json.loads(argument))
    completion = parse_output_payload(data)
    try:
        result = engine.enforce_output(
            policy=policy,
            completion=completion,
            deadline_monotonic=deadline_monotonic,
        )
    except GuardrailRejected as exc:
        failure = exc.failure
        return encode_output_decision(
            action=str(failure.safe_details.get("action") or GuardrailAction.ERROR.value),
            failure={
                "failure_class": failure.failure_class.value,
                "safe_message": failure.safe_message,
            },
        )
    if result.text != completion.text:
        return encode_output_decision(
            action=GuardrailAction.MODIFY.value,
            replacement_text=result.text,
        )
    return encode_output_decision(action=GuardrailAction.ALLOW.value)

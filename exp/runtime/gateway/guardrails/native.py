"""JSON-typed native-boundary helpers for input and output enforcement."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.guardrails.bounded import run_on_native_loop
from exp.runtime.gateway.guardrails.client import assert_not_internal_classification
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCompletion,
    GuardrailPolicy,
    GuardrailRejected,
    GuardrailToolCall,
    OutputGuardrailMode,
)
from exp.runtime.gateway.guardrails.deterministic import NativeDetector, native_input_request
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine


def enforce_native_input(
    engine: GuardrailEngine | None,
    *,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    deadline_monotonic: float,
    detectors: Mapping[str, NativeDetector] | None = None,
) -> tuple[GatewayRequest, GuardrailPolicy | None]:
    """Apply input enforcement after continuation and before native routing.

    A chain built only from adapters with a compiled native detector runs
    inline here, so it pays neither the contract projection nor the
    isolation-worker round trip. Every other chain uses the engine.

    Args:
        engine: Optional composed engine. ``None`` skips all guardrail work.
        authorization: Frozen authenticated identity.
        request: Canonical request after continuation expansion.
        deadline_monotonic: Remaining request-wide deadline.
        detectors: Compiled deterministic detectors, keyed by adapter.

    Returns:
        The validated or transformed request and the assigned policy, if any.

    Raises:
        GuardrailRejected: The input chain blocked or fail-closed.
        GuardrailRecursionError: A classifier re-entered the public route.
    """
    assert_not_internal_classification()
    if engine is None:
        return request, None
    policy = engine.policy_for(authorization.organization_id, authorization.identity_id)
    if policy is None:
        return request, None
    if detectors:
        native = native_input_request(
            policy,
            detectors,
            request,
            monotonic=time.monotonic,
            deadline_monotonic=deadline_monotonic,
        )
        if native is not None:
            return native, policy
    return (
        run_on_native_loop(
            engine.enforce_input(
                policy=policy,
                request=request,
                deadline_monotonic=deadline_monotonic,
            )
        ),
        policy,
    )


def native_output_mode(
    engine: GuardrailEngine | None,
    policy: GuardrailPolicy | None,
    request: GatewayRequest,
) -> OutputGuardrailMode:
    """Return the output enforcement shape one admission must use.

    Args:
        engine: Optional composed engine. ``None`` leaves the stream untouched.
        policy: Policy resolved during input enforcement, if any.
        request: Canonical request after continuation expansion.

    Returns:
        ``off``, ``buffer``, or ``stream`` for the data plane.
    """
    if engine is None:
        return OutputGuardrailMode.OFF
    return engine.output_mode(
        policy,
        streaming=request.stream,
        tools_offered=bool(
            request.tools or request.provider_native_tools or request.provider_server_tools
        ),
        reasoning_text_requested=bool(
            request.reasoning_summary is not None
            or request.reasoning_effort is not None
            or request.thinking_default_enable
        ),
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


def _guardrail_failure_payload(safe_message: str, failure_class: str = "guardrail") -> JsonObject:
    """Return one sanitized failure body for a native decision."""
    return {"failure_class": failure_class, "safe_message": safe_message}


def _settled_bytes(data: JsonObject) -> int:
    """Return how many provider completion bytes already left the buffer."""
    value = data.get("settled_bytes")
    return value if isinstance(value, int) else 0


def enforce_native_output_segment(
    engine: GuardrailEngine | None,
    policy: GuardrailPolicy | None,
    argument: str,
    *,
    deadline_monotonic: float,
) -> str:
    """Redact and release the settled part of one streamed completion tail.

    The data plane owns the buffer: it presents the tail it is holding and
    receives back the text it may send now plus the text it must keep. The
    call is synchronous on the caller's thread, because a deterministic
    redactor is bounded CPU work and any hop would reintroduce the latency
    this path exists to remove.

    Args:
        engine: Optional composed engine.
        policy: Policy captured at admission. ``None`` means unguarded.
        argument: JSON object with ``pending``, ``final``, and
            ``settled_bytes``.
        deadline_monotonic: Remaining request-wide deadline.

    Returns:
        JSON decision with ``action`` plus either ``release``, ``pending``,
        and ``flagged``, or a sanitized ``failure``.
    """
    data = cast(JsonObject, json.loads(argument))
    if engine is None or policy is None:
        return _encode_segment_failure(
            _guardrail_failure_payload("A gateway guardrail could not complete this request.")
        )
    try:
        segment = engine.release_output_segment(
            policy=policy,
            pending=str(data.get("pending") or ""),
            final=bool(data.get("final")),
            settled_bytes=_settled_bytes(data),
            deadline_monotonic=deadline_monotonic,
        )
    except GuardrailRejected as exc:
        return _encode_segment_failure(
            _guardrail_failure_payload(exc.failure.safe_message, exc.failure.failure_class.value)
        )
    return json.dumps(
        {
            "action": GuardrailAction.ALLOW.value,
            "release": segment.release,
            "pending": segment.pending,
            "flagged": segment.flagged,
        },
        separators=(",", ":"),
    )


def _encode_segment_failure(failure: JsonObject) -> str:
    """Encode one fail-closed streaming decision that releases nothing."""
    return json.dumps(
        {"action": GuardrailAction.ERROR.value, "failure": failure},
        separators=(",", ":"),
    )


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
        result = run_on_native_loop(
            engine.enforce_output(
                policy=policy,
                completion=completion,
                deadline_monotonic=deadline_monotonic,
            )
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

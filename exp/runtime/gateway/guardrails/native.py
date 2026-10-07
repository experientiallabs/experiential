"""Native callbacks for the shared request-owned guardrail session."""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, cast

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCompletion,
    GuardrailRejected,
    GuardrailToolCall,
    OutputGuardrailMode,
)
from exp.runtime.gateway.guardrails.session import GuardrailSession, denied_input_failure
from exp.runtime.gateway.native_accounting_errors import NativeBridgeError
from exp.runtime.gateway.native_capture import (
    CaptureController,
    PendingCapture,
    begin_capture,
    capture_unavailable_failure,
    discard_capture,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

if TYPE_CHECKING:
    from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting


def require_capture(
    controller: CaptureController | None,
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    pending: PendingCapture | None,
) -> None:
    """Register approved input or terminalize unavailable synchronous capture admission.

    Args:
        controller: Optional host-configured capture collector and policy.
        accounting: Owner of the already accepted request's durable terminal write.
        authorization: Exact authenticated request identity.
        pending: Approved public context, or None when capture is disabled.

    Returns:
        None once capture is registered or does not apply.

    Raises:
        NativeBridgeError: Capture is unavailable; accounting retains the failed request
            before the sanitized 503 is raised, without dispatching a provider.
    """
    if pending is None or begin_capture(
        controller, authorization, pending.request, session_id=pending.session_id
    ):
        return
    failure = capture_unavailable_failure()
    accounting.finish_request_quietly(authorization, failure)
    raise NativeBridgeError(
        OpenAIProtocolError(
            status_code=503, code="capture_unavailable", message=failure.safe_message
        )
    )


def validate_guardrail_engine(engine: GuardrailEngine | None) -> None:
    """Require the coordinated native guardrail contract before serving guarded traffic."""
    if engine is None:
        return
    native = importlib.import_module("exp_gateway_native")
    if getattr(native, "GUARDRAIL_CONTRACT_VERSION", None) != 3:
        raise ValueError(
            "guardrails require native GUARDRAIL_CONTRACT_VERSION=3; "
            "install the coordinated native package"
        )


def open_admission_guardrails(
    engine: GuardrailEngine | None,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    data: JsonObject,
    *,
    deadline_monotonic: float,
) -> GuardrailSession | None:
    """Freeze admission policy and reject a replay claim made under a different snapshot."""
    session = (
        None
        if engine is None
        else engine.open_request(authorization, deadline_monotonic=deadline_monotonic)
    )
    if request.idempotency_key is not None:
        revision = None if session is None else session.engine.policy_revision(session.policies)
        if (
            "claimed_guardrail_revision" not in data
            or data["claimed_guardrail_revision"] != revision
        ):
            raise NativeBridgeError(
                OpenAIProtocolError(
                    status_code=409,
                    code="idempotency_conflict",
                    message="Guardrail policy changed during admission. Retry the request.",
                    param="Idempotency-Key",
                )
            )
    return session


def require_unguarded_surface(
    engine: GuardrailEngine | None,
    authorization: AuthorizationSnapshot,
    surface: str,
) -> None:
    """Reject applicable policies before accepting a surface this engine cannot inspect."""
    policies = (
        ()
        if engine is None
        else engine.policies_for(authorization.organization_id, authorization.identity_id)
    )
    if engine is not None:
        engine.record_unsupported_observations(policies)
    if any(policy.mode == "enforce" for policy in policies):
        raise NativeBridgeError(
            OpenAIProtocolError(
                status_code=400,
                code="unsupported_capability",
                message=(
                    f"Guardrail policies do not support native {surface}. "
                    "Use a supported text endpoint."
                ),
                param="model",
            )
        )


def native_output_mode(
    session: GuardrailSession | None,
    request: GatewayRequest,
    *,
    image_output: bool = False,
    wire_route: Sequence[JsonObject] = (),
) -> OutputGuardrailMode:
    """Choose one output execution strategy for the entire frozen policy set."""
    if session is None:
        return OutputGuardrailMode.OFF
    return session.output_mode(
        request,
        reasoning=upstream_requests_reasoning(wire_route),
        images=image_output or any(wire.get("image_output") is True for wire in wire_route),
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
    raw_context = data.get("context", [])
    if not isinstance(raw_context, list) or any(not isinstance(item, str) for item in raw_context):
        raise GuardrailRejected(denied_input_failure())
    return GuardrailCompletion(
        text=str(data.get("text") or ""),
        refusal=bool(data.get("refusal")),
        tool_calls=tuple(calls),
        context=tuple(str(item) for item in raw_context),
    )


def encode_output_decision(
    *,
    action: str,
    replacement_text: str | None = None,
    failure: JsonObject | None = None,
) -> str:
    """Encode a release decision with sanitized failure metadata."""
    payload: JsonObject = {"action": action}
    if replacement_text is not None:
        payload["replacement_text"] = replacement_text
    if failure is not None:
        payload["failure"] = failure
    return json.dumps(payload, separators=(",", ":"))


def enforce_native_output_segment(session: GuardrailSession | None, argument: str) -> str:
    """Apply the session's deterministic streaming strategy before releasing bytes."""
    data = cast(JsonObject, json.loads(argument))
    try:
        if session is None:
            raise GuardrailRejected(denied_input_failure())
        value = data.get("settled_bytes")
        segment = session.release_output_segment(
            pending=str(data.get("pending") or ""),
            final=bool(data.get("final")),
            settled_bytes=value if isinstance(value, int) else 0,
        )
    except GuardrailRejected as exc:
        return encode_output_decision(
            action=str(exc.failure.safe_details.get("action", "error")),
            failure=exc.failure.model_dump(mode="json"),
        )
    return json.dumps(
        {
            "action": "allow",
            "release": segment.release,
            "pending": segment.pending,
            "flagged": segment.flagged,
        }
    )


def enforce_native_output(session: GuardrailSession | None, argument: str) -> str:
    """Apply every scoped policy to the same buffered completion before releasing it."""
    try:
        if session is None:
            raise GuardrailRejected(denied_input_failure())
        completion = parse_output_payload(cast(JsonObject, json.loads(argument)))
        result = session.inspect_output(completion)
    except GuardrailRejected as exc:
        return encode_output_decision(
            action=str(exc.failure.safe_details.get("action", "error")),
            failure=exc.failure.model_dump(mode="json"),
        )
    if result != completion:
        return encode_output_decision(
            action=GuardrailAction.MODIFY.value, replacement_text=result.text
        )
    return encode_output_decision(action=GuardrailAction.ALLOW.value)


def upstream_requests_reasoning(wire_route: Sequence[JsonObject]) -> bool:
    """Return whether any rung's built payload asks for readable reasoning.

    Reasoning display defaults add an OpenAI ``reasoning.summary`` or an
    Anthropic ``thinking.display`` the caller never sent; a streamed output
    chain must then buffer, because it cannot judge or redact reasoning.

    Args:
        wire_route: The admission's ordered wire entries.

    Returns:
        ``True`` when a payload carries a summary or a thinking display.
    """
    for wire in wire_route:
        payload = wire.get("upstream_payload")
        if not isinstance(payload, dict):
            continue
        reasoning = payload.get("reasoning")
        thinking = payload.get("thinking")
        if (isinstance(reasoning, dict) and "summary" in reasoning) or (
            isinstance(thinking, dict) and "display" in thinking
        ):
            return True
    return False


class _GuardrailPlane(Protocol):
    """Services used by each admitted request's sole guardrail session.

    Attributes:
        _accounting: Owner of live requests and their terminal settlement.
        _capture: Optional collector bound to the host's authenticated capture policy.
    """

    _accounting: NativeAttemptAccounting
    _capture: CaptureController | None


class NativeGuardrailsMixin:
    """Resolve native operations to one frozen session, independent of policy scope."""

    def guardrail_input_status(self: _GuardrailPlane, argument: str) -> str:
        """Poll the shared input session and register capture once before allowing release.

        An occupied request lock returns pending rather than blocking a bridge worker.
        Capture failure is retained in the same session for zero-charge settlement;
        cancellation discards capture registered while its owner was closing.

        Args:
            argument: JSON object containing the authenticated admission's request_id.

        Returns:
            A JSON pending, allow, error, or closed decision. Allow requires both current
            session approval and successful required capture registration.

        Raises:
            NativeBridgeError: A concurrent abandonment's durable terminal write failed;
                accounting keeps the request for retry and the native gate fails closed.
        """
        request_id = str(json.loads(argument).get("request_id") or "")
        entry = self._accounting.entry(request_id)
        if entry is None or entry.pending_abandon is not None:
            return '{"action":"closed"}'
        decision = (
            {"action": "error", "failure": denied_input_failure().model_dump(mode="json")}
            if entry.guardrails is None
            else entry.guardrails.input_decision()
        )
        if decision["action"] == "allow":
            # Reservation owns this same lock during ledger I/O. Poll without
            # waiting so unrelated dispatch and settlement workers remain available.
            if not entry.execution_lock.acquire(blocking=False):
                return '{"action":"pending"}'
            began = False
            try:
                if self._accounting.entry(request_id) is not entry or entry.pending_abandon:
                    return '{"action":"closed"}'
                session = entry.guardrails
                assert session is not None  # Only the same session's approval reaches capture.
                decision = session.input_decision()
                pending = entry.pending_capture
                if decision["action"] == "allow" and pending is not None:
                    began = begin_capture(
                        self._capture,
                        entry.authorization,
                        pending.request,
                        entry.route.snapshot.exact_model_id,
                        session_id=pending.session_id,
                    )
                    if not began:
                        session.cancel(capture_unavailable_failure())
                    entry.pending_capture = None
                    decision = session.input_decision()
            finally:
                entry.execution_lock.release()
                try:
                    if entry.pending_abandon is not None:
                        self._accounting.abandon(argument)
                finally:
                    if began and (
                        entry.pending_abandon is not None
                        or self._accounting.entry(request_id) is not entry
                    ):
                        discard_capture(self._capture, request_id)
        return (
            json.dumps(decision)
            if self._accounting.entry(request_id) is entry and entry.pending_abandon is None
            else '{"action":"closed"}'
        )

    def enforce_output_segment(self: _GuardrailPlane, argument: str) -> str:
        """Release one deterministic segment through the admitted session."""
        entry = self._accounting.entry(str(json.loads(argument).get("request_id") or ""))
        return enforce_native_output_segment(None if entry is None else entry.guardrails, argument)

    def enforce_output(self: _GuardrailPlane, argument: str) -> str:
        """Inspect one complete output through the admitted session."""
        entry = self._accounting.entry(str(json.loads(argument).get("request_id") or ""))
        return enforce_native_output(None if entry is None else entry.guardrails, argument)

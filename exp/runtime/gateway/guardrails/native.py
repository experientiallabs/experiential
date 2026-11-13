"""Native callbacks for the shared request-owned guardrail session."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, cast

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure, GatewayRequest
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
)
from exp.runtime.gateway.native_components import SyncWriteLedger
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

if TYPE_CHECKING:
    from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting


def require_capture(
    controller: CaptureController | None,
    authorization: AuthorizationSnapshot,
    pending: PendingCapture | None,
) -> None:
    """Register approved input or return its exact failure to the admission owner.

    Args:
        controller: Optional host-configured capture collector and policy.
        authorization: Exact authenticated request identity.
        pending: Approved public context, or None when capture is disabled.

    Returns:
        None once capture is registered or does not apply.

    Raises:
        NativeBridgeError: Capture is unavailable; the caller must retain the attached
            failure before returning the sanitized 503 without dispatching a provider.
    """
    if pending is None or begin_capture(
        controller, authorization, pending.request, session_id=pending.session_id
    ):
        return
    failure = capture_unavailable_failure()
    raise NativeBridgeError(
        OpenAIProtocolError(
            status_code=503, code="capture_unavailable", message=failure.safe_message
        ),
        failure=failure,
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
    if session is not None:
        session.native_execution = True
    if session is not None:
        session.native_execution = True
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
    """Record missed coverage and reject only strict unsupported policies."""
    policies = () if engine is None else engine.policies_for(authorization)
    if engine is not None:
        engine.record_unsupported_observations(
            policies, request_id=authorization.request_id, surface=surface
        )
    if any(policy.mode == "enforce" and not policy.input_fail_open for policy in policies):
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


def discard_unaccepted_input(session: GuardrailSession | None) -> None:
    """Cancel unowned inspections and record bounded loss without a financial admission."""
    if session is None:
        return
    inspections = list(session.native_completed)
    if session.native_observation is not None:
        inspections.append(session.native_observation)
    session.cancel()
    for inspection in inspections:
        inspection.cancel()
    session.native_observation = None
    session.native_completed.clear()
    if inspections:
        logging.getLogger(__name__).warning(
            "moderation records dropped before request acceptance",
            extra={
                "operation": "native_moderation_unaccepted",
                "samples": len(inspections),
                "outcome": "dropped",
            },
        )


class _GuardrailPlane(Protocol):
    """Services used by each admitted request's sole guardrail session.

    Attributes:
        _accounting: Owner of live requests and their terminal settlement.
        _capture: Optional collector bound to the host's authenticated capture policy.
    """

    _accounting: NativeAttemptAccounting
    _capture: CaptureController | None


class NativeGuardrailsMixin:
    """Administrative inspection calls use the request's native resources."""

    _accounting: NativeAttemptAccounting

    _write_ledger: SyncWriteLedger

    def record_rejected_input(
        self,
        authorization: AuthorizationSnapshot,
        session: GuardrailSession,
        failure: GatewayFailure,
    ) -> None:
        """Retain a serial denial without dispatching or replacing the public refusal."""
        try:
            self._write_ledger.accept_request(authorization=authorization)
        except Exception:  # noqa: BLE001 - diagnostic storage cannot change a decided refusal.
            logging.getLogger(__name__).warning("could not record rejected input admission")
            discard_unaccepted_input(session)
            return
        self._accounting.finish_request_quietly(authorization, failure, input_session=session)

    def guardrail_input_status(self, argument: str) -> str:
        """Read one immediate native status; serving awaits its asynchronous signal."""
        return self._accounting.call("guardrail_input_status", argument)

    def enforce_output_segment(self, argument: str) -> str:
        """Inspect a deterministic segment with the admitted native policy."""
        return self._accounting.call("enforce_output_segment", argument)

    def enforce_output(self, argument: str) -> str:
        """Inspect complete output with the admitted native policy."""
        return self._accounting.call("enforce_output", argument)

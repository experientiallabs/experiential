"""Resolve deterministic guardrail chains for in-plane native enforcement.

A deterministic adapter (today the local ``regex`` rule) has an exact native
implementation in the Rust data plane, so a chain built only from such
adapters can run beside the buffered completion with no Python callback, no
GIL acquisition, and no JSON round trip of the completion text. Every other
adapter, including ``http_json`` and future model-based classifiers, keeps
crossing the existing Python boundary unchanged.

Nothing here logs request text, completions, detector payloads, or
replacements.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from typing import Protocol

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCheck,
    GuardrailPolicy,
    GuardrailRejected,
    guardrail_failure,
    request_content_bytes,
)
from exp.runtime.gateway.guardrails.enforcement import restored_provider_authority

_logger = logging.getLogger(__name__)


class NativeDetector(Protocol):
    """One compiled native rule shared with the data plane.

    The incremental streaming path can call the same handle on a delta
    window, since both operations are pure over the subject they receive.
    """

    def redact(self, text: str) -> str | None:
        """Return the rewritten subject, or ``None`` when nothing matched."""
        ...

    def matches(self, text: str) -> bool:
        """Return whether the subject matches at all."""
        ...


def compile_native_detectors(specifications: Mapping[str, str]) -> dict[str, NativeDetector]:
    """Compile every deterministic rule once, outside the request path.

    A rule the native detector cannot express is skipped, and its chain then
    keeps running through the Python adapter with identical results.

    Args:
        specifications: Adapter identity to its content-free JSON rule.

    Returns:
        Adapter identity to compiled detector, for the adapters that
        compiled natively. An environment without the extension gets an
        empty mapping and keeps the Python path.
    """
    if not specifications:
        return {}
    try:
        native = importlib.import_module("exp_gateway_native")
    except ModuleNotFoundError:
        return {}
    detectors: dict[str, NativeDetector] = {}
    for adapter_id, specification in specifications.items():
        try:
            detectors[adapter_id] = native.RegexDetector(specification)
        except ValueError:
            _logger.info(
                "native guardrail detector declined adapter_id=%s; keeping the python adapter",
                adapter_id,
            )
    return detectors


def native_input_request(
    policy: GuardrailPolicy,
    detectors: Mapping[str, NativeDetector],
    request: GatewayRequest,
    *,
    monotonic: Callable[[], float],
    deadline_monotonic: float,
) -> GatewayRequest | None:
    """Run a fully deterministic input chain inline on the admitting thread.

    Admission already holds the interpreter, so the saving here is the
    contract construction, the isolation-worker hop, and the future round
    trip the general engine pays per check. Detection itself runs in Rust
    with the interpreter released.

    Args:
        policy: The assigned identity policy for this admission.
        detectors: Compiled deterministic detectors, keyed by adapter.
        request: Canonical request after continuation expansion.
        monotonic: Process-local clock in seconds.
        deadline_monotonic: Remaining request-wide deadline.

    Returns:
        The validated or redacted request, or ``None`` when the chain binds
        an adapter the native detector cannot evaluate and the caller must
        use the Python engine.

    Raises:
        GuardrailRejected: A check blocked, errored, or fail-closed.
    """
    if not policy.input_checks:
        return request
    if any(check.adapter_id not in detectors for check in policy.input_checks):
        return None
    if request_content_bytes(request) > policy.max_request_bytes:
        _record(policy, None, GuardrailAction.ERROR)
        raise GuardrailRejected(guardrail_failure(action=GuardrailAction.ERROR))
    current = request
    for check in policy.input_checks:
        detector = detectors[check.adapter_id]
        remaining = deadline_monotonic - monotonic()
        # The check runs under the tighter of its authored timeout and the
        # remaining request deadline, exactly as the engine bounds it.
        budget = min(check.timeout_ms / 1000.0, remaining)
        if budget <= 0:
            _uncertain(policy, check)
            continue
        started = monotonic()
        try:
            messages, flagged, tool_match = _redacted_messages(detector, current)
        except ValueError:
            _uncertain(policy, check)
            continue
        # A scan that overran its budget is uncertain: an inspection the
        # engine would have abandoned never returns a verdict here.
        if monotonic() - started > budget:
            _uncertain(policy, check)
            continue
        if not flagged:
            _record(policy, check, GuardrailAction.ALLOW, monotonic() - started)
            continue
        _record(policy, check, check.action, monotonic() - started)
        current = _applied(policy, check, current, messages, tool_match=tool_match)
    return current


def _redacted_messages(
    detector: NativeDetector,
    request: GatewayRequest,
) -> tuple[tuple[GatewayMessage, ...], bool, bool]:
    """Redact message text and flag tool arguments, which are never rewritten.

    Returns:
        The rewritten messages, whether anything matched, and whether a
        match landed in tool-call arguments.

    Raises:
        ValueError: The subject exceeded an inspection bound.
    """
    messages: list[GatewayMessage] = []
    flagged = False
    tool_match = False
    for message in request.messages:
        rewritten = detector.redact(message.content or "")
        flagged |= rewritten is not None
        messages.append(
            message if rewritten is None else message.model_copy(update={"content": rewritten})
        )
        for call in message.tool_calls:
            found = detector.matches(call.arguments_json())
            flagged |= found
            tool_match |= found
    return tuple(messages), flagged, tool_match


def _applied(
    policy: GuardrailPolicy,
    check: GuardrailCheck,
    request: GatewayRequest,
    messages: tuple[GatewayMessage, ...],
    *,
    tool_match: bool,
) -> GatewayRequest:
    """Apply one flagged deterministic input action.

    Raises:
        GuardrailRejected: The action blocks, or a modification cannot be
            applied without leaking a matched tool argument or breaking
            hidden provider replay authority.
    """
    del policy
    if check.action is GuardrailAction.ALLOW:
        return request
    if check.action is GuardrailAction.MODIFY:
        if tool_match:
            raise GuardrailRejected(
                guardrail_failure(action=GuardrailAction.ERROR, check_id=check.check_id)
            )
        restored = restored_provider_authority(request.messages, messages)
        if restored is None:
            raise GuardrailRejected(
                guardrail_failure(action=GuardrailAction.ERROR, check_id=check.check_id)
            )
        return request.model_copy(update={"messages": restored})
    raise GuardrailRejected(guardrail_failure(action=check.action, check_id=check.check_id))


def _uncertain(policy: GuardrailPolicy, check: GuardrailCheck) -> None:
    """Fail closed for a protected identity, or skip the uncertain check.

    Raises:
        GuardrailRejected: The identity is protected.
    """
    _record(policy, check, GuardrailAction.ERROR)
    if policy.protected:
        raise GuardrailRejected(
            guardrail_failure(action=GuardrailAction.ERROR, check_id=check.check_id)
        )


def _record(
    policy: GuardrailPolicy,
    check: GuardrailCheck | None,
    action: GuardrailAction,
    latency_seconds: float = 0.0,
) -> None:
    """Emit the same content-free decision metadata the engine emits."""
    _logger.info(
        "guardrail decision policy_id=%s organization_id=%s identity_id=%s "
        "check_id=%s capability=%s action=%s latency_ms=%.1f",
        policy.policy_id,
        policy.organization_id,
        policy.identity_id,
        None if check is None else check.check_id,
        None if check is None else check.capability.value,
        action.value,
        latency_seconds * 1000.0,
    )


def native_output_plan(
    policy: GuardrailPolicy | None,
    detectors: Mapping[str, NativeDetector],
) -> JsonObject | None:
    """Return the resolved output chain when the data plane can run it alone.

    Args:
        policy: The assigned identity policy, or ``None`` for unguarded
            traffic.
        detectors: Compiled deterministic detectors, keyed by adapter.

    Returns:
        The authored-order chain, or ``None`` when the policy has no output
        stage or binds any adapter the data plane cannot evaluate.
    """
    if policy is None or not policy.output_checks:
        return None
    if any(check.adapter_id not in detectors for check in policy.output_checks):
        return None
    return {
        "protected": policy.protected,
        "max_response_bytes": policy.max_response_bytes,
        "policy_id": policy.policy_id,
        "organization_id": policy.organization_id,
        "identity_id": policy.identity_id,
        "checks": [
            {
                "action": check.action.value,
                "adapter_id": check.adapter_id,
                "check_id": check.check_id,
                "capability": check.capability.value,
                "timeout_ms": check.timeout_ms,
            }
            for check in policy.output_checks
        ],
    }

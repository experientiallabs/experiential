"""Tool-search round callback for the native data plane.

The data plane withheld the model's ``tool_search`` call(s) on one dial and
asks for the next dispatch of the same rung. The control plane runs the
search, extends the conversation, rebuilds that depth's wire entry from the
retained admission material, and returns the wire plus the facts the
response renders.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass, GatewayRequest
from exp.runtime.gateway.guardrails.contracts import GuardrailRejected
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.inspection import (
    open_inspection,
    require_output,
)
from exp.runtime.gateway.guardrails.streaming import GuardrailFragment, GuardrailOutput
from exp.runtime.gateway.native_accounting import NativeBridgeError, internal_protocol_error
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_reasoning import rung_provider_request
from exp.runtime.gateway.native_rungs import build_rung_dispatch
from exp.runtime.gateway.tool_search.round import WithheldSearchCall, parse_calls, perform_round


class _Registry(Protocol):
    def entry(self, request_id: str) -> InflightRequest | None:
        """Return the in-flight request, or ``None``."""
        ...


class _Plane(Protocol):
    _accounting: _Registry
    _guardrails: GuardrailEngine | None


def _search_output(request_id: str, calls: Sequence[WithheldSearchCall]) -> GuardrailOutput:
    """Validate complete generated calls before a gateway-owned search action.

    Args:
        request_id: Exact admitted request whose output session owns the calls.
        calls: Complete withheld search calls in provider order.

    Returns:
        A bounded output segment without truncating any call argument.

    Raises:
        GuardrailRejected: A call exceeds the inspection contract's coverage.
    """
    try:
        return GuardrailOutput(
            request_id=request_id,
            final=False,
            fragments=tuple(
                GuardrailFragment(
                    kind="tool",
                    channel=f"search:{call.call_id}",
                    text=call.raw_arguments,
                    name=call.name,
                )
                for call in calls
            ),
        )
    except ValueError:
        raise GuardrailRejected(
            GatewayFailure(
                failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
                safe_message="Content inspection does not support this generated tool call.",
            )
        ) from None


class NativeToolSearchMixin:
    """Answer one gateway tool-search round with a rebuilt dispatch."""

    def tool_search_round(self: _Plane, argument: str) -> str:
        """Search the deferred tools, extend the conversation, rebuild the rung.

        Args:
            argument: JSON object with ``request_id``, ``route_depth``,
                ``round``, and ``calls`` (``{call_id, name, arguments}`` each).

        Returns:
            JSON dispatch and completed search rounds, or an inspection failure
            and only the search rounds that actually executed.

        Raises:
            NativeBridgeError: The request is unknown, carries no tool search,
                or names a depth outside its route.
        """
        data = json.loads(argument)
        entry = self._accounting.entry(str(data.get("request_id") or ""))
        depth = data.get("route_depth")
        if (
            entry is None
            or entry.tool_search is None
            or entry.resolved_wires is None
            or entry.public_request is None
            or not isinstance(depth, int)
            or not 0 <= depth < len(entry.route.deployments)
            or depth >= len(entry.resolved_wires)
        ):
            raise NativeBridgeError(internal_protocol_error())
        provider_request = entry.request
        if not isinstance(provider_request, GatewayRequest):
            raise NativeBridgeError(internal_protocol_error())
        calls = parse_calls(data.get("calls"))
        rounds: list[JsonObject] = []
        try:
            if entry.guardrail_inspection is not None:
                require_output(
                    entry.guardrail_inspection,
                    _search_output(entry.authorization.request_id, calls),
                    deadline_monotonic=entry.deadline_monotonic,
                )
            outcome = perform_round(provider_request, entry.tool_search, calls)
            rounds = list(outcome.rounds)
            session = open_inspection(
                self._guardrails,
                authorization=entry.authorization,
                request=outcome.request,
                deadline_monotonic=entry.deadline_monotonic,
            )
            if (session is None) != (entry.guardrail_inspection is None):
                raise GuardrailRejected(
                    GatewayFailure(
                        failure_class=GatewayFailureClass.UNAVAILABLE,
                        safe_message="Content inspection changed during this request. Retry later.",
                    )
                )
            entry.guardrail_inspection = session
        except GuardrailRejected as exc:
            return json.dumps(
                {"inspection_failure": exc.failure.model_dump(mode="json"), "rounds": rounds}
            )
        deployment = entry.route.deployments[depth]
        profile, client = entry.resolved_wires[depth]
        budget = (
            entry.throttle_redial_budgets[depth]
            if depth < len(entry.throttle_redial_budgets)
            else 0
        )
        dispatch = build_rung_dispatch(
            entry.route,
            deployment,
            profile,
            client,
            provider_request=rung_provider_request(entry.route, deployment, outcome.request),
            public_request=entry.public_request,
            authorization=entry.authorization,
            throttle_redial_budget=budget,
        )
        entry.request = outcome.request
        output_bounds = list(entry.reserved_output_tokens_by_depth)
        output_bounds[depth] = dispatch.reserved_output_tokens
        entry.reserved_output_tokens_by_depth = tuple(output_bounds)
        signers = list(entry.signers)
        bindings = list(entry.dispatch_bindings)
        authorities = list(entry.reasoning_carrier_authorities)
        if depth < len(signers):
            signers[depth] = dispatch.signer
            entry.signers = tuple(signers)
        if depth < len(bindings):
            bindings[depth] = dispatch.binding
            entry.dispatch_bindings = tuple(bindings)
        if depth < len(authorities):
            authorities[depth] = dispatch.carrier_authority
            entry.reasoning_carrier_authorities = tuple(authorities)
        return json.dumps(
            {"wire": dispatch.wire_entry, "rounds": outcome.rounds, "exhausted": outcome.exhausted},
            separators=(",", ":"),
        )

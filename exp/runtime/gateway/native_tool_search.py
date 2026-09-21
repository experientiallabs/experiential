"""Tool-search round callback for the native data plane.

The data plane withheld the model's ``tool_search`` call(s) on one dial and
asks for the next dispatch of the same rung. The control plane runs the
search, extends the conversation, rebuilds that depth's wire entry from the
retained admission material, and returns the wire plus the facts the
response renders.
"""

from __future__ import annotations

import json
from typing import Protocol

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.native_accounting import NativeBridgeError, internal_protocol_error
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_reasoning import rung_provider_request
from exp.runtime.gateway.native_rungs import build_rung_dispatch
from exp.runtime.gateway.tool_search.round import parse_calls, perform_round


class _Registry(Protocol):
    def entry(self, request_id: str) -> InflightRequest | None:
        """Return the in-flight request, or ``None``."""
        ...


class _Plane(Protocol):
    _accounting: _Registry


class NativeToolSearchMixin:
    """Answer one gateway tool-search round with a rebuilt dispatch."""

    def tool_search_round(self: _Plane, argument: str) -> str:
        """Search the deferred tools, extend the conversation, rebuild the rung.

        Args:
            argument: JSON object with ``request_id``, ``route_depth``,
                ``round``, and ``calls`` (``{call_id, name, arguments}`` each).

        Returns:
            JSON ``{"wire": <DeploymentWire>, "rounds": [...], "exhausted": bool}``.

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
        outcome = perform_round(provider_request, entry.tool_search, parse_calls(data.get("calls")))
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

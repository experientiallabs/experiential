"""Recovery history hashes actual prefix input independently from cache routing hints."""

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.native_recovery import recovery_prefix_digest, session_cache_key


def request(system: str = "Stable instructions", user: str = "First turn") -> GatewayRequest:
    """Build two requests deliberately sharing the same caller affinity hint."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="system", content=system),
            GatewayMessage(role="user", content=user),
        ),
        prompt_cache_key="same-public-hint",
        provider_prompt_cache_key="xpl-same-namespaced-hint",
    )


def test_actual_prefix_change_with_same_hint_loses_history() -> None:
    """Caller-chosen hints cannot establish that unrelated prompt prefixes match."""
    first, changed = request(), request("Completely different system")
    route = _route()
    entries = [
        InflightRequest(
            authorization=route.snapshot.authorization,
            route=route,
            request=r,
            deadline_monotonic=10,
        )
        for r in (first, changed)
    ]
    assert session_cache_key(entries[0]) != session_cache_key(entries[1])
    key = session_cache_key(entries[0])
    assert key is not None and key.prefix_key == recovery_prefix_digest(first)
    assert recovery_prefix_digest(first) == recovery_prefix_digest(request(user="Different suffix"))
    assert recovery_prefix_digest(first) == recovery_prefix_digest(
        first.model_copy(update={"prompt_cache_key": "changed"})
    )


def test_prefix_tools_and_order_are_part_of_cache_evidence_identity() -> None:
    """Tool changes and reordered system roles invalidate the actual cached prefix."""
    first = request()
    tool = GatewayToolDefinition(name="read", parameters={"type": "object"})
    with_tool = first.model_copy(update={"tools": (tool,)})
    assert recovery_prefix_digest(first) != recovery_prefix_digest(with_tool)
    changed_tool = with_tool.model_copy(
        update={"tools": (tool.model_copy(update={"parameters": {"type": "string"}}),)}
    )
    assert recovery_prefix_digest(with_tool) != recovery_prefix_digest(changed_tool)
    ordered = first.model_copy(
        update={
            "messages": (
                GatewayMessage(role="system", content="a"),
                GatewayMessage(role="developer", content="b"),
                first.messages[-1],
            )
        }
    )
    reordered = ordered.model_copy(
        update={"messages": (ordered.messages[1], ordered.messages[0], ordered.messages[2])}
    )
    assert recovery_prefix_digest(ordered) != recovery_prefix_digest(reordered)

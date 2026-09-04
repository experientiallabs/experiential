"""Fireworks-specific opaque tool-continuation contracts."""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlsplit

from exp.common.core.artifacts import Sha256, sha256_json
from exp.common.models import ModelSnapshot
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.errors import ProviderParameterError


def is_fireworks_base_url(base_url: str) -> bool:
    """Return whether one endpoint is the exact public Fireworks inference root."""
    parsed = urlsplit(base_url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "api.fireworks.ai"
        and parsed.username is None
        and parsed.password is None
        and parsed.port in {None, 443}
        and parsed.path.rstrip("/") == "/inference/v1"
        and not parsed.query
        and not parsed.fragment
    )


def reasoning_content_route_sha256(model: ModelSnapshot) -> Sha256:
    """Bind opaque reasoning to one exact provider model revision and connection."""
    return sha256_json(
        {
            "schema_version": 1,
            "provider": model.provider,
            "model_id": model.model_id,
            "revision": model.revision,
            "connection_sha256": model.connection_sha256,
        }
    )


def require_responses_continuation_channel(request: GatewayRequest) -> None:
    """Reject tool-capable Fireworks Responses turns with no replay channel."""
    if (
        request.surface == GatewayApiSurface.RESPONSES
        and request.tools
        and request.tool_choice != "none"
        and request.response_store is False
        and not request.include_encrypted_reasoning
    ):
        raise ProviderParameterError(
            message=(
                "Fireworks tool continuations require either 'store: true' or "
                "'include: [\"reasoning.encrypted_content\"]'."
            ),
            param="store",
            code="unsupported_parameter",
        )


def prepare_gateway_reasoning_history(
    messages: Sequence[GatewayMessage],
    *,
    route_sha256: Sha256 | None,
) -> tuple[tuple[GatewayMessage, ...], bool]:
    """Strip stale Fireworks reasoning and validate active tool history."""
    last_user = max(
        (index for index, message in enumerate(messages) if message.role == "user"),
        default=-1,
    )
    active = False
    prepared: list[GatewayMessage] = []
    for index, message in enumerate(messages):
        blocks = tuple(
            block for block in message.provider_reasoning if block.kind == "reasoning_content"
        )
        keep = index > last_user
        retained = tuple(
            block
            for block in message.provider_reasoning
            if block.kind != "reasoning_content" or keep
        )
        if keep and blocks:
            active = True
            _require_active_carrier(message, blocks, route_sha256)
        prepared.append(
            message
            if retained == message.provider_reasoning
            else message.model_copy(update={"provider_reasoning": retained})
        )
    if active:
        _require_complete_tool_results(prepared[last_user + 1 :])
    return tuple(prepared), active


def _reasoning_parameter_error(message: str) -> ProviderParameterError:
    """Build one stable field-specific error for an unsafe replay."""
    return ProviderParameterError(
        message=message,
        param="messages.reasoning_content",
        code="invalid_parameter",
    )


def _require_active_carrier(
    message: GatewayMessage,
    blocks: tuple[object, ...],
    route_sha256: Sha256 | None,
) -> None:
    """Require one route-matched carrier on an assistant tool-call message."""
    if message.role != "assistant" or not message.tool_calls or len(blocks) != 1:
        raise _reasoning_parameter_error(
            "Active reasoning_content requires one assistant tool-call carrier."
        )
    block = blocks[0]
    if route_sha256 is None or getattr(block, "route_sha256", None) != route_sha256:
        raise _reasoning_parameter_error("reasoning_content belongs to a different provider route.")


def _require_complete_tool_results(messages: Sequence[GatewayMessage]) -> None:
    """Require exactly one result for every active assistant tool call.

    A provider answers some tool rounds without reasoning, so an active window mixes
    carrier-bearing turns with plain ones. Every tool call in the window is tracked for
    result identity, while completion is required only of the rounds that carry state.
    """
    pending: set[str] = set()
    window_call_ids: set[str] = set()
    completed_call_ids: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            if pending:
                raise _reasoning_parameter_error(
                    "reasoning_content tool calls need complete tool results."
                )
            call_ids = tuple(call.call_id for call in message.tool_calls)
            if len(call_ids) != len(set(call_ids)) or window_call_ids.intersection(call_ids):
                raise _reasoning_parameter_error("reasoning_content tool-call IDs must be unique.")
            window_call_ids.update(call_ids)
            if any(block.kind == "reasoning_content" for block in message.provider_reasoning):
                pending = set(call_ids)
        elif message.role == "tool" and window_call_ids:
            call_id = message.tool_call_id
            if call_id not in window_call_ids or call_id in completed_call_ids:
                raise _reasoning_parameter_error(
                    "reasoning_content requires exactly one result per tool call."
                )
            pending.discard(call_id)
            completed_call_ids.add(call_id)
    if pending or not messages or messages[-1].role != "tool":
        raise _reasoning_parameter_error(
            "reasoning_content can replay only in a completed tool continuation."
        )

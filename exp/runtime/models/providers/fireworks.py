"""Fireworks-specific reasoning effort and opaque tool-history contracts."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from exp.common.core.artifacts import Sha256, sha256_json
from exp.common.models import (
    ModelMessage,
    ModelSnapshot,
    OpaqueReasoningContentBlock,
    ReasoningEffort,
)
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.errors import (
    ProviderParameterError,
    UnsupportedReasoningEffortError,
)

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

_FIREWORKS_REASONING_EFFORTS: dict[str, tuple[ReasoningEffort, ...]] = {
    "deepseek-v4-flash-0731": ("none", "high", "max"),
    "glm-5p2": ("none", "high", "max"),
    "kimi-k2p7-code": ("none", "low", "medium", "high", "max"),
}
_EFFORT_ORDER: tuple[ReasoningEffort, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "ultra",
    "max",
)


def require_responses_continuation_channel(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
) -> None:
    """Reject Fireworks Responses when neither continuation channel is available."""
    if (
        request.surface == GatewayApiSurface.RESPONSES
        and request.response_store is False
        and not request.include_encrypted_reasoning
        and any(profile.fireworks_reasoning_route_sha256 is not None for profile in profiles)
    ):
        raise ProviderParameterError(
            message=(
                "Fireworks Responses tool continuations require either gateway storage or "
                "include=['reasoning.encrypted_content']. Enable one continuation channel or "
                "choose another provider route."
            ),
            param="include",
            code="unsupported_parameter",
        )


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


def fireworks_reasoning_efforts(
    model_id: str,
    *,
    explicit_efforts: Collection[str] | None = None,
) -> tuple[ReasoningEffort, ...] | None:
    """Return semantically distinct documented efforts for one Fireworks model.

    The Fireworks API accepts aliases that it silently promotes for DeepSeek V4
    and GLM 5.2. Those aliases are excluded because accepting a caller's
    ``low`` request while executing ``high`` would violate the gateway's exact
    generation-parameter contract.
    """
    supported = _FIREWORKS_REASONING_EFFORTS.get(_fireworks_model_name(model_id))
    if supported is None:
        return None
    if explicit_efforts is None:
        return supported
    normalized = {
        normalized_effort
        for effort in explicit_efforts
        if (normalized_effort := normalized_fireworks_default(model_id, effort)) is not None
    }
    return tuple(effort for effort in _EFFORT_ORDER if effort in supported and effort in normalized)


def fireworks_reasoning_effort(model_id: str, effort: str) -> str:
    """Return one truthful Fireworks effort or reject a silently promoted alias."""
    supported = fireworks_reasoning_efforts(model_id)
    if supported is None:
        return effort
    if effort in supported:
        return effort
    raise UnsupportedReasoningEffortError(
        effort=effort,
        supported_efforts=supported,
        param="reasoning_effort",
    )


def normalized_fireworks_default(
    model_id: str,
    effort: str | None,
) -> str | None:
    """Normalize an operator default to the provider's actual semantic tier."""
    if effort is None:
        return None
    name = _fireworks_model_name(model_id)
    if name in {"deepseek-v4-flash-0731", "glm-5p2"}:
        if effort in {"low", "medium"}:
            return "high"
        if effort == "xhigh":
            return "max"
    return effort


def prepare_gateway_reasoning_history(
    messages: Sequence[GatewayMessage],
    *,
    route_sha256: Sha256 | None,
) -> tuple[tuple[GatewayMessage, ...], bool]:
    """Strip old Fireworks reasoning and validate active gateway tool history.

    Fireworks ``interleaved`` history ignores reasoning at and before the last
    user message. Only assistant tool calls after that boundary are replayed,
    and those calls must have complete tool results before the continuation.

    Args:
        messages: Canonical gateway conversation.
        route_sha256: Exact Fireworks route selected for this payload, or
            ``None`` for every other Chat-compatible provider.

    Returns:
        Provider-visible messages and whether active Fireworks reasoning exists.

    Raises:
        ProviderParameterError: Active reasoning is malformed or belongs to a
            different route.
    """
    last_user = _last_user_index(tuple(message.role for message in messages))
    active = False
    prepared: list[GatewayMessage] = []
    for index, message in enumerate(messages):
        fireworks_blocks = tuple(
            block for block in message.provider_reasoning if block.kind == "reasoning_content"
        )
        keep_fireworks = index > last_user
        retained = tuple(
            block
            for block in message.provider_reasoning
            if block.kind != "reasoning_content" or keep_fireworks
        )
        if keep_fireworks and fireworks_blocks:
            active = True
            _require_active_gateway_carrier(message, fireworks_blocks, route_sha256)
        prepared.append(
            message
            if retained == message.provider_reasoning
            else message.model_copy(update={"provider_reasoning": retained})
        )
    if active:
        _require_complete_gateway_tool_results(prepared[last_user + 1 :])
    return tuple(prepared), active


def prepare_model_reasoning_history(
    messages: Sequence[ModelMessage],
    *,
    route_sha256: Sha256 | None,
) -> tuple[tuple[ModelMessage, ...], bool]:
    """Strip old Fireworks reasoning and validate active direct-client history."""
    last_user = _last_user_index(tuple(message.role for message in messages))
    active = False
    prepared: list[ModelMessage] = []
    for index, message in enumerate(messages):
        action = message.assistant_action
        blocks = () if action is None else action.provider_reasoning
        keep = index > last_user
        retained = blocks if keep else ()
        if keep and blocks:
            active = True
            _require_active_model_carrier(message, route_sha256)
        if action is not None and retained != blocks:
            action = action.model_copy(update={"provider_reasoning": retained})
            message = message.model_copy(update={"assistant_action": action})
        prepared.append(message)
    if active:
        _require_complete_model_tool_results(prepared[last_user + 1 :])
    return tuple(prepared), active


def _fireworks_model_name(model_id: str) -> str:
    """Return the final normalized Fireworks resource segment."""
    return model_id.lower().rstrip("/").rsplit("/", 1)[-1]


def _last_user_index(roles: tuple[str, ...]) -> int:
    """Return the last user boundary, or minus one for instruction-only history."""
    return max((index for index, role in enumerate(roles) if role == "user"), default=-1)


def _reasoning_parameter_error(message: str) -> ProviderParameterError:
    """Build one stable local error for an unsafe Fireworks replay."""
    return ProviderParameterError(
        message=message,
        param="messages.reasoning_content",
        code="invalid_parameter",
    )


def _require_active_gateway_carrier(
    message: GatewayMessage,
    blocks: tuple[OpaqueReasoningContentBlock, ...],
    route_sha256: Sha256 | None,
) -> None:
    """Require one route-matched carrier on an assistant tool-call message."""
    if message.role != "assistant" or not message.tool_calls or len(blocks) != 1:
        raise _reasoning_parameter_error(
            "Active Fireworks reasoning_content requires one assistant tool-call carrier."
        )
    if route_sha256 is None or blocks[0].route_sha256 != route_sha256:
        raise _reasoning_parameter_error(
            "Fireworks reasoning_content belongs to a different provider route."
        )


def _require_active_model_carrier(
    message: ModelMessage,
    route_sha256: Sha256 | None,
) -> None:
    """Require one route-matched direct-client carrier and assistant tool call."""
    action = message.assistant_action
    if message.role != "assistant" or action is None or not action.tool_calls:
        raise _reasoning_parameter_error(
            "Active Fireworks reasoning_content requires one assistant tool-call carrier."
        )
    if len(action.provider_reasoning) != 1:
        raise _reasoning_parameter_error(
            "Active Fireworks reasoning_content requires exactly one carrier."
        )
    if route_sha256 is None or action.provider_reasoning[0].route_sha256 != route_sha256:
        raise _reasoning_parameter_error(
            "Fireworks reasoning_content belongs to a different provider route."
        )


def _require_complete_gateway_tool_results(messages: Sequence[GatewayMessage]) -> None:
    """Require every active assistant call to receive a result before continuation."""
    pending: set[str] = set()
    active_call_ids: set[str] = set()
    completed_call_ids: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            if pending:
                raise _reasoning_parameter_error(
                    "Fireworks reasoning_content tool calls need complete tool results."
                )
            if any(block.kind == "reasoning_content" for block in message.provider_reasoning):
                call_ids = tuple(call.call_id for call in message.tool_calls)
                if len(call_ids) != len(set(call_ids)) or active_call_ids.intersection(call_ids):
                    raise _reasoning_parameter_error(
                        "Fireworks reasoning_content tool-call IDs must be unique."
                    )
                pending = set(call_ids)
                active_call_ids.update(call_ids)
        elif message.role == "tool" and active_call_ids:
            call_id = message.tool_call_id
            if call_id not in active_call_ids or call_id in completed_call_ids:
                raise _reasoning_parameter_error(
                    "Fireworks reasoning_content requires exactly one result per tool call."
                )
            pending.remove(call_id)
            completed_call_ids.add(call_id)
    if pending or messages[-1].role != "tool":
        raise _reasoning_parameter_error(
            "Fireworks reasoning_content can replay only in a completed tool continuation."
        )


def _require_complete_model_tool_results(messages: Sequence[ModelMessage]) -> None:
    """Require complete direct-client tool results after every active carrier."""
    pending: set[str] = set()
    active_call_ids: set[str] = set()
    completed_call_ids: set[str] = set()
    for message in messages:
        action = message.assistant_action
        if message.role == "assistant":
            if pending:
                raise _reasoning_parameter_error(
                    "Fireworks reasoning_content tool calls need complete tool results."
                )
            if action is not None and action.provider_reasoning:
                call_ids = tuple(call.call_id for call in action.tool_calls)
                if len(call_ids) != len(set(call_ids)) or active_call_ids.intersection(call_ids):
                    raise _reasoning_parameter_error(
                        "Fireworks reasoning_content tool-call IDs must be unique."
                    )
                pending = set(call_ids)
                active_call_ids.update(call_ids)
        elif message.role == "tool" and active_call_ids:
            call_id = message.tool_call_id
            if call_id not in active_call_ids or call_id in completed_call_ids:
                raise _reasoning_parameter_error(
                    "Fireworks reasoning_content requires exactly one result per tool call."
                )
            pending.remove(call_id)
            completed_call_ids.add(call_id)
    if pending or messages[-1].role != "tool":
        raise _reasoning_parameter_error(
            "Fireworks reasoning_content can replay only in a completed tool continuation."
        )

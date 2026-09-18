"""Prompt-cache checkpoint preservation across provider payloads."""

from collections.abc import Sequence

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import MessageContentPart
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.errors import ProviderParameterError


def retain_multimodal_cache_boundaries(
    parts: Sequence[MessageContentPart],
) -> tuple[tuple[MessageContentPart, ...], bool]:
    """Drop empty text while retaining checkpoints in the complete media order.

    An empty marked part attaches only to the immediately preceding retained
    block. A leading checkpoint has no representable boundary and is refused.
    Identical colocated markers collapse; conflicting declarations are refused.
    Return whether an unsupported media boundary lost a marker so the decoder
    can disclose that omission instead of moving it to a different prefix.
    """
    retained: list[MessageContentPart] = []
    unsupported = False
    for part in parts:
        if part.kind == "text" and not part.text:
            marker = part.cache_control
            if marker is None:
                continue
            if retained:
                previous = retained[-1]
                if (
                    previous.kind == "text"
                    or previous.kind == "image"
                    or previous.kind == "document"
                ):
                    merged = merge_cache_checkpoint(previous.cache_control, marker)
                    retained[-1] = previous.model_copy(update={"cache_control": merged})
                else:
                    unsupported = True
            else:
                raise ProviderParameterError(
                    message=(
                        "A cache checkpoint needs preceding content; "
                        "remove the leading empty checkpoint."
                    ),
                    param="cache_control",
                    code="invalid_parameter",
                )
            continue
        retained.append(part)
    return tuple(retained), unsupported


def merge_cache_checkpoint(existing: JsonObject | None, incoming: JsonObject) -> JsonObject:
    """Merge one boundary's validated declarations, with the documented five-minute default."""
    if existing is not None and {"ttl": "5m", **existing} != {"ttl": "5m", **incoming}:
        raise ProviderParameterError(
            message=(
                "Remove conflicting cache checkpoints at the same content boundary; "
                "keep one declaration."
            ),
            param="cache_control",
            code="invalid_parameter",
        )
    return incoming if existing is None or "ttl" in incoming else existing


def multimodal_text_cache_blocks(parts: Sequence[MessageContentPart]) -> tuple[JsonObject, ...]:
    """Project aligned text carriers only when a retained text checkpoint exists."""
    if not any(part.kind == "text" and part.cache_control is not None for part in parts):
        return ()
    return tuple(
        {
            "type": "text",
            "text": part.text,
            **({"cache_control": part.cache_control} if part.cache_control is not None else {}),
        }
        for part in parts
        if part.kind == "text"
    )


def cache_markers(request: GatewayRequest) -> tuple[JsonObject, ...]:
    """Return all explicit cache hints without serializing prompt content."""
    markers: list[JsonObject | None] = [request.provider_cache_control]
    markers.extend(tool.cache_control for tool in request.tools)
    for tool in request.provider_server_tools:
        marker = tool.get("cache_control")
        if isinstance(marker, dict):
            markers.append(marker)
    for message in request.messages:
        markers.append(message.cache_control)
        markers.extend(call.cache_control for call in message.tool_calls)
        for part in message.content_parts:
            if part.kind == "text" or part.kind == "image" or part.kind == "document":
                markers.append(part.cache_control)
        blocks = (*message.provider_text_blocks, *(message.provider_anthropic_blocks or ()))
        if message.provider_anthropic_block is not None:
            blocks = (*blocks, message.provider_anthropic_block)
        for block in blocks:
            marker = block.get("cache_control")
            if isinstance(marker, dict):
                markers.append(marker)
    return tuple(marker for marker in markers if marker is not None)

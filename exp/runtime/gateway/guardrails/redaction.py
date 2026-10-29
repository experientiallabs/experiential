"""Preserve provider authority when a guardrail rewrites canonical messages."""

from __future__ import annotations

from collections.abc import Sequence

from exp.runtime.gateway.contracts import GatewayMessage


def restored_provider_authority(
    original: Sequence[GatewayMessage],
    replacement: Sequence[GatewayMessage],
) -> tuple[GatewayMessage, ...] | None:
    """Validate visible edits and restore hidden provider replay authority.

    Hosted classifiers receive only the normal serialized message projection,
    because replay-only reasoning, raw arguments, provider identity, status,
    and phase are excluded from that contract. A valid replacement must keep
    the classifier-visible authenticated prefix exact. The gateway then uses
    the original prefix objects, reattaching every hidden field without asking
    the classifier to receive or echo it.
    """

    def has_authority(message: GatewayMessage) -> bool:
        """Identify fields that must replay byte-exact on a provider continuation."""
        return bool(
            message.provider_reasoning
            or message.provider_item_id is not None
            or message.provider_output_index is not None
            or message.provider_status is not None
            or message.provider_phase is not None
            or message.provider_tool_name is not None
            or message.provider_tool_namespace is not None
            or message.provider_tool_caller is not None
            or message.tool_is_error
            or any(
                call.raw_arguments is not None
                or call.provider_item_id is not None
                or call.provider_output_index is not None
                or call.provider_status is not None
                or call.provider_namespace is not None
                or call.provider_caller is not None
                for call in message.tool_calls
            )
        )

    original_carrier_indexes = tuple(
        index for index, message in enumerate(original) if has_authority(message)
    )
    if not original_carrier_indexes:
        return (
            None if any(has_authority(message) for message in replacement) else tuple(replacement)
        )
    original_fireworks_carriers = tuple(
        index
        for index, message in enumerate(original)
        if any(
            block.kind in {"reasoning_content", "sealed_reasoning_content"}
            for block in message.provider_reasoning
        )
    )
    replacement_fireworks_carriers = tuple(
        index
        for index, message in enumerate(replacement)
        if any(
            block.kind in {"reasoning_content", "sealed_reasoning_content"}
            for block in message.provider_reasoning
        )
    )
    if original_fireworks_carriers:
        if not replacement_fireworks_carriers:
            return (
                tuple(replacement)
                if all(message.role in {"system", "developer", "user"} for message in replacement)
                else None
            )
        if replacement_fireworks_carriers != original_fireworks_carriers:
            return None
        for index in original_fireworks_carriers:
            if original[index] != replacement[index]:
                return None
    bound = original_carrier_indexes[-1]
    if len(replacement) <= bound:
        return None
    original_visible = tuple(message.model_dump(mode="json") for message in original[: bound + 1])
    replacement_visible = tuple(
        message.model_dump(mode="json") for message in replacement[: bound + 1]
    )
    if replacement_visible != original_visible:
        return None
    if any(has_authority(message) for message in replacement[bound + 1 :]):
        return None
    return (*original[: bound + 1], *replacement[bound + 1 :])

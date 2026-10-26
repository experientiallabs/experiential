"""Estimated reasoning tokens, the python mirror of ``reasoning_estimate.rs``.

Anthropic Messages and Bedrock Converse bill a turn's thinking inside
``output_tokens`` and publish no separate count, so ``GatewayUsage`` keeps
``reasoning_tokens`` unknown. Callers that log
``completion_tokens_details.reasoning_tokens`` (Chat) or
``output_tokens_details.reasoning_tokens`` (Responses) then read 0 on a turn
that thought, where OpenRouter reports an estimate: the output total less the
visible answer. The encoded wire reports the same estimate,
``output_tokens - ceil(visible_bytes / 4)`` floored at zero, only when the
provider published no count and the stream carried thinking, and marks it
``estimated_token_details: ["reasoning_tokens"]``. ``GatewayUsage``,
settlement and the ledger are never changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway.contracts import GatewayEvent, GatewayEventKind, GatewayUsage

ESTIMATED_TOKEN_DETAILS = "estimated_token_details"
"""The usage extension naming the token details that are estimates."""

_BYTES_PER_TOKEN = 4
"""Bytes per estimated token, matching the data plane's input approximation."""

_THINKING_KINDS = frozenset(
    {
        GatewayEventKind.THINKING_DELTA,
        GatewayEventKind.THINKING_SIGNATURE,
        GatewayEventKind.REDACTED_THINKING,
        GatewayEventKind.REASONING_SUMMARY_DELTA,
        GatewayEventKind.ENCRYPTED_REASONING,
    }
)


@dataclass
class ReasoningEstimate:
    """Visible output and thinking observed on one attempt's event stream."""

    visible_bytes: int = 0
    thinking: bool = False

    @classmethod
    def from_events(cls, events: tuple[GatewayEvent, ...]) -> ReasoningEstimate:
        """Observe every event of one stream in order."""
        estimate = cls()
        for event in events:
            estimate.observe(event)
        return estimate

    def observe(self, event: GatewayEvent) -> None:
        """Record one event's visible bytes or thinking evidence."""
        if event.kind in (GatewayEventKind.TEXT_DELTA, GatewayEventKind.REFUSAL_DELTA):
            self.visible_bytes += len((event.text_delta or "").encode())
        elif event.kind == GatewayEventKind.TOOL_ARGUMENTS_DELTA:
            self.visible_bytes += len((event.raw_arguments_delta or "").encode())
        elif event.kind in _THINKING_KINDS:
            self.thinking = True

    def estimate(self, usage: GatewayUsage | None) -> int | None:
        """The estimated count; ``None`` if one was reported, none ran or output is unknown."""
        if usage is None or usage.reasoning_tokens is not None or not self.thinking:
            return None
        if usage.output_tokens is None:
            return None
        visible = -(-self.visible_bytes // _BYTES_PER_TOKEN)
        return max(usage.output_tokens - visible, 0)

    def annotate(
        self, wire: JsonObject | None, usage: GatewayUsage | None, details_key: str
    ) -> JsonObject | None:
        """Write the estimate into one encoded usage object and return it."""
        estimate = self.estimate(usage)
        if wire is None or estimate is None:
            return wire
        details = wire.get(details_key)
        updated: JsonObject = dict(cast("JsonObject", details)) if isinstance(details, dict) else {}
        updated["reasoning_tokens"] = estimate
        wire[details_key] = updated
        unreported = wire.get("unreported_token_details")
        if isinstance(unreported, list):
            kept: list[JsonValue] = [name for name in unreported if name != "reasoning_tokens"]
            if kept:
                wire["unreported_token_details"] = kept
            else:
                del wire["unreported_token_details"]
        wire[ESTIMATED_TOKEN_DETAILS] = ["reasoning_tokens"]
        return wire

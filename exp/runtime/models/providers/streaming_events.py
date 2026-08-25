"""Normalized event policies and OpenAI Responses stream helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.models.providers.errors import (
    ProviderResponseError,
    require_integer,
    require_string,
)

SEMANTIC_EVENT_KINDS = frozenset(
    {
        GatewayEventKind.TEXT_DELTA,
        GatewayEventKind.REFUSAL_DELTA,
        GatewayEventKind.REASONING_SUMMARY_DELTA,
        GatewayEventKind.TOOL_CALL_STARTED,
        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
    }
)
TERMINAL_EVENT_KINDS = frozenset(
    {
        GatewayEventKind.COMPLETED,
        GatewayEventKind.INCOMPLETE,
        GatewayEventKind.FAILED,
    }
)

_REASONING_SUMMARY_DELTA = "response.reasoning_summary_text.delta"
_REASONING_SUMMARY_DONE = "response.reasoning_summary_text.done"


@dataclass
class OpenAIReasoningSummaryParser:
    """Validate and normalize one Responses reasoning-summary stream."""

    _summaries: dict[tuple[int, int], str] = field(default_factory=dict)

    def consume(
        self,
        event_type: JsonValue | None,
        payload: JsonObject,
        *,
        create: Callable[..., GatewayEvent],
    ) -> tuple[bool, GatewayEvent | None]:
        """Consume a reasoning-summary event when the event type matches.

        Args:
            event_type: Provider event discriminator.
            payload: Decoded provider event object.
            create: Sequence-aware normalized event factory.

        Returns:
            Whether the event was consumed and any non-empty normalized delta.

        Raises:
            ProviderResponseError: Provider fragments disagree with the done event.
        """
        if event_type not in {_REASONING_SUMMARY_DELTA, _REASONING_SUMMARY_DONE}:
            return False, None
        output_index = require_integer(payload.get("output_index"), "OpenAI reasoning output_index")
        summary_index = require_integer(
            payload.get("summary_index"), "OpenAI reasoning summary_index"
        )
        key = (output_index, summary_index)
        if event_type == _REASONING_SUMMARY_DELTA:
            delta = _optional_string(payload.get("delta"), "OpenAI reasoning summary delta")
            self._summaries[key] = self._summaries.get(key, "") + delta
        else:
            final_text = require_string(payload.get("text"), "OpenAI reasoning summary text")
            streamed = self._summaries.get(key, "")
            if streamed and streamed != final_text:
                raise ProviderResponseError("OpenAI reasoning summary fragments changed at done")
            delta = final_text if not streamed else ""
            self._summaries[key] = final_text
        if not delta:
            return True, None
        return True, create(
            GatewayEventKind.REASONING_SUMMARY_DELTA,
            reasoning_summary_output_index=output_index,
            reasoning_summary_index=summary_index,
            text_delta=delta,
        )


def provider_refusal_failure() -> GatewayFailure:
    """Return the shared sanitized terminal classification for provider refusals."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.REFUSAL,
        safe_message="provider refused the request",
        safe_details={"signal": "content_policy"},
    )


def _optional_string(value: JsonValue | None, label: str) -> str:
    """Accept one optional string while rejecting every other wire type."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ProviderResponseError(f"{label} must be text")
    return value

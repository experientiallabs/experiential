"""Normalized event policies and OpenAI Responses stream helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue, ValidationError

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
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
MAXIMUM_RETAINED_OUTPUT_BYTES = 64 * 1024 * 1024
MAXIMUM_RETAINED_PROVIDER_ENTRIES = 4_096
PROVIDER_OUTPUT_OVERFLOW_MESSAGE = "provider output exceeded the gateway response limit"


def require_retained_provider_entry_capacity(retained_entries: int) -> None:
    """Fail before adding another provider-indexed stream accumulator.

    Args:
        retained_entries: Number of entries already retained by the accumulator.

    Raises:
        ProviderResponseError: The bounded provider-state ceiling is exhausted.
    """
    if retained_entries >= MAXIMUM_RETAINED_PROVIDER_ENTRIES:
        raise ProviderResponseError(PROVIDER_OUTPUT_OVERFLOW_MESSAGE)


def require_retained_provider_byte_capacity(retained_bytes: int, additional_bytes: int) -> None:
    """Fail before retaining provider output beyond the aggregate byte ceiling.

    Args:
        retained_bytes: UTF-8 bytes already retained by the accumulator.
        additional_bytes: UTF-8 bytes the next fragment would retain.

    Raises:
        ProviderResponseError: The bounded provider-state ceiling is exhausted.
    """
    if retained_bytes + additional_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES:
        raise ProviderResponseError(PROVIDER_OUTPUT_OVERFLOW_MESSAGE)


@dataclass
class ProviderOutputRetentionBudget:
    """Enforce one aggregate UTF-8 byte ceiling across retained stream state."""

    _retained_bytes: int = 0

    def retain(self, fragment: str) -> None:
        """Account for one provider-controlled string before retaining it.

        Args:
            fragment: Provider text that an accumulator is about to retain.

        Raises:
            ProviderResponseError: The aggregate retained-output ceiling is exhausted.
        """
        additional_bytes = len(fragment.encode("utf-8"))
        require_retained_provider_byte_capacity(self._retained_bytes, additional_bytes)
        self._retained_bytes += additional_bytes


def retain_provider_entry[KeyT, ValueT](
    entries: dict[KeyT, ValueT], key: KeyT, value: ValueT
) -> None:
    """Insert or replace one provider entry without exceeding the hard ceiling.

    Args:
        entries: Provider-indexed accumulator map.
        key: Provider-controlled entry key.
        value: New retained accumulator value.
    """
    if key not in entries:
        require_retained_provider_entry_capacity(len(entries))
    entries[key] = value


@dataclass
class ProviderToolAccumulator:
    """Provider-order state for one incrementally emitted function call."""

    index: int
    call_id: str
    name: str
    raw_arguments: str = ""
    completed: bool = False

    def complete(self) -> ToolCall:
        """Parse the accumulated raw JSON once the provider closes the call.

        Returns:
            A complete tool call retaining provider-order argument text.

        Raises:
            ProviderResponseError: The raw arguments are not one JSON object.
        """
        try:
            arguments = json.loads(self.raw_arguments)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("streamed tool arguments are not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ProviderResponseError("streamed tool arguments must decode to an object")
        try:
            return ToolCall(
                call_id=self.call_id,
                name=self.name,
                arguments=arguments,
                raw_arguments=self.raw_arguments,
            )
        except ValidationError as exc:
            raise ProviderResponseError("streamed tool call is incomplete") from exc


def start_provider_tool_accumulator(
    tools: dict[int, ProviderToolAccumulator],
    *,
    index: int,
    call_id: str,
    name: str,
    budget: ProviderOutputRetentionBudget,
) -> ProviderToolAccumulator:
    """Retain one provider tool identity under the shared output budget."""
    budget.retain(call_id)
    budget.retain(name)
    tool = ProviderToolAccumulator(index=index, call_id=call_id, name=name)
    retain_provider_entry(tools, index, tool)
    return tool


def append_provider_tool_arguments(
    tool: ProviderToolAccumulator,
    fragment: str,
    *,
    budget: ProviderOutputRetentionBudget,
) -> None:
    """Append one raw argument fragment only after reserving aggregate capacity."""
    budget.retain(fragment)
    tool.raw_arguments += fragment


def require_provider_tool(
    tools: Mapping[int, ProviderToolAccumulator],
    index: int,
) -> ProviderToolAccumulator:
    """Return one previously started provider tool call."""
    try:
        return tools[index]
    except KeyError as exc:
        raise ProviderResponseError("provider emitted arguments before a tool start") from exc


@dataclass
class OpenAIReasoningSummaryParser:
    """Validate and normalize one Responses reasoning-summary stream."""

    budget: ProviderOutputRetentionBudget = field(
        default_factory=ProviderOutputRetentionBudget,
        repr=False,
    )
    _summaries: dict[tuple[int, int], str] = field(default_factory=dict)

    def _append(self, key: tuple[int, int], fragment: str) -> None:
        """Retain one non-empty UTF-8 fragment under both hard ceilings."""
        if key not in self._summaries:
            require_retained_provider_entry_capacity(len(self._summaries))
        self.budget.retain(fragment)
        self._summaries[key] = self._summaries.get(key, "") + fragment

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
            if delta:
                self._append(key, delta)
        else:
            final_text = require_string(payload.get("text"), "OpenAI reasoning summary text")
            streamed = self._summaries.get(key, "")
            if streamed and streamed != final_text:
                raise ProviderResponseError("OpenAI reasoning summary fragments changed at done")
            delta = final_text if not streamed else ""
            if final_text and not streamed:
                self._append(key, final_text)
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

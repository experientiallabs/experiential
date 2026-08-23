"""Build and rewrite the winning normalized completion for output checks."""

from __future__ import annotations

from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import GatewayEvent, GatewayEventKind
from exp.runtime.gateway.guardrails.contracts import GuardrailCompletion, GuardrailToolCall
from exp.runtime.openai_protocol.response import assistant_message


def completion_from_events(events: tuple[GatewayEvent, ...]) -> GuardrailCompletion:
    """Project one event sequence into the output-check subject.

    Args:
        events: Winning normalized provider events.

    Returns:
        Text, refusal presence, and completed tool calls.
    """
    refusal = any(event.kind is GatewayEventKind.REFUSAL_DELTA for event in events)
    assistant = assistant_message(events)
    if assistant is None:
        text = ""
        tool_calls: tuple[GuardrailToolCall, ...] = ()
    else:
        text = assistant.content or ""
        tool_calls = tuple(_tool_call(call) for call in assistant.tool_calls)
    return GuardrailCompletion(text=text, refusal=refusal, tool_calls=tool_calls)


def apply_text_replacement(
    events: tuple[GatewayEvent, ...],
    replacement: str,
) -> tuple[GatewayEvent, ...]:
    """Replace text deltas with one rewritten delta and leave other events intact.

    Tool-call events are never rewritten. Callers must block those completions
    instead of asking this helper to edit arguments.

    Args:
        events: Winning normalized provider events.
        replacement: Sanitized replacement text.

    Returns:
        A new event sequence with a single text delta when text was present,
        or with one inserted text delta before the terminal event.
    """
    rewritten: list[GatewayEvent] = []
    inserted = False
    for event in events:
        if event.kind is GatewayEventKind.TEXT_DELTA:
            if inserted:
                continue
            rewritten.append(event.model_copy(update={"text_delta": replacement}))
            inserted = True
            continue
        if (
            event.kind
            in {
                GatewayEventKind.COMPLETED,
                GatewayEventKind.INCOMPLETE,
                GatewayEventKind.FAILED,
            }
            and not inserted
        ):
            rewritten.append(
                GatewayEvent(
                    kind=GatewayEventKind.TEXT_DELTA,
                    sequence_number=event.sequence_number,
                    text_delta=replacement,
                )
            )
            inserted = True
        rewritten.append(event)
    return tuple(rewritten)


def _tool_call(call: ToolCall) -> GuardrailToolCall:
    """Copy one completed tool call's identity and raw argument text."""
    return GuardrailToolCall(
        call_id=call.call_id,
        name=call.name,
        arguments=call.arguments_json(),
    )

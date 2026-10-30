"""Readable calibration transcripts for one normalized production trace."""

from __future__ import annotations

from rich.console import Console

from exp.common.models import ModelSnapshot
from exp.common.traces import Trace, TraceSpan
from exp.common.traces.attributes import assistant_completion, jsonish_text, user_input


def model_display_name(model: ModelSnapshot) -> str:
    """Return the plain provider and model identity without internal hashes.

    Args:
        model: Model snapshot with provider, model, and optional revision fields.

    Returns:
        Human-readable exact provider and model identity.
    """
    suffix = f" (revision {model.revision})" if model.revision is not None else ""
    return f"{model.provider}/{model.model_id}{suffix}"


def render_trace(console: Console, trace: Trace, *, character_limit: int | None) -> None:
    """Render one normalized trace as a role-separated readable transcript.

    Args:
        console: Destination for the rendered fields.
        trace: Verified immutable normalized production trace.
        character_limit: Maximum characters per field, or ``None`` for the full value.
    """
    render_field(console, "Original user request", trace.task, character_limit=character_limit)
    if trace.initial_context:
        render_field(
            console,
            "Initial context",
            jsonish_text(trace.initial_context),
            character_limit=character_limit,
        )
    completion_positions = tuple(
        index for index, span in enumerate(trace.spans) if assistant_completion(span.attributes)
    )
    final_completion = completion_positions[-1] if completion_positions else None
    for index, span in enumerate(trace.spans):
        _render_span(
            console,
            span,
            final_response=index == final_completion,
            character_limit=character_limit,
        )
    outcome = trace.outcome
    if outcome is None:
        render_field(console, "Final outcome", "Not recorded", character_limit=character_limit)
        return
    outcome_text = outcome.status
    if outcome.outcome_name is not None:
        outcome_text += f" ({outcome.outcome_name})"
    render_field(console, "Final outcome", outcome_text, character_limit=character_limit)
    if outcome.failure is not None:
        render_field(
            console,
            "Final failure",
            f"{outcome.failure.code.value}: {outcome.failure.message} "
            f"(retryable={str(outcome.failure.retryable).lower()})",
            character_limit=character_limit,
        )


def _render_span(
    console: Console,
    span: TraceSpan,
    *,
    final_response: bool,
    character_limit: int | None,
) -> None:
    """Render recognized assistant, tool-call, tool-result, and failure evidence.

    Args:
        console: Destination for the rendered fields.
        span: One normalized chronological trace span.
        final_response: Whether this span holds the last captured assistant response.
        character_limit: Maximum characters per field, or ``None`` for the full value.
    """
    attributes = span.attributes
    operation = attributes.get("gen_ai.operation.name")
    tool_name = attributes.get("gen_ai.tool.name")
    arguments = attributes.get("gen_ai.tool.call.arguments")
    result = attributes.get("gen_ai.tool.message")
    if result is None:
        result = attributes.get("gen_ai.tool.output")
    completion = assistant_completion(attributes)
    user_message = user_input(attributes)
    console.print(f"\nEvidence span {span.span_id} ({span.name})", style="dim", markup=False)
    if user_message is not None:
        render_field(console, "User message", user_message, character_limit=character_limit)
    if span.model is not None:
        render_field(
            console,
            "Assistant model",
            model_display_name(span.model),
            character_limit=character_limit,
        )
    if completion:
        render_field(
            console,
            "Final response" if final_response else "Assistant message",
            completion,
            character_limit=character_limit,
        )
    if operation != "execute_tool" and isinstance(tool_name, str):
        render_field(console, "Tool call", tool_name, character_limit=character_limit)
        if arguments is not None:
            render_field(
                console,
                "Tool arguments",
                jsonish_text(arguments),
                character_limit=character_limit,
            )
    if operation == "execute_tool":
        result_name = tool_name if isinstance(tool_name, str) else span.name
        result_text = result_name
        if result is not None:
            result_text += "\n" + jsonish_text(result)
        render_field(console, "Tool result", result_text, character_limit=character_limit)
    if span.failure is not None:
        render_field(
            console,
            "Span failure",
            f"{span.failure.code.value}: {span.failure.message}",
            character_limit=character_limit,
        )


def span_evidence_text(span: TraceSpan) -> str:
    """Return role- and tool-labeled visible content from one cited span.

    Args:
        span: Judge-cited normalized trace span.

    Returns:
        Captured evidence text without inferred facts.
    """
    attributes = span.attributes
    pieces: list[str] = []
    user_message = user_input(attributes)
    completion = assistant_completion(attributes)
    if user_message is not None:
        pieces.append("User message: " + user_message)
    if completion is not None:
        pieces.append("Assistant message: " + completion)
    tool_name = attributes.get("gen_ai.tool.name")
    if isinstance(tool_name, str):
        pieces.append("Tool: " + tool_name)
    arguments = attributes.get("gen_ai.tool.call.arguments")
    if arguments is not None:
        pieces.append("Tool arguments: " + jsonish_text(arguments))
    result = attributes.get("gen_ai.tool.message")
    if result is None:
        result = attributes.get("gen_ai.tool.output")
    if result is not None:
        pieces.append("Tool result: " + jsonish_text(result))
    if span.failure is not None:
        pieces.append(f"Failure: {span.failure.code.value}: {span.failure.message}")
    return "\n".join(pieces) if pieces else f"Captured span {span.name}"


def render_field(console: Console, label: str, value: str, *, character_limit: int | None) -> None:
    """Render one safely wrapped transcript field with truthful truncation.

    Args:
        console: Destination for the rendered field.
        label: Human role or evidence label.
        value: Captured field text.
        character_limit: Maximum characters, or ``None`` for the full value.
    """
    shown = value
    if character_limit is not None and len(value) > character_limit:
        omitted = len(value) - character_limit
        shown = (
            value[:character_limit]
            + f"\n... [truncated {omitted} characters; use --page for the full transcript]"
        )
    console.print(f"{label}:", style="bold", markup=False)
    console.print(shown, markup=False, overflow="fold")

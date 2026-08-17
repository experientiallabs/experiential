"""Role-separated transcript rendering for manual judge CLI review."""

from __future__ import annotations

import json

from pydantic import JsonValue
from rich.console import Console

from wmo.common.models import ModelSnapshot
from wmo.common.traces import Trace, TraceSpan


def render_trace(console: Console, trace: Trace, *, character_limit: int | None) -> None:
    """Render one normalized trace as a role-separated readable transcript.

    Args:
        console: Command-owned console used for operator output.
        trace: Verified immutable normalized production trace.
        character_limit: Maximum characters per field, or ``None`` for the full value.
    """
    render_field(console, "User / task", trace.task, character_limit=character_limit)
    if trace.initial_context:
        render_field(
            console,
            "Initial context",
            jsonish_text(trace.initial_context),
            character_limit=character_limit,
        )
    for span in trace.spans:
        render_span(console, span, character_limit=character_limit)
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


def render_span(console: Console, span: TraceSpan, *, character_limit: int | None) -> None:
    """Render recognized assistant, tool-call, tool-result, and failure evidence.

    Args:
        console: Command-owned console used for operator output.
        span: One normalized chronological trace span.
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
    user_input = user_message(attributes)
    if user_input is not None:
        render_field(console, "User message", user_input, character_limit=character_limit)
    if span.model is not None:
        render_field(
            console, "Assistant / model", model_name(span.model), character_limit=character_limit
        )
    if completion:
        render_field(console, "Assistant output", completion, character_limit=character_limit)
    if operation != "execute_tool" and isinstance(tool_name, str):
        render_field(console, "Tool call", tool_name, character_limit=character_limit)
        if arguments is not None:
            render_field(
                console, "Tool arguments", jsonish_text(arguments), character_limit=character_limit
            )
    if operation == "execute_tool":
        render_field(
            console,
            "Tool result",
            tool_name if isinstance(tool_name, str) else span.name,
            character_limit=character_limit,
        )
        if result is not None:
            render_field(
                console, "Tool output", jsonish_text(result), character_limit=character_limit
            )
    if span.failure is not None:
        render_field(
            console,
            "Span failure",
            f"{span.failure.code.value}: {span.failure.message}",
            character_limit=character_limit,
        )


def assistant_completion(attributes: dict[str, JsonValue]) -> str | None:
    """Extract readable assistant content from supported normalized attributes.

    Args:
        attributes: Canonical normalized span attributes.

    Returns:
        Assistant content when captured, otherwise ``None``.
    """
    for key in ("gen_ai.completion", "gen_ai.response.text"):
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return last_role_message(
        decoded_json_value(attributes.get("gen_ai.output.messages")),
        frozenset({"assistant", "model"}),
    )


def user_message(attributes: dict[str, JsonValue]) -> str | None:
    """Extract the latest readable user message from one normalized span.

    Args:
        attributes: Canonical normalized span attributes.

    Returns:
        Latest user content when captured, otherwise the legacy prompt field.
    """
    text = last_role_message(
        decoded_json_value(attributes.get("gen_ai.input.messages")),
        frozenset({"user", "human"}),
    )
    if text is not None:
        return text
    prompt = attributes.get("gen_ai.prompt")
    return prompt.strip() if isinstance(prompt, str) and prompt.strip() else None


def last_role_message(value: JsonValue, roles: frozenset[str]) -> str | None:
    """Return the latest visible message for one set of transcript roles.

    Args:
        value: Decoded normalized message collection.
        roles: Accepted lowercase role names.

    Returns:
        Latest nonempty message content, or ``None`` when absent.
    """
    if not isinstance(value, list):
        return None
    for item in reversed(value):
        if not isinstance(item, dict) or item.get("role") not in roles:
            continue
        text = message_text(item.get("content"))
        if text is not None:
            return text
    return None


def decoded_json_value(value: JsonValue | None) -> JsonValue:
    """Decode JSON-encoded semantic attributes without guessing malformed text.

    Args:
        value: Native or JSON-encoded normalized attribute value.

    Returns:
        Decoded JSON value, or the original value when it is not encoded JSON.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def message_text(value: JsonValue | None) -> str | None:
    """Read plain text from one normalized message content value.

    Args:
        value: String or structured content parts.

    Returns:
        Joined text content, or ``None`` when no text was captured.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, list):
        return None
    texts = tuple(
        item["text"].strip()
        for item in value
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()
    )
    return "\n".join(texts) if texts else None


def jsonish_text(value: JsonValue) -> str:
    """Format native or JSON-encoded transcript evidence for a human.

    Args:
        value: Captured transcript value.

    Returns:
        Stable indented JSON when possible, otherwise its original text.
    """
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(decoded, indent=2, sort_keys=True, ensure_ascii=False)


def render_field(console: Console, label: str, value: str, *, character_limit: int | None) -> None:
    """Render one safely wrapped transcript field with truthful truncation.

    Args:
        console: Command-owned console used for operator output.
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


def model_name(model: ModelSnapshot) -> str:
    """Return the plain provider and model identity without internal hashes.

    Args:
        model: Model snapshot with provider, model, and optional revision fields.

    Returns:
        Human-readable exact provider and model identity.
    """
    suffix = f" (revision {model.revision})" if model.revision is not None else ""
    return f"{model.provider}/{model.model_id}{suffix}"

"""Optional source-backed instruction cleanup without provider-side calls in mining."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject, canonical_json_bytes
from wmo.common.core.text import normalize_durable_text
from wmo.common.tasks import ToolSchema
from wmo.common.traces import Trace

_WORD_PATTERN = re.compile(r"[a-z0-9_]+(?:[./-][a-z0-9_]+)*", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
_TOOL_REFERENCE_PATTERN = re.compile(
    r"\b(?:call|invoke|run)\s+(?:the\s+)?(?:tool\s+)?[`\"']?([a-z][a-z0-9_.-]*)"
    r"|\buse\s+(?:the\s+)?tool\s+[`\"']?([a-z][a-z0-9_.-]*)",
    re.IGNORECASE,
)
_BACKTICK_IDENTIFIER_PATTERN = re.compile(r"`([a-z][a-z0-9_.-]*)`", re.IGNORECASE)
_ALLOWED_NEW_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "please",
        "the",
        "to",
        "using",
        "with",
        "your",
    }
)


class InstructionCleanupModel(Protocol):
    """Injected small-model interface for one optional instruction cleanup proposal."""

    def cleanup_instruction(
        self,
        *,
        instruction: str,
        initial_context: JsonObject,
        tools: tuple[ToolSchema, ...],
    ) -> str:
        """Propose a clearer instruction without changing the source task.

        Args:
            instruction: Original production instruction.
            initial_context: Request-visible source context.
            tools: Tools available at task start.

        Returns:
            Proposed cleaned instruction. WMO independently validates it before use.
        """


@dataclass(frozen=True)
class InstructionCleanupResult:
    """One cleanup proposal's acceptance decision and source-preserving final instruction.

    Args:
        instruction: Final instruction, always source text when validation rejects a proposal.
        proposed_instruction: Normalized model proposal, when cleanup was requested.
        accepted: Whether source back-checks approved the proposal.
        reason: Deterministic acceptance or rejection reason for task review.
    """

    instruction: str
    proposed_instruction: str | None
    accepted: bool
    reason: str


def clean_instruction(
    trace: Trace,
    model: InstructionCleanupModel | None,
) -> InstructionCleanupResult:
    """Optionally clean one task instruction and reject unsafe changes against source evidence.

    The injected interface is never resolved from credentials and this function never invokes a
    provider on its own. A caller that passes ``None`` receives the original source instruction.

    Args:
        trace: Canonical source trace that supplies instruction, tools, context, and later evidence.
        model: Explicit injected cleanup model, or ``None`` to keep the source instruction
            unchanged.

    Returns:
        The accepted cleanup or a reviewable fallback to the original source instruction.
    """
    source_instruction = normalize_durable_text(trace.task.strip())
    if model is None:
        return InstructionCleanupResult(
            instruction=source_instruction,
            proposed_instruction=None,
            accepted=False,
            reason="cleanup not requested",
        )
    proposed = normalize_durable_text(
        " ".join(
            model.cleanup_instruction(
                instruction=source_instruction,
                initial_context=trace.initial_context,
                tools=trace.tools,
            ).split()
        )
    )
    if not proposed:
        return _rejected(source_instruction, proposed, "cleanup proposal is blank")
    if proposed == source_instruction:
        return InstructionCleanupResult(
            instruction=source_instruction,
            proposed_instruction=proposed,
            accepted=True,
            reason="cleanup matches source instruction",
        )
    reason = _source_back_check(trace, source_instruction, proposed)
    if reason is not None:
        return _rejected(source_instruction, proposed, reason)
    return InstructionCleanupResult(
        instruction=proposed,
        proposed_instruction=proposed,
        accepted=True,
        reason="source back-check passed",
    )


def _rejected(
    source_instruction: str, proposed_instruction: str, reason: str
) -> InstructionCleanupResult:
    """Build an explicit source-preserving rejection result."""
    return InstructionCleanupResult(
        instruction=source_instruction,
        proposed_instruction=proposed_instruction,
        accepted=False,
        reason=reason,
    )


def _source_back_check(
    trace: Trace, source_instruction: str, proposed_instruction: str
) -> str | None:
    """Reject answer leakage, invented tools, dropped requirements, and new hard facts."""
    lowered_source = source_instruction.casefold()
    lowered_proposal = proposed_instruction.casefold()
    tool_reason = _invented_tool_reason(trace.tools, lowered_proposal)
    if tool_reason is not None:
        return tool_reason
    leakage_reason = _answer_leakage_reason(trace, lowered_source, lowered_proposal)
    if leakage_reason is not None:
        return leakage_reason
    missing = _missing_source_terms(lowered_source, lowered_proposal)
    if missing:
        return f"cleanup proposal drops source requirement term {missing[0]!r}"
    invented = _invented_hard_facts(trace, lowered_source, lowered_proposal)
    if invented:
        return f"cleanup proposal invents unsupported requirement term {invented[0]!r}"
    return None


def _invented_tool_reason(tools: tuple[ToolSchema, ...], proposed: str) -> str | None:
    """Reject a proposal that names an unavailable tool as an action requirement."""
    allowed = {tool.name.casefold() for tool in tools}
    references = {
        reference for groups in _TOOL_REFERENCE_PATTERN.findall(proposed) for reference in groups
    }
    references.update(
        reference
        for reference in _BACKTICK_IDENTIFIER_PATTERN.findall(proposed)
        if any(character in reference for character in ("_", ".", "-"))
    )
    for reference in sorted(reference.casefold() for reference in references if reference):
        if reference not in allowed:
            return f"cleanup proposal invents unavailable tool {reference!r}"
    return None


def _answer_leakage_reason(trace: Trace, source: str, proposed: str) -> str | None:
    """Reject later answer or observation content newly copied into the task instruction."""
    for output in _later_outputs(trace):
        lowered_output = output.casefold()
        if len(lowered_output) >= 8 and lowered_output in proposed and lowered_output not in source:
            return "cleanup proposal leaks later source output"
        output_numbers = set(_NUMBER_PATTERN.findall(lowered_output)) - set(
            _NUMBER_PATTERN.findall(source)
        )
        if any(number in proposed for number in output_numbers):
            return "cleanup proposal leaks a later source numeric answer"
        tokens = _tokens(lowered_output)
        for index in range(len(tokens) - 1):
            phrase = " ".join(tokens[index : index + 2])
            if phrase in proposed and phrase not in source and len(phrase) >= 7:
                return "cleanup proposal leaks later source answer text"
    return None


def _later_outputs(trace: Trace) -> tuple[str, ...]:
    """Collect assistant completions and tool observations that occur after the initial request."""
    values: list[str] = []
    for span in trace.spans:
        for key in ("gen_ai.completion", "gen_ai.tool.message", "gen_ai.tool.output"):
            value = _text_value(span.attributes.get(key))
            if value:
                values.append(value)
        values.extend(_output_message_texts(span.attributes.get("gen_ai.output.messages")))
    return tuple(values)


def _output_message_texts(value: JsonValue | None) -> tuple[str, ...]:
    """Extract assistant text from native or JSON-encoded GenAI output message arrays."""
    decoded = _decoded_json_value(value)
    messages = decoded.get("messages") if isinstance(decoded, dict) else decoded
    if not isinstance(messages, list):
        return ()
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not isinstance(role, str) or role.casefold() not in {"assistant", "model", "ai"}:
            continue
        text = _message_text(message.get("content"))
        if text is None:
            text = _message_text(message.get("text"))
        if text is not None:
            texts.append(text)
    return tuple(texts)


def _decoded_json_value(value: JsonValue | None) -> JsonValue | None:
    """Decode a JSON string when possible while preserving ordinary source text."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _message_text(value: JsonValue | None) -> str | None:
    """Read plain or multipart text from one standard GenAI message payload."""
    if isinstance(value, str) and value.strip():
        return normalize_durable_text(value.strip())
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            return normalize_durable_text(text.strip())
        return None
    if not isinstance(value, list):
        return None
    texts = [
        item["text"].strip()
        for item in value
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
        and item["text"].strip()
    ]
    return normalize_durable_text("\n".join(texts)) if texts else None


def _text_value(value: JsonValue | None) -> str | None:
    """Read a plain or JSON-encoded output string from canonical source attributes."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        if isinstance(decoded, str) and decoded.strip():
            return normalize_durable_text(decoded.strip())
        if not isinstance(decoded, str):
            rendered = canonical_json_bytes(decoded).decode("utf-8")
            return rendered if rendered != "null" else None
    return None


def _missing_source_terms(source: str, proposed: str) -> tuple[str, ...]:
    """Require source-specific content terms to survive a supposedly cleanup-only rewrite."""
    required = {
        token
        for token in _tokens(source)
        if len(token) >= 5 or token.isnumeric() or "_" in token or "/" in token
    }
    missing = sorted(token for token in required if token not in _tokens(proposed))
    return tuple(missing)


def _invented_hard_facts(trace: Trace, source: str, proposed: str) -> tuple[str, ...]:
    """Reject unsupported identifiers, numbers, and requirement terms newly introduced."""
    allowed_text = " ".join(
        (
            source,
            canonical_json_bytes(trace.initial_context).decode("utf-8"),
            " ".join(tool.name + " " + tool.description for tool in trace.tools),
        )
    ).casefold()
    known = set(_tokens(allowed_text)) | _ALLOWED_NEW_WORDS
    proposed_tokens = set(_tokens(proposed))
    invented = sorted(
        token
        for token in proposed_tokens
        if token not in known and (token.isnumeric() or "_" in token or len(token) >= 5)
    )
    return tuple(invented)


def _tokens(value: str) -> tuple[str, ...]:
    """Return normalized word-like tokens for deterministic source back-checks."""
    return tuple(_WORD_PATTERN.findall(value.casefold()))

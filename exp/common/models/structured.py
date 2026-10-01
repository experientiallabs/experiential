"""Normalization of visible provider text that carries a strict JSON protocol."""

from __future__ import annotations

import json

from exp.common.core.artifacts import JsonValue
from exp.common.models.model import ModelFinishReason, ModelResponse

_FENCE = "```"


class StructuredReplyError(ValueError):
    """A completed reply could not carry the strict JSON payload it promised."""


def structured_reply_json(response: ModelResponse) -> JsonValue:
    """Return the strict JSON payload of one completed single-shot reply.

    Shared handling for callers that prompt a model for a JSON-only answer:
    a reply that stopped at its output-token limit, carried no visible text
    (tool calls only), or failed strict JSON parsing after fence
    normalization is rejected loudly. Shape validation of the parsed value
    stays with the caller's own contract.

    Args:
        response: The completed model response.

    Returns:
        The parsed JSON value of the reply's visible text.

    Raises:
        StructuredReplyError: The reply was truncated, textless, or non-JSON;
            rerun the call or raise the output-token budget.
    """
    if response.finish_reason is ModelFinishReason.LENGTH:
        msg = "the model stopped at its output-token limit; raise the output-token budget"
        raise StructuredReplyError(msg)
    content = response.output.content
    if content is None:
        raise StructuredReplyError("the model returned no text output; rerun the call")
    try:
        parsed: JsonValue = json.loads(structured_json_text(content))
    except json.JSONDecodeError as error:
        msg = "the model returned non-JSON output; rerun the call"
        raise StructuredReplyError(msg) from error
    return parsed


def structured_json_text(content: str) -> str:
    """Return the JSON payload carried by one visible provider text response.

    Supported providers, including Anthropic models served through Bedrock, commonly return a
    JSON-only answer inside a single Markdown code fence and sometimes precede it with an
    explanation, even when the prompt and schema forbid both. This returns the body of exactly one
    unlabeled or ``json``-labeled fenced block that closes the response, and otherwise returns the
    stripped text unchanged. Callers keep parsing strict JSON, so unfenced prose, several fenced
    blocks, text after the closing fence, other fence labels, and truncated output all stay
    invalid.

    Args:
        content: Visible assistant text returned by the provider.

    Returns:
        Candidate JSON text with surrounding whitespace and at most one Markdown fence removed.
    """
    text = content.strip()
    if text.count(_FENCE) != 2 or not text.endswith(_FENCE):
        return text
    start = text.index(_FENCE) + len(_FENCE)
    inner = text[start : len(text) - len(_FENCE)]
    head, separator, remainder = inner.partition("\n")
    label = head.strip()
    if label and label.lower() != "json":
        return inner.strip() if not separator else text
    if separator:
        return remainder.strip()
    return inner.strip()

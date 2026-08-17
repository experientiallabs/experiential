"""Normalization of visible provider text that carries a strict JSON protocol."""

from __future__ import annotations

_FENCE = "```"


def structured_json_text(content: str) -> str:
    """Return the JSON payload carried by one visible provider text response.

    Supported providers, including Anthropic models served through Bedrock, commonly return a
    JSON-only answer inside a single Markdown code fence and sometimes precede it with an
    explanation, even when the prompt and schema forbid both. This returns the body of exactly one
    fenced block when the response contains exactly one, and otherwise returns the stripped text
    unchanged. Callers keep parsing strict JSON, so unfenced prose, several fenced blocks, and
    truncated output all stay invalid.

    Args:
        content: Visible assistant text returned by the provider.

    Returns:
        Candidate JSON text with surrounding whitespace and at most one Markdown fence removed.
    """
    text = content.strip()
    if text.count(_FENCE) != 2:
        return text
    start = text.index(_FENCE) + len(_FENCE)
    inner = text[start : text.index(_FENCE, start)]
    head, separator, remainder = inner.partition("\n")
    label = head.strip()
    if separator and (not label or label.isalnum()):
        return remainder.strip()
    return inner.strip()

"""Normalization of visible provider text that carries a strict JSON protocol."""

from __future__ import annotations

_FENCE = "```"


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

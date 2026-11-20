"""Ignore Chat-only generation controls on Anthropic ``count_tokens`` bodies.

Claude Desktop in third-party mode (``claude-desktop-3p``) speaks OpenAI Chat
Completions for generation and reuses the same body extras on Anthropic's
``/v1/messages/count_tokens``, so a count arrives carrying Chat's
``reasoning_effort`` (and, from the same family of clients, ``verbosity`` and
``service_tier``). None of them is a Messages field, and none can change how
many tokens the prompt holds: a count dispatches nothing and generates
nothing. Each is removed with disclosure before the manifest reads the body,
so the count answers instead of refusing the client's whole request.

This applies to ``count_tokens`` only. On ``/v1/messages`` these fields would
change what the call does (effort, reply length, priced capacity), so that
surface keeps its named refusal and the caller's request is never silently
weakened.
"""

from __future__ import annotations

from typing import Final

from exp.common.core.artifacts import JsonObject

COUNT_NEUTRAL_FIELDS: Final = ("reasoning_effort", "verbosity", "service_tier")
"""Chat-only top-level fields a token count ignores, in disclosure order."""


def count_neutral_disclosure(field: str) -> str:
    """Return the disclosure recorded when ``field`` is ignored by a count."""
    return f"{field}->ignored(count_tokens)"


def drop_count_neutral_fields(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove Chat-only fields that cannot change a ``count_tokens`` answer.

    A null value asks for nothing and is removed silently; any other value is
    removed with a disclosure naming the field. The value is not validated,
    since nothing reads it.

    Args:
        payload: Parsed Anthropic ``count_tokens`` body.

    Returns:
        The payload without the count-neutral fields, and any disclosures.
    """
    present = [field for field in COUNT_NEUTRAL_FIELDS if field in payload]
    if not present:
        return payload, ()
    remaining = {key: value for key, value in payload.items() if key not in present}
    disclosures = tuple(
        count_neutral_disclosure(field) for field in present if payload[field] is not None
    )
    return remaining, disclosures

"""OpenAI's cache-write lifetime split rendered on gateway usage."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.stream_contracts import GatewayUsage


def cache_write_lifetimes(usage: GatewayUsage) -> JsonObject | None:
    """Return ``cache_write_tokens_details`` when a 12-hour write count was observed.

    OpenAI reports ``ttl_30m_tokens`` and ``ttl_12h_tokens`` beside
    ``cache_write_tokens``; the two never overlap and sum to the writes
    (https://developers.openai.com/api/docs/guides/prompt-caching). Any one-hour
    subset (Anthropic's lifetime) is excluded from the 30-minute count. Mirrors
    the native encoder's ``cache_write_lifetimes``.

    Args:
        usage: Normalized gateway usage.

    Returns:
        The split object, or None when no 12-hour count was observed.
    """
    half_day = usage.cache_creation_12h_input_tokens
    if half_day is None:
        return None
    written = usage.cache_creation_input_tokens or 0
    short = max(written - half_day - (usage.cache_creation_1h_input_tokens or 0), 0)
    return {"ttl_30m_tokens": short, "ttl_12h_tokens": half_day}

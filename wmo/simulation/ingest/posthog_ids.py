"""Deterministic W3C identifiers for PostHog's opaque trace and span keys."""

from __future__ import annotations

import hashlib
import re

_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")


def posthog_w3c_id(value: str, *, kind: str, namespace: str) -> str:
    """Preserve valid W3C IDs or deterministically map an opaque PostHog ID.

    Args:
        value: Source trace or span identity.
        kind: ``trace`` for a 32-character identity, otherwise ``span``.
        namespace: Stable source-role namespace preventing cross-role collisions.

    Returns:
        Lowercase nonzero W3C-shaped trace or span identity.
    """
    normalized = value.casefold()
    pattern = _TRACE_ID_PATTERN if kind == "trace" else _SPAN_ID_PATTERN
    if pattern.fullmatch(normalized) and set(normalized) != {"0"}:
        return normalized
    width = 32 if kind == "trace" else 16
    return hashlib.sha256(f"posthog\0{namespace}\0{value}".encode()).hexdigest()[:width]

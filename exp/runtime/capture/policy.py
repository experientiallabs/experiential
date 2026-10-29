"""Define exact provider host filters without importing the interception backend."""

from __future__ import annotations

import re

DEFAULT_CAPTURE_DOMAINS = ("api.openai.com", "chatgpt.com", "api.anthropic.com")
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


def validate_domains(domains: tuple[str, ...]) -> tuple[str, ...]:
    """Validate normalized public DNS names and return unique hosts in sorted order.

    Args:
        domains: One to 32 lowercase hostnames without trailing dots.

    Returns:
        Sorted distinct exact hostnames, without wildcards or local aliases.

    Raises:
        ValueError: A hostname is invalid or the input exceeds the target limit.
    """
    if not domains or len(domains) > 32:
        raise ValueError("Capture needs between 1 and 32 explicit provider hostnames.")
    normalized = tuple(sorted(set(domains)))
    for domain in normalized:
        labels = domain.split(".")
        if (
            len(domain) > 253
            or len(labels) < 2
            or any(not _LABEL.fullmatch(label) for label in labels)
            or labels[-1] in {"localhost", "local", "internal", "test", "invalid"}
            or labels[-1].isdigit()
        ):
            raise ValueError(f"Invalid provider hostname {domain!r}; use a literal DNS name.")
    return normalized

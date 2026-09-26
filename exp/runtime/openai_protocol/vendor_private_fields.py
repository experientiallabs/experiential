"""Drop underscore-prefixed vendor-private top-level fields with disclosure.

Routers and proxies in front of the gateway attach their own bookkeeping to the
Chat body under an underscore-prefixed name (``_omnirouteSkipContextRelay`` is
the one seen most). The name is private to that hop and carries no meaning for
any downstream provider, so the field is removed here, before the compatibility
manifest sees the body, and named in the request's disclosed
``ignored_parameters``.

This is a whole-class rule rather than one entry per spelling: an underscore
prefix is the convention such hops already use, and each new spelling would
otherwise be a fresh pre-admission rejection. Dropping is never silent, so a
caller that believed the field did something still learns that it did not.
Every other unknown top-level field stays rejected by name.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject

VENDOR_PRIVATE_PREFIX = "_"
"""Prefix marking a top-level field as private to an intermediate hop."""

MAXIMUM_DISCLOSED_FIELDS = 8
"""How many dropped field names are listed before the rest become a count."""

MAXIMUM_DISCLOSED_NAME_CHARACTERS = 64
"""Longest disclosed field name; a longer one is shortened to this prefix."""

_TRUNCATION_MARKER = "..."


def vendor_private_disclosure(field: str) -> str:
    """Build the disclosure naming one dropped vendor-private field.

    Args:
        field: Top-level field name as the caller spelled it.

    Returns:
        The ``path->dropped(reason)`` disclosure recorded for the request.
    """
    return f"{field}->dropped(vendor_private)"


def _disclosed_name(field: str) -> str:
    """Shorten one field name to the length the disclosure will carry.

    Args:
        field: Top-level field name as the caller spelled it.

    Returns:
        The name unchanged when it is within the limit, otherwise its leading
        characters followed by a marker, which still identifies a real router
        field while refusing to echo an arbitrary-length one.
    """
    if len(field) <= MAXIMUM_DISCLOSED_NAME_CHARACTERS:
        return field
    return f"{field[:MAXIMUM_DISCLOSED_NAME_CHARACTERS]}{_TRUNCATION_MARKER}"


def drop_vendor_private_fields(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove underscore-prefixed top-level fields from one request body.

    Disclosures are bounded in both count and name length. A streaming response
    repeats the whole disclosure list on every chunk, so echoing caller-chosen
    names verbatim would let one request inflate a long completion. Fields past
    the limit are still reported, as a count rather than by name.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The original payload when no such field is present; otherwise a shallow
        copy without them, paired with the disclosures recorded for the request
        in the order the caller sent them.
    """
    dropped = tuple(field for field in payload if field.startswith(VENDOR_PRIVATE_PREFIX))
    if not dropped:
        return payload, ()
    remaining = {key: value for key, value in payload.items() if key not in dropped}
    named = dropped[:MAXIMUM_DISCLOSED_FIELDS]
    disclosures = [vendor_private_disclosure(_disclosed_name(field)) for field in named]
    unnamed = len(dropped) - len(named)
    if unnamed:
        disclosures.append(f"vendor_private->dropped({unnamed}_more)")
    return remaining, tuple(disclosures)

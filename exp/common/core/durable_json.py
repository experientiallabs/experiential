"""Durable JSON normalization shared by request projection and batch validation."""

from __future__ import annotations

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.core.text import normalize_durable_text

_REPLACEMENT_CHAR = "\ufffd"


def _replace_in_text(text: str) -> tuple[str, int]:
    """Return ``text`` made durable by the engine's normalizer, plus the count.

    ``normalize_durable_text`` replaces NUL and each LONE surrogate with U+FFFD
    and folds a valid surrogate pair into its scalar (the value jsonb decodes
    the pair to anyway), so the replacements are exactly the U+FFFD it added.
    """
    normalized = normalize_durable_text(text)
    return normalized, normalized.count(_REPLACEMENT_CHAR) - text.count(_REPLACEMENT_CHAR)


def count_unstorable_text(value: JsonValue) -> int:
    """Count the code points inside ``value`` that no Postgres text column can hold.

    NUL and each LONE UTF-16 surrogate, in strings AND object keys, anywhere in
    the JSON tree; a valid surrogate pair is one storable scalar and counts as
    nothing. The batch lane's submit boundary refuses a JSONL line or filename
    on a non-zero count (``the host's batch admission``)
    instead of letting the job document's ``jsonb`` insert fail as a 500.
    """
    match value:
        case str():
            return _replace_in_text(value)[1]
        case dict():
            return sum(
                _replace_in_text(key)[1] + count_unstorable_text(item)
                for key, item in value.items()
            )
        case list():
            return sum(count_unstorable_text(item) for item in value)
        case _:
            return 0


def _replace_in_object(value: JsonObject) -> tuple[JsonObject, int]:
    r"""Sanitize one JSON object's keys and values without merging entries.

    Keys that needed no replacement are placed first, verbatim. A sanitized
    key that then collides with an existing key (``"a\x00"`` beside a literal
    ``"a\ufffd"``) is disambiguated with a ``~<n>`` suffix instead of
    overwriting the other entry; the disambiguation counts as a replacement so
    the stamp reflects it.
    """
    entries: JsonObject = {}
    total = 0
    renamed: list[tuple[str, JsonValue]] = []
    for key, item in value.items():
        cleaned_item, item_count = _replace_unstorable(item)
        total += item_count
        cleaned_key, key_count = _replace_in_text(key)
        if key_count == 0:
            entries[key] = cleaned_item
        else:
            total += key_count
            renamed.append((cleaned_key, cleaned_item))
    for cleaned_key, cleaned_item in renamed:
        unique_key = cleaned_key
        suffix = 1
        while unique_key in entries:
            suffix += 1
            unique_key = f"{cleaned_key}~{suffix}"
            total += 1
        entries[unique_key] = cleaned_item
    return entries, total


def _replace_unstorable(value: JsonValue) -> tuple[JsonValue, int]:
    """Return ``value`` with every unstorable code point replaced, plus the count.

    Walks the JSON value ``model_dump(mode="json")`` produces: strings (values
    and object keys) are rewritten, arrays and objects recurse, scalars pass
    through untouched.
    """
    match value:
        case dict():
            return _replace_in_object(value)
        case list():
            items: list[JsonValue] = []
            list_total = 0
            for item in value:
                cleaned_item, count = _replace_unstorable(item)
                items.append(cleaned_item)
                list_total += count
            return items, list_total
        case str():
            return _replace_in_text(value)
        case _:
            return value, 0


def normalize_durable_object(message: JsonObject) -> tuple[JsonObject, int]:
    """Make one dumped message storable as ``jsonb``.

    Args:
        message: One ``GatewayMessage.model_dump(mode="json")`` object.

    Returns:
        The same object and 0 when nothing needed replacing; otherwise a copy
        with every unstorable code point replaced by U+FFFD (colliding
        sanitized keys disambiguated, never merged) and the replacement count,
        disambiguations included.
    """
    cleaned, total = _replace_in_object(message)
    return (message, 0) if total == 0 else (cleaned, total)

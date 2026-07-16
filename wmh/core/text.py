"""Text canonicalization for UTF-8 files, transports, and durable JSON stores."""

from __future__ import annotations

_REPLACEMENT_CHARACTER = "\N{REPLACEMENT CHARACTER}"


def normalize_durable_text(value: str) -> str:
    """Replace code points that cannot safely cross every WMH persistence boundary.

    PostgreSQL JSONB rejects embedded NULs and lone UTF-16 surrogates; UTF-8 filesystem and HTTP
    clients reject surrogates as well. A valid surrogate pair is folded into its Unicode scalar.
    """
    normalized: list[str] = []
    index = 0
    while index < len(value):
        code_point = ord(value[index])
        if code_point == 0:
            normalized.append(_REPLACEMENT_CHARACTER)
        elif 0xD800 <= code_point <= 0xDBFF:
            if index + 1 < len(value) and 0xDC00 <= (low := ord(value[index + 1])) <= 0xDFFF:
                scalar = 0x10000 + ((code_point - 0xD800) << 10) + (low - 0xDC00)
                normalized.append(chr(scalar))
                index += 1
            else:
                normalized.append(_REPLACEMENT_CHARACTER)
        elif 0xDC00 <= code_point <= 0xDFFF:
            normalized.append(_REPLACEMENT_CHARACTER)
        else:
            normalized.append(value[index])
        index += 1
    return "".join(normalized)


def validate_durable_text(value: str, *, field: str) -> None:
    """Reject content-addressed text that canonicalization would change."""
    if "\x00" in value:
        raise ValueError(f"{field} contains an embedded NUL character")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError(f"{field} contains an unpaired UTF-16 surrogate")

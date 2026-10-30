"""Lone UTF-16 surrogates in request JSON: replaced in content, refused in names.

JavaScript's well-formed ``JSON.stringify`` writes a string cut inside a
surrogate pair (an emoji sliced by length) as a lone ``\\udXXX`` escape. That
is valid JSON, but ``json.loads`` keeps the lone surrogate in the ``str``, which
has no UTF-8 form: left alone it fails the first ``.encode("utf-8")`` deep
inside admission as an unclassified 500. A raw byte that is not UTF-8 never
reaches here; the data plane refuses it as ``invalid_json``.

Content (message text, parts, tool results, system prompts, tool-call
arguments) is repaired the way a lenient UTF-8 decoder would, one U+FFFD per
lone surrogate, and each repaired field is disclosed as
``<path>->replaced(lone_surrogate)``. A field that names something (the
model, a tool, a call, an opaque provider token) or an object key is refused
with a field-specific 400 instead, because a rewritten name would no longer
match the caller's own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from exp.common.core.artifacts import JsonObject, JsonValue

SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F][0-9a-fA-F]{2}")
"""A JSON surrogate escape; a body without one decodes to no surrogate code point.

It also matches the escape text inside a nested JSON string (``\\\\ud83d``), so
one search of the raw body gates both levels."""
LONE_SURROGATE = re.compile("[\ud800-\udfff]")
"""A surrogate left in decoded text, which ``json.loads`` leaves only when it was unpaired."""
REPLACED_LONE_SURROGATE = "->replaced(lone_surrogate)"
"""Suffix of the disclosure naming a field whose lone surrogates became U+FFFD."""

IDENTIFIER_FIELDS = frozenset(
    {
        "model",
        "name",
        "id",
        "call_id",
        "tool_call_id",
        "tool_use_id",
        "previous_response_id",
        "type",
        "role",
        "signature",
        "thought_signature",
        "encrypted_content",
        "data",
    }
)
"""Fields that name a model, tool, call, protocol member, or opaque provider token."""

NESTED_JSON_FIELDS = frozenset({"arguments"})
"""String fields whose value is itself JSON text the decoders parse again."""

OPEN_OBJECT_FIELDS = frozenset(
    {"parameters", "input_schema", "schema", "metadata", "properties", "input"}
)
"""Fields holding caller-keyed JSON (schemas, metadata, a tool call's ``input``).

Below one, every string is content and every key is the caller's own name, so
paths stop at the field: no caller key is echoed in a disclosure or an error.
The top-level Responses ``input`` is the protocol's item list, not one of these."""

_ESCAPE = re.compile(r"\\(?:u([0-9a-fA-F]{4})|.)", re.DOTALL)
"""One JSON escape; scanning these left to right tokenizes every backslash correctly."""
_REPLACEMENT_ESCAPE = "\\ufffd"


class LoneSurrogateRefusal(ValueError):
    """A lone surrogate where it cannot be replaced; ``message`` is safe to return.

    Messages name the field path only, never the caller's text or keys. This
    module stays free of protocol imports, so each boundary maps the refusal
    to its own error shape.
    """

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param
        self.message = message


def has_lone_surrogate(text: str) -> bool:
    """Whether decoded ``text`` holds a surrogate code point (necessarily unpaired)."""
    return not text.isascii() and LONE_SURROGATE.search(text) is not None


def repaired_text(text: str) -> tuple[str, bool]:
    """Return ``text`` with each lone surrogate replaced by U+FFFD, and whether any was."""
    if not has_lone_surrogate(text):
        return text, False
    return LONE_SURROGATE.sub("\N{REPLACEMENT CHARACTER}", text), True


def repaired_escapes(json_text: str) -> tuple[str, bool]:
    """Replace each lone surrogate ESCAPE inside JSON text with ``\\ufffd``.

    A tool call's ``arguments`` is JSON text the decoders parse again, so a
    lone escape inside it survives the outer decode as six ASCII characters
    and becomes a lone surrogate only at that second parse. The rewrite is
    textual, so every other byte of the caller's arguments is kept exactly.

    Args:
        json_text: JSON document text (valid or not; only escapes are read).

    Returns:
        The text with every unpaired ``\\uD800``-``\\uDFFF`` escape replaced,
        and whether any was.
    """
    if SURROGATE_ESCAPE.search(json_text) is None:
        return json_text, False
    pieces: list[str] = []
    last = 0
    escapes = list(_ESCAPE.finditer(json_text))
    index = 0
    while index < len(escapes):
        escape = escapes[index]
        code = int(escape.group(1), 16) if escape.group(1) is not None else None
        index += 1
        if code is None or not 0xD800 <= code <= 0xDFFF:
            continue
        if code <= 0xDBFF and index < len(escapes):
            low = escapes[index]
            if (
                low.start() == escape.end()
                and low.group(1) is not None
                and 0xDC00 <= int(low.group(1), 16) <= 0xDFFF
            ):
                index += 1
                continue
        pieces.extend((json_text[last : escape.start()], _REPLACEMENT_ESCAPE))
        last = escape.end()
    if not pieces:
        return json_text, False
    pieces.append(json_text[last:])
    return "".join(pieces), True


def refuse_lone_surrogates(payload: JsonObject) -> None:
    """Refuse a body holding any lone surrogate, naming the first field found.

    For surfaces with no disclosure channel (embeddings, images, audio), where
    a silent rewrite would change the input unannounced.

    Raises:
        LoneSurrogateRefusal: A string or object key holds a lone surrogate.
    """
    for member in _members(payload):
        if isinstance(member.key, str) and has_lone_surrogate(member.key):
            raise _key_error(_public(member.path)[0])
        if isinstance(member.value, str) and has_lone_surrogate(member.value):
            param = _param(_public(member.field)[0])
            raise LoneSurrogateRefusal(
                param,
                f"'{param}' contains a lone UTF-16 surrogate (a \\uD800-\\uDFFF escape without "
                "its pair), which has no UTF-8 form. Resend it as valid Unicode.",
            )


def replace_lone_surrogates(payload: JsonObject) -> tuple[str, ...]:
    """Replace lone surrogates in content in place and return the disclosures.

    Args:
        payload: A freshly decoded request object, owned by the caller; its
            strings are rewritten in place.

    Returns:
        One ``<path>->replaced(lone_surrogate)`` per repaired field path
        (list positions omitted), sorted.

    Raises:
        LoneSurrogateRefusal: An identifier field or an object key holds a
            lone surrogate.
    """
    disclosed: set[str] = set()
    for member in _members(payload):
        if isinstance(member.key, str) and has_lone_surrogate(member.key):
            raise _key_error(_public(member.path)[0])
        if not isinstance(member.value, str):
            continue
        field, caller_keyed = _public(member.field)
        named = next((part for part in reversed(member.field) if isinstance(part, str)), "")
        if not caller_keyed and _names_something(member, named):
            if has_lone_surrogate(member.value):
                param = _param(field)
                raise LoneSurrogateRefusal(
                    param,
                    f"'{param}' must not contain a lone UTF-16 surrogate (a \\uD800-\\uDFFF "
                    "escape without its pair). It names a model, tool, call, or provider token, "
                    "so the gateway does not replace it; resend it as valid Unicode.",
                )
            continue
        repaired, changed = repaired_text(member.value)
        if named in NESTED_JSON_FIELDS:
            repaired, nested = repaired_escapes(repaired)
            changed |= nested
        if changed:
            member.assign(repaired)
            disclosed.add(".".join(part for part in field if isinstance(part, str)))
    return tuple(f"{path}{REPLACED_LONE_SURROGATE}" for path in sorted(disclosed))


@dataclass(frozen=True, slots=True)
class _Member:
    """One dict entry or list element, with the container that holds it."""

    container: dict[str, JsonValue] | list[JsonValue]
    path: tuple[str | int, ...]
    key: str | int
    value: JsonValue

    @property
    def field(self) -> tuple[str | int, ...]:
        """The member's own path: its container's path plus its key."""
        return (*self.path, self.key)

    def assign(self, value: str) -> None:
        """Replace this member's value in its container."""
        match self.container, self.key:
            case dict(), str():
                self.container[self.key] = value
            case list(), int():
                self.container[self.key] = value
            case _:  # pragma: no cover - members are built with matching keys.
                raise TypeError("member key does not fit its container")


def _members(payload: JsonObject) -> list[_Member]:
    """List every member of the tree, iteratively.

    A list (not a generator) lets callers rewrite values while reading, and
    iteration rather than recursion keeps a deeply nested body from raising
    ``RecursionError``.
    """
    members: list[_Member] = []
    pending: list[tuple[tuple[str | int, ...], JsonValue]] = [((), payload)]
    while pending:
        path, value = pending.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                members.append(_Member(value, path, key, child))
                pending.append(((*path, key), child))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                members.append(_Member(value, path, index, child))
                pending.append(((*path, index), child))
    return members


def _public(field: tuple[str | int, ...]) -> tuple[tuple[str | int, ...], bool]:
    """Cut ``field`` after its first caller-keyed object, and say whether it was cut.

    Args:
        field: A member's full path.

    Returns:
        The path a refusal or disclosure may name, and whether the member
        lies inside (or is) a caller-keyed object.
    """
    for index, part in enumerate(field):
        if part in OPEN_OBJECT_FIELDS and (part != "input" or index > 0):
            return field[: index + 1], True
    return field, False


def _names_something(member: _Member, named: str) -> bool:
    """Whether ``member``'s string is an identifier rather than content.

    A Chat message's own ``name`` is a participant label (removed before
    dispatch, never matched against a tool), so it is content; on a ``tool``
    or ``function`` message it names the function the result answers.
    """
    if named not in IDENTIFIER_FIELDS:
        return False
    field = member.field
    if len(field) == 3 and field[0] == "messages" and field[2] == "name":
        return isinstance(member.container, dict) and member.container.get("role") in {
            "tool",
            "function",
        }
    return True


def _param(field: tuple[str | int, ...]) -> str:
    """Return the public dotted field path, list positions included."""
    return ".".join(str(part) for part in field) or "body"


def _key_error(path: tuple[str | int, ...]) -> LoneSurrogateRefusal:
    """Refuse an object key holding a lone surrogate without echoing the key."""
    param = _param(path)
    return LoneSurrogateRefusal(
        param,
        f"An object key in '{param}' contains a lone UTF-16 surrogate (a \\uD800-\\uDFFF escape "
        "without its pair). Keys are names, so the gateway does not replace them; resend the "
        "key as valid Unicode.",
    )

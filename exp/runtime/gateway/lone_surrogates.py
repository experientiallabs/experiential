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
model, a tool, a call, an opaque provider token) or an object key (including
a key inside a tool call's JSON ``arguments``) is refused with a
field-specific 400 instead, because a rewritten name would no longer
match the caller's own.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from exp.common.core.artifacts import JsonObject, JsonValue

SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F][0-9a-fA-F]{2}")
"""A JSON surrogate escape; a body without one decodes to no surrogate code point.

It also matches the escape text inside a nested JSON string (``\\\\ud83d``), so
one search of the raw body gates both levels."""
_ESCAPED_ASCII = re.compile(r"\\u00[2-7][0-9a-fA-F]")
"""A printable ASCII character written as an escape, which can spell a nested escape."""
LONE_SURROGATE = re.compile("[\ud800-\udfff]")
"""A surrogate left in decoded text, which ``json.loads`` leaves only when it was unpaired."""
REPLACED_LONE_SURROGATE = "->replaced(lone_surrogate)"
"""Suffix of the disclosure naming a field whose lone surrogates became U+FFFD."""
MAX_DISCLOSED_PATHS = 64
"""Full field paths disclosed; past this, a repair is disclosed by its top-level field,
and past twice this, as ``body``.

A body can repeat one long path prefix in many distinct fields, so the bound
keeps the disclosures no larger than the body itself."""
MAX_DISCLOSED_PATH_CHARS = 256
"""Longest field name a disclosure carries; a longer one is disclosed as a shorter form."""
_PATH_BUDGET = 256
"""Full paths built per body for disclosure; past it a repair names its top-level field.

Building a path costs its depth, so the budget keeps a body of many deep
repairs linear in its size."""

IDENTIFIER_FIELDS = frozenset(
    {
        "model",
        "name",
        "namespace",
        "id",
        "call_id",
        "tool_call_id",
        "tool_use_id",
        "previous_response_id",
        "approval_request_id",
        "container_id",
        "vector_store_ids",
        "file_ids",
        "tunnel_id",
        "connector_id",
        "working_directory",
        "agent_name",
        "effort",
        "speed",
        "keys",
        "skill_id",
        "container",
        "fingerprint",
        "authorization",
        "allowed_tools",
        "tool_names",
        "stop",
        "stop_sequences",
        "file_id",
        "user",
        "safety_identifier",
        "prompt_cache_key",
        "promptCacheKey",
        "server_label",
        "tool_name",
        "allowed_callers",
        "url",
        "image_url",
        "file_url",
        "server_url",
        "inference_geo",
        "include_domains",
        "exclude_domains",
        "allowed_domains",
        "blocked_domains",
        "type",
        "role",
        "signature",
        "thought_signature",
        "encrypted_content",
        "data",
    }
)
"""Fields that name a model, tool, call, protocol member, caller, cache key, or provider token."""

IDENTIFIER_PATHS = frozenset({("metadata", "user_id"), ("thinking", "display")})
"""Identifiers one level below a top-level object: the Messages end-user id in
``metadata`` and the Messages reasoning-display selector in ``thinking``."""

OPAQUE_FIELDS = frozenset(
    {
        "caller",
        "context_management",
        "diagnostics",
        "safeguards",
        "output_config",
        "client_metadata",
        "citations",
        "user_location",
        "filters",
        "headers",
        "domain_secrets",
        "env",
    }
)
"""Provider objects that round-trip verbatim; any string below one is an identifier."""

TOP_LEVEL_OPAQUE_FIELDS = frozenset({"provider"})
"""Top-level objects whose strings all name something (provider routing preferences)."""

MAX_ARGUMENTS_CHARS = 4_000_000
"""The decoders' own bound on tool-call ``arguments``; longer text is refused there."""

NESTED_JSON_FIELDS = frozenset({"arguments"})
"""String fields whose value is itself JSON text the decoders parse again."""

ITEM_JSON_FIELDS = frozenset({("local_shell_call_output", "output")})
"""(item type, field) pairs whose string is JSON text, repaired like ``arguments``."""

ITEM_IDENTIFIER_FIELDS = frozenset(
    {
        ("image_generation_call", "result"),
        ("agent_message", "author"),
        ("agent_message", "recipient"),
    }
)
"""(item type, field) pairs that name something or carry an opaque payload in that item."""

OPEN_OBJECT_FIELDS = frozenset(
    {
        "parameters",
        "input_schema",
        "output_schema",
        "schema",
        "metadata",
        "properties",
        "input",
        "input_examples",
        "provider_specific_fields",
    }
)
"""Fields holding caller-keyed JSON (schemas, metadata, a tool call's ``input``).

Below one, every string is content and every key is the caller's own name, so
paths stop at the field: no caller key is echoed in a disclosure or an error.
The top-level Responses ``input`` is the protocol's item list, not one of these."""

_ESCAPE = re.compile(r"\\(?:u([0-9a-fA-F]{4})|.)", re.DOTALL)
"""One JSON escape; scanning these left to right tokenizes every backslash correctly."""
_REPLACEMENT_ESCAPE = "\\ufffd"
_LAYERED_REPLACEMENT = "\\\\ufffd"
_INNER_ESCAPE = re.compile(r"(?:u|\\u0075)((?:[0-9a-fA-F]|\\u00(?:3[0-9]|4[1-6]|6[1-6])){4})")
"""The ``uXXXX`` of an inner escape, any character possibly spelled as an outer escape."""
_SPELLED_ASCII = re.compile(r"\\u00([0-9a-fA-F]{2})")
_STRING_TOKEN = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"(\s*:)?', re.DOTALL)
"""One JSON string token, with the colon that makes it an object key when present."""
_TRAILING_SPACE = re.compile(r"[ \t\r\n]*\Z")
"""Only JSON whitespace left: a key whose value never started."""
_LAYERED_ESCAPE = re.compile(r"\\\\u[dD][89a-fA-F][0-9a-fA-F]{2}")
"""A surrogate escape behind an escaped backslash: an escape one JSON layer deeper."""


class LoneSurrogateRefusal(ValueError):
    """A lone surrogate where it cannot be replaced; ``message`` is safe to return.

    Messages name the field path only, never the caller's text or keys. This
    module stays free of protocol imports, so each boundary maps the refusal
    to its own error shape.
    """

    def __init__(self, param: str, message: str) -> None:
        """Record the refused field path and the caller-facing message."""
        super().__init__(message)
        self.param = param
        self.message = message


def may_hold_lone_surrogate(body: str) -> bool:
    """Whether raw JSON ``body`` can decode to a lone surrogate at any nesting level.

    True when the body holds a surrogate escape, or an escaped printable ASCII
    character (``\\u005c`` for a backslash, ``\\u0075`` for ``u``) that can
    spell one inside nested JSON text such as tool-call ``arguments``. Bodies
    with neither skip the walk.
    """
    return SURROGATE_ESCAPE.search(body) is not None or _ESCAPED_ASCII.search(body) is not None


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
    escapes = _ESCAPE.finditer(json_text)
    escape = next(escapes, None)
    while escape is not None:
        following = next(escapes, None)
        code = int(escape.group(1), 16) if escape.group(1) is not None else None
        if code is None or not 0xD800 <= code <= 0xDFFF:
            escape = following
            continue
        if (
            code <= 0xDBFF
            and following is not None
            and following.start() == escape.end()
            and following.group(1) is not None
            and 0xDC00 <= int(following.group(1), 16) <= 0xDFFF
        ):
            escape = next(escapes, None)
            continue
        pieces.extend((json_text[last : escape.start()], _REPLACEMENT_ESCAPE))
        last = escape.end()
        escape = following
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
            path = member.path
            raise _key_error(_identifier_path(path, _public(path)[0]))
        if isinstance(member.value, str) and has_lone_surrogate(member.value):
            param = _param(_public(member.field)[0])
            raise LoneSurrogateRefusal(
                param,
                f"'{param}' contains a lone UTF-16 surrogate (a \\uD800-\\uDFFF escape without "
                "its pair), which has no UTF-8 form. Resend it as valid Unicode.",
            )


def replace_lone_surrogates(
    payload: JsonObject, *, object_prefix: Callable[[str], bool]
) -> tuple[str, ...]:
    """Replace lone surrogates in content in place and return the disclosures.

    Args:
        payload: A freshly decoded request object, owned by the caller; its
            strings are rewritten in place.
        object_prefix: Whether tool-call ``arguments`` text that does not
            parse is a cut-off object prefix the decoder closes (the
            decoder's own predicate, passed in because this module stays
            free of protocol imports).

    Returns:
        One ``<path>->replaced(lone_surrogate)`` per repaired field path
        (list positions omitted), sorted.

    Raises:
        LoneSurrogateRefusal: An identifier field, an object key, or a key
            inside a tool call's JSON ``arguments`` holds a lone surrogate.
    """
    disclosed: set[str] = set()
    budget = _PATH_BUDGET
    for member in _members(payload):
        if isinstance(member.key, str) and has_lone_surrogate(member.key):
            path = member.path
            raise _key_error(_identifier_path(path, _public(path)[0]))
        if not isinstance(member.value, str):
            continue
        lone = has_lone_surrogate(member.value)
        argument_text = member.key in NESTED_JSON_FIELDS or member.item_field in ITEM_JSON_FIELDS
        escaped_text = argument_text and may_hold_lone_surrogate(member.value)
        if not lone and not escaped_text:
            continue
        caller_keyed = member.caller_keyed
        if _names_something(member, caller_keyed=caller_keyed):
            if lone:
                full_field = member.field
                param = _param(_identifier_path(full_field, _public(full_field)[0]))
                raise LoneSurrogateRefusal(
                    param,
                    f"'{param}' must not contain a lone UTF-16 surrogate (a \\uD800-\\uDFFF "
                    "escape without its pair). It names a model, tool, call, caller, or provider "
                    "token, so the gateway does not replace it; resend it as valid Unicode.",
                )
            continue
        if argument_text and not caller_keyed:
            repaired, changed = _repaired_arguments(
                member.value, member, layers=2, object_prefix=object_prefix
            )
        else:
            repaired, changed = repaired_text(member.value)
        if not changed:
            continue
        member.assign(repaired)
        disclosed.add(
            _disclosure(
                member,
                full=len(disclosed) < MAX_DISCLOSED_PATHS and budget > 0,
                wide=len(disclosed) < 2 * MAX_DISCLOSED_PATHS,
            )
        )
        budget -= 1
    return tuple(f"{path}{REPLACED_LONE_SURROGATE}" for path in sorted(disclosed))


@dataclass(frozen=True, slots=True)
class _Path:
    """One container's position: its parent's position plus its own key.

    Positions are linked rather than copied, so a deep body costs one node
    per open container instead of one full path tuple per member.

    Each node also carries the facts a member's classification needs, so a
    deep body is classified in constant time per member rather than by
    rebuilding its path.

    Attributes:
        parent: The enclosing container's position, or ``None`` at the body.
        key: This container's key or list index in its parent.
        named: The nearest string key on the path (``""`` when none).
        depth: The number of keys on this container's path.
        top: The body's top-level key on this path (``""`` when none).
        caller_keyed: Whether the path enters a caller-keyed object.
        opaque: Whether the path enters a verbatim provider object outside any
            caller-keyed object; below one, nothing is caller-keyed.
    """

    parent: _Path | None
    key: str | int
    named: str
    depth: int
    top: str
    caller_keyed: bool
    opaque: bool

    @staticmethod
    def child(parent: _Path | None, key: str | int) -> _Path:
        """The position of ``key``'s container below ``parent``."""
        if parent is None:
            return _Path(
                parent=None,
                key=key,
                named=key if isinstance(key, str) else "",
                depth=1,
                top=key if isinstance(key, str) else "",
                caller_keyed=_opens(key, 0),
                opaque=key in OPAQUE_FIELDS or key in TOP_LEVEL_OPAQUE_FIELDS,
            )
        return _Path(
            parent=parent,
            key=key,
            named=key if isinstance(key, str) else parent.named,
            depth=parent.depth + 1,
            top=parent.top,
            caller_keyed=not parent.opaque
            and (parent.caller_keyed or _opens(key, parent.depth) or key in NESTED_JSON_FIELDS),
            opaque=parent.opaque or (not parent.caller_keyed and key in OPAQUE_FIELDS),
        )


@dataclass(frozen=True, slots=True)
class _Member:
    """One dict entry or list element, with the container that holds it.

    Attributes:
        container: The dict or list holding this member.
        position: The container's position in the body, ``None`` for the body.
        key: This member's key or list index in ``container``.
        value: This member's value.
    """

    container: dict[str, JsonValue] | list[JsonValue]
    position: _Path | None
    key: str | int
    value: JsonValue

    @property
    def path(self) -> tuple[str | int, ...]:
        """The container's path from the body."""
        parts: list[str | int] = []
        node = self.position
        while node is not None:
            parts.append(node.key)
            node = node.parent
        parts.reverse()
        return tuple(parts)

    @property
    def field(self) -> tuple[str | int, ...]:
        """The member's own path: its container's path plus its key."""
        return (*self.path, self.key)

    @property
    def named(self) -> str:
        """The nearest string key on the member's own path (``""`` when none)."""
        if isinstance(self.key, str):
            return self.key
        return "" if self.position is None else self.position.named

    @property
    def item_field(self) -> tuple[str, str] | None:
        """``(type, key)`` when the member sits directly in an object with a string ``type``."""
        if not isinstance(self.container, dict) or not isinstance(self.key, str):
            return None
        kind = self.container.get("type")
        return (kind, self.key) if isinstance(kind, str) else None

    @property
    def depth(self) -> int:
        """The number of keys on the container's path."""
        return 0 if self.position is None else self.position.depth

    @property
    def top(self) -> str:
        """The body's top-level key on the member's own path (``""`` when none)."""
        if self.position is None:
            return self.key if isinstance(self.key, str) else ""
        return self.position.top

    @property
    def caller_keyed(self) -> bool:
        """Whether the member lies inside (or is) a caller-keyed object."""
        if self.opaque:
            return False
        inside = self.position is not None and self.position.caller_keyed
        return inside or _opens(self.key, self.depth)

    @property
    def opaque(self) -> bool:
        """Whether the member lies inside a verbatim provider object."""
        return self.position is not None and self.position.opaque

    @property
    def identifier_path(self) -> bool:
        """Whether the member's own path is one of ``IDENTIFIER_PATHS``."""
        position = self.position
        return (
            position is not None
            and position.parent is None
            and (position.key, self.key) in IDENTIFIER_PATHS
        )

    def assign(self, value: str) -> None:
        """Replace this member's value in its container."""
        match self.container, self.key:
            case dict(), str():
                self.container[self.key] = value
            case list(), int():
                self.container[self.key] = value
            case _:  # pragma: no cover - members are built with matching keys.
                raise TypeError("member key does not fit its container")


def _entries(
    container: dict[str, JsonValue] | list[JsonValue],
) -> Iterator[tuple[str | int, JsonValue]]:
    """Iterate a container's keys (or indexes) and values."""
    if isinstance(container, dict):
        return iter(container.items())
    return enumerate(container)


def _members(payload: JsonValue) -> Iterator[_Member]:
    """Yield every member of the tree, depth first, one at a time.

    Only the open containers are held (one frame per nesting level), so a
    wide body is walked in constant extra memory, and iteration rather than
    recursion keeps a deeply nested body from raising ``RecursionError``.
    A caller may replace a yielded string member's value before the next
    member is read; containers are entered from the value as yielded.
    """
    if not isinstance(payload, (dict, list)):
        return
    stack: list[
        tuple[
            _Path | None,
            Iterator[tuple[str | int, JsonValue]],
            dict[str, JsonValue] | list[JsonValue],
        ]
    ] = [(None, _entries(payload), payload)]
    while stack:
        position, entries, container = stack[-1]
        entry = next(entries, None)
        if entry is None:
            stack.pop()
            continue
        key, child = entry
        yield _Member(container, position, key, child)
        if isinstance(child, (dict, list)):
            stack.append((_Path.child(position, key), _entries(child), child))


def _repaired_arguments(
    json_text: str, member: _Member, *, layers: int, object_prefix: Callable[[str], bool]
) -> tuple[str, bool]:
    """Repair a tool call's JSON ``arguments`` text, refusing a lone surrogate in a key.

    Lone surrogates in values become U+FFFD textually, so every other byte of
    the caller's arguments is kept. A key names a parameter, so one holding a
    lone surrogate (raw or escaped) is refused rather than renamed, in a
    complete object and in a cut-off object prefix the decoder later closes.
    Double-encoded
    arguments (a JSON string whose value encodes the object, which the
    Responses decoder unwraps) are repaired at the inner layer and
    re-encoded; only that string layer is ever parsed.

    Args:
        json_text: The caller's ``arguments`` text.
        member: The ``arguments`` member, for a refusal's field path.
        layers: How many JSON string layers may still be unwrapped.
        object_prefix: The decoder's cut-off object predicate.

    Returns:
        The repaired text and whether anything was repaired.

    Raises:
        LoneSurrogateRefusal: An object key in the arguments holds a lone
            surrogate.
    """
    repaired, changed = repaired_text(json_text)
    repaired, escaped = repaired_escapes(repaired)
    changed |= escaped
    if changed and _argument_key_has_lone_surrogate(json_text, object_prefix):
        raise _key_error(_public(member.field)[0])
    if layers <= 1 or (
        _LAYERED_ESCAPE.search(json_text) is None and _ESCAPED_ASCII.search(json_text) is None
    ):
        return repaired, changed
    if not json_text.lstrip().startswith('"'):
        return repaired, changed
    try:
        parsed: JsonValue = json.loads(json_text)
    except ValueError:
        return repaired, changed
    if not isinstance(parsed, str):
        return repaired, changed
    inner, inner_changed = _repaired_arguments(
        parsed, member, layers=layers - 1, object_prefix=object_prefix
    )
    if not inner_changed:
        return repaired, changed
    # Prefer a textual repair of the outer string, which keeps every other
    # caller byte; re-encode only for spellings the textual pass cannot see.
    textual = _repaired_layered_escapes(repaired)
    try:
        layered: JsonValue = json.loads(textual)
    except ValueError:
        layered = None
    if layered == inner:
        return textual, True
    return json.dumps(inner, ensure_ascii=False), True


def _repaired_layered_escapes(json_text: str) -> str:
    """Replace each lone surrogate escape one JSON string layer down, textually.

    In a JSON string, each inner backslash is an escaped backslash (or its
    ``\\\\u005c`` spelling). Within a run of them, inner backslashes pair up,
    so an inner escape starts only where a run of odd length is followed by
    ``uXXXX``. Unpaired surrogate escapes become ``\\\\ufffd`` in place, so
    every other byte of the outer text is kept.
    """
    pieces: list[str] = []
    last = 0
    run = 0
    run_end = -1
    pending: tuple[int, int, int] | None = None
    consumed = -1
    for escape in _ESCAPE.finditer(json_text):
        if escape.start() < consumed:
            # Part of an inner escape already read (a spelled ``\\u0075``).
            continue
        backslash = escape.group(0) == "\\\\" or (escape.group(1) or "").lower() == "005c"
        run = run + 1 if backslash and escape.start() == run_end else int(backslash)
        run_end = escape.end() if backslash else -1
        inner = None
        inner_end = escape.end()
        if backslash and run % 2 == 1:
            spelled = _INNER_ESCAPE.match(json_text, escape.end())
            if spelled is not None:
                digits = _SPELLED_ASCII.sub(
                    lambda hex_digit: chr(int(hex_digit[1], 16)), spelled[1]
                )
                inner, inner_end = int(digits, 16), spelled.end()
        if pending is not None:
            start, end, code = pending
            pending = None
            if (
                code <= 0xDBFF
                and inner is not None
                and escape.start() == end
                and 0xDC00 <= inner <= 0xDFFF
            ):
                run, run_end = 0, -1
                continue
            pieces.extend((json_text[last:start], _LAYERED_REPLACEMENT))
            last = end
        if inner is not None:
            # The inner escape consumes this backslash and its ``uXXXX``.
            run, run_end = 0, -1
            consumed = inner_end
            if 0xD800 <= inner <= 0xDFFF:
                pending = (escape.start(), inner_end, inner)
    if pending is not None:
        start, end, _code = pending
        pieces.extend((json_text[last:start], _LAYERED_REPLACEMENT))
        last = end
    if not pieces:
        return json_text
    pieces.append(json_text[last:])
    return "".join(pieces)


def _argument_key_has_lone_surrogate(json_text: str, object_prefix: Callable[[str], bool]) -> bool:
    """Whether arguments the decoder serves as an object hold a lone surrogate in a key.

    Only text the decoder turns into an object has keys: a complete JSON
    object (judged on its parsed keys) or a cut-off object prefix (judged on
    its string tokens followed by ``:``, minus a final key whose value never
    started, which the decoder drops). Any other text is served verbatim as
    an unparsed stand-in, so nothing in it is a key. Text past the decoders'
    own length bound is refused there and never parsed here.
    """
    if len(json_text) > MAX_ARGUMENTS_CHARS:
        return False
    try:
        parsed: JsonValue = json.loads(json_text)
    except ValueError:
        return object_prefix(json_text) and _prefix_key_has_lone_surrogate(json_text)
    except RecursionError:
        # Valid JSON too deep to materialize: its string tokens are its keys.
        return _prefix_key_has_lone_surrogate(json_text)
    if not isinstance(parsed, dict):
        return False
    return any(
        isinstance(member.key, str) and has_lone_surrogate(member.key)
        for member in _members(parsed)
    )


def _prefix_key_has_lone_surrogate(json_text: str) -> bool:
    """Whether a key in a valid cut-off object prefix holds a lone surrogate.

    Scanning string tokens left to right tokenizes a valid prefix correctly,
    since text between tokens holds no quote.
    """
    for token in _STRING_TOKEN.finditer(json_text):
        if token.group(2) is None or _TRAILING_SPACE.match(json_text, token.end()):
            continue
        key = token.group(1)
        if has_lone_surrogate(key) or repaired_escapes(key)[1]:
            return True
    return False


def _public(field: tuple[str | int, ...]) -> tuple[tuple[str | int, ...], bool]:
    """Cut ``field`` after its first caller-keyed object, and say whether it was cut.

    Args:
        field: A member's full path.

    Returns:
        The path a refusal or disclosure may name, and whether the member
        lies inside (or is) a caller-keyed object.
    """
    for index, part in enumerate(field):
        if _opens(part, index) or (part in NESTED_JSON_FIELDS and index < len(field) - 1):
            return field[: index + 1], True
    return field, False


def _disclosure(member: _Member, *, full: bool, wide: bool) -> str:
    """Name a repaired member's field for its disclosure, never longer than the bound.

    Args:
        member: The repaired member.
        full: Whether the full public path may still be built.
        wide: Whether a top-level field may still be named.

    Returns:
        The dotted public path, else the top-level field, else ``body``; any
        name over ``MAX_DISCLOSED_PATH_CHARS`` falls back to the next form.
    """
    if full:
        parts = [part for part in _public(member.field)[0] if isinstance(part, str)]
        if sum(len(part) + 1 for part in parts) <= MAX_DISCLOSED_PATH_CHARS:
            return ".".join(parts)
    top = member.top
    if wide and top and len(top) <= MAX_DISCLOSED_PATH_CHARS:
        return top
    return "body"


def _opens(part: str | int, index: int) -> bool:
    """Whether path part ``part`` at position ``index`` enters a caller-keyed object.

    The top-level Responses ``input`` is the protocol's item list, not one.
    """
    return part in OPEN_OBJECT_FIELDS and (part != "input" or index > 0)


def _names_something(member: _Member, *, caller_keyed: bool) -> bool:
    """Whether ``member``'s string is an identifier rather than content.

    A Chat message's own ``name`` is a participant label (removed before
    dispatch, never matched against a tool), so it is content; on a ``tool``
    or ``function`` message it names the function the result answers. Inside
    a caller-keyed object only the listed ``IDENTIFIER_PATHS`` name anything.
    """
    if member.identifier_path:
        return True
    if caller_keyed:
        return False
    if member.opaque:
        return True
    position = member.position
    if member.key == "path" and position is not None and position.key == "operation":
        # An apply-patch operation names the file it changes.
        return True
    if member.item_field in ITEM_IDENTIFIER_FIELDS:
        return True
    if member.named not in IDENTIFIER_FIELDS:
        return False
    if member.depth == 2 and member.top == "messages" and member.key == "name":
        return isinstance(member.container, dict) and member.container.get("role") in {
            "tool",
            "function",
        }
    return True


def _identifier_path(
    field: tuple[str | int, ...], public: tuple[str | int, ...]
) -> tuple[str | int, ...]:
    """The path a refusal names: the full path for a listed identifier, else ``public``."""
    if field in IDENTIFIER_PATHS:
        return field
    # A verbatim object counts only outside caller-keyed content (``public``
    # already stops there), so a caller key is never echoed.
    for index, part in enumerate(field[: len(public)]):
        if part in OPAQUE_FIELDS or (index == 0 and part in TOP_LEVEL_OPAQUE_FIELDS):
            return field[: index + 1]
    return public


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

"""Render pydantic validation failures as field-specific public protocol errors.

Split from ``requests`` for the module line budget: the decoders there own
manifest gating and canonical translation; this module owns the one
translation from a pydantic ``ValidationError`` (raised by the official SDK
schema or by this gateway's strict wire models) to a stable OpenAI-shaped 400
whose ``param`` names the caller's field and whose message states what shape
was expected against what arrived.
"""

from __future__ import annotations

import re

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from exp.runtime.openai_protocol.errors import OpenAIProtocolError, invalid_field
from exp.runtime.openai_protocol.responses_input_items import EXTRA_INPUT_ITEM_TYPES
from exp.runtime.openai_protocol.wire_models import (
    HOSTED_TOOL_ITEM_TYPES_ASSISTANT,
    HOSTED_TOOL_ITEM_TYPES_TOOL,
)

_LOCATION_NOISE = {"body", "non-streaming", "streaming"}
_UNION_BRANCH_TYPES = {"str", "int", "float", "bool", "list", "tuple", "dict", "NoneType"}
_OUTPUT_ITEM_VARIANTS = {
    "message",
    "function_call",
    "function_call_output",
    "reasoning",
    "additional_tools",
    "custom_tool_call",
    "custom_tool_call_output",
    *EXTRA_INPUT_ITEM_TYPES,
    # Hosted-tool echo variants share the same union-branch label shape.
    *HOSTED_TOOL_ITEM_TYPES_TOOL,
    *HOSTED_TOOL_ITEM_TYPES_ASSISTANT,
}

_CHAT_ONLY_PART_TAGS = frozenset({"text", "image_url", "video_url", "file"})
"""Content-part discriminators the Responses surface never accepts.

The two surfaces share one content-part union, so a Responses rejection would
otherwise advertise the Chat spellings the provider itself refuses (probed
live 2026-09-15, api.openai.com: ``type: "text"`` in a Responses message is
"Invalid value: 'text'").
"""

_TAG_TOKEN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
"""A discriminator tag short and identifier-like enough to echo.

Messages never echo caller VALUES (prompts, keys, URLs); the ``type`` tag of
a content part is a vocabulary token pydantic itself reports in the error
context, and naming the one that arrived is what lets the caller see which
spelling it used (the provider's own error does the same).
"""


def cleaned_location(
    location: tuple[str | int, ...], *, keep_last: bool = False
) -> tuple[str, ...]:
    """Drop pydantic union-branch labels so the path names request fields.

    A discriminated-union branch label is never the LAST segment of a location
    (an error inside a branch names the field after it), so a final segment
    that merely spells like one (a message key named ``reasoning``) is the
    caller's own field and is kept.
    """
    cleaned: list[str] = []
    last_index = len(location) - 1
    # The cleaned depth at which an item's variant label was dropped: the
    # segment after it is that item's own field even when it spells like a
    # variant (a ``configuration_update`` item's ``reasoning`` object).
    variant_dropped_at: int | None = None
    for index, part in enumerate(location):
        text = str(part)
        if keep_last and index == last_index:
            # An ``extra_forbidden`` location ends in the caller's own key,
            # whatever it spells (``_meta``, ``Foo``, a parent's name).
            cleaned.append(text)
            continue
        if text in _LOCATION_NOISE:
            continue
        if (
            tuple(cleaned) == ("gateway", "retry", "backoff")
            and text in {"none", "exponential"}
            and index != last_index
        ):
            continue
        # Typed-dict union branches are labeled with their class name, which
        # no request field ever shares: every public field is lower case.
        if isinstance(part, str) and (
            part.startswith("_") or "[" in text or text in _UNION_BRANCH_TYPES or text[:1].isupper()
        ):
            continue
        if (
            text in _OUTPUT_ITEM_VARIANTS
            and cleaned
            and cleaned[-1].isdigit()
            and index != last_index
            and variant_dropped_at != len(cleaned)
        ):
            variant_dropped_at = len(cleaned)
            continue
        # A discriminated part whose tag is also its payload field name
        # (``file.file``, ``image_url.image_url``) reports the tag once.
        if cleaned and cleaned[-1] == text and not text.isdigit():
            continue
        cleaned.append(text)
    return tuple(cleaned)


_WIRE_TYPE_NAMES = {
    "str": "a string",
    "int": "an integer",
    "float": "a number",
    "bool": "a boolean",
    "list": "an array",
    "tuple": "an array",
    "dict": "an object",
    "NoneType": "null",
}
"""JSON-shape names for python input types, used in expected/got messages."""

_EXPECTED_BY_ERROR_TYPE = {
    "string_type": "a string",
    "string_too_short": "a non-empty string",
    "int_type": "an integer",
    "int_parsing": "an integer",
    "float_type": "a number",
    "float_parsing": "a number",
    "bool_type": "a boolean",
    "list_type": "an array",
    "tuple_type": "an array",
    "dict_type": "an object",
    "model_type": "an object",
    "model_attributes_type": "an object",
    "missing": "a value",
    "none_required": "null",
}
"""Shape-level expectations for the pydantic error types worth naming."""

_QUOTED = re.compile(r"'([^']*)'")


def _quoted_values(rendered: object) -> list[str]:
    """Return the quoted vocabulary values pydantic renders in an ``expected`` string."""
    return _QUOTED.findall(rendered) if isinstance(rendered, str) else []


def _join_quoted(values: list[str]) -> str:
    """Render ``'a', 'b' or 'c'`` the way pydantic renders one literal's choices."""
    quoted = [f"'{value}'" for value in values]
    if len(quoted) <= 1:
        return "".join(quoted)
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]


def _discriminator_message(param: str, detail: ErrorDetails) -> tuple[str, str] | None:
    """Name the accepted ``type`` tags when a discriminated part carries a wrong or missing one.

    Returns the (param, message) pair naming the tag field itself, or ``None``
    when the detail is not a discriminator failure.
    """
    if detail["type"] not in {"union_tag_invalid", "union_tag_not_found"}:
        return None
    context = detail.get("ctx") or {}
    discriminator = _quoted_values(context.get("discriminator"))
    field = discriminator[0] if discriminator else "type"
    tag_param = f"{param}.{field}"
    expected = _quoted_values(context.get("expected_tags"))
    if param.startswith("input."):
        expected = [tag for tag in expected if tag not in _CHAT_ONLY_PART_TAGS]
    if detail["type"] == "union_tag_not_found":
        message = f"Invalid value for '{tag_param}': the field is required"
        if expected:
            message += f"; expected one of {_join_quoted(expected)}"
        return tag_param, message + "."
    tag = context.get("tag")
    got = f"'{tag}'" if isinstance(tag, str) and _TAG_TOKEN.match(tag) else "an unsupported value"
    if expected:
        return tag_param, (
            f"Invalid value for '{tag_param}': expected one of {_join_quoted(expected)}, "
            f"but got {got} instead."
        )
    return tag_param, f"Invalid value for '{tag_param}': got {got}."


def _declared_unknown_tag(details: list[ErrorDetails]) -> tuple[str, str] | None:
    """Name the ``type`` of an item that declared a tag no union branch accepts.

    An item union pairs a model with a defaulted tag (the Responses message,
    whose ``type`` defaults to ``message``) with a discriminated union of
    every other item. An item that states a tag neither side knows makes the
    defaulted branch report each field it lacks, and its "role: expected a
    value" would otherwise win as the most specific complaint, naming a field
    the caller never meant to send. The defaulted branch's literal fault on
    the discriminator itself proves the item declared a tag, so the tag is the
    field to name, against the whole accepted vocabulary of both sides.

    Sibling literal faults are indexed once by their item location, so the
    lookup stays linear in the number of details (a body of thousands of
    malformed items must not cost quadratic time on the error path).

    Args:
        details: Every pydantic error detail of one failed validation.

    Returns:
        The ``(param, message)`` pair naming the declared tag, or ``None``
        when no item declared an unknown tag.
    """
    literal_tags: dict[tuple[tuple[str | int, ...], str | int], list[str]] = {}
    for detail in details:
        location = tuple(detail["loc"])
        if detail["type"] == "literal_error" and len(location) >= 2:
            literal_tags.setdefault((location[:-2], location[-1]), []).extend(
                _quoted_values((detail.get("ctx") or {}).get("expected"))
            )
    if not literal_tags:
        return None
    for detail in details:
        if detail["type"] != "union_tag_invalid":
            continue
        context = detail.get("ctx") or {}
        discriminator = _quoted_values(context.get("discriminator"))
        field = discriminator[0] if discriminator else "type"
        location = tuple(detail["loc"])
        sibling_tags = literal_tags.get((location[:-1], field))
        if not sibling_tags:
            continue
        tags = [*sibling_tags, *_quoted_values(context.get("expected_tags"))]
        merged: ErrorDetails = {
            **detail,
            "ctx": {
                **context,
                "expected_tags": ", ".join(f"'{tag}'" for tag in dict.fromkeys(tags)),
            },
        }
        return _discriminator_message(".".join(cleaned_location(location)) or "body", merged)
    return None


def _without_rejected_branches(details: list[ErrorDetails]) -> list[ErrorDetails]:
    """Drop the complaints of union branches whose own ``type`` refused the value.

    The official schema's unions are not discriminated, so every branch
    reports. A branch whose ``type`` literal failed is not the one the caller
    meant, yet its deeper complaints can outrank the meant branch's: a replayed
    ``web_search_call`` lacking one field would be reported at ``action.type``
    against the computer-use and local-shell actions ("expected one of
    'click', ... or 'exec'"). A refused branch is dropped only while a sibling
    branch of the same union still reports, so a value that no branch accepts
    keeps every complaint and the pooled vocabulary.

    Branch prefixes are indexed once, so the pass stays linear in the number
    of details times the location depth, never quadratic in the item count.

    Args:
        details: Every pydantic error detail of one failed validation.

    Returns:
        The details outside every dropped branch, in their original order.
    """
    rejected = {
        tuple(detail["loc"][:-1])
        for detail in details
        if detail["type"] == "literal_error" and detail["loc"] and detail["loc"][-1] == "type"
    }
    if not rejected:
        return details
    # Each union location mapped to the branches below it that report a
    # complaint deeper than the branch label itself.
    reporting: dict[tuple[str | int, ...], set[tuple[str | int, ...]]] = {}
    for detail in details:
        location = tuple(detail["loc"])
        for depth in range(len(location) - 1):
            reporting.setdefault(location[:depth], set()).add(location[: depth + 1])
    dropped = {
        branch
        for branch in rejected
        if any(sibling not in rejected for sibling in reporting.get(branch[:-1], ()))
    }
    if not dropped:
        return details
    return [
        detail
        for detail in details
        if not any(
            tuple(detail["loc"][:depth]) in dropped for depth in range(len(detail["loc"]) + 1)
        )
    ]


def shape_message(param: str, details: list[ErrorDetails]) -> str | None:
    """Describe what shape a field expected versus what arrived.

    Only structural facts appear: expectations come from this gateway's own
    wire models (or the official schema), and the got side is the JSON type of
    the caller's value, never the value itself and never provider prose. A
    literal/enum member fault names the members and no arriving type: the
    value arrived as the right JSON type, so "got a string instead" would
    misdescribe a value fault as a type fault (#951).
    """
    expected: list[str] = []
    allowed: list[str] = []
    got: str | None = None
    member_error = False
    open_string = False
    for detail in details:
        if detail["type"] == "string_too_long":
            # The bound and the arriving LENGTH are both display-safe facts
            # (the value itself is never echoed); stating them saves the
            # caller from bisecting the ceiling out of a bare rejection.
            context = detail.get("ctx") or {}
            maximum = context.get("max_length")
            value = detail.get("input")
            if isinstance(maximum, int) and isinstance(value, str):
                return (
                    f"Invalid value for '{param}': expected at most "
                    f"{maximum:,} characters, but got {len(value):,}."
                )
        phrase = _EXPECTED_BY_ERROR_TYPE.get(detail["type"])
        open_string = open_string or detail["type"] == "string_type"
        if detail["type"] in {"literal_error", "enum"}:
            member_error = True
            context = detail.get("ctx") or {}
            for value in _quoted_values(context.get("expected")):
                if value not in allowed:
                    allowed.append(value)
        if phrase is not None and phrase not in expected:
            expected.append(phrase)
        # A missing-field complaint carries the parent object as its input,
        # so it contributes no honest "got" type.
        if detail["type"] != "missing" and "input" in detail:
            got = _WIRE_TYPE_NAMES.get(type(detail["input"]).__name__, got)
    if open_string:
        # An open vocabulary (``str | Literal[...]``, the SDK's ``model``)
        # admits any string, so its member list names no constraint; listing
        # it would bury "expected a string" behind every known model id.
        allowed = []
        member_error = False
    if allowed:
        expected.insert(0, f"one of {_join_quoted(allowed)}")
    if not expected:
        return None
    description = " or ".join(expected)
    if got is not None and not member_error:
        return f"Invalid value for '{param}': expected {description}, but got {got} instead."
    return f"Invalid value for '{param}': expected {description}."


def unknown_parameter_message(param: str) -> str:
    """State that a field the gateway does not define is unsupported and must go.

    Pydantic reports an undefined key as ``extra_forbidden``, which carries no
    expected shape; a bare ``Invalid value for '<param>'.`` would read as a
    value fault when the key itself is the problem (an agent framework's own
    flag on a function tool, such as Agno's ``requires_confirmation``). The
    code stays ``invalid_parameter``; only the message names the fix.
    """
    return f"Unknown parameter '{param}': this field is not supported. Remove it and resend."


def _names_unknown_field(
    groups: dict[tuple[str | int, ...], list[tuple[tuple[str, ...], ErrorDetails]]],
    location: tuple[str, ...],
    details: list[ErrorDetails],
) -> bool:
    """Whether the chosen field is a key no union branch defines.

    Every complaint at the chosen location must be ``extra_forbidden``, and
    every closed sibling branch (one that reported some other stray key beside
    it) must have called this key extra too: a closed branch that defines the
    key reports either a value fault there or nothing there at all, and in
    both cases the caller may have meant that branch, so "unknown parameter"
    would misdescribe it.
    """
    if not details or any(detail["type"] != "extra_forbidden" for detail in details):
        return False
    parent = location[:-1]
    for members in groups.values():
        siblings = [
            (cleaned, detail)
            for cleaned, detail in members
            if len(cleaned) == len(location) and cleaned[:-1] == parent
        ]
        if not any(detail["type"] == "extra_forbidden" for _, detail in siblings):
            # A branch that never reports a stray key (an open shape, or one
            # that failed before its extras were checked) says nothing about
            # whether it defines this one.
            continue
        here = [detail for cleaned, detail in siblings if cleaned == location]
        if not here or any(detail["type"] != "extra_forbidden" for detail in here):
            return False
    return True


_MAXIMUM_ECHOED_PARAM_CHARACTERS = 256
"""Longest field path echoed back in an error's ``param`` and message.

An undefined key is the caller's own text and may be arbitrarily long; the
error model caps ``param`` at 512 characters and the message at 2,048, so an
unbounded echo would fail to build the 400 it describes.
"""


def _bounded_param(param: str) -> str:
    """Abbreviate an over-long field path so the refusal can always be built.

    Args:
        param: Dotted field path, possibly ending in a caller-chosen key.

    Returns:
        The path unchanged when short enough, else its leading characters
        followed by ``...``.
    """
    if len(param) <= _MAXIMUM_ECHOED_PARAM_CHARACTERS:
        return param
    return param[: _MAXIMUM_ECHOED_PARAM_CHARACTERS - 3] + "..."


def validation_protocol_error(error: ValidationError) -> OpenAIProtocolError:
    """Convert Pydantic locations into stable dotted OpenAI ``param`` paths.

    Union validation reports every branch's complaints. Errors group by
    their branch (the location minus its final field segment); among the
    most field-specific groups, the branch the caller actually meant is the
    one with the fewest complaints, so its deepest cleaned location names
    the real field (an echoed item's ``input.1.caller``), never a union
    branch label such as ``input.str``. The chosen field's own complaints
    then name the expected shape against the arriving JSON type; a
    vocabulary field pools the choices every sibling branch accepts at that
    same path, so the message lists the whole accepted set rather than the
    one branch that happened to complain least.
    """
    every_detail = error.errors(include_url=False)
    declared = _declared_unknown_tag(every_detail)
    if declared is not None:
        return invalid_field(*declared)
    groups: dict[tuple[str | int, ...], list[tuple[tuple[str, ...], ErrorDetails]]] = {}
    for detail in _without_rejected_branches(every_detail):
        groups.setdefault(tuple(detail["loc"][:-1]), []).append(
            (
                cleaned_location(detail["loc"], keep_last=detail["type"] == "extra_forbidden"),
                detail,
            )
        )
    if not groups:
        return invalid_field("body")
    deepest = max(len(location) for members in groups.values() for location, _ in members)
    candidates = [
        members
        for members in groups.values()
        if any(len(location) == deepest for location, _ in members)
    ]
    best = min(candidates, key=len)
    location = max((cleaned for cleaned, _ in best), key=len, default=())
    param = _bounded_param(".".join(location) or "body")
    details = [detail for cleaned, detail in best if cleaned == location]
    if _names_unknown_field(groups, location, details):
        return invalid_field(param, unknown_parameter_message(param))
    if param.startswith("gateway."):
        for detail in details:
            if detail["type"] in {
                "greater_than_equal",
                "less_than_equal",
                "finite_number",
                "string_pattern_mismatch",
            }:
                return invalid_field(
                    param,
                    f"Invalid value for '{param}': {detail['msg']}. Correct the field and resend.",
                )
    if param == "body":
        # A whole-request rule (such as the attachment count ceiling) has no
        # field of its own, so its own wording is the only useful message.
        for detail in details:
            if detail["type"] == "value_error":
                return invalid_field(param, detail["msg"].removeprefix("Value error, ") + ".")
    for detail in details:
        # A field validator's own wording states this gateway's exact value
        # constraint (only our wire models raise these, so the text is
        # display-safe and never echoes the caller's value).
        if detail["type"] == "value_error":
            return invalid_field(
                param,
                f"Invalid value for {param!r}: "
                + detail["msg"].removeprefix("Value error, ")
                + ".",
            )
    if not any(detail["type"] in _EXPECTED_BY_ERROR_TYPE for detail in details):
        # A part that IS an object but carries a wrong or missing tag: name
        # the tag field and the accepted vocabulary. A part of the wrong JSON
        # shape altogether (a bare string where an object belongs) is a shape
        # complaint below, not a tag complaint.
        for detail in details:
            discriminated = _discriminator_message(param, detail)
            if discriminated is not None:
                tag_param, message = discriminated
                return invalid_field(tag_param, message)
    if all(detail["type"] in {"literal_error", "enum"} for detail in details):
        # Pool the sibling branches' choices for the same vocabulary field.
        details = [
            detail
            for members in groups.values()
            for cleaned, detail in members
            if cleaned == location and detail["type"] in {"literal_error", "enum"}
        ]
    return invalid_field(param, shape_message(param, details))

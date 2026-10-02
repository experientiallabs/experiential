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


def cleaned_location(location: tuple[str | int, ...]) -> tuple[str, ...]:
    """Drop pydantic union-branch labels so the path names request fields.

    A discriminated-union branch label is never the LAST segment of a location
    (an error inside a branch names the field after it), so a final segment
    that merely spells like one (a message key named ``reasoning``) is the
    caller's own field and is kept.
    """
    cleaned: list[str] = []
    last_index = len(location) - 1
    for index, part in enumerate(location):
        text = str(part)
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
        ):
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
    if allowed:
        expected.insert(0, f"one of {_join_quoted(allowed)}")
    if not expected:
        return None
    description = " or ".join(expected)
    if got is not None and not member_error:
        return f"Invalid value for '{param}': expected {description}, but got {got} instead."
    return f"Invalid value for '{param}': expected {description}."


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
    groups: dict[tuple[str | int, ...], list[tuple[tuple[str, ...], ErrorDetails]]] = {}
    for detail in error.errors(include_url=False):
        groups.setdefault(tuple(detail["loc"][:-1]), []).append(
            (cleaned_location(detail["loc"]), detail)
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
    param = ".".join(location) or "body"
    details = [detail for cleaned, detail in best if cleaned == location]
    if param.startswith("gateway."):
        for detail in details:
            if detail["type"] == "extra_forbidden":
                return invalid_field(param, f"Unknown parameter '{param}'. Remove it and resend.")
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

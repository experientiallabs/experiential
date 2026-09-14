"""Public rejection shaping for the OpenAI-protocol request decoders.

Every strict wire model in :mod:`exp.runtime.openai_protocol.wire_models` is
validated through this module, so a caller that sends a value the gateway
cannot serve reads a message that names the field, the constraint it violated,
and what arrived. Only structural facts appear: expectations come from this
gateway's own wire models, and the arrived side is the JSON type of the
caller's value, never the value itself and never provider prose.

Two rejection families are shaped here:

- Shape and bound failures (a wrong JSON type, an out-of-range number, an
  over-long string, a list of the wrong length) name the constraint and the
  arriving shape, so the caller never has to bisect a ceiling out of a bare
  field path.
- Discriminated-union failures name the offending tag and the accepted ones.
  Pydantic reports a bad tag at the union's own location and then adds
  speculative complaints from the arm it tried first (a ``missing`` sibling
  field); those arm complaints are dropped in favor of the tag diagnosis,
  because the caller's actual mistake is the tag.
"""

from __future__ import annotations

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
    "bool_parsing": "a boolean",
    "list_type": "an array",
    "tuple_type": "an array",
    "dict_type": "an object",
    "model_type": "an object",
    "model_attributes_type": "an object",
    "missing": "a value",
    "none_required": "null",
}
"""Shape-level expectations for the pydantic error types worth naming."""

_BOUND_ERROR_TYPES = {
    "less_than": ("lt", "less than"),
    "less_than_equal": ("le", "at most"),
    "greater_than": ("gt", "greater than"),
    "greater_than_equal": ("ge", "at least"),
}
"""Numeric-bound error types mapped to their context key and comparison phrase.

Each pydantic comparison error carries its own limit in ``ctx`` under this key,
so the message can state the exact bound instead of only the field path. The
numeric limits are this gateway's own wire-model constraints, so stating them
leaks nothing about the caller's value.
"""


def _format_bound(value: object) -> str:
    """Render one numeric limit without a trailing ``.0`` on whole floats."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _cleaned_location(location: tuple[str | int, ...]) -> tuple[str, ...]:
    """Drop pydantic union-branch labels so the path names request fields."""
    cleaned: list[str] = []
    for part in location:
        text = str(part)
        if text in _LOCATION_NOISE:
            continue
        # Typed-dict union branches are labeled with their class name, which
        # no request field ever shares: every public field is lower case.
        if isinstance(part, str) and (
            part.startswith("_") or "[" in text or text in _UNION_BRANCH_TYPES or text[:1].isupper()
        ):
            continue
        if text in _OUTPUT_ITEM_VARIANTS and cleaned and cleaned[-1].isdigit():
            continue
        # A discriminated part whose tag is also its payload field name
        # (``file.file``, ``image_url.image_url``) reports the tag once.
        if cleaned and cleaned[-1] == text and not text.isdigit():
            continue
        cleaned.append(text)
    return tuple(cleaned)


def _tag_message(param: str, detail: ErrorDetails) -> tuple[str, str]:
    """Name the missing or rejected union tag and the accepted tags there.

    Returns the field path (the union location plus its discriminator) and the
    message together, so a caller's path and prose can never disagree.

    Pydantic supplies the discriminator's property name and the accepted tag
    list, both of which are this gateway's own vocabulary. The arriving tag is
    deliberately not echoed, exactly like every other value in these messages:
    the caller learns which field is wrong and which selectors are valid, which
    is what the rejection has to say.
    """
    context = detail.get("ctx") or {}
    discriminator = context.get("discriminator")
    tag_field = discriminator.strip("'\"") if isinstance(discriminator, str) else None
    field = f"{param}.{tag_field}" if tag_field else param
    expected = context.get("expected_tags")
    if isinstance(expected, str) and expected:
        return field, f"Invalid value for '{field}': expected one of {expected}."
    return field, f"Invalid value for '{field}': a type selector is required here."


def _tag_rejection(
    reported: list[ErrorDetails],
    selected: list[ErrorDetails],
    param: str,
) -> tuple[str, str] | None:
    """Prefer a bad-discriminator tag over a union arm's speculative complaint.

    A caller that sends an unknown item or content-part ``type`` gets pydantic's
    ``union_tag_invalid`` at the union's own location plus complaints from the
    arm it happened to try first (typically a ``missing`` field of a sibling
    shape). Reporting the arm complaint names a field the caller never sent, so
    when the selected detail is speculative and a tag rejection exists, the tag
    diagnosis wins. When the selected detail *is* the tag rejection, its own
    location becomes the param so the path and the message agree.
    """
    tag_types = {"union_tag_invalid", "union_tag_not_found"}
    chosen: ErrorDetails | None = None
    if any(detail["type"] in tag_types for detail in selected):
        chosen = next(detail for detail in selected if detail["type"] in tag_types)
    elif selected and all(detail["type"] == "missing" for detail in selected):
        chosen = next(
            (detail for detail in reported if detail["type"] == "union_tag_invalid"),
            None,
        )
    if chosen is None:
        return None
    location = _cleaned_location(chosen["loc"])
    return _tag_message(".".join(location) or param, chosen)


def _shape_message(param: str, details: list[ErrorDetails]) -> str | None:
    """Describe what shape a field expected versus what arrived.

    Only structural facts appear: expectations come from this gateway's own
    wire models and the got side is the JSON type of the caller's value,
    never the value itself and never provider prose.
    """
    expected: list[str] = []
    got: str | None = None
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
        bound = _BOUND_ERROR_TYPES.get(detail["type"])
        if bound is not None:
            key, phrase = bound
            limit = (detail.get("ctx") or {}).get(key)
            if limit is not None:
                return f"Invalid value for '{param}': expected {phrase} {_format_bound(limit)}."
        if detail["type"] == "too_short":
            shortage = _shortage_message(param, detail)
            if shortage is not None:
                return shortage
        if detail["type"] == "extra_forbidden":
            # An undeclared key inside a nested object (a tool entry, a text
            # block) is the same rejection the manifest already names at the
            # top level: state it consistently instead of leaving a bare path.
            return f"Unknown parameter '{param}'. Remove the field and resend the request."
        phrase = _EXPECTED_BY_ERROR_TYPE.get(detail["type"])
        if detail["type"] in {"literal_error", "enum"}:
            context = detail.get("ctx") or {}
            allowed = context.get("expected")
            if isinstance(allowed, str):
                phrase = f"one of {allowed}"
        if phrase is not None and phrase not in expected:
            expected.append(phrase)
        # A missing-field complaint carries the parent object as its input,
        # so it contributes no honest "got" type.
        if detail["type"] != "missing" and "input" in detail:
            got = _WIRE_TYPE_NAMES.get(type(detail["input"]).__name__, got)
    if not expected:
        return None
    description = " or ".join(expected)
    if got is not None:
        return f"Invalid value for '{param}': expected {description}, but got {got} instead."
    return f"Invalid value for '{param}': expected {description}."


def _shortage_message(param: str, detail: ErrorDetails) -> str | None:
    """Name a list too short for its declared minimum, with both counts."""
    context = detail.get("ctx") or {}
    minimum = context.get("min_length")
    actual = context.get("actual_length")
    if isinstance(minimum, int) and isinstance(actual, int):
        return (
            f"Invalid value for '{param}': expected at least {minimum:,} "
            f"{'entry' if minimum == 1 else 'entries'}, but got {actual:,}."
        )
    return None


def _validation_protocol_error(error: ValidationError) -> OpenAIProtocolError:
    """Convert Pydantic locations into stable dotted OpenAI ``param`` paths.

    Union validation reports every branch's complaints. Errors group by
    their branch (the location minus its final field segment); among the
    most field-specific groups, the branch the caller actually meant is the
    one with the fewest complaints, so its deepest cleaned location names
    the real field (an echoed item's ``input.1.caller``), never a union
    branch label such as ``input.str``. The chosen field's own complaints
    then name the expected shape against the arriving JSON type.
    """
    reported = error.errors(include_url=False)
    groups: dict[tuple[str | int, ...], list[tuple[tuple[str, ...], ErrorDetails]]] = {}
    for detail in reported:
        groups.setdefault(tuple(detail["loc"][:-1]), []).append(
            (_cleaned_location(detail["loc"]), detail)
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
    # A tag rejection can carry the field itself as its location, so it may
    # rename the param; the message and the path are returned together so the
    # two never disagree.
    tag = _tag_rejection(reported, details, param)
    if tag is not None:
        return invalid_field(tag[0], tag[1])
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
    return invalid_field(param, _shape_message(param, details))

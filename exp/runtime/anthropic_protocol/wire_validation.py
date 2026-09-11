"""Wire validation and public error rendering for the Anthropic Messages decoder.

The strict wire models live beside the decoder in ``requests.py``; this module
turns their Pydantic rejections into the stable, field-specific public errors
the HTTP layer renders in the Anthropic envelope, and recognizes the
known-but-unsupported content blocks (video, audio, documents inside a
``tool_result``) whose union miss would otherwise misdirect the caller.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, invalid_field

_REJECTED_BLOCK_HINTS = {
    kind: (
        f"{kind} blocks are not supported: the Anthropic Messages wire defines no "
        f"{kind} content, so send {kind} on the Chat Completions surface"
    )
    for kind in ("video", "audio")
}
_REJECTED_TOOL_RESULT_BLOCK_HINTS = {
    "document": "document blocks are not supported inside tool_result content",
    **_REJECTED_BLOCK_HINTS,
}


def validate_wire[WireModelT: BaseModel](payload: JsonObject, wire: type[WireModelT]) -> WireModelT:
    """Validate ``payload`` against the strict wire model ``wire``.

    Args:
        payload: Parsed JSON request body.
        wire: The closed Messages wire model to validate against (the
            generation request, or the ``count_tokens`` variant).

    Returns:
        The validated wire model.

    Raises:
        OpenAIProtocolError: The body fails validation; the error names the
            exact offending field (a known-but-unsupported block first, else
            the deepest Pydantic location) and states what was expected.
    """
    try:
        return wire.model_validate(payload)
    except ValidationError as exc:
        hint = rejected_block_hint(payload)
        if hint is not None:
            param, message = hint
            raise invalid_field(param, message) from exc
        # A union miss reports one error PER ARM, and the first arm is the
        # scalar one: naming it ("content.str: Input should be a valid
        # string") misdirects a caller whose list merely held an unsupported
        # block. The deepest location is the arm that actually matched the
        # payload's shape, so its error names the offending element.
        errors = exc.errors(include_url=False)
        first = max(errors, key=lambda error: len(error["loc"]))
        raise validation_error(first) from exc


def validation_error(first: ErrorDetails) -> OpenAIProtocolError:
    """Convert one Pydantic error location into a stable dotted field error.

    The public message keeps the expected-vs-got shape: it names the field
    and states what the decoder expected there (Pydantic's own expectation
    text, which never echoes the caller's value), so a rejected request says
    what to fix instead of only where it failed.
    """
    location = first["loc"]
    cleaned: list[str] = []
    for part in location:
        text = str(part)
        # Union arm labels in pydantic locations are noise for callers: wire
        # model class names (private or public), scalar type names, and
        # constrained-type spellings. Real wire fields are snake_case.
        if isinstance(part, str) and (
            part.startswith("_")
            or "[" in text
            or text[:1].isupper()
            or text in ("str", "int", "float", "bool", "none", "list", "dict")
        ):
            continue
        cleaned.append(text)
    param = ".".join(cleaned) or "body"
    if first["type"] == "extra_forbidden":
        return invalid_field(
            param,
            f"Unknown parameter '{param}'. Remove the field and resend the request.",
        )
    if param == "body" and first["type"] == "value_error":
        # A whole-request rule (such as the attachment count ceiling) has no
        # field of its own, so its own wording is the only useful message.
        return invalid_field(param, first["msg"].removeprefix("Value error, ") + ".")
    return invalid_field(param, f"Invalid value for '{param}': {first['msg']}.")


def rejected_block_hint(payload: JsonObject) -> tuple[str, str] | None:
    """Return the field path and message for a known-but-unsupported block.

    The path names the exact offending block (and, for a ``tool_result``, the
    offending sub-block), so the caller is never sent to the union's string
    arm for a list-shaped problem.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block_index, block in enumerate(cast(list[object], message["content"])):
            if not isinstance(block, dict):
                continue
            block_object = cast(JsonObject, block)
            param = f"messages.{message_index}.content.{block_index}"
            hint = _REJECTED_BLOCK_HINTS.get(str(block_object.get("type")))
            if hint is not None:
                return param, hint
            if block_object.get("type") == "tool_result" and isinstance(
                block_object.get("content"), list
            ):
                for inner_index, inner in enumerate(cast(list[object], block_object["content"])):
                    if not isinstance(inner, dict):
                        continue
                    inner_type = str(cast(JsonObject, inner).get("type"))
                    inner_param = f"{param}.content.{inner_index}"
                    hint = _REJECTED_TOOL_RESULT_BLOCK_HINTS.get(inner_type)
                    if hint is not None:
                        return inner_param, hint
                    if inner_type not in ("text", "image"):
                        return inner_param, (
                            f"unsupported block type '{inner_type}' inside tool_result "
                            "content; only text and image sub-blocks are supported."
                        )
    return None

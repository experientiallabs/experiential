"""Anthropic's programmatic tool-calling ``caller`` on a history ``tool_use`` block.

Programmatic tool calling (stable, no beta header) stamps every response
``tool_use`` with how it was invoked, and Messages clients such as Claude Code
echo the block back in history. The decoder admits exactly the official SDK's
request shapes (``DirectCallerParam``, ``ServerToolCallerParam``,
``ServerToolCaller20260120Param``), validated closed, so anything else stays a
field-specific 400.
"""

from __future__ import annotations

from typing import Annotated, Literal, get_args

from pydantic import Discriminator, Field, Tag

from exp.runtime.anthropic_protocol.media_blocks import AnthropicWireModel


class _DirectCaller(AnthropicWireModel):
    """A ``tool_use`` the model invoked directly (traditional tool use).

    Attributes:
        type: Always ``direct``.
    """

    type: Literal["direct"]


class _CodeExecutionCaller(AnthropicWireModel):
    """A ``tool_use`` invoked programmatically from a code execution run.

    Attributes:
        type: The code execution tool version. Anthropic tags responses
            ``code_execution_20260120``; the SDK's request params also admit the
            older ``code_execution_20250825``.
        tool_id: The id of the code execution ``server_tool_use`` block that made
            the call.
    """

    type: Literal["code_execution_20250825", "code_execution_20260120"]
    tool_id: str = Field(min_length=1, max_length=256)


_CODE_EXECUTION_CALLER_TYPES = frozenset(
    get_args(_CodeExecutionCaller.model_fields["type"].annotation)
)
"""The code execution ``type`` tags, read from the arm so a new version is one edit."""


def _caller_arm(value: object) -> str | None:
    """Pick the ``caller`` arm by its ``type`` tag.

    The tags are wire-model class names so the error renderer strips them from
    the field path, exactly as it strips ordinary union arm labels.
    """
    tag = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    if tag == "direct":
        return "_DirectCaller"
    if tag in _CODE_EXECUTION_CALLER_TYPES:
        return "_CodeExecutionCaller"
    return None


ToolUseCaller = Annotated[
    Annotated[_DirectCaller, Tag("_DirectCaller")]
    | Annotated[_CodeExecutionCaller, Tag("_CodeExecutionCaller")],
    Discriminator(
        _caller_arm,
        custom_error_type="invalid_caller",
        custom_error_message=(
            "caller must be an object whose type is 'direct', "
            "'code_execution_20250825', or 'code_execution_20260120'"
        ),
    ),
]
"""The ``caller`` a history ``tool_use`` block may carry, discriminated on ``type``."""

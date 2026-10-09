"""Strict wire schemas for OpenAI custom (freeform) tools on Chat and Responses.

OpenAI spells one custom tool two ways. Chat nests the declaration and the call
under a ``custom`` object (``{"type": "custom", "custom": {"name", "format"}}``,
``{"id", "type": "custom", "custom": {"name", "input"}}``) and nests a grammar
format one level deeper (``{"type": "grammar", "grammar": {"syntax",
"definition"}}``). Responses keeps every field flat on the item. Decode turns
the Chat spelling into the flat Responses one, which is the gateway's canonical
custom tool (see :mod:`exp.runtime.openai_protocol.custom_tools`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import MAXIMUM_TOOL_CALL_ID_CHARACTERS

EchoedItemStatus = Literal["in_progress", "completed", "incomplete"]
"""Lifecycle marker carried by echoed output items; validated and dropped."""


class _StrictWire(BaseModel):
    """Strict private wire model: unknown keys are refused."""

    model_config = ConfigDict(extra="forbid")


class ChatCustomToolGrammar(_StrictWire):
    """The grammar a Chat custom tool's input must match.

    Attributes:
        syntax: OpenAI's grammar dialect, ``lark`` or ``regex``.
        definition: The grammar text, forwarded verbatim.
    """

    syntax: Literal["lark", "regex"]
    definition: str = Field(min_length=1)


class ChatCustomToolFormat(_StrictWire):
    """A Chat custom tool's input format: free text or a grammar.

    Attributes:
        type: ``text`` for unconstrained input, ``grammar`` for a grammar.
        grammar: The grammar, present exactly when ``type`` is ``grammar``.
    """

    type: Literal["text", "grammar"]
    grammar: ChatCustomToolGrammar | None = None

    @model_validator(mode="after")
    def _require_grammar_body(self) -> ChatCustomToolFormat:
        """Require the grammar body exactly for the grammar format."""
        if (self.type == "grammar") != (self.grammar is not None):
            raise ValueError("format.grammar must be present exactly when format.type is 'grammar'")
        return self


class ChatCustomTool(_StrictWire):
    """The ``custom`` body of one Chat custom tool declaration.

    Attributes:
        name: Required nonempty tool name, at most 256 characters.
        description: Optional verbatim instructions.
        format: Optional input format; unconstrained text when absent.
    """

    name: str = Field(min_length=1, max_length=256)
    description: str | None = None
    format: ChatCustomToolFormat | None = None


class ChatCustomCall(_StrictWire):
    """The ``custom`` body of one Chat custom tool call.

    Attributes:
        name: The called custom tool.
        input: The model's freeform input, verbatim.
    """

    name: str = Field(min_length=1, max_length=256)
    input: str = Field(max_length=4_000_000)


class ChatCustomToolCall(_StrictWire):
    """One assistant custom tool call retained in Chat request history.

    Attributes:
        id: The call identity a later ``role: "tool"`` result names.
        type: Always ``custom``.
        custom: The tool name and freeform input.
        index: Streaming delta ordinal, validated and dropped like a function call's.
    """

    id: str = Field(min_length=1, max_length=MAXIMUM_TOOL_CALL_ID_CHARACTERS)
    type: Literal["custom"]
    custom: ChatCustomCall
    index: int | None = Field(default=None, ge=0)


class CustomToolCallItem(_StrictWire):
    """One freeform (custom) tool call echoed as Responses assistant history.

    ``namespace`` attributes the call to its declaring nested tool tree and
    forwards verbatim with the rest of the raw item (the whole item replays
    byte-for-byte on native Responses rungs), mirroring the typed
    ``function_call`` namespace round trip.
    """

    type: Literal["custom_tool_call"]
    id: str | None = Field(default=None, min_length=1, max_length=256)
    status: EchoedItemStatus | None = None
    call_id: str = Field(min_length=1, max_length=MAXIMUM_TOOL_CALL_ID_CHARACTERS)
    name: str = Field(min_length=1, max_length=256)
    namespace: str | None = Field(default=None, min_length=1, max_length=256)
    caller: JsonObject | None = None
    """Opaque SDK 3.0 programmatic tool-calling attribution, forwarded
    verbatim with the rest of the raw item like ``namespace``."""
    input: str = Field(max_length=4_000_000)


class CustomToolCallOutputItem(_StrictWire):
    """One freeform tool result echoed as Responses tool history."""

    type: Literal["custom_tool_call_output"]
    id: str | None = Field(default=None, min_length=1, max_length=256)
    status: EchoedItemStatus | None = None
    call_id: str = Field(min_length=1, max_length=MAXIMUM_TOOL_CALL_ID_CHARACTERS)
    output: JsonValue

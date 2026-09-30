"""Strict private schema for OpenAI Responses multi-agent input items."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exp.common.core.artifacts import JsonValue


class _AgentMessageModel(BaseModel):
    """Closed field model used by Responses multi-agent input schemas."""

    model_config = ConfigDict(extra="forbid", strict=True)


class _AgentMessageAttribution(_AgentMessageModel):
    """Optional agent attribution on one Responses agent message.

    Attributes:
        agent_name: Attributed agent name supplied by the caller.
    """

    agent_name: str


class _AgentMessageInputTextPart(_AgentMessageModel):
    """Visible text part accepted inside a Responses agent message.

    Attributes:
        type: Required ``input_text`` content discriminator.
        text: Visible content forwarded with the raw item on native Responses routes.
    """

    type: Literal["input_text"]
    text: str


class _AgentMessageEncryptedContentPart(_AgentMessageModel):
    """Opaque encrypted part accepted only inside a Responses agent message.

    Attributes:
        type: Required ``encrypted_content`` content discriminator.
        encrypted_content: Opaque string forwarded unchanged on native Responses routes.
    """

    type: Literal["encrypted_content"]
    encrypted_content: str


_AgentMessageContentPart = Annotated[
    _AgentMessageInputTextPart | _AgentMessageEncryptedContentPart,
    Field(discriminator="type"),
]


class _AgentMessageInput(_AgentMessageModel):
    """One validated OpenAI Responses multi-agent input item.

    Attributes:
        type: Required ``agent_message`` discriminator.
        id: Optional provider item identity, omitted when the caller has no ID.
        author: Name of the sending agent.
        recipient: Name of the receiving agent.
        content: Documented text or opaque encrypted parts in caller order.
        agent: Optional recipient attribution, preserved when supplied.
    """

    type: Literal["agent_message"]
    id: str | None = Field(default=None, validate_default=False)
    author: str
    recipient: str
    content: list[_AgentMessageContentPart]
    agent: _AgentMessageAttribution | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _reject_explicit_null_id(cls, value: JsonValue) -> JsonValue:
        """Require absent optional IDs to stay omitted instead of becoming null."""
        if value is None:
            raise ValueError("Omit id when no response item ID is available.")
        return value

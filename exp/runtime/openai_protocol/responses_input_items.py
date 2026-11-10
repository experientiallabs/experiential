"""Responses input items that carry no canonical message: agent, compaction, control, reference.

Split from ``wire_models`` for the module line budget. Each item here is
valid OpenAI Responses input that this gateway re-models only shallowly,
because its meaning belongs to the provider that issued or interprets it:

- ``agent_message``: a message routed between agents (OpenAI's multi-agent
  Responses beta, ``openai-python`` 3.37 ``AgentMessage``). Codex records
  every inter-agent delivery this way (``ResponseItem::AgentMessage``,
  codex-rs ``protocol/src/models.rs``): a sub-agent's task from its parent
  and a child's result for its parent, so a sub-agent thread carries one
  early in its input and a parent carries them wherever a child reported.
- ``compaction_trigger``: the request for an encrypted context compaction
  (``openai-python`` 3.0 ``CompactionTrigger``, "Must be the final input
  item"). Codex appends it to the whole history whenever the provider is the
  built-in OpenAI one (remote compaction v2), so it arrives at a large index.
- ``context_compaction``: the opaque compacted history the OpenAI backend
  returns (``ResponseItem::ContextCompaction``), replayed like ``compaction``.
- ``configuration_update``: Codex's positional reasoning-effort change
  (``ResponseItem::ConfigurationUpdate``), interpreted by the backend at its
  position in history.
- ``item_reference``: a stored item named by id (``openai-python`` 3.0
  ``ItemReference``), which the AI SDK and LiteLLM send for stored reasoning.

The strict wire union in ``wire_models`` includes every model here, so none
of them falls into the message branch (which would read it as a message
without a role). The decoder carries the raw caller item verbatim (``ReplayedNativeItem``): a
native OpenAI Responses rung receives it byte-for-byte, and a foreign wire
translates or drops it with disclosure (``codex_tools``).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import MAXIMUM_IMAGE_BASE64_BYTES

_MAXIMUM_AGENT_PATH_CHARACTERS = 1_024
"""Bound on an agent identity (Codex agent paths such as ``/root/worker``)."""

_MAXIMUM_ITEM_ID_CHARACTERS = 256
"""Bound on an echoed provider item id, matching the other Responses items."""

_MAXIMUM_IMAGE_URL_CHARACTERS = MAXIMUM_IMAGE_BASE64_BYTES + 128
"""Room for the largest inline image plus its ``data:`` URL preamble, as on a message part."""

_MAXIMUM_FILE_ID_CHARACTERS = 512
"""Bound on an uploaded file id, as on a message ``input_image`` part."""


class _AgentMessageText(BaseModel):
    """One plaintext part of an agent message.

    Attributes:
        type: Required ``input_text`` discriminator.
        text: Required text, preserved verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["input_text"]
    text: str


class _AgentMessageEncrypted(BaseModel):
    """One opaque part only the issuing OpenAI backend can decrypt.

    Attributes:
        type: Required ``encrypted_content`` discriminator.
        encrypted_content: Required nonempty ciphertext, forwarded verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["encrypted_content"]
    encrypted_content: str = Field(min_length=1)


class _AgentMessageImage(BaseModel):
    """One image part; its fields are the Responses ``input_image`` shape the provider owns.

    Attributes:
        type: Required ``input_image`` discriminator.
        image_url: Optional nonempty image URL or data URL; exactly one carrier is required.
        file_id: Optional nonempty uploaded file id; exactly one carrier is required.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["input_image"]
    image_url: str | None = Field(
        default=None, min_length=1, max_length=_MAXIMUM_IMAGE_URL_CHARACTERS
    )
    file_id: str | None = Field(default=None, min_length=1, max_length=_MAXIMUM_FILE_ID_CHARACTERS)

    @model_validator(mode="after")
    def _require_one_carrier(self) -> _AgentMessageImage:
        """Require exactly one of ``image_url`` or ``file_id``, like a message image part."""
        if (self.image_url is None) == (self.file_id is None):
            raise ValueError("input_image needs exactly one of image_url or file_id")
        return self


_AgentMessagePart = Annotated[
    _AgentMessageText | _AgentMessageEncrypted | _AgentMessageImage,
    Field(discriminator="type"),
]


class AgentMessageItem(BaseModel):
    """A message routed between agents, replayed as input history.

    Attributes:
        type: Required ``agent_message`` discriminator.
        id: Optional nonempty echoed item id.
        author: Required sending agent identity.
        recipient: Required destination agent identity.
        content: Ordered plaintext, image, or encrypted parts.
        agent: Optional producing-agent attribution (``{"agent_name": ...}``), forwarded verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["agent_message"]
    id: str | None = Field(default=None, min_length=1, max_length=_MAXIMUM_ITEM_ID_CHARACTERS)
    author: str = Field(min_length=1, max_length=_MAXIMUM_AGENT_PATH_CHARACTERS)
    recipient: str = Field(min_length=1, max_length=_MAXIMUM_AGENT_PATH_CHARACTERS)
    content: tuple[_AgentMessagePart, ...]
    agent: JsonObject | None = None


class CompactionTriggerItem(BaseModel):
    """The request for an encrypted context compaction; it has no other field.

    Attributes:
        type: Required ``compaction_trigger`` discriminator.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["compaction_trigger"]


class ContextCompactionItem(BaseModel):
    """Opaque compacted history an OpenAI backend issued, replayed verbatim.

    Validation is deliberately shallow, like the hosted-tool echoes: the item
    is provider-owned vocabulary and its other fields pass through unchanged.

    Attributes:
        type: Required ``context_compaction`` discriminator.
        id: Optional nonempty echoed item id.
        encrypted_content: Optional nonempty opaque payload.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["context_compaction"]
    id: str | None = Field(default=None, min_length=1, max_length=_MAXIMUM_ITEM_ID_CHARACTERS)
    encrypted_content: str | None = Field(default=None, min_length=1)


class _ConfigurationReasoning(BaseModel):
    """The reasoning setting a configuration update applies from its position on.

    Attributes:
        effort: Required nonempty effort name; the interpreting provider owns the vocabulary.
    """

    model_config = ConfigDict(extra="forbid")

    effort: str = Field(min_length=1, max_length=64)


class ConfigurationUpdateItem(BaseModel):
    """Codex's positional reasoning-effort change.

    Attributes:
        type: Required ``configuration_update`` discriminator.
        reasoning: Required reasoning setting.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["configuration_update"]
    reasoning: _ConfigurationReasoning


class ItemReferenceItem(BaseModel):
    """A stored item named by id; only the issuing provider can resolve it.

    The SDK also admits the id alone without ``type``; the client-field
    repairs give such an item its ``item_reference`` tag before validation.

    Attributes:
        type: Required ``item_reference`` discriminator.
        id: Required nonempty referenced item id.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["item_reference"]
    id: str = Field(min_length=1, max_length=_MAXIMUM_ITEM_ID_CHARACTERS)


EXTRA_INPUT_ITEM_TYPES = frozenset(
    {
        "agent_message",
        "compaction_trigger",
        "context_compaction",
        "configuration_update",
        "item_reference",
    }
)
"""The discriminators of every item model in this module."""

SDK_UNKNOWN_INPUT_ITEM_TYPES = frozenset(
    {"agent_message", "context_compaction", "configuration_update"}
)
"""Item types the installed ``openai`` SDK schema does not enumerate.

``agent_message`` is in the SDK's beta Responses types only, and the other
two are Codex-recorded backend items; the strict models above own them, so
the official-schema probe does not see them (``official_responses_probe``)."""

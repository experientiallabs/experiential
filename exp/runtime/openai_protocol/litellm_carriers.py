"""LiteLLM's reasoning carriers on an echoed assistant message, validated and dropped.

LiteLLM's ``Message.model_dump()`` writes ``thinking_blocks`` and
``reasoning_items`` on every assistant message, and with a reasoning model it
populates them: Anthropic-style ``thinking`` / ``redacted_thinking`` blocks
(``ChatCompletionThinkingBlock`` / ``ChatCompletionRedactedThinkingBlock`` in
``litellm/types/llms/openai.py``) or OpenAI Responses reasoning items
(``ChatCompletionReasoningItem``). Agent loops keep the message object and
echo the dump on every later turn, so rejecting a populated carrier fails the
whole session.

The models here cover exactly those documented shapes, so anything else is
still a field-named 400. Like ``provider_specific_fields`` the carriers are
never forwarded and their drop is disclosed: the same turn already carries
its reasoning as ``reasoning_content`` (which LiteLLM sets beside them and the
gateway replays), a thinking signature or encrypted item is bound to the
issuing provider account, and a summary is display copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.reasoning_carrier import MAXIMUM_REASONING_CARRIER_BYTES


class _LiteLLMModel(BaseModel):
    """Strict base: an undocumented key inside a carrier block is a named rejection."""

    model_config = ConfigDict(extra="forbid")


_BoundedReasoningText = Annotated[str, Field(max_length=MAXIMUM_REASONING_CARRIER_BYTES)]
"""One reasoning payload string inside a LiteLLM carrier, bounded like the replay field."""


class LiteLLMThinkingBlock(_LiteLLMModel):
    """LiteLLM's ``ChatCompletionThinkingBlock`` on an echoed assistant message.

    LiteLLM (``litellm/types/llms/openai.py``) surfaces an Anthropic-style
    reasoning model's thinking as ``{"type": "thinking", "thinking": ...,
    "signature": ...}`` with optional ``cache_control`` and ``index``. The same
    turn carries the plaintext as ``reasoning_content``, so the block is
    validated and dropped with disclosure, never forwarded.

    Attributes:
        type: The discriminator, always ``"thinking"``.
        thinking: The block's plaintext reasoning, bounded by
            ``MAXIMUM_REASONING_CARRIER_BYTES``; ``None`` when LiteLLM omitted it
            (its non-streaming Bedrock Converse path builds a signature-only block).
        signature: The issuing provider's signature, bounded like ``thinking``;
            ``None`` when LiteLLM recorded none.
        cache_control: LiteLLM's optional cache marker object, never interpreted.
        index: LiteLLM's optional non-negative block position.
    """

    type: Literal["thinking"]
    thinking: _BoundedReasoningText | None = None
    signature: _BoundedReasoningText | None = None
    cache_control: JsonObject | None = None
    index: int | None = Field(default=None, ge=0)


class LiteLLMRedactedThinkingBlock(_LiteLLMModel):
    """LiteLLM's ``ChatCompletionRedactedThinkingBlock``: an opaque ``data`` payload.

    Attributes:
        type: The discriminator, always ``"redacted_thinking"``.
        data: The provider's opaque redacted payload, bounded by
            ``MAXIMUM_REASONING_CARRIER_BYTES``.
        cache_control: LiteLLM's optional cache marker object, never interpreted.
        index: LiteLLM's optional non-negative block position.
    """

    type: Literal["redacted_thinking"]
    data: _BoundedReasoningText
    cache_control: JsonObject | None = None
    index: int | None = Field(default=None, ge=0)


LiteLLMThinkingBlockItem = Annotated[
    LiteLLMThinkingBlock | LiteLLMRedactedThinkingBlock, Field(discriminator="type")
]
"""One ``thinking_blocks`` entry, discriminated on LiteLLM's documented ``type``."""


class LiteLLMReasoningSummaryText(_LiteLLMModel):
    """LiteLLM's ``ChatCompletionReasoningSummaryTextBlock`` inside a reasoning item.

    Attributes:
        type: The discriminator, always ``"summary_text"``.
        text: The summary's display text, bounded by ``MAXIMUM_REASONING_CARRIER_BYTES``.
    """

    type: Literal["summary_text"]
    text: _BoundedReasoningText


class LiteLLMReasoningItem(_LiteLLMModel):
    """LiteLLM's ``ChatCompletionReasoningItem``: an OpenAI Responses reasoning item.

    LiteLLM keeps it for round-tripping a Responses-bridged reasoning model.
    ``encrypted_content`` is bound to the issuing provider account and the
    summary is display copy, so neither is replayable here (the same rule
    :mod:`exp.runtime.openai_protocol.reasoning_replay` applies to OpenRouter's
    encrypted and summary blocks): validated and dropped with disclosure.

    Attributes:
        type: The discriminator, always ``"reasoning"``.
        id: The provider's optional item id, 1 to 256 characters.
        encrypted_content: The provider-bound encrypted reasoning, bounded by
            ``MAXIMUM_REASONING_CARRIER_BYTES``; ``None`` when absent.
        summary: Optional ``summary_text`` parts; ``None`` when absent.
    """

    type: Literal["reasoning"]
    id: str | None = Field(default=None, min_length=1, max_length=256)
    encrypted_content: _BoundedReasoningText | None = None
    summary: tuple[LiteLLMReasoningSummaryText, ...] | None = None


class _CarrierMessage(Protocol):
    """The two carrier fields a wire message exposes."""

    @property
    def thinking_blocks(self) -> tuple[LiteLLMThinkingBlockItem, ...] | None:
        """LiteLLM's echoed thinking blocks."""
        ...

    @property
    def reasoning_items(self) -> tuple[LiteLLMReasoningItem, ...] | None:
        """LiteLLM's echoed Responses reasoning items."""
        ...


def litellm_carrier_disclosures(
    messages: Sequence[_CarrierMessage], prefix: str
) -> tuple[str, ...]:
    """Disclose each populated LiteLLM reasoning carrier once per request.

    No wire takes these fields, so the disclosure does not depend on the route.

    Args:
        messages: The request's validated wire messages.
        prefix: The request's message path (``messages`` or ``input``).

    Returns:
        ``<prefix>.thinking_blocks`` and/or ``<prefix>.reasoning_items``, each
        present when any message populated that carrier.
    """
    carried = (
        ("thinking_blocks", any(message.thinking_blocks for message in messages)),
        ("reasoning_items", any(message.reasoning_items for message in messages)),
    )
    return tuple(f"{prefix}.{field}" for field, present in carried if present)

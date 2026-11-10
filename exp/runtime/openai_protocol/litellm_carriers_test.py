"""LiteLLM reasoning-carrier block shapes and their once-per-request disclosure."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from exp.runtime.openai_protocol.litellm_carriers import (
    LiteLLMReasoningItem,
    LiteLLMRedactedThinkingBlock,
    LiteLLMThinkingBlock,
    LiteLLMThinkingBlockItem,
    litellm_carrier_disclosures,
)

_BLOCKS = TypeAdapter(tuple[LiteLLMThinkingBlockItem, ...])


def test_thinking_blocks_accept_exactly_litellms_documented_shapes() -> None:
    """Both block kinds decode by ``type``; an undocumented shape is refused."""
    blocks = _BLOCKS.validate_python(
        [
            {"type": "thinking", "thinking": "t", "signature": "s", "index": 0},
            {"type": "thinking", "thinking": "t", "signature": None, "cache_control": {}},
            # LiteLLM's non-streaming Bedrock Converse path omits ``thinking``.
            {"type": "thinking", "signature": "s"},
            {"type": "redacted_thinking", "data": "d"},
        ]
    )
    assert [type(block) for block in blocks] == [
        LiteLLMThinkingBlock,
        LiteLLMThinkingBlock,
        LiteLLMThinkingBlock,
        LiteLLMRedactedThinkingBlock,
    ]
    for junk in (
        [{"type": "redacted_thinking", "thinking": "t"}],
        [{"type": "text", "text": "t"}],
        [{"type": "thinking", "thinking": "t", "extra": 1}],
        ["thinking"],
    ):
        with pytest.raises(ValidationError):
            _BLOCKS.validate_python(junk)


def test_reasoning_items_accept_litellms_responses_item_shape() -> None:
    """A Responses reasoning item with a summary decodes; another item type does not."""
    item = LiteLLMReasoningItem.model_validate(
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "x"}]}
    )
    assert item.summary is not None and item.summary[0].text == "x"
    with pytest.raises(ValidationError):
        LiteLLMReasoningItem.model_validate({"type": "message", "id": "m"})


class _Message:
    """A minimal carrier-bearing message."""

    def __init__(
        self,
        thinking_blocks: tuple[LiteLLMThinkingBlockItem, ...] | None = None,
        reasoning_items: tuple[LiteLLMReasoningItem, ...] | None = None,
    ) -> None:
        """Hold the two optional carriers; ``None`` means the key was absent."""
        self.thinking_blocks = thinking_blocks
        self.reasoning_items = reasoning_items


def test_disclosures_name_each_populated_carrier_once() -> None:
    """Empty carriers disclose nothing; a populated one is named once however often it recurs."""
    block = LiteLLMRedactedThinkingBlock(type="redacted_thinking", data="d")
    assert litellm_carrier_disclosures([_Message(()), _Message(None, ())], "messages") == ()
    assert litellm_carrier_disclosures([_Message((block,)), _Message((block,))], "input") == (
        "input.thinking_blocks",
    )

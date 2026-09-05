"""Wire-message translation is covered through the payload builders in
:mod:`exp.runtime.models.providers.streaming_requests_test`; the Anthropic
empty-block conversion contract is pinned here beside its module."""

from exp.common.models.content import ImageContentPart, TextContentPart
from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.models.providers.wire_messages import anthropic_blocks


def test_empty_text_blocks_drop_loss_free_on_the_anthropic_wire() -> None:
    """The wire rejects empty text content blocks post-dispatch ("text content
    blocks must be non-empty"; 2026-09-05, six orgs on claude-fable routes), and
    an empty block carries nothing, so conversion drops it wherever other
    content keeps the array non-empty."""

    multimodal = GatewayMessage(
        role="user",
        content="look",
        content_parts=(
            TextContentPart(text=""),
            ImageContentPart(media_type="image/png", data="aGk="),
            TextContentPart(text="look"),
        ),
    )
    _role, blocks = anthropic_blocks(multimodal)
    assert [block["type"] for block in blocks] == ["image", "text"]
    assert blocks[1]["text"] == "look"

    marked = GatewayMessage(
        role="user",
        content="real",
        provider_text_blocks=(
            {"type": "text", "text": ""},
            {"type": "text", "text": "real", "cache_control": {"type": "ephemeral"}},
        ),
    )
    _role, blocks = anthropic_blocks(marked)
    assert blocks == [{"type": "text", "text": "real", "cache_control": {"type": "ephemeral"}}]

    # An all-empty marked run can only exist on an all-empty turn (the model
    # requires the blocks to flatten to the content); it falls back to the
    # flattened string, and route admission refuses the whole-empty user turn
    # by name before an Anthropic dispatch.
    hollow = GatewayMessage(
        role="user",
        content="",
        provider_text_blocks=({"type": "text", "text": ""},),
    )
    _role, blocks = anthropic_blocks(hollow)
    assert blocks == [{"type": "text", "text": ""}]

    tool = GatewayMessage(
        role="tool",
        content="",
        tool_call_id="call-1",
        content_parts=(
            TextContentPart(text=""),
            ImageContentPart(media_type="image/png", data="aGk="),
        ),
    )
    _role, blocks = anthropic_blocks(tool)
    run = blocks[0]["content"]
    assert isinstance(run, list)
    assert [block["type"] for block in run if isinstance(block, dict)] == ["image"]

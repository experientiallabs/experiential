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

    # A breakpoint on a dropped TRAILING empty block migrates to the retained
    # neighbor: an empty block adds no bytes, so the cache boundary is
    # byte-identical and the prefix keeps billing cached (the block-cache
    # incident class).
    trailing_marker = GatewayMessage(
        role="user",
        content="real",
        provider_text_blocks=(
            {"type": "text", "text": "real"},
            {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
        ),
    )
    _role, blocks = anthropic_blocks(trailing_marker)
    assert blocks == [{"type": "text", "text": "real", "cache_control": {"type": "ephemeral"}}]

    # A marker on a dropped LEADING empty block lands on the first retained
    # block (a wider, still valid breakpoint), and never overwrites one the
    # retained block already carries.
    leading_marker = GatewayMessage(
        role="user",
        content="real",
        provider_text_blocks=(
            {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "real"},
        ),
    )
    _role, blocks = anthropic_blocks(leading_marker)
    assert blocks == [{"type": "text", "text": "real", "cache_control": {"type": "ephemeral"}}]

    # The same migration applies on the multimodal path, where Claude Code's
    # turn-final marker can land on an empty trailing block.
    multimodal_marker = GatewayMessage(
        role="user",
        content="look",
        content_parts=(
            TextContentPart(text="look"),
            ImageContentPart(media_type="image/png", data="aGk="),
            TextContentPart(text=""),
        ),
        provider_text_blocks=(
            {"type": "text", "text": "look"},
            {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
        ),
    )
    _role, blocks = anthropic_blocks(multimodal_marker)
    assert blocks[0] == {"type": "text", "text": "look", "cache_control": {"type": "ephemeral"}}
    assert [block["type"] for block in blocks] == ["text", "image"]

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


def test_caller_minted_tool_ids_sanitize_deterministically_for_the_anthropic_wire() -> None:
    """Anthropic rejects tool ids outside ``^[a-zA-Z0-9_-]+$`` post-dispatch
    (probed live 2026-09-05, prod rows 2026-09-05 09:48); a caller-minted id
    rewrites deterministically with the SAME mapping on the paired
    tool_result so history stays linked, and conforming ids pass through
    byte-identical."""
    from exp.common.models.model import ToolCall
    from exp.runtime.models.providers.wire_messages import anthropic_tool_use_id

    assert anthropic_tool_use_id("toolu_01AbC-xyz") == "toolu_01AbC-xyz"

    rewritten = anthropic_tool_use_id("call:with:colons")
    assert rewritten.startswith("call_with_colons-")
    assert anthropic_tool_use_id("call:with:colons") == rewritten

    # Distinct originals that clean to the same bytes stay distinct.
    sibling = anthropic_tool_use_id("call.with.colons")
    assert sibling != rewritten

    call = ToolCall(
        call_id="call:with:colons",
        name="get_weather",
        arguments={"city": "Paris"},
        raw_arguments='{"city": "Paris"}',
    )
    _role, blocks = anthropic_blocks(
        GatewayMessage(role="assistant", content=None, tool_calls=(call,))
    )
    tool_use = blocks[-1]
    assert tool_use["id"] == rewritten

    _role, result_blocks = anthropic_blocks(
        GatewayMessage(role="tool", content="sunny", tool_call_id="call:with:colons")
    )
    assert result_blocks[0]["tool_use_id"] == rewritten


def test_overlong_call_ids_bound_deterministically_for_the_responses_wire() -> None:
    """The Responses wire caps call_id at 64 characters (probed live
    2026-09-05: "Invalid 'input[N].call_id': string too long"; one prod
    gpt-6-astra row). A longer caller-minted id maps to a bounded
    prefix+digest with the same mapping on the paired output."""
    from exp.common.models.model import ToolCall
    from exp.runtime.models.providers.wire_messages import responses_call_id, responses_items

    assert responses_call_id("c" * 64) == "c" * 64

    long_id = "x" * 100
    bounded = responses_call_id(long_id)
    assert len(bounded) == 64
    assert bounded.startswith("x" * 55)
    assert responses_call_id(long_id) == bounded
    assert responses_call_id("y" + "x" * 99) != bounded

    call = ToolCall(
        call_id=long_id,
        name="get_weather",
        arguments={"city": "Paris"},
        raw_arguments='{"city": "Paris"}',
    )
    call_items = responses_items(GatewayMessage(role="assistant", content=None, tool_calls=(call,)))
    assert call_items[-1]["call_id"] == bounded
    output_items = responses_items(
        GatewayMessage(role="tool", content="sunny", tool_call_id=long_id)
    )
    assert output_items[0]["call_id"] == bounded

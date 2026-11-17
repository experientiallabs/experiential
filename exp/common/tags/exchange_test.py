"""Tests for the tagging exchange shape, the opening key and free exchange facts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.tags.exchange import (
    TagExchange,
    TagFunction,
    TagImagePart,
    TagMessage,
    TagTextPart,
    TagTool,
    TagToolCall,
    opening_key,
    tag_facts,
)
from exp.common.tags.records import TagFacts

SYSTEM = TagMessage(role="system", content="You are a coding agent.")
FIRST_USER = TagMessage(role="user", content="Fix the date parser.")
LOOKUP = TagTool(name="read_file", description="Read a file.", parameters={"type": "object"})
SHELL = TagTool(name="bash", description="Run a command.", parameters={"type": "object"})


def _call(call_id: str, name: str = "bash", arguments: str = '{"command":"ls"}') -> TagToolCall:
    """Build one tool call.

    Args:
        call_id: Call id.
        name: Tool name.
        arguments: JSON arguments.

    Returns:
        The call.
    """
    return TagToolCall(id=call_id, function=TagFunction(name=name, arguments=arguments))


def _exchange(
    messages: tuple[TagMessage, ...],
    response: tuple[TagMessage, ...] = (TagMessage(role="assistant", content="Done."),),
    tools: tuple[TagTool, ...] = (LOOKUP, SHELL),
) -> TagExchange:
    """Build an exchange.

    Args:
        messages: Request messages.
        response: Response turns.
        tools: Offered tools.

    Returns:
        The exchange.
    """
    return TagExchange(exchange_id="req_1", tools=tools, messages=messages, response=response)


def test_later_turns_share_the_opening_key() -> None:
    """Every turn of one conversation resends its opening and so shares the key."""
    first = _exchange((SYSTEM, FIRST_USER))
    later = _exchange(
        (
            SYSTEM,
            FIRST_USER,
            TagMessage(role="assistant", tool_calls=(_call("c1"),)),
            TagMessage(role="tool", tool_call_id="c1", content="parser.py"),
            TagMessage(role="system", content="Context was compacted."),
            TagMessage(role="user", content="Now run the tests."),
        ),
        tools=(SHELL, LOOKUP),
    )
    assert opening_key(first) == opening_key(later)
    assert len(opening_key(first)) == 64


@pytest.mark.parametrize(
    "changed",
    [
        _exchange((TagMessage(role="system", content="You are a support agent."), FIRST_USER)),
        _exchange((SYSTEM, TagMessage(role="user", content="Write a poem."))),
        _exchange((SYSTEM, FIRST_USER), tools=(LOOKUP,)),
        _exchange(
            (SYSTEM, FIRST_USER),
            tools=(LOOKUP, SHELL.model_copy(update={"parameters": {"type": "string"}})),
        ),
        _exchange(
            (SYSTEM, FIRST_USER),
            tools=(LOOKUP, SHELL.model_copy(update={"description": "Delete everything."})),
        ),
    ],
)
def test_opening_key_changes_with_instructions_tools_or_first_request(
    changed: TagExchange,
) -> None:
    """Different instructions, tools or first requests are different conversations.

    Args:
        changed: An exchange that differs from the base opening in one part.
    """
    assert opening_key(changed) != opening_key(_exchange((SYSTEM, FIRST_USER)))


def test_opening_key_without_a_user_message() -> None:
    """An instruction-only request still has a deterministic key."""
    key = opening_key(_exchange((SYSTEM,)))
    assert key == opening_key(_exchange((SYSTEM,)))
    assert key != opening_key(_exchange((SYSTEM, FIRST_USER)))


def test_facts_count_response_calls_users_and_request_images() -> None:
    """Free facts read counts and flags directly from the exchange."""
    picture = TagMessage(
        role="user",
        content=(TagTextPart(text="What is this?"), TagImagePart(media_type="image/png")),
    )
    exchange = TagExchange(
        exchange_id="req_2",
        messages=(SYSTEM, FIRST_USER, TagMessage(role="assistant", content="Sure."), picture),
        response=(
            TagMessage(role="assistant", content="Looking.", tool_calls=(_call("a"), _call("b"))),
            TagMessage(role="assistant", tool_calls=(_call("c"),)),
        ),
        truncated=True,
        structured_output=True,
    )
    assert tag_facts(exchange) == TagFacts(
        tool_calls=3, turns=2, truncated=True, has_images=True, structured_output=True
    )
    plain = tag_facts(_exchange((SYSTEM, FIRST_USER)))
    assert (plain.tool_calls, plain.turns, plain.has_images) == (0, 1, False)
    assert not plain.truncated and not plain.structured_output


def test_text_marks_images_and_joins_parts() -> None:
    """Multipart text joins in order with an ``[image]`` marker per image."""
    message = TagMessage(
        role="user",
        content=(
            TagTextPart(text="Compare "),
            TagImagePart(url="https://example.com/a.png"),
            TagTextPart(text=" with "),
            TagImagePart(),
        ),
    )
    assert message.text() == "Compare [image] with [image]"
    assert message.has_image()
    assert TagMessage(role="assistant").text() == ""
    assert not FIRST_USER.has_image()


def test_message_fields_must_match_the_role() -> None:
    """Tool calls belong to assistants and call ids to tool results."""
    with pytest.raises(ValidationError, match="only assistant messages"):
        TagMessage(role="user", tool_calls=(_call("x"),))
    with pytest.raises(ValidationError, match="only tool messages"):
        TagMessage(role="assistant", tool_call_id="x")


def test_image_markers_are_spaced_from_adjacent_text() -> None:
    """An image next to text without whitespace still reads as a separate token."""
    message = TagMessage(
        role="user", content=(TagTextPart(text="Weather?"), TagImagePart(), TagTextPart(text="ok"))
    )
    assert message.text() == "Weather? [image] ok"


def test_opening_key_covers_every_message_the_opening_shows() -> None:
    """Messages before the first user message, of any role, are part of the opening."""
    greeting_a = _exchange((TagMessage(role="assistant", content="A"), FIRST_USER))
    greeting_b = _exchange((TagMessage(role="assistant", content="B"), FIRST_USER))
    assert opening_key(greeting_a) != opening_key(greeting_b)


def test_image_parts_never_hold_inline_data() -> None:
    """Only remote URLs are kept; inline image data is rejected."""
    assert TagImagePart(url="https://example.com/a.png").url == "https://example.com/a.png"
    with pytest.raises(ValidationError, match="remote http"):
        TagImagePart(url="data:image/png;base64,AAAA")

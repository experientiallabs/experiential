"""Tests for pydantic-to-protocol error rendering; the decoders' end-to-end
messages are exercised in requests_test.py."""

from __future__ import annotations

import pytest
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from exp.runtime.openai_protocol.validation_errors import (
    cleaned_location,
    unknown_parameter_message,
    validation_protocol_error,
)


class _Closed(BaseModel):
    """A strict wire object, the shape every gateway wire model shares."""

    model_config = ConfigDict(extra="forbid")


class _Function(_Closed):
    """A function declaration with one optional defined key."""

    name: str
    strict: bool = False


class _Tool(_Closed):
    """A tool wrapping one function declaration."""

    function: _Function


class _Tools(_Closed):
    """A body carrying a list of tools."""

    tools: list[_Tool]


class _Search(_Closed):
    """A search action branch."""

    query: str


class _Fetch(_Closed):
    """A fetch action branch that defines ``limit``."""

    url: str
    limit: int


class _Action(_Closed):
    """A body whose ``action`` is an undiscriminated union."""

    action: _Search | _Fetch


class _Gateway(_Closed):
    """A gateway extension object with one defined key."""

    retry: bool = False


class _Body(_Closed):
    """A body carrying the gateway extension object."""

    gateway: _Gateway


def _rendered(model: type[BaseModel], payload: object) -> tuple[str | None, str]:
    """Return the (param, message) one validation failure renders as."""
    with pytest.raises(ValidationError) as caught:
        model.model_validate(payload)
    detail = validation_protocol_error(caught.value).detail
    return detail.param, detail.message


def test_cleaned_location_drops_branch_labels_but_keeps_a_same_named_final_field() -> None:
    """A union-branch label is never the last segment; a caller field may be."""
    assert cleaned_location(("input", 1, "reasoning", "id")) == ("input", "1", "id")
    assert cleaned_location(("messages", 0, "reasoning")) == ("messages", "0", "reasoning")
    assert cleaned_location(("body", "input", "tuple[...]", 0, "_ResponseMessage", "role")) == (
        "input",
        "0",
        "role",
    )
    assert cleaned_location(("messages", 0, "content", 1, "image_url", "image_url", "url")) == (
        "messages",
        "0",
        "content",
        "1",
        "image_url",
        "url",
    )


def test_an_undefined_key_is_named_unknown_and_unsupported_not_an_invalid_value() -> None:
    """Agno's ``requires_confirmation`` on a function tool is named unknown, not a bad value."""
    param, message = _rendered(
        _Tools, {"tools": [{"function": {"name": "f", "requires_confirmation": True}}]}
    )
    assert param == "tools.0.function.requires_confirmation"
    assert message == (
        "Unknown parameter 'tools.0.function.requires_confirmation': "
        "this field is not supported. Remove it and resend."
    )
    assert message == unknown_parameter_message(param)


def test_a_defined_key_with_a_bad_value_keeps_its_value_message() -> None:
    """Only an undefined key is "unknown"; a defined one still names its expected shape."""
    param, message = _rendered(_Tools, {"tools": [{"function": {"name": "f", "strict": "maybe"}}]})
    assert param == "tools.0.function.strict"
    assert message.startswith("Invalid value for 'tools.0.function.strict'")


def test_a_key_some_union_branch_defines_is_never_called_unknown() -> None:
    """A sibling branch defining the key with a value fault means the caller may have meant it.

    ``limit`` is extra on the search branch but defined (wrong type) on the
    fetch branch, so "unknown parameter" would misdescribe the request.
    """
    param, message = _rendered(_Action, {"action": {"query": "q", "limit": "ten"}})
    assert param == "action.limit"
    assert message.startswith("Invalid value for 'action.limit'")
    # A key NO branch defines is still named unknown inside the union.
    assert _rendered(_Action, {"action": {"query": "q", "bogus": 1}}) == (
        "action.bogus",
        unknown_parameter_message("action.bogus"),
    )


def test_an_undefined_gateway_key_keeps_the_unknown_parameter_wording() -> None:
    """A stray key on the gateway extension object uses the same unknown-parameter wording."""
    param, message = _rendered(_Body, {"gateway": {"retries": 3}})
    assert param == "gateway.retries"
    assert message == unknown_parameter_message("gateway.retries")


@pytest.mark.parametrize("key", ["_meta", "Requires", "function", "str"])
def test_an_undefined_key_is_named_even_when_it_spells_like_a_branch_label(key: str) -> None:
    """The caller's own key is named, never its parent object (which is defined)."""
    param, message = _rendered(_Tools, {"tools": [{"function": {"name": "f", key: 1}}]})
    assert param == f"tools.0.function.{key}"
    assert message == unknown_parameter_message(f"tools.0.function.{key}")


def test_a_key_a_sibling_branch_accepts_is_never_called_unknown() -> None:
    """``limit`` is valid on the fetch branch, which failed only on its other fields."""
    param, message = _rendered(_Action, {"action": {"query": "q", "limit": 3}})
    assert param == "action.limit"
    assert not message.startswith("Unknown parameter")


@pytest.mark.parametrize("key", ["A" * 513, "x" * 3_000])
def test_an_over_long_undefined_key_still_renders_a_bounded_refusal(key: str) -> None:
    """A caller key longer than the error model allows is abbreviated, never a build failure."""
    param, message = _rendered(_Tools, {"tools": [{"function": {"name": "f", key: 1}}]})
    assert param is not None
    assert len(param) == 256
    assert param.startswith("tools.0.function.")
    assert param.endswith("...")
    assert message == unknown_parameter_message(param)


def test_an_open_string_vocabulary_names_the_string_not_its_members() -> None:
    """``str | Literal[...]`` (the SDK's ``model``) admits any string: no member list is shown."""
    adapter = TypeAdapter[object](CompletionCreateParamsNonStreaming)
    with pytest.raises(ValidationError) as raised:
        adapter.validate_python({"model": {"value": "coding"}, "messages": []})
    error = validation_protocol_error(raised.value)
    assert error.detail.param == "model"
    assert error.detail.message == (
        "Invalid value for 'model': expected a string, but got an object instead."
    )

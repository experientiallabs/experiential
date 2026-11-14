"""Tests for reading OpenAI-nullable top-level nulls as absent."""

from __future__ import annotations

import types
import typing

import pytest
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.nullable_params import (
    CHAT_NULL_AS_ABSENT,
    CHAT_NULLABLE_PARAMS,
    CHAT_SERVED_NULL_PARAMS,
    RESPONSES_NULL_AS_ABSENT,
    RESPONSES_NULLABLE_PARAMS,
    RESPONSES_SERVED_NULL_PARAMS,
    drop_null_defaults,
)


def _admits_null(annotation: object) -> bool:
    """Whether one generated TypedDict annotation admits ``None``."""
    origin = typing.get_origin(annotation)
    if origin in {typing.Required, typing.NotRequired, typing.Annotated}:
        return _admits_null(typing.get_args(annotation)[0])
    if origin in {typing.Union, types.UnionType}:
        return any(_admits_null(argument) for argument in typing.get_args(annotation))
    return annotation is type(None)


def _sdk_nullable(params: type) -> frozenset[str]:
    """The top-level fields an official SDK request TypedDict declares ``Optional``."""
    hints = typing.get_type_hints(params, include_extras=True)
    return frozenset(name for name, annotation in hints.items() if _admits_null(annotation))


@pytest.mark.parametrize(
    ("pinned", "params"),
    [
        (CHAT_NULLABLE_PARAMS, CompletionCreateParamsNonStreaming),
        (RESPONSES_NULLABLE_PARAMS, ResponseCreateParamsNonStreaming),
    ],
)
def test_nullable_vocabularies_match_the_installed_official_sdk(
    pinned: frozenset[str], params: type
) -> None:
    """The pinned literals are exactly OpenAI's nullable fields; a pin bump moving them fails."""
    assert pinned == _sdk_nullable(params)


@pytest.mark.parametrize(
    ("served", "params"),
    [
        (CHAT_SERVED_NULL_PARAMS, CompletionCreateParamsNonStreaming),
        (RESPONSES_SERVED_NULL_PARAMS, ResponseCreateParamsNonStreaming),
    ],
)
def test_served_null_vocabularies_are_optional_sdk_fields_beyond_its_nullable_ones(
    served: frozenset[str], params: type
) -> None:
    """Each served-null field is an optional SDK key the SDK does not already mark nullable."""
    assert served <= frozenset(vars(params)["__optional_keys__"])
    assert not served & _sdk_nullable(params)


def test_required_fields_are_never_dropped() -> None:
    """``model`` and ``messages``/``input`` keep their named refusal on a null."""
    for field in ("model", "messages"):
        assert field not in CHAT_NULL_AS_ABSENT
    for field in ("model", "input"):
        assert field not in RESPONSES_NULL_AS_ABSENT


def test_drop_null_defaults_removes_only_nullable_nulls() -> None:
    """A null on a nullable field is removed; a value, or a null elsewhere, is kept as sent."""
    payload: JsonObject = {
        "model": None,
        "stream": None,
        "temperature": 0.5,
        "metadata": None,
        "user": None,
    }
    assert drop_null_defaults(payload, CHAT_NULLABLE_PARAMS) == {
        "model": None,
        "temperature": 0.5,
        "user": None,
    }


def test_drop_null_defaults_returns_the_same_payload_when_nothing_is_null() -> None:
    """No copy is made when there is nothing to remove."""
    payload: JsonObject = {"model": "coding", "stream": False, "user": None}
    assert drop_null_defaults(payload, CHAT_NULLABLE_PARAMS) is payload

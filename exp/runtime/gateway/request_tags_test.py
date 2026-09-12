"""Typed tag constraints and attribution-only request identity."""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from exp.common.core.artifacts import sha256_json
from exp.runtime.gateway.native_decode import NativeDecodeError, decode_native_body
from exp.runtime.gateway.replay_identity import canonical_request_sha256, provider_replay_authority
from exp.runtime.gateway.request_tags import RequestTags

_TAGS = TypeAdapter(RequestTags)


@pytest.mark.parametrize(
    "tags",
    [
        {"team": ""},
        {"team": 1},
        {"team": True},
        {"team": None},
        {"team": {}},
        {"team": []},
        {"1team": "v"},
        {"_team": "v"},
        {"é": "v"},
        {"a b": "v"},
        {"a\n": "v"},
        {"k" * 65: "v"},
        {"team": "x" * 257},
        {"team": "\x00"},
        {"team": "\x85"},
        {"team": "\ud800"},
        {"explabs.team": "v"},
        {"ExPlAbS.team": "v"},
        {f"k{i}": "v" for i in range(17)},
        [],
        "bad",
    ],
)
def test_invalid_typed_tags(tags: object) -> None:
    """Typed boundaries refuse malformed or reserved attribution, never coerce it."""
    with pytest.raises(ValidationError):
        _TAGS.validate_python(tags)


def test_valid_exact_limits_and_key_case() -> None:
    """Character counts are Unicode scalar counts, keys otherwise stay exact."""
    tags = {f"k{i}": "v" for i in range(13)}
    tags.update({"K": "v", "cost-center.prod": "café", "k" * 64: "é" * 256})
    assert _TAGS.validate_python(tags) == tags
    assert _TAGS.validate_python({}) == {}


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_decoded_tags_are_not_provider_input(surface: str) -> None:
    """Every surface retains validated tags without serializing them to providers."""
    payload = {"model": "coding", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    if surface == "responses":
        payload = {"model": "coding", "input": "hi"}
    plain = decode_native_body(json.dumps(payload), surface=surface).request
    tagged = decode_native_body(
        json.dumps(payload), surface=surface, request_tags={"team": "a", "env": "prod"}
    ).request
    reordered = tagged.model_copy(update={"request_tags": {"env": "prod", "team": "a"}})
    changed = tagged.model_copy(update={"request_tags": {"team": "b", "env": "prod"}})
    assert tagged.request_tags == {"team": "a", "env": "prod"}
    assert tagged.model_dump(mode="json") == plain.model_dump(mode="json")
    assert provider_replay_authority(tagged) == provider_replay_authority(plain)
    assert canonical_request_sha256(plain) == sha256_json(plain)
    assert canonical_request_sha256(tagged) == canonical_request_sha256(reordered)
    assert canonical_request_sha256(tagged) != canonical_request_sha256(changed)
    assert canonical_request_sha256(tagged) != canonical_request_sha256(plain)


def test_native_decode_revalidates_tags() -> None:
    """A malformed map at the native bridge still returns a safe named 400."""
    with pytest.raises(NativeDecodeError) as raised:
        decode_native_body(
            '{"model":"coding","messages":[{"role":"user","content":"hi"}]}',
            request_tags={"explabs.team": "do-not-echo-value"},
        )
    assert raised.value.error.status_code == 400
    assert raised.value.error.detail.param == "X-Explabs-Tags"
    assert "do-not-echo-value" not in str(raised.value)

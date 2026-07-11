"""Tests for the Bedrock adapter (boto3 stubbed — no AWS calls)."""

from __future__ import annotations

import io
import json
import sys
import threading
import types
from typing import TYPE_CHECKING, Any, cast

import pytest

from llm_waterfall.adapters.bedrock import BedrockAdapter
from llm_waterfall.types import Backend, ChatRequest, Message

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FakeClient:
    """Records invoke_model calls and returns canned Anthropic/Titan responses."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.converse_stop_reason = "end_turn"

    def invoke_model(self, *, modelId: str, body: str) -> dict[str, Any]:  # noqa: N803
        parsed = json.loads(body)
        self.calls.append({"modelId": modelId, "body": parsed})
        if "inputText" in parsed:  # Titan embedding request
            payload = {"embedding": [0.1, 0.2], "inputTextTokenCount": 3}
        else:
            payload = {
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        return {"body": io.BytesIO(json.dumps(payload).encode())}

    def converse(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "output": {"message": {"role": "assistant", "content": []}},
            "stopReason": self.converse_stop_reason,
            "usage": {"inputTokens": 5, "outputTokens": 0},
        }


class _FakeSession:
    def __init__(self, **kwargs: str | None) -> None:
        self.kwargs = kwargs
        self.client_args: dict[str, Any] = {}

    def client(self, service: str, **kwargs: Any) -> _FakeClient:
        self.client_args = {"service": service, **kwargs}
        fake = _FakeClient()
        self.made_client = fake
        return fake


@pytest.fixture
def fake_boto3(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_FakeSession]]:
    sessions: list[_FakeSession] = []
    boto3 = types.ModuleType("boto3")

    def session_factory(**kwargs: str | None) -> _FakeSession:
        s = _FakeSession(**kwargs)
        sessions.append(s)
        return s

    setattr(boto3, "Session", session_factory)  # noqa: B010 - fake module attr

    botocore = types.ModuleType("botocore")
    botocore_config = types.ModuleType("botocore.config")

    class Config:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    setattr(botocore_config, "Config", Config)  # noqa: B010 - fake module attr
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)
    yield sessions


def test_named_profile_and_region_flow_into_session(fake_boto3: list[_FakeSession]) -> None:
    backend = Backend(
        "bedrock", "us.anthropic.claude-opus-4-6-v1", profile="endflow", region="us-west-1"
    )
    adapter = BedrockAdapter(backend)
    adapter.complete("sys", [Message(role="user", content="hi")], temperature=None, max_tokens=64)
    (session,) = fake_boto3
    assert session.kwargs == {"profile_name": "endflow", "region_name": "us-west-1"}
    assert session.client_args["service"] == "bedrock-runtime"


def test_sdk_retries_disabled_and_timeouts_bound(fake_boto3: list[_FakeSession]) -> None:
    adapter = BedrockAdapter(Backend("bedrock", "m", region="us-west-2"))
    adapter.complete("", [Message(role="user", content="hi")], temperature=None, max_tokens=64)
    (session,) = fake_boto3
    config = session.client_args["config"]
    # total_max_attempts counts the initial request; 1 == zero botocore retries.
    assert config.kwargs["retries"] == {"total_max_attempts": 1}
    assert config.kwargs["connect_timeout"] == 15.0
    assert config.kwargs["read_timeout"] == 600.0


def test_request_body_shape(fake_boto3: list[_FakeSession]) -> None:
    adapter = BedrockAdapter(Backend("bedrock", "model-x"))
    text, usage = adapter.complete(
        "be terse", [Message(role="user", content="hi")], temperature=None, max_tokens=99
    )
    call = fake_boto3[0].made_client.calls[0]
    body = call["body"]
    assert call["modelId"] == "model-x"
    assert body["system"] == "be terse"
    assert body["max_tokens"] == 99
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in body  # None → not forwarded
    assert text == "hello"
    assert usage.input_tokens == 10 and usage.output_tokens == 5


def test_titan_embed_loop_and_dimensions(fake_boto3: list[_FakeSession]) -> None:
    adapter = BedrockAdapter(Backend("bedrock", "m", embed_dim=512))
    vectors, usage = adapter.embed(["a", "b"])
    calls = fake_boto3[0].made_client.calls
    assert len(calls) == 2  # one invoke per text
    assert calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"
    assert calls[0]["body"] == {"inputText": "a", "dimensions": 512, "normalize": True}
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert usage.input_tokens == 6


def test_client_built_once_under_concurrency(fake_boto3: list[_FakeSession]) -> None:
    adapter = BedrockAdapter(Backend("bedrock", "m"))
    barrier = threading.Barrier(8)

    def hit() -> None:
        barrier.wait()
        adapter.complete("", [Message(role="user", content="x")], temperature=None, max_tokens=8)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(fake_boto3) == 1  # exactly one Session/client constructed


@pytest.mark.parametrize("stop_reason", ["content_filtered", "guardrail_intervened"])
def test_structured_chat_preserves_filtered_stops(
    fake_boto3: list[_FakeSession], stop_reason: str
) -> None:
    """Blocked Bedrock waterfall turns retain the OpenAI safety signal."""
    adapter = BedrockAdapter(Backend("bedrock", "model-x"))
    client = cast("_FakeClient", adapter._get_client())  # noqa: SLF001 - configure SDK fake
    client.converse_stop_reason = stop_reason

    response = adapter.complete_chat(
        ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})
    )

    assert response.choices[0].finish_reason == "content_filter"

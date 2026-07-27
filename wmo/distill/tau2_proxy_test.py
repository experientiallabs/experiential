"""Tests for the per-episode OpenAI-compatible proxy."""

from __future__ import annotations

import json
import urllib.request

import pytest
from llm_waterfall.types import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatUsage,
)

from wmo.distill.tau2_proxy import EpisodeProxy


class _FakeProvider:
    """A ToolCallingProvider stand-in that echoes what it was asked."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="hello"),
                    finish_reason="stop",
                )
            ],
            usage=ChatUsage(prompt_tokens=11, completion_tokens=3),
            model="tinker://x",
        )


class _ExplodingProvider:
    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        raise RuntimeError("sampler wedged")


def _post(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.fixture
def proxy() -> EpisodeProxy:
    instance = EpisodeProxy()
    instance.start()
    yield instance
    instance.stop()


class TestRegistry:
    def test_duplicate_alias_is_rejected(self) -> None:
        instance = EpisodeProxy()
        instance.register("ep-1", _FakeProvider())
        with pytest.raises(ValueError, match="already registered"):
            instance.register("ep-1", _FakeProvider())

    def test_release_is_idempotent(self) -> None:
        instance = EpisodeProxy()
        instance.register("ep-1", _FakeProvider())
        instance.release("ep-1")
        instance.release("ep-1")

    def test_empty_alias_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="nonempty"):
            EpisodeProxy().register("", _FakeProvider())


class TestServing:
    def test_round_trip_openai_shape(self, proxy: EpisodeProxy) -> None:
        provider = _FakeProvider()
        proxy.register("ep-a", provider)
        status, body = _post(
            f"{proxy.base_url}/chat/completions",
            {
                "model": "ep-a",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 1.0,
                "max_tokens": 64,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_user", "parameters": {"type": "object"}},
                    }
                ],
            },
        )
        assert status == 200
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "hello"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["total_tokens"] == 14
        # The provider saw the structured request, tools included.
        [request] = provider.requests
        assert request.tools is not None and request.tools[0].function.name == "get_user"
        assert request.max_tokens == 64

    def test_unknown_alias_is_404(self, proxy: EpisodeProxy) -> None:
        status, body = _post(
            f"{proxy.base_url}/chat/completions",
            {"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 404
        assert "unknown episode alias" in body["error"]["message"]

    def test_provider_failure_is_502(self, proxy: EpisodeProxy) -> None:
        proxy.register("ep-b", _ExplodingProvider())
        status, body = _post(
            f"{proxy.base_url}/chat/completions",
            {"model": "ep-b", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 502
        assert "sampler wedged" in body["error"]["message"]

    def test_released_alias_stops_serving(self, proxy: EpisodeProxy) -> None:
        proxy.register("ep-c", _FakeProvider())
        proxy.release("ep-c")
        status, _ = _post(
            f"{proxy.base_url}/chat/completions",
            {"model": "ep-c", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 404


class TestLifecycle:
    def test_base_url_before_start_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before start"):
            _ = EpisodeProxy().base_url

    def test_double_start_raises(self, proxy: EpisodeProxy) -> None:
        with pytest.raises(RuntimeError, match="twice"):
            proxy.start()

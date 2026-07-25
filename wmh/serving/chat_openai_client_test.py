"""OpenAI compatibility proven with the REAL `openai` client library, not our own requests.

`chat_test.py` exercises the route with FastAPI's TestClient, which only proves we can parse
our own output. These tests boot an actual uvicorn server and drive it with the official
`openai` SDK - the same code path a customer's integration uses - covering non-streaming
response parsing, streaming chunk framing with `stream_options.include_usage`, and error body
shapes (the SDK must raise its typed exception AND surface our message from
`body["error"]["message"]`). The upstream provider is a deterministic in-process fake, so the
tests prove wire compatibility without network or keys.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import cast

import openai
import pytest
import uvicorn
from fastapi import FastAPI

from wmh.optimize.policy import RoutingPolicy
from wmh.providers.base import (
    Completion,
    Message,
    ProviderKind,
    StreamChunk,
    TokenUsage,
    VerifyResult,
)
from wmh.providers.pool import PoolEntry
from wmh.serving.chat import (
    EndpointRuntime,
    RequestLog,
    create_chat_router,
    install_openai_error_shapes,
)


class _FakeProvider:
    """Deterministic upstream: echoes a fixed reply, reports cache-split usage."""

    def __init__(self, entry: PoolEntry) -> None:
        self.config = entry.provider_config()
        self.name = entry.name

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Completion:
        return Completion(
            text=f"served by {self.name}",
            usage=TokenUsage(input_tokens=10, output_tokens=5, cached_input_tokens=4),
        )

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[StreamChunk]:
        yield StreamChunk(delta="served ")
        yield StreamChunk(delta="by ")
        yield StreamChunk(delta=self.name)
        yield StreamChunk(
            done=True,
            usage=TokenUsage(input_tokens=10, output_tokens=5, cached_input_tokens=4),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError


@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A real uvicorn server on an ephemeral port, serving one static endpoint."""
    pool = [PoolEntry(name="haiku-4-5", kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5")]
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=pool),
        provider_factory=_FakeProvider,
        log=RequestLog(tmp_path_factory.mktemp("serving") / "requests.jsonl"),
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime}))
    install_openai_error_shapes(app)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("uvicorn thread died during startup")
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start within 10s")
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=5)


def _client(base_url: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=base_url, api_key="test-key", max_retries=0)


def test_real_client_parses_completion(live_server: str) -> None:
    completion = _client(live_server).chat.completions.create(
        model="tau-bench",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert completion.choices[0].message.content == "served by haiku-4-5"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.model == "tau-bench"  # endpoint name, never the routed pool model
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 10
    assert completion.usage.completion_tokens == 5
    details = completion.usage.prompt_tokens_details
    assert details is not None and details.cached_tokens == 4


def test_real_client_streams_with_usage_chunk(live_server: str) -> None:
    stream = _client(live_server).chat.completions.create(
        model="tau-bench",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    parts: list[str] = []
    finish_reasons: list[str] = []
    usage = None
    for chunk in stream:
        if chunk.usage is not None:
            # OpenAI framing: the usage chunk carries NO choices.
            assert chunk.choices == []
            usage = chunk.usage
        for choice in chunk.choices:
            if choice.delta.content:
                parts.append(choice.delta.content)
            if choice.finish_reason:
                finish_reasons.append(choice.finish_reason)
    assert "".join(parts) == "served by haiku-4-5"
    assert finish_reasons == ["stop"]
    assert usage is not None and usage.prompt_tokens == 10 and usage.completion_tokens == 5


def test_real_client_stream_without_usage_opt_in_gets_none(live_server: str) -> None:
    stream = _client(live_server).chat.completions.create(
        model="tau-bench",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    assert all(chunk.usage is None for chunk in stream)


def test_real_client_gets_typed_error_with_message(live_server: str) -> None:
    with pytest.raises(openai.NotFoundError) as excinfo:
        _client(live_server).chat.completions.create(
            model="no-such-endpoint",
            messages=[{"role": "user", "content": "hello"}],
        )
    assert isinstance(excinfo.value.body, dict)
    body = cast("dict[str, str]", excinfo.value.body)
    assert body["code"] == "model_not_found"
    assert "no endpoint 'no-such-endpoint'" in body["message"]
    assert "tau-bench" in body["message"]  # the error names what IS available


def test_real_client_validation_error_is_openai_shaped_400(live_server: str) -> None:
    with pytest.raises(openai.BadRequestError) as excinfo:
        _client(live_server).chat.completions.create(
            model="tau-bench",
            messages=[],  # violates min_length=1
        )
    assert isinstance(excinfo.value.body, dict)
    body = cast("dict[str, str]", excinfo.value.body)
    assert body["code"] == "invalid_request"
    assert "messages" in body["message"]


def test_real_client_tools_rejected_not_silently_dropped(live_server: str) -> None:
    with pytest.raises(openai.BadRequestError) as excinfo:
        _client(live_server).chat.completions.create(
            model="tau-bench",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        )
    body = cast("dict[str, str]", excinfo.value.body)
    assert body["code"] == "unsupported_parameter"
    assert "tools" in body["message"]


def test_real_client_developer_role_and_content_parts(live_server: str) -> None:
    completion = _client(live_server).chat.completions.create(
        model="tau-bench",
        messages=[
            {"role": "developer", "content": "be terse"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
    )
    assert completion.choices[0].message.content == "served by haiku-4-5"

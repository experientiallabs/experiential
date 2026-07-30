"""OpenAI compatibility proven with the REAL `openai` client library, not our own requests.

`chat_test.py` exercises the route with FastAPI's TestClient, which only proves we can parse
our own output. These tests boot an actual uvicorn server and drive it with the official
`openai` SDK - the same code path a customer's integration uses - covering non-streaming
response parsing, streaming chunk framing with `stream_options.include_usage`, tool calling
(including the SDK's own replay idiom, where the assistant message it parsed goes straight back
into `messages`), and error body shapes (the SDK must raise its typed exception AND surface our
message from `body["error"]["message"]`). The upstream provider is a deterministic in-process
fake, so the tests prove wire compatibility without network or keys.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import cast

import openai
import pytest
import uvicorn
from fastapi import FastAPI
from llm_waterfall.types import (
    ChatChoice,
    ChatFunctionCall,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatToolCall,
    ChatUsage,
)
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

from wmo.optimize.policy import RoutingPolicy
from wmo.providers.base import (
    Completion,
    Message,
    ProviderKind,
    StreamChunk,
    TokenUsage,
    VerifyResult,
)
from wmo.providers.pool import PoolEntry
from wmo.serving.chat import (
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


_TOOL_ARGUMENTS = '{"table": "superheroes", "limit": 10}'

_TOOLS: list[ChatCompletionToolUnionParam] = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "read rows from a table",
            "parameters": {
                "type": "object",
                "properties": {"table": {"type": "string"}, "limit": {"type": "integer"}},
            },
        },
    }
]


class _ToolFakeProvider(_FakeProvider):
    """Deterministic structured upstream: calls `lookup`, then answers from the tool result."""

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        results = [m for m in request.messages if m.role == "tool"]
        if results:
            return ChatResponse(
                choices=[
                    ChatChoice(
                        message=ChatMessage(
                            role="assistant", content=f"{self.name} read {results[-1].content}"
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=ChatUsage(prompt_tokens=13, completion_tokens=9),
            )
        return ChatResponse(
            choices=[
                ChatChoice(
                    message=ChatMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatToolCall(
                                id="call_1",
                                function=ChatFunctionCall(name="lookup", arguments=_TOOL_ARGUMENTS),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=ChatUsage(prompt_tokens=11, completion_tokens=7),
        )


@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A real uvicorn server on an ephemeral port, serving two static endpoints.

    `tau-bench`'s pool model is text-only (its injected fake has no `complete_chat`,
    the shape of a provider that never grew the structured seam); `tools-bench`'s speaks
    the structured contract. Both are needed to prove the two
    halves of the tool surface: a real round trip, and the honest error when the routed model
    cannot make one.
    """
    pool = [PoolEntry(name="haiku-4-5", kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5")]
    log = RequestLog(tmp_path_factory.mktemp("serving") / "requests.jsonl")
    runtime = EndpointRuntime(
        name="tau-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=pool),
        provider_factory=_FakeProvider,
        log=log,
    )
    tool_runtime = EndpointRuntime(
        name="tools-bench",
        policy=RoutingPolicy(kind="static", default_model="haiku-4-5", pool=pool),
        provider_factory=_ToolFakeProvider,
        log=log,
    )
    app = FastAPI()
    app.include_router(create_chat_router({"tau-bench": runtime, "tools-bench": tool_runtime}))
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


def test_real_client_round_trips_a_tool_call(live_server: str) -> None:
    """The SDK's own agent loop: parse tool_calls, replay the message it parsed, get the answer.

    Appending `completion.choices[0].message` verbatim is the documented idiom, and it sends
    fields our schema never declared (`refusal`, `annotations`, a null `content`), so this also
    proves the endpoint tolerates them instead of 400ing a client's own output.
    """
    client = _client(live_server)
    completion = client.chat.completions.create(
        model="tools-bench",
        messages=[{"role": "user", "content": "how many superheroes are there?"}],
        tools=_TOOLS,
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content is None
    assert choice.message.tool_calls is not None
    call = choice.message.tool_calls[0]
    assert call.id == "call_1"
    assert call.type == "function"
    assert call.function.name == "lookup"
    assert json.loads(call.function.arguments) == {"table": "superheroes", "limit": 10}

    followup = client.chat.completions.create(
        model="tools-bench",
        messages=[
            {"role": "user", "content": "how many superheroes are there?"},
            cast("ChatCompletionMessageParam", choice.message),
            {"role": "tool", "tool_call_id": call.id, "content": "42 rows"},
        ],
        tools=_TOOLS,
    )
    assert followup.choices[0].message.content == "haiku-4-5 read 42 rows"
    assert followup.choices[0].finish_reason == "stop"
    assert followup.usage is not None and followup.usage.completion_tokens == 9


def test_real_client_reassembles_streamed_tool_call_arguments(live_server: str) -> None:
    stream = _client(live_server).chat.completions.create(
        model="tools-bench",
        messages=[{"role": "user", "content": "how many superheroes are there?"}],
        tools=_TOOLS,
        stream=True,
        stream_options={"include_usage": True},
    )
    arguments: dict[int, str] = {}
    names: dict[int, str] = {}
    ids: dict[int, str] = {}
    finish_reasons: list[str] = []
    usage = None
    for chunk in stream:
        if chunk.usage is not None:
            assert chunk.choices == []
            usage = chunk.usage
        for choice in chunk.choices:
            for call in choice.delta.tool_calls or []:
                if call.id is not None:
                    ids[call.index] = call.id
                if call.function is not None and call.function.name is not None:
                    names[call.index] = call.function.name
                if call.function is not None and call.function.arguments:
                    arguments[call.index] = arguments.get(call.index, "") + call.function.arguments
            if choice.finish_reason:
                finish_reasons.append(choice.finish_reason)
    assert ids == {0: "call_1"} and names == {0: "lookup"}
    assert json.loads(arguments[0]) == {"table": "superheroes", "limit": 10}
    assert finish_reasons == ["tool_calls"]
    assert usage is not None and usage.completion_tokens == 7


def test_real_client_tools_on_a_text_only_pool_model_are_not_silently_dropped(
    live_server: str,
) -> None:
    # `tau-bench`'s pool model has no structured backend. Answering in prose would look like a
    # weak model, so the endpoint fails with the pool entry named (501, not a 400: the request is
    # valid, the routed model cannot serve it).
    with pytest.raises(openai.InternalServerError) as excinfo:
        _client(live_server).chat.completions.create(
            model="tau-bench",
            messages=[{"role": "user", "content": "hi"}],
            tools=_TOOLS,
        )
    assert excinfo.value.status_code == 501
    body = cast("dict[str, str]", excinfo.value.body)
    assert body["code"] == "tool_calling_unsupported"
    assert "haiku-4-5" in body["message"]
    assert "cannot serve tool calls" in body["message"]


def test_real_client_developer_role_and_content_parts(live_server: str) -> None:
    completion = _client(live_server).chat.completions.create(
        model="tau-bench",
        messages=[
            {"role": "developer", "content": "be terse"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
    )
    assert completion.choices[0].message.content == "served by haiku-4-5"

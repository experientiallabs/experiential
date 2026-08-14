"""Official OpenAI SDK routing endpoint tests."""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)
from openai.types.responses import Response, ResponseStreamEvent
from openai.types.responses.function_tool_param import FunctionToolParam

from wmo.common.models import (
    AssistantAction,
    ModelFinishReason,
    ModelRequest,
    ModelResponse,
    OperationEconomics,
)
from wmo.runtime.router.application import create_project_router_app
from wmo.runtime.router.completion import RouterCompletionConflictError
from wmo.runtime.router.endpoint import (
    HttpResponseRequest,
    _chat_completion,
    _openai_response,
)
from wmo.runtime.router.runtime import RoutedModelResponse, RouterRuntime
from wmo.runtime.router.runtime_test import _Client, _runtime, _snapshot


class _ReplayService:
    """Thread-safe test double for the durable journal completion contract."""

    def __init__(self, runtime: RouterRuntime) -> None:
        """Initialize the replay service around a router runtime."""
        self.runtime = runtime
        self._lock = threading.Lock()
        self._completed: dict[str, tuple[ModelRequest, RoutedModelResponse]] = {}

    def complete(
        self,
        request: ModelRequest,
        *,
        idempotency_key: str,
        conversation_id: str | None = None,
    ) -> RoutedModelResponse:
        """Complete a request once per key and replay the durable result.

        Args:
            request: Provider-neutral request to route.
            idempotency_key: Caller key that owns the replay entry.
            conversation_id: Optional stable conversation identity.

        Returns:
            Newly completed or previously retained routed response.

        Raises:
            RouterCompletionConflictError: The key is reused for another request.
        """
        if not idempotency_key:
            return self.runtime.complete(request, episode_id=conversation_id)
        with self._lock:
            existing = self._completed.get(idempotency_key)
            if existing is not None:
                if existing[0] != request:
                    raise RouterCompletionConflictError("key reused for another request")
                return existing[1]
            result = self.runtime.complete(request, episode_id=conversation_id or idempotency_key)
            self._completed[idempotency_key] = (request, result)
            return result


def _clients(
    *, candidate_tools: bool = True, durable: bool = False
) -> tuple[OpenAI, TestClient, RouterRuntime, _Client]:
    """Build official and loopback clients over one test router runtime.

    Args:
        candidate_tools: Whether the routed candidate declares tool support.
        durable: Whether to inject the replaying durable completion service.

    Returns:
        Official OpenAI client, loopback HTTP client, router runtime, and model client.
    """
    runtime, model_client = _runtime(candidate_tools=candidate_tools)
    service = _ReplayService(runtime) if durable else None
    app = create_project_router_app("router-a", runtime, completion_service=service)
    http = TestClient(app)
    openai = OpenAI(api_key="local-test", base_url="http://testserver/v1", http_client=http)
    return openai, http, runtime, model_client


def test_official_chat_client_preserves_tools_without_cross_caller_affinity() -> None:
    """Official SDK chat calls need no WMO header and do not join equal transcripts."""
    openai, _http, _runtime_value, model_client = _clients()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-in",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"a"}'},
                }
            ],
        },
        {"role": "tool", "content": "result", "tool_call_id": "call-in"},
    ]
    tools: list[ChatCompletionToolUnionParam] = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    completion = openai.chat.completions.create(model="router-a", messages=messages, tools=tools)

    assert isinstance(completion, ChatCompletion)
    assert completion.model == "router-a"
    assert completion.choices[0].message.tool_calls is not None
    assert completion.choices[0].message.tool_calls[0].id == "call-out"
    assert completion.model_extra is None or "routing_decision" not in completion.model_extra
    captured = model_client.requests[-1]
    assert [message.role for message in captured.messages] == ["user", "assistant", "tool"]
    assert captured.messages[1].assistant_action is not None
    assert captured.messages[1].assistant_action.tool_calls[0].call_id == "call-in"
    assert captured.messages[2].tool_call_id == "call-in"

    next_turn = openai.chat.completions.create(
        model="router-a",
        messages=[
            *messages,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-out",
                        "type": "function",
                        "function": {"name": "write", "arguments": '{"x":1}'},
                    }
                ],
            },
            {"role": "tool", "content": "done", "tool_call_id": "call-out"},
            {"role": "user", "content": "next turn"},
        ],
        tools=tools,
    )

    assert next_turn.model == "router-a"
    assert model_client.embed_calls == 2
    assert len(model_client.requests[-1].messages) == 6

    openai.chat.completions.create(model="router-a", messages=messages, tools=tools)
    assert model_client.embed_calls == 3


@pytest.mark.parametrize("path", ("/v1/chat/completions", "/v1/responses"))
def test_tool_request_rejects_an_incapable_selected_model(path: str) -> None:
    """OpenAI tool requests fail explicitly before an incapable provider call."""
    _openai, http, _runtime_value, model_client = _clients(candidate_tools=False)
    payload = (
        {
            "model": "router-a",
            "input": "read",
            "tools": [
                {
                    "type": "function",
                    "name": "read",
                    "parameters": {"type": "object"},
                }
            ],
        }
        if path.endswith("responses")
        else {
            "model": "router-a",
            "messages": [{"role": "user", "content": "read"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )

    response = http.post(path, json=payload)

    assert response.status_code == 501
    assert response.json()["error"]["code"] == "tool_calling_unsupported"
    assert model_client.complete_calls == 0


def test_official_responses_client_continues_with_previous_response_id() -> None:
    """Responses continuation keeps routing affinity through the standard response ID."""
    openai, _http, _runtime_value, model_client = _clients()

    first = openai.responses.create(model="router-a", input="Help me")
    second = openai.responses.create(
        model="router-a", input="Continue", previous_response_id=first.id
    )

    assert isinstance(first, Response)
    assert isinstance(second, Response)
    assert second.previous_response_id == first.id
    assert model_client.embed_calls == 1
    assert [message.role for message in model_client.requests[-1].messages] == [
        "user",
        "assistant",
        "user",
    ]


def test_responses_continuation_does_not_carry_prior_instructions() -> None:
    """Prove Responses instructions remain scoped to their provider call.

    A continued request retains visible conversation content while excluding the earlier
    request-scoped instruction from the next provider transcript.
    """
    openai, _http, _runtime_value, model_client = _clients()

    first = openai.responses.create(model="router-a", input="one", instructions="FIRST")
    openai.responses.create(
        model="router-a",
        input="two",
        instructions="SECOND",
        previous_response_id=first.id,
    )

    assert [(item.role, item.content) for item in model_client.requests[-1].messages] == [
        ("user", "one"),
        ("assistant", None),
        ("system", "SECOND"),
        ("user", "two"),
    ]


def test_openai_text_content_parts_are_preserved() -> None:
    """Chat and Responses text parts reach the routed model as complete visible text."""
    openai, _http, _runtime_value, model_client = _clients()

    openai.chat.completions.create(
        model="router-a",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first "},
                    {"type": "text", "text": "second"},
                ],
            }
        ],
    )
    openai.responses.create(
        model="router-a",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "third "},
                    {"type": "input_text", "text": "fourth"},
                ],
            }
        ],
    )

    assert model_client.requests[-2].messages[0].content == "first second"
    assert model_client.requests[-1].messages[0].content == "third fourth"


def test_request_validation_uses_an_openai_error_envelope() -> None:
    """Public schema failures do not leak FastAPI's proprietary detail body."""
    _openai, http, _runtime_value, model_client = _clients()

    response = http.post(
        "/v1/chat/completions",
        json={"model": "router-a", "messages": [{"role": "invalid", "content": "x"}]},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Invalid OpenAI request",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_request",
        }
    }
    assert model_client.embed_calls == 0
    assert model_client.complete_calls == 0


def test_official_clients_parse_buffered_streams() -> None:
    """Both official SDK streaming iterators parse WMO's OpenAI SSE events."""
    openai, _http, _runtime_value, _model_client = _clients()

    chat_stream = openai.chat.completions.create(
        model="router-a", messages=[{"role": "user", "content": "stream"}], stream=True
    )
    chat_chunks: list[ChatCompletionChunk] = list(chat_stream)
    response_stream = openai.responses.create(model="router-a", input="stream", stream=True)
    response_events: list[ResponseStreamEvent] = list(response_stream)

    assert chat_chunks
    assert all(isinstance(item, ChatCompletionChunk) for item in chat_chunks)
    assert chat_chunks[-1].choices[0].finish_reason == "tool_calls"
    assert response_events
    assert response_events[0].type == "response.created"
    assert response_events[-1].type == "response.completed"


def test_responses_tool_stream_emits_the_official_item_and_argument_events() -> None:
    """Prove tool streams expose the complete official Responses lifecycle.

    The official SDK observes item creation, argument delta and completion, item completion, and
    the final response event in protocol order.
    """
    openai, _http, _runtime_value, _model_client = _clients()

    events = list(
        openai.responses.create(
            model="router-a",
            input="write",
            tools=[
                FunctionToolParam(
                    type="function",
                    name="write",
                    parameters={"type": "object"},
                    strict=None,
                )
            ],
            stream=True,
        )
    )

    assert [event.type for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]


def test_nested_unsupported_official_fields_fail_before_dispatch() -> None:
    """Prove unsupported official nested fields fail before provider dispatch.

    Strict tool schemas and serial tool-call requirements are rejected at validation, and neither
    request reaches the model client.
    """
    _openai, http, _runtime_value, model_client = _clients()

    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "router-a",
            "messages": [{"role": "user", "content": "read", "name": "customer"}],
        },
    )
    strict = http.post(
        "/v1/responses",
        json={
            "model": "router-a",
            "input": "read",
            "tools": [{"type": "function", "name": "read", "parameters": {}, "strict": True}],
        },
    )

    assert response.status_code == strict.status_code == 400
    assert model_client.embed_calls == model_client.complete_calls == 0


def test_length_and_responses_request_metadata_are_preserved() -> None:
    """Prove truncation and supported request metadata survive public adaptation.

    The response preserves instructions, tool settings, temperature, token limits, usage, and the
    incomplete terminal state produced by length termination.
    """
    model_response = ModelResponse(
        output=AssistantAction(content="partial"),
        model=_snapshot("cheap"),
        economics=OperationEconomics(),
        finish_reason=ModelFinishReason.LENGTH,
    )
    request = HttpResponseRequest.model_validate(
        {
            "model": "router-a",
            "input": "write",
            "instructions": "be brief",
            "temperature": 0.2,
            "max_output_tokens": 10,
            "parallel_tool_calls": True,
            "tool_choice": {"type": "function", "name": "write"},
            "tools": [
                {
                    "type": "function",
                    "name": "write",
                    "parameters": {"type": "object"},
                    "strict": None,
                }
            ],
        }
    )

    chat = _chat_completion("router-a", model_response.output, model_response)
    response = _openai_response(
        "router-a",
        model_response.output,
        model_response,
        request=request,
        idempotency_key=None,
        previous_response_id=None,
    )

    assert chat.choices[0].finish_reason == "length"
    assert response.status == "incomplete"
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"
    assert response.instructions == "be brief"
    assert response.temperature == 0.2
    assert response.max_output_tokens == 10
    assert response.parallel_tool_calls is True
    assert not isinstance(response.tool_choice, str)
    assert response.tool_choice.type == "function"
    assert response.tools[0].type == "function"


def test_idempotency_key_fails_closed_without_a_durable_service() -> None:
    """Prove a keyed request fails closed without a durable completion service.

    The endpoint rejects the claim before router selection or provider dispatch can create state
    that cannot be replayed durably.
    """
    _openai, http, runtime, model_client = _clients()

    response = http.post(
        "/v1/chat/completions",
        json={"model": "router-a", "messages": [{"role": "user", "content": "retry"}]},
        headers={"Idempotency-Key": "interaction-a"},
    )

    assert response.status_code == 409
    assert model_client.embed_calls == model_client.complete_calls == 0
    assert not runtime._request_decisions  # noqa: SLF001


@pytest.mark.parametrize("path", ("/v1/chat/completions", "/v1/responses"))
def test_empty_idempotency_key_is_rejected_before_dispatch(path: str) -> None:
    """Prove an empty standard idempotency key never becomes an unkeyed call.

    Args:
        path: OpenAI endpoint path exercised by the parameterized regression.
    """
    _openai, http, runtime, model_client = _clients(durable=True)
    payload = (
        {"model": "router-a", "input": "retry"}
        if path.endswith("responses")
        else {"model": "router-a", "messages": [{"role": "user", "content": "retry"}]}
    )

    response = http.post(path, json=payload, headers={"Idempotency-Key": " "})

    assert response.status_code == 400
    assert model_client.embed_calls == model_client.complete_calls == 0
    assert not runtime._request_decisions  # noqa: SLF001


def test_durable_service_replays_the_same_public_completion() -> None:
    """Prove durable Chat Completion replay avoids dispatch and envelope drift.

    Two calls with the same key return identical public JSON while producing only one model client
    completion.
    """
    _openai, http, _runtime_value, model_client = _clients(durable=True)
    payload = {"model": "router-a", "messages": [{"role": "user", "content": "same"}]}
    headers = {"Idempotency-Key": "interaction-a"}

    first = http.post("/v1/chat/completions", json=payload, headers=headers)
    second = http.post("/v1/chat/completions", json=payload, headers=headers)

    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert first.json()["id"].startswith("chatcmpl-")
    assert model_client.embed_calls == model_client.complete_calls == 1


def test_durable_service_replays_the_same_public_response() -> None:
    """Prove durable Responses replay preserves nested public identities.

    Reusing a caller key returns identical response and child-item IDs without another provider
    invocation.
    """
    _openai, http, _runtime_value, model_client = _clients(durable=True)
    payload = {"model": "router-a", "input": "same"}
    headers = {"Idempotency-Key": "interaction-response"}

    first = http.post("/v1/responses", json=payload, headers=headers)
    second = http.post("/v1/responses", json=payload, headers=headers)

    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert model_client.embed_calls == model_client.complete_calls == 1


@pytest.mark.parametrize("path", ("/v1/chat/completions", "/v1/responses"))
def test_idempotency_key_rejects_a_different_request(path: str) -> None:
    """Prove one idempotency key cannot merge divergent OpenAI requests.

    Args:
        path: OpenAI endpoint path exercised by the parameterized conflict regression.
    """
    _openai, http, _runtime_value, model_client = _clients(durable=True)
    headers = {"Idempotency-Key": "one-logical-request"}
    if path.endswith("responses"):
        first = {"model": "router-a", "input": "first"}
        changed = {"model": "router-a", "input": "changed"}
    else:
        first = {"model": "router-a", "messages": [{"role": "user", "content": "first"}]}
        changed = {
            "model": "router-a",
            "messages": [{"role": "user", "content": "changed"}],
        }

    assert http.post(path, json=first, headers=headers).status_code == 200
    conflict = http.post(path, json=changed, headers=headers)

    assert conflict.status_code == 409
    assert (
        conflict.json()["error"]["message"]
        == "Idempotency-Key conflicts with durable request state"
    )
    assert model_client.embed_calls == 1
    assert model_client.complete_calls == 1


@pytest.mark.parametrize(
    "path,payload",
    [
        (
            "/v1/chat/completions",
            {
                "model": "router-a",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "x",
                                "function": {"name": "read", "arguments": "[1]"},
                            }
                        ],
                    }
                ],
            },
        ),
        (
            "/v1/chat/completions",
            {
                "model": "router-a",
                "messages": [{"role": "system", "content": "no user"}],
            },
        ),
        (
            "/v1/responses",
            {"model": "router-a", "input": "next", "previous_response_id": "missing"},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "router-a",
                "messages": [{"role": "user", "content": "unsupported"}],
                "logprobs": True,
            },
        ),
        (
            "/v1/responses",
            {"model": "router-a", "input": "unsupported", "background": True},
        ),
    ],
)
def test_invalid_official_request_never_reaches_provider(
    path: str, payload: dict[str, object]
) -> None:
    """Malformed OpenAI requests fail as 4xx before embedding or model dispatch."""
    _openai, http, _runtime_value, model_client = _clients()

    response = http.post(path, json=payload)

    assert response.status_code in {400, 422}
    assert model_client.embed_calls == 0
    assert model_client.complete_calls == 0

"""Tests for the per-episode OpenAI-compatible proxy."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest
from llm_waterfall.types import (
    ChatChoice,
    ChatFunctionCall,
    ChatFunctionDefinition,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatTool,
    ChatToolCall,
    ChatUsage,
)

from wmo.distill.tau2_proxy import EpisodeProxy, realign_tool_argument_types


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
def proxy() -> Iterator[EpisodeProxy]:
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
        # The class name crosses the wire; the detail stays in the collector log
        # (CodeQL: stack-trace exposure), mirroring wmo.serving.chat's split.
        assert "RuntimeError" in body["error"]["message"]
        assert "sampler wedged" not in body["error"]["message"]

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


class TestToolArgumentRealignment:
    """The cookbook XML parser JSON-decodes parameter values schema-blind; the
    proxy must re-align them with the declared schema (measured live: retail's
    all-numeric string ids came out as ints and every DB lookup failed)."""

    @staticmethod
    def _arguments(response: ChatResponse) -> dict:
        calls = response.choices[0].message.tool_calls
        assert calls is not None
        return json.loads(calls[0].function.arguments)

    @staticmethod
    def _response(arguments: str, name: str = "find_user_id_by_name_zip") -> ChatResponse:
        return ChatResponse(
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatToolCall(
                                id="call-1",
                                function=ChatFunctionCall(name=name, arguments=arguments),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ]
        )

    @staticmethod
    def _tools(properties: dict) -> list[ChatTool]:
        return [
            ChatTool(
                function=ChatFunctionDefinition(
                    name="find_user_id_by_name_zip",
                    parameters={"type": "object", "properties": properties},
                )
            )
        ]

    def test_the_live_failure_shape_is_fixed(self) -> None:
        # The exact arguments observed in the failing teacher episode.
        response = self._response('{"first_name": "Yusuf", "last_name": "Rossi", "zip": 19122}')
        tools = self._tools(
            {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "zip": {"type": "string"},
            }
        )
        realign_tool_argument_types(response, tools)
        arguments = self._arguments(response)
        assert arguments == {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}

    def test_reverse_direction_and_booleans(self) -> None:
        response = self._response('{"quantity": "5", "ratio": "3.5", "confirm": "true"}')
        tools = self._tools(
            {
                "quantity": {"type": "integer"},
                "ratio": {"type": "number"},
                "confirm": {"type": "boolean"},
            }
        )
        realign_tool_argument_types(response, tools)
        arguments = self._arguments(response)
        assert arguments == {"quantity": 5, "ratio": 3.5, "confirm": True}

    def test_lossy_or_unknown_conversions_are_left_alone(self) -> None:
        response = self._response(
            '{"zip": "007", "note": "hello", "count": "not-a-number", "flag": true}'
        )
        tools = self._tools(
            {
                "zip": {"type": "integer"},  # "007" -> 7 would lose the leading zeros
                "note": {"type": "string"},
                "count": {"type": "integer"},
                "flag": {"type": "string"},  # bools are never stringified
            }
        )
        realign_tool_argument_types(response, tools)
        arguments = self._arguments(response)
        assert arguments == {"zip": "007", "note": "hello", "count": "not-a-number", "flag": True}

    def test_unknown_tool_and_absent_tools_are_untouched(self) -> None:
        response = self._response('{"zip": 19122}', name="unknown_tool")
        realign_tool_argument_types(response, self._tools({"zip": {"type": "string"}}))
        assert self._arguments(response) == {"zip": 19122}
        response2 = self._response('{"zip": 19122}')
        realign_tool_argument_types(response2, None)
        assert self._arguments(response2) == {"zip": 19122}

    def test_array_items_align_per_element(self) -> None:
        # Retail's reward-bearing write tools take List[str] of all-numeric item
        # ids; a model emitting them unquoted must be repaired element-wise.
        response = self._response('{"item_ids": [9612497925, 8124970213], "note": "x"}')
        tools = self._tools(
            {
                "item_ids": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string"},
            }
        )
        realign_tool_argument_types(response, tools)
        assert self._arguments(response) == {
            "item_ids": ["9612497925", "8124970213"],
            "note": "x",
        }

    def test_anyof_optional_string_aligns(self) -> None:
        # Optional[str] renders as anyOf [string, null]; the single non-null
        # branch is the declared type.
        response = self._response('{"origin": 90210, "leave_after": null}')
        tools = self._tools(
            {
                "origin": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "leave_after": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            }
        )
        realign_tool_argument_types(response, tools)
        assert self._arguments(response) == {"origin": "90210", "leave_after": None}

    def test_untyped_array_items_are_left_alone(self) -> None:
        response = self._response('{"values": [1, "two"]}')
        tools = self._tools({"values": {"type": "array"}})
        realign_tool_argument_types(response, tools)
        assert self._arguments(response) == {"values": [1, "two"]}

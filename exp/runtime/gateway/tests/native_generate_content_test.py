"""End-to-end Gemini generateContent tests against the served native engine.

The official ``google-genai`` client points its base URL at the shared native
serving subprocess from ``native_messages_test`` (an OpenAI-compatible SSE mock
behind the ``coding`` alias, or a Gemini mock for the ``gemini-budget`` variant)
and drives ``/v1beta/models/{model}:generateContent``,
``:streamGenerateContent?alt=sse``, ``:countTokens`` and the model listing
through the real Rust data plane and shared python control plane.
"""

from __future__ import annotations

import httpx
import pytest
from google import genai
from google.genai import errors, types

from exp.runtime.gateway.tests.native_messages_test import (
    _completed_attempts,
    _engine,  # noqa: F401 - module fixture `engine`.
    _ServingEngine,
    _SseUpstream,
)

pytest.importorskip("exp_gateway_native")


def _client(engine: _ServingEngine, api_key: str | None = None) -> genai.Client:
    """The official SDK pointed at the gateway with an ``xpl_``-style virtual key."""
    return genai.Client(
        api_key=api_key or engine.raw_key,
        http_options=types.HttpOptions(
            base_url=engine.base,
            api_version="v1beta",
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def test_non_streaming_generate_content_answers_googles_shape_and_accounts(
    engine: _ServingEngine,
) -> None:
    """One SDK call returns a GenerateContentResponse and settles one attempt."""
    client = _client(engine)
    before = _completed_attempts(engine.base)
    response = client.models.generate_content(
        model="coding",
        contents="fast-token",
        config=types.GenerateContentConfig(system_instruction="be terse", max_output_tokens=64),
    )
    assert response.text == "hello world"
    assert response.candidates is not None
    assert response.candidates[0].finish_reason == types.FinishReason.STOP
    usage = response.usage_metadata
    assert usage is not None
    # The OpenAI mock reports 9 prompt tokens with 2 cached and 4 completion tokens.
    assert (usage.prompt_token_count, usage.candidates_token_count, usage.total_token_count) == (
        9,
        4,
        13,
    )
    assert usage.cached_content_token_count == 2
    assert response.model_version == "coding"
    assert _completed_attempts(engine.base) == before + 1


def test_streaming_generate_content_streams_sse_chunks_with_a_terminal_usage(
    engine: _ServingEngine,
) -> None:
    """The SDK's stream iterator sees text chunks and a final chunk with finish and usage."""
    client = _client(engine)
    chunks = list(client.models.generate_content_stream(model="coding", contents="fast-token"))
    text = "".join(chunk.text or "" for chunk in chunks)
    assert text == "hello world"
    last = chunks[-1]
    assert last.candidates is not None
    assert last.candidates[0].finish_reason == types.FinishReason.STOP
    assert last.usage_metadata is not None and last.usage_metadata.total_token_count == 13
    assert all(chunk.response_id == chunks[0].response_id for chunk in chunks)


def test_tool_calls_render_as_function_call_parts(engine: _ServingEngine) -> None:
    """A tool turn answers a functionCall part the SDK exposes as function_calls."""
    client = _client(engine)
    response = client.models.generate_content(
        model="coding",
        contents="tool-token",
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="search",
                            description="look up",
                            parameters_json_schema={"type": "object"},
                        )
                    ]
                )
            ]
        ),
    )
    calls = response.function_calls
    assert calls is not None and len(calls) == 1
    assert (calls[0].name, calls[0].args, calls[0].id) == ("search", {"q": "x"}, "call-1")


def test_count_tokens_and_model_listing_answer_googles_shapes(engine: _ServingEngine) -> None:
    """countTokens is a read (no attempt); the listing names exactly the granted aliases."""
    client = _client(engine)
    before = _completed_attempts(engine.base)
    counted = client.models.count_tokens(model="coding", contents="hello there")
    assert counted.total_tokens is not None and counted.total_tokens > 0
    assert _completed_attempts(engine.base) == before
    assert [model.name for model in client.models.list()] == ["models/coding"]
    detail = client.models.get(model="coding")
    assert detail.name == "models/coding"
    assert detail.supported_actions == ["generateContent", "streamGenerateContent", "countTokens"]
    with pytest.raises(errors.ClientError) as missing:
        client.models.get(model="not-granted")
    assert missing.value.code == 404
    # A colon in a GET segment belongs to the alias (`coding:free`), never to an action.
    free = httpx.get(
        f"{engine.base}/v1beta/models/coding:free",
        headers={"x-goog-api-key": engine.raw_key},
        timeout=10.0,
    )
    assert free.status_code == 404
    assert "Unknown model action" not in free.json()["error"]["message"]
    assert free.json()["error"]["details"][0]["reason"] != "unknown_route"


def test_auth_and_protocol_failures_wear_googles_error_envelope(engine: _ServingEngine) -> None:
    """Bad keys, unknown fields and raw REST conventions answer Google-shaped errors."""
    bad_client = _client(engine, api_key="exp_vk_invalid")
    with pytest.raises(errors.ClientError) as bad_key:
        bad_client.models.generate_content(model="coding", contents="fast-token")
    assert (bad_key.value.code, bad_key.value.status) == (401, "UNAUTHENTICATED")
    unknown_field = httpx.post(
        f"{engine.base}/v1beta/models/coding:generateContent",
        params={"key": engine.raw_key},
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "unknown_field": 3},
        timeout=10.0,
    )
    assert unknown_field.status_code == 400
    error = unknown_field.json()["error"]
    assert error["status"] == "INVALID_ARGUMENT"
    assert error["details"][0]["reason"] == "unsupported_parameter"
    assert error["details"][0]["metadata"]["param"] == "unknown_field"
    no_sse = httpx.post(
        f"{engine.base}/v1beta/models/coding:streamGenerateContent",
        headers={"x-goog-api-key": engine.raw_key},
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        timeout=10.0,
    )
    assert no_sse.status_code == 400
    assert "alt=sse" in no_sse.json()["error"]["message"]
    unknown_action = httpx.post(
        f"{engine.base}/v1beta/models/coding:embedContent",
        headers={"x-goog-api-key": engine.raw_key},
        json={},
        timeout=10.0,
    )
    assert unknown_action.status_code == 404
    assert unknown_action.json()["error"]["status"] == "NOT_FOUND"


def test_query_key_auth_serves_a_raw_rest_request(engine: _ServingEngine) -> None:
    """Google's ``?key=`` convention authenticates without any header."""
    response = httpx.post(
        f"{engine.base}/v1beta/models/coding:generateContent",
        params={"key": engine.raw_key},
        json={"contents": [{"role": "user", "parts": [{"text": "fast-token"}]}]},
        timeout=30.0,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidates"][0]["content"]["parts"] == [{"text": "hello world"}]
    assert response.headers["x-gateway-alias"] == "coding"


@pytest.mark.parametrize("engine", ["gemini-budget"], indirect=True)
def test_a_gemini_lane_receives_the_native_generate_content_payload(
    engine: _ServingEngine,
) -> None:
    """On a Gemini lane the request round-trips Google's own wire with its thinking usage."""
    with _SseUpstream.payloads_lock:
        _SseUpstream.payloads.clear()
    client = _client(engine)
    response = client.models.generate_content(
        model="coding",
        contents="hi",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            max_output_tokens=32,
        ),
    )
    assert response.text == "hello world"
    usage = response.usage_metadata
    assert usage is not None
    # The Gemini mock reports 9 prompt, 4 candidate and 2 thought tokens.
    assert (usage.prompt_token_count, usage.candidates_token_count, usage.thoughts_token_count) == (
        9,
        4,
        2,
    )
    assert usage.total_token_count == 15
    with _SseUpstream.payloads_lock:
        dispatched = _SseUpstream.payloads[-1]
    assert dispatched["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    generation = dispatched["generationConfig"]
    assert isinstance(generation, dict)
    assert generation["maxOutputTokens"] == 32
    assert "thinkingConfig" in generation

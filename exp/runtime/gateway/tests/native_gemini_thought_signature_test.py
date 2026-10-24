"""Gemini thought signatures round-trip through the served gateway across turns.

A fake Gemini upstream answers the first turn with two parallel function
calls, of which Google signs only the first. The caller then sends the next
turn the way real clients do (echoing the assistant message or output items,
or naming ``previous_response_id``), and the upstream must receive the real
signature back on the first call and none on the second.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import httpx
import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.runtime.gateway.catalog_authority import upsert_connection, upsert_singleton_deployment
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests.launch_test import _ServedGateway, _unused_port
from exp.runtime.gateway.tests.native_json_object_test import _frame
from exp.runtime.models import registry
from exp.runtime.models.providers.gemini_requests import GEMINI_THOUGHT_SIGNATURE_BYPASS

_SIGNATURE = "Q2lJQkFqczFYMmp0aGVyZWFsc2lnbmF0dXJl"
"""A base64 stand-in for the opaque signature Gemini issues."""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for one city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]

_CALLS: JsonObject = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                            "id": "call_paris",
                        },
                        "thoughtSignature": _SIGNATURE,
                    },
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Tokyo"},
                            "id": "call_tokyo",
                        }
                    },
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 6},
}

_ANSWER: JsonObject = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "Sunny in both."}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 4},
}


@dataclass
class _Upstream:
    """A served fake Gemini endpoint and the request bodies it received."""

    bodies: list[JsonObject]
    server: ThreadingHTTPServer


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, str, _Upstream]]:
    """Serve one Gemini alias against a fake upstream: calls first, then an answer."""
    bodies: list[JsonObject] = []

    class Provider(BaseHTTPRequestHandler):
        """Answer the first request with signed calls and later ones with text."""

        def do_POST(self) -> None:  # noqa: N802
            """Record the native request body and stream the next canned turn."""
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            bodies.append(body)
            chunk = _frame(_CALLS if len(bodies) == 1 else _ANSWER)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Suppress synthetic content from logs."""
            del format, args

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    factory, _ = registry._HTTP_PROVIDERS["gemini"]
    monkeypatch.setattr(
        registry,
        "_HTTP_PROVIDERS",
        {
            **registry._HTTP_PROVIDERS,
            "gemini": (factory, f"http://127.0.0.1:{upstream.server_port}/v1"),
        },
    )
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-key")
    key = _configure(tmp_path)
    gateway = _ServedGateway(tmp_path, _unused_port())
    gateway.start()
    try:
        yield f"http://127.0.0.1:{gateway.port}/v1", key, _Upstream(bodies, upstream)
    finally:
        gateway.stop()
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)


def _configure(root: Path) -> str:
    """Seed one tool-capable Gemini deployment and return a granted key."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider",
        connection=ConnectionConfig(provider="gemini", api_key_env="TEST_PROVIDER_KEY"),
        replace=False,
    )
    normalized, snapshot, _ = upsert_singleton_deployment(
        root,
        deployment_alias="gemini",
        connection_name="provider",
        provider_model="gemini-test",
        exact_model_id="gemini-test",
        revision=None,
        capabilities=ModelCapabilities(supports_tools=True),
        gateway_capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_streaming_tool_arguments=True,
            supports_parallel_tool_calls=True,
        ),
        prices=GatewayTokenPrices(
            input_nano_usd_per_million_tokens=1_000_000_000,
            output_nano_usd_per_million_tokens=2_000_000_000,
        ),
        pricing_source="synthetic-test-prices",
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="gemini",
        alias_name="gemini",
        revision_id="rev",
        pool_id="gemini",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="gemini")
    return manager.issue_key(identity_id="default", key_id="key").raw_key


def _post(base: str, key: str, route: str, body: JsonObject) -> httpx.Response:
    """POST one bounded public request and require success."""
    limit = "max_tokens" if route == "chat/completions" else "max_output_tokens"
    body = {limit: 64, **body}
    response = httpx.post(
        f"{base}/{route}", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=10
    )
    assert response.status_code == 200, response.text
    return response


def _stream_chat_message(response: httpx.Response) -> JsonObject:
    """Fold Chat stream deltas into one assistant message, as OpenAI SDKs do."""
    calls: dict[int, JsonObject] = {}
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    for line in response.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for choice in json.loads(line[6:]).get("choices", []):
            for delta in choice["delta"].get("tool_calls", []):
                index = delta["index"]
                call = calls.setdefault(index, {})
                function = delta.get("function", {})
                names[index] = function.get("name") or names.get(index, "")
                arguments[index] = arguments.get(index, "") + function.get("arguments", "")
                call.update({k: v for k, v in delta.items() if k not in {"index", "function"}})
    tool_calls: list[JsonValue] = [
        {**call, "function": {"name": names[index], "arguments": arguments[index]}}
        for index, call in sorted(calls.items())
    ]
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def _replayed_model_parts(upstream: _Upstream) -> list[JsonValue]:
    """The model turn's parts the second upstream request replayed."""
    assert len(upstream.bodies) == 2
    contents = upstream.bodies[1]["contents"]
    assert isinstance(contents, list)
    model_turns = [turn for turn in contents if isinstance(turn, dict) and turn["role"] == "model"]
    assert len(model_turns) == 1
    parts = model_turns[0]["parts"]
    assert isinstance(parts, list)
    return parts


def _signatures(parts: list[JsonValue]) -> list[JsonValue]:
    """Each replayed functionCall part's signature, or None when it has none."""
    return [
        part.get("thoughtSignature")
        for part in parts
        if isinstance(part, dict) and "functionCall" in part
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_chat_callers_echoing_the_message_return_the_real_signature(
    served: tuple[str, str, _Upstream], stream: bool
) -> None:
    """The signature reaches the caller on the first call and returns to Gemini."""
    base, key, upstream = served
    turn: list[JsonObject] = [{"role": "user", "content": "Weather in Paris and Tokyo?"}]
    first = _post(
        base,
        key,
        "chat/completions",
        {"model": "gemini", "messages": turn, "tools": _TOOLS, "stream": stream},
    )
    message = _stream_chat_message(first) if stream else first.json()["choices"][0]["message"]
    calls = cast("list[JsonObject]", message["tool_calls"])
    assert [call["id"] for call in calls] == ["call_paris", "call_tokyo"]
    assert calls[0]["extra_content"] == {"google": {"thought_signature": _SIGNATURE}}
    assert "extra_content" not in calls[1]
    history = [
        *turn,
        message,
        *({"role": "tool", "tool_call_id": call["id"], "content": "sunny"} for call in calls),
    ]
    second = _post(
        base, key, "chat/completions", {"model": "gemini", "messages": history, "tools": _TOOLS}
    )
    assert second.json()["choices"][0]["message"]["content"] == "Sunny in both."
    assert _signatures(_replayed_model_parts(upstream)) == [_SIGNATURE, None]


def test_a_chat_caller_echoing_the_signature_directly_on_the_call_returns_it(
    served: tuple[str, str, _Upstream],
) -> None:
    """``thought_signature`` on the call (some Gemini bridges' spelling) replays the same way.

    The response is served with no ignored-parameters disclosure: the Gemini
    rung reads the signature, so nothing the caller sent was dropped.
    """
    base, key, upstream = served
    turn: list[JsonObject] = [{"role": "user", "content": "Weather in Paris and Tokyo?"}]
    first = _post(
        base, key, "chat/completions", {"model": "gemini", "messages": turn, "tools": _TOOLS}
    )
    calls = first.json()["choices"][0]["message"]["tool_calls"]
    flattened = []
    for call in calls:
        flat = {key: value for key, value in call.items() if key != "extra_content"}
        if "extra_content" in call:
            flat["thought_signature"] = call["extra_content"]["google"]["thought_signature"]
        flattened.append(flat)
    history = [
        *turn,
        {"role": "assistant", "content": None, "tool_calls": flattened},
        *({"role": "tool", "tool_call_id": call["id"], "content": "sunny"} for call in calls),
    ]
    second = _post(
        base, key, "chat/completions", {"model": "gemini", "messages": history, "tools": _TOOLS}
    )
    assert "x-experiential-ignored-parameters" not in second.json()
    assert _signatures(_replayed_model_parts(upstream)) == [_SIGNATURE, None]


def test_a_chat_caller_that_drops_the_signature_falls_back_to_the_placeholder(
    served: tuple[str, str, _Upstream],
) -> None:
    """History without signatures still continues, on Google's documented placeholder."""
    base, key, upstream = served
    turn: list[JsonObject] = [{"role": "user", "content": "Weather in Paris and Tokyo?"}]
    first = _post(
        base, key, "chat/completions", {"model": "gemini", "messages": turn, "tools": _TOOLS}
    )
    calls = first.json()["choices"][0]["message"]["tool_calls"]
    stripped = [
        {key: value for key, value in call.items() if key != "extra_content"} for call in calls
    ]
    history = [
        *turn,
        {"role": "assistant", "content": None, "tool_calls": stripped},
        *({"role": "tool", "tool_call_id": call["id"], "content": "sunny"} for call in calls),
    ]
    _post(base, key, "chat/completions", {"model": "gemini", "messages": history, "tools": _TOOLS})
    assert _signatures(_replayed_model_parts(upstream)) == [GEMINI_THOUGHT_SIGNATURE_BYPASS] * 2


_RESPONSES_TOOLS = [{"type": "function", **_TOOLS[0]["function"]}]


@pytest.mark.parametrize("stream", [False, True])
def test_responses_callers_echoing_output_items_return_the_real_signature(
    served: tuple[str, str, _Upstream], stream: bool
) -> None:
    """Echoed ``function_call`` items carry the signature back to Gemini."""
    base, key, upstream = served
    turn: list[JsonObject] = [{"role": "user", "content": "Weather in Paris and Tokyo?"}]
    first = _post(
        base,
        key,
        "responses",
        {
            "model": "gemini",
            "input": turn,
            "tools": _RESPONSES_TOOLS,
            "stream": stream,
            "store": False,
        },
    )
    if stream:
        output = [
            json.loads(line[6:])["item"]
            for line in first.text.splitlines()
            if line.startswith("data: ") and '"response.output_item.done"' in line
        ]
    else:
        output = first.json()["output"]
    calls = [item for item in output if item["type"] == "function_call"]
    assert calls[0]["extra_content"] == {"google": {"thought_signature": _SIGNATURE}}
    assert "extra_content" not in calls[1]
    _post(
        base,
        key,
        "responses",
        {
            "model": "gemini",
            "input": [
                *turn,
                *output,
                *(
                    {"type": "function_call_output", "call_id": call["call_id"], "output": "sunny"}
                    for call in calls
                ),
            ],
            "tools": _RESPONSES_TOOLS,
            "store": False,
        },
    )
    assert _signatures(_replayed_model_parts(upstream)) == [_SIGNATURE, None]


def test_a_previous_response_id_continuation_replays_the_retained_signature(
    served: tuple[str, str, _Upstream],
) -> None:
    """The gateway's own retained turn carries the signature with its call."""
    base, key, upstream = served
    first = _post(
        base,
        key,
        "responses",
        {"model": "gemini", "input": "Weather in Paris and Tokyo?", "tools": _RESPONSES_TOOLS},
    ).json()
    calls = [item for item in first["output"] if item["type"] == "function_call"]
    _post(
        base,
        key,
        "responses",
        {
            "model": "gemini",
            "previous_response_id": first["id"],
            "input": [
                {"type": "function_call_output", "call_id": call["call_id"], "output": "sunny"}
                for call in calls
            ],
            "tools": _RESPONSES_TOOLS,
        },
    )
    assert _signatures(_replayed_model_parts(upstream)) == [_SIGNATURE, None]

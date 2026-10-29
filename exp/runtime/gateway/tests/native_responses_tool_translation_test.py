"""Real root-only serving regression, also runnable against the released source fixture."""

from __future__ import annotations

import inspect
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import cast

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayDeploymentCapabilities
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.tests import native_waterfall_test as waterfall
from exp.runtime.gateway.tests.native_waterfall_test import _ServingEngine, _sse_frame


class _ToolUpstream(BaseHTTPRequestHandler):
    """Stream translated functions, without live credentials or external calls."""

    calls = 0
    stopped = threading.Event()
    release = threading.Event()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Emit fragmented custom input and an unchanged ordinary function."""
        type(self).calls += 1
        payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
        prompt = payload["messages"][-1]["content"]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        if any(tool["function"]["name"] == "apply_patch_3" for tool in payload["tools"]):
            self._collision_stream(payload)
            return
        names = ["apply_patch", "agents__close", "plain"]
        assert {tool["function"]["name"] for tool in payload["tools"]} == set(names)
        arguments = (
            '{"input":7}' if prompt == "malformed" else json.dumps({"input": 'patch\n"café 😀"'}),
            '{"id":"a"}',
            '{"n":1}',
        )
        try:
            if prompt in {"text-cancel", "text-gated", "text-success"}:
                self.wfile.write(waterfall._content_chunk("visible-text"))
                self.wfile.flush()
                if prompt == "text-gated":
                    assert type(self).release.wait(5), "Test did not release gated provider"
                if prompt == "text-cancel":
                    while True:
                        self.wfile.write(waterfall._content_chunk("more"))
                        self.wfile.flush()
                        time.sleep(0.02)
                self.wfile.write(waterfall._terminal_frames())
                return
            for index, name in enumerate(names):
                self.wfile.write(
                    _sse_frame(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": index,
                                                "id": f"call-{index}",
                                                "type": "function",
                                                "function": {"name": name, "arguments": ""},
                                            }
                                        ]
                                    },
                                }
                            ]
                        }
                    )
                )
            self.wfile.flush()
            if prompt == "provider-error":
                self.wfile.write(
                    _sse_frame(
                        {
                            "error": {
                                "message": "fixture provider failed",
                                "type": "server_error",
                                "code": "server_error",
                            }
                        }
                    )
                )
                return
            if prompt in {"cancel", "always-500"}:
                fragment = '{"input":"'
                while True:
                    self.wfile.write(
                        _sse_frame(
                            {
                                "choices": [
                                    {
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": 0,
                                                    "function": {"arguments": fragment},
                                                }
                                            ]
                                        }
                                    }
                                ]
                            }
                        )
                    )
                    self.wfile.flush()
                    fragment = "x" * 512
                    time.sleep(0.02)
            for offset in range(max(map(len, arguments))):
                for index, argument in enumerate(arguments):
                    if offset < len(argument):
                        self.wfile.write(
                            _sse_frame(
                                {
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "index": index,
                                                        "function": {"arguments": argument[offset]},
                                                    }
                                                ]
                                            },
                                        }
                                    ]
                                }
                            )
                        )
            self.wfile.write(
                _sse_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
            )
            self.wfile.write(waterfall._terminal_frames(prompt_tokens=7, completion_tokens=3))
        except OSError:
            type(self).stopped.set()
            return

    def _collision_stream(self, payload: JsonObject) -> None:
        """Require declaration/history agreement before returning colliding public names."""
        names = [
            "apply_patch",
            "apply_patch_2",
            "agents__close",
            "apply_patch_3",
            "agents__close_2",
        ]
        tools = cast("list[JsonObject]", payload["tools"])
        assert [cast("JsonObject", tool["function"])["name"] for tool in tools] == names
        messages = cast("list[JsonObject]", payload["messages"])
        calls = [
            call
            for message in messages
            for call in cast("list[JsonObject]", message.get("tool_calls", []))
        ]
        assert [(call["id"], cast("JsonObject", call["function"])["name"]) for call in calls] == [
            ("old-custom", "apply_patch_3"),
            ("old-ns", "agents__close_2"),
        ]
        assert [message["tool_call_id"] for message in messages if message["role"] == "tool"] == [
            "old-custom",
            "old-ns",
        ]
        for index, name in enumerate(names):
            arguments = json.dumps(
                {"input": "new patch"} if name == "apply_patch_3" else {"id": name}
            )
            self.wfile.write(
                _sse_frame(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": f"new-{index}",
                                            "function": {"name": name, "arguments": arguments},
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                )
            )
        self.wfile.write(
            _sse_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        )
        self.wfile.write(waterfall._terminal_frames(prompt_tokens=7, completion_tokens=3))

    def log_message(self, format: str, *args: object) -> None:
        """Keep loopback request output quiet."""


@pytest.fixture(name="engine")
def root_engine(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_ServingEngine]:
    """Use the actual control plane with a certified root pool, never model stages."""
    original = waterfall._configured_pool_gateway

    def configured(
        root: Path, *, refusal_failover: bool, base_urls: tuple[str, str]
    ) -> tuple[GatewayManagement, str]:
        """Declare the fixture provider's real streaming tool capability."""
        capabilities = GatewayDeploymentCapabilities(
            supports_streaming=True, supports_streaming_tool_arguments=True
        )
        return original(
            root,
            refusal_failover=refusal_failover,
            base_urls=base_urls,
            gateway_capabilities=(capabilities, capabilities),
        )

    _ToolUpstream.calls = 0
    _ToolUpstream.stopped.clear()
    _ToolUpstream.release.clear()
    monkeypatch.setattr(waterfall, "_configured_pool_gateway", configured)
    monkeypatch.setattr(waterfall, "_PrimaryUpstream", _ToolUpstream)
    monkeypatch.setattr(waterfall, "_SecondaryUpstream", _ToolUpstream)
    fixture = cast(
        "Callable[[pytest.TempPathFactory], Iterator[_ServingEngine]]",
        inspect.unwrap(waterfall._engine),
    )
    yield from fixture(tmp_path_factory)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("malformed", [False, True])
def test_actual_root_responses_translated_custom_and_namespace(
    engine: _ServingEngine, stream: bool, malformed: bool
) -> None:
    """Starts, input deltas, completion and settlement agree on the real compiled path."""
    response = httpx.post(
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={
            "model": "coding",
            "input": "malformed" if malformed else "success",
            "stream": stream,
            "tools": [
                {"type": "custom", "name": "apply_patch"},
                {
                    "type": "namespace",
                    "name": "agents",
                    "tools": [
                        {"type": "function", "name": "close", "parameters": {"type": "object"}},
                    ],
                },
                {"type": "function", "name": "plain", "parameters": {"type": "object"}},
            ],
        },
        timeout=30,
    )
    assert _ToolUpstream.calls == 1, "Never replay a committed custom tool on another deployment"
    with sqlite3.connect(engine.database_path) as db:
        rows = db.execute(
            "SELECT deployment_id,state FROM gateway_attempts ORDER BY attempt_ordinal"
        ).fetchall()
    assert rows == [("alpha", "failed" if malformed else "completed")], response.text
    if malformed:
        if stream:
            assert response.status_code == 200
            assert "response.failed" in response.text
            assert "response.completed" not in response.text
            assert "response.custom_tool_call_input.delta" not in response.text
        else:
            assert response.status_code == 502
        return
    assert response.status_code == 200, response.text
    if stream:
        events = [
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        ]
        body = next(event["response"] for event in events if event["type"] == "response.completed")
        custom_delta = [
            event for event in events if event["type"] == "response.custom_tool_call_input.delta"
        ]
        assert len(custom_delta) == 1 and custom_delta[0]["delta"] == 'patch\n"café 😀"'
        added = [event["item"] for event in events if event["type"] == "response.output_item.added"]
        assert [item["type"] for item in added] == [
            "custom_tool_call",
            "function_call",
            "function_call",
        ]
        assert custom_delta[0]["item_id"] == added[0]["id"]
    else:
        body = response.json()
    assert body["model"] == "coding"
    output = body["output"]
    assert [item["call_id"] for item in output] == ["call-0", "call-1", "call-2"]
    assert output[0]["type"] == "custom_tool_call" and output[0]["input"] == 'patch\n"café 😀"'
    assert output[1]["name"] == "close" and output[1]["namespace"] == "agents"
    assert output[1]["arguments"] == '{"id":"a"}'
    assert output[2]["name"] == "plain" and output[2]["arguments"] == '{"n":1}'


@pytest.mark.parametrize(
    "boundary",
    ["response.created", "response.output_item.added", "response.completed", "response.failed"],
)
def test_custom_start_disconnect_cancels_without_replay(
    engine: _ServingEngine, boundary: str
) -> None:
    """Client closure cancels idle output, but never overwrites an observed provider terminal."""
    terminal = boundary in {"response.completed", "response.failed"}
    prompt = (
        "provider-error" if boundary == "response.failed" else "success" if terminal else "cancel"
    )
    with httpx.stream(
        "POST",
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={
            "model": "coding",
            "input": prompt,
            "stream": True,
            "tools": [
                {"type": "custom", "name": "apply_patch"},
                {
                    "type": "namespace",
                    "name": "agents",
                    "tools": [
                        {"type": "function", "name": "close", "parameters": {"type": "object"}},
                    ],
                },
                {"type": "function", "name": "plain", "parameters": {"type": "object"}},
            ],
        },
        timeout=30,
    ) as response:
        assert response.status_code == 200
        request_id = response.headers["x-request-id"]
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event["type"] == boundary:
                    if boundary == "response.output_item.added":
                        assert event["item"]["type"] == "custom_tool_call"
                    break
    expected = (
        "failed" if boundary == "response.failed" else "completed" if terminal else "cancelled"
    )
    assert [state for _, _, state in waterfall._attempt_rows(engine, request_id)] == [expected]
    assert _ToolUpstream.calls == 1


def _collision_request(stream: bool) -> JsonObject:
    """Replay custom and namespaced calls whose flattened names are plain declarations."""
    return {
        "model": "coding",
        "stream": stream,
        "tools": [
            {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}},
            {"type": "function", "name": "apply_patch_2", "parameters": {"type": "object"}},
            {"type": "function", "name": "agents__close", "parameters": {"type": "object"}},
            {"type": "custom", "name": "apply_patch"},
            {
                "type": "namespace",
                "name": "agents",
                "tools": [
                    {"type": "function", "name": "close", "parameters": {"type": "object"}},
                ],
            },
        ],
        "input": [
            {
                "type": "custom_tool_call",
                "call_id": "old-custom",
                "name": "apply_patch",
                "input": "old patch",
            },
            {"type": "custom_tool_call_output", "call_id": "old-custom", "output": "done"},
            {
                "type": "function_call",
                "call_id": "old-ns",
                "name": "close",
                "namespace": "agents",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "old-ns", "output": "done"},
            {"role": "user", "content": "always-500"},
        ],
    }


def _assert_collision_response(response: httpx.Response, stream: bool) -> None:
    """Require exact public custom/function identity without corrupting ordinary calls."""
    assert response.status_code == 200, response.text
    if stream:
        events = [
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        ]
        body = next(event["response"] for event in events if event["type"] == "response.completed")
        added = [event["item"] for event in events if event["type"] == "response.output_item.added"]
        assert [item["type"] for item in added] == ["function_call"] * 3 + [
            "custom_tool_call",
            "function_call",
        ]
    else:
        body = response.json()
    output = body["output"]
    assert [item["call_id"] for item in output] == [f"new-{i}" for i in range(5)]
    assert [(item["type"], item["name"], item.get("namespace")) for item in output] == [
        ("function_call", "apply_patch", None),
        ("function_call", "apply_patch_2", None),
        ("function_call", "agents__close", None),
        ("custom_tool_call", "apply_patch", None),
        ("function_call", "close", "agents"),
    ]
    assert output[3]["input"] == "new patch"
    assert json.loads(output[0]["arguments"]) == {"id": "apply_patch"}


@pytest.mark.parametrize("stream", [False, True])
def test_actual_native_collision_history_and_inverse_agree(
    engine: _ServingEngine, stream: bool
) -> None:
    """Actual Rust buffered and streamed responses preserve every colliding tool origin."""
    response = httpx.post(
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=_collision_request(stream),
        timeout=30,
    )
    _assert_collision_response(response, stream)
    assert _ToolUpstream.calls == 1
    assert [
        state for _, _, state in waterfall._attempt_rows(engine, response.headers["x-request-id"])
    ] == ["completed"]

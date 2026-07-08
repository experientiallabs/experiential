"""RunnerLink conformance tests: drive the whole broker offline via a scripted runner peer.

No socket, no node, no Bedrock. A `FakeChannel` plays the runner side (emits tool_request /
llm_request / done frames and records what the host sent back); a fake `AgentEnvironment` stands in
for the world model; `worker_fn` is injected so the worker-LLM callback needs no provider. This is
the same seam the E2B fake-sandbox test used, now for the frame protocol. The frame codec and the
Bedrock translation (shared with the SSH shim) are unit-tested directly.
"""

from __future__ import annotations

from typing import Any, cast

from wmh.core.types import Action, JsonObject, Observation
from wmh.harness.runner_link import (
    RunnerLink,
    bedrock_to_completion,
    openai_to_bedrock,
    read_frame,
    write_frame,
)
from wmh.harness.runtime import StopReason
from wmh.harness.tools import SUBMIT, TOOL_REGISTRY


class _Env:
    def __init__(self) -> None:
        self.actions: list[Action] = []

    def execute(self, action: Action) -> Observation:
        self.actions.append(action)
        return Observation(content=f"ran {action.name}")

    def close(self) -> None:
        pass


class _FakeChannel:
    """Plays the runner peer: recv() yields scripted frames in order; send() records host output."""

    def __init__(self, script: list) -> None:
        self.sent: list = []
        self._script = list(script)

    def send(self, frame: JsonObject) -> None:
        self.sent.append(frame)

    def recv(self) -> JsonObject | None:
        return self._script.pop(0) if self._script else None


def _tools() -> list:
    return [TOOL_REGISTRY["bash"], SUBMIT]


def _link(channel: _FakeChannel, **kw) -> RunnerLink:  # noqa: ANN003
    # worker_fn returns a fixed completion so the llm_request path needs no provider.
    return RunnerLink(
        channel,
        worker_fn=lambda body: {"choices": [{"message": {"content": "ok"}}]},
        **kw,
    )


def _sent(channel: _FakeChannel, kind: str) -> list:
    # cast to Any so deep-indexing assertions on frame payloads stay readable in tests.
    return [cast(Any,f) for f in channel.sent if f.get("type") == kind]


# --- frame codec ---
class _PipeSock:
    def __init__(self) -> None:
        self.buf = bytearray()

    def sendall(self, data: bytes) -> None:
        self.buf += data

    def recv(self, n: int) -> bytes:
        if not self.buf:
            return b""
        chunk = bytes(self.buf[:n])
        del self.buf[:n]
        return chunk


def test_frame_codec_roundtrip_and_eof() -> None:
    sock = _PipeSock()
    hello: JsonObject = {"type": "hello", "n": 1, "s": "x" * 5000}
    done: JsonObject = {"type": "done", "answer": "café"}
    write_frame(sock, hello)
    write_frame(sock, done)
    assert read_frame(sock) == hello
    assert read_frame(sock) == done
    assert read_frame(sock) is None  # clean EOF


# --- episode broker ---
def test_episode_start_carries_task_and_tools() -> None:
    ch = _FakeChannel([{"type": "done", "answer": "x"}])
    _link(ch, system_prompt="sys", files={"src/agent.ts": "// a"}).run(
        "t1", "do it", _Env(), tools=_tools()
    )
    start = _sent(ch, "episode_start")
    assert len(start) == 1
    s = start[0]
    assert s["instruction"] == "do it" and s["system"] == "sys"
    assert s["files"] == {"src/agent.ts": "// a"}
    assert {t["name"] for t in s["tools"]} >= {"bash", "submit"}


def test_tool_request_routes_to_env_and_records_step() -> None:
    env = _Env()
    script = [
        {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {"command": "ls"}},
        {"type": "done", "answer": "done-42"},
    ]
    ch = _FakeChannel(script)
    result = _link(ch).run("t1", "do it", env, tools=_tools())
    assert result.stop_reason is StopReason.SUBMITTED
    assert result.answer == "done-42"
    assert [a.name for a in env.actions] == ["bash"]  # the host WM answered the call
    tr = _sent(ch, "tool_response")
    assert tr[0]["content"] == "ran bash" and tr[0]["is_error"] is False
    assert tr[0]["req_id"] == 1  # correlation id echoed
    assert len(result.steps) == 1


def test_env_action_budget_enforced() -> None:
    env = _Env()
    script = [
        {"type": "tool_request", "req_id": i, "name": "bash", "arguments": {}} for i in range(4)
    ]
    script.append({"type": "done", "answer": "ok"})
    ch = _FakeChannel(script)
    result = _link(ch, max_env_actions=2).run("t1", "x", env, tools=_tools())
    assert len(env.actions) == 2  # only budgeted calls reached the environment
    responses = _sent(ch, "tool_response")
    assert responses[2]["is_error"] is True and "budget" in responses[2]["content"]
    assert result.stop_reason is StopReason.SUBMITTED


def test_llm_request_answered_via_worker_fn() -> None:
    calls: list[JsonObject] = []

    def worker(body: JsonObject) -> JsonObject:
        calls.append(body)
        return {"choices": [{"message": {"content": "hi", "role": "assistant"}}]}

    script = [
        {"type": "llm_request", "req_id": 7, "openai_body": {"messages": [{"role": "user"}]}},
        {"type": "done", "answer": "fin"},
    ]
    ch = _FakeChannel(script)
    RunnerLink(ch, worker_fn=worker).run("t1", "x", _Env(), tools=_tools())
    assert len(calls) == 1  # the worker callback fired host-side
    resp = _sent(ch, "llm_response")[0]
    assert resp["req_id"] == 7
    assert resp["completion"]["choices"][0]["message"]["content"] == "hi"


def test_worker_fn_error_is_reported_not_crashed() -> None:
    def boom(body: JsonObject) -> JsonObject:
        raise RuntimeError("provider down")

    script = [
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
        {"type": "done", "answer": "ok"},
    ]
    ch = _FakeChannel(script)
    result = RunnerLink(ch, worker_fn=boom).run("t1", "x", _Env(), tools=_tools())
    resp = _sent(ch, "llm_response")[0]
    assert "provider down" in resp["error"]  # surfaced to the runner, host survives
    assert result.stop_reason is StopReason.SUBMITTED


def test_channel_close_without_done_reports_error() -> None:
    ch = _FakeChannel(
        [{"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {}}]  # then EOF
    )
    result = _link(ch).run("t1", "x", _Env(), tools=_tools())
    assert result.stop_reason is StopReason.MAX_TURNS  # a step ran, so not a bare ERROR
    assert result.steps[-1].observation.is_error


def test_episode_error_frame_reports_error() -> None:
    ch = _FakeChannel([{"type": "episode_error", "note": "pi fatal"}])
    result = _link(ch).run("t1", "x", _Env(), tools=_tools())
    assert result.stop_reason is StopReason.ERROR  # no steps -> hard error
    assert "pi fatal" in result.steps[-1].observation.content


# --- shared Bedrock translation (offline) ---
def test_openai_to_bedrock_maps_tools_and_tool_results() -> None:
    body: JsonObject = {
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "look up u1"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "get_user", "arguments": '{"id":"u1"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "found u1"},
        ],
        "tools": [
            {"function": {"name": "get_user", "description": "d", "parameters": {"type": "object"}}}
        ],
    }
    system, msgs, tool_config = openai_to_bedrock(body)
    assert system == [{"text": "be nice"}]
    assert tool_config is not None
    assert cast(Any,tool_config)["tools"][0]["toolSpec"]["name"] == "get_user"
    # assistant toolUse + tool result present
    blocks = [b for m in cast(Any,msgs) for b in m["content"]]
    assert any("toolUse" in b for b in blocks)
    assert any("toolResult" in b for b in blocks)


def test_bedrock_to_completion_shape() -> None:
    resp: JsonObject = {
        "output": {
            "message": {
                "content": [
                    {"text": "sure"},
                    {"toolUse": {"toolUseId": "t1", "name": "get_user", "input": {"id": "u1"}}},
                ]
            }
        },
        "stopReason": "tool_use",
    }
    completion = cast(Any,bedrock_to_completion(resp))
    choice = completion["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "sure"
    tc = choice["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_user"
    assert tc["function"]["arguments"] == '{"id": "u1"}'

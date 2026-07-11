"""RunnerLink conformance tests: drive the whole broker offline via a scripted runner peer.

No socket, no node, no Bedrock. A `FakeChannel` plays the runner side (emits tool_request /
llm_request / done frames and records what the host sent back); a fake `AgentEnvironment` stands in
for the world model; `worker_fn` is injected so the worker-LLM callback needs no provider.
"""

from __future__ import annotations

from typing import Any, cast

from llm_waterfall import ChatRequest, ChatResponse

from wmh.core.types import Action, JsonObject, Observation
from wmh.harness.runner_link import RunnerLink, read_frame, write_frame
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


def _completion(
    content: str = "ok", *, input_tokens: int = 0, output_tokens: int = 0
) -> ChatResponse:
    return ChatResponse.model_validate(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            },
        }
    )


def _link(channel: _FakeChannel, **kw) -> RunnerLink:  # noqa: ANN003
    # worker_fn returns a fixed completion so the llm_request path needs no provider.
    return RunnerLink(channel, worker_fn=lambda request: _completion(), **kw)


def _sent(channel: _FakeChannel, kind: str) -> list:
    # cast to Any so deep-indexing assertions on frame payloads stay readable in tests.
    return [cast(Any, f) for f in channel.sent if f.get("type") == kind]


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
    calls: list[ChatRequest] = []

    def worker(request: ChatRequest) -> ChatResponse:
        calls.append(request)
        return _completion("hi")

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
    def boom(request: ChatRequest) -> ChatResponse:
        del request
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


def test_tools_bound_at_construction_satisfy_runtime_contract() -> None:
    # RunnerLink(tools=...) is a drop-in runtime: run(task_id, instruction, env), no tools kwarg.
    env = _Env()
    script = [
        {"type": "tool_request", "req_id": 1, "name": "bash", "arguments": {"command": "ls"}},
        {"type": "done", "answer": "done"},
    ]
    ch = _FakeChannel(script)
    link = RunnerLink(ch, tools=_tools(), worker_fn=lambda request: _completion())
    result = link.run("t1", "do it", env)  # no tools= : uses the constructor's
    assert result.stop_reason is StopReason.SUBMITTED
    assert [a.name for a in env.actions] == ["bash"]
    assert {t["name"] for t in _sent(ch, "episode_start")[0]["tools"]} >= {"bash", "submit"}


def test_multiple_episodes_over_one_channel() -> None:
    # A persistent channel drives episodes sequentially (what closed-loop eval / the search do).
    script = [
        {"type": "done", "answer": "a1"},  # episode 1
        {"type": "done", "answer": "a2"},  # episode 2
    ]
    ch = _FakeChannel(script)
    link = _link(ch, tools=_tools())
    r1 = link.run("t1", "first", _Env())
    r2 = link.run("t2", "second", _Env())
    assert (r1.answer, r2.answer) == ("a1", "a2")
    starts = _sent(ch, "episode_start")
    assert len(starts) == 2
    assert starts[0]["episode_id"] != starts[1]["episode_id"]  # fresh id per episode
    assert (starts[0]["instruction"], starts[1]["instruction"]) == ("first", "second")


def test_doc_runtime_dispatches_runner_link_under_pi_transport_link() -> None:
    import os as _os

    from wmh.harness.doc import RUNTIME_KIND_ID, TOOL_POLICY_ID, HarnessDoc, Surface, SurfaceKind
    from wmh.harness.runner_link import RunnerLink, set_active_channel
    from wmh.providers.base import Provider, ProviderConfig, ProviderKind

    class _P:
        config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

        def complete(self, *a, **k) -> object:  # noqa: ANN002, ANN003
            raise NotImplementedError

        def complete_chat(self, request: ChatRequest) -> ChatResponse:
            return _completion()

        def embed(self, texts) -> list:  # noqa: ANN001
            return [[0.0] for _ in texts]

        def verify(self) -> object:
            raise NotImplementedError

    doc = HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            Surface(id="code:a", kind=SurfaceKind.CODE, path="src/agent.ts", content="// a"),
        ],
    )
    provider = cast(Provider, _P())
    prev = _os.environ.get("PI_TRANSPORT")
    _os.environ["PI_TRANSPORT"] = "link"
    try:
        set_active_channel(None)
        try:
            doc.runtime(provider)
            raise AssertionError("expected RuntimeError with no active channel")
        except RuntimeError as exc:
            assert "no active runner channel" in str(exc)
        set_active_channel(_FakeChannel([]))
        assert isinstance(doc.runtime(provider), RunnerLink)
    finally:
        set_active_channel(None)
        if prev is None:
            _os.environ.pop("PI_TRANSPORT", None)
        else:
            _os.environ["PI_TRANSPORT"] = prev


def test_worker_usage_accumulates_across_llm_requests() -> None:
    """Each answered llm_request adds its completion usage to RunResult.worker_usage."""
    replies = iter([_completion("a", input_tokens=100, output_tokens=7), _completion("b")])
    script = [
        {"type": "llm_request", "req_id": 1, "openai_body": {}},
        {"type": "llm_request", "req_id": 2, "openai_body": {}},
        {"type": "done", "answer": "fin"},
    ]
    ch = _FakeChannel(script)
    result = RunnerLink(ch, worker_fn=lambda request: next(replies)).run(
        "t1", "x", _Env(), tools=_tools()
    )
    assert result.worker_usage is not None
    assert result.worker_usage.calls == 2
    assert result.worker_usage.input_tokens == 100
    assert result.worker_usage.output_tokens == 7
    # No llm_request at all -> usage stays None (not zero: the runtime reported nothing).
    quiet = RunnerLink(
        _FakeChannel([{"type": "done", "answer": "ok"}]),
        worker_fn=lambda request: _completion(),
    ).run("t2", "x", _Env(), tools=_tools())
    assert quiet.worker_usage is None

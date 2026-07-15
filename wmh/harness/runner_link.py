"""RunnerLink: the transport that replaces per-episode SSH + reverse tunnel for the pi runner.

The control plane (this process) holds the model credentials and the world-model session state; a
long-lived pi *runner* — local, on nucbox, or any remote box — dials the host and blocks reading
frames. One episode is driven over one bidirectional frame channel: the host sends an
`episode_start`, then answers the two callbacks the runner pushes up — `llm_request` (the worker
LLM completion, produced host-side so no creds ever reach the runner) and `tool_request` (the
environment tool call, routed to the `AgentEnvironment` / world model) — until `done`.

Frames are length-prefixed JSON (4-byte big-endian length + UTF-8 body) over a raw socket, so the
transport adds ZERO dependency on either side (Python stdlib here; Node stdlib in the runner). The
episode-driving logic is decoupled from the socket behind the `Channel` protocol so a scripted
in-process peer can exercise the whole broker offline (see runner_link_test.py).

The link is provider-neutral: the caller supplies a structured tool-calling provider, which owns
authentication, routing, wire translation, retries, and failover. RunnerLink only validates and
brokers frames plus environment tool calls.
"""

from __future__ import annotations

import json
import struct
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, cast

from llm_waterfall import ChatRequest, ChatResponse

from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Observation, Step
from wmh.harness.environment import AgentEnvironment, is_env_action
from wmh.harness.runtime import RunResult, StopReason, TokenUsage
from wmh.harness.tools import ToolSpec
from wmh.providers.base import ToolCallingProvider

DEFAULT_MAX_ENV_ACTIONS = 40


# --------------------------------------------------------------------------------------------------
# Wire framing: length-prefixed JSON over a raw socket (stdlib only, both sides).
# --------------------------------------------------------------------------------------------------
class _SupportsSocket(Protocol):
    def sendall(self, data: bytes) -> None: ...
    def recv(self, n: int) -> bytes: ...


def write_frame(sock: _SupportsSocket, frame: JsonObject) -> None:
    """Send one JSON frame: 4-byte big-endian length prefix + UTF-8 body."""
    body = json.dumps(frame).encode("utf-8")
    sock.sendall(struct.pack(">I", len(body)) + body)


def read_frame(sock: _SupportsSocket) -> JsonObject | None:
    """Read one framed JSON message, or None if the peer closed the connection cleanly."""
    header = _recv_exactly(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    body = _recv_exactly(sock, length)
    if body is None:
        return None
    return cast("JsonObject", json.loads(body))


def _recv_exactly(sock: _SupportsSocket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class Channel(Protocol):
    """A bidirectional frame channel to the runner peer (a socket, or a test double)."""

    def send(self, frame: JsonObject) -> None: ...
    def recv(self) -> JsonObject | None: ...


class SocketChannel:
    """A `Channel` backed by a connected socket, using the length-prefixed JSON framing."""

    def __init__(self, sock: _SupportsSocket) -> None:
        self._sock = sock

    def send(self, frame: JsonObject) -> None:
        write_frame(self._sock, frame)

    def recv(self) -> JsonObject | None:
        return read_frame(self._sock)


# The process-wide runner channel doc.runtime(PI_TRANSPORT=link) drives. A search/eval sets it once
# (its runner connection is process-scoped infra), so create_harness's internal doc.runtime() calls
# reach the runner without threading a channel through every signature; cleared at teardown.
_ACTIVE_CHANNEL: Channel | None = None


def set_active_channel(channel: Channel | None) -> None:
    global _ACTIVE_CHANNEL
    _ACTIVE_CHANNEL = channel


def active_channel() -> Channel | None:
    return _ACTIVE_CHANNEL


def params_schema(tool: ToolSpec) -> JsonObject:
    """A JSON-schema `parameters` object for a tool, as the model's function-calling API expects."""
    props: JsonObject = {
        name: {"type": "string", "description": desc} for name, desc in tool.arguments.items()
    }
    return {"type": "object", "properties": props, "required": list(tool.arguments)}


# --------------------------------------------------------------------------------------------------
# Host-side episode: environment tool routing, budget, and transcript recording.
# --------------------------------------------------------------------------------------------------
@dataclass
class HostEpisode:
    """Per-episode host state: routes tool calls to the environment under a budget, records Steps.

    Same budget/step-recording contract the SSH shim's `_Episode` had, minus the HTTP specifics.
    """

    instruction: str
    tools: list[ToolSpec]
    environment: AgentEnvironment
    max_env_actions: int = DEFAULT_MAX_ENV_ACTIONS
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    _env_calls: int = 0

    def tool_specs(self) -> list[JsonObject]:
        return [
            {"name": t.name, "description": t.description, "parameters": params_schema(t)}
            for t in self.tools
        ]

    def run_tool(self, name: str, arguments: JsonObject) -> JsonObject:
        """Answer one environment tool call: enforce the budget, record the Step."""
        action = Action(kind=ActionKind.TOOL_CALL, name=name, arguments=arguments)
        if self._env_calls >= self.max_env_actions:
            obs = Observation(content="environment action budget exhausted", is_error=True)
        elif name not in {t.name for t in self.tools} or not is_env_action(action):
            obs = Observation(content=f"tool {name!r} not available", is_error=True)
        else:
            self._env_calls += 1
            obs = self.environment.execute(action)
        self.steps.append(
            Step(action=action, observation=obs, state_before=EnvState(), task=self.instruction)
        )
        return {"content": obs.content, "is_error": obs.is_error}


# The worker function the host uses to answer llm_request frames; injectable for tests.
WorkerFn = Callable[[ChatRequest], ChatResponse]


class RunnerLink:
    """Drives one pi episode over a `Channel` to the runner peer.

    Sends `episode_start`, then answers `llm_request` (worker LLM, host-side) and `tool_request`
    (environment) frames until `done`/`episode_error`, returning a `RunResult` shaped exactly like
    the other runtimes. One `RunnerLink.run` == one episode; concurrent episodes multiplex over the
    same channel by `episode_id` (a later migration step).
    """

    def __init__(
        self,
        channel: Channel,
        *,
        tools: list[ToolSpec] | None = None,
        provider: ToolCallingProvider | None = None,
        worker_fn: WorkerFn | None = None,
        files: dict[str, str] | None = None,
        system_prompt: str = "",
        max_env_actions: int = DEFAULT_MAX_ENV_ACTIONS,
    ) -> None:
        self._channel = channel
        # Tools bound at construction make RunnerLink satisfy the runtime contract closed-loop eval
        # drives — `run(task_id, instruction, environment)` — while `run(..., tools=...)` still lets
        # a caller (or the conformance tests) override per episode.
        self._tools = tools or []
        if worker_fn is None and provider is None:
            raise ValueError("RunnerLink needs a ToolCallingProvider or worker_fn")
        # worker_fn lets tests answer llm_request without a real provider.
        if worker_fn is not None:
            self._worker_fn = worker_fn
        else:
            assert provider is not None
            self._worker_fn = provider.complete_chat
        self._files = files or {}
        self._system_prompt = system_prompt
        self._max_env_actions = max_env_actions

    def run(
        self,
        task_id: str,
        instruction: str,
        environment: AgentEnvironment,
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RunResult:
        episode = HostEpisode(
            instruction=instruction,
            tools=tools if tools is not None else self._tools,
            environment=environment,
            max_env_actions=self._max_env_actions,
        )
        episode_id = uuid.uuid4().hex
        self._channel.send(
            {
                "type": "episode_start",
                "episode_id": episode_id,
                "task_id": task_id,
                "instruction": instruction,
                "system": self._system_prompt,
                "tools": episode.tool_specs(),
                "files": self._files,
                "max_env_actions": self._max_env_actions,
            }
        )
        usage = TokenUsage()
        while True:
            frame = self._channel.recv()
            if frame is None:  # channel closed before the episode finished
                return self._error_result(
                    task_id, episode, instruction, "runner channel closed", usage=usage
                )
            kind = frame.get("type")
            if kind == "llm_request":
                self._answer_llm(episode_id, frame, usage)
            elif kind == "tool_request":
                name = frame.get("name")
                args = frame.get("arguments")
                obs = episode.run_tool(
                    name if isinstance(name, str) else "",
                    args if isinstance(args, dict) else {},
                )
                self._channel.send(
                    {
                        "type": "tool_response",
                        "episode_id": episode_id,
                        "req_id": frame.get("req_id"),
                        **obs,
                    }
                )
            elif kind == "done":
                answer = frame.get("answer")
                episode.answer = answer if isinstance(answer, str) else ""
                return RunResult(
                    task_id=task_id,
                    steps=episode.steps,
                    stop_reason=StopReason.SUBMITTED,
                    answer=episode.answer,
                    turns=len(episode.steps),
                    worker_usage=usage if usage.calls else None,
                )
            elif kind == "episode_error":
                note = frame.get("note")
                return self._error_result(
                    task_id,
                    episode,
                    instruction,
                    note if isinstance(note, str) else "runner error",
                    usage=usage,
                )
            # unknown frame types are ignored (forward-compatible)

    def _answer_llm(self, episode_id: str, frame: JsonObject, usage: TokenUsage) -> None:
        req_id = frame.get("req_id")
        body = frame.get("openai_body")
        try:
            request = ChatRequest.model_validate(body if isinstance(body, dict) else {})
            completion = self._worker_fn(request)
            # Meter the worker leg from the provider's structured response.
            usage.calls += 1
            reported = completion.token_usage()
            usage.input_tokens += reported.input_tokens
            usage.output_tokens += reported.output_tokens
            response: JsonObject = {
                "type": "llm_response",
                "episode_id": episode_id,
                "req_id": req_id,
                "completion": completion.wire_payload(),
            }
        except Exception as exc:  # noqa: BLE001 - report to the runner, never crash the host
            response = {
                "type": "llm_response",
                "episode_id": episode_id,
                "req_id": req_id,
                "error": str(exc),
            }
        # A send timeout is transport failure with an uncertain delivery state. Let it propagate
        # so the owning runtime retires the channel instead of sending a second response frame.
        self._channel.send(response)

    @staticmethod
    def _error_result(
        task_id: str,
        episode: HostEpisode,
        instruction: str,
        note: str,
        *,
        usage: TokenUsage | None = None,
    ) -> RunResult:
        stop = StopReason.MAX_TURNS if episode.steps else StopReason.ERROR
        episode.steps.append(
            Step(
                action=Action(kind=ActionKind.MESSAGE, content="(runner link)"),
                observation=Observation(content=note, is_error=True),
                state_before=EnvState(),
                task=instruction,
            )
        )
        return RunResult(
            task_id=task_id,
            steps=episode.steps,
            stop_reason=stop,
            answer="",
            turns=len(episode.steps),
            worker_usage=usage if usage is not None and usage.calls else None,
        )

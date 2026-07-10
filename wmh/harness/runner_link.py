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

Only the frame transport is new; the worker-LLM translation (`openai_to_bedrock`,
`bedrock_to_completion`) and the tool budget/step recording (`HostEpisode`) are the same logic the
SSH shim in `pi_runtime.py` runs, lifted here so both transports share one implementation.
"""

from __future__ import annotations

import json
import os
import struct
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Observation, Step
from wmh.harness.environment import AgentEnvironment, is_env_action
from wmh.harness.runtime import RunResult, StopReason, TokenUsage
from wmh.harness.tools import ToolSpec
from wmh.providers.base import ProviderConfig, ProviderKind

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


# --------------------------------------------------------------------------------------------------
# Worker-LLM completion (host-side; creds never leave this process).
# --------------------------------------------------------------------------------------------------
@dataclass
class WorkerConfig:
    """How the host answers worker-LLM requests. Creds are read from the environment, host-side."""

    backend: str = "openai"  # "openai": OpenAI-compatible endpoint | "bedrock": Converse via boto3
    model: str = ""
    region: str = "us-east-1"
    base_url: str = ""
    key_env: str = ""


_PI_AGENT_ENV_VARS = (
    "PI_AGENT_BACKEND",
    "PI_AGENT_MODEL",
    "PI_AGENT_REGION",
    "PI_AGENT_BASE_URL",
    "PI_AGENT_KEY_ENV",
)


def worker_config_from_env() -> WorkerConfig:
    """Build the worker config from the same PI_AGENT_* env knobs PiRuntime reads.

    So `doc.runtime()` under PI_TRANSPORT=link answers the worker LLM the same way the SSH shim
    does (e.g. PI_AGENT_BACKEND=bedrock + PI_AGENT_MODEL=<Haiku profile>).
    """
    # Defaults mirror pi_runtime.PI_AGENT_* exactly, so PI_TRANSPORT=link on an otherwise-default
    # (deepseek) SSH setup works without extra env, not sending keyless/model-less requests.
    return WorkerConfig(
        backend=os.environ.get("PI_AGENT_BACKEND", "openai"),
        model=os.environ.get("PI_AGENT_MODEL", "deepseek-chat"),
        region=os.environ.get("PI_AGENT_REGION", os.environ.get("AWS_REGION", "us-east-1")),
        base_url=os.environ.get("PI_AGENT_BASE_URL", "https://api.deepseek.com/v1"),
        key_env=os.environ.get("PI_AGENT_KEY_ENV", "DEEPSEEK_API_KEY"),
    )


def worker_config_for(config: ProviderConfig) -> WorkerConfig:
    """The worker config for an eval, preferring explicit env over provider derivation.

    Precedence: any PI_AGENT_* env var set -> `worker_config_from_env` exactly as before (the
    operator asked for a specific worker). Otherwise, a Bedrock agent provider is fully derivable
    (model id + region; AWS creds stay host-side as always) — this is what lets the hosted
    platform point pi at the agent's catalog model with zero env plumbing. Any other kind keeps
    the env defaults: their auth shapes (Azure api-version query strings, deployment paths) do
    not fit the single openai-completions POST `worker_completion` sends.
    """
    if any(os.environ.get(name) for name in _PI_AGENT_ENV_VARS):
        return worker_config_from_env()
    if config.kind is ProviderKind.BEDROCK and config.model:
        return WorkerConfig(
            backend="bedrock",
            model=config.model,
            region=config.region or os.environ.get("AWS_REGION", "us-east-1"),
        )
    return worker_config_from_env()


# The process-wide runner channel doc.runtime(PI_TRANSPORT=link) drives. A search/eval sets it once
# (its runner connection is process-scoped infra), so create_harness's internal doc.runtime() calls
# reach the runner without threading a channel through every signature; cleared at teardown.
_ACTIVE_CHANNEL: Channel | None = None


def set_active_channel(channel: Channel | None) -> None:
    global _ACTIVE_CHANNEL
    _ACTIVE_CHANNEL = channel


def active_channel() -> Channel | None:
    return _ACTIVE_CHANNEL


def worker_completion(body: JsonObject, cfg: WorkerConfig) -> JsonObject:
    """Answer one worker-LLM request, returning a single OpenAI chat.completion object.

    Unlike the SSH shim (which streamed raw SSE), the framed transport carries one finished
    completion object — the runner synthesizes whatever its parser needs locally.
    """
    if cfg.backend == "bedrock":
        return _bedrock_completion(body, cfg)
    return _openai_completion(body, cfg)


def _normalized_openai_body(body: JsonObject, cfg: WorkerConfig) -> JsonObject:
    """The runner's request body rewritten for a single non-streaming worker call.

    The runner's pi client asks for a stream (`stream: true` + `stream_options`), but the frame
    transport carries exactly one finished completion, so the request must be non-streaming —
    and strict OpenAI-compatible servers (DeepSeek) 400 on `stream_options` without
    `stream=true`. `max_completion_tokens` is translated to the widely supported `max_tokens`.
    """
    b = dict(body)
    if cfg.model:
        b["model"] = cfg.model
    b["stream"] = False
    b.pop("stream_options", None)
    # Always removed — strict backends 400 on the field itself; an explicit max_tokens wins.
    max_completion = b.pop("max_completion_tokens", None)
    if max_completion is not None and "max_tokens" not in b:
        b["max_tokens"] = max_completion
    return b


def _openai_completion(body: JsonObject, cfg: WorkerConfig) -> JsonObject:
    import urllib.request

    b = _normalized_openai_body(body, cfg)
    key = os.environ.get(cfg.key_env, "") if cfg.key_env else ""
    req = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(b).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310 - fixed https endpoint
        return cast("JsonObject", json.loads(resp.read()))


def _bedrock_completion(body: JsonObject, cfg: WorkerConfig) -> JsonObject:
    import boto3

    system, messages, tool_config = openai_to_bedrock(body)
    mt = body.get("max_tokens")
    inf: dict[str, Any] = {"maxTokens": mt if isinstance(mt, int) else 4096}
    if body.get("temperature") is not None:
        inf["temperature"] = body["temperature"]
    kwargs: dict[str, Any] = {"modelId": cfg.model, "messages": messages, "inferenceConfig": inf}
    if system:
        kwargs["system"] = system
    if tool_config:
        kwargs["toolConfig"] = tool_config
    client = boto3.client("bedrock-runtime", region_name=cfg.region)
    return bedrock_to_completion(client.converse(**kwargs))


def _text(content: object) -> str:
    return content if isinstance(content, str) else str(content)


def openai_to_bedrock(body: JsonObject) -> tuple[list, list, dict | None]:
    """OpenAI chat request -> (system blocks, Bedrock messages, toolConfig).

    Maps assistant tool_calls -> toolUse, role:"tool" results -> toolResult (grouped into a user
    turn), and OpenAI function tools -> Bedrock toolSpec. Consecutive same-role turns are merged so
    the Converse alternation contract holds.
    """
    b = cast("Any", body)
    system: list[Any] = []
    msgs: list[Any] = []

    def _push(role: str, content: list[Any]) -> None:
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"].extend(content)
        else:
            msgs.append({"role": role, "content": content})

    for m in b.get("messages", []):
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            if content:
                system.append({"text": _text(content)})
        elif role == "tool":
            result = {
                "toolResult": {
                    "toolUseId": m.get("tool_call_id", ""),
                    "content": [{"text": _text(content)}],
                }
            }
            _push("user", [result])
        elif role == "assistant":
            blocks: list[Any] = []
            if content:
                blocks.append({"text": _text(content)})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (ValueError, TypeError):
                    args = {}
                use = {"toolUseId": tc.get("id", ""), "name": fn.get("name", ""), "input": args}
                blocks.append({"toolUse": use})
            if blocks:
                _push("assistant", blocks)
        elif content:  # user (and any other) -> plain text user turn
            _push("user", [{"text": _text(content)}])

    tools = b.get("tools") or []
    tool_config: dict[str, Any] | None = None
    if tools:
        specs = []
        for t in tools:
            fn = t.get("function", t)
            params = fn.get("parameters", {"type": "object", "properties": {}})
            specs.append(
                {
                    "toolSpec": {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "inputSchema": {"json": params},
                    }
                }
            )
        tool_config = {"tools": specs}
        # Carry OpenAI tool_choice -> Bedrock toolChoice so a forced/required tool call is not
        # silently downgraded to auto. "auto"/absent = Bedrock's default (no explicit toolChoice).
        choice = b.get("tool_choice")
        if choice == "required":
            tool_config["toolChoice"] = {"any": {}}
        elif isinstance(choice, dict) and choice.get("type") == "function":
            name = choice.get("function", {}).get("name")
            if name:
                tool_config["toolChoice"] = {"tool": {"name": name}}
        elif choice == "none":
            tool_config = None  # caller asked the model NOT to use tools
    return system, msgs, tool_config


def bedrock_to_completion(resp: JsonObject) -> JsonObject:
    """Bedrock Converse output -> one OpenAI chat.completion object (non-streaming)."""
    message = cast("Any", resp).get("output", {}).get("message", {})
    text_parts: list[str] = []
    tool_calls: list[Any] = []
    for block in message.get("content", []):
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(
                {
                    "id": tu.get("toolUseId", ""),
                    "type": "function",
                    "function": {
                        "name": tu.get("name", ""),
                        "arguments": json.dumps(tu.get("input", {})),
                    },
                }
            )
    msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    stop = cast("Any", resp).get("stopReason", "end_turn")
    # Map Bedrock stop reasons to OpenAI finish_reasons. A blocked/filtered response is surfaced as
    # "content_filter" rather than looking like a normal "stop", so downstream can tell it apart.
    finish = {
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "content_filtered": "content_filter",
        "guardrail_intervened": "content_filter",
    }.get(stop, "stop")
    completion: JsonObject = {
        "id": "chatcmpl-runnerlink",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
    }
    usage = cast("Any", resp).get("usage", {})
    if usage:
        # OpenAI usage shape, so worker-token accounting reads one format on every backend.
        completion["usage"] = {
            "prompt_tokens": int(usage.get("inputTokens", 0)),
            "completion_tokens": int(usage.get("outputTokens", 0)),
        }
    return completion


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
WorkerFn = Any  # Callable[[JsonObject], JsonObject]


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
        worker: WorkerConfig | None = None,
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
        cfg = worker or WorkerConfig()
        self._worker_cfg = cfg
        # worker_fn lets tests answer llm_request without a real provider; defaults to the real one.
        self._worker_fn: WorkerFn = worker_fn or (lambda body: worker_completion(body, cfg))
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
            completion = self._worker_fn(body if isinstance(body, dict) else {})
            # Meter the worker leg host-side: these calls bypass the Provider abstraction, so
            # the completion's usage block is the only spend record (failed calls cost nothing).
            usage.calls += 1
            reported = completion.get("usage")
            if isinstance(reported, dict):
                prompt = reported.get("prompt_tokens")
                out = reported.get("completion_tokens")
                usage.input_tokens += prompt if isinstance(prompt, int) else 0
                usage.output_tokens += out if isinstance(out, int) else 0
            self._channel.send(
                {
                    "type": "llm_response",
                    "episode_id": episode_id,
                    "req_id": req_id,
                    "completion": completion,
                }
            )
        except Exception as exc:  # noqa: BLE001 - report to the runner, never crash the host
            self._channel.send(
                {
                    "type": "llm_response",
                    "episode_id": episode_id,
                    "req_id": req_id,
                    "error": str(exc),
                }
            )

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

"""Adapter from an installed Pi JSON-event process to the WMO agent contract."""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
from contextlib import AbstractContextManager
from contextvars import copy_context
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from typing import cast
from urllib.parse import unquote, urlparse

from pydantic import TypeAdapter, ValidationError

from wmo.common.core.artifacts import (
    FailureAttribution,
    FailureCode,
    JsonObject,
    StructuredFailure,
)
from wmo.common.models import (
    AssistantAction,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from wmo.common.rollouts import RolloutEventKind, RolloutSpan, StopReason
from wmo.common.tasks import TaskCase, ToolSchema
from wmo.runtime.agents.interface import AgentEpisode
from wmo.runtime.environments import EnvironmentSession, Observation
from wmo.runtime.router.endpoint import HttpMessage, chat_completion, model_messages
from wmo.runtime.router.streaming import chat_stream

_JSON_OBJECT = TypeAdapter(JsonObject)
_HTTP_MESSAGES = TypeAdapter(tuple[HttpMessage, ...])
_DETERMINISTIC_EVENT_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_WMO_PI_PROVIDER = "wmo-injected"
_WMO_PI_MODEL = "wmo-injected-model"
_DEFAULT_TIMEOUT_SECONDS = 300.0
_UNIX_MILLISECOND_MAGNITUDE = 100_000_000_000


class PiRuntimePreflightError(RuntimeError):
    """An installed Pi executable is unavailable or cannot complete its local process run."""


class PiInvocationTimeoutError(TimeoutError):
    """The bounded installed-Pi process did not finish before its episode deadline."""


class PiTranscriptError(ValueError):
    """An installed Pi JSON event stream cannot be represented as an agent episode."""


@dataclass(frozen=True)
class _PiInvocation:
    """The process arguments, input, environment, and root for one installed-Pi execution."""

    command: tuple[str, ...]
    environment: dict[str, str]
    input_text: str
    cwd: Path


class PiAgentRuntime:
    """Run installed Pi in local JSON-event mode through WMO's injected dependencies.

    The adapter configures Pi with an ephemeral custom provider that sends every model request to
    the model WMO supplied for this episode. Its explicit extension forwards task-visible tool
    calls to the supplied execute-only environment. This configuration prevents ambient Pi model
    selection. Its private working directory isolates evaluator inputs from caller project Pi
    settings and prompts. It is not a security sandbox for the installed Pi process.

    Args:
        executable: Name or path of the externally installed Pi executable.
        timeout_seconds: Positive upper bound for the installed Pi subprocess.
    """

    def __init__(
        self,
        *,
        executable: str = "pi",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not executable:
            raise ValueError("Pi executable must be a non-empty command or path")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Pi timeout_seconds must be a finite positive value")
        self._executable = executable
        self._timeout_seconds = timeout_seconds

    def preflight(self) -> None:
        """Confirm that WMO can call the configured installed Pi executable.

        Raises:
            PiRuntimePreflightError: The configured Pi executable cannot be found.
        """
        self._resolve_executable()

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        """Run Pi and normalize its final local JSON event stream to an episode.

        Args:
            task: Task and tool schemas for the installed Pi process.
            model: Candidate model supplied through WMO's canonical model-client contract.
            environment: Execute-only session supplied by the simulator.

        Returns:
            The canonical in-memory episode reconstructed from Pi's JSON events.

        Raises:
            PiRuntimePreflightError: Pi cannot be found or its local process exits unsuccessfully.
            PiInvocationTimeoutError: The installed Pi process exceeds its configured deadline.
            PiTranscriptError: Pi emitted an invalid or incomplete JSON event transcript.
        """
        executable = self._resolve_executable()
        with _PiBridge(task, model, environment) as bridge:
            output = _invoke_installed_pi(
                executable,
                bridge.invocation(executable, task.instruction),
                self._timeout_seconds,
            )
        return _episode_from_pi_events(output)

    def _resolve_executable(self) -> str:
        """Resolve the configured Pi command and raise an actionable local-install error."""
        executable_path = which(self._executable)
        if executable_path is None:
            raise PiRuntimePreflightError(
                "PiAgentRuntime could not find an installed Pi executable named "
                f"{self._executable!r}. Install Pi outside WMO, or configure the executable path."
            )
        return executable_path


class _PiBridge(AbstractContextManager["_PiBridge"]):
    """Bind one Pi process to one WMO model and execute-only environment session."""

    def __init__(
        self,
        task: TaskCase,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> None:
        self._task = task
        self._model = model
        self._environment = environment
        self._tools_by_name = {tool.name: tool for tool in task.tools}
        if len(self._tools_by_name) != len(task.tools):
            raise ValueError("Pi bridge requires task-visible tool names to be unique")
        self._model_context = copy_context()
        self._model_lock = threading.Lock()
        self._environment_context = copy_context()
        self._tool_lock = threading.Lock()
        self._temporary_directory: TemporaryDirectory[str] | None = None
        self._server: _PiBridgeHttpServer | None = None
        self._server_thread: threading.Thread | None = None

    def __enter__(self) -> _PiBridge:
        """Start the ephemeral local bridge and write Pi's per-invocation configuration."""
        temporary_directory = TemporaryDirectory(prefix="wmo-pi-")
        self._temporary_directory = temporary_directory
        root = Path(temporary_directory.name)
        config_directory = root / "agent"
        config_directory.mkdir()
        server = _PiBridgeHttpServer(("127.0.0.1", 0), self)
        self._server = server
        thread = threading.Thread(target=server.serve_forever, name="wmo-pi-bridge", daemon=True)
        self._server_thread = thread
        thread.start()
        bridge_url = f"http://127.0.0.1:{server.server_port}"
        (config_directory / "models.json").write_text(
            json.dumps(_pi_models_config(bridge_url), separators=(",", ":")), encoding="utf-8"
        )
        (root / "tools.json").write_text(
            json.dumps(_pi_tools_config(self._task.tools), separators=(",", ":")), encoding="utf-8"
        )
        (root / "wmo-pi-tools.mjs").write_text(_PI_EXTENSION_SOURCE, encoding="utf-8")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> bool:
        """Stop the bridge before removing its ephemeral configuration files."""
        server = self._server
        if server is not None:
            server.shutdown()
            server.server_close()
        thread = self._server_thread
        if thread is not None:
            thread.join()
        temporary_directory = self._temporary_directory
        if temporary_directory is not None:
            temporary_directory.cleanup()
        self._server = None
        self._server_thread = None
        self._temporary_directory = None
        return False

    def invocation(self, executable: str, instruction: str) -> _PiInvocation:
        """Build one Pi invocation that can use only this bridge's selected model and tools."""
        temporary_directory = self._temporary_directory
        server = self._server
        if temporary_directory is None or server is None:
            raise RuntimeError("Pi bridge invocation requested outside its active context")
        root = Path(temporary_directory.name)
        process_environment = os.environ.copy()
        process_environment.update(
            {
                "PI_CODING_AGENT_DIR": str(root / "agent"),
                "PI_OFFLINE": "1",
                "PI_TELEMETRY": "0",
                "PI_SKIP_VERSION_CHECK": "1",
                "WMO_PI_BRIDGE_URL": f"http://127.0.0.1:{server.server_port}",
                "WMO_PI_TOOLS_PATH": str(root / "tools.json"),
            }
        )
        return _PiInvocation(
            command=(
                executable,
                "--mode",
                "json",
                "--no-session",
                "--no-context-files",
                "--no-extensions",
                "-e",
                str(root / "wmo-pi-tools.mjs"),
                "--no-skills",
                "--no-prompt-templates",
                "--no-themes",
                "--offline",
                "--no-builtin-tools",
                "--provider",
                _WMO_PI_PROVIDER,
                "--model",
                _WMO_PI_MODEL,
            ),
            environment=process_environment,
            input_text=instruction,
            cwd=root,
        )

    def complete_model(self, payload: JsonObject) -> ModelResponse:
        """Convert one Pi OpenAI-compatible request and call the WMO-injected model client."""
        request = ModelRequest(
            messages=_model_messages_from_pi(payload),
            tools=self._task.tools,
        )
        with self._model_lock:
            return self._model_context.run(self._model.complete, request)

    def execute_tool(self, tool_name: str, payload: JsonObject) -> Observation:
        """Execute one Pi extension tool through WMO's supplied environment session."""
        if tool_name not in self._tools_by_name:
            raise _PiBridgeError(
                f"Pi requested tool {tool_name!r}, which is not visible to this task"
            )
        call_id = payload.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise _PiBridgeError("Pi tool bridge request requires a non-empty call_id")
        try:
            arguments = _JSON_OBJECT.validate_python(payload.get("arguments", {}))
        except ValidationError as exc:
            raise _PiBridgeError("Pi tool bridge request has non-object arguments") from exc
        with self._tool_lock:
            return self._environment_context.run(
                self._environment.execute,
                ToolCall(call_id=call_id, name=tool_name, arguments=arguments),
            )


class _PiBridgeHttpServer(ThreadingHTTPServer):
    """Threading HTTP server whose handler delegates only to one active Pi bridge."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], bridge: _PiBridge) -> None:
        super().__init__(address, _PiBridgeRequestHandler)
        self.bridge = bridge


class _PiBridgeRequestHandler(BaseHTTPRequestHandler):
    """Handle the two local HTTP endpoints that Pi's custom provider and extension require."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler hook name
        """Route one local Pi model or tool request without logging its contents."""
        try:
            payload = self._read_payload()
            parsed_path = urlparse(self.path)
            if parsed_path.path == "/v1/chat/completions":
                self._write_completion(payload)
                return
            prefix = "/tools/"
            if parsed_path.path.startswith(prefix):
                self._write_tool_result(unquote(parsed_path.path.removeprefix(prefix)), payload)
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "unknown Pi bridge endpoint"})
        except _PiBridgeError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception:  # noqa: BLE001 - bridge failures must not expose episode payloads
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Pi bridge execution failed"}
            )

    def log_message(self, format: str, *args: object) -> None:
        """Suppress HTTP request logging because bridge payloads are episode-visible data."""

    def _read_payload(self) -> JsonObject:
        """Read a JSON object body with an explicit content-length requirement."""
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise _PiBridgeError("Pi bridge request requires Content-Length")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise _PiBridgeError("Pi bridge request has an invalid Content-Length") from exc
        if length < 0:
            raise _PiBridgeError("Pi bridge request has a negative Content-Length")
        try:
            return _JSON_OBJECT.validate_json(self.rfile.read(length))
        except ValidationError as exc:
            raise _PiBridgeError("Pi bridge request body must be a JSON object") from exc

    def _write_completion(self, payload: JsonObject) -> None:
        """Return a model response in the OpenAI-compatible shape Pi's custom provider uses."""
        response = self._bridge.complete_model(payload)
        completion = chat_completion(
            _WMO_PI_MODEL, response.output, response, idempotency_key="wmo-pi-bridge"
        )
        if payload.get("stream") is True:
            body = "".join(chat_stream(completion)).encode("utf-8")
            self._write_body(HTTPStatus.OK, "text/event-stream", body)
            return
        self._write_body(
            HTTPStatus.OK,
            "application/json",
            completion.model_dump_json(exclude_none=True).encode("utf-8"),
        )

    def _write_tool_result(self, tool_name: str, payload: JsonObject) -> None:
        """Run an explicit task tool and return its canonical observation to Pi's extension."""
        observation = self._bridge.execute_tool(tool_name, payload)
        self._write_json(
            HTTPStatus.OK,
            {
                "content": observation.content,
                "is_error": observation.is_error,
                "metadata": observation.metadata,
            },
        )

    @property
    def _bridge(self) -> _PiBridge:
        """Return the bridge bound to this server's single active process invocation."""
        return cast(_PiBridgeHttpServer, self.server).bridge

    def _write_json(self, status: HTTPStatus, payload: JsonObject) -> None:
        """Write one compact JSON response with no cacheable transport state."""
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._write_body(status, "application/json", body)

    def _write_body(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        """Write one complete response body with no cacheable transport state."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class _PiBridgeError(ValueError):
    """A local conversion or execution request cannot cross the Pi bridge."""


def _pi_models_config(bridge_url: str) -> JsonObject:
    """Return Pi's ephemeral custom-provider configuration for one injected WMO model."""
    return {
        "providers": {
            _WMO_PI_PROVIDER: {
                "baseUrl": f"{bridge_url}/v1",
                "api": "openai-completions",
                "apiKey": "wmo-local-bridge",
                "compat": {
                    "maxTokensField": "max_tokens",
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "supportsUsageInStreaming": False,
                },
                "models": [
                    {
                        "id": _WMO_PI_MODEL,
                        "name": _WMO_PI_MODEL,
                        "reasoning": False,
                        "input": ["text"],
                        "contextWindow": 128000,
                        "maxTokens": 16384,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    }
                ],
            }
        }
    }


def _pi_tools_config(tools: tuple[ToolSchema, ...]) -> JsonObject:
    """Serialize exactly the task-visible tools for Pi's explicit bridge extension."""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
    }


def _invoke_installed_pi(
    executable: str,
    invocation: _PiInvocation,
    timeout_seconds: float,
) -> str:
    """Run one bounded installed-Pi JSON process with its per-episode binding configuration."""
    try:
        result = subprocess.run(
            invocation.command,
            capture_output=True,
            check=False,
            cwd=invocation.cwd,
            env=invocation.environment,
            input=invocation.input_text,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise PiInvocationTimeoutError(
            "PiAgentRuntime's installed Pi process exceeded its "
            f"{timeout_seconds:g}-second deadline"
        ) from exc
    except OSError as exc:
        raise PiRuntimePreflightError(
            f"PiAgentRuntime could not start installed Pi at {executable!r}: {type(exc).__name__}"
        ) from exc
    if result.returncode != 0:
        raise PiRuntimePreflightError(
            "PiAgentRuntime's installed Pi process exited with "
            f"status {result.returncode}. Check that the executable supports Pi JSON mode."
        )
    return result.stdout


def _model_messages_from_pi(payload: JsonObject) -> tuple[ModelMessage, ...]:
    """Map Pi's OpenAI-compatible request messages to WMO's canonical request shape.

    Args:
        payload: Parsed JSON body of one Pi chat-completions bridge request.

    Returns:
        Canonical model messages in Pi's request order.

    Raises:
        _PiBridgeError: The messages are absent, empty, or not valid OpenAI chat messages.
    """
    if not payload.get("messages"):
        raise _PiBridgeError("Pi model bridge request requires a non-empty messages list")
    try:
        return model_messages(_HTTP_MESSAGES.validate_python(payload["messages"]))
    except ValueError as exc:
        raise _PiBridgeError(f"Pi model bridge request has invalid messages: {exc}") from exc


def _episode_from_pi_events(output: str) -> AgentEpisode:
    """Translate Pi JSON mode's ordered local events into the WMO episode representation."""
    spans: list[RolloutSpan] = []
    final_action: AssistantAction | None = None
    failure: StructuredFailure | None = None
    last_event_time: datetime | None = None
    saw_agent_end = False

    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        event = _decode_event(line, line_number)
        event_type = event.get("type")
        if not isinstance(event_type, str):
            raise PiTranscriptError(f"Pi JSON event line {line_number} has no string type")
        if event_type == "message_end":
            message = _require_object(event, "message", line_number)
            span, action, message_failure = _message_span(message, line_number, last_event_time)
            spans.append(span)
            last_event_time = span.ended_at
            if action is not None:
                final_action = action
            if message_failure is not None:
                failure = message_failure
        elif event_type in {"tool_execution_start", "tool_execution_end"}:
            span = _tool_span(event, line_number, last_event_time)
            spans.append(span)
            last_event_time = span.ended_at
        elif event_type == "agent_end":
            saw_agent_end = True

    if not saw_agent_end:
        raise PiTranscriptError("Pi JSON transcript ended before an agent_end event")
    if failure is not None:
        return AgentEpisode(
            events=tuple(spans),
            final_action=final_action,
            stop_reason=StopReason.FAILURE,
            failure=failure,
        )
    return AgentEpisode(
        events=tuple(spans),
        final_action=final_action,
        stop_reason=StopReason.COMPLETED,
    )


def _decode_event(line: str, line_number: int) -> JsonObject:
    """Decode one Pi JSONL event while rejecting non-object payloads with a useful error."""
    try:
        raw_event = json.loads(line)
        return _JSON_OBJECT.validate_python(raw_event)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise PiTranscriptError(f"Pi JSON event line {line_number} is not a JSON object") from exc


def _require_object(event: JsonObject, field: str, line_number: int) -> JsonObject:
    """Read one required nested JSON object from a Pi event."""
    try:
        return _JSON_OBJECT.validate_python(event.get(field))
    except ValidationError as exc:
        raise PiTranscriptError(
            f"Pi JSON event line {line_number} needs an object {field!r} field"
        ) from exc


def _message_span(
    message: JsonObject,
    line_number: int,
    previous_time: datetime | None,
) -> tuple[RolloutSpan, AssistantAction | None, StructuredFailure | None]:
    """Convert one completed Pi message to an ordered message span and terminal evidence."""
    role = message.get("role")
    if not isinstance(role, str):
        raise PiTranscriptError(f"Pi JSON event line {line_number} message has no string role")
    text, tool_calls = _message_contents(message, line_number)
    timestamp = _event_timestamp(message, line_number, previous_time)
    payload: JsonObject = {
        "source": "installed-pi",
        "event": "message_end",
        "role": role,
    }
    if text is not None:
        payload["content"] = text
    action = None
    if role == "assistant" and (text is not None or tool_calls):
        action = AssistantAction(content=text, tool_calls=tool_calls)
    failure = _assistant_stop_failure(message, line_number) if role == "assistant" else None
    if failure is not None:
        payload["stop_reason"] = str(failure.details["pi_stop_reason"])
    return (
        RolloutSpan(
            span_id=f"pi-message-{line_number}",
            kind=RolloutEventKind.MESSAGE,
            started_at=timestamp,
            ended_at=timestamp,
            payload=payload,
            failure=failure,
        ),
        action,
        failure,
    )


def _assistant_stop_failure(message: JsonObject, line_number: int) -> StructuredFailure | None:
    """Map Pi's unsuccessful assistant stop reasons to explicit WMO terminal failures."""
    stop_reason = message.get("stopReason")
    if stop_reason == "error":
        return StructuredFailure(
            code=FailureCode.PROVIDER,
            message=f"installed Pi assistant event line {line_number} stopped with error",
            attribution=FailureAttribution.MODEL,
            details={"pi_stop_reason": stop_reason},
        )
    if stop_reason == "aborted":
        return StructuredFailure(
            code=FailureCode.CANCELLED,
            message=f"installed Pi assistant event line {line_number} was aborted",
            attribution=FailureAttribution.AGENT,
            details={"pi_stop_reason": stop_reason},
        )
    return None


def _tool_span(
    event: JsonObject,
    line_number: int,
    previous_time: datetime | None,
) -> RolloutSpan:
    """Convert one Pi tool lifecycle event to the corresponding WMO span."""
    event_type = event.get("type")
    tool_name = event.get("toolName")
    if not isinstance(event_type, str) or not isinstance(tool_name, str):
        raise PiTranscriptError(f"Pi JSON event line {line_number} has no string tool name")
    timestamp = _event_timestamp(event, line_number, previous_time)
    payload: JsonObject = {"source": "installed-pi", "event": event_type}
    if event_type == "tool_execution_end":
        is_error = event.get("isError")
        if isinstance(is_error, bool):
            payload["is_error"] = is_error
    return RolloutSpan(
        span_id=f"pi-tool-{line_number}",
        kind=(
            RolloutEventKind.TOOL_CALL
            if event_type == "tool_execution_start"
            else RolloutEventKind.OBSERVATION
        ),
        started_at=timestamp,
        ended_at=timestamp,
        payload=payload,
        tool_name=tool_name,
    )


def _message_contents(
    message: JsonObject, line_number: int
) -> tuple[str | None, tuple[ToolCall, ...]]:
    """Extract visible text and complete Pi tool calls from one completed assistant message."""
    raw_contents = message.get("content")
    if not isinstance(raw_contents, list):
        return None, ()
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for content in raw_contents:
        try:
            block = _JSON_OBJECT.validate_python(content)
        except ValidationError as exc:
            raise PiTranscriptError(
                f"Pi JSON event line {line_number} has a non-object message content block"
            ) from exc
        content_type = block.get("type")
        if content_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise PiTranscriptError(
                    f"Pi JSON event line {line_number} has a text block without string text"
                )
            text_parts.append(text)
        elif content_type == "toolCall":
            tool_calls.append(_tool_call(block, line_number))
    return ("".join(text_parts) if text_parts else None), tuple(tool_calls)


def _tool_call(block: JsonObject, line_number: int) -> ToolCall:
    """Convert one complete Pi tool-call content block to WMO's canonical action shape."""
    call_id = block.get("id")
    name = block.get("name")
    if not isinstance(call_id, str) or not isinstance(name, str):
        raise PiTranscriptError(f"Pi JSON event line {line_number} has an invalid Pi tool call")
    try:
        arguments = _JSON_OBJECT.validate_python(block.get("arguments", {}))
    except ValidationError as exc:
        raise PiTranscriptError(
            f"Pi JSON event line {line_number} tool call {name!r} has non-object arguments"
        ) from exc
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _event_timestamp(
    event: JsonObject,
    line_number: int,
    previous_time: datetime | None,
) -> datetime:
    """Return an ordered event timestamp without manufacturing wall-clock provenance.

    A supplied timezone-aware Pi timestamp or finite Unix-seconds or Unix-milliseconds value is
    used unless it would move ordering backwards. Numeric values with at least the magnitude of a
    plausible modern millisecond timestamp use milliseconds; smaller values use seconds. A
    missing timestamp becomes the Unix epoch plus its one-based JSONL line number in microseconds.
    That explicit synthetic policy keeps all-missing transcripts deterministic and source-ordered.
    """
    raw_timestamp = event.get("timestamp")
    if raw_timestamp is None:
        timestamp = _DETERMINISTIC_EVENT_EPOCH + timedelta(microseconds=line_number)
    elif isinstance(raw_timestamp, (int, float)) and not isinstance(raw_timestamp, bool):
        if isinstance(raw_timestamp, float) and not math.isfinite(raw_timestamp):
            raise PiTranscriptError(f"Pi JSON event line {line_number} has an invalid timestamp")
        unit = "milliseconds" if abs(raw_timestamp) >= _UNIX_MILLISECOND_MAGNITUDE else "seconds"
        try:
            timestamp = _DETERMINISTIC_EVENT_EPOCH + timedelta(**{unit: raw_timestamp})
        except OverflowError as exc:
            raise PiTranscriptError(
                f"Pi JSON event line {line_number} has an invalid timestamp"
            ) from exc
    elif not isinstance(raw_timestamp, str):
        raise PiTranscriptError(f"Pi JSON event line {line_number} has a non-string timestamp")
    else:
        try:
            timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PiTranscriptError(
                f"Pi JSON event line {line_number} has an invalid timestamp"
            ) from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise PiTranscriptError(
                f"Pi JSON event line {line_number} timestamp must include a timezone"
            )
    return max(timestamp, previous_time) if previous_time is not None else timestamp


_PI_EXTENSION_SOURCE = r"""import { readFile } from "node:fs/promises";
import { Type } from "typebox";

const bridgeUrl = process.env.WMO_PI_BRIDGE_URL;
const toolsPath = process.env.WMO_PI_TOOLS_PATH;

async function request(path, body) {
  if (!bridgeUrl) throw new Error("WMO_PI_BRIDGE_URL is required");
  const response = await fetch(`${bridgeUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error("WMO Pi bridge rejected the request");
  return payload;
}

export default async function (pi) {
  if (!toolsPath) throw new Error("WMO_PI_TOOLS_PATH is required");
  const config = JSON.parse(await readFile(toolsPath, "utf8"));
  for (const tool of config.tools) {
    pi.registerTool({
      name: tool.name,
      label: tool.name,
      description: tool.description,
      parameters: Type.Unsafe(tool.input_schema),
      async execute(toolCallId, params) {
        const result = await request(`/tools/${encodeURIComponent(tool.name)}`, {
          call_id: toolCallId,
          arguments: params,
        });
        return {
          content: [{ type: "text", text: result.content }],
          details: result.metadata,
          isError: result.is_error,
        };
      },
    });
  }
}
"""

"""Adapter from an installed Pi JSON-event process to the WMO agent contract."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from shutil import which

from pydantic import TypeAdapter, ValidationError

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import AssistantAction, ToolCall
from wmo.common.rollouts import RolloutEventKind, RolloutSpan, StopReason
from wmo.common.tasks import TaskCase
from wmo.runtime.agents.interface import AgentEpisode
from wmo.runtime.environments import EnvironmentSession

type PiTranscriptRunner = Callable[[TaskCase, object, EnvironmentSession], JsonObject]
"""A deterministic test seam that returns a WMO-shaped Pi episode transcript."""

_JSON_OBJECT = TypeAdapter(JsonObject)
_PI_JSON_OPTIONS = (
    "--mode",
    "json",
    "--no-session",
    "--no-context-files",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--offline",
    "--no-tools",
)


class PiRuntimePreflightError(RuntimeError):
    """An installed Pi executable is unavailable or could not complete its local process run."""


class PiTranscriptError(ValueError):
    """An installed Pi JSON event stream cannot be represented as an agent episode."""


class PiAgentRuntime:
    """Run installed Pi in its local JSON-event mode and return a WMO episode.

    Pi's executable remains an external dependency. The default invokes its JSON-event mode with
    sessions, context discovery, built-in tools, and Pi startup network activity disabled. A
    supplied ``transcript_runner`` remains a deterministic test seam for WMO-shaped transcripts.

    Args:
        executable: Name or path of the externally installed Pi executable.
        transcript_runner: Deterministic test bridge returning a JSON-compatible WMO transcript.
    """

    def __init__(
        self,
        *,
        executable: str = "pi",
        transcript_runner: PiTranscriptRunner | None = None,
    ) -> None:
        if not executable:
            raise ValueError("Pi executable must be a non-empty command or path")
        self._executable = executable
        self._transcript_runner = transcript_runner

    def preflight(self) -> None:
        """Confirm that WMO can call the configured installed Pi executable.

        The deterministic transcript seam is deliberately exempt because it invokes no process.

        Raises:
            PiRuntimePreflightError: The configured Pi executable cannot be found.
        """
        if self._transcript_runner is None:
            self._resolve_executable()

    def run(
        self,
        task: TaskCase,
        *,
        model: object,  # W3 restack: replace with the canonical ModelClient.
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        """Run Pi and normalize its final local JSON event stream to an episode.

        Args:
            task: Task and tool schemas for the installed Pi process.
            model: Candidate model supplied by WMO's pending common model-client contract.
            environment: Execute-only session supplied by the simulator.

        Returns:
            The canonical in-memory episode reconstructed from Pi's JSON events.

        Raises:
            PiRuntimePreflightError: Pi cannot be found or its local process exits unsuccessfully.
            PiTranscriptError: Pi emitted an invalid or incomplete JSON event transcript.
        """
        runner = self._transcript_runner
        if runner is not None:
            return _episode_from_wmo_transcript(runner(task, model, environment))
        executable = self._resolve_executable()
        return _episode_from_pi_events(_invoke_installed_pi(executable, task))

    def _resolve_executable(self) -> str:
        """Resolve the configured Pi command and raise an actionable local-install error."""
        executable_path = which(self._executable)
        if executable_path is None:
            raise PiRuntimePreflightError(
                "PiAgentRuntime could not find an installed Pi executable named "
                f"{self._executable!r}. Install Pi outside WMO, or configure the executable path."
            )
        return executable_path


def _invoke_installed_pi(executable: str, task: TaskCase) -> str:
    """Run one isolated installed-Pi JSON process without triggering Pi startup network work."""
    try:
        result = subprocess.run(
            (executable, *_PI_JSON_OPTIONS, task.instruction),
            capture_output=True,
            check=False,
            text=True,
        )
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


def _episode_from_wmo_transcript(transcript: JsonObject) -> AgentEpisode:
    """Validate a deterministic WMO-shaped test transcript returned by an injected bridge."""
    try:
        return AgentEpisode.model_validate(transcript)
    except ValidationError as exc:
        raise PiTranscriptError(
            "Pi transcript does not satisfy the WMO AgentEpisode contract. "
            "Update the deterministic Pi bridge to emit ordered events and terminal state."
        ) from exc


def _episode_from_pi_events(output: str) -> AgentEpisode:
    """Translate Pi JSON mode's ordered local events into the WMO episode representation."""
    spans: list[RolloutSpan] = []
    final_action: AssistantAction | None = None
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
            span, action = _message_span(message, line_number, last_event_time)
            spans.append(span)
            last_event_time = span.ended_at
            if action is not None:
                final_action = action
        elif event_type in {"tool_execution_start", "tool_execution_end"}:
            span = _tool_span(event, line_number, last_event_time)
            spans.append(span)
            last_event_time = span.ended_at
        elif event_type == "agent_end":
            saw_agent_end = True

    if not saw_agent_end:
        raise PiTranscriptError("Pi JSON transcript ended before an agent_end event")
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
) -> tuple[RolloutSpan, AssistantAction | None]:
    """Convert one completed Pi message to an ordered message span and possible final action."""
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
    return (
        RolloutSpan(
            span_id=f"pi-message-{line_number}",
            kind=RolloutEventKind.MESSAGE,
            started_at=timestamp,
            ended_at=timestamp,
            payload=payload,
        ),
        action,
    )


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
    """Return an ordered Pi timestamp, falling back to the local process clock."""
    raw_timestamp = event.get("timestamp")
    if raw_timestamp is None:
        timestamp = datetime.now(UTC)
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

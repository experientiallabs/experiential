"""Import coding-agent session files as canonical trace evidence.

The ``agent-sessions`` source reads the session files coding agents write and Hugging Face
"agent traces" datasets publish: Claude Code transcripts, pi sessions, and one-record-per-session
exports (see :mod:`exp.common.traces.ingest.agent_session_formats`). The path may be one session
file or a directory, whose ``*.jsonl`` files are read recursively in path order.

    result = load_agent_sessions(Path("datasets/claude-code-sessions"))

Each session becomes one trace:

- ``task`` is the first typed user prompt; harness-injected text (command wrappers, environment
  context, ``AGENTS.md`` preambles) is request-visible context, not the task.
- ``initial_context`` holds the system and injected messages before the first prompt as
  ``instruction_messages``. Working directory, branch, titles, and skipped-record counts are
  trace metadata (``exp.trace.metadata``), never request context.
- Every assistant turn is a model span whose input is a prefix of the shared message history.
  A turn requesting tools yields one model span per call; each tool result is a tool span
  paired with its call by recorded call identity, with ``gen_ai.tool.is_error`` from the
  session.
- Model identity is retained only when the session records both provider and model (pi);
  a model named without a provider (Claude Code, exports) stays ``gen_ai.request.model``.
- A session whose last assistant turn failed has a failure outcome; a session with only earlier
  failed turns has an ``unknown`` outcome; otherwise the outcome is not derivable and is omitted.
- Subagent (sidechain) transcripts are separate traces whose conversation identity is their
  parent session's, so lineage grouping keeps a session and its subagents together.

Trace identity derives from the dataset label, the file path below the input, and the session
key, so re-importing the same file is idempotent. A session key repeated within one file (an
export holding several records with one ``id``) is numbered by occurrence (``KEY~2``). The
dataset label is the caller's ``source_id`` when given, otherwise the input directory's name
(or a file's parent directory).

Every string is redacted with :class:`exp.common.traces.ingest.session_redaction.SessionRedactor`
before it enters a trace, and each trace records the policy and its replacement counts under
``exp.trace.metadata.redaction``. Files are read as data only; nothing in them is executed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject, SourceIdentity
from exp.common.core.text import normalize_durable_text
from exp.common.traces.ingest.agent_session_formats import (
    AgentSession,
    SessionEvent,
    SessionToolCall,
    detect_session_format,
    read_sessions,
)
from exp.common.traces.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult
from exp.common.traces.ingest.session_redaction import SessionRedactor
from exp.common.traces.ingest.trace_extensions import (
    OUTCOME_FAILURE_CODE_KEY,
    OUTCOME_FAILURE_MESSAGE_KEY,
    OUTCOME_STATUS_KEY,
    REQUEST_CONTEXT_KEY,
    REQUEST_TOOLS_KEY,
)
from exp.common.traces.ingest.vendor_observations import (
    VendorModelIdentity,
    VendorObservation,
    VendorToolCall,
)
from exp.common.traces.ingest.vendor_records import VendorTraceFormatError, json_text
from exp.common.traces.ingest.vendor_trace import build_vendor_traces

AGENT_SESSIONS_VENDOR = "agent-sessions"
"""Canonical source name and span vendor label of imported agent sessions."""

ADJUSTED_TIME_ATTRIBUTE = "exp.source.time.adjusted"
"""Span attribute marking a recorded instant moved forward to keep session order strict."""

RECORDED_AT_ATTRIBUTE = "exp.source.recorded_at"
"""Span attribute holding the recorded instant of an adjusted span, when one was recorded."""

_INFERRED_DESCRIPTION = "Schema inferred from recorded calls; the session declares no definition."
_UNSAFE_TEXT = re.compile("[\x00\ud800-\udfff]")
_SYNTHETIC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_STEP = timedelta(microseconds=1)


@dataclass
class AgentSessionReader:
    """Read one session file or directory into declared vendor observations.

    The constructor fixes the file inventory and the source identity, whose digest covers every
    selected file's path and bytes. :meth:`observations` then reads one file at a time, so a
    caller can stage observations without holding the corpus in memory.

    Attributes:
        path: Session file or directory.
        source: Immutable identity of the selected files.
        dataset: Dataset label used in trace identity.
        files: Selected files with their labels relative to the input, in path order.
        issues: Files, lines, and sessions excluded so far, in read order.
        sessions_by_file: Sessions read per file label, filled while reading.
    """

    path: Path
    source: SourceIdentity
    dataset: str
    files: tuple[tuple[Path, str], ...]
    issues: list[TraceNormalizationIssue] = field(default_factory=list)
    sessions_by_file: dict[str, int] = field(default_factory=dict)

    @classmethod
    def open(cls, path: Path, *, source_id: str | None = None) -> AgentSessionReader:
        """Select the session files of one input path and fix its source identity.

        Args:
            path: A session file, or a directory whose ``*.jsonl`` files are read recursively.
            source_id: Optional durable source label, also used as the dataset label.

        Returns:
            A reader over the selected files.

        Raises:
            VendorTraceFormatError: The path is missing, unreadable, or selects no file.
        """
        if path.is_dir():
            selected = sorted(item for item in path.rglob("*.jsonl") if item.is_file())
            files = tuple((item, item.relative_to(path).as_posix()) for item in selected)
            dataset = path.resolve().name
        elif path.is_file():
            files = ((path, path.name),)
            dataset = path.resolve().parent.name
        else:
            raise VendorTraceFormatError(
                f"agent session path not found: {path}; pass a session file or a directory"
            )
        if not files:
            raise VendorTraceFormatError(f"agent session directory has no *.jsonl files: {path}")
        digest = hashlib.sha256()
        for item, label in files:
            digest.update(label.encode("utf-8") + b"\0" + _file_sha256(item).encode() + b"\n")
        source = SourceIdentity(
            kind="file",
            source_id=source_id or f"{AGENT_SESSIONS_VENDOR}:{path}",
            sha256=digest.hexdigest(),
        )
        return cls(
            path=path, source=source, dataset=source_id or dataset or "sessions", files=files
        )

    def observations(self) -> Iterator[VendorObservation]:
        """Yield every session's observations, file by file, with global source ordinals.

        Yields:
            Declared observations in source order. Excluded files, lines, and sessions are
            appended to :attr:`issues` as they are found.
        """
        ordinal = 0
        for file_path, label in self.files:
            repeats: dict[str, int] = {}
            for session in self._file_sessions(file_path, label):
                repeats[session.key] = repeats.get(session.key, 0) + 1
                suffix = "" if repeats[session.key] == 1 else f"~{repeats[session.key]}"
                trace_key = f"{self.dataset}/{label}#{session.key}{suffix}"
                try:
                    emitted = session_observations(
                        session,
                        trace_key=trace_key,
                        conversation_key=f"{self.dataset}#{session.conversation_id or session.key}",
                        ordinal=ordinal,
                        provenance={"dataset": self.dataset, "file": label},
                    )
                except VendorTraceFormatError as exc:
                    self.issues.append(TraceNormalizationIssue(f"session-{trace_key}", str(exc)))
                    continue
                ordinal += len(emitted)
                yield from emitted

    def _file_sessions(self, file_path: Path, label: str) -> tuple[AgentSession, ...]:
        """Decode one file and read its sessions, retaining every exclusion.

        Args:
            file_path: Session file.
            label: File label relative to the input.

        Returns:
            The file's sessions, empty when the file is excluded.
        """
        records: list[JsonObject] = []
        try:
            with file_path.open("rb") as stream:
                for number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    record = _decode_line(line)
                    if isinstance(record, dict):
                        records.append(record)
                    else:
                        reason = "invalid JSON" if record is None else "not a JSON object"
                        self.issues.append(
                            TraceNormalizationIssue(f"{label}:line-{number}", f"{reason} record")
                        )
        except OSError as exc:
            self.issues.append(TraceNormalizationIssue(f"file-{label}", f"unreadable: {exc}"))
            return ()
        session_format = detect_session_format(records)
        if session_format is None:
            self.issues.append(
                TraceNormalizationIssue(
                    f"file-{label}",
                    "not a recognized agent session file (Claude Code, pi, or session export)",
                )
            )
            return ()
        try:
            sessions = read_sessions(session_format, records)
        except VendorTraceFormatError as exc:
            self.issues.append(TraceNormalizationIssue(f"file-{label}", str(exc)))
            return ()
        self.sessions_by_file[label] = len(sessions)
        return sessions


def load_agent_sessions(path: Path, *, source_id: str | None = None) -> TraceNormalizationResult:
    """Normalize one agent session file or directory into canonical traces.

    Args:
        path: A session file, or a directory whose ``*.jsonl`` files are read recursively.
        source_id: Optional durable source label that replaces the local path in provenance
            and names the dataset in trace identity.

    Returns:
        Canonical traces and every retained file, line, and session exclusion.

    Raises:
        VendorTraceFormatError: The path is missing, unreadable, or selects no file.
    """
    reader = AgentSessionReader.open(path, source_id=source_id)
    observations = tuple(reader.observations())
    return build_vendor_traces(
        observations,
        vendor=AGENT_SESSIONS_VENDOR,
        source=reader.source,
        initial_issues=reader.issues,
    )


def session_observations(
    session: AgentSession,
    *,
    trace_key: str,
    conversation_key: str,
    ordinal: int,
    provenance: JsonObject,
) -> tuple[VendorObservation, ...]:
    """Convert one redacted neutral session into declared vendor observations.

    Args:
        session: Neutral session read from a file.
        trace_key: Source trace key: dataset, file label, and session key.
        conversation_key: Conversation identity shared with the session's subagents.
        ordinal: Source order position of the first observation.
        provenance: Dataset and file labels recorded in trace metadata.

    Returns:
        Observations in session order; the first carries the task, history, and trace context.

    Raises:
        VendorTraceFormatError: The session has no user text or no assistant turn.
    """
    redactor = SessionRedactor()
    events = tuple(_redacted_event(event, redactor) for event in session.events)
    task = next((event.text for event in events if event.kind == "user" and event.prompt), None)
    if task is None:
        task = next((event.text for event in events if event.kind == "user"), None)
    if task is None or not task.strip():
        raise VendorTraceFormatError("session has no user prompt")
    if not any(event.kind == "assistant" for event in events):
        raise VendorTraceFormatError("session has no assistant turn")
    calls = {call.call_id: call for event in events for call in event.tool_calls if call.call_id}
    times = _span_times(session, events)
    history: list[JsonObject] = []
    instructions: list[JsonValue] = []
    observations: list[VendorObservation] = []
    orphans = 0
    prompted = False
    for index, event in enumerate(events):
        if event.kind in ("system", "user"):
            message: JsonObject = {"role": event.kind, "content": event.text}
            if not prompted and not event.prompt:
                instructions.append(message)
            prompted = prompted or event.prompt
            history.append(message)
            continue
        started, adjusted = times[index]
        base = _Base(
            trace_key=trace_key,
            source_span_id=event.source_id,
            ordinal=ordinal + len(observations),
            started_at=started,
            synthetic=session.synthetic_time,
            recorded=None if session.synthetic_time or not adjusted else event.timestamp,
        )
        if event.kind == "assistant":
            observations.append(_model_observation(event, base, len(history)))
            history.append(_assistant_message(event))
            continue
        call = calls.get(event.tool_call_id) if event.tool_call_id else None
        name = event.tool_name or (call.name if call is not None else None)
        if name is None:
            orphans += 1
            continue
        observations.append(_tool_observation(event, base, name, call))
        history.append(
            {
                "role": "tool",
                "tool_call_id": event.tool_call_id,
                "name": name,
                "content": event.text,
            }
        )
    metadata = redactor.json(
        {
            "source_format": session.format,
            **provenance,
            **session.metadata,
            "tool_schemas": _tool_schema_origin(session, events),
        }
    )
    assert isinstance(metadata, dict)
    if orphans:
        metadata["unpaired_tool_results"] = orphans
    tools = _tools(session, events, redactor)
    metadata["redaction"] = redactor.provenance()
    first = observations[0]
    extensions: JsonObject = {
        "exp.conversation.id": conversation_key,
        "exp.trace.metadata": metadata,
        REQUEST_TOOLS_KEY: tools,
        **_outcome(events),
    }
    if instructions:
        extensions[REQUEST_CONTEXT_KEY] = {"instruction_messages": instructions}
    observations[0] = replace(
        first,
        request_text=task,
        message_history=tuple(history),
        extensions={**first.extensions, **extensions},
    )
    return tuple(observations)


@dataclass(frozen=True)
class _Base:
    """Shared identity and timing of one emitted observation.

    Attributes:
        trace_key: Source trace key.
        source_span_id: Recorded event identity.
        ordinal: Source order position.
        started_at: Strictly ordered span instant.
        synthetic: Whether the instant is positional rather than recorded.
        recorded: Recorded instant of an adjusted span, when one was recorded.
    """

    trace_key: str
    source_span_id: str
    ordinal: int
    started_at: datetime
    synthetic: bool
    recorded: datetime | None


def _model_observation(event: SessionEvent, base: _Base, prefix: int) -> VendorObservation:
    """Declare one assistant turn as a model observation over a history prefix.

    Args:
        event: Redacted assistant event.
        base: Identity and timing.
        prefix: Number of history messages that form the turn's input.

    Returns:
        A model observation with its tool calls, model evidence, usage, and failure.
    """
    declared = _time_attributes(base)
    model = None
    if event.model is not None and event.provider is not None:
        model = VendorModelIdentity(provider=event.provider, model_id=event.model)
    elif event.model is not None:
        declared["gen_ai.request.model"] = event.model
    return VendorObservation(
        source_trace_id=base.trace_key,
        source_span_id=base.source_span_id,
        ordinal=base.ordinal,
        started_at=base.started_at,
        ended_at=base.started_at,
        kind="model",
        input_history_prefix=prefix,
        completion_text=event.text or None,
        tool_calls=tuple(
            VendorToolCall(
                name=call.name, arguments=json_text(call.arguments), call_id=call.call_id
            )
            for call in event.tool_calls
        ),
        model=model,
        usage=event.usage,
        failure_message=event.error,
        declared_attributes=declared,
        synthetic_time=base.synthetic,
    )


def _tool_observation(
    event: SessionEvent, base: _Base, name: str, call: SessionToolCall | None
) -> VendorObservation:
    """Declare one tool result, carrying its paired call's arguments.

    Args:
        event: Redacted tool-result event.
        base: Identity and timing.
        name: Resolved tool name.
        call: The paired call, when the session recorded it.

    Returns:
        A tool-result observation with the recorded error flag.
    """
    declared = _time_attributes(base)
    declared["gen_ai.tool.is_error"] = event.is_error
    return VendorObservation(
        source_trace_id=base.trace_key,
        source_span_id=base.source_span_id,
        ordinal=base.ordinal,
        started_at=base.started_at,
        ended_at=base.started_at,
        kind="tool_result",
        tool_name=name,
        tool_arguments=None if call is None else json_text(call.arguments),
        tool_message=event.text,
        tool_call_id=event.tool_call_id,
        declared_attributes=declared,
        synthetic_time=base.synthetic,
    )


def _time_attributes(base: _Base) -> JsonObject:
    """Return the adjustment attributes of one observation.

    Args:
        base: Identity and timing.

    Returns:
        The adjusted flag and recorded instant when the span instant was moved, else empty.
    """
    if base.recorded is None:
        return {}
    return {ADJUSTED_TIME_ATTRIBUTE: True, RECORDED_AT_ATTRIBUTE: base.recorded.isoformat()}


def _assistant_message(event: SessionEvent) -> JsonObject:
    """Render one assistant turn as a history message.

    Args:
        event: Redacted assistant event.

    Returns:
        An assistant message with its text and OpenAI-style tool calls.
    """
    message: JsonObject = {"role": "assistant", "content": event.text}
    if event.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": json_text(call.arguments)},
            }
            for call in event.tool_calls
        ]
    return message


def _span_times(
    session: AgentSession, events: Sequence[SessionEvent]
) -> dict[int, tuple[datetime, bool]]:
    """Assign strictly increasing instants to the events that become spans.

    Recorded instants are kept when they advance; an absent or non-advancing instant moves one
    microsecond past the previous span and is marked adjusted. A session with positional
    order, or with no recorded instant at all, uses positional instants from its start.

    Args:
        session: Neutral session.
        events: Its redacted events.

    Returns:
        Span instant and adjustment flag by event position.
    """
    positional = session.synthetic_time or all(event.timestamp is None for event in events)
    start = session.started_at or next(
        (event.timestamp for event in events if event.timestamp is not None), _SYNTHETIC_EPOCH
    )
    cursor: datetime | None = None
    times: dict[int, tuple[datetime, bool]] = {}
    for index, event in enumerate(events):
        if event.kind not in ("assistant", "tool_result"):
            continue
        if positional:
            instant = start + _STEP * index
            times[index] = (instant, False)
            continue
        recorded = event.timestamp
        if recorded is not None and (cursor is None or recorded > cursor):
            instant, adjusted = recorded, False
        else:
            instant, adjusted = (cursor or start) + _STEP, True
        times[index] = (instant, adjusted)
        cursor = instant
    return times


def _outcome(events: Sequence[SessionEvent]) -> JsonObject:
    """Derive the outcome extension from recorded assistant failures.

    Args:
        events: Redacted session events.

    Returns:
        A failure outcome when the last assistant turn failed, an ``unknown`` outcome when only
        earlier turns failed, and no extension otherwise.
    """
    turns = [event for event in events if event.kind == "assistant"]
    if turns and turns[-1].error is not None:
        return {
            OUTCOME_STATUS_KEY: "failure",
            OUTCOME_FAILURE_CODE_KEY: "provider",
            OUTCOME_FAILURE_MESSAGE_KEY: normalize_durable_text(turns[-1].error[:2000]),
        }
    if any(event.error is not None for event in turns):
        return {OUTCOME_STATUS_KEY: "unknown"}
    return {}


def _tools(
    session: AgentSession, events: Sequence[SessionEvent], redactor: SessionRedactor
) -> list[JsonValue]:
    """Return declared tool definitions plus inferred schemas for undeclared called tools.

    Args:
        session: Neutral session with its declared definitions.
        events: Redacted session events.
        redactor: Redactor applied to declared definitions.

    Returns:
        Tool definitions in the shape ``exp.request.tools`` accepts, one per name.
    """
    tools: dict[str, JsonValue] = {}
    for raw in session.declared_tools:
        definition = _declared_definition(raw)
        if definition is not None:
            tools[str(definition["name"])] = redactor.json(definition)
    observed: dict[str, dict[str, set[str]]] = {}
    for event in events:
        for call in event.tool_calls:
            properties = observed.setdefault(call.name, {})
            if isinstance(call.arguments, dict):
                for key, value in call.arguments.items():
                    properties.setdefault(key, set()).add(_json_type(value))
    for name in sorted(set(observed) - set(tools)):
        tools[name] = {
            "name": name,
            "description": _INFERRED_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    key: {"type": sorted(kinds)[0] if len(kinds) == 1 else sorted(kinds)}
                    for key, kinds in sorted(observed[name].items())
                },
            },
        }
    return [tools[name] for name in sorted(tools)]


def _declared_definition(raw: JsonObject) -> JsonObject | None:
    """Normalize one declared tool definition, or drop one without a usable schema.

    Args:
        raw: OpenAI function-style or flat tool definition.

    Returns:
        A flat ``{name, description, input_schema}`` definition, or ``None``.
    """
    candidate = raw.get("function") if raw.get("type") == "function" else raw
    if not isinstance(candidate, dict):
        return None
    name = candidate.get("name")
    schema = candidate.get("parameters", candidate.get("input_schema"))
    if not isinstance(name, str) or not name.strip() or not isinstance(schema, dict):
        return None
    description = candidate.get("description")
    return {
        "name": name.strip(),
        "description": description
        if isinstance(description, str) and description.strip()
        else "No description captured.",
        "input_schema": schema,
    }


def _tool_schema_origin(session: AgentSession, events: Sequence[SessionEvent]) -> str:
    """Describe where the trace's tool schemas come from.

    Args:
        session: Neutral session.
        events: Its events.

    Returns:
        ``declared``, ``inferred``, ``mixed``, or ``none``.
    """
    declared = {
        str(item["name"])
        for raw in session.declared_tools
        if (item := _declared_definition(raw)) is not None
    }
    called = {call.name for event in events for call in event.tool_calls}
    if not declared and not called:
        return "none"
    if not declared:
        return "inferred"
    return "declared" if called <= declared else "mixed"


def _json_type(value: JsonValue) -> str:
    """Return the JSON Schema type name of one recorded argument value.

    Args:
        value: Recorded argument value.

    Returns:
        ``object``, ``array``, ``string``, ``integer``, ``number``, ``boolean``, or ``null``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


def _redacted_event(event: SessionEvent, redactor: SessionRedactor) -> SessionEvent:
    """Redact and normalize every string an event carries.

    Args:
        event: Neutral event read from a file.
        redactor: The session's redactor, which counts replacements.

    Returns:
        The event with redacted, durable text, tool arguments, and error text.
    """
    calls = tuple(
        replace(call, arguments=_durable_json(redactor.json(call.arguments)))
        for call in event.tool_calls
    )
    return replace(
        event,
        text=_durable(redactor.text(event.text)),
        tool_calls=calls,
        error=None if event.error is None else _durable(redactor.text(event.error)),
    )


def _durable(value: str) -> str:
    """Replace NULs and lone surrogates only when the text contains them.

    Args:
        value: Decoded text.

    Returns:
        Text that every persistence boundary accepts.
    """
    return normalize_durable_text(value) if _UNSAFE_TEXT.search(value) else value


def _durable_json(value: JsonValue) -> JsonValue:
    """Apply :func:`_durable` to every string leaf of a JSON value.

    Args:
        value: JSON value.

    Returns:
        The value with durable string leaves.
    """
    if isinstance(value, str):
        return _durable(value)
    if isinstance(value, list):
        return [_durable_json(item) for item in value]
    if isinstance(value, dict):
        return {_durable(key): _durable_json(item) for key, item in value.items()}
    return value


def _decode_line(line: bytes) -> JsonValue | None:
    """Decode one JSONL line, mapping non-standard constants such as ``NaN`` to null.

    Args:
        line: Raw line bytes.

    Returns:
        The decoded value, or ``None`` when the line is not valid UTF-8 JSON.
    """
    try:
        return json.loads(line.decode("utf-8"), parse_constant=lambda _: None)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _file_sha256(path: Path) -> str:
    """Hash one file's bytes in bounded chunks.

    Args:
        path: File to hash.

    Returns:
        Hex SHA-256 digest.

    Raises:
        VendorTraceFormatError: The file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VendorTraceFormatError(f"cannot read agent session file {path}: {exc}") from exc
    return digest.hexdigest()

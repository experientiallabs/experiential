"""Read coding-agent session files into one format-neutral event sequence per session.

Coding agents persist each session as JSONL, and Hugging Face "agent traces" datasets publish
those files as recorded. Three wire formats are supported:

- Claude Code transcripts: ``user``, ``assistant``, ``system``, and bookkeeping records sharing a
  ``sessionId``, with Anthropic-style content blocks (``text``, ``thinking``, ``tool_use``,
  ``tool_result``). One API response may be split across consecutive ``assistant`` records that
  repeat ``message.id``; they are merged into one turn. Subagent records (``isSidechain``) form
  their own session keyed by ``agentId``. Several sessions may share one file.
- Pi sessions: a ``session`` header followed by ``message``, ``model_change``, and extension
  entries linked by ``id`` and ``parentId``. When every message entry links to its parent, only
  the active branch (the path from the last message to the root) is read.
- One-record-per-session exports: ``{id, createdAt, projectPath, source, messages}`` with flat
  string messages whose tool calls are declared by ``toolName`` and ``toolCallId``. Their
  timestamps are not trusted for order, because such exports fuzz them; message order is used.
  A repeated tool call or tool result for one call identity is dropped and counted.

Every reader returns :class:`AgentSession` values that hold only request-visible messages, model
turns, and tool results. Bookkeeping records are counted in the session metadata, never
converted. Reading is pure data handling: nothing in a session is executed.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.traces.ingest.vendor_observations import VendorTokenUsage
from exp.common.traces.ingest.vendor_records import (
    VendorTraceFormatError,
    json_text,
    source_timestamp,
)

SessionFormat = Literal["claude-code", "pi", "session-export"]
EventKind = Literal["system", "user", "assistant", "tool_result"]

_INJECTED_PREFIXES = (
    "<task-notification>",
    "[Request interrupted",
    "<local-command-",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<system-reminder>",
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",
    "<user-prompt-submit-hook>",
    "Caveat:",
    "# AGENTS.md instructions",
    "<environment_context>",
    "<user_instructions>",
    "<INSTRUCTIONS>",
    "<permissions instructions>",
    "<turn_aborted>",
)
_CLAUDE_MESSAGE_TYPES = frozenset({"user", "assistant"})
_EXPORT_TOOL_RESULT_ROLES = frozenset({"tool-result", "tool_result", "toolresult", "tool"})
_CALL_PREFIX = "[Tool call: "


@dataclass(frozen=True)
class SessionToolCall:
    """One tool call requested by an assistant turn.

    Attributes:
        call_id: Recorded call identity, when the session declares one.
        name: Recorded tool name.
        arguments: Recorded arguments as a JSON value.
    """

    call_id: str | None
    name: str
    arguments: JsonValue


@dataclass(frozen=True)
class SessionEvent:
    """One request-visible message, model turn, or tool result in session order.

    Attributes:
        kind: Event role. ``user`` events include injected context that is not a typed prompt.
        source_id: Recorded record identity, or a positional label when none is recorded.
        timestamp: Recorded instant, when the session declares one.
        text: Visible text: the prompt, instruction, completion, or tool output.
        prompt: Whether a ``user`` event is a typed request rather than injected context.
        tool_calls: Calls requested by an ``assistant`` event.
        tool_call_id: Call identity answered by a ``tool_result`` event.
        tool_name: Tool name of a ``tool_result`` event, when known.
        is_error: Whether a ``tool_result`` event reports a tool failure.
        model: Recorded model name of an ``assistant`` event.
        provider: Recorded provider of an ``assistant`` event.
        usage: Recorded complete token accounting of an ``assistant`` event.
        error: Recorded error text of a failed ``assistant`` event.
    """

    kind: EventKind
    source_id: str
    timestamp: datetime | None
    text: str
    prompt: bool = False
    tool_calls: tuple[SessionToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    model: str | None = None
    provider: str | None = None
    usage: VendorTokenUsage | None = None
    error: str | None = None


@dataclass(frozen=True)
class AgentSession:
    """One coding-agent session in format-neutral form.

    Attributes:
        key: Session identity inside its file, ``sessionId`` plus an agent suffix for subagents.
        format: Wire format the session was read from.
        conversation_id: Parent conversation shared by a session and its subagents.
        events: Request-visible events in session order.
        declared_tools: Tool definitions the session itself declares, in declaration order.
        metadata: Recorded working directory, branch, titles, and skipped-record counts.
        synthetic_time: Whether recorded timestamps are not usable and order is positional.
        started_at: Recorded session start instant, when any.
    """

    key: str
    format: SessionFormat
    conversation_id: str | None
    events: tuple[SessionEvent, ...]
    declared_tools: tuple[JsonObject, ...] = ()
    metadata: JsonObject = field(default_factory=dict)
    synthetic_time: bool = False
    started_at: datetime | None = None


def detect_session_format(records: Iterable[JsonObject]) -> SessionFormat | None:
    """Recognize a session file's wire format from its leading records.

    Args:
        records: Decoded records in file order; only the first few are inspected.

    Returns:
        The recognized format, or ``None`` when the records are not an agent session file.
    """
    for index, record in enumerate(records):
        if index >= 20:
            break
        kind = record.get("type")
        if kind == "session" and isinstance(record.get("id"), str):
            return "pi"
        if isinstance(record.get("sessionId"), str) and isinstance(kind, str):
            return "claude-code"
        if isinstance(record.get("messages"), list) and kind is None:
            if any(key in record for key in ("projectPath", "createdAt", "source")):
                return "session-export"
    return None


def read_sessions(
    session_format: SessionFormat, records: Sequence[JsonObject]
) -> tuple[AgentSession, ...]:
    """Read every session declared by one file's records.

    Args:
        session_format: Format recognized by :func:`detect_session_format`.
        records: Decoded file records in order.

    Returns:
        Sessions in first-appearance order.

    Raises:
        VendorTraceFormatError: A record of the declared format is malformed.
    """
    if session_format == "claude-code":
        return _claude_sessions(records)
    if session_format == "pi":
        return (_pi_session(records),)
    return tuple(export_session(record, position=index) for index, record in enumerate(records))


def is_injected_text(text: str) -> bool:
    """Return whether user-role text is harness-injected context rather than a typed prompt.

    Args:
        text: Visible user-role text.

    Returns:
        Whether the text starts with a known harness wrapper.
    """
    return text.lstrip().startswith(_INJECTED_PREFIXES)


def _claude_sessions(records: Sequence[JsonObject]) -> tuple[AgentSession, ...]:
    """Group Claude Code records by session and subagent, then read each group.

    Args:
        records: Decoded transcript records in file order.

    Returns:
        One session per ``sessionId`` and per subagent ``agentId``.
    """
    groups: dict[str, list[JsonObject]] = {}
    titles: dict[str, JsonObject] = {}
    skipped: dict[str, Counter[str]] = {}
    for record in records:
        session_id = record.get("sessionId")
        kind = record.get("type")
        if not isinstance(session_id, str) or not session_id.strip():
            continue
        if kind in ("ai-title", "custom-title"):
            title = record.get("aiTitle", record.get("customTitle"))
            if isinstance(title, str):
                titles.setdefault(session_id, {})[str(kind)] = title
            continue
        key = _claude_key(record, session_id)
        groups.setdefault(key, [])
        if kind in _CLAUDE_MESSAGE_TYPES and isinstance(record.get("message"), dict):
            groups[key].append(record)
        else:
            skipped.setdefault(key, Counter())[str(kind)] += 1
    sessions: list[AgentSession] = []
    for key, group in groups.items():
        if not group:
            continue
        session_id = str(group[0]["sessionId"])
        metadata: JsonObject = {**titles.get(session_id, {})}
        counts = skipped.get(key, Counter())
        sessions.append(_claude_session(key, session_id, group, metadata, counts))
    return tuple(sessions)


def _claude_key(record: JsonObject, session_id: str) -> str:
    """Return the session key of one record: the session or one of its subagents.

    Args:
        record: One transcript record.
        session_id: The record's ``sessionId``.

    Returns:
        ``sessionId`` for main-chain records, with an ``agent:`` suffix for subagent records.
    """
    if record.get("isSidechain") is not True:
        return session_id
    agent = record.get("agentId")
    return f"{session_id}/agent:{agent if isinstance(agent, str) and agent else 'sidechain'}"


def _claude_session(
    key: str,
    session_id: str,
    records: Sequence[JsonObject],
    metadata: JsonObject,
    skipped: Counter[str],
) -> AgentSession:
    """Read one Claude Code session or subagent group into neutral events.

    Args:
        key: Session key.
        session_id: Parent ``sessionId`` shared with subagents.
        records: The group's ``user`` and ``assistant`` records in file order.
        metadata: Titles recorded for the session.
        skipped: Counts of bookkeeping records that are not converted.

    Returns:
        The neutral session.
    """
    events: list[SessionEvent] = []
    seen_records: set[str] = set()
    seen_calls: set[str] = set()
    answered: set[str] = set()
    names: dict[str, str] = {}
    duplicates = 0
    last_response: str | None = None
    first = records[0]
    for record in records:
        uuid = record.get("uuid")
        if isinstance(uuid, str):
            if uuid in seen_records:
                duplicates += 1
                continue
            seen_records.add(uuid)
        message = record["message"]
        assert isinstance(message, dict)
        timestamp = _optional_timestamp(record.get("timestamp"))
        source_id = uuid if isinstance(uuid, str) else f"record-{len(events)}"
        if record.get("type") == "assistant":
            event = _claude_assistant(record, message, source_id, timestamp, seen_calls)
            for call in event.tool_calls:
                if call.call_id is not None:
                    names[call.call_id] = call.name
            response_id = message.get("id") if isinstance(message.get("id"), str) else None
            if response_id is not None and response_id == last_response and events:
                events[-1] = _merge_turns(events[-1], event)
            else:
                events.append(event)
            last_response = response_id
            continue
        emitted = _claude_user(
            record, message, source_id, timestamp, names=names, answered=answered
        )
        if emitted:
            last_response = None
        events.extend(emitted)
    if duplicates:
        skipped["duplicate-record"] += duplicates
    agent = first.get("agentId")
    metadata.update(
        {
            "session_id": session_id,
            "sidechain": first.get("isSidechain") is True,
            "skipped_records": dict(sorted(skipped.items())),
        }
    )
    for name, source in (("cwd", "cwd"), ("git_branch", "gitBranch"), ("agent_version", "version")):
        value = first.get(source)
        if isinstance(value, str) and value:
            metadata[name] = value
    if isinstance(agent, str) and agent:
        metadata["agent_id"] = agent
    return AgentSession(
        key=key,
        format="claude-code",
        conversation_id=session_id,
        events=tuple(events),
        metadata=metadata,
        started_at=_optional_timestamp(first.get("timestamp")),
    )


def _merge_turns(previous: SessionEvent, current: SessionEvent) -> SessionEvent:
    """Merge two assistant records of one API response into one turn.

    Args:
        previous: Earlier part, whose identity and start instant are kept.
        current: Later part, whose usage and error supersede the earlier part's.

    Returns:
        One assistant turn holding both parts' text and tool calls.
    """
    text = "\n".join(part for part in (previous.text, current.text) if part)
    return replace(
        previous,
        text=text,
        tool_calls=previous.tool_calls + current.tool_calls,
        usage=current.usage or previous.usage,
        error=current.error or previous.error,
        model=previous.model or current.model,
    )


def _claude_assistant(
    record: JsonObject,
    message: JsonObject,
    source_id: str,
    timestamp: datetime | None,
    seen_calls: set[str],
) -> SessionEvent:
    """Read one Claude Code assistant record.

    Args:
        record: The transcript record.
        message: Its Anthropic-style ``message`` object.
        source_id: Record identity.
        timestamp: Recorded instant.
        seen_calls: Tool-use identities already read; repeated calls are dropped.

    Returns:
        One assistant event. Thinking blocks are not visible output and are omitted.
    """
    texts: list[str] = []
    calls: list[SessionToolCall] = []
    for block in _blocks(message.get("content")):
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(str(block["text"]))
        elif kind == "tool_use" and isinstance(block.get("name"), str):
            call_id = _text_or(block.get("id"), None)
            if call_id is not None:
                if call_id in seen_calls:
                    continue
                seen_calls.add(call_id)
            calls.append(SessionToolCall(call_id, str(block["name"]), block.get("input", {})))
    model = message.get("model")
    error = None
    if record.get("isApiErrorMessage") is True or record.get("error") is not None:
        error = json_text(record.get("error")) or "\n".join(texts) or "API error"
    return SessionEvent(
        kind="assistant",
        source_id=source_id,
        timestamp=timestamp,
        text="\n".join(text for text in texts if text.strip()),
        tool_calls=tuple(calls),
        model=model if isinstance(model, str) and model and model != "<synthetic>" else None,
        usage=_usage(message.get("usage")),
        error=error,
    )


def _claude_user(
    record: JsonObject,
    message: JsonObject,
    source_id: str,
    timestamp: datetime | None,
    *,
    names: dict[str, str],
    answered: set[str],
) -> list[SessionEvent]:
    """Read one Claude Code user record into prompt and tool-result events.

    Args:
        record: The transcript record.
        message: Its ``message`` object.
        source_id: Record identity.
        timestamp: Recorded instant.
        names: Tool names by call identity, from earlier assistant turns.
        answered: Call identities already answered; repeated results are dropped.

    Returns:
        Tool results in block order, then one user event for any visible text.
    """
    content = message.get("content")
    if isinstance(content, str):
        return [_user_event(record, source_id, timestamp, content)]
    events: list[SessionEvent] = []
    texts: list[str] = []
    for index, block in enumerate(_blocks(content)):
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(str(block["text"]))
        elif kind == "image":
            texts.append("[image]")
        elif kind == "tool_result":
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str) or call_id in answered:
                continue
            answered.add(call_id)
            events.append(
                SessionEvent(
                    kind="tool_result",
                    source_id=f"{source_id}#{index}",
                    timestamp=timestamp,
                    text=content_text(block.get("content")),
                    tool_call_id=call_id,
                    tool_name=names.get(call_id),
                    is_error=block.get("is_error") is True,
                )
            )
    text = "\n".join(texts)
    if text.strip():
        events.append(_user_event(record, source_id, timestamp, text))
    return events


def _user_event(
    record: JsonObject, source_id: str, timestamp: datetime | None, text: str
) -> SessionEvent:
    """Build one Claude Code user-text event, classifying typed prompts.

    Args:
        record: The transcript record.
        source_id: Record identity.
        timestamp: Recorded instant.
        text: Visible text.

    Returns:
        A user event; meta, compaction-summary, and harness-wrapped text is not a prompt.
    """
    injected = (
        record.get("isMeta") is True
        or record.get("isCompactSummary") is True
        or is_injected_text(text)
    )
    return SessionEvent(
        kind="user",
        source_id=source_id,
        timestamp=timestamp,
        text=text,
        prompt=bool(text.strip()) and not injected,
    )


def _pi_session(records: Sequence[JsonObject]) -> AgentSession:
    """Read one pi session file into neutral events along its active branch.

    Args:
        records: Decoded session entries in file order.

    Returns:
        The neutral session.

    Raises:
        VendorTraceFormatError: The file has no ``session`` header.
    """
    header = next((item for item in records if item.get("type") == "session"), None)
    if header is None or not isinstance(header.get("id"), str):
        raise VendorTraceFormatError("pi session files need a session header with an id")
    entries, off_branch = _active_branch(records)
    skipped: Counter[str] = Counter()
    events: list[SessionEvent] = []
    tools: list[JsonObject] = []
    provider: str | None = None
    model: str | None = None
    for entry in entries:
        kind = entry.get("type")
        if kind == "model_change":
            provider = _text_or(entry.get("provider"), provider)
            model = _text_or(entry.get("modelId"), model)
        elif kind == "custom" and entry.get("customType") == "teich-available-tools":
            data = entry.get("data")
            declared = data.get("tools") if isinstance(data, dict) else None
            if isinstance(declared, list):
                tools.extend(item for item in declared if isinstance(item, dict))
        elif kind == "message" and isinstance(entry.get("message"), dict):
            event = _pi_event(entry, len(events), provider=provider, model=model)
            if event is None:
                skipped["message"] += 1
            else:
                events.append(event)
        elif kind != "session":
            skipped[str(kind)] += 1
    if off_branch:
        skipped["off-branch"] += off_branch
    metadata: JsonObject = {
        "session_id": str(header["id"]),
        "skipped_records": dict(sorted(skipped.items())),
    }
    if isinstance(header.get("cwd"), str):
        metadata["cwd"] = str(header["cwd"])
    if isinstance(header.get("version"), int | str):
        metadata["agent_version"] = str(header["version"])
    return AgentSession(
        key=str(header["id"]),
        format="pi",
        conversation_id=str(header["id"]),
        events=tuple(events),
        declared_tools=tuple(tools),
        metadata=metadata,
        started_at=_optional_timestamp(header.get("timestamp")),
    )


def _active_branch(records: Sequence[JsonObject]) -> tuple[list[JsonObject], int]:
    """Select the entries on the path from the last message to the root.

    The path is used only when every entry declares ``parentId`` and the chain from the last
    message reaches a root without a missing link; otherwise file order is kept.

    Args:
        records: Session entries in file order.

    Returns:
        The selected entries in file order and the number of message entries left out.
    """
    linked = [item for item in records if item.get("type") != "session"]
    messages = [item for item in linked if item.get("type") == "message"]
    if not messages or any("parentId" not in item or "id" not in item for item in linked):
        return list(records), 0
    by_id = {item.get("id"): item for item in linked}
    path: set[str] = set()
    current: JsonObject | None = messages[-1]
    while current is not None:
        identity = current.get("id")
        if not isinstance(identity, str) or identity in path:
            return list(records), 0
        path.add(identity)
        parent = current.get("parentId")
        if parent is None:
            current = None
        elif parent in by_id:
            current = by_id[parent]
        else:
            return list(records), 0
    kept = [item for item in records if item.get("type") == "session" or item.get("id") in path]
    return kept, sum(1 for item in messages if item.get("id") not in path)


def _pi_event(
    entry: JsonObject, position: int, *, provider: str | None, model: str | None
) -> SessionEvent | None:
    """Read one pi message entry.

    Args:
        entry: One ``message`` entry.
        position: Events read so far, used when the entry declares no identity.
        provider: Provider of the latest ``model_change`` entry.
        model: Model of the latest ``model_change`` entry.

    Returns:
        The event, or ``None`` for a message role that is not request-visible.
    """
    message = entry["message"]
    assert isinstance(message, dict)
    role = message.get("role")
    source_id = entry.get("id") if isinstance(entry.get("id"), str) else f"message-{position}"
    assert isinstance(source_id, str)
    timestamp = _optional_timestamp(entry.get("timestamp")) or _optional_timestamp(
        message.get("timestamp")
    )
    if role in ("user", "system"):
        text = content_text(message.get("content"))
        if not text.strip():
            return None
        return SessionEvent(
            kind="user" if role == "user" else "system",
            source_id=source_id,
            timestamp=timestamp,
            text=text,
            prompt=role == "user" and not is_injected_text(text),
        )
    if role == "bashExecution":
        if message.get("excludeFromContext") is True:
            return None
        command = message.get("command")
        output = message.get("output")
        text = f"$ {_text_or(command, '')}\n{_text_or(output, '')}"
        return SessionEvent(kind="user", source_id=source_id, timestamp=timestamp, text=text)
    if role == "toolResult":
        call_id = message.get("toolCallId")
        name = message.get("toolName")
        return SessionEvent(
            kind="tool_result",
            source_id=source_id,
            timestamp=timestamp,
            text=content_text(message.get("content")),
            tool_call_id=call_id if isinstance(call_id, str) else None,
            tool_name=name if isinstance(name, str) and name else None,
            is_error=message.get("isError") is True,
        )
    if role != "assistant":
        return None
    texts: list[str] = []
    calls: list[SessionToolCall] = []
    for block in _blocks(message.get("content")):
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(str(block["text"]))
        elif block.get("type") == "toolCall" and isinstance(block.get("name"), str):
            call_id = block.get("id")
            calls.append(
                SessionToolCall(
                    call_id if isinstance(call_id, str) else None,
                    str(block["name"]),
                    block.get("arguments", {}),
                )
            )
    error = message.get("errorMessage")
    failed = message.get("stopReason") == "error" or isinstance(error, str)
    return SessionEvent(
        kind="assistant",
        source_id=source_id,
        timestamp=timestamp,
        text="\n".join(text for text in texts if text.strip()),
        tool_calls=tuple(calls),
        model=_text_or(message.get("model"), model),
        provider=_text_or(message.get("provider"), provider),
        usage=_usage(message.get("usage")),
        error=(error if isinstance(error, str) and error else "model error") if failed else None,
    )


def export_session(record: JsonObject, *, position: int) -> AgentSession:
    """Read one one-record-per-session export line.

    Args:
        record: One ``{id, createdAt, projectPath, source, messages}`` record.
        position: Line position, used when the record declares no identity.

    Returns:
        The neutral session, ordered by message position with synthetic time.

    Raises:
        VendorTraceFormatError: The record has no message array.
    """
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise VendorTraceFormatError("session export records need a messages array")
    identity = record.get("id")
    key = identity if isinstance(identity, str) and identity else f"session-{position}"
    names: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    events: list[SessionEvent] = []
    for index, raw in enumerate(messages):
        if not isinstance(raw, dict):
            raise VendorTraceFormatError("session export messages must be objects")
        event = _export_event(raw, f"message-{index}", names)
        if event is None:
            continue
        identity = event.tool_calls[0].call_id if event.tool_calls else event.tool_call_id
        if identity is not None:
            if (event.kind, identity) in seen:
                duplicates += 1
                continue
            seen.add((event.kind, identity))
        events.append(event)
    metadata: JsonObject = {
        "session_id": key,
        "skipped_records": {"duplicate-record": duplicates} if duplicates else {},
    }
    for name, source in (("cwd", "projectPath"), ("agent", "source"), ("title", "name")):
        value = record.get(source)
        if isinstance(value, str) and value:
            metadata[name] = value
    return AgentSession(
        key=key,
        format="session-export",
        conversation_id=key,
        events=tuple(events),
        metadata=metadata,
        synthetic_time=True,
        started_at=_optional_timestamp(record.get("createdAt")),
    )


def _export_event(
    message: JsonObject, source_id: str, names: dict[str, str]
) -> SessionEvent | None:
    """Read one flat export message.

    Args:
        message: One message object with string content.
        source_id: Positional message label.
        names: Tool names by call identity, extended by tool-call messages.

    Returns:
        The event, or ``None`` for an empty or unsupported role.
    """
    role = message.get("role")
    text = content_text(message.get("content"))
    call_id = message.get("toolCallId", message.get("tool_call_id"))
    call_id = call_id if isinstance(call_id, str) and call_id else None
    tool_name = message.get("toolName", message.get("name"))
    tool_name = tool_name if isinstance(tool_name, str) and tool_name else None
    if role in ("user", "system"):
        if not text.strip():
            return None
        return SessionEvent(
            kind="user" if role == "user" else "system",
            source_id=source_id,
            timestamp=None,
            text=text,
            prompt=role == "user" and not is_injected_text(text),
        )
    if isinstance(role, str) and role.casefold() in _EXPORT_TOOL_RESULT_ROLES:
        return SessionEvent(
            kind="tool_result",
            source_id=source_id,
            timestamp=None,
            text=text,
            tool_call_id=call_id,
            tool_name=tool_name or (names.get(call_id) if call_id else None),
        )
    if role != "assistant":
        return None
    model = message.get("model")
    calls: tuple[SessionToolCall, ...] = ()
    completion = text
    if tool_name is not None:
        calls = (SessionToolCall(call_id, tool_name, _export_arguments(text, tool_name)),)
        completion = ""
        if call_id is not None:
            names[call_id] = tool_name
    return SessionEvent(
        kind="assistant",
        source_id=source_id,
        timestamp=None,
        text=completion,
        tool_calls=calls,
        model=model if isinstance(model, str) and model else None,
    )


def _export_arguments(text: str, tool_name: str) -> JsonValue:
    """Recover recorded arguments from an export's ``[Tool call: name] {json}`` content.

    Args:
        text: Tool-call message content.
        tool_name: Declared tool name.

    Returns:
        The decoded JSON arguments, or the remaining text when it is not JSON.
    """
    prefix = f"{_CALL_PREFIX}{tool_name}]"
    body = text[len(prefix) :].strip() if text.startswith(prefix) else text.strip()
    try:
        return json.loads(body, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError):
        return body


def _reject_constant(constant: str) -> JsonValue:
    """Refuse non-standard JSON constants such as ``NaN`` in decoded arguments.

    Args:
        constant: The constant name.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"non-standard JSON constant {constant}")


def content_text(value: JsonValue | None) -> str:
    """Read visible text from string content or a list of content blocks.

    Args:
        value: Content as text, text blocks, or nested tool-result content.

    Returns:
        Joined text; image blocks are rendered as ``[image]`` without their data.
    """
    if isinstance(value, str):
        return value
    parts: list[str] = []
    for block in _blocks(value):
        if isinstance(block.get("text"), str):
            parts.append(str(block["text"]))
        elif block.get("type") == "image":
            parts.append("[image]")
        elif "content" in block:
            parts.append(content_text(block.get("content")))
    return "\n".join(part for part in parts if part)


def _blocks(value: JsonValue | None) -> tuple[JsonObject, ...]:
    """Return the object blocks of a content list.

    Args:
        value: Content value.

    Returns:
        Object blocks in order, empty when the content is not a list.
    """
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _usage(value: JsonValue | None) -> VendorTokenUsage | None:
    """Read complete token accounting in the Anthropic or pi usage shape.

    Args:
        value: Recorded usage object.

    Returns:
        Usage when both input and output counts are non-negative integers, else ``None``.
    """
    if not isinstance(value, dict):
        return None
    counts: list[int | None] = []
    for keys in (
        ("input_tokens", "input"),
        ("output_tokens", "output"),
        ("cache_read_input_tokens", "cacheRead"),
    ):
        found = next((value[key] for key in keys if key in value), None)
        valid = isinstance(found, int) and not isinstance(found, bool) and found >= 0
        counts.append(found if valid and isinstance(found, int) else None)
    if counts[0] is None or counts[1] is None:
        return None
    return VendorTokenUsage(
        input_tokens=counts[0], output_tokens=counts[1], cached_input_tokens=counts[2]
    )


def _optional_timestamp(value: JsonValue | None) -> datetime | None:
    """Parse a recorded timestamp, treating absent, zero, and malformed values as unknown.

    Args:
        value: ISO-8601 text or epoch seconds or milliseconds.

    Returns:
        The UTC instant, or ``None`` when the value is not a usable timestamp.
    """
    if value is None or value == 0 or value == "":
        return None
    try:
        return source_timestamp(value, "session timestamp")
    except VendorTraceFormatError:
        return None


def _text_or(value: JsonValue | None, fallback: str | None) -> str | None:
    """Return non-empty text, otherwise the fallback.

    Args:
        value: Candidate value.
        fallback: Value used when the candidate is not non-empty text.

    Returns:
        The text or the fallback.
    """
    return value if isinstance(value, str) and value.strip() else fallback

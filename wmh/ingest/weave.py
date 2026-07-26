"""Weights & Biases Weave adapter — Weave Call objects into normalized ``Trace``s.

Weave (https://wandb.ai/site/weave) records agent executions as **Calls**, not OTLP spans.
Each Call represents one invocation of a tracked **Op** and carries:

  - ``id`` — unique call identifier (used as ``span_id``).
  - ``trace_id`` — groups calls belonging to the same agent run.
  - ``parent_id`` — parent call id (``None`` for root calls), forming a call tree.
  - ``op_name`` — a ``weave:///entity/project/op/<name>:<hash>`` URI *or* a plain function name.
  - ``started_at`` / ``ended_at`` — ISO-8601 timestamps.
  - ``inputs`` — the arguments passed to the op (a JSON object).
  - ``output`` — the return value (a string, object, or ``None``).
  - ``exception`` — error message when the call failed (``None`` on success).
  - ``summary`` — aggregated metadata (token counts, latency) added by Weave post-call.
  - ``attributes`` — custom user-attached metadata.

Classification (LLM vs tool):

  Weave does not tag calls with a semantic convention like OpenInference ``span.kind``.
  Instead, the adapter classifies by **op name heuristics**: ops whose name contains an
  LLM-related marker (``chat``, ``complete``, ``generate``, ``predict``, ``llm``,
  ``openai``, ``anthropic``) are treated as LLM/agent calls (Action spans); everything
  else with a non-empty ``output`` is treated as a tool execution (Observation spans).
  Root calls with children are treated as agent orchestration (LLM).

  The adapter emits ``SpanRecord``s with OTel GenAI attribute keys so the shared
  normalizer (``wmh.ingest.normalize``) pairs them into Steps. ``inputs`` map to tool
  arguments; ``output`` maps to ``gen_ai.tool.message`` (tool) or ``gen_ai.completion``
  (LLM).

Accepted file shapes (``from_file``): a single Call object, a JSON array of Calls, a
``calls/stream_query`` response wrapper (``[{...}, {..}]`` JSONL from the streaming endpoint), or
JSONL (one Call per line). Grouping is by ``trace_id``.

Pull: live pull via the Weave service API is implemented in ``_pull_payloads``; it queries
``/calls/stream_query`` with a project id. Export to a file and use ``from_file`` if you prefer
offline ingestion.
"""

from __future__ import annotations

import json
import os

import httpx
from pydantic import JsonValue

from wmh.core.types import JsonObject
from wmh.ingest.adapter import VendorPull, register_adapter
from wmh.ingest.base import BaseTraceAdapter
from wmh.ingest.normalize import SpanRecord, as_text, iso_to_ordinal

# Weave service API base. Configurable for self-hosted / dedicated instances.
_API_HOST = os.environ.get("WEAVE_API_HOST", "https://trace.wandb.ai").rstrip("/")
_API_KEY_ENV = "WANDB_API_KEY"

# Op-name substrings that mark a call as an LLM/agent invocation (case-insensitive).
_LLM_MARKERS = frozenset(
    {
        "chat",
        "complete",
        "completion",
        "generate",
        "predict",
        "llm",
        "openai",
        "anthropic",
        "bedrock",
        "invoke_model",
        "create_message",
    }
)


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _extract_op_name(raw: JsonValue) -> str:
    """Extract the human-readable op name from a Weave op URI or plain string.

    Weave op URIs look like ``weave:///entity/project/op/my_func:abc123``; we want ``my_func``.
    Plain function names (``my_func``) pass through unchanged.
    """
    if not isinstance(raw, str) or not raw:
        return ""
    # Strip the weave URI prefix and hash suffix if present.
    name = raw
    if "/op/" in name:
        name = name.split("/op/")[-1]
    if ":" in name:
        name = name.rsplit(":", 1)[0]
    return name


def _is_llm_call(op_name: str) -> bool:
    """Heuristic: does this op name look like an LLM/agent call?"""
    lower = op_name.lower()
    return any(marker in lower for marker in _LLM_MARKERS)


def _call_trace_id(call: JsonObject) -> str:
    """The trace id grouping this call with its siblings."""
    for key in ("trace_id", "traceId"):
        value = call.get(key)
        if isinstance(value, str) and value:
            return value
    # Fallback: root calls without a trace_id use their own call id.
    call_id = call.get("id")
    return call_id if isinstance(call_id, str) and call_id else ""


def _call_id(call: JsonObject) -> str:
    """The unique call identifier (used as span_id)."""
    cid = call.get("id")
    return cid if isinstance(cid, str) else ""


def _call_parent_id(call: JsonObject) -> str:
    """The parent call id (empty string for root calls)."""
    pid = call.get("parent_id")
    return pid if isinstance(pid, str) else ""


def _call_inputs(call: JsonObject) -> JsonObject:
    """The call's input arguments as a dict."""
    inputs = call.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _call_output(call: JsonObject) -> str:
    """The call's output rendered as a string."""
    output = call.get("output")
    if output is None:
        return ""
    return output if isinstance(output, str) else as_text(output)


def _call_exception(call: JsonObject) -> str | None:
    """The exception message if the call failed, else None."""
    exc = call.get("exception")
    if isinstance(exc, str) and exc.strip():
        return exc.strip()
    return None


def _call_is_error(call: JsonObject) -> bool:
    """True when the call ended with an error (exception present or status flagged)."""
    if _call_exception(call) is not None:
        return True
    status = call.get("status")
    if isinstance(status, str):
        return status.lower() in ("error", "failed")
    return False


def _user_message_text(inputs: JsonObject) -> str | None:
    """Extract the first user message from an LLM call's inputs (OpenAI-style messages list)."""
    messages = inputs.get("messages")
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if isinstance(msg, dict) and _as_str(msg.get("role")).lower() in {"user", "human"}:
            content = msg.get("content")
            if content is not None:
                return as_text(content)
    return None


def _weave_spans(call: JsonObject, ordinal: int) -> list[SpanRecord]:
    """Map one Weave Call to one or more ``SpanRecord``s.

    An LLM call with multiple parallel tool calls (increasingly common with
    OpenAI-style parallel function calling) emits one action ``SpanRecord``
    per tool call so the normalizer can pair each with its corresponding
    child tool-execution Call. Returns an empty list if the call has no
    trace id.
    """
    trace_id = _call_trace_id(call)
    if not trace_id:
        return []

    op_name = _extract_op_name(call.get("op_name"))
    is_llm = _is_llm_call(op_name)
    inputs = _call_inputs(call)
    output = _call_output(call)
    error = _call_is_error(call)
    exception = _call_exception(call)
    call_id = _call_id(call) or f"weave-{ordinal:08d}"
    parent_id = _call_parent_id(call)
    start = iso_to_ordinal(call.get("started_at"), ordinal)
    end = iso_to_ordinal(call.get("ended_at"), ordinal)

    if not is_llm:
        attrs: JsonObject = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": op_name or call.get("display_name", ""),
        }
        if inputs:
            attrs["gen_ai.tool.call.arguments"] = as_text(inputs)
        attrs["gen_ai.tool.message"] = exception if error and exception else output
        return [
            SpanRecord(
                trace_id=trace_id,
                span_id=call_id,
                parent_span_id=parent_id,
                name="execute_tool",
                start_nano=start,
                end_nano=end,
                attributes=attrs,
                status_error=error,
            )
        ]

    # LLM call: check for tool calls in the output.
    tool_calls = _extract_tool_calls(call.get("output"))
    task = _user_message_text(inputs)

    if not tool_calls:
        # Plain completion (no tool calls).
        attrs = {"gen_ai.operation.name": "chat"}
        attrs["gen_ai.completion"] = exception if error and exception else output
        if task is not None:
            attrs["gen_ai.prompt"] = task
        return [
            SpanRecord(
                trace_id=trace_id,
                span_id=call_id,
                parent_span_id=parent_id,
                name="chat",
                start_nano=start,
                end_nano=end,
                attributes=attrs,
                status_error=error,
            )
        ]

    if len(tool_calls) == 1:
        # Single tool call: emit one action span for the normalizer
        # to pair with the child tool Call's execution span.
        tc_name, tc_args = tool_calls[0]
        attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.tool.name": tc_name,
            "gen_ai.tool.call.arguments": tc_args,
        }
        if task is not None:
            attrs["gen_ai.prompt"] = task
        return [
            SpanRecord(
                trace_id=trace_id,
                span_id=call_id,
                parent_span_id=parent_id,
                name="chat",
                start_nano=start,
                end_nano=end,
                attributes=attrs,
                status_error=error,
            )
        ]

    # Multiple parallel tool calls: do NOT emit action spans from the
    # LLM call. Each child tool Call is processed separately and
    # produces a complete Step (action from op_name + inputs,
    # observation from output). Emitting LLM-side action spans here
    # would create an action,action,...,tool,tool ordering that the
    # normalizer cannot pair correctly, leading to mismatched
    # action/observation Steps.
    return []


def _extract_tool_calls(output: JsonValue) -> list[tuple[str, str]]:
    """Extract (name, arguments_json) pairs from an LLM call's output."""
    calls: list[tuple[str, str]] = []
    if not isinstance(output, dict):
        return calls
    # OpenAI-style: output has a choices list or a direct tool_calls list.
    tool_calls = output.get("tool_calls")
    if not isinstance(tool_calls, list):
        choices = output.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict):
                        tc = message.get("tool_calls")
                        if isinstance(tc, list):
                            tool_calls = tc
                            break
    if not isinstance(tool_calls, list):
        return calls
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if isinstance(fn, dict):
            name = _as_str(fn.get("name"))
            args = fn.get("arguments")
            args_str = args if isinstance(args, str) else as_text(args)
            if name:
                calls.append((name, args_str))
    return calls


class WeaveAdapter(BaseTraceAdapter):
    """Map Weights & Biases Weave Call exports into normalized ``Trace``s."""

    name = "weave"

    def spans_from_payload(self, payload: JsonValue) -> list[SpanRecord]:
        """Weave Call dicts -> SpanRecords.

        Accepts a single Call, a list of Calls, or a calls/stream_query response (JSONL lines or a
        JSON array). Each Call is mapped independently; the shared normalizer groups by trace_id.
        """
        calls = self._extract_calls(payload)
        spans: list[SpanRecord] = []
        for ordinal, call in enumerate(calls):
            spans.extend(_weave_spans(call, ordinal))
        return spans

    def _extract_calls(self, payload: JsonValue) -> list[JsonObject]:
        """Normalize a payload into a flat list of Weave Call objects."""
        if isinstance(payload, list):
            out: list[JsonObject] = []
            for item in payload:
                out.extend(self._extract_calls(item))
            return out
        if not isinstance(payload, dict):
            return []
        # Wrapper shapes: {"calls": [...]}, {"data": [...]}, {"results": [...]}.
        for wrapper_key in ("calls", "data", "results"):
            inner = payload.get(wrapper_key)
            if isinstance(inner, list):
                out = []
                for item in inner:
                    out.extend(self._extract_calls(item))
                return out
        # A dict with an id or op_name is a single Call.
        if "id" in payload or "op_name" in payload or "trace_id" in payload:
            return [payload]
        return []

    def _pull_payloads(self, pull: VendorPull) -> list[JsonValue]:
        """Query the Weave service API for Calls and return them for normalization.

        ``pull.project`` is the W&B ``entity/project`` path; ``pull.api_key`` (else
        ``$WANDB_API_KEY``) is a W&B API key. Fetches recent calls via ``/calls/stream_query``.
        """
        api_key = pull.api_key or os.environ.get(_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"weave pull needs an API key: pass --api-key or set ${_API_KEY_ENV}")
        if not pull.project:
            raise ValueError(
                "weave pull needs --project (the W&B entity/project path, e.g. 'myteam/myproject')"
            )
        limit = pull.limit if pull.limit is not None else 1000
        resp = httpx.post(
            f"{_API_HOST}/calls/stream_query",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "project_id": pull.project,
                "limit": limit,
                "sort_by": [{"field": "started_at", "direction": "desc"}],
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        # The streaming endpoint returns JSONL (one Call per line).
        calls: list[JsonValue] = []
        for line in resp.text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                calls.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
        return [calls]


register_adapter(WeaveAdapter())

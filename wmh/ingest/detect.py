"""Trace-format auto-detection: which registered adapter reads this payload?

The D-INGEST streaming contract (and `wmh ingest` without `--source`) starts from raw payloads
with no declared format: a file upload names no adapter, and a Postgres payload column can hold
any shape a file could. Each source's export carries an unambiguous structural marker (the same
markers the adapters key their own parsing on), so detection is a cheap shape check, not a parse:

  - OTLP envelope (`resourceSpans`) or bare OTLP span (`traceId` + span fields) -> `otel-genai`
  - `{"messages": [{"role": ...}]}` or a bare message list                      -> `chat-json`
  - a trace with an `observations` list                                          -> `langfuse`
  - a run carrying `run_type`                                                    -> `langsmith`
  - a span row carrying `root_span_id`                                           -> `braintrust`
  - `$ai_*` analytics events                                                     -> `posthog`
  - AI-tracing spans typed by `type`/`spanType`                                  -> `mastra`
  - OpenInference span dicts with a `context` id block                           -> `phoenix`

API page wrappers (`{"data"|"events"|"results"|"runs"|"spans"|"traces": [...]}`) and JSON arrays
defer to their elements. Detection samples the first few payloads and requires agreement;
an unknown or conflicting corpus raises with "pass --source" guidance instead of guessing.
"""

from __future__ import annotations

from pydantic import JsonValue

from wmh.ingest.adapter import list_adapters

# Mastra `ExportedSpan.type` values (post- and pre-2025-11-rename), per `wmh/ingest/mastra.py`.
_MASTRA_TYPES = frozenset(
    {
        "model_generation",
        "llm_generation",
        "tool_call",
        "mcp_tool_call",
        "agent_run",
        "model_chunk",
        "model_step",
        "generic",
    }
)
_WRAPPER_KEYS = ("data", "events", "results", "runs", "spans", "traces")
# How many payloads must agree before we trust the verdict (JSONL corpora are homogeneous, so a
# small prefix is plenty; a conflict inside the sample is a hard error either way).
_SAMPLE = 8


def _is_message(value: JsonValue) -> bool:
    return isinstance(value, dict) and isinstance(value.get("role"), str)


def _detect_dict(payload: dict[str, JsonValue]) -> str | None:
    if isinstance(payload.get("resourceSpans"), list):
        return "otel-genai"
    messages = payload.get("messages")
    if isinstance(messages, list) and messages and all(_is_message(m) for m in messages):
        return "chat-json"
    if isinstance(payload.get("observations"), list):
        return "langfuse"
    if "run_type" in payload:
        return "langsmith"
    if "root_span_id" in payload:
        return "braintrust"
    event = payload.get("event")
    properties = payload.get("properties")
    if (isinstance(event, str) and event.startswith("$ai_")) or (
        isinstance(properties, dict) and "$ai_trace_id" in properties
    ):
        return "posthog"
    span_type = payload.get("type") or payload.get("spanType")
    if isinstance(span_type, str) and (
        span_type in _MASTRA_TYPES or span_type.startswith("workflow_")
    ):
        return "mastra"
    context = payload.get("context")
    if isinstance(context, dict) and ("span_id" in context or "trace_id" in context):
        return "phoenix"
    if isinstance(payload.get("traceId"), str) and (
        "spanId" in payload or "attributes" in payload or "startTimeUnixNano" in payload
    ):
        return "otel-genai"
    for key in _WRAPPER_KEYS:
        wrapped = payload.get(key)
        if isinstance(wrapped, list):
            return _detect_one(wrapped)
    return None


def _detect_one(payload: JsonValue) -> str | None:
    if isinstance(payload, dict):
        return _detect_dict(payload)
    if isinstance(payload, list):
        if payload and all(_is_message(m) for m in payload):
            return "chat-json"
        for item in payload:
            verdict = _detect_one(item)
            if verdict is not None:
                return verdict
    return None


def detect_format(payloads: list[JsonValue]) -> str:
    """The registered adapter name for these decoded payloads.

    Samples the first payloads with a recognizable shape and requires them to agree. Raises
    `ValueError` (telling the caller to pass `--source` explicitly) when nothing is recognized
    or when the sample is mixed: silent misdetection would normalize a corpus wrongly.
    """
    verdicts: list[str] = []
    for payload in payloads:
        verdict = _detect_one(payload)
        if verdict is not None:
            verdicts.append(verdict)
        if len(verdicts) >= _SAMPLE:
            break
    if not verdicts:
        raise ValueError(
            "could not auto-detect the trace format: no payload matched a known source shape; "
            f"pass --source explicitly (one of {list_adapters()})"
        )
    distinct = sorted(set(verdicts))
    if len(distinct) > 1:
        raise ValueError(
            f"ambiguous trace format: payloads look like {' and '.join(distinct)}; "
            "pass --source explicitly to pin one"
        )
    return distinct[0]

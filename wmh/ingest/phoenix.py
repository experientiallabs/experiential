"""Arize Phoenix adapter — OpenInference spans, in Phoenix's native export shape.

Phoenix (https://github.com/Arize-ai/phoenix) stores **OpenInference** spans: each span carries a
flat `attributes` dict using OpenInference keys (`openinference.span.kind`, `tool.name`,
`input.value`, `output.value`, `llm.input_messages`, `llm.model_name`, ...). The shared normalizer
(`wmh.ingest.normalize`) already classifies those keys, so this adapter is almost entirely about
*transport + field naming*: turning Phoenix's exported span objects into `SpanRecord`s.

Phoenix exports spans in two shapes, both handled here:

  1. **OTLP envelope** — `{"resourceSpans": [...]}` (or a JSON array of bare OTLP spans with
     `traceId`/`spanId`/`startTimeUnixNano`). This is standard OTLP-JSON, so we hand it straight to
     `collect_spans`.

  2. **Phoenix native span dicts** — what `px.Client().get_spans_dataframe(...)` /
     the Phoenix UI "export" produce. Two flavours are handled:
       - Nested (UI/JSON export): ids under a `context` object, `attributes` a nested dict.
       - FLAT (the dataframe: `get_spans_dataframe().reset_index().to_dict("records")`): the id and
         attributes are **dotted top-level columns** — `context.trace_id`, `context.span_id`,
         `attributes.openinference.span.kind`, `attributes.llm.output_messages.0...` — and
         `start_time`/`end_time` are pandas `Timestamp`/`datetime`, not ISO strings.

         {
           "name": "agent",
           "context.trace_id": "abc123...", "context.span_id": "def456...",
           "parent_id": null, "status_code": "OK",
           "start_time": Timestamp("2024-01-01T00:00:00Z"),
           "attributes.openinference.span.kind": "LLM",
           "attributes.llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get"
         }

     `collect_spans` -> `parse_span` looks for OTLP `traceId`/`spanId`, which these dicts lack, so
     it would drop them. `_phoenix_span` reads `context.<field>` from either shape and rebuilds
     `attributes` from both a nested dict and the flat `attributes.*` columns; the shared classifier
     (which understands OpenInference indexed tool-call keys) does the rest. `start_time`/`end_time`
     accept an ISO string OR a datetime; a missing/unparseable one falls back to the array index,
     which only needs to be monotonic within a trace.

Optional `wmh.*` enrichments (`wmh.state.*`, `wmh.trace.metadata`) are honored by the shared
normalizer if present in a span's attributes.

Live pull: Phoenix's query API/SDK is left as the `BaseTraceAdapter` default (a friendly
"export to a file" error). Phoenix's recommended export path is a file (or a pandas dataframe dumped
to JSON), and the SDK surface is version-dependent; we prefer correctness over a guessed endpoint.
Export from Phoenix to a file and use `from_file` / `wmh ingest run --source phoenix --file ...`.
"""

from __future__ import annotations

from pydantic import JsonValue

from wmh.core.types import JsonObject
from wmh.ingest.adapter import register_adapter
from wmh.ingest.base import BaseTraceAdapter
from wmh.ingest.normalize import SpanRecord, attrs_to_dict, collect_spans, iso_to_ordinal


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _ctx_field(span: JsonObject, field: str) -> str:
    """Read `context.<field>` whether `context` is a NESTED dict (Phoenix UI export) or a FLAT
    dotted column `context.<field>` (`get_spans_dataframe().reset_index().to_dict("records")`)."""
    context = span.get("context")
    if isinstance(context, dict):
        value = context.get(field)
        if isinstance(value, str) and value:
            return value
    flat = span.get(f"context.{field}")
    if isinstance(flat, str) and flat:
        return flat
    bare = span.get(field)
    return bare if isinstance(bare, str) else ""


def _phoenix_attributes(span: JsonObject) -> JsonObject:
    """Reconstruct a span's attributes from a nested `attributes` dict AND/OR flat `attributes.*`
    dotted columns (the dataframe flattens attributes to top-level `attributes.<key>` cols)."""
    attrs = attrs_to_dict(span.get("attributes"))
    for key, value in span.items():
        if isinstance(key, str) and key.startswith("attributes."):
            attrs[key[len("attributes.") :]] = value
    return attrs


def _phoenix_span(raw: JsonValue, ordinal: int) -> SpanRecord | None:
    """Map ONE Phoenix native span dict to a `SpanRecord` (None if it carries no trace id)."""
    if not isinstance(raw, dict):
        return None
    trace_id = _ctx_field(raw, "trace_id")
    if not trace_id:
        return None
    status = _as_str(raw.get("status_code")).upper()
    return SpanRecord(
        trace_id=trace_id,
        span_id=_ctx_field(raw, "span_id"),
        parent_span_id=_as_str(raw.get("parent_id")),
        name=_as_str(raw.get("name")),
        # `start_time`/`end_time` may be ISO strings (UI export) or datetimes (dataframe records).
        start_nano=iso_to_ordinal(raw.get("start_time"), ordinal),
        end_nano=iso_to_ordinal(raw.get("end_time"), ordinal),
        attributes=_phoenix_attributes(raw),
        status_error=status in ("ERROR", "STATUS_CODE_ERROR"),
    )


def _phoenix_spans(payload: JsonValue) -> list[SpanRecord]:
    """Map Phoenix native span dicts (a single dict or an array) to `SpanRecord`s."""
    items = payload if isinstance(payload, list) else [payload]
    spans: list[SpanRecord] = []
    for ordinal, item in enumerate(items):
        parsed = _phoenix_span(item, ordinal)
        if parsed is not None:
            spans.append(parsed)
    return spans


def _is_otlp(payload: JsonValue) -> bool:
    """True when the payload is an OTLP envelope or a bare OTLP span (has `traceId`)."""
    if isinstance(payload, dict):
        return "resourceSpans" in payload or "traceId" in payload
    if isinstance(payload, list):
        return any(_is_otlp(item) for item in payload)
    return False


class PhoenixAdapter(BaseTraceAdapter):
    """Normalize Arize Phoenix OpenInference span exports into `Trace`s. SDK-free."""

    name = "phoenix"

    def spans_from_payload(self, payload: JsonValue) -> list[SpanRecord]:
        """Phoenix native span dicts -> SpanRecords; OTLP envelopes delegate to `collect_spans`.

        A list may mix shapes, so route per item rather than all-or-nothing — otherwise a single
        OTLP-shaped element would send the whole list to `collect_spans`, silently dropping the
        native dicts (which lack `traceId`).
        """
        if isinstance(payload, list):
            spans: list[SpanRecord] = []
            for item in payload:
                spans.extend(self.spans_from_payload(item))
            return spans
        if _is_otlp(payload):
            return collect_spans(payload)
        return _phoenix_spans(payload)


register_adapter(PhoenixAdapter())

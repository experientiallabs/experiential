"""Record-sized views of supported trace export collections."""

from collections.abc import Iterator

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.traces.ingest.json_archive import JsonNode, records
from exp.common.traces.ingest.otlp import _extract_raw_spans
from exp.common.traces.ingest.vendor_records import VendorTraceFormatError

_COLLECTIONS = {
    "chat-json": (("conversations", "data", "results"), ("messages", "conversation")),
    "experiential": (("captures", "records"), ("request",)),
    "braintrust": (
        ("events", "rows", "data", "results", "items"),
        ("span_id", "root_span_id", "span_attributes"),
    ),
    "langfuse": (("data", "traces", "results", "items"), ("observations", "traceId", "trace_id")),
    "langsmith": (("runs", "data", "results", "items"), ("run_type", "runType")),
    "mastra": (("spans", "traces", "data", "results", "items"), ("traceId", "trace_id")),
    "phoenix": (
        ("spans", "data", "results", "items", "traces"),
        ("context", "span_id", "context.span_id", "spanId"),
    ),
    "otel-genai": (
        ("spans", "data", "results", "items", "resource_spans"),
        ("trace_id", "traceId", "context", "span_id", "spanId"),
    ),
    "posthog": (("results", "events"), ("event", "properties")),
}


def source_records(source: str, node: JsonNode) -> Iterator[JsonValue]:
    """Split declared collection wrappers while leaving a conversation or span intact."""
    if source == "otlp":
        if node.kind == "array":
            for child in node.children():
                yield from source_records(source, child)
        elif (resources := node.get("resourceSpans")) is not None and resources.kind == "array":
            yield from _otlp_records(node, snake=False)
        else:
            yield node.value()
        return
    if (
        source in {"phoenix", "otel-genai"}
        and node.kind == "map"
        and (node.get("resourceSpans") is not None or node.get("resource_spans") is not None)
    ):
        yield from _otlp_records(node, snake=source == "phoenix")
        return
    wrappers, keys = _COLLECTIONS[source]
    for record in records(
        node,
        vendor="PostHog" if source == "posthog" else source,
        wrappers=wrappers,
        keys=keys,
        chat=source == "chat-json",
    ):
        yield record.value()


def _array(node: JsonNode | None, message: str) -> Iterator[JsonNode]:
    """Require an explicit array before traversing its children."""
    if node is None or node.kind != "array":
        raise VendorTraceFormatError(message)
    yield from node.children()


def _otlp_records(node: JsonNode, *, snake: bool) -> Iterator[JsonObject]:
    """Yield one inherited-resource OTLP span at a time, preserving resource context."""
    resources = node.get("resourceSpans") or (node.get("resource_spans") if snake else None)
    for resource in _array(resources, "OTLP envelope needs a resourceSpans array"):
        if resource.kind != "map":
            raise VendorTraceFormatError("resourceSpans entries must be objects")
        context = resource.get("resource")
        resource_value = context.value() if context else None
        if not snake:
            _extract_raw_spans(
                {"resourceSpans": [{"resource": resource_value, "scopeSpans": []}]}, 0
            )
        scopes = resource.get("scopeSpans") or (resource.get("scope_spans") if snake else None)
        for scope in _array(scopes, "resourceSpans entries must contain scopeSpans arrays"):
            if scope.kind != "map":
                raise VendorTraceFormatError("scopeSpans entries must be objects")
            for span in _array(scope.get("spans"), "scopeSpans entries must contain spans arrays"):
                if span.kind != "map":
                    raise VendorTraceFormatError("OTLP span entries must be objects")
                yield {
                    "resourceSpans": [
                        {
                            "resource": resource_value,
                            "scopeSpans": [{"spans": [span.value()]}],
                        }
                    ]
                }

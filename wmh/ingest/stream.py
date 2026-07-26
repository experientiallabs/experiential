"""Streaming ingest: normalize any source to OTel JSONL while emitting progress events.

This is the library seam behind `wmh ingest` and the platform's SSE trace-ingest endpoint. The
event vocabulary is pinned by the D-INGEST contract (mirrored in the platform frontend's
`parseTraceIngestEvent`): exactly one `detected {format, traces}`, then repeated
`progress {normalized, total|null, note?}`, then a terminal `done {traces, steps, otel_object}`,
or a terminal `error {message, code?}` at any point. `event_json` serializes an event to exactly
those keys (optional fields dropped when unset, `total` kept even when null).

The pipeline: load payloads (file) or pull them (vendor/database source) -> auto-detect the format
unless pinned -> collect spans through the chosen adapter -> group into traces (the `detected`
count) -> normalize group-by-group (the `progress` ticks) -> persist as OTel-GenAI span JSONL via
`wmh.ingest.otel_writer` (`done.otel_object` is that path, readable by `--source otel-genai` and
every downstream `wmh` command).
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from wmh.core.types import JsonObject, Trace
from wmh.ingest.adapter import SourceCredentialError, VendorPull, get_adapter
from wmh.ingest.base import BaseTraceAdapter, load_payloads
from wmh.ingest.detect import detect_format
from wmh.ingest.normalize import SpanRecord, group_spans, trace_from_group
from wmh.ingest.otel_writer import write_traces_jsonl


class DetectedEvent(BaseModel):
    """The source format has been identified and the trace count is known."""

    type: Literal["detected"] = "detected"
    format: str
    traces: int


class ProgressEvent(BaseModel):
    """`normalized` of `total` traces done (`total` null for open-ended sources)."""

    type: Literal["progress"] = "progress"
    normalized: int
    total: int | None
    note: str | None = None


class DoneEvent(BaseModel):
    """Terminal success: corpus persisted at `otel_object` (a path/storage key)."""

    type: Literal["done"] = "done"
    traces: int
    steps: int
    otel_object: str


class ErrorCode(StrEnum):
    """Machine-readable failure class, so a UI can distinguish bad keys from a dead backend."""

    BAD_CREDENTIALS = "bad_credentials"
    UNREACHABLE = "unreachable"
    BAD_FORMAT = "bad_format"
    EMPTY = "empty"
    INTERNAL = "internal"


class ErrorEvent(BaseModel):
    """Terminal failure."""

    type: Literal["error"] = "error"
    message: str
    code: ErrorCode | None = None


IngestEvent = DetectedEvent | ProgressEvent | DoneEvent | ErrorEvent


def event_json(event: IngestEvent) -> JsonObject:
    """Serialize one event to exactly the D-INGEST wire keys.

    Optional fields (`note`, `code`) are dropped when unset, but `progress.total` stays present
    as null: the pinned frontend parser requires `total` to be a number or null, not absent.
    """
    dump = event.model_dump(mode="json", exclude_none=True)
    if isinstance(event, ProgressEvent) and "total" not in dump:
        dump["total"] = None
    return dump


# Cap on progress events per ingest: enough for a smooth bar, no event flood on huge corpora.
_MAX_PROGRESS_EVENTS = 100


class _IngestFailure(Exception):
    """Internal: a classified failure to surface as a terminal `error` event."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _classify(exc: Exception) -> _IngestFailure:
    if isinstance(exc, _IngestFailure):
        return exc
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code in (401, 403):
            return _IngestFailure(
                ErrorCode.BAD_CREDENTIALS,
                f"the source rejected the credentials ({exc.response.status_code}): {exc}",
            )
        return _IngestFailure(ErrorCode.UNREACHABLE, f"the source returned an error: {exc}")
    if isinstance(exc, httpx.TransportError):
        return _IngestFailure(ErrorCode.UNREACHABLE, f"could not reach the source: {exc}")
    # Adapters raise SourceCredentialError/ConnectionError (both stdlib subclasses, no driver
    # import needed here) for source-side failures; order matters, both are OSError subclasses.
    # A BARE PermissionError (an unreadable local file) is deliberately NOT a credential error:
    # it falls through to the OSError branch as bad_format.
    if isinstance(exc, SourceCredentialError):
        return _IngestFailure(ErrorCode.BAD_CREDENTIALS, str(exc))
    if isinstance(exc, ConnectionError):
        return _IngestFailure(ErrorCode.UNREACHABLE, str(exc))
    if isinstance(exc, (ValueError, OSError)):
        return _IngestFailure(ErrorCode.BAD_FORMAT, str(exc))
    return _IngestFailure(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")


def _collect_spans(
    file: str | None, pull: VendorPull | None, source: str | None
) -> tuple[str, list[SpanRecord]]:
    """Resolve the adapter (detecting the format when unpinned) and collect all spans."""
    if file is not None:
        payloads = load_payloads(Path(file).read_text(encoding="utf-8"))
        fmt = source or detect_format(payloads)
        adapter = get_adapter(fmt)
        if not isinstance(adapter, BaseTraceAdapter):
            raise _IngestFailure(
                ErrorCode.INTERNAL, f"adapter {fmt!r} does not support streaming ingest"
            )
        return fmt, adapter.collect_all(payloads)
    if pull is not None:
        if source is None:
            raise _IngestFailure(
                ErrorCode.BAD_FORMAT, "a vendor/database pull needs an explicit source name"
            )
        adapter = get_adapter(source)
        if not isinstance(adapter, BaseTraceAdapter):
            raise _IngestFailure(
                ErrorCode.INTERNAL, f"adapter {source!r} does not support streaming ingest"
            )
        return source, adapter.spans_from_pull(pull)
    raise _IngestFailure(ErrorCode.BAD_FORMAT, "ingest needs either a file path or a pull")


def ingest_events(
    *,
    file: str | None = None,
    pull: VendorPull | None = None,
    source: str | None = None,
    out: Path,
    limit: int | None = None,
) -> Iterator[IngestEvent]:
    """Normalize one source into OTel JSONL at `out`, yielding D-INGEST progress events.

    Exactly one of `file`/`pull` selects the transport. `source` pins the adapter; when omitted
    (file transport only) the format is auto-detected. `limit` caps the normalized traces for any
    transport (a pull's own `pull.limit` additionally bounds what is fetched vendor-side). The
    generator never raises: every failure becomes a terminal `ErrorEvent`, so a consumer can pipe
    events straight onto a wire.
    """
    try:
        fmt, spans = _collect_spans(file, pull, source)
        groups = group_spans(spans)
        cap = limit if limit is not None else (pull.limit if pull is not None else None)
        if cap is not None:
            groups = groups[:cap]
        if not groups:
            raise _IngestFailure(
                ErrorCode.EMPTY,
                "the source contained no usable traces (nothing normalized); check that the "
                "export carries agent steps, not just container/metric events",
            )
        yield DetectedEvent(format=fmt, traces=len(groups))

        pulled_from = pull.project if pull is not None and pull.project else "vendor"
        label = f"{fmt}:{file}" if file is not None else f"{fmt}:{pulled_from}"
        every = max(1, len(groups) // _MAX_PROGRESS_EVENTS)
        traces: list[Trace] = []
        for i, group in enumerate(groups, start=1):
            traces.append(trace_from_group(group, source=label))
            if i % every == 0 or i == len(groups):
                yield ProgressEvent(normalized=i, total=len(groups))

        out.parent.mkdir(parents=True, exist_ok=True)
        n_spans = write_traces_jsonl(traces, out)
        yield ProgressEvent(
            normalized=len(traces), total=len(groups), note=f"wrote {n_spans} spans"
        )
        yield DoneEvent(
            traces=len(traces),
            steps=sum(len(t.steps) for t in traces),
            otel_object=str(out),
        )
    except Exception as exc:  # noqa: BLE001 - the wire contract IS "never raise, emit error"
        failure = _classify(exc)
        yield ErrorEvent(message=str(failure), code=failure.code)

"""Disk-grouped source normalization for ingestion without materializing the corpus."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import JsonValue, TypeAdapter

from exp.common.core.artifacts import JsonObject, SourceIdentity
from exp.common.traces import Trace
from exp.common.traces.ingest.braintrust import BRAINTRUST_SOURCE
from exp.common.traces.ingest.chat_json import CHAT_JSON_SOURCE
from exp.common.traces.ingest.environment_capture import (
    _SPAN_KEYS,
    canonicalize_environment_capture_payloads,
)
from exp.common.traces.ingest.experiential import EXPERIENTIAL_SOURCE
from exp.common.traces.ingest.json_archive import JsonArchive
from exp.common.traces.ingest.langfuse import LANGFUSE_SOURCE
from exp.common.traces.ingest.langsmith import LANGSMITH_SOURCE
from exp.common.traces.ingest.mastra import MASTRA_SOURCE
from exp.common.traces.ingest.model_identity import (
    TraceModelIdentityEvidence,
    normalized_model_identity_evidence,
)
from exp.common.traces.ingest.otel_genai import _otlp_span
from exp.common.traces.ingest.otlp import (
    GENAI_SEMANTIC_CONVENTION_VERSION,
    OtlpTraceFormatError,
    TraceNormalizationIssue,
    _extract_raw_spans,
    _normalize_trace_group,
    _RawSpan,
)
from exp.common.traces.ingest.phoenix import PHOENIX_SOURCE
from exp.common.traces.ingest.posthog_canonical import (
    _PostHogEvent,
    _source_event_order_key,
    _source_trace_id,
    _trace_observations,
)
from exp.common.traces.ingest.stream_sources import source_records
from exp.common.traces.ingest.vendor_observations import VendorObservation
from exp.common.traces.ingest.vendor_records import VendorTraceFormatError
from exp.common.traces.ingest.vendor_source import VendorSource
from exp.common.traces.ingest.vendor_trace import build_vendor_traces

_OBSERVATION = TypeAdapter(VendorObservation)
_RAW_SPAN = TypeAdapter(_RawSpan)
_EVENT = TypeAdapter(_PostHogEvent)
_ISSUES = TypeAdapter(tuple[TraceNormalizationIssue, ...])
_IDENTITIES = TypeAdapter(tuple[TraceModelIdentityEvidence, ...])
_VENDORS = {
    "braintrust": BRAINTRUST_SOURCE,
    "chat-json": CHAT_JSON_SOURCE,
    "experiential": EXPERIENTIAL_SOURCE,
    "langfuse": LANGFUSE_SOURCE,
    "langsmith": LANGSMITH_SOURCE,
    "mastra": MASTRA_SOURCE,
}


@dataclass(frozen=True)
class IngestSummary:
    """Counts and normalization diagnostics without retaining trace payloads.

    Attributes:
        trace_count: Number of accepted canonical traces.
        issues: Complete ordered exclusions.
        source: Exact input source identity.
    """

    trace_count: int
    issues: tuple[TraceNormalizationIssue, ...]
    source: SourceIdentity


class NormalizedSource:
    """Private disk grouping and sorting with one complete trace in memory at a time."""

    def __init__(self, database: sqlite3.Connection, source_format: str, directory: Path) -> None:
        """Initialize temporary grouping tables that are never shared with gateway capture."""
        self.database = database
        self.directory = directory
        self.source_format = source_format
        self.issues: list[TraceNormalizationIssue] = []
        self.ordinal = 0
        self.source = SourceIdentity(kind="manual", source_id="pending")
        self.identity_available = True
        database.executescript(
            "CREATE TABLE observations(trace_id TEXT,ordinal INTEGER,payload BLOB);"
            "CREATE INDEX observation_order ON observations(trace_id,ordinal);"
            "CREATE TABLE normalized(started TEXT,trace_id TEXT,payload TEXT);"
            "CREATE INDEX normalized_order ON normalized(started,trace_id);"
        )

    def _append(self, identity: str, ordinal: int, payload: bytes) -> None:
        """Persist one typed observation under its complete-trace grouping identity."""
        self.database.execute(
            "INSERT INTO observations VALUES (?,?,?)", (identity, ordinal, payload)
        )

    def vendor[T](self, vendor: VendorSource[T], payload: JsonValue) -> None:
        """Convert one declared vendor record while preserving global source ordinals."""
        for record in vendor.records(payload):
            observations = vendor.convert(record, self.ordinal)
            for observation in observations:
                self._append(
                    observation.source_trace_id,
                    observation.ordinal,
                    _OBSERVATION.dump_json(observation),
                )
            self.ordinal += len(observations)

    def record(self, payload: JsonValue, document: int) -> None:
        """Stage a record or retain its normalization exclusion without ending the corpus."""
        try:
            if self.source_format in {"otlp", "otel-genai"}:
                if (
                    self.source_format == "otel-genai"
                    and isinstance(payload, dict)
                    and "resourceSpans" not in payload
                    and "resource_spans" not in payload
                ):
                    payload = _otlp_span(payload)
                for span in _extract_raw_spans(payload, self.ordinal):
                    identity = span.raw.get("traceId")
                    if not isinstance(identity, str):
                        self.issues.append(
                            TraceNormalizationIssue(
                                f"span-{span.ordinal}", "OTLP span is missing string traceId"
                            )
                        )
                    else:
                        self._append(identity, span.ordinal, _RAW_SPAN.dump_json(span))
                    self.ordinal += 1
            elif self.source_format == "posthog":
                if not isinstance(payload, dict):
                    raise VendorTraceFormatError("PostHog exports must contain record objects")
                identity = _source_trace_id(payload)
                event = _PostHogEvent(
                    payload, self.ordinal, _source_event_order_key(payload, self.ordinal)
                )
                self._append(identity, self.ordinal, _EVENT.dump_json(event))
                self.ordinal += 1
            elif self.source_format == "phoenix":
                self.vendor(PHOENIX_SOURCE, payload)
            else:
                self.vendor(_VENDORS[self.source_format], payload)
        except (VendorTraceFormatError, OtlpTraceFormatError) as exc:
            self.issues.append(TraceNormalizationIssue(f"record-{document}", str(exc)))

    def file(self, path: Path) -> None:
        """Parse a stable private snapshot and group interleaved records on disk."""
        archive = JsonArchive(path, self.directory, self.database, self.source_format)
        self.source = archive.source
        self.issues.extend(archive.issues)
        profile_eligible = True
        for index, node in enumerate(archive.documents(), 1):
            profile_eligible = (
                profile_eligible
                and node.kind == "map"
                and {
                    row[0]
                    for row in self.database.execute(
                        "SELECT name FROM json_nodes WHERE parent=?", (node.identity,)
                    )
                }
                == _SPAN_KEYS
            )
            self.database.execute("SAVEPOINT source_document")
            ordinal, issue_count = self.ordinal, len(self.issues)
            try:
                for payload in source_records(self.source_format, node):
                    self.record(payload, index)
            except (VendorTraceFormatError, OtlpTraceFormatError) as exc:
                self.database.execute("ROLLBACK TO source_document")
                self.ordinal = ordinal
                del self.issues[issue_count:]
                if self.source_format == "posthog":
                    self.database.execute("DELETE FROM observations")
                    self.identity_available = False
                    self.issues = [
                        *archive.issues,
                        TraceNormalizationIssue("posthog-payload", str(exc)),
                    ]
                    break
                self.issues.append(TraceNormalizationIssue(f"record-{index}", str(exc)))
            finally:
                self.database.execute("RELEASE source_document")
        if (
            self.source_format == "otlp"
            and archive.jsonl
            and profile_eligible
            and not archive.duplicate_keys
            and not self.issues
        ):
            self._environment_profile()
        self.finish()

    def _environment_profile(self) -> None:
        """Apply the complete environment profile only if every contiguous trace qualifies."""
        self.database.execute("CREATE TABLE profile(trace_id TEXT,ordinal INTEGER,payload BLOB)")
        eligible = True
        for (identity,) in self.database.execute(
            "SELECT DISTINCT trace_id FROM observations ORDER BY trace_id"
        ):
            rows = list(
                self.database.execute(
                    "SELECT ordinal,payload FROM observations WHERE trace_id=? ORDER BY ordinal",
                    (identity,),
                )
            )
            ordinals = [row[0] for row in rows]
            spans = [_RAW_SPAN.validate_json(row[1]) for row in rows]
            if ordinals != list(range(ordinals[0], ordinals[0] + len(ordinals))) or any(
                span.resource_attributes for span in spans
            ):
                eligible = False
                break
            converted = canonicalize_environment_capture_payloads([span.raw for span in spans])
            if converted is None:
                eligible = False
                break
            for ordinal, value in zip(ordinals, converted, strict=True):
                if not isinstance(value, dict):
                    raise ValueError("invalid canonical environment span")
                self.database.execute(
                    "INSERT INTO profile VALUES (?,?,?)",
                    (identity, ordinal, _RAW_SPAN.dump_json(_RawSpan(value, {}, ordinal))),
                )
        if eligible:
            self.database.execute("DELETE FROM observations")
            self.database.execute("INSERT INTO observations SELECT * FROM profile")
        self.database.execute("DROP TABLE profile")

    def finish(self, *, transform: Callable[[Trace], Trace] | None = None) -> None:
        """Normalize one complete trace group and sort the resulting evidence on disk."""
        for (identity,) in self.database.execute(
            "SELECT DISTINCT trace_id FROM observations ORDER BY trace_id"
        ):
            rows = self.database.execute(
                "SELECT payload FROM observations WHERE trace_id=? ORDER BY ordinal", (identity,)
            )
            try:
                if self.source_format in {"otlp", "otel-genai"}:
                    trace = _normalize_trace_group(
                        [_RAW_SPAN.validate_json(row[0]) for row in rows],
                        source=self.source,
                        semantic_convention_version=GENAI_SEMANTIC_CONVENTION_VERSION,
                    )
                    traces = (trace,)
                else:
                    if self.source_format == "posthog":
                        observations = _trace_observations(
                            identity, [_EVENT.validate_json(row[0]) for row in rows]
                        )
                    else:
                        observations = tuple(_OBSERVATION.validate_json(row[0]) for row in rows)
                    result = build_vendor_traces(
                        observations,
                        vendor=self.source_format,
                        source=self.source,
                        strict_tool_pairing=self.source_format == "posthog",
                    )
                    self.issues.extend(result.issues)
                    traces = result.traces
                for trace in traces:
                    order = (
                        trace.spans[0].started_at.astimezone(UTC).isoformat(timespec="microseconds")
                    )
                    if transform is not None:
                        trace = transform(trace)
                    self.database.execute(
                        "INSERT INTO normalized VALUES (?,?,?)",
                        (order, trace.trace_id, trace.model_dump_json()),
                    )
            except (OtlpTraceFormatError, VendorTraceFormatError) as exc:
                self.issues.append(TraceNormalizationIssue(f"trace-{identity}", str(exc)))
        self.database.commit()

    def traces(self) -> Iterator[Trace]:
        """Read accepted traces in exactly the canonical order without loading the corpus."""
        for row in self.database.execute(
            "SELECT payload FROM normalized ORDER BY started,trace_id"
        ):
            yield Trace.model_validate_json(row[0])

    def summary(self) -> IngestSummary:
        """Return lightweight counts and every retained normalization exclusion."""
        return IngestSummary(
            self.database.execute("SELECT COUNT(*) FROM normalized").fetchone()[0],
            tuple(self.issues),
            self.source,
        )

    def metadata(self) -> JsonObject:
        """Collect compact provenance metadata separately from large trace payloads."""
        identities = [
            evidence
            for trace in self.traces()
            for evidence in normalized_model_identity_evidence((trace,))
        ]
        identities.sort(key=lambda item: (item.trace_id, item.span_id))
        return {
            "schema_version": 1,
            "issues": _ISSUES.dump_python(tuple(self.issues), mode="json"),
            "identity_evidence": (
                _IDENTITIES.dump_python(tuple(identities), mode="json")
                if self.identity_available
                else None
            ),
        }


@contextmanager
def normalization_workspace(source: str) -> Iterator[NormalizedSource]:
    """Own disk-backed normalization state until a source adapter and persistence finish.

    Args:
        source: Canonical format used to convert staged observations into traces.

    Yields:
        Private normalization storage shared only with the caller's source adapter.
    """
    try:
        with TemporaryDirectory(prefix="exp-ingest-") as name:
            directory = Path(name)
            database_path = directory / "normalization.db"
            with closing(sqlite3.connect(database_path)) as database:
                database_path.chmod(0o600)
                database.execute("PRAGMA cache_size=-2048")
                database.execute("PRAGMA temp_store=FILE")
                yield NormalizedSource(database, source, directory)
    except (OSError, sqlite3.Error) as exc:
        raise ValueError(
            "Cannot stage trace import; check source access and temporary disk space, then retry."
        ) from exc


@contextmanager
def normalized_source(source: str, path: Path) -> Iterator[NormalizedSource]:
    """Normalize one declared file source while owning its temporary storage."""
    with normalization_workspace(source) as normalized:
        normalized.file(path)
        yield normalized

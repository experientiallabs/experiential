"""Spec-driven normalization loop shared by declarative vendor trace sources.

Most vendor exports normalize the same way: decode a local file into JSON payloads, flatten each
payload through the vendor's declared envelope keys into records, convert each record into
declared vendor observations, and build canonical traces while retaining every parse or
validation failure as an explicit per-record exclusion. :class:`VendorSource` owns that loop
once, so a vendor module only declares its wire format: which envelope keys wrap records, which
keys identify a bare record, and how one record converts.

A vendor whose conversion cannot be expressed per record (for example OTLP envelope batching)
keeps its own normalize function and registers its loader directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject, SourceIdentity
from wmo.simulation.ingest.otlp import (
    GENAI_SEMANTIC_CONVENTION_VERSION,
    OtlpTraceFormatError,
    TraceNormalizationIssue,
    TraceNormalizationResult,
)
from wmo.simulation.ingest.vendor_observations import VendorObservation
from wmo.simulation.ingest.vendor_records import (
    VendorTraceFormatError,
    flatten_records,
    read_vendor_export,
)
from wmo.simulation.ingest.vendor_trace import build_vendor_traces


def record_flattener(
    *, vendor: str, wrapper_keys: tuple[str, ...], record_keys: tuple[str, ...]
) -> Callable[[JsonValue], tuple[JsonObject, ...]]:
    """Declare a record reader over the shared strict wrapper-flattening rules.

    Args:
        vendor: Vendor label used in error messages.
        wrapper_keys: Envelope keys whose array values contain records or nested wrappers.
        record_keys: Keys whose presence identifies a bare vendor record.

    Returns:
        Reader that flattens one decoded payload into vendor record objects.
    """

    def flatten(payload: JsonValue) -> tuple[JsonObject, ...]:
        """Flatten one decoded payload into vendor record objects.

        Args:
            payload: One decoded export document, array, or record.

        Returns:
            Vendor record objects in source order.

        Raises:
            VendorTraceFormatError: The payload is not a supported wrapper or record shape.
        """
        return flatten_records(
            payload, vendor=vendor, wrapper_keys=wrapper_keys, record_keys=record_keys
        )

    return flatten


@dataclass(frozen=True)
class VendorSource[RecordT]:
    """One declarative vendor trace source: a record reader plus a record converter.

    Args:
        vendor: Vendor label retained on every canonical span and used in error messages.
        records: Reader that flattens one decoded payload into vendor records.
        convert: Converter that turns one record at a source ordinal into observations.
    """

    vendor: str
    records: Callable[[JsonValue], tuple[RecordT, ...]]
    convert: Callable[[RecordT, int], tuple[VendorObservation, ...]]

    def normalize(
        self,
        payloads: Sequence[JsonValue],
        *,
        source: SourceIdentity,
        semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
        initial_issues: Sequence[TraceNormalizationIssue] = (),
    ) -> TraceNormalizationResult:
        """Normalize decoded vendor payloads into canonical traces.

        Args:
            payloads: Decoded vendor documents in source order.
            source: Immutable identity of the source bytes or transport result.
            semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
            initial_issues: Parse exclusions collected before record mapping.

        Returns:
            Canonical traces and every retained validation exclusion.
        """
        issues = list(initial_issues)
        observations: list[VendorObservation] = []
        ordinal = 0
        for index, payload in enumerate(payloads, start=1):
            try:
                records = self.records(payload)
            except (VendorTraceFormatError, OtlpTraceFormatError) as exc:
                issues.append(TraceNormalizationIssue(f"record-{index}", str(exc)))
                continue
            for record in records:
                try:
                    emitted = self.convert(record, ordinal)
                except (VendorTraceFormatError, OtlpTraceFormatError) as exc:
                    issues.append(TraceNormalizationIssue(f"record-{index}", str(exc)))
                    continue
                observations.extend(emitted)
                ordinal += len(emitted)
        return build_vendor_traces(
            observations,
            vendor=self.vendor,
            source=source,
            semantic_convention_version=semantic_convention_version,
            initial_issues=issues,
        )

    def load(
        self,
        path: Path,
        *,
        semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
        source_id: str | None = None,
    ) -> TraceNormalizationResult:
        """Read one vendor JSON or JSONL export into canonical trace evidence.

        Args:
            path: Local vendor export file.
            semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
            source_id: Optional durable source label. The local path is used when omitted.

        Returns:
            Canonical traces and every retained parse or validation exclusion.

        Raises:
            VendorTraceFormatError: The export cannot be read or decoded.
        """
        export = read_vendor_export(path, vendor=self.vendor, source_id=source_id)
        return self.normalize(
            export.payloads,
            source=export.source,
            semantic_convention_version=semantic_convention_version,
            initial_issues=export.issues,
        )

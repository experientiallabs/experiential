"""Tests for the spec-driven vendor source normalization loop."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject, SourceIdentity
from wmo.simulation.ingest.vendor_observations import VendorObservation
from wmo.simulation.ingest.vendor_records import VendorTraceFormatError
from wmo.simulation.ingest.vendor_source import VendorSource, record_flattener

_INSTANT = datetime(2026, 1, 1, tzinfo=UTC)


def _observation(ordinal: int) -> VendorObservation:
    """Return one minimal model observation at the given ordinal.

    Args:
        ordinal: Source order position for the observation.

    Returns:
        Minimal declared model observation.
    """
    return VendorObservation(
        source_trace_id="trace-1",
        source_span_id=f"span-{ordinal}",
        ordinal=ordinal,
        started_at=_INSTANT,
        ended_at=_INSTANT,
        kind="model",
        request_text="hello",
        completion_text="hi",
    )


def test_normalize_labels_failures_and_advances_ordinals_by_emission() -> None:
    """A failing record excludes only itself, and ordinals advance by emitted count."""
    seen_ordinals: list[int] = []

    def convert(record: JsonObject, ordinal: int) -> tuple[VendorObservation, ...]:
        """Emit the declared number of observations or fail for a bad record.

        Args:
            record: Fixture record declaring its kind and emission count.
            ordinal: Source order offset for the first emitted observation.

        Returns:
            Declared observations for this record.

        Raises:
            VendorTraceFormatError: The record is marked bad.
        """
        if record.get("kind") == "bad":
            raise VendorTraceFormatError("bad record")
        seen_ordinals.append(ordinal)
        count = record.get("emit")
        assert isinstance(count, int)
        return tuple(_observation(ordinal + offset) for offset in range(count))

    source = VendorSource(
        vendor="example",
        records=record_flattener(vendor="example", wrapper_keys=("items",), record_keys=("kind",)),
        convert=convert,
    )
    payloads: list[JsonValue] = [
        {"items": [{"kind": "good", "emit": 2}, {"kind": "bad"}, {"kind": "good", "emit": 1}]},
        "not-a-record",
    ]

    result = source.normalize(
        payloads,
        source=SourceIdentity(
            kind="file", source_id="example:test", sha256=hashlib.sha256(b"test").hexdigest()
        ),
    )

    record_issues = [issue for issue in result.issues if issue.source_record.startswith("record-")]
    assert [issue.source_record for issue in record_issues] == ["record-1", "record-2"]
    assert seen_ordinals == [0, 2]

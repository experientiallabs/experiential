"""Disk preparation and bounded batches for immutable SQLite trace imports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import TemporaryFile
from typing import BinaryIO

from exp.common.core.artifacts import JsonObject, SourceIdentity, canonical_json_bytes
from exp.common.traces.trace import Trace

BATCH_BYTES = 1024 * 1024
BATCH_RECORDS = 64


@dataclass(frozen=True)
class ImportRecord:
    """One canonical record buffered for a shared-database transaction.

    Attributes:
        digest: Full SHA256 of the canonical trace without import provenance.
        trace_id: Canonical trace identity.
        payload: Canonical trace JSON.
        source: Per-import trace provenance JSON.
    """

    digest: str
    trace_id: str
    payload: str
    source: str


@dataclass(frozen=True)
class PreparedImport:
    """Private disk spool with a completed immutable identity.

    Attributes:
        import_id: Identity including all ordered records and normalization metadata.
        count: Number of records in the complete spool.
        file: Caller-owned temporary file, closed when preparation's context exits.
    """

    import_id: str
    count: int
    file: BinaryIO

    def batches(self) -> Iterator[tuple[ImportRecord, ...]]:
        """Yield bounded batches, admitting a large individual trace without truncation."""
        self.file.seek(0)
        batch: list[ImportRecord] = []
        size = 0
        for line in self.file:
            record = ImportRecord(*json.loads(line))
            record_size = len(line)
            if batch and (size + record_size > BATCH_BYTES or len(batch) >= BATCH_RECORDS):
                yield tuple(batch)
                batch = []
                size = 0
            batch.append(record)
            size += record_size
        if batch:
            yield tuple(batch)


@contextmanager
def prepare_import(
    source_format: str, source: SourceIdentity, traces: Iterable[Trace], metadata: JsonObject
) -> Iterator[PreparedImport]:
    """Hash and spool one pass without retaining the corpus or acquiring a shared writer lock.

    Canonical object keys follow canonical_json_bytes, so streaming and materialized
    import identities are identical. Buffer sizes affect throughput, never admission.
    """
    digest = hashlib.sha256()
    digest.update(b'{"metadata":' + canonical_json_bytes(metadata) + b',"records":[')
    with TemporaryFile() as spool:
        count = 0
        for trace in traces:
            payload = canonical_json_bytes(trace.model_dump(mode="json", exclude={"source"}))
            record_digest = hashlib.sha256(payload).hexdigest()
            provenance = canonical_json_bytes(trace.source)
            if count:
                digest.update(b",")
            digest.update(b'{"sha256":' + canonical_json_bytes(record_digest))
            digest.update(b',"source":' + provenance + b"}")
            spool.write(
                canonical_json_bytes(
                    [record_digest, trace.trace_id, payload.decode(), provenance.decode()]
                )
                + b"\n"
            )
            count += 1
        digest.update(b'],"source":' + canonical_json_bytes(source))
        digest.update(b',"source_format":' + canonical_json_bytes(source_format) + b"}")
        spool.flush()
        yield PreparedImport(f"import-{digest.hexdigest()[:20]}", count, spool)

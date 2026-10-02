"""Adapt retained gateway snapshots to shared disk-backed trace ingestion."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from pathlib import Path

from exp.common.core.artifacts import SourceIdentity, canonical_json_bytes
from exp.common.traces.ingest.chat_json import CHAT_JSON_SOURCE
from exp.common.traces.ingest.otlp import TraceNormalizationIssue
from exp.common.traces.ingest.persistence import persist_normalized_import
from exp.common.traces.ingest.streaming import (
    IngestSummary,
    NormalizedSource,
    normalization_workspace,
)
from exp.common.traces.sqlite import TraceImportReceipt
from exp.runtime.gateway.ingest.conversion import _conversation
from exp.runtime.gateway.ingest.metrics import measured_trace
from exp.runtime.gateway.local_capture import GATEWAY_CAPTURE_APPLICATION
from exp.runtime.gateway.local_capture_contracts import CapturedExchange, LocalCaptureScope
from exp.runtime.gateway.local_capture_store import LocalCaptureStore


def ingest_gateway_capture(
    project_id: str,
    *,
    root: Path,
    path: Path,
    identity_id: str,
    dry_run: bool = False,
) -> tuple[IngestSummary, TraceImportReceipt | None]:
    """Normalize one identity's retained gateway snapshot and publish a shared trace import.

    Args:
        project_id: Project namespace receiving the immutable import.
        root: Local workspace containing the destination content database.
        path: Existing gateway capture database supplying the source snapshot.
        identity_id: Explicit authenticated identity whose captures are selected.
        dry_run: Validate without publishing an import or creating destination storage.

    Returns:
        Normalization counts and exclusions, plus a receipt unless dry_run.
    """
    with normalized_gateway_capture(path, identity_id=identity_id) as normalized:
        return persist_normalized_import(
            project_id,
            root=root,
            source_format="gateway",
            normalized=normalized,
            dry_run=dry_run,
        )


@contextmanager
def normalized_gateway_capture(path: Path, *, identity_id: str) -> Iterator[NormalizedSource]:
    """Own one identity's frozen, normalized capture snapshot before shared writes.

    Args:
        path: Existing native capture database.
        identity_id: Explicit identity whose retained evidence is selected.

    Yields:
        Private normalized evidence for validation, publication, and scenario building.
    """
    with normalization_workspace("chat-json") as normalized:
        _stage_gateway(normalized, path, identity_id)
        yield normalized


def _stage_gateway(normalized: NormalizedSource, path: Path, identity_id: str) -> None:
    """Copy one retained capture snapshot before normalization or shared writes begin."""
    normalized.database.execute(
        "CREATE TABLE captured(sequence INTEGER PRIMARY KEY,response_id TEXT UNIQUE,payload TEXT)"
    )
    scope = LocalCaptureScope(user_id=identity_id, application_id=GATEWAY_CAPTURE_APPLICATION)
    digest = hashlib.sha256(b"[")
    count = 0
    with closing(LocalCaptureStore(path, scope).iter_snapshot()) as rows:
        for row in rows:
            if count:
                digest.update(b",")
            digest.update(canonical_json_bytes(row.experience))
            normalized.database.execute(
                "INSERT INTO captured VALUES (?,?,?)",
                (row.sequence, row.experience.response_id, row.experience.model_dump_json()),
            )
            count += 1
    digest.update(b"]")
    normalized.source = SourceIdentity(
        kind="production", source_id=f"gateway:{identity_id}", sha256=digest.hexdigest()
    )
    responses = _CapturedResponses(normalized.database)
    for (payload,) in normalized.database.execute("SELECT payload FROM captured ORDER BY sequence"):
        experience = CapturedExchange.model_validate_json(payload)
        try:
            normalized.vendor(CHAT_JSON_SOURCE, _conversation(experience, responses))
        except ValueError:
            normalized.issues.append(
                TraceNormalizationIssue(
                    experience.experience_id,
                    "Capture lacks supported effective context or a complete response; "
                    "collect fresh traffic with capture enabled.",
                )
            )
    normalized.finish(transform=measured_trace)


class _CapturedResponses(Mapping[str, CapturedExchange]):
    """Resolve response lineage from the private frozen capture snapshot on demand."""

    def __init__(self, database: sqlite3.Connection) -> None:
        """Bind the private snapshot used for every parent lookup."""
        self.database = database

    def __getitem__(self, key: str) -> CapturedExchange:
        """Load one parent response, preserving Mapping's missing-key contract."""
        row = self.database.execute(
            "SELECT payload FROM captured WHERE response_id=?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return CapturedExchange.model_validate_json(row[0])

    def __iter__(self) -> Iterator[str]:
        """Iterate response identities without retaining payloads."""
        return (row[0] for row in self.database.execute("SELECT response_id FROM captured"))

    def __len__(self) -> int:
        """Return the snapshot's response count."""
        return self.database.execute("SELECT COUNT(*) FROM captured").fetchone()[0]

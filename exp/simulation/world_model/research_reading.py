"""Immutable original-evidence archives, separate from compact recalled research notes."""

import json
from datetime import datetime

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, sha256_json
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.retrieval.evidence_reading import ReadingResult


def archive_reading(
    store: ArtifactStore,
    identity: str,
    result: ReadingResult,
    sources: tuple[ArtifactInput, ...],
    *,
    created_at: datetime,
    code_revision: str,
) -> JsonObject:
    """Save or exactly replay original passages and return compact recall metadata.

    Args:
        store: Immutable artifact owner.
        identity: Owning investigation's stable identity.
        result: Completed reader pass including rejected-source inspection coverage.
        sources: Full source closure of all inspected records, including rejected records.
        created_at: Publication time, irrelevant to replay of an existing identical packet.
        code_revision: Producing implementation.

    Returns:
        Compact reference and coverage. Original passages never recurse through recalled notes.

    Raises:
        ValueError: A packet or its source dependencies changed under a saved identity.
    """
    packet = result.model_dump(mode="json", exclude={"proposal"})
    inputs = tuple(sorted(set(sources), key=lambda ref: ref.artifact_id))
    for source in inputs:
        if artifact_input(store.read(source.artifact_id).manifest) != source:
            raise ValueError("reader source differs from its immutable reference")
    packet_id = identity + "-reading"
    if store.exists(packet_id):
        manifest = store.read(packet_id).manifest
        saved = json.loads(store.read_bytes(packet_id, "packet.json"))
        if (
            manifest.artifact_type != "investigation-evidence"
            or manifest.inputs != inputs
            or sha256_json(saved) != sha256_json(packet)
        ):
            raise ValueError("reading packet changed on resumption; inspect the saved evidence")
    else:
        manifest = store.write_json(
            artifact_id=packet_id,
            artifact_type="investigation-evidence",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=created_at, code_revision=code_revision, inputs=inputs
            ),
            files={"packet.json": packet},
        )
    return {
        "receipt": artifact_input(manifest).model_dump(mode="json"),
        "inspected_ids": list(result.inspected_ids),
        "selected_ids": [row.source_id for row in result.passages],
        "unread_count": len(result.unread_ids),
        "stop_reason": result.stop_reason,
    }

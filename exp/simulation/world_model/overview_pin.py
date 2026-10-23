"""The overview a memory version shows, fixed the first time the version is read."""

import json
from datetime import datetime

from exp.common.core.artifacts import ArtifactEnvelope, stable_id
from exp.common.project import ArtifactStore
from exp.simulation.world_model.experience_memory import (
    MemoryVersion,
    memory_lock,
    summary_generations,
)
from exp.simulation.world_model.memory_tree import summary_id


def pinned_summaries(
    store: ArtifactStore, version: MemoryVersion | None, *, created_at: datetime, code_revision: str
) -> frozenset[tuple[int, int, int]]:
    """The summaries a version's overview may show, fixed the first time the version is read.

    Compression runs after publication and may finish later, so the summaries that exist
    change over time. The first reader records the built ``(lo, hi, generation)`` summaries
    of the version's log; every later reader of the same version renders from exactly that
    set, so one checkpoint always delivers the same overview.

    Args:
        store: Project artifact owner.
        version: Pinned memory version, or None before any consumption.
        created_at: Pin creation time, recorded only by the first reader.
        code_revision: Recording implementation identity.

    Returns:
        Summaries this version's overview may show.
    """
    if version is None or not version.log_length:
        return frozenset()
    identity = stable_id(
        "memory-overview-pin", {"lineage": version.lineage, "version": version.version}
    )
    with memory_lock(store, version.lineage):
        if not store.exists(identity):
            generations = summary_generations(version)
            built: list[list[int]] = []
            size = 2
            while size <= version.log_length:
                for lo in range(0, version.log_length - size + 1, size):
                    generation = generations.get((lo, lo + size), 0)
                    if store.exists(summary_id(version.lineage, lo, lo + size, generation)):
                        built.append([lo, lo + size, generation])
                size *= 2
            store.write_json(
                artifact_id=identity,
                artifact_type="memory-overview-pin",
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=created_at, code_revision=code_revision
                ),
                files={"pin.json": {"version": version.version, "summaries": built}},
            )
    saved = json.loads(store.read_bytes(identity, "pin.json"))
    return frozenset((int(lo), int(hi), int(g)) for lo, hi, g in saved["summaries"])

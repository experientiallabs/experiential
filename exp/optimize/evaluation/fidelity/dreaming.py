"""Consolidate completed learning evidence before its single atomic memory publication."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from itertools import zip_longest
from typing import TYPE_CHECKING

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity.memory import trajectory_record
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.engines.transition import Completion
from exp.simulation.world_model.dreaming import DreamSource, consolidate_memory
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    memory_view,
    read_memory_version,
    reserved_task_lineages,
)
from exp.simulation.world_model.experience_sources import (
    persist_trajectory_source,
)
from exp.simulation.world_model.resumption import Resumption

if TYPE_CHECKING:
    from exp.optimize.evaluation.fidelity.runner import PreparedFidelity


def dream_pending(project: ProjectStore, prepared: PreparedFidelity) -> bool:
    """Whether a fully observed learning trace still needs its registered consolidation."""
    memory = prepared.spec.memory
    if (
        memory is None
        or memory.dreaming is None
        or project.artifacts.exists(prepared.id + "-dream")
    ):
        return False
    evidence_id = prepared.id + "-report-intent"
    if not project.artifacts.exists(evidence_id):
        return False
    evidence = json.loads(project.artifacts.read_bytes(evidence_id, "evidence.json"))
    receipts = evidence.get("sources", {}).get("online_feedback", {}).get("receipts", [])
    return evidence.get("stop_reason") == "completed" and len(receipts) == len(
        prepared.spec.workload.actions
    )


def run_dream(
    project: ProjectStore,
    prepared: PreparedFidelity,
    *,
    complete: Completion,
    decide: Callable[[DecisionRequest], JsonObject],
    clock: Callable[[], datetime],
    resumption: Resumption | None,
) -> None:
    """Use only committed prediction feedback and pinned prior memory for consolidation.

    Args:
        project: Owning project under the fidelity run's exclusive lease.
        prepared: Registered learning run, including consolidation budgets.
        complete: Existing admitted investigator completion.
        decide: Existing admitted decision reader.
        clock: Publication clock.
        resumption: Explicit recovery authority, if this run was interrupted.
    """
    if not dream_pending(project, prepared):
        return
    memory = prepared.spec.memory
    assert memory is not None and memory.dreaming is not None
    store = project.artifacts
    execution = json.loads(store.read_bytes(prepared.id + "-execution", "execution.json"))
    number = execution["memory"]["version"]
    parent = read_memory_version(store, memory.lineage, number) if number else None
    eligibility = Eligibility(
        strict=True, reserved_task_lineages=reserved_task_lineages(store, memory.lineage)
    )
    source = persist_trajectory_source(
        store,
        trajectory_record(prepared.spec),
        created_at=clock(),
        code_revision=prepared.code_revision,
    )
    evidence = json.loads(store.read_bytes(prepared.id + "-report-intent", "evidence.json"))
    receipts = evidence.get("sources", {}).get("online_feedback", {}).get("receipts", [])
    fresh: dict[str, DreamSource] = {}
    for index, (action, reply) in enumerate(
        zip(prepared.spec.workload.actions, prepared.spec.references.steps, strict=True)
    ):
        key = source.artifact_id + f"-step-{index}"
        fresh[key] = DreamSource(
            original={
                "id": key,
                "kind": "original-observation",
                "action": {"name": action.name, "arguments": action.arguments},
                "observed_reply": reply.reply,
                "task_lineage": prepared.spec.task.lineage_group_id,
                "authority": "Original disclosed observation of a completed historical instance",
            },
            sources=(source,),
        )
    for row in receipts:
        if isinstance(row, dict) and isinstance(row.get("investigation"), dict):
            dependencies = tuple(
                ArtifactInput.model_validate(ref) for ref in row.get("depends_on", [])
            )
            fresh[str(row["id"])] = DreamSource(original=row, sources=(source, *dependencies))
    view = memory_view(store, parent, eligibility)
    wiki = {
        key: DreamSource(original=row, sources=view.sources[key])
        for key, row in view.records.items()
        if row.get("kind") == "wiki-claim"
    }
    prior = {
        key: DreamSource(original=row, sources=view.sources[key])
        for key, row in view.records.items()
        if row.get("kind") != "wiki-claim"
    }
    # Alternate recent, consolidated and older evidence so large new traces cannot crowd out
    # every older claim and counterexample. Unselected originals remain explicitly unread.
    catalog = {}
    for group in zip_longest(
        reversed(fresh.items()), reversed(wiki.items()), reversed(prior.items())
    ):
        for item in group:
            if item is not None:
                catalog[item[0]] = item[1]
    consolidate_memory(
        store,
        prepared.id + "-dream",
        catalog,
        policy=memory.dreaming,
        eligibility=eligibility,
        complete=complete,
        decide=decide,
        clock=clock,
        code_revision=prepared.code_revision,
        resumption=resumption,
    )

"""Propose, select and publish behavioral edits through the existing shared library."""

import json
from collections.abc import Callable, Sequence
from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.optimize.simulation.library import (
    LibraryCase,
    LibraryEvaluation,
    LibrarySelection,
    select_library_edit,
)
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.reconstruction.library import LibraryEntry, LibrarySnapshot, extend_library
from exp.simulation.reconstruction.revision import ResearchNote


class LibraryTrainingCase(LibraryCase):
    """An exposed development example with explicit admitted-source identities.

    Attributes:
        source_trace_ids: Original episodes present in this example's supplied evidence.
            This is caller-declared provenance, not automatic semantic entailment.
    """

    source_trace_ids: tuple[str, ...] = Field(min_length=1)


class BehavioralEdit(ContractModel):
    """An evidenced semantic proposal, without authority to change the current world.

    Attributes:
        rules: Complete replacement rules, allowing additions, corrections and removals.
        rationale: Explanation of the observed failure and the proposed general correction.
        source_trace_ids: Admitted source episodes supporting the proposal.
    """

    rules: str
    rationale: str = Field(min_length=1)
    source_trace_ids: tuple[str, ...] = Field(min_length=1)


class LibraryEvolution(ContractModel):
    """Completed development attempt and its reusable, immutable library snapshot.

    Attributes:
        decision: Paired held-out/protected selection; never release approval.
        library: Child retaining the attempt and research, including rejected proposals.
        note: Investigation result, distinguishable from original native evidence.
    """

    decision: LibrarySelection
    library: LibrarySnapshot
    note: ResearchNote


def evolve_library(
    parent: SimulationKnowledge,
    *,
    library: LibrarySnapshot,
    store: ArtifactStore,
    run_id: str,
    training: Sequence[LibraryTrainingCase],
    held_out: Sequence[LibraryCase],
    protected: Sequence[LibraryCase],
    evaluator_sha256: Sha256,
    complete: Callable[[JsonObject, str], JsonObject],
    evaluate: Callable[[SimulationKnowledge, LibraryCase], LibraryEvaluation],
    record: Callable[[JsonObject], None],
    created_at: datetime,
    code_revision: str,
    minimum_gain: float = 0,
    supersedes: str | None = None,
) -> LibraryEvolution:
    """Inspect training failures, propose one edit, select and publish for future construction.

    Args:
        parent: Frozen knowledge; contracts, state and evidence stay outside semantic editing.
        library: Existing shared snapshot containing the source observations.
        store: Maintained artifact owner for requests, decisions and library references.
        run_id: Fresh registered attempt identity; repeated attempts require a new identity.
        training: Exposed development errors; only these logs reach the proposer.
        held_out: Reserved development selection cases, excluded from the proposal input.
        protected: Previously established behavior evaluated without permissible regressions.
        evaluator_sha256: Frozen execution/metric/judge identity, shared by both candidates.
        complete: Existing admitted structured completion with attempt/budget ownership.
        evaluate: Existing pipeline evaluator, not a second simulator or scoring loop.
        record: Durable event sink for registration, outputs and the selection decision.
        created_at: Aware knowledge-clock timestamp.
        code_revision: Source implementation identity.
        minimum_gain: Required strict held-out improvement margin.
        supersedes: Optional active rule entry replaced only after acceptance.

    Returns:
        Selected knowledge and a growing library usable by construct_world_model.
    """
    if sha256_json(json.loads(store.read_bytes(library.library_id, "library.json"))) != sha256_json(
        library
    ):
        raise ValueError("library differs from its persisted snapshot; reload the exact version")
    if not training or not parent.source_trace_ids:
        raise ValueError("editing requires training logs and admitted parent source episodes")
    if any(not set(case.source_trace_ids) <= set(parent.source_trace_ids) for case in training):
        raise ValueError("training cites episodes outside the admitted parent source pool")
    if supersedes is not None:
        if any(entry.supersedes == supersedes for entry in library.entries):
            raise ValueError(
                "library parent is already superseded; branch from its earlier snapshot"
            )
        old = next((entry for entry in library.entries if entry.entry_id == supersedes), None)
        if old is None or old.kind != "rule" or old.disposition != "accepted":
            raise ValueError("supersedes must identify an existing accepted rule entry")
        payload = json.loads(store.read_bytes(old.artifact.artifact_id, old.path))
        if not isinstance(payload, dict) or payload.get("knowledge") != parent.model_dump(
            mode="json"
        ):
            raise ValueError("superseded entry does not contain the exact active parent knowledge")
    selection_ids = {case.case_id for case in held_out}
    selection_lineages = {case.lineage_id for case in held_out}
    if (
        not held_out
        or not protected
        or selection_ids & {case.case_id for case in (*training, *protected)}
        or selection_lineages & {case.lineage_id for case in (*training, *protected)}
        or {case.case_id for case in training} & {case.case_id for case in protected}
        or {case.lineage_id for case in training} & {case.lineage_id for case in protected}
        or len({case.case_id for case in training}) != len(training)
        or len({c.case_id for c in (*held_out, *protected)}) != len(held_out) + len(protected)
        or not 0 <= minimum_gain <= 1
    ):
        raise ValueError("register distinct held-out lineages and nonempty protected cases")
    intent_id = stable_id("library-edit-intent", {"run_id": run_id})
    if store.exists(intent_id):
        raise ValueError(
            "library attempt already registered; replay is disabled. "
            "Inspect saved attempt evidence, "
            "reconcile unfinished I/O, then register a new run_id"
        )
    registration: JsonObject = {
        "run_id": run_id,
        "parent_sha256": parent.identity,
        "library_id": library.library_id,
        "training": [case.model_dump(mode="json") for case in training],
        "held_out": [case.model_dump(mode="json") for case in held_out],
        "protected": [case.model_dump(mode="json") for case in protected],
        "evaluator_sha256": evaluator_sha256,
        "minimum_gain": minimum_gain,
        "supersedes": supersedes,
        "authority": "Case/source inventories and evaluator identity are caller-declared. "
        "The source-ID checks do not prove semantic support or detect facts hidden in prose.",
    }
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        inputs=(artifact_input(store.read(library.library_id).manifest),),
    )
    store.write_json(
        artifact_id=intent_id,
        artifact_type="library-edit-intent",
        envelope=envelope,
        files={"registration.json": registration},
    )
    record({"library_edit_registration": registration})
    exposed = []
    for case in training:
        outcome = evaluate(parent, case)
        exposed.append(
            {"case": case.model_dump(mode="json"), "outcome": outcome.model_dump(mode="json")}
        )
    body: JsonObject = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Revise behavioral rules using only the admitted training evidence. "
                    "Return complete replacement rules and source episode IDs. Remove unsupported "
                    "claims rather than inventing certainty. Preserve unrelated behavior. "
                    "Historical "
                    "entity values are examples, never current-instance values. "
                    "Research hypotheses "
                    "are not native truth. Example: an unconditional duration-preservation "
                    "rule that "
                    "fails when an end time is explicitly given needs an omitted-versus-explicit "
                    "condition, not a special case for the failed event ID. "
                    "Do not infer this example's "
                    "calendar behavior for an unrelated target."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"parent": parent.model_dump(mode="json"), "training": exposed}
                ),
            },
        ],
        "response_schema": BehavioralEdit.model_json_schema(),
    }
    proposal = BehavioralEdit.model_validate(complete(body, intent_id + ":proposal"))
    supplied = {source for case in training for source in case.source_trace_ids}
    if not set(proposal.source_trace_ids) <= supplied:
        raise ValueError("proposal cites episodes absent from its supplied training evidence")
    candidate = parent.model_copy(update={"rules": proposal.rules})
    receipts: list[JsonObject] = []

    def retain(value: JsonObject) -> None:
        """Keep the same paired events in the immutable attempt and external ledger."""
        receipts.append(value)
        record(value)

    decision = select_library_edit(
        parent,
        candidate,
        training=training,
        held_out=held_out,
        protected=protected,
        evaluator_sha256=evaluator_sha256,
        evaluate=evaluate,
        record=retain,
        minimum_gain=minimum_gain,
    )
    attempt_id = stable_id(
        "library-edit", {"intent": intent_id, "proposal": proposal.model_dump(mode="json")}
    )
    investigation: JsonObject = {
        "registration": registration,
        "proposal": proposal.model_dump(mode="json"),
        "training_outcomes": exposed,
        "paired_evaluations": receipts,
        "decision": decision.model_dump(mode="json"),
    }
    note = ResearchNote(
        id=attempt_id,
        content=proposal.rationale
        + " Development decision: "
        + ("accepted" if decision.accepted else "rejected")
        + "; held-out gain="
        + str(decision.held_out_gain)
        + "; protected regressions="
        + str(decision.protected_regressions),
        investigation_sha256=sha256_json(investigation),
        source_trace_ids=proposal.source_trace_ids,
        disposition="accepted" if decision.accepted else "rejected",
    )
    manifest = store.write_json(
        artifact_id=attempt_id,
        artifact_type="behavioral-library-edit",
        envelope=envelope,
        files={
            "investigation.json": investigation,
            "knowledge.json": {
                "knowledge": candidate.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            },
            "research.json": note.model_dump(mode="json"),
        },
    )
    ref = artifact_input(manifest)
    tools = parent.contracts.get("tools", [])
    cues = (
        tuple(
            tool["name"]
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        )
        if isinstance(tools, list)
        else ()
    )
    additions = (
        LibraryEntry(
            entry_id=attempt_id + ":rules",
            kind="rule",
            artifact=ref,
            path="knowledge.json",
            payload_sha256=sha256_bytes(store.read_bytes(attempt_id, "knowledge.json")),
            disposition=note.disposition,
            cues=cues,
            supersedes=supersedes if decision.accepted else None,
        ),
        LibraryEntry(
            entry_id=attempt_id + ":research",
            kind="research",
            artifact=ref,
            path="research.json",
            payload_sha256=sha256_bytes(store.read_bytes(attempt_id, "research.json")),
            disposition=note.disposition,
            cues=(proposal.rationale,),
        ),
    )
    child = extend_library(
        store, additions, parent=library, created_at=created_at, code_revision=code_revision
    )
    return LibraryEvolution(decision=decision, library=child, note=note)

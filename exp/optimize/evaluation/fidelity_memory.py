"""Memory lifecycle of a fidelity run: partition checks, publication, maintenance, inspection.

The run's single owner, ``execute_fidelity_evaluation``, calls these stages in order. They
read the saved registration and evaluation artifacts only, so replay can complete them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.runtime.sessions.store import SessionStore
from exp.simulation.world_model.experience_memory import (
    ReservedTaskError,
    consumed_task_lineages,
    is_reserved,
    materialize_log,
    memory_head,
    publish_trajectory,
    read_memory_version,
    retired_lines,
    summary_generations,
)
from exp.simulation.world_model.memory_tree import (
    LineKind,
    ambiguous_blocks,
    build_pending,
    fit_line,
    refused_blocks,
)

if TYPE_CHECKING:
    from exp.optimize.evaluation.fidelity import FidelitySpec, PreparedFidelity


INCOMPLETE = "incomplete trajectory is not consumed"
DUPLICATE = "duplicate trajectory already consumed"


def saved_memory_receipt(project: ProjectStore, prepared: PreparedFidelity) -> JsonObject | None:
    """Read the memory version a completed learning trajectory produced, if any."""
    identity = prepared.id + "-memory"
    if not project.artifacts.exists(identity):
        return None
    return json.loads(project.artifacts.read_bytes(identity, "memory.json"))


def consume_trajectory(project: ProjectStore, prepared: PreparedFidelity, now: datetime) -> None:
    """Publish a completed learning trajectory exactly once; replay resumes publication.

    Deterministic code decides admission: only a learning spec whose every planned step
    was predicted and disclosed is consumed. Model-authored investigations enter as
    unresolved research, never as accepted rules.
    """
    spec = prepared.spec
    receipt_id = prepared.id + "-memory"
    if (
        spec.memory is None
        or spec.memory.mode != "learn"
        or spec.online_research is None
        or project.artifacts.exists(receipt_id)
    ):
        return
    pinned = json.loads(project.artifacts.read_bytes(prepared.id + "-execution", "execution.json"))
    memory = pinned.get("memory")
    if not isinstance(memory, dict) or memory.get("lineage") != spec.memory.lineage:
        raise ValueError("learning execution did not pin its memory version")
    evidence = json.loads(
        project.artifacts.read_bytes(prepared.id + "-report-intent", "evidence.json")
    )
    feedback = evidence.get("sources", {}).get("online_feedback", {})
    receipts = [row for row in feedback.get("receipts", []) if isinstance(row, dict)]
    research = [
        str(row["id"])
        for row in receipts
        if isinstance(row.get("investigation"), dict) or row.get("reply_matched") is False
    ]
    outcome: JsonObject = {
        "lineage": spec.memory.lineage,
        "parent_version": memory["version"],
        "research_ids": list(research),
        "feedback_receipts": len(receipts),
    }
    if evidence.get("stop_reason") != "completed" or len(receipts) != len(spec.workload.actions):
        outcome |= {"published": False, "reason": INCOMPLETE}
    else:
        parent_number = memory["version"]
        parent = (
            read_memory_version(project.artifacts, spec.memory.lineage, int(parent_number))
            if parent_number
            else None
        )
        try:
            version, consumed = publish_trajectory(
                project.artifacts,
                lineage=spec.memory.lineage,
                parent=parent,
                world_model=spec.world_model,
                selection_policy_sha256=str(spec.memory_selection_identity),
                trajectory=trajectory_record(spec),
                research=research,
                lines=memory_lines(spec, receipts),
                created_at=now,
                code_revision=prepared.code_revision,
                publisher=prepared.id,
            )
        except ReservedTaskError as error:
            outcome |= {"published": False, "reason": str(error)}
        else:
            outcome |= {
                "published": consumed,
                "reason": "consumed" if consumed else DUPLICATE,
                "version": version.version,
                "consumed_trajectories": len(version.consumed),
                "log_length": version.log_length,
                "library": version.library.model_dump(mode="json"),
            }
    project.artifacts.write_json(
        artifact_id=receipt_id,
        artifact_type="fidelity-memory",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=now,
            code_revision=prepared.code_revision,
            inputs=(artifact_input(project.artifacts.read(prepared.id + "-report").manifest),),
        ),
        files={"memory.json": outcome},
    )


def memory_lines(spec: FidelitySpec, receipts: list[JsonObject]) -> list[tuple[LineKind, str, str]]:
    """Derive this trajectory's log lines: one observed record, then each transferable rule.

    The observed line is deterministic code over public actions only. It never records how
    well the simulator predicted: those predictions read evidence retrieved from other
    lineages, so a match count would carry that evidence back into memory without its
    provenance. Mismatches stay in the investigation records, which carry it.
    Only investigations the investigator marked transferable become rule lines, phrased by the
    observable cue that triggers them; every investigation stays recallable as a record.
    """
    names = [action.name for action in spec.workload.actions]
    observed = (
        f"[observed] task {spec.task.lineage_group_id}: {len(names)} actions ({', '.join(names)})"
    )
    lines: list[tuple[LineKind, str, str]] = [
        ("observed", fit_line(observed), spec.task.lineage_group_id)
    ]
    for row in receipts:
        investigation = row.get("investigation")
        if not isinstance(investigation, dict):
            continue
        if investigation.get("transferable") is not True:
            continue
        superseded = investigation.get("supersedes") or []
        text = (
            f"[unresolved rule] when {investigation.get('observable_cue')}: "
            f"{investigation.get('hypothesis')}"
        )
        if superseded:
            text += f" (supersedes {', '.join(str(item) for item in superseded)})"
        lines.append(("hypothesis", fit_line(text), str(row["id"])))
    return lines


def maintain_memory(
    project: ProjectStore,
    prepared: PreparedFidelity,
    complete: Callable[[JsonObject, str], JsonObject],
    now: datetime,
) -> None:
    """After a fresh learning run, buy at most the registered number of compressions.

    Runs only where a provider is available; replay never compresses, and readers open
    unbuilt blocks instead. Failures are recorded rather than raised, because the report
    and the memory version are already durable.
    """
    spec = prepared.spec
    identity = prepared.id + "-memory-maintenance"
    receipt = saved_memory_receipt(project, prepared)
    if (
        spec.memory is None
        or spec.memory.mode != "learn"
        or receipt is None
        or not receipt.get("published")
        or project.artifacts.exists(identity)
    ):
        return
    head = memory_head(project.artifacts, spec.memory.lineage)
    materialize_log(project.artifacts, head)
    total = 0 if head is None else head.log_length or 0
    generations = summary_generations(head)
    outcome: JsonObject = {"log_length": total}
    try:
        built, remaining = build_pending(
            project.artifacts,
            spec.memory.lineage,
            total,
            complete,
            limit=spec.memory.maximum_compressions,
            created_at=now,
            code_revision=prepared.code_revision,
            generations=generations,
            retired=retired_lines(project.artifacts, head),
        )
        outcome |= {
            "built": list(built),
            "pending": remaining,
            "refused": refused_blocks(project.artifacts, spec.memory.lineage, total, generations),
            "ambiguous": ambiguous_blocks(
                project.artifacts, spec.memory.lineage, total, generations
            ),
        }
    except Exception as error:  # noqa: BLE001 - durable report must not be undone.
        outcome |= {"error_type": type(error).__name__, "error": str(error)}
    project.artifacts.write_json(
        artifact_id=identity,
        artifact_type="fidelity-memory-maintenance",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=now, code_revision=prepared.code_revision
        ),
        files={"maintenance.json": outcome},
    )


def check_partition(project: ProjectStore, spec: FidelitySpec) -> None:
    """Refuse learning a reserved task lineage, or reserving one already learned."""
    memory = spec.memory
    if memory is None:
        return
    task_lineage = spec.task.lineage_group_id
    if memory.mode == "learn" and is_reserved(project.artifacts, memory.lineage, task_lineage):
        raise ValueError(
            f"task lineage {task_lineage!r} is reserved for evaluation in memory "
            f"{memory.lineage!r}; it cannot be learned"
        )
    if memory.reserved and task_lineage in consumed_task_lineages(
        project.artifacts, memory_head(project.artifacts, memory.lineage)
    ):
        raise ValueError(
            f"task lineage {task_lineage!r} was already consumed by memory "
            f"{memory.lineage!r}; it cannot serve as reserved evaluation"
        )


def trajectory_record(spec: FidelitySpec) -> JsonObject:
    """Public task, context and actions plus the disclosed replies consumed after prediction."""
    return {
        "task": spec.task.instruction,
        "task_lineage": spec.task.lineage_group_id,
        "initial_context": spec.task.initial_context,
        "actions": [{"name": a.name, "arguments": a.arguments} for a in spec.workload.actions],
        "observed_replies": [row.reply for row in spec.references.steps],
        "source": None if spec.online_research is None else spec.online_research.feedback_source,
    }


def fidelity_memory_plan(
    project: ProjectStore, spec: FidelitySpec, prepared: PreparedFidelity | None = None
) -> str | None:
    """Describe the memory a run read (once executed) or would read, without provider calls."""
    if spec.memory is None:
        return None
    intent = None if prepared is None else prepared.id + "-execution"
    if intent is not None and project.artifacts.exists(intent):
        pinned = json.loads(project.artifacts.read_bytes(intent, "execution.json")).get("memory")
        if isinstance(pinned, dict):
            return (
                f"{spec.memory.lineage} {pinned.get('mode')} run read v{pinned.get('version')} "
                f"({pinned.get('consumed_trajectories')} consumed trajectories)"
            )
    if spec.memory.checkpoint is not None:
        frozen = read_memory_version(project.artifacts, spec.memory.lineage, spec.memory.checkpoint)
        return (
            f"{spec.memory.lineage} frozen checkpoint v{frozen.version} "
            f"({len(frozen.consumed)} consumed trajectories); outcomes never update memory"
        )
    head = memory_head(project.artifacts, spec.memory.lineage)
    count = 0 if head is None else len(head.consumed)
    return (
        f"{spec.memory.lineage} learning from v{0 if head is None else head.version} "
        f"({count} consumed trajectories); a completed trajectory publishes the next version"
    )


def fidelity_decision_trace(project: ProjectStore, prepared: PreparedFidelity) -> list[JsonObject]:
    """Return per-step model decisions and deterministic commits from saved artifacts only."""
    path = project.artifacts.project_directory / (prepared.id + "-sessions.sqlite")
    if not path.exists():
        return []
    state = SessionStore(path).read_session(prepared.id).state
    transitions = state.get("transitions")
    receipts: dict[str, JsonObject] = {}
    evidence_id = prepared.id + "-report-intent"
    if project.artifacts.exists(evidence_id):
        evidence = json.loads(project.artifacts.read_bytes(evidence_id, "evidence.json"))
        feedback = evidence.get("sources", {}).get("online_feedback", {})
        receipts = {
            str(row["operation_id"]): row
            for row in feedback.get("receipts", [])
            if isinstance(row, dict)
        }
    rows: list[JsonObject] = []
    for index, transition in enumerate(transitions if isinstance(transitions, list) else []):
        if not isinstance(transition, dict):
            continue
        evidence = transition.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        selection = evidence.get("session_research_selection")
        feedback_row = receipts.get(str(transition.get("operation_id")), {})
        investigation = feedback_row.get("investigation")
        rows.append(
            {
                "step": index,
                "committed_version": index + 1,
                "request_sha256": evidence.get("request_sha256"),
                "originals": evidence.get("grounded_transition_ids"),
                "memory_selected": selection.get("selected_ids")
                if isinstance(selection, dict)
                else None,
                "proposal": transition.get("proposal"),
                "reply_matched": feedback_row.get("reply_matched"),
                "research_id": feedback_row.get("id") if isinstance(investigation, dict) else None,
                "hypothesis": investigation.get("hypothesis")
                if isinstance(investigation, dict)
                else None,
            }
        )
    return rows


def saved_maintenance_receipt(
    project: ProjectStore, prepared: PreparedFidelity
) -> JsonObject | None:
    """Read the compression outcome a learning run recorded, if any."""
    identity = prepared.id + "-memory-maintenance"
    if not project.artifacts.exists(identity):
        return None
    return json.loads(project.artifacts.read_bytes(identity, "maintenance.json"))

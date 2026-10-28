"""Explicit observation-only settlement of finalized failed fidelity investigations.

Repair never redraws predictions or investigations. It retains the original report and paid
responses, records an operator's decision to abstain from the missing hypotheses, and consumes
the disclosed observations against an explicitly named memory head.
"""

import json
from datetime import UTC, datetime
from typing import Literal

from filelock import FileLock
from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    stable_id,
    strict_equal,
)
from exp.common.evaluations.reconstruction import WorkflowEvidence
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.fidelity.memory import (
    DUPLICATE,
    INCOMPLETE,
    check_partition,
    memory_lines,
    trajectory_record,
)
from exp.optimize.evaluation.fidelity.runner import PreparedFidelity, saved_fidelity_report
from exp.simulation.world_model.experience_memory import (
    MemoryVersion,
    consumed_trajectory_digest,
    memory_head,
    publish_trajectory,
    read_memory_version,
    require_memory_scope,
)
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


class LearningRepairSpec(ContractModel):
    """Operator-authorized abstention from failed hypotheses without replacing measurements.

    Attributes:
        run: Exact finalized fidelity registration.
        report: Original scored report, retained byte-for-byte.
        parent: Explicit current memory version to append to, or None for an empty lineage.
        reason: Operator's explanation for retaining observations without failed hypotheses.
        resolution: Observation-only settlement; no generated replacement is implied.
    """

    run: ArtifactInput
    report: ArtifactInput
    parent: ArtifactInput | None = None
    reason: str = Field(min_length=1, max_length=4000)
    resolution: Literal["retain-observations"] = "retain-observations"


def repair_learning(
    project: ProjectStore,
    spec: LearningRepairSpec,
    *,
    code_revision: str,
    dry_run: bool = False,
) -> JsonObject:
    """Settle failed research explicitly with no model calls or session-state mutation.

    The original prediction report, failed investigation responses and original memory receipt
    remain immutable. An advanced head must be supplied explicitly and is checked by the memory
    owner during publication. Restart recovers its own publication without consuming twice.

    Args:
        project: Existing project containing the finalized run and its memory lineage.
        spec: Exact source references, publication parent and operator disposition.
        code_revision: Producing implementation identity.
        dry_run: Validate only; no artifacts, reservations or memory versions are written.

    Returns:
        Explicit publication outcome, or a validated dry-run description.
    """
    store = project.artifacts
    with FileLock(store.project_directory / (spec.run.artifact_id + ".lock"), timeout=0):
        for ref in (spec.run, spec.report, *((spec.parent,) if spec.parent else ())):
            if artifact_input(store.read(ref.artifact_id).manifest) != ref:
                raise ValueError("learning repair source changed; supply its exact saved reference")
        prepared = PreparedFidelity.model_validate_json(
            store.read_bytes(spec.run.artifact_id, "plan.json")
        )
        if (
            prepared.id != spec.run.artifact_id
            or spec.report.artifact_id != prepared.id + "-report"
        ):
            raise ValueError("learning repair report must belong to its exact fidelity run")
        if not spec.reason.strip():
            raise ValueError("learning repair needs a nonblank operator reason")
        identity = stable_id("learning-repair", spec)
        if store.exists(identity):
            saved = LearningRepairSpec.model_validate_json(store.read_bytes(identity, "spec.json"))
            if saved != spec:
                raise ValueError("learning repair already registered with a different disposition")
        result_id = prepared.id + "-learning-repair-result"
        if store.exists(result_id):
            outcome = json.loads(store.read_bytes(result_id, "memory.json"))
            if outcome.get("repair") != identity:
                raise ValueError("learning repair already settled with a different disposition")
            return outcome
        original = json.loads(store.read_bytes(prepared.id + "-memory", "memory.json"))
        policy = prepared.spec.memory
        if (
            policy is None
            or policy.mode != "learn"
            or prepared.spec.online_research is None
            or original.get("reason") != INCOMPLETE
            or saved_fidelity_report(project, prepared) is None
        ):
            raise ValueError("repair requires finalized, unconsumed learning with failed research")
        evidence = WorkflowEvidence.model_validate_json(
            store.read_bytes(prepared.id + "-report-intent", "evidence.json")
        )
        if (
            evidence.stop_reason != "research_update_failure"
            or len(evidence.steps) != len(prepared.spec.workload.actions)
            or any(step.status != "valid" for step in evidence.steps)
        ):
            raise ValueError("learning repair requires every original prediction to be committed")
        references = prepared.spec.references.steps
        if len(references) != len(evidence.steps) or any(
            not row.reply_available for row in references
        ):
            raise ValueError("learning repair requires a recorded reply for every committed step")
        feedback = evidence.sources.get("online_feedback")
        rows = feedback.get("receipts") if isinstance(feedback, dict) else None
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("failed learning did not retain its feedback receipt inventory")
        receipts: list[JsonObject] = [row for row in rows if isinstance(row, dict)]
        by_operation = {str(row["operation_id"]): row for row in receipts}
        if len(by_operation) != len(receipts):
            raise ValueError("failed learning contains repeated feedback operations")
        parent = (
            None
            if spec.parent is None
            else MemoryVersion.model_validate_json(
                store.read_bytes(spec.parent.artifact_id, "memory.json")
            )
        )
        if parent is not None and parent.lineage != policy.lineage:
            raise ValueError("learning repair parent belongs to another memory lineage")
        check_partition(project, prepared.spec)
        require_memory_scope(
            parent,
            world_model=prepared.spec.world_model,
            artifact=load_grounded_world_model_artifact(store, prepared.spec.world_model),
            selection_policy_sha256=str(prepared.spec.memory_selection_identity),
        )
        missing = []
        for step, action, reference in zip(
            evidence.steps, prepared.spec.workload.actions, references, strict=True
        ):
            if step.action != {"name": action.name, "arguments": action.arguments}:
                raise ValueError("learning repair action differs from its original scored step")
            existing = by_operation.get(action.call_id)
            if existing is not None:
                if not strict_equal(existing.get("observed_reply"), reference.reply):
                    raise ValueError("saved feedback differs from the recorded observation")
                if not strict_equal(existing.get("predicted_reply"), step.reply):
                    raise ValueError("saved feedback differs from the original prediction")
                continue
            missing.append(step.step)
        if not missing:
            raise ValueError("learning repair has no missing feedback to settle")
        head = memory_head(store, policy.lineage)
        digest = consumed_trajectory_digest(trajectory_record(prepared.spec))
        recovered = (
            head is not None
            and digest in head.consumed
            and read_memory_version(
                store, policy.lineage, head.consumed.index(digest) + 1
            ).publisher
            == identity
        )
        if head is not None and digest in head.consumed and not recovered:
            previous = read_memory_version(store, policy.lineage, head.consumed.index(digest) + 1)
            if previous.publisher is not None and store.exists(previous.publisher):
                manifest = store.read(previous.publisher).manifest
                if manifest.artifact_type == "fidelity-learning-repair":
                    earlier = LearningRepairSpec.model_validate_json(
                        store.read_bytes(previous.publisher, "spec.json")
                    )
                    if earlier.run == spec.run:
                        raise ValueError(
                            "recover the original published repair before changing its disposition"
                        )
        if head != parent and not recovered:
            raise ValueError("memory advanced; explicitly name its current head before repair")
        if dry_run:
            return {"validated": True, "missing_steps": missing, "provider_calls": 0}
        now = datetime.now(UTC)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=now,
            code_revision=code_revision,
            inputs=tuple(
                sorted(
                    {spec.run, spec.report, *((spec.parent,) if spec.parent else ())},
                    key=lambda ref: ref.artifact_id,
                )
            ),
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="fidelity-learning-repair",
                envelope=envelope,
                files={"spec.json": spec},
            )
        repair_ref = artifact_input(store.read(identity).manifest)
        version, consumed = publish_trajectory(
            store,
            lineage=policy.lineage,
            parent=parent,
            world_model=prepared.spec.world_model,
            selection_policy_sha256=str(prepared.spec.memory_selection_identity),
            trajectory=trajectory_record(prepared.spec),
            research=[
                str(row["id"])
                for row in receipts
                if isinstance(row.get("investigation"), dict) or row.get("reply_matched") is False
            ],
            lines=memory_lines(prepared.spec, receipts),
            created_at=now,
            code_revision=code_revision,
            publisher=identity,
        )
        outcome: JsonObject = {
            "published": consumed,
            "reason": "consumed" if consumed else DUPLICATE,
            "lineage": policy.lineage,
            "version": version.version,
            "parent_version": 0 if parent is None else parent.version,
            "original_parent_version": original.get("parent_version"),
            "repair": identity,
            "resolution": spec.resolution,
            "repaired_steps": missing,
            "provider_calls": 0,
            "original_report": spec.report.model_dump(mode="json"),
        }
        store.write_json(
            artifact_id=result_id,
            artifact_type="fidelity-learning-repair-result",
            envelope=envelope.model_copy(update={"inputs": (repair_ref,)}),
            files={"memory.json": outcome},
        )
        return outcome

"""Explicit operator settlements of learning runs that cannot complete on their own.

Repair never redraws predictions or investigations. Observation retention keeps the original
report and paid responses, records an operator's decision to abstain from the missing
hypotheses, and consumes the disclosed observations against an explicitly named memory head.
Trace retirement records an operator's decision never to learn one named trace: it stays
unscored, every report counts it as not learned, and memory is not edited
(:mod:`exp.optimize.evaluation.learning.retirement`).
"""

import json
from datetime import UTC, datetime
from typing import Literal

from filelock import Timeout
from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    stable_id,
    strict_equal,
)
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.reconstruction import WorkflowEvidence
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.source_access import refuse_known_ineligible_sources
from exp.optimize.evaluation.fidelity.memory import (
    DUPLICATE,
    INCOMPLETE,
    check_partition,
    memory_lines,
    trajectory_record,
)
from exp.optimize.evaluation.fidelity.replay import DEADLINE
from exp.optimize.evaluation.fidelity.runner import PreparedFidelity, saved_fidelity_report
from exp.optimize.evaluation.learning.retirement import (
    TraceRetirement,
    retire,
    retirement_of,
    trace_held,
    trace_refusals,
)
from exp.optimize.evaluation.learning.traces import (
    Scope,
    learning_cases,
    learning_plan,
    learning_spec,
    recorded_plan,
    registered_attempts,
)
from exp.simulation.world_model.experience_memory import (
    MemoryVersion,
    memory_head,
    publish_trajectory,
    read_memory_version,
    require_memory_scope,
)
from exp.simulation.world_model.experience_sources import (
    consumed_trajectory_digest,
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


class TraceRetirementRequest(ContractModel):
    """Operator retirement of one named learning trace that will never be learned.

    Attributes:
        resolution: Selects trace retirement in the ``repair-learning`` action.
        scope: Learning memory to retire it from: the audit's ``fit`` memory or ``all``.
        trace_id: Trace to retire; it must be in scope, unlearned and without a saved score.
        reason: Operator's explanation, recorded with the retirement.
    """

    resolution: Literal["retire-trace"]
    scope: Scope = "fit"
    trace_id: str = Field(min_length=1, max_length=512)
    reason: str = Field(min_length=1, max_length=4000)


def retire_trace(
    project: ProjectStore,
    request: TraceRetirementRequest,
    *,
    code_revision: str,
    dry_run: bool = False,
) -> JsonObject:
    """Retire one named trace from a learning memory, with no model calls and no memory edit.

    The trace must be one of the scope's learning traces (held-out traces are never learned,
    so they are never retired), not learned, without a saved measurement, and not running in
    another process. Refusals recorded for it are attached as evidence. Retiring it again with
    the same reason replays.

    Args:
        project: Project with a recorded learning plan in ``request.scope``.
        request: The trace and the operator's reason.
        code_revision: Producing implementation identity.
        dry_run: Validate only; nothing is written.

    Returns:
        The recorded retirement, or a validated dry-run description.

    Raises:
        ValueError: The trace cannot be retired; the message says why.
    """
    if not request.reason.strip():
        raise ValueError("trace retirement needs a nonblank operator reason")
    if not recorded_plan(project, request.scope):
        raise ValueError(f"no {request.scope} learning is recorded; learn before retiring")
    plan = learning_plan(project, scope=request.scope)
    case = next(
        (row for row in learning_cases(project, plan.scope) if row.trace_id == request.trace_id),
        None,
    )
    if case is None:
        raise ValueError(
            f"trace {request.trace_id} is not a {request.scope} learning trace of this build; "
            "held-out traces are never learned, so they cannot be retired"
        )
    saved = retirement_of(project, plan.lineage, case.trace_id)
    if saved is not None and saved.cause == "operator" and saved.reason == request.reason:
        return _retired(saved, replayed=True)
    # Held through the record, so no learning pass starts the trace between check and commit.
    with trace_held(project, plan.lineage, case.trace_id):
        head = memory_head(project.artifacts, plan.lineage)
        spec = learning_spec(project, plan, case, deadline=datetime.now(UTC) + DEADLINE)
        digest = consumed_trajectory_digest(trajectory_record(spec))
        if head is not None and digest in head.consumed:
            raise ValueError(f"trace {case.trace_id} is already learned into memory {plan.lineage}")
        attempts = registered_attempts(project, plan, case)
        if any(saved_fidelity_report(project, prepared) is not None for prepared in attempts):
            raise ValueError(
                f"trace {case.trace_id} already has a saved learning measurement; it is counted as "
                "learned or not learned already and cannot be retired"
            )
        for prepared in attempts:
            lock = CoordinationLock(project.artifacts.project_directory / (prepared.id + ".lock"))
            try:
                lock.acquire(timeout=0)
            except Timeout as error:
                raise ValueError(
                    f"trace {case.trace_id} is being learned by another process; retry once "
                    "it stops"
                ) from error
            lock.release()
        refusals = trace_refusals(project, plan.lineage, case.trace_id)
        retirement = TraceRetirement(
            lineage=plan.lineage,
            trace_id=case.trace_id,
            cause="operator",
            reason=request.reason,
            run=attempts[-1].id if attempts else None,
            refusals=refusals,
            head_version=0 if head is None else head.version,
        )
        if dry_run:
            return {"validated": True, **_retired(retirement, replayed=False)}
        retire(
            project,
            retirement,
            now=datetime.now(UTC),
            code_revision=code_revision,
            unconsumed=digest,
        )
    return _retired(retirement, replayed=False)


def _retired(retirement: TraceRetirement, *, replayed: bool) -> JsonObject:
    """The JSON result of a trace retirement."""
    return {
        "retired": retirement.trace_id,
        "lineage": retirement.lineage,
        "cause": retirement.cause,
        "reason": retirement.reason,
        "refusals": len(retirement.refusals),
        "replayed": replayed,
        "provider_calls": 0,
    }


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
    with CoordinationLock(store.project_directory / (spec.run.artifact_id + ".lock"), timeout=0):
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
        # The repaired memory cites the run and report, so an excluded one never enters it.
        refuse_known_ineligible_sources(store, (spec.run, spec.report), maximum_artifacts=100_000)
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
            evidence=(spec.run, spec.report),
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

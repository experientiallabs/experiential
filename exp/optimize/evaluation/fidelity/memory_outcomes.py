"""Record later predictions exposed to checked memory claims, without promoting those claims."""

import json
from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    strict_equal,
)
from exp.common.evaluations.reconstruction import (
    DualReport,
    EvaluationReferences,
    WorkflowEvidence,
    turn_outcomes,
)
from exp.common.project import ArtifactStore, artifact_input


class DeliveredClaim(ContractModel):
    """An exact full claim supplied in the predictor's required memory channels.

    Attributes:
        id: Original immutable wiki record identity.
        sha256: Digest of the complete record delivered before prediction.
    """

    id: str
    sha256: Sha256


class ClaimDelivery(ContractModel):
    """Memory exposure attached to a committed prediction, before reference feedback.

    Attributes:
        step: Planned workflow slot.
        operation_id: Tool call identity in the registered workload.
        request_sha256: Exact framed predictor request digest.
        claims: Full wiki records delivered directly, excluding indirect summary exposure.
    """

    step: int = Field(ge=0)
    operation_id: str
    request_sha256: Sha256
    claims: tuple[DeliveredClaim, ...]


class ClaimPredictionOutcome(ContractModel):
    """One later measured prediction with the original claim it could read.

    Attributes:
        claim: Immutable checked historical hypothesis, not a promoted behavioral fact.
        step: Measured workflow slot.
        request_sha256: Prospectively recorded request digest.
        exact_reply: Recorded-reply equality, or None when no reference exists.
        fields_matched: Matching leaf fields under the saved workflow's reference.
        fields_total: Complete reply leaf-field denominator.
    """

    claim: ArtifactInput
    step: int
    request_sha256: Sha256
    exact_reply: bool | None
    fields_matched: int | None
    fields_total: int | None


class MemoryOutcomeReport(ContractModel):
    """Observational claim-use ledger with an explicit denominator and measurement scope.

    Attributes:
        evaluation: Saved fidelity report from which all labels are recomputed.
        pinned_memory: Exact execution memory descriptor, or None for a run without one.
        planned_steps: Every workflow slot, including failures and blocked suffixes.
        recorded_delivery_steps: Committed steps with explicit memory delivery evidence.
        outcomes: Every directly delivered claim and that step's independently recorded reply.
        scope: No causal attribution, automatic confirmation, retirement or memory publication.
    """

    evaluation: ArtifactInput
    pinned_memory: JsonObject | None
    planned_steps: int
    recorded_delivery_steps: tuple[int, ...]
    outcomes: tuple[ClaimPredictionOutcome, ...]
    scope: str = (
        "observed outcomes after direct full-claim delivery; not evidence of claim use or causal "
        "benefit; summary-only exposure and uncommitted predictions excluded; no rule promotion"
    )


def record_memory_outcomes(
    store: ArtifactStore, run_id: str, *, created_at: datetime, code_revision: str
) -> ArtifactInput | None:
    """Join prospectively saved claim deliveries to independent reply measurements once.

    The ledger never enters memory catalogs or dream evidence. Held-out measurements therefore
    remain evaluation data, including when this audit record is created for a frozen run.
    Older reports without delivery metadata remain explicitly unsupported and are not inferred.
    """
    report_id = run_id + "-report"
    intent_id = report_id + "-intent"
    evidence = WorkflowEvidence.model_validate_json(store.read_bytes(intent_id, "evidence.json"))
    raw = evidence.audit_metadata.get("memory_claim_deliveries")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("memory delivery inventory is not an ordered list")
    deliveries = tuple(ClaimDelivery.model_validate(row) for row in raw)
    positions = tuple(row.step for row in deliveries)
    if positions != tuple(sorted(set(positions))):
        raise ValueError("memory deliveries repeat or reorder prediction slots")
    report = DualReport.model_validate_json(store.read_bytes(report_id, "report.json"))
    references = EvaluationReferences.model_validate_json(
        store.read_bytes(intent_id, "references.json")
    )
    if report.evidence_sha256 != sha256_json(evidence) or report.reference_sha256 != sha256_json(
        references
    ):
        raise ValueError("memory outcomes do not match the saved evaluation inputs")
    turns = turn_outcomes(evidence, references)
    evaluation = artifact_input(store.read(report_id).manifest)
    dependencies = {evaluation, artifact_input(store.read(intent_id).manifest)}
    pinned: JsonObject | None = None
    execution_id = run_id + "-execution"
    if store.exists(execution_id):
        execution = json.loads(store.read_bytes(execution_id, "execution.json"))
        memory = execution.get("memory")
        pinned = memory if isinstance(memory, dict) else None
        dependencies.add(artifact_input(store.read(execution_id).manifest))
    outcomes = []
    for delivery in deliveries:
        if delivery.step >= len(evidence.steps) or evidence.steps[delivery.step].status != "valid":
            raise ValueError("memory delivery does not name a committed prediction")
        if len({claim.id for claim in delivery.claims}) != len(delivery.claims):
            raise ValueError("memory delivery repeats the same claim")
        step = evidence.steps[delivery.step]
        expected = references.steps[delivery.step] if references.steps else None
        exact = (
            None
            if expected is None or not expected.reply_available
            else strict_equal(step.reply, expected.reply)
        )
        turn = turns[delivery.step]
        for claim in delivery.claims:
            record = json.loads(store.read_bytes(claim.id, "record.json"))
            if record.get("kind") != "wiki-claim" or sha256_json(record) != claim.sha256:
                raise ValueError("delivered memory claim differs from its original full record")
            reference = artifact_input(store.read(claim.id).manifest)
            dependencies.add(reference)
            outcomes.append(
                ClaimPredictionOutcome(
                    claim=reference,
                    step=delivery.step,
                    request_sha256=delivery.request_sha256,
                    exact_reply=exact,
                    fields_matched=turn.fields_matched,
                    fields_total=turn.fields_total,
                )
            )
    result = MemoryOutcomeReport(
        evaluation=evaluation,
        pinned_memory=pinned,
        planned_steps=len(evidence.steps),
        recorded_delivery_steps=positions,
        outcomes=tuple(outcomes),
    )
    identity = run_id + "-memory-outcomes"
    sources = tuple(sorted(dependencies, key=lambda ref: ref.artifact_id))
    if store.exists(identity):
        if (
            MemoryOutcomeReport.model_validate_json(store.read_bytes(identity, "outcomes.json"))
            != result
            or store.read(identity).manifest.inputs != sources
        ):
            raise ValueError("saved memory outcome ledger no longer reproduces")
    else:
        store.write_json(
            artifact_id=identity,
            artifact_type="memory-prediction-outcomes",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=created_at, code_revision=code_revision, inputs=sources
            ),
            files={"outcomes.json": result},
        )
    return artifact_input(store.read(identity).manifest)

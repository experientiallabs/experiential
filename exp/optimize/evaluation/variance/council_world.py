"""Native per-job simulator authority inside one qualified physical council envelope."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import cache
from typing import Literal

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    saved_continuation_job,
)
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.provider_grant import ProviderLimits
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec, _pick
from exp.optimize.evaluation.variance.council_checkpoint import CouncilCheckpoint, point_job
from exp.optimize.evaluation.variance.council_initial import saved_initial_feedback
from exp.optimize.evaluation.variance.council_inventory import (
    CouncilPhaseInventory,
    phase_inventory,
)
from exp.optimize.evaluation.variance.council_qualification import (
    QUALIFICATION_TYPE,
    QualifiedCouncil,
    council_sources,
)
from exp.optimize.evaluation.variance.evidence import load_references
from exp.optimize.evaluation.variance.evidence_tools import TRANSCRIPT_TYPE, AnalysisTranscript
from exp.optimize.evaluation.variance.role_prompts import assign_prompts
from exp.optimize.evaluation.variance.roles import validate_judge
from exp.runtime.models.scoped_spending import (
    SpendEnvelopeRef,
    VerifiedSpendEnvelope,
    verify_spend_envelope,
)
from exp.runtime.models.spend_ceiling import current_spend_ceiling
from exp.simulation.continuations import load_continuation_checkpoint

INVENTORY_TYPE = "council-native-phase-inventory"
RUNTIME_GRANT_TYPE = "council-runtime-grant"


def _qualified(project: ProjectStore, reference: ArtifactInput) -> QualifiedCouncil:
    """Read exact immutable domain authority, never a caller-authored replacement job."""
    stored = project.artifacts.read(reference.artifact_id)
    value = QualifiedCouncil.model_validate_json(stored.payloads["qualification.json"])
    if (
        artifact_input(stored.manifest) != reference
        or stored.manifest.artifact_type != QUALIFICATION_TYPE
        or stored.manifest.code_revision != value.producer_revision
        or value.contract.sources != council_sources()
    ):
        raise ValueError("runtime authority lost its exact qualified council producer")
    return value


def _open() -> VerifiedSpendEnvelope:
    """Refuse stale, unscoped or different-operation runtime construction."""
    ceiling = current_spend_ceiling()
    if ceiling is None or ceiling.envelope is None:
        raise ValueError("council runtime requires its active native spend envelope")
    return verify_spend_envelope(ceiling, ceiling.envelope)


def save_council_inventory(
    project: ProjectStore,
    qualification: ArtifactInput,
    spec: ActorCriticSpec,
    checkpoint: CouncilCheckpoint,
    *,
    code_revision: str,
) -> ArtifactInput:
    """Price and persist the actual native wave before any reservation or client creation."""
    qualified = _qualified(project, qualification)
    if code_revision != qualified.producer_revision or spec not in qualified.contract.studies:
        raise ValueError("inventory is outside the exact qualified study")
    inventory = phase_inventory(project, spec, checkpoint, code_revision=code_revision)
    identity = stable_id(
        INVENTORY_TYPE,
        {
            "qualification": qualification.model_dump(mode="json"),
            "inventory": inventory.model_dump(mode="json"),
            "producer": code_revision,
        },
    )
    inputs = tuple(
        sorted(
            {
                qualification,
                *inventory.replayed,
                *(job.study.checkpoint for job in checkpoint.jobs),
                *([checkpoint.group] if checkpoint.group else []),
                *([checkpoint.initial_judged] if checkpoint.initial_judged else []),
            },
            key=lambda ref: ref.artifact_id,
        )
    )
    if not project.artifacts.exists(identity):
        project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=INVENTORY_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=inputs,
            ),
            files={"inventory.json": inventory},
        )
    return artifact_input(project.artifacts.read(identity).manifest)


def _inventory(
    project: ProjectStore,
    reference: ArtifactInput,
    qualification: ArtifactInput,
    qualified: QualifiedCouncil,
) -> tuple[CouncilPhaseInventory, ActorCriticSpec]:
    """Require a genuine saved native wave and its exact configured study."""
    stored = project.artifacts.read(reference.artifact_id)
    inventory = CouncilPhaseInventory.model_validate_json(stored.payloads["inventory.json"])
    spec = next(
        (row for row in qualified.contract.studies if sha256_json(row) == inventory.spec_sha256),
        None,
    )
    if (
        artifact_input(stored.manifest) != reference
        or stored.manifest.artifact_type != INVENTORY_TYPE
        or stored.manifest.code_revision != qualified.producer_revision
        or qualification not in stored.manifest.inputs
        or spec is None
        or inventory.checkpoint.run_identity != spec.identity(qualified.producer_revision)
    ):
        raise ValueError("runtime authority requires the exact native wave inventory")
    return inventory, spec


def _selected_jobs(
    project: ProjectStore,
    inventory: CouncilPhaseInventory,
    spec: ActorCriticSpec,
    *,
    code_revision: str,
) -> None:
    """Independently bind suffix jobs to actual selectors and the seeded group's checkpoints."""
    checkpoint = inventory.checkpoint
    group = saved_continuation_job(project, spec.initial, code_revision=code_revision)
    if group is None or checkpoint.group != continuation_result_reference(
        project, spec.initial, code_revision=code_revision
    ):
        raise ValueError("suffix runtime has no exact committed initial group")
    loader = cache(lambda ref: load_continuation_checkpoint(project.artifacts, ref))
    _, packet = _pick(
        group,
        spec.identity(code_revision),
        spec,
        loader,
        load_references(project.artifacts, spec.references),
    )
    if checkpoint.initial_judged is None:
        raise ValueError("suffix runtime is missing the actual initial LLM feedback")
    assert checkpoint.group is not None
    packet = saved_initial_feedback(
        project, spec, checkpoint.group, checkpoint.initial_judged, packet
    )
    chosen: set[int] = set()
    prompts = assign_prompts(spec.selector_prompts, seed=spec.seed, draws=len(spec.judges))
    for draw, prompt in enumerate(prompts):
        key = f"{checkpoint.run_identity}-judge-{draw}-transcript"
        if not project.artifacts.exists(key):
            raise ValueError("suffix runtime is missing a mandatory selector receipt")
        stored = project.artifacts.read(key)
        transcript = AnalysisTranscript.model_validate_json(stored.payloads["transcript.json"])
        if (
            stored.manifest.artifact_type != TRANSCRIPT_TYPE
            or stored.manifest.code_revision != code_revision
            or transcript.profile != spec.selector_profile
            or transcript.prompt != prompt
            or transcript.answer_tool != "choose_points"
        ):
            raise ValueError("suffix runtime selector differs from its exact native draw")
        if transcript.stop != "answered" or transcript.failure is not None:
            continue
        try:
            selection = validate_judge(transcript.answer, packet, points=spec.points)
        except ValueError:
            continue
        chosen.update(point.step for point in selection.points)
    if (
        tuple(sorted(chosen)) != checkpoint.selected
        or len(set(checkpoint.wave)) != len(checkpoint.wave)
        or not set(checkpoint.wave) <= chosen - set(checkpoint.completed)
        or checkpoint.jobs != tuple(point_job(spec, packet.pool.points[p]) for p in checkpoint.wave)
    ):
        raise ValueError("suffix runtime changed the selected native checkpoint or required group")


class CouncilRuntimeRecord(ContractModel):
    """One immutable job's first authority; a later wave cannot refill the same world owner."""

    protocol: Literal["council-runtime-authority-v1"] = "council-runtime-authority-v1"
    qualification: ArtifactInput
    inventory: ArtifactInput
    envelope: SpendEnvelopeRef
    job: ContinuationJob
    limits: ProviderLimits
    producer_revision: str


class CouncilRuntimeGrant(ContractModel):
    """Simulator-only operational allowance, not an executable-trial receipt or world override."""

    registration: ArtifactInput
    limits: ProviderLimits

    def identity_payload(self) -> JsonObject:
        """Keep the saved registration as this job's durable physical-world identity."""
        return self.model_dump(mode="json")

    def require_configuration(
        self,
        project: ProjectStore,
        configuration: JsonObject,
        *,
        runtime_sha256: Sha256 | None = None,
    ) -> None:
        """Verify the first native grant and its currently active exact paid wave."""
        stored = project.artifacts.read(self.registration.artifact_id)
        record = CouncilRuntimeRecord.model_validate_json(stored.payloads["grant.json"])
        opened = _open()
        if (
            artifact_input(stored.manifest) != self.registration
            or stored.manifest.artifact_type != RUNTIME_GRANT_TYPE
            or stored.manifest.code_revision != record.producer_revision
            or record.limits != self.limits
            or record.envelope != opened.reference
            or opened.policy.execution != record.qualification
            or opened.plan.inventory != record.inventory
            or configuration != record.job.configuration
            or (runtime_sha256 is not None and runtime_sha256 != record.job.runtime_sha256)
        ):
            raise ValueError("council runtime authority differs from its exact job or native wave")
        qualified = _qualified(project, record.qualification)
        inventory, spec = _inventory(project, record.inventory, record.qualification, qualified)
        if (
            record.producer_revision != qualified.producer_revision
            or record.producer_revision != opened.policy.producer_revision
            or record.job not in inventory.checkpoint.jobs
            or datetime.now(UTC) >= record.limits.deadline
            or record.limits.deadline > opened.policy.deadline
            or record.limits.deadline > record.job.deadline
        ):
            raise ValueError("council runtime lost its qualified job, producer or lifecycle")
        runtime = GroundedRuntimeDefinition.model_validate(configuration)
        if record.limits.maximum_physical_attempts != runtime.maximum_physical_attempts:
            raise ValueError("council runtime cannot change its frozen world allowance")
        if inventory.checkpoint.phase == "actors":
            if inventory.checkpoint.jobs != (spec.initial,) or record.job != spec.initial:
                raise ValueError("initial runtime cannot substitute another actor group")
        elif inventory.checkpoint.phase == "continuations":
            _selected_jobs(project, inventory, spec, code_revision=record.producer_revision)
        else:
            raise ValueError("this native phase has no admitted runtime job")


def register_council_runtime(
    project: ProjectStore,
    qualification: ArtifactInput,
    inventory: ArtifactInput,
    job: ContinuationJob,
    *,
    code_revision: str,
) -> CouncilRuntimeGrant:
    """Register exact per-job authority under a genuine open original-operation envelope.

    The domain allocator owns the operation lock and the native cost/future-work gate.
    An existing job grant cannot be re-keyed under another envelope after partial execution.
    """
    opened = _open()
    qualified = _qualified(project, qualification)
    priced, spec = _inventory(project, inventory, qualification, qualified)
    if (
        opened.policy.execution != qualification
        or opened.plan.inventory != inventory
        or opened.policy.producer_revision != code_revision
        or qualified.producer_revision != code_revision
        or job not in priced.checkpoint.jobs
        or opened.plan.maximum_attempts < priced.attempts
        or opened.plan.maximum_known_usd < priced.known_usd
    ):
        raise ValueError("per-job runtime has no exact admitted native wave")
    if priced.checkpoint.phase == "actors":
        if priced.checkpoint.jobs != (spec.initial,) or job != spec.initial:
            raise ValueError("initial runtime cannot substitute another actor group")
    elif priced.checkpoint.phase == "continuations":
        _selected_jobs(project, priced, spec, code_revision=code_revision)
    else:
        raise ValueError("this native phase has no admitted runtime job")
    runtime = GroundedRuntimeDefinition.model_validate(job.configuration)
    limits = ProviderLimits(
        maximum_physical_attempts=runtime.maximum_physical_attempts,
        deadline=min(opened.policy.deadline, job.deadline),
    )
    record = CouncilRuntimeRecord(
        qualification=qualification,
        inventory=inventory,
        envelope=opened.reference,
        job=job,
        limits=limits,
        producer_revision=code_revision,
    )
    identity = stable_id(
        RUNTIME_GRANT_TYPE,
        {
            "qualification": qualification.model_dump(mode="json"),
            "job": job.model_dump(mode="json"),
            "producer": code_revision,
        },
    )
    if project.artifacts.exists(identity):
        stored = project.artifacts.read(identity)
        if CouncilRuntimeRecord.model_validate_json(stored.payloads["grant.json"]) != record:
            raise ValueError("registered council runtime cannot refill or change its native wave")
    else:
        project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=RUNTIME_GRANT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(
                    sorted(
                        {qualification, inventory, job.study.checkpoint},
                        key=lambda ref: ref.artifact_id,
                    )
                ),
            ),
            files={"grant.json": record},
        )
    result = CouncilRuntimeGrant(
        registration=artifact_input(project.artifacts.read(identity).manifest), limits=limits
    )
    result.require_configuration(project, job.configuration, runtime_sha256=job.runtime_sha256)
    return result

"""Native initial-job runtime authority before full council studies exist."""

from __future__ import annotations

from datetime import UTC, datetime

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.provider_grant import ProviderLimits
from exp.optimize.evaluation.variance.bootstrap_source import current_bootstrap_execution_source
from exp.optimize.evaluation.variance.council_prelude import (
    PRELUDE_TYPE,
    QualifiedCouncilPrelude,
    _save,
    prelude_sources,
)
from exp.optimize.evaluation.variance.council_prelude_hook import CouncilPreludeCheckpoint
from exp.optimize.evaluation.variance.council_prelude_inventory import (
    CouncilPreludeInventory,
    prelude_inventory,
)
from exp.optimize.evaluation.variance.council_preparation import _pinned
from exp.optimize.evaluation.variance.council_world import _open
from exp.runtime.models.scoped_spending import SpendEnvelopeRef

INVENTORY_TYPE = "council-native-prelude-runtime-inventory"
RUNTIME_TYPE = "council-prelude-runtime-grant"


def _qualified(
    project: ProjectStore, reference: ArtifactInput, revision: str
) -> QualifiedCouncilPrelude:
    """Read the immutable proof already bound by the original ledger's parent policy."""
    _pinned(project, reference, PRELUDE_TYPE)
    stored = project.artifacts.read(reference.artifact_id)
    qualified = QualifiedCouncilPrelude.model_validate_json(stored.payloads["qualification.json"])
    if (
        stored.manifest.code_revision != revision
        or qualified.producer_revision != revision
        or qualified.contract.sources != prelude_sources()
        or qualified.contract.executor_source_sha256
        != current_bootstrap_execution_source(revision).source_sha256
    ):
        raise ValueError("prelude runtime lost its exact qualified source or producer")
    required = {
        qualified.preparation,
        qualified.resume_binding,
        qualified.task_start,
        qualified.contract.preparation.exposure,
        qualified.contract.preparation.family_simulator,
    }
    if not required.issubset(stored.manifest.inputs):
        raise ValueError("prelude runtime lost its exact preparation ancestry")
    return qualified


def _inputs(
    qualification: ArtifactInput, inventory: CouncilPreludeInventory
) -> tuple[ArtifactInput, ...]:
    """Bind every actual job checkpoint and native completed owner used by the quote."""
    checkpoint = inventory.checkpoint
    return tuple(
        sorted(
            {
                qualification,
                *(job.study.checkpoint for job in inventory.jobs),
                *inventory.replayed,
                *checkpoint.completed_labels,
                *checkpoint.completed_error_points,
                *(() if checkpoint.group is None else (checkpoint.group,)),
                *(() if checkpoint.completed_rubric is None else (checkpoint.completed_rubric,)),
            },
            key=lambda ref: ref.artifact_id,
        )
    )


def _identity(
    qualification: ArtifactInput, inventory: CouncilPreludeInventory, revision: str
) -> str:
    """Key the complete native inventory and exact qualified prelude."""
    return stable_id(
        INVENTORY_TYPE,
        {
            "qualification": qualification.model_dump(mode="json"),
            "inventory": inventory.model_dump(mode="json"),
            "producer": revision,
        },
    )


def save_prelude_inventory(
    project: ProjectStore,
    qualification: ArtifactInput,
    checkpoint: CouncilPreludeCheckpoint,
    *,
    code_revision: str,
) -> ArtifactInput:
    """Persist the actual native prelude quote before opening its physical spend envelope."""
    qualified = _qualified(project, qualification, code_revision)
    inventory = prelude_inventory(project, qualified, checkpoint)
    return _save(
        project,
        _identity(qualification, inventory, code_revision),
        INVENTORY_TYPE,
        "inventory.json",
        inventory,
        _inputs(qualification, inventory),
        code_revision,
    )


def _inventory(
    project: ProjectStore, qualification: ArtifactInput, reference: ArtifactInput, revision: str
) -> CouncilPreludeInventory:
    """Independently recompute the native quote and reject substitutions or completed jobs."""
    qualified = _qualified(project, qualification, revision)
    _pinned(project, reference, INVENTORY_TYPE)
    stored = project.artifacts.read(reference.artifact_id)
    inventory = CouncilPreludeInventory.model_validate_json(stored.payloads["inventory.json"])
    if (
        stored.manifest.code_revision != revision
        or reference.artifact_id != _identity(qualification, inventory, revision)
        or stored.manifest.inputs != _inputs(qualification, inventory)
        or inventory != prelude_inventory(project, qualified, inventory.checkpoint)
    ):
        raise ValueError("prelude runtime inventory changed its exact native owner or quote")
    return inventory


class PreludeRuntimeRecord(ContractModel):
    """Durable initial-job world allowance under one actual native prelude wave.

    Attributes:
        qualification: Exact registered prelude proof and parent execution identity.
        inventory: Recomputed native quote for this initial-job wave.
        envelope: Exact current original-ledger envelope.
        job: Entire frozen initial job, including actor, checkpoint and runtime.
        limits: Unchanged world ceiling and bounded operational deadline.
        producer_revision: Exact current reviewed executor.
    """

    qualification: ArtifactInput
    inventory: ArtifactInput
    envelope: SpendEnvelopeRef
    job: ContinuationJob
    limits: ProviderLimits
    producer_revision: str


class PreludeRuntimeGrant(ContractModel):
    """Exact native initial-job authority accepted by the trusted frozen runtime.

    Attributes:
        registration: Immutable native world-grant receipt, never a fabricated trial.
        limits: Exact world ceiling and finite registered lifecycle.
    """

    registration: ArtifactInput
    limits: ProviderLimits

    def identity_payload(self) -> JsonObject:
        """Preserve this job's durable physical-world identity across repeated restores."""
        return self.model_dump(mode="json")

    def require_job(self, project: ProjectStore, job: ContinuationJob) -> None:
        """Refuse a different actor, checkpoint, runtime or count before source import."""
        record = PreludeRuntimeRecord.model_validate_json(
            project.artifacts.read_bytes(self.registration.artifact_id, "grant.json")
        )
        if record.job != job:
            raise ValueError("prelude runtime authority belongs to another exact incoming job")
        self.require_configuration(project, job.configuration, runtime_sha256=job.runtime_sha256)

    def require_configuration(
        self,
        project: ProjectStore,
        configuration: JsonObject,
        *,
        runtime_sha256: Sha256 | None = None,
    ) -> None:
        """Require the exact initial job and its still-active original paid wave."""
        opened = _open()
        stored = project.artifacts.read(self.registration.artifact_id)
        record = PreludeRuntimeRecord.model_validate_json(stored.payloads["grant.json"])
        if (
            artifact_input(stored.manifest) != self.registration
            or stored.manifest.artifact_type != RUNTIME_TYPE
            or stored.manifest.code_revision != record.producer_revision
            or record.limits != self.limits
            or opened.reference != record.envelope
            or opened.policy.execution != record.qualification
            or opened.policy.producer_revision != record.producer_revision
            or opened.plan.inventory != record.inventory
            or configuration != record.job.configuration
            or (runtime_sha256 is not None and runtime_sha256 != record.job.runtime_sha256)
        ):
            raise ValueError("prelude runtime authority differs from its exact job or native wave")
        inventory = _inventory(
            project, record.qualification, record.inventory, record.producer_revision
        )
        runtime = GroundedRuntimeDefinition.model_validate(configuration)
        if (
            inventory.phase not in ("traceml_group", "behavior_reference")
            or inventory.jobs != (record.job,)
            or record.limits.maximum_physical_attempts != runtime.maximum_physical_attempts
            or not datetime.now(UTC)
            < record.limits.deadline
            <= min(opened.policy.deadline, record.job.deadline)
        ):
            raise ValueError("prelude runtime lost its initial job or finite world allowance")


def register_prelude_runtime(
    project: ProjectStore,
    qualification: ArtifactInput,
    reference: ArtifactInput,
    job: ContinuationJob,
    *,
    code_revision: str,
) -> PreludeRuntimeGrant:
    """Grant only a genuinely pending native initial job within its exact paid envelope."""
    opened = _open()
    inventory = _inventory(project, qualification, reference, code_revision)
    if (
        opened.policy.execution != qualification
        or opened.plan.inventory != reference
        or opened.policy.producer_revision != code_revision
        or inventory.phase not in ("traceml_group", "behavior_reference")
        or inventory.jobs != (job,)
        or opened.plan.maximum_attempts < inventory.attempts
        or opened.plan.maximum_known_usd < inventory.known_usd
        or opened.plan.remaining_minimum_attempts < inventory.remaining.attempts
        or opened.plan.requirements_sha256 != inventory.contract_sha256
    ):
        raise ValueError("runtime job has no exact admitted native prelude")
    runtime = GroundedRuntimeDefinition.model_validate(job.configuration)
    record = PreludeRuntimeRecord(
        qualification=qualification,
        inventory=reference,
        envelope=opened.reference,
        job=job,
        limits=ProviderLimits(
            maximum_physical_attempts=runtime.maximum_physical_attempts,
            deadline=min(opened.policy.deadline, job.deadline),
        ),
        producer_revision=code_revision,
    )
    identity = stable_id(
        RUNTIME_TYPE,
        {
            "qualification": qualification.model_dump(mode="json"),
            "job": job.model_dump(mode="json"),
            "producer": code_revision,
        },
    )
    if project.artifacts.exists(identity):
        stored = project.artifacts.read(identity)
        if PreludeRuntimeRecord.model_validate_json(stored.payloads["grant.json"]) != record:
            raise ValueError("prelude runtime cannot refill or change its native wave")
    reference = _save(
        project,
        identity,
        RUNTIME_TYPE,
        "grant.json",
        record,
        (qualification, record.inventory, job.study.checkpoint),
        code_revision,
    )
    grant = PreludeRuntimeGrant(registration=reference, limits=record.limits)
    grant.require_job(project, job)
    return grant

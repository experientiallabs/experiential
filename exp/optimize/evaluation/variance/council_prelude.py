"""Frozen conditional admission for native council work before a full study exists.

Prelude quotes use native jobs, rubric profiles, judging and tagging bounds. They do not create
a rubric, run a provider, fabricate an actor-critic registration, or qualify training records.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.models import RoutedCandidateSnapshot, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    preflight_continuation_job,
)
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.fidelity.retrieval_binding import index_embedder_alias
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap import resolve_bootstrap
from exp.optimize.evaluation.variance.bootstrap_source import current_bootstrap_execution_source
from exp.optimize.evaluation.variance.council_preparation import (
    CouncilPreparationContract,
    _pinned,
    qualify_prepared_council,
)
from exp.optimize.evaluation.variance.council_qualification import (
    CONSENT_TYPE,
    CouncilConditionalConsent,
    configured_council_actors,
    council_sources,
)
from exp.optimize.evaluation.variance.pipeline_behaviors import behavior_bank
from exp.optimize.evaluation.variance.pipeline_build import BUILD_STAGE
from exp.optimize.evaluation.variance.pipeline_errors import tier_error_bank
from exp.optimize.evaluation.variance.pipeline_family_rag import native_world_catalog
from exp.optimize.evaluation.variance.pipeline_inputs import local_id
from exp.optimize.evaluation.variance.pipeline_spec import (
    COUNCIL_ACTORS,
    DecisionVarianceSpec,
)
from exp.optimize.evaluation.variance.pipeline_stages import (
    group_job,
)
from exp.optimize.evaluation.variance.rubric_author import load_rubric
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.scoped_spending import (
    ParentSpendPolicy,
    ParentSpendRef,
    register_parent_policy,
)
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.mining.traceml import traceml_bank
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

PRELUDE_TYPE = "conditional-council-prelude-qualification"
PRELUDE_INVENTORY_TYPE = "council-native-prelude-inventory"
_PRELUDE_SOURCES = (
    "council_preparation.py",
    "council_prelude.py",
    "council_prelude_inventory.py",
    "council_prelude_world.py",
    "council_prelude_execution.py",
    "pipeline_spec.py",
    "pipeline_stages.py",
    "pipeline_inputs.py",
    "pipeline_errors.py",
    "pipeline_behaviors.py",
    "rubric_agent.py",
    "rubric_critic.py",
    "behavior_pairs.py",
    "behavior_differences.py",
    "council_descendant.py",
)


def prelude_sources() -> dict[str, Sha256]:
    """Freeze the actual prelude owners as well as the reviewed council execution sources."""
    root = Path(__file__).parent
    return {
        **council_sources(),
        **{
            f"exp/optimize/evaluation/variance/{name}": sha256_bytes((root / name).read_bytes())
            for name in _PRELUDE_SOURCES
        },
    }


def prelude_banks(pipeline: DecisionVarianceSpec) -> dict[str, Sha256]:
    """Bind complete native question-bank content rather than mutable directory names."""
    banks: dict[str, Sha256] = {}
    if pipeline.traceml.enabled:
        banks["traceml"] = sha256_json(traceml_bank(pipeline.traceml.bank_root))
        errors = tier_error_bank(pipeline)
        if errors is not None:
            banks["errors"] = sha256_json(errors)
    behavior = behavior_bank(pipeline)
    if behavior is not None:
        banks["behavior"] = sha256_json(behavior)
    return banks


class CouncilPreludeContract(ContractModel):
    """Frozen native pipeline and genuine task inputs before rubric or labels exist.

    Attributes:
        preparation: Exact native preparation, start, exposure and family scope.
        pipeline: Actual full native pipeline settings, including roles, prompts and seed.
        actor_model_ids: Explicit five declared open model IDs, separate from alias metadata.
        banks: Complete frozen local tagging-bank content digests.
        sources: Reviewed native prelude and execution-source fingerprints.
        executor_source_sha256: Actual full installed source of the new council executor.
    """

    preparation: CouncilPreparationContract
    pipeline: DecisionVarianceSpec
    actor_model_ids: tuple[str, ...] = Field(default=COUNCIL_ACTORS, min_length=5, max_length=5)
    banks: dict[str, Sha256]
    sources: dict[str, Sha256]
    executor_source_sha256: Sha256


class QualifiedCouncilPrelude(ContractModel):
    """Read-only actual preparation and five native initial jobs, without spending authority.

    Attributes:
        contract: Exact frozen prelude and future-study settings.
        preparation: Genuine completed native preparation.
        resume_binding: Exact original physical-ledger ancestry.
        task_start: Native selected task-start and mandatory QA registration.
        jobs: Separate native group job for each of the five configured actor models.
        actors: Actual configured snapshots in pipeline actor order.
        producer_revision: Reviewed actual executor.
    """

    contract: CouncilPreludeContract
    preparation: ArtifactInput
    resume_binding: ArtifactInput
    task_start: ArtifactInput
    jobs: tuple[ContinuationJob, ...] = Field(min_length=5, max_length=5)
    actors: tuple[RoutedCandidateSnapshot, ...] = Field(min_length=5, max_length=5)
    producer_revision: str


class ConditionalCouncilPreludeRegistration(ContractModel):
    """A genuine native prelude proof and the unchanged original ledger's parent policy.

    Attributes:
        qualification: Saved native prelude qualification carrying the real pipeline and jobs.
        policy: Additive original-ledger policy, retained for later full-study descendants.
    """

    qualification: ArtifactInput
    policy: ParentSpendRef


def freeze_council_prelude(
    preparation: CouncilPreparationContract,
    pipeline: DecisionVarianceSpec,
    *,
    actor_model_ids: tuple[str, ...] = COUNCIL_ACTORS,
    code_revision: str,
) -> CouncilPreludeContract:
    """Freeze exact native pipeline settings, model declarations, bank content and producer."""
    return CouncilPreludeContract(
        preparation=preparation,
        pipeline=pipeline,
        actor_model_ids=actor_model_ids,
        banks=prelude_banks(pipeline),
        sources=prelude_sources(),
        executor_source_sha256=current_bootstrap_execution_source(code_revision).source_sha256,
    )


def qualify_council_prelude(
    project: ProjectStore,
    prepared: ArtifactInput,
    resume_binding: ArtifactInput,
    spec: CouncilPreludeContract,
    *,
    code_revision: str,
) -> QualifiedCouncilPrelude:
    """Verify native preparation and exact pending jobs before any rubric or study exists."""
    proof = qualify_prepared_council(
        project,
        prepared,
        resume_binding,
        spec.preparation,
        code_revision=code_revision,
    )
    pipeline = spec.pipeline
    mining = pipeline.mining
    if (
        mining.bootstrap != spec.preparation.bootstrap
        or mining.family_simulator != spec.preparation.family_simulator
        or mining.job not in (None, proof.start.job)
        or pipeline.build.saved_report != proof.build_report
        or mining.schedule is not None
        or (mining.attempts, mining.selector_draws, mining.critic_draws, mining.continuations)
        != (16, 8, 8, 8)
        or not 1 <= mining.points <= 4
        or proof.start.job.maximum_model_calls != 20
        or mining.continuation_physical_attempts < mining.continuations
    ):
        raise ValueError("prelude changed its native task or mandatory G16/J8/C8/K<=4/M8/H20")
    if (
        spec.sources != prelude_sources()
        or spec.banks != prelude_banks(pipeline)
        or spec.executor_source_sha256
        != current_bootstrap_execution_source(code_revision).source_sha256
    ):
        raise ValueError("native prelude sources, prompts, protocols or tagging banks changed")
    configured_council_actors(
        project,
        aliases=mining.actors,
        model_ids=spec.actor_model_ids,
    )
    resolved = resolve_bootstrap(project, pipeline)
    costs, _ = BUILD_STAGE.plan(project, resolved, code_revision)
    if costs:
        raise ValueError("council preludes must reuse the genuine saved preparation build")
    jobs = tuple(group_job(project, resolved, proof.start.job, alias) for alias in mining.actors)
    catalog = load_model_catalog(project.model_catalog_path)
    registry = RuntimeModelCatalog(catalog, environment={})
    roles = [mining.judge, mining.selector, mining.critic, pipeline.authoring.rubric_author]
    if pipeline.traceml.enabled:
        assert pipeline.traceml.move_judge is not None
        roles.append(pipeline.traceml.move_judge)
    if mining.behaviors is not None:
        roles.extend((mining.behaviors.author, mining.behaviors.reviewer))
    for role in roles:
        if (
            registry.snapshot(role.model_alias)[0] != role.model
            or role.catalog_sha256 != proof.start.job.catalog_sha256
        ):
            raise ValueError("native prelude role changed its configured snapshot or route")
    for job in jobs:
        if job.maximum_physical_attempts < mining.attempts:
            raise ValueError("native group allowance cannot complete all mandatory actor attempts")
        preflight_continuation_job(project, job)
    return QualifiedCouncilPrelude(
        contract=spec,
        preparation=prepared,
        resume_binding=resume_binding,
        task_start=proof.task_start,
        jobs=jobs,
        actors=tuple(
            RoutedCandidateSnapshot(alias=job.actor_alias, model=job.study.actor_model)
            for job in jobs
        ),
        producer_revision=code_revision,
    )


def _resolved(project: ProjectStore, qualified: QualifiedCouncilPrelude) -> DecisionVarianceSpec:
    """Resolve only the native saved job while retaining every frozen pipeline setting."""
    return resolve_bootstrap(project, qualified.contract.pipeline)


def _job(qualified: QualifiedCouncilPrelude, alias: str | None) -> ContinuationJob:
    """Select an exact frozen native actor job without aliases inferred from display text."""
    matches = tuple(job for job in qualified.jobs if job.actor_alias == alias)
    if len(matches) != 1:
        raise ValueError("prelude checkpoint requires one exact configured actor tier")
    return matches[0]


def _rubric_proof(
    project: ProjectStore,
    qualified: QualifiedCouncilPrelude,
    reference: ArtifactInput,
) -> None:
    """Require an approved saved rubric or the native whole-task authoring owner."""
    _pinned(project, reference, "episode-rubric")
    configured = qualified.contract.pipeline.mining.rubric
    if configured is not None:
        if reference != configured:
            raise ValueError("prelude substituted the explicitly frozen mining rubric")
        return
    rubric = load_rubric(project, reference)
    manifest = project.artifacts.read(reference.artifact_id).manifest
    task = load_continuation_checkpoint(project.artifacts, qualified.jobs[0].study.checkpoint).task
    if manifest.code_revision != qualified.producer_revision or rubric.family_id != local_id(
        f"mining-{task.task_id}"
    ):
        raise ValueError("prelude rubric is not the native selected-task authoring result")


def prelude_aliases(project: ProjectStore, qualified: QualifiedCouncilPrelude) -> tuple[str, ...]:
    """Resolve real actor, role, tagger, world, reader, recall and embedder aliases directly."""
    spec = qualified.contract.pipeline
    aliases = set(spec.mining.actors)
    roles = [
        spec.mining.judge,
        spec.mining.selector,
        spec.mining.critic,
        spec.authoring.rubric_author,
    ]
    if spec.traceml.enabled:
        aliases.add(spec.traceml.decision_alias)
        assert spec.traceml.move_judge is not None
        roles.append(spec.traceml.move_judge)
    if spec.mining.behaviors is not None:
        aliases.add(spec.traceml.decision_alias)
        roles.extend((spec.mining.behaviors.author, spec.mining.behaviors.reviewer))
    aliases.update(role.model_alias for role in roles)
    runtime = GroundedRuntimeDefinition.model_validate(qualified.jobs[0].configuration)
    catalog = native_world_catalog(project, runtime)
    world = load_grounded_world_model_artifact(project.artifacts, runtime.world_model)
    aliases.add(world.model_alias)
    embedder = index_embedder_alias(
        project,
        catalog,
        world,
        configured_alias=(
            None if runtime.catalog_binding is None else runtime.catalog_binding.embedder_alias
        ),
    )
    if embedder is not None:
        aliases.add(embedder)
    if runtime.memory is not None:
        aliases.add(runtime.memory.recall.jev_alias)
    if runtime.rule_memory is not None:
        aliases.add(runtime.rule_memory.recall.jev_alias)
    if runtime.prediction_reading is not None:
        aliases.add(runtime.prediction_reading.reader_alias)
    return tuple(sorted(aliases))


def register_council_prelude(
    project: ProjectStore,
    prepared: ArtifactInput,
    resume_binding: ArtifactInput,
    spec: CouncilPreludeContract,
    consent: ArtifactInput,
    *,
    code_revision: str,
) -> ConditionalCouncilPreludeRegistration:
    """Register exact prelude consent under the same original physical ledger and grants."""
    qualified = qualify_council_prelude(
        project,
        prepared,
        resume_binding,
        spec,
        code_revision=code_revision,
    )
    _pinned(project, consent, CONSENT_TYPE)
    approved = CouncilConditionalConsent.model_validate_json(
        project.artifacts.read_bytes(consent.artifact_id, "consent.json")
    )
    from exp.optimize.evaluation.variance.bootstrap_history import (
        read_bootstrap_preparation_provenance,
    )

    binding = read_bootstrap_preparation_provenance(
        project,
        resume_binding,
        spec=spec.preparation.preparation_spec,
    ).binding
    manifest = project.artifacts.read(consent.artifact_id).manifest
    if (
        approved.operation != binding.original_operation
        or approved.ledger_sha256 != binding.ledger_sha256
        or approved.contract_sha256 != sha256_json(spec)
        or approved.preparation != prepared
        or approved.resume_binding != resume_binding
        or approved.open_actors != qualified.actors
        or approved.legacy_attempt_limit != binding.unpriced_attempts
        or approved.maximum_known_usd != binding.ceiling_usd
        or approved.producer_revision != code_revision
        or not datetime.now(UTC) < approved.deadline
        or not manifest.created_at < approved.deadline <= manifest.created_at + timedelta(hours=6)
    ):
        raise ValueError("conditional prelude consent differs from exact original native authority")
    inputs = (
        prepared,
        resume_binding,
        consent,
        qualified.task_start,
        spec.preparation.exposure,
        spec.preparation.family_simulator,
    )
    identity = stable_id(
        PRELUDE_TYPE,
        {
            "qualified": sha256_json(qualified),
            "consent": consent.model_dump(mode="json"),
        },
    )
    reference = _save(
        project, identity, PRELUDE_TYPE, "qualification.json", qualified, inputs, code_revision
    )
    policy = register_parent_policy(
        operation_spend_ceiling(project, binding.original_operation),
        ParentSpendPolicy(
            operation=binding.original_operation,
            legacy_attempt_limit=approved.legacy_attempt_limit,
            parent_attempt_limit=approved.parent_attempt_limit,
            maximum_known_usd=approved.maximum_known_usd,
            deadline=approved.deadline,
            aliases=prelude_aliases(project, qualified),
            authorization=consent,
            preparation=prepared,
            execution=reference,
            producer_revision=code_revision,
        ),
    )
    return ConditionalCouncilPreludeRegistration(qualification=reference, policy=policy)


def _save(
    project: ProjectStore,
    identity: str,
    kind: str,
    file: str,
    value: ContractModel,
    inputs: tuple[ArtifactInput, ...],
    revision: str,
) -> ArtifactInput:
    """Save one exact native proof idempotently without substituting an existing owner."""
    ordered = tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id))
    if project.artifacts.exists(identity):
        stored = project.artifacts.read(identity)
        if (
            stored.manifest.artifact_type != kind
            or stored.manifest.code_revision != revision
            or stored.manifest.inputs != ordered
            or type(value).model_validate_json(stored.payloads[file]) != value
        ):
            raise ValueError("saved native council proof differs from its frozen registration")
    else:
        project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=kind,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=revision,
                inputs=ordered,
            ),
            files={file: value},
        )
    return artifact_input(project.artifacts.read(identity).manifest)


def load_qualified_prelude(
    project: ProjectStore,
    reference: ArtifactInput,
) -> QualifiedCouncilPrelude:
    """Read and independently re-qualify an exact native prelude without writes or calls."""
    _pinned(project, reference, PRELUDE_TYPE)
    stored = project.artifacts.read(reference.artifact_id)
    qualified = QualifiedCouncilPrelude.model_validate_json(stored.payloads["qualification.json"])
    if stored.manifest.code_revision != qualified.producer_revision:
        raise ValueError("native prelude proof lost its actual executor producer")
    checked = qualify_council_prelude(
        project,
        qualified.preparation,
        qualified.resume_binding,
        qualified.contract,
        code_revision=qualified.producer_revision,
    )
    if checked != qualified:
        raise ValueError("saved native prelude proof changed its preparation or exact jobs")
    required = {
        qualified.preparation,
        qualified.resume_binding,
        qualified.task_start,
        qualified.contract.preparation.exposure,
        qualified.contract.preparation.family_simulator,
    }
    if not required.issubset(stored.manifest.inputs):
        raise ValueError("native prelude proof lost its exact input ancestry")
    return qualified

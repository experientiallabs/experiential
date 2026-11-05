"""Qualify native preparation evidence before registering conditional council authority.

Qualification reads immutable owners and configured snapshots without constructing clients.
It does not qualify downstream training records or dispatch any actor, world, or QA call.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field

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
    continuation_result_reference,
    preflight_continuation_job,
)
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
)
from exp.optimize.evaluation.fidelity.retrieval_binding import index_embedder_alias
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec
from exp.optimize.evaluation.variance.behavior_pairs import load_behavior_differences
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    BootstrapBinding,
    BootstrapRegistration,
)
from exp.optimize.evaluation.variance.bootstrap_history import (
    PreparationSourceType,
    read_bootstrap_preparation_provenance,
)
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapPreparationSpec,
)
from exp.optimize.evaluation.variance.council_preparation import (
    CouncilPreparationContract,
    _pinned,
    qualify_prepared_council,
)
from exp.optimize.evaluation.variance.council_preparation import (
    _native_start as _native_start,
)
from exp.optimize.evaluation.variance.episode_evidence import run_task_sha256, run_world_sha256
from exp.optimize.evaluation.variance.evidence import load_references
from exp.optimize.evaluation.variance.family_simulators import load_family_simulator
from exp.optimize.evaluation.variance.pipeline_family_rag import native_world_catalog
from exp.optimize.evaluation.variance.pipeline_spec import COUNCIL_ACTORS, MiningSettings
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.scoped_spending import (
    ParentSpendPolicy,
    ParentSpendRef,
    register_parent_policy,
)
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

CONSENT_TYPE = "conditional-council-consent"
QUALIFICATION_TYPE = "conditional-council-qualification"
_SOURCE_FILES = (
    "exp/optimize/evaluation/variance/actor_critic.py",
    "exp/optimize/evaluation/variance/bootstrap.py",
    "exp/optimize/evaluation/variance/bootstrap_capture.py",
    "exp/optimize/evaluation/variance/bootstrap_contracts.py",
    "exp/optimize/evaluation/variance/bootstrap_history.py",
    "exp/optimize/evaluation/variance/bootstrap_source.py",
    "exp/optimize/evaluation/variance/bootstrap_resume.py",
    "exp/optimize/evaluation/variance/bootstrap_protocols.py",
    "exp/optimize/evaluation/variance/bootstrap_selection.py",
    "exp/optimize/evaluation/variance/council_checkpoint.py",
    "exp/optimize/evaluation/variance/council_initial.py",
    "exp/optimize/evaluation/variance/council_inventory.py",
    "exp/optimize/evaluation/variance/council_parallel.py",
    "exp/optimize/evaluation/variance/council_qualification.py",
    "exp/optimize/evaluation/variance/council_world.py",
    "exp/optimize/evaluation/variance/council_preparation.py",
    "exp/optimize/evaluation/variance/council_prelude_hook.py",
    "exp/optimize/evaluation/variance/pipeline_council_tiers.py",
    "exp/optimize/evaluation/variance/pipeline_inputs.py",
    "exp/optimize/evaluation/variance/pipeline_stages.py",
    "exp/optimize/evaluation/variance/pipeline_spec.py",
    "exp/optimize/evaluation/variance/pipeline_behaviors.py",
    "exp/optimize/evaluation/variance/pipeline_errors.py",
    "exp/optimize/evaluation/variance/pipeline_family_rag.py",
    "exp/optimize/evaluation/variance/trace_search_spec.py",
    "exp/optimize/evaluation/fidelity/retrieval_binding.py",
    "exp/optimize/evaluation/executable/catalog_binding.py",
    "exp/optimize/evaluation/variance/judge_prompt.py",
    "exp/optimize/evaluation/variance/judged.py",
    "exp/optimize/evaluation/variance/role_prompts.py",
    "exp/optimize/evaluation/variance/roles.py",
    "exp/optimize/evaluation/continuations/job.py",
    "exp/optimize/evaluation/executable/grounded.py",
    "exp/optimize/evaluation/executable/loader.py",
    "exp/optimize/evaluation/executable/provider_grant.py",
    "exp/optimize/evaluation/executable/runtime_authority.py",
    "exp/runtime/models/scoped_spending.py",
    "exp/runtime/models/spend_ceiling.py",
    "exp/runtime/models/providers/admitted_transport.py",
)


def council_sources() -> dict[str, Sha256]:
    """Freeze actual domain, runtime authority, prompt and physical-admission owners."""
    root = Path(__file__).resolve().parents[4]
    return {name: sha256_bytes((root / name).read_bytes()) for name in _SOURCE_FILES}


class CouncilStudyContract(ContractModel):
    """Frozen native per-model study and the exact preparation evidence it consumes.

    Attributes:
        preparation_spec: Original preparation settings understood by the resume owner.
        bootstrap: Selected native task-start binding, never a hand-authored substitute.
        mining: Configured mining settings supplying the exact five actor aliases.
        studies: Separate full native actor-critic registration per configured actor.
        exposure: Exposure plan of the genuine preparation build.
        family_simulator: Saved shared family runtime, rules and retrieval binding.
        sources: Actual maintained domain source fingerprints, including prompts.
    """

    preparation_spec: BootstrapPreparationSpec
    preparation_producer: str = Field(min_length=1)
    preparation_source_sha256: Sha256
    preparation_provenance: ArtifactInput
    preparation_provenance_type: PreparationSourceType
    bootstrap: BootstrapBinding
    mining: MiningSettings
    studies: tuple[ActorCriticSpec, ...] = Field(min_length=5, max_length=5)
    exposure: ArtifactInput
    family_simulator: ArtifactInput
    sources: dict[str, Sha256]


class CouncilConditionalConsent(ContractModel):
    """Explicit host consent for one exact study under its original physical ledger.

    Attributes:
        protocol: Versioned conditional, native-only authority.
        operation: Exact original preparation operation.
        ledger_sha256: Native original physical-ledger identity.
        contract_sha256: Full frozen study, controls and source identity.
        preparation: Genuine completed native preparation, not a build report.
        resume_binding: Exact original resume lineage.
        open_actors: Host-approved open generative snapshots and configured routes.
        legacy_attempt_limit: Unchanged original preparation allowance.
        maximum_known_usd: Unchanged original dollar authority.
        parent_attempt_limit: Whole physical operation limit, including retained claims.
        deadline: Finite operational deadline for scoped authority.
        producer_revision: Actual reviewed producer allowed to register and execute.
    """

    protocol: Literal["conditional-native-council-v1"] = "conditional-native-council-v1"
    operation: str = Field(min_length=1)
    ledger_sha256: Sha256
    contract_sha256: Sha256
    preparation: ArtifactInput
    resume_binding: ArtifactInput
    open_actors: tuple[RoutedCandidateSnapshot, ...] = Field(min_length=5, max_length=5)
    legacy_attempt_limit: int = Field(ge=1, le=100_000, strict=True)
    maximum_known_usd: float = Field(ge=0)
    parent_attempt_limit: Literal[100000] = 100000
    deadline: AwareDatetime
    producer_revision: str = Field(min_length=1)


class QualifiedCouncil(ContractModel):
    """Read-only qualified native inputs, with no provider or ledger authority.

    Attributes:
        contract: Exact frozen domain specification.
        preparation: Completed native preparation proof.
        resume_binding: Original physical operation ancestry.
        task_start: Native saved task-start registration carrying its actual QA.
        actors: Exact configured per-model snapshots and connection digests.
        producer_revision: Actual verified source producer.
    """

    contract: CouncilStudyContract
    preparation: ArtifactInput
    resume_binding: ArtifactInput
    task_start: ArtifactInput
    actors: tuple[RoutedCandidateSnapshot, ...]
    producer_revision: str


class ConditionalCouncilRegistration(ContractModel):
    """Immutable domain qualification and its original-ledger native parent policy.

    Attributes:
        qualification: Saved exact domain qualification, including all native jobs.
        policy: Native additive parent policy in the unchanged original ledger.
    """

    qualification: ArtifactInput
    policy: ParentSpendRef


def qualify_council(
    project: ProjectStore,
    prepared: ArtifactInput,
    resume_binding: ArtifactInput,
    spec: CouncilStudyContract,
    *,
    code_revision: str,
) -> QualifiedCouncil:
    """Verify real completed preparation, exact world scope and five configured native studies.

    This reader never creates artifacts, installs a ceiling, executes QA or resolves credentials.
    Missing preparation, failed execution or a refused target cannot grant parent authority.
    """
    proof = qualify_prepared_council(
        project,
        prepared,
        resume_binding,
        CouncilPreparationContract(
            preparation_spec=spec.preparation_spec,
            preparation_producer=spec.preparation_producer,
            preparation_source_sha256=spec.preparation_source_sha256,
            preparation_provenance=spec.preparation_provenance,
            preparation_provenance_type=spec.preparation_provenance_type,
            bootstrap=spec.bootstrap,
            exposure=spec.exposure,
            family_simulator=spec.family_simulator,
        ),
        code_revision=code_revision,
    )
    start, task_start = proof.start, proof.task_start
    if spec.sources != council_sources():
        raise ValueError("council producer, source or prompt protocol changed")
    actors = validate_council_studies(project, spec, start, code_revision=code_revision)
    return QualifiedCouncil(
        contract=spec,
        preparation=prepared,
        resume_binding=resume_binding,
        task_start=task_start,
        actors=actors,
        producer_revision=code_revision,
    )


def validate_council_studies(
    project: ProjectStore,
    spec: CouncilStudyContract,
    start: BootstrapRegistration,
    *,
    code_revision: str,
) -> tuple[RoutedCandidateSnapshot, ...]:
    """Bind native study controls and aliases directly to configured snapshots and routes."""
    family = load_family_simulator(project.artifacts, spec.family_simulator)
    runtime = GroundedRuntimeDefinition.model_validate(start.job.configuration)
    for study in spec.studies:
        search = study.retrieval
        if search is not None and (
            search.index != family.index
            or search.scope != family.corpus
            or search.task_id != start.task_id
            or search.maximum_top_k != spec.mining.retrieval_top_k
            or search.catalog_binding != runtime.catalog_binding
        ):
            raise ValueError("native study retrieval differs from the exact family RAG scope")
    mining = spec.mining
    aliases = mining.actors
    if (
        len(aliases) != 5
        or len(set(aliases)) != 5
        or (mining.attempts, mining.selector_draws, mining.critic_draws, mining.continuations)
        != (16, 8, 8, 8)
        or not 1 <= mining.points <= 4
        or mining.bootstrap != spec.bootstrap
        or mining.family_simulator != spec.family_simulator
        or mining.schedule is not None
    ):
        raise ValueError("council requires five separate configured G16/J8/C8/K<=4/M8 studies")
    if tuple(row.initial.actor_alias for row in spec.studies) != aliases:
        raise ValueError("native studies do not preserve the five configured actor tiers")
    catalog = load_model_catalog(project.model_catalog_path)
    registry = RuntimeModelCatalog(catalog, environment={})
    actors: list[RoutedCandidateSnapshot] = []
    for alias, study in zip(aliases, spec.studies, strict=True):
        job = study.initial
        snapshot, capabilities = registry.snapshot(alias)
        if capabilities.supports_completions is False or capabilities.supports_tools is not True:
            raise ValueError("council actor must be an explicitly configured generative tool model")
        if snapshot.model_id in {row.model.model_id for row in actors}:
            raise ValueError("five configured actor tiers require distinct model IDs")
        actors.append(RoutedCandidateSnapshot(alias=alias, model=snapshot))
        expected = start.job.model_copy(
            update={
                "actor_alias": alias,
                "maximum_physical_attempts": mining.group_physical_attempts
                or start.job.maximum_physical_attempts,
                "study": start.job.study.model_copy(
                    update={"actor_model": snapshot, "repeats": 16}
                ),
            }
        )
        if job != expected or job.maximum_model_calls != 20:
            raise ValueError("native actor job changed its registered start, H20 or runtime")
        if (
            study.actor_tier != alias
            or study.judges != (mining.selector,) * 8
            or study.critics != (mining.critic,) * 8
            or study.points != mining.points
            or study.continuations != 8
            or study.continuation_physical_attempts != mining.continuation_physical_attempts
            or study.seed != mining.seed
            or study.selector_profile != mining.selector_profile
            or study.critic_profile != mining.critic_profile
            or study.episode_judge.settings != mining.judge
            or study.selector_prompts != spec.studies[0].selector_prompts
            or study.critic_prompts != spec.studies[0].critic_prompts
            or study.episode_judge != spec.studies[0].episode_judge
        ):
            raise ValueError("native study reduced controls or changed configured roles/protocol")
        if mining.retrieval and study.retrieval is None:
            raise ValueError("native council cannot drop required fit-only retrieval")
        checkpoint = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
        definition = study.episode_judge.definition
        if definition.task_sha256 != run_task_sha256(
            checkpoint
        ) or definition.world_sha256 != run_world_sha256(job.configuration):
            raise ValueError("native episode judge binds another task or frozen world")
        for role in (*study.judges, *study.critics, study.episode_judge.settings):
            if (
                registry.snapshot(role.model_alias)[0] != role.model
                or role.catalog_sha256 != job.catalog_sha256
            ):
                raise ValueError("native selector, critic or episode-judge route snapshot changed")
        preflight_continuation_job(project, job)
        load_references(project.artifacts, study.references)
        shared = tuple(ref for ref in study.references if ref.kind not in MODEL_REFERENCE_KINDS)
        first = tuple(
            ref for ref in spec.studies[0].references if ref.kind not in MODEL_REFERENCE_KINDS
        )
        if shared != first:
            raise ValueError("native studies changed their shared non-model evidence")
        council_reference_owners(project, study, mining, code_revision=code_revision)
    return tuple(actors)


MODEL_REFERENCE_KINDS = frozenset({"traceml_labels", "error_points", "behavior_differences"})


def council_reference_owners(
    project: ProjectStore,
    study: ActorCriticSpec,
    mining: MiningSettings,
    *,
    code_revision: str,
    behavior_bank_sha256: Sha256 | None = None,
) -> tuple[ArtifactInput, ...]:
    """Verify each model's generated references against its actual native group owner."""
    owned = tuple(ref for ref in study.references if ref.kind in MODEL_REFERENCE_KINDS)
    if not owned:
        return ()
    group = continuation_result_reference(project, study.initial, code_revision=code_revision)
    owners = {group}
    for reference in owned:
        owners.add(reference.artifact)
        manifest = project.artifacts.read(reference.artifact.artifact_id).manifest
        if reference.kind == "traceml_labels" and group not in manifest.inputs:
            raise ValueError("full council labels do not belong to the native model group")
        if reference.kind == "error_points" and not any(
            group in project.artifacts.read(parent.artifact_id).manifest.inputs
            for parent in manifest.inputs
        ):
            raise ValueError("full council errors belong to another native model group")
        if reference.kind == "behavior_differences":
            raw = json.loads(
                project.artifacts.read_bytes(reference.artifact.artifact_id, reference.file)
            )
            source = ArtifactInput.model_validate(raw["source"])
            report = load_behavior_differences(project, source)
            if (
                manifest.inputs != (source,)
                or report.study != group
                or report.tier != study.initial.actor_alias
                or report.judge != study.episode_judge.identity
                or report.spec != mining.behaviors
                or (
                    behavior_bank_sha256 is not None
                    and sha256_json(report.bank) != behavior_bank_sha256
                )
            ):
                raise ValueError("full council behavior belongs to another group or judge")
            owners.add(source)
    return tuple(sorted(owners, key=lambda ref: ref.artifact_id))


def register_conditional_council(
    project: ProjectStore,
    prepared: ArtifactInput,
    resume_binding: ArtifactInput,
    spec: CouncilStudyContract,
    consent: ArtifactInput,
    *,
    code_revision: str,
) -> ConditionalCouncilRegistration:
    """Register exact host consent only after real native domain qualification succeeds.

    The kernel preserves all original grants and retained claims in the same physical ledger.
    Per-job runtime grants and phase envelopes remain mandatory before any later dispatch.
    """
    qualified = qualify_council(
        project, prepared, resume_binding, spec, code_revision=code_revision
    )
    _pinned(project, consent, CONSENT_TYPE)
    approved = CouncilConditionalConsent.model_validate_json(
        project.artifacts.read_bytes(consent.artifact_id, "consent.json")
    )
    binding = read_bootstrap_preparation_provenance(
        project, resume_binding, spec=spec.preparation_spec
    ).binding
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
        or datetime.now(UTC) >= approved.deadline
        or approved.deadline > binding.stop
        or not (
            project.artifacts.read(consent.artifact_id).manifest.created_at
            < approved.deadline
            <= project.artifacts.read(consent.artifact_id).manifest.created_at + timedelta(hours=6)
        )
    ):
        raise ValueError("conditional council consent differs from qualified original authority")
    inputs = tuple(
        sorted(
            {
                prepared,
                resume_binding,
                consent,
                qualified.task_start,
                spec.exposure,
                spec.family_simulator,
            },
            key=lambda ref: ref.artifact_id,
        )
    )
    identity = stable_id(
        QUALIFICATION_TYPE,
        {"qualified": sha256_json(qualified), "consent": consent.model_dump(mode="json")},
    )
    if project.artifacts.exists(identity):
        stored = project.artifacts.read(identity)
        if (
            stored.manifest.artifact_type != QUALIFICATION_TYPE
            or stored.manifest.code_revision != code_revision
            or stored.manifest.inputs != inputs
            or QualifiedCouncil.model_validate_json(stored.payloads["qualification.json"])
            != qualified
        ):
            raise ValueError("saved council qualification differs from the exact native proof")
    else:
        project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=QUALIFICATION_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=inputs,
            ),
            files={"qualification.json": qualified},
        )
    reference = artifact_input(project.artifacts.read(identity).manifest)
    policy = register_parent_policy(
        operation_spend_ceiling(project, binding.original_operation),
        ParentSpendPolicy(
            operation=binding.original_operation,
            legacy_attempt_limit=approved.legacy_attempt_limit,
            parent_attempt_limit=approved.parent_attempt_limit,
            maximum_known_usd=approved.maximum_known_usd,
            deadline=approved.deadline,
            aliases=council_aliases(project, qualified),
            authorization=consent,
            preparation=prepared,
            execution=reference,
            producer_revision=code_revision,
        ),
    )
    return ConditionalCouncilRegistration(qualification=reference, policy=policy)


def council_aliases(project: ProjectStore, qualified: QualifiedCouncil) -> tuple[str, ...]:
    """Resolve exact native actor, role, world, recall, reader and embedding routes.

    Cost component display names never supply authority or model identity.
    """
    aliases: set[str] = set()
    for study in qualified.contract.studies:
        aliases.add(study.initial.actor_alias)
        aliases.update(
            role.model_alias
            for role in (
                *study.judges,
                *study.critics,
                study.episode_judge.settings,
            )
        )
        runtime = GroundedRuntimeDefinition.model_validate(study.initial.configuration)
        world = load_grounded_world_model_artifact(project.artifacts, runtime.world_model)
        aliases.add(world.model_alias)
        embedder = index_embedder_alias(
            project,
            native_world_catalog(project, runtime),
            world,
            configured_alias=None
            if runtime.catalog_binding is None
            else runtime.catalog_binding.embedder_alias,
        )
        if embedder is not None:
            aliases.add(embedder)
        if runtime.memory is not None:
            aliases.add(runtime.memory.recall.jev_alias)
        if runtime.rule_memory is not None:
            aliases.add(runtime.rule_memory.recall.jev_alias)
        if runtime.prediction_reading is not None:
            aliases.add(runtime.prediction_reading.reader_alias)
        if study.initial.study.cascade is not None:
            aliases.add(study.initial.study.cascade.decider.alias)
    return tuple(sorted(aliases))


def configured_council_actors(
    project: ProjectStore,
    *,
    aliases: tuple[str, ...],
    model_ids: tuple[str, ...] = COUNCIL_ACTORS,
) -> tuple[RoutedCandidateSnapshot, ...]:
    """Resolve declared open council model IDs to unique actual configured native routes.

    The model set comes from the caller or native council defaults. Alias spelling, catalog
    iteration order and cost display labels never choose a model or a fallback.
    """
    if (
        len(model_ids) != 5
        or len(set(model_ids)) != 5
        or len(aliases) != 5
        or len(set(aliases)) != 5
    ):
        raise ValueError("declare exactly five distinct open council models and aliases")
    catalog = load_model_catalog(project.model_catalog_path)
    registry = RuntimeModelCatalog(catalog, environment={})
    selected: list[RoutedCandidateSnapshot] = []
    for model_id in model_ids:
        matched = tuple(
            alias
            for alias in aliases
            if alias in catalog.models and catalog.models[alias].model == model_id
        )
        if len(matched) != 1:
            raise ValueError(f"council model {model_id!r} needs one exact configured alias")
        alias = matched[0]
        snapshot, capabilities = registry.snapshot(alias)
        if capabilities.supports_completions is not True or capabilities.supports_tools is not True:
            raise ValueError(f"council model {model_id!r} lacks explicit generative tool support")
        selected.append(RoutedCandidateSnapshot(alias=alias, model=snapshot))
    return tuple(selected)

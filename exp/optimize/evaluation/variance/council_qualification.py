"""Qualify native preparation evidence before registering conditional council authority.

Qualification reads immutable owners and configured snapshots without constructing clients.
It does not qualify downstream training records or dispatch any actor, world, or QA call.
"""

from __future__ import annotations

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
from exp.optimize.evaluation.continuations.job import preflight_continuation_job
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    preflight_grounded_runtime,
)
from exp.optimize.evaluation.fidelity.retrieval_binding import index_embedder_alias
from exp.optimize.evaluation.rules.agent_loop import TRANSCRIPT_TYPE, AgentTranscript
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec
from exp.optimize.evaluation.variance.bootstrap import frozen_report, registration_entry
from exp.optimize.evaluation.variance.bootstrap_agents import RevealAnswer
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    REGISTRATION_TYPE,
    BootstrapBinding,
    BootstrapRegistration,
    BootstrapSettings,
    ReviewAnswer,
    StartDraft,
)
from exp.optimize.evaluation.variance.bootstrap_plan import (
    PREPARATION_TYPE,
    BootstrapPreparationReport,
    BootstrapPreparationSpec,
)
from exp.optimize.evaluation.variance.bootstrap_resume import (
    bootstrap_execution_spec,
    load_bootstrap_resume,
)
from exp.optimize.evaluation.variance.bootstrap_selection import select_bootstrap_families
from exp.optimize.evaluation.variance.bootstrap_stage import bootstrapped_registration
from exp.optimize.evaluation.variance.build_records import (
    REPORT_TYPE,
    VarianceBuildReport,
    load_exposure_plan,
)
from exp.optimize.evaluation.variance.episode_evidence import run_task_sha256, run_world_sha256
from exp.optimize.evaluation.variance.evidence import load_references
from exp.optimize.evaluation.variance.family_simulators import load_family_simulator
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
    "actor_critic.py",
    "bootstrap.py",
    "bootstrap_capture.py",
    "bootstrap_contracts.py",
    "council_inventory.py",
    "council_qualification.py",
    "judge_prompt.py",
    "judged.py",
    "role_prompts.py",
    "roles.py",
)


def council_sources() -> dict[str, Sha256]:
    """Freeze maintained domain owners and protocols without importing runtime factories."""
    root = Path(__file__).parent
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


def _pinned(project: ProjectStore, reference: ArtifactInput, kind: str) -> None:
    """Require exact immutable bytes and the actual native owner's artifact type."""
    if not project.artifacts.exists(reference.artifact_id):
        raise ValueError(f"completed native {kind} is missing: {reference.artifact_id}")
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != kind:
        raise ValueError(f"{reference.artifact_id} is not the exact native {kind}")


def _native_start(
    project: ProjectStore,
    binding: BootstrapBinding,
    report: BootstrapPreparationReport,
    settings: BootstrapSettings,
) -> tuple[BootstrapRegistration, ArtifactInput]:
    """Independently verify the selected committed start and its provenance-required QA."""
    if binding not in report.bindings or binding.batch not in report.batches:
        raise ValueError("selected task has no binding in the completed preparation")
    entry = registration_entry(project, binding)
    assert entry.outcome is not None
    _pinned(project, entry.outcome, REGISTRATION_TYPE)
    start = bootstrapped_registration(project, binding)
    validation = start.validation
    if (
        start.task_id != binding.task_id
        or start.checkpoint != start.job.study.checkpoint
        or validation.provenance != start.provenance
        or not validation.passed
        or not start.rounds
        or start.rounds[-1].verdict != "accept"
    ):
        raise ValueError("native task-start acceptance and provenance QA are incomplete")
    draws = [start.rounds[-1].author, *start.rounds[-1].reviews]
    if tuple(draw.variant for draw in start.rounds[-1].reviews) != settings.review_variants:
        raise ValueError("native task-start has no complete mandatory reviewer receipts")
    draws.extend(validation.checks)
    manifest = project.artifacts.read(entry.outcome.artifact_id).manifest
    if manifest.code_revision != report.plan.code_revision:
        raise ValueError("native task-start producer differs from completed preparation")
    if StartDraft.model_validate(start.rounds[-1].author.answer) != start.draft:
        raise ValueError("native final author receipt describes another task-start draft")
    if any(
        ReviewAnswer.model_validate(draw.answer).verdict != "accept"
        for draw in start.rounds[-1].reviews
    ):
        raise ValueError("native mandatory reviewer did not accept the task start")
    if any(not RevealAnswer.model_validate(draw.answer).consistent for draw in validation.checks):
        raise ValueError("native first-reveal reviewer found a contradiction")
    for draw in draws:
        if draw.transcript is None or draw.failure is not None:
            raise ValueError("native task-start author/reviewer/reveal receipt is unavailable")
        _pinned(project, draw.transcript, TRANSCRIPT_TYPE)
        transcript = AgentTranscript.model_validate_json(
            project.artifacts.read_bytes(draw.transcript.artifact_id, "transcript.json")
        )
        if (
            transcript.answer != draw.answer
            or transcript.stop != "answered"
            or transcript.failure is not None
            or transcript.role != draw.role
            or transcript.variant != draw.variant
        ):
            raise ValueError("native task-start role receipt differs from its completed transcript")
        stored = project.artifacts.read(draw.transcript.artifact_id).manifest
        if artifact_input(stored) != draw.transcript or draw.transcript not in manifest.inputs:
            raise ValueError("native task-start lost its exact role receipt ancestry")
    methods = {
        "observed": "exact_restore",
        "reconstructed": "prefix_replay",
        "authored": "coherence_and_first_reveal",
    }
    if validation.method != methods[start.provenance]:
        raise ValueError("native task-start used another provenance QA method")
    if start.provenance == "reconstructed" and (
        len(validation.replay) != start.draft.prefix_steps
        or any(not row.agrees for row in validation.replay)
    ):
        raise ValueError("native prefix replay QA is incomplete")
    if start.provenance == "authored" and (
        tuple(draw.variant for draw in validation.checks) != settings.reveal_variants
        or len(validation.reveals) != len(start.draft.reveal_probes)
    ):
        raise ValueError("native authored start lacks complete first-reveal QA receipts")
    return start, entry.outcome


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
    _pinned(project, prepared, PREPARATION_TYPE)
    report = BootstrapPreparationReport.model_validate_json(
        project.artifacts.read_bytes(prepared.artifact_id, "report.json")
    )
    binding = load_bootstrap_resume(
        project, resume_binding, spec=spec.preparation_spec, code_revision=code_revision
    )
    if (
        prepared.artifact_id != binding.original_operation + "-result"
        or report.identity != binding.original_operation
        or report.plan.identity != report.identity
        or report.plan.resume_binding != resume_binding
        or report.plan.code_revision != code_revision
        or report.plan.build_identity != binding.build_identity
        or report.failed_tasks
        or spec.bootstrap.task_id in report.refused_tasks
        or not report.bindings
    ):
        raise ValueError("native preparation is not complete under the exact original resume")
    manifest = project.artifacts.read(prepared.artifact_id).manifest
    expected = {report.build_report, *report.batches, *report.simulators, resume_binding}
    if manifest.code_revision != code_revision or set(manifest.inputs) != expected:
        raise ValueError("native completed preparation producer or ancestry differs")
    _pinned(project, report.build_report, REPORT_TYPE)
    build = VarianceBuildReport.model_validate_json(
        project.artifacts.read_bytes(report.build_report.artifact_id, "report.json")
    )
    if build.identity != binding.build_identity or build.build != binding.source_build:
        raise ValueError("native preparation changed its original source build")
    execution = bootstrap_execution_spec(project, binding, spec.preparation_spec)
    selected = select_bootstrap_families(
        project,
        build,
        execution.bootstrap,
        execution.limits,
        task_ids=execution.task_ids,
        simulators=execution.simulators,
    )
    start, task_start = _native_start(project, spec.bootstrap, report, execution.bootstrap)
    families = [row for row in selected if row.simulator.family_id == start.family_id]
    if len(families) != 1 or spec.family_simulator not in report.simulators:
        raise ValueError("selected start has no exact prepared family simulator")
    family = load_family_simulator(project.artifacts, spec.family_simulator)
    unchanged = {"rule_book", "rule_memory", "rules_report", "runtime", "tasks"}
    if family.model_dump(exclude=unchanged) != families[0].simulator.model_dump(exclude=unchanged):
        raise ValueError("prepared family compatibility or scoped source inventory differs")
    if family.family_id != start.family_id or family.runtime is None:
        raise ValueError("prepared family runtime is unavailable or belongs to another family")
    if (
        family.world_model != families[0].scope.world_model
        or family.index != families[0].scope.index
        or family.corpus != families[0].scope.retrieval_scope
        or family.rule_book != start.rulebook
        or family.rule_memory != start.rule_memory
        or family.rules_report != start.rules_report
        or start.rules_report is None
        or spec.exposure != build.exposure
    ):
        raise ValueError("family world, fit-only RAG scope, exposure or frozen rules differ")
    exposure = load_exposure_plan(project, spec.exposure)
    exposure.require(family.members, "world_learning")
    exposure.require(family.rules_traces, "world_learning")
    checkpoint = load_continuation_checkpoint(project.artifacts, start.checkpoint)
    exposure.require_lineage(checkpoint.task.lineage_group_id, "mining_probes")
    runtime = GroundedRuntimeDefinition.model_validate(start.job.configuration)
    excluded = {"task", "manifest", "initial_state", "probe_paths", "facts"}
    if runtime.model_dump(exclude=excluded) != family.runtime.model_dump(exclude=excluded):
        raise ValueError("task runtime changed its frozen family definition")
    preflight_grounded_runtime(project, runtime)
    if frozen_report(project, start.rules_report).rulebook != start.rulebook:
        raise ValueError("native rules report binds another rule book")
    for study in spec.studies:
        search = study.retrieval
        if search is not None and (
            search.index != family.index
            or search.scope != family.corpus
            or search.task_id != start.task_id
            or search.maximum_top_k != spec.mining.retrieval_top_k
        ):
            raise ValueError("native study retrieval differs from the exact family RAG scope")
    if spec.sources != council_sources():
        raise ValueError("council producer, source or prompt protocol changed")
    actors = validate_council_studies(project, spec, start)
    return QualifiedCouncil(
        contract=spec,
        preparation=prepared,
        resume_binding=resume_binding,
        task_start=task_start,
        actors=actors,
        producer_revision=code_revision,
    )


def validate_council_studies(
    project: ProjectStore, spec: CouncilStudyContract, start: BootstrapRegistration
) -> tuple[RoutedCandidateSnapshot, ...]:
    """Bind native study controls and aliases directly to configured snapshots and routes."""
    mining = spec.mining
    aliases = mining.actors
    if (
        len(aliases) != 5
        or len(set(aliases)) != 5
        or (mining.attempts, mining.selector_draws, mining.critic_draws, mining.continuations)
        != (16, 8, 8, 8)
        or not 1 <= mining.points <= 4
        or mining.bootstrap != spec.bootstrap
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
            or study.references != spec.studies[0].references
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
    return tuple(actors)


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
    binding = load_bootstrap_resume(
        project, resume_binding, spec=spec.preparation_spec, code_revision=code_revision
    )
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
    catalog = load_model_catalog(project.model_catalog_path)
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
        embedder = index_embedder_alias(project, catalog, world)
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

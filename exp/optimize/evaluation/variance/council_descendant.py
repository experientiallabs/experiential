"""Bind actual post-prelude council studies to the same original parent policy."""

from __future__ import annotations

from exp.common.core.artifacts import ArtifactInput, ContractModel, sha256_json, stable_id
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.job import continuation_result_reference
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec
from exp.optimize.evaluation.variance.bootstrap_stage import bootstrapped_registration
from exp.optimize.evaluation.variance.council_prelude import (
    QualifiedCouncilPrelude,
    _rubric_proof,
    _save,
    load_qualified_prelude,
    prelude_sources,
)
from exp.optimize.evaluation.variance.council_preparation import _pinned
from exp.optimize.evaluation.variance.council_qualification import (
    CouncilStudyContract,
    QualifiedCouncil,
    council_reference_owners,
    council_sources,
    qualify_council,
    validate_council_studies,
)
from exp.optimize.evaluation.variance.pipeline_stages import mining_profile

DESCENDANT_TYPE = "conditional-council-study-descendant"


class CouncilStudyDescendant(ContractModel):
    """Real full studies produced after authorized preludes, under the same parent policy.

    Attributes:
        prelude: Exact original native prelude qualification and parent execution reference.
        qualified: Genuine full study qualification including produced rubric and references.
    """

    prelude: ArtifactInput
    qualified: QualifiedCouncil


def _validate_descendant(
    project: ProjectStore,
    prelude: ArtifactInput,
    studies: tuple[ActorCriticSpec, ...],
    *,
    code_revision: str,
) -> tuple[CouncilStudyDescendant, tuple[ArtifactInput, ...]]:
    """Verify genuine full studies, generated references and immutable prelude ancestry."""
    frozen = load_qualified_prelude(project, prelude)
    if frozen.producer_revision != code_revision or frozen.contract.sources != prelude_sources():
        raise ValueError("full council descendant changed its native prelude producer or sources")
    core = frozen.contract.preparation
    pipeline = frozen.contract.pipeline
    mining = pipeline.mining
    spec = CouncilStudyContract(
        preparation_spec=core.preparation_spec,
        bootstrap=core.bootstrap,
        preparation_producer=core.preparation_producer,
        preparation_source_sha256=core.preparation_source_sha256,
        preparation_provenance=core.preparation_provenance,
        preparation_provenance_type=core.preparation_provenance_type,
        exposure=core.exposure,
        family_simulator=core.family_simulator,
        mining=mining,
        studies=studies,
        sources=council_sources(),
    )
    qualified = qualify_council(
        project,
        frozen.preparation,
        frozen.resume_binding,
        spec,
        code_revision=code_revision,
    )
    return _validated_studies(project, prelude, frozen, qualified)


def _validated_studies(
    project: ProjectStore,
    prelude: ArtifactInput,
    frozen: QualifiedCouncilPrelude,
    qualified: QualifiedCouncil,
) -> tuple[CouncilStudyDescendant, tuple[ArtifactInput, ...]]:
    """Verify native generated studies and owners within an exact immutable prelude contract."""
    code_revision = qualified.producer_revision
    pipeline = frozen.contract.pipeline
    mining = pipeline.mining
    studies = qualified.contract.studies
    _rubric_proof(project, frozen, studies[0].episode_judge.definition.rubric)
    if qualified.actors != frozen.actors or tuple(row.initial for row in studies) != frozen.jobs:
        raise ValueError("full council descendant changed a native actor job or configured model")
    owners: set[ArtifactInput] = {
        prelude,
        qualified.task_start,
        frozen.preparation,
        frozen.resume_binding,
    }
    reference_kinds = (
        *(("behavior_differences",) if mining.behaviors is not None else ()),
        *(("traceml_labels",) if pipeline.traceml.enabled else ()),
        *(
            ("error_points",)
            if pipeline.traceml.enabled and pipeline.traceml.error_bank is not None
            else ()
        ),
    )
    for study in studies:
        group = continuation_result_reference(project, study.initial, code_revision=code_revision)
        owners.add(group)
        rubric = study.episode_judge.definition.rubric
        owners.add(rubric)
        if study.episode_judge.definition.profile != mining_profile(pipeline) or (
            mining.rubric is not None and rubric != mining.rubric
        ):
            raise ValueError("full council descendant changed its frozen native judge or rubric")
        expected = ActorCriticSpec.error_mining(
            study.initial,
            episode_judge=study.episode_judge,
            selector=mining.selector,
            critic=mining.critic,
            concurrency=mining.concurrency,
            attempts=mining.attempts,
            selector_draws=mining.selector_draws,
            critic_draws=mining.critic_draws,
            points=mining.points,
            continuations=mining.continuations,
            seed=mining.seed,
            continuation_physical_attempts=mining.continuation_physical_attempts,
            selector_profile=mining.selector_profile,
            critic_profile=mining.critic_profile,
            references=study.references,
            actor_tier=study.initial.actor_alias,
        ).model_copy(update={"retrieval": study.retrieval})
        if study != expected:
            raise ValueError("full council prompts or generation controls changed")
        if tuple(reference.kind for reference in study.references) != reference_kinds:
            raise ValueError("full council descendant changed mandatory native reference controls")
        owners.update(
            council_reference_owners(
                project,
                study,
                mining,
                code_revision=code_revision,
                behavior_bank_sha256=frozen.contract.banks.get("behavior"),
            )
        )
    return CouncilStudyDescendant(prelude=prelude, qualified=qualified), tuple(owners)


def register_full_council_from_prelude(
    project: ProjectStore,
    prelude: ArtifactInput,
    studies: tuple[ActorCriticSpec, ...],
    *,
    code_revision: str,
) -> ArtifactInput:
    """Bind actual post-prelude studies without replacing the original parent policy or grants."""
    descendant, owners = _validate_descendant(
        project,
        prelude,
        studies,
        code_revision=code_revision,
    )
    identity = stable_id(DESCENDANT_TYPE, {"descendant": sha256_json(descendant)})
    return _save(
        project, identity, DESCENDANT_TYPE, "descendant.json", descendant, owners, code_revision
    )


def _read_descendant(project: ProjectStore, reference: ArtifactInput) -> CouncilStudyDescendant:
    """Read only the exact registered descendant identity and its original producer."""
    _pinned(project, reference, DESCENDANT_TYPE)
    stored = project.artifacts.read(reference.artifact_id)
    saved = CouncilStudyDescendant.model_validate_json(stored.payloads["descendant.json"])
    if (
        stored.manifest.code_revision != saved.qualified.producer_revision
        or reference.artifact_id != stable_id(DESCENDANT_TYPE, {"descendant": sha256_json(saved)})
    ):
        raise ValueError("full council descendant lost its exact identity or producer")
    return saved


def load_parent_bound_council_descendant(
    project: ProjectStore, reference: ArtifactInput, *, parent_execution: ArtifactInput
) -> CouncilStudyDescendant:
    """Verify a descendant against its already registered immutable original parent.

    This reader returns no spending authority. Runtime callers separately require the exact
    live original-ledger envelope and lifecycle. Historical preparation was qualified when
    the parent policy was registered; all current jobs, catalogs, source and generated owners
    are rechecked here without replaying that historical qualification.
    """
    # The initial-world owner imports council_world for the shared live envelope boundary.
    from exp.optimize.evaluation.variance.council_prelude_world import _qualified

    saved = _read_descendant(project, reference)
    if saved.prelude != parent_execution:
        raise ValueError("full council descendant belongs to another original parent")
    revision = saved.qualified.producer_revision
    frozen = _qualified(project, parent_execution, revision)
    core = frozen.contract.preparation
    expected = CouncilStudyContract(
        preparation_spec=core.preparation_spec,
        bootstrap=core.bootstrap,
        preparation_producer=core.preparation_producer,
        preparation_source_sha256=core.preparation_source_sha256,
        preparation_provenance=core.preparation_provenance,
        preparation_provenance_type=core.preparation_provenance_type,
        exposure=core.exposure,
        family_simulator=core.family_simulator,
        mining=frozen.contract.pipeline.mining,
        studies=saved.qualified.contract.studies,
        sources=council_sources(),
    )
    actual = saved.qualified
    if (
        actual.contract != expected
        or actual.preparation != frozen.preparation
        or actual.resume_binding != frozen.resume_binding
        or actual.task_start != frozen.task_start
        or actual.actors != frozen.actors
    ):
        raise ValueError("full council descendant changed its original parent contract")
    start = bootstrapped_registration(project, core.bootstrap)
    actors = validate_council_studies(project, expected, start, code_revision=revision)
    if actors != actual.actors:
        raise ValueError("full council descendant changed its current actor catalog")
    checked, owners = _validated_studies(project, parent_execution, frozen, actual)
    stored = project.artifacts.read(reference.artifact_id)
    if checked != saved or stored.manifest.inputs != tuple(
        sorted(set(owners), key=lambda ref: ref.artifact_id)
    ):
        raise ValueError("full council descendant changed its genuine native proof or ancestry")
    return saved


def load_council_descendant(
    project: ProjectStore,
    reference: ArtifactInput,
) -> CouncilStudyDescendant:
    """Read and independently verify a full study descendant without writes or dispatch."""
    saved = _read_descendant(project, reference)
    stored = project.artifacts.read(reference.artifact_id)
    revision = saved.qualified.producer_revision
    checked, owners = _validate_descendant(
        project,
        saved.prelude,
        saved.qualified.contract.studies,
        code_revision=revision,
    )
    if checked != saved or stored.manifest.inputs != tuple(
        sorted(set(owners), key=lambda ref: ref.artifact_id)
    ):
        raise ValueError("full council descendant changed its genuine native proof or ancestry")
    return saved

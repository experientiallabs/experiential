"""Bind actual post-prelude council studies to the same original parent policy."""

from __future__ import annotations

import json

from exp.common.core.artifacts import ArtifactInput, ContractModel, sha256_json, stable_id
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.job import continuation_result_reference
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec
from exp.optimize.evaluation.variance.behavior_pairs import load_behavior_differences
from exp.optimize.evaluation.variance.council_prelude import (
    _save,
    load_qualified_prelude,
    prelude_sources,
)
from exp.optimize.evaluation.variance.council_preparation import _pinned
from exp.optimize.evaluation.variance.council_qualification import (
    CouncilStudyContract,
    QualifiedCouncil,
    council_sources,
    qualify_council,
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
        for reference in study.references:
            owners.add(reference.artifact)
            manifest = project.artifacts.read(reference.artifact.artifact_id).manifest
            if reference.kind == "traceml_labels" and group not in manifest.inputs:
                raise ValueError("full council labels do not belong to the native prelude group")
            if reference.kind == "error_points":
                if not any(
                    group in project.artifacts.read(parent.artifact_id).manifest.inputs
                    for parent in manifest.inputs
                ):
                    raise ValueError("full council errors belong to another native group")
            if reference.kind == "behavior_differences":
                raw = json.loads(
                    project.artifacts.read_bytes(
                        reference.artifact.artifact_id,
                        reference.file,
                    )
                )
                source = ArtifactInput.model_validate(raw["source"])
                report = load_behavior_differences(project, source)
                if (
                    manifest.inputs != (source,)
                    or report.study != group
                    or report.tier != study.initial.actor_alias
                    or report.judge != study.episode_judge.identity
                    or report.spec != mining.behaviors
                    or sha256_json(report.bank) != frozen.contract.banks["behavior"]
                ):
                    raise ValueError("full council behavior belongs to another group or judge")
                owners.add(source)
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


def load_council_descendant(
    project: ProjectStore,
    reference: ArtifactInput,
) -> CouncilStudyDescendant:
    """Read and independently verify a full study descendant without writes or dispatch."""
    _pinned(project, reference, DESCENDANT_TYPE)
    stored = project.artifacts.read(reference.artifact_id)
    saved = CouncilStudyDescendant.model_validate_json(stored.payloads["descendant.json"])
    revision = saved.qualified.producer_revision
    if stored.manifest.code_revision != revision:
        raise ValueError("full council descendant lost its actual executor producer")
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

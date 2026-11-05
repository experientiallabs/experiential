"""Typed pilot revisions bind exact existing compatible evidence without rebuilding it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import load_task_set
from exp.optimize.evaluation.variance.build_records import (
    REPORT_TYPE,
    VarianceBuildReport,
    load_exposure_plan,
)
from exp.simulation.mining.environment_families import (
    family_review_identity,
    load_environment_families,
)
from exp.simulation.mining.environment_review import review_protocol_sha256
from exp.simulation.rules.contracts import tool_schema_coverage

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationSpec


class BootstrapPilotRevision(ContractModel):
    """Exact selection changes and immutable evidence for a completed family build.

    Attributes:
        task_id: Explicit replacement mining probe from the saved ranked selection.
        maximum_family_tools: Full selected family tool surface, without trimming.
        maximum_topics: One topic per declared tool plus the environment topic.
        saved_build: Actual completed variance report, never a renamed copy.
        build_producer_revision: Historical producer of that completed report.
        family_id: Existing compatible placement of the replacement task.
        scope_sha256: Exact saved family scope, including corpus and runtime references.
        contracts_sha256: All registered family tool contracts.
        task_set: Original task set containing the selected probe.
        exposure: Existing mutually exclusive learning, mining and final-evaluation roles.
        families: Existing placed family record.
        reviewed_families: Original review input identity, before mining tasks were placed.
        index: Existing scope's retrieval index.
        world_model: Existing scope's grounded world.
        tag_retrieval: Existing scope's tag retrieval binding.
        protocol_sha256: Review protocol under which the completed build is reused.
    """

    task_id: str = Field(min_length=1)
    maximum_family_tools: int = Field(ge=1, le=10000, strict=True)
    maximum_topics: int = Field(ge=1, le=256, strict=True)
    saved_build: ArtifactInput
    build_producer_revision: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    scope_sha256: Sha256
    contracts_sha256: Sha256
    task_set: ArtifactInput
    exposure: ArtifactInput
    families: ArtifactInput
    reviewed_families: ArtifactInput
    index: ArtifactInput
    world_model: ArtifactInput
    tag_retrieval: ArtifactInput
    protocol_sha256: Sha256

    @property
    def references(self) -> tuple[ArtifactInput, ...]:
        """Return the exact immutable evidence roots included in successor ancestry."""
        return (
            self.saved_build,
            self.task_set,
            self.exposure,
            self.families,
            self.reviewed_families,
            self.index,
            self.world_model,
            self.tag_retrieval,
        )


def load_pilot_build(
    project: ProjectStore, revision: BootstrapPilotRevision
) -> VarianceBuildReport:
    """Verify every bound artifact and the actual historical build report's producer."""
    for reference in revision.references:
        if artifact_input(project.artifacts.read(reference.artifact_id).manifest) != reference:
            raise ValueError("pilot evidence reference changed")
    manifest = project.artifacts.read(revision.saved_build.artifact_id).manifest
    report = VarianceBuildReport.model_validate_json(
        project.artifacts.read_bytes(revision.saved_build.artifact_id, "report.json")
    )
    if (
        manifest.artifact_type != REPORT_TYPE
        or manifest.code_revision != revision.build_producer_revision
        or revision.saved_build.artifact_id != report.identity + "-result"
        or report.plan.identity != report.identity
        or report.build != project.load_project().build
    ):
        raise ValueError("pilot saved build identity, producer or original source differs")
    return report


def validate_pilot_revision(
    project: ProjectStore,
    revision: BootstrapPilotRevision,
    baseline: BootstrapPreparationSpec,
    *,
    expected_build: str,
) -> VarianceBuildReport:
    """Require a ranked, exactly covered mining probe in the predecessor's completed build.

    Args:
        project: Owner of the unchanged source and saved variance build.
        revision: Concrete reviewed selection and evidence values.
        baseline: Predecessor's verified execution settings.
        expected_build: Predecessor's exact build identity.

    Returns:
        The real saved report, after scope, contracts and exposure verification.

    Raises:
        ValueError: A reference, scope, role, contract, bound or protocol differs.
    """
    report = load_pilot_build(project, revision)
    scoping = report.environment_families
    if (
        report.identity != expected_build
        or report.spec != baseline.build.spec
        or scoping is None
        or report.build.task_set != revision.task_set
        or report.exposure != revision.exposure
        or scoping.families != revision.families
        or revision.protocol_sha256 != review_protocol_sha256()
    ):
        raise ValueError("pilot build settings, ancestry or protocol differs")
    placed_manifest = project.artifacts.read(revision.families.artifact_id).manifest
    reviewed = load_environment_families(project.artifacts, revision.reviewed_families)
    reviewer = baseline.build.reviewer
    assert reviewer is not None
    expected_review = family_review_identity(
        clustering=reviewed.clustering,
        model=reviewed.model,
        configuration_sha256=sha256_json(reviewer),
        spec=reviewed.spec,
        evidence_sha256=reviewed.evidence_sha256,
        protocol_sha256=revision.protocol_sha256,
        review_window=reviewed.review_window,
        unclustered=reviewed.unclustered,
    )
    if (
        placed_manifest.inputs != (revision.reviewed_families,)
        or revision.reviewed_families.artifact_id != expected_review
        or reviewed.model != reviewer.model.model_dump(mode="json")
        or baseline.build.spec.environment_families is None
        or reviewed.spec != baseline.build.spec.environment_families.review
        or reviewed.review_window != baseline.build.spec.environment_families.review_window
    ):
        raise ValueError("pilot historical review protocol or configuration differs")
    record = load_environment_families(project.artifacts, revision.families)
    families = {family.family_id: family for family in record.families}
    placements = scoping.placements_by_task()
    scopes = {scope.family_id: scope for scope in scoping.scopes}
    tasks = {
        task.task_id: task
        for task in load_task_set(project.artifacts, revision.task_set.artifact_id).tasks
    }
    exposure = load_exposure_plan(project, revision.exposure)
    eligible: list[str] = []
    for task_id in report.selected_probes:
        for family_id in placements.get(task_id, ()):
            if family_id not in scopes or family_id not in families:
                continue
            family, scope, task = families[family_id], scopes[family_id], tasks[task_id]
            if family.status != "compatible" or scope.status != "compatible":
                continue
            exposure.require_lineage(task.lineage_group_id, "mining_probes")
            exposure.require(task.source_trace_ids, "mining_probes")
            exposure.require(scope.members, "world_learning")
            exposure.require(scope.rules_traces, "world_learning")
            exposure.require(scope.retrieval_scope.ids, "world_learning")
            if tool_schema_coverage(task.tools, family.tool_contracts).status == "schema_matches":
                eligible.append(task_id)
                break
    if not eligible or revision.task_id != eligible[0]:
        raise ValueError(
            "pilot task must be the first ranked exactly covered compatible mining probe"
        )
    if revision.family_id not in scopes or revision.family_id not in families:
        raise ValueError("pilot names no existing compatible family")
    scope, family = scopes[revision.family_id], families[revision.family_id]
    if (
        revision.family_id not in placements[revision.task_id]
        or tool_schema_coverage(tasks[revision.task_id].tools, family.tool_contracts).status
        != "schema_matches"
        or sha256_json(scope) != revision.scope_sha256
        or sha256_json(family.model_dump(mode="json")["tool_contracts"])
        != revision.contracts_sha256
        or set(scope.tools) != set(family.tool_contracts)
        or scope.index != revision.index
        or scope.world_model != revision.world_model
        or scope.tag_retrieval != revision.tag_retrieval
        or len(scope.rules_traces) > baseline.limits.maximum_family_traces
        or revision.maximum_family_tools != len(scope.tools)
        or revision.maximum_topics != len(scope.tools) + 1
        or baseline.bootstrap.rules is None
        or baseline.bootstrap.rules.memory is None
        or baseline.simulators
    ):
        raise ValueError("pilot scope, contracts or exact tool/topic bounds differ")
    return report


def apply_pilot_revision(
    spec: BootstrapPreparationSpec, revision: BootstrapPilotRevision
) -> BootstrapPreparationSpec:
    """Change only the explicit pin, full family-tool bound and memory-topic bound."""
    rules = spec.bootstrap.rules
    if rules is None or rules.memory is None:
        raise ValueError("pilot topic revision requires registered rule-memory settings")
    rules = rules.model_copy(
        update={
            "memory": rules.memory.model_copy(update={"maximum_topics": revision.maximum_topics})
        }
    )
    return type(spec).model_validate(
        spec.model_copy(
            update={
                "task_ids": (revision.task_id,),
                "limits": spec.limits.model_copy(
                    update={"maximum_family_tools": revision.maximum_family_tools}
                ),
                "bootstrap": spec.bootstrap.model_copy(update={"rules": rules}),
            }
        ).model_dump(mode="json")
    )

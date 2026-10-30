"""Bounded multi-family authoring, measurement and delivery through existing stage owners.

This application qualifies declared development observable contracts. It does not upgrade
synthetic controls to native truth or claim protected release certification.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from filelock import Timeout
from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.actors import (
    ActorPanelSpec,
    PanelActor,
    preflight_actor_panel,
)
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.factory.candidates import (
    CandidateBatch,
    CandidateBatchSpec,
    preflight_candidate_batch,
    run_candidate_batch,
    saved_candidate_batch,
)
from exp.optimize.evaluation.factory.proposals import (
    TaskProposalSpec,
    preflight_task_proposals,
    propose_tasks,
    saved_task_proposals,
)
from exp.optimize.evaluation.factory.quality import TaskQualityReview
from exp.optimize.evaluation.factory.quality_proxy import (
    QualityProxySpec,
    preflight_quality_proxy,
    proxy_settings,
)
from exp.simulation.capability_briefs import load_capability_brief
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.evaluation.calibration import CalibrationCase, CalibrationPack
from exp.simulation.task_distribution import (
    DistributionSelection,
    FamilyCandidate,
    TaskDistribution,
    select_distribution,
)


class FactoryFamily(ContractModel):
    """One seed world/intent/verifier and its independent control inventory.

    Attributes:
        proposal: Existing bounded author request; all two-to-four proposals are measured.
        controls: Private independently witnessed controls, never supplied to the author.
        strata: Intended coverage cells; semantic applicability remains a declared contract.
        information: Declared information condition of this seed world.
        horizon: Declared workflow horizon, unrelated to model count or attempts.
        quality_reviews: Human task-quality attestations, separate from mechanical controls.
        quality_proxy: Optional frozen judge override; default is the last panel model.
        replay_native_records: Explicit permission for the child batch's native witness replay.
    """

    proposal: TaskProposalSpec
    controls: tuple[CalibrationCase, ...] = Field(min_length=1)
    strata: tuple[str, ...] = Field(min_length=1)
    information: Literal["available", "discoverable", "conflicting", "unavailable"]
    horizon: Literal["single_decision", "multi_step", "long_horizon"]
    quality_reviews: tuple[TaskQualityReview, ...] = ()
    quality_proxy: QualityProxySpec | None = None
    replay_native_records: bool = Field(
        default=False, strict=True, exclude_if=lambda value: not value
    )


class TaskFactorySpec(ContractModel):
    """A bounded development factory with a frozen population, mixture and allowance.

    Attributes:
        families: Seed tasks from distinct needs or workflows; related seeds remain grouped.
        distribution: Desired user/capability mix, fixed before seeing candidate outcomes.
        actors: Model panel shared across families; tier separation never grants correctness.
        maximum_author_actor_attempts: Conservative sum of author, actor and proxy ledger ceilings.
            Simulator/runtime calls retain their separately declared trusted runtime budgets.
        deadline: Outer dispatch limit; every child deadline must fit within it.
        destination: Export parent, with unique content-addressed directories per family.
        namespace: Harbor task namespace.
    """

    families: tuple[FactoryFamily, ...] = Field(min_length=1, max_length=100)
    distribution: TaskDistribution
    actors: tuple[PanelActor, ...] = Field(min_length=2, max_length=10)
    maximum_author_actor_attempts: int = Field(ge=1, le=100_000, strict=True)
    deadline: AwareDatetime
    destination: Path
    namespace: str = Field(default="factory", pattern=r"^[a-z0-9][a-z0-9.-]*$")

    @model_validator(mode="after")
    def distinct(self) -> Self:
        """Repeated seeds or model aliases cannot manufacture a larger denominator."""
        if len({row.proposal.base for row in self.families}) != len(self.families):
            raise ValueError("factory seed task references must be distinct")
        if len({actor.alias for actor in self.actors}) != len(self.actors):
            raise ValueError("factory actor aliases must be distinct")
        if any(row.proposal.deadline > self.deadline for row in self.families):
            raise ValueError("factory author deadline exceeds the outer deadline")
        return self

    def identity(self, code_revision: str) -> str:
        """Bind population, delivery, policy and producing source before any calls."""
        return stable_id("task-factory", {"spec": sha256_json(self), "code": code_revision})


class FactoryCell(ContractModel):
    """One seed's complete disposition, including absent or failed child work.

    Attributes:
        seed: Exact original task; its variants never add source-family breadth.
        proposals: Saved original candidate inventory, when available.
        batch: Complete feasibility, qualification, panel and package measurements.
        status: Scoped qualification, pending review, rejection, failure or no dispatch.
        phase: Last application phase, useful without leaking raw exception text.
        error_type: Sanitized error class, never an implicit retry instruction.
    """

    seed: ArtifactInput
    proposals: ArtifactInput | None = None
    batch: ArtifactInput | None = None
    status: Literal[
        "qualified", "rejected", "needs_review", "operationally_incomplete", "not_started"
    ]
    phase: Literal["coverage", "author", "measurement", "complete"]
    error_type: str | None = None


class TaskFactoryResult(ContractModel):
    """A reproducible development package inventory with honest family denominators.

    Attributes:
        spec: Exact registered pipeline manifest.
        cells: Every seed family in registration order, including errors and rejections.
        coverage: Quotas filled by qualified distinct components, never by attempts or variants.
        status: Overall disposition under the fixed coverage policy.
        counts: Separate seeds, components, candidates, packages and operational failures.
        scope: Explicit boundaries on qualification and coverage claims.
    """

    spec: TaskFactorySpec
    cells: tuple[FactoryCell, ...]
    coverage: DistributionSelection
    status: Literal[
        "development_qualified", "needs_review", "insufficient_evidence", "operationally_incomplete"
    ]
    counts: JsonObject
    scope: str = (
        "development observable contracts with provisional rubric-proxy task quality "
        "and declared coverage; "
        "no protected release confirmation, subjective judgment calibration, native fidelity, "
        "semantic novelty, deployed containment or training-benefit claim; author/actor budgets "
        "exclude separately bounded runtime inference; no weights are trained"
    )


def factory_population(project: ProjectStore, spec: TaskFactorySpec) -> tuple[FamilyCandidate, ...]:
    """Resolve grouping from saved source tasks, not mutable family labels or variant counts."""
    strata = {row.id: row for row in spec.distribution.strata}
    population = []
    for family in spec.families:
        base = load_executable_task(project, family.proposal.base)
        point = load_continuation_checkpoint(project.artifacts, base.job.study.checkpoint)
        brief = load_capability_brief(project.artifacts, family.proposal.brief)
        brief.require_task(point.task, base.intent)
        for name in family.strata:
            row = strata.get(name)
            if row is None or (
                row.user,
                row.capability,
                row.category,
                row.information,
                row.horizon,
            ) != (brief.user, brief.capability, brief.category, family.information, family.horizon):
                raise ValueError(
                    "family coverage differs from its saved brief or declared conditions"
                )
        population.append(
            FamilyCandidate(
                id=family.proposal.base.artifact_id,
                strata=family.strata,
                lineages=(point.task.lineage_group_id,),
                traces=point.task.source_trace_ids,
                workflow=sha256_json(
                    {
                        "intent": base.intent,
                        "verifier": base.job.study.verifier.model_dump(mode="json"),
                        "tools": [tool.model_dump(mode="json") for tool in point.task.tools],
                    }
                ),
            )
        )
    return tuple(population)


def factory_attempt_bound(project: ProjectStore, spec: TaskFactorySpec) -> int:
    """Sum the worst case for every proposed candidate, without counting saved work as free."""
    return sum(
        family.proposal.maximum_physical_attempts
        + 4
        * (1 + len(spec.actors))
        * load_executable_task(project, family.proposal.base).job.maximum_physical_attempts
        + 4
        * proxy_settings(
            family.quality_proxy,
            load_executable_task(project, family.proposal.base),
            spec.actors[-1],
        ).maximum_physical_attempts
        for family in spec.families
    )


def _batch(
    project: ProjectStore, spec: TaskFactorySpec, family: FactoryFamily, revision: str
) -> CandidateBatchSpec:
    """Compose the existing one-family measurement owner without widening its authority."""
    identity = family.proposal.identity(revision)
    proposals = artifact_input(project.artifacts.read(identity).manifest)
    return CandidateBatchSpec(
        proposals=proposals,
        controls=family.controls,
        actors=spec.actors,
        destination=spec.destination / family.proposal.base.artifact_id,
        namespace=spec.namespace,
        quality_reviews=family.quality_reviews,
        quality_proxy=family.quality_proxy,
        replay_native_records=family.replay_native_records,
    )


def preflight_task_factory(
    project: ProjectStore, spec: TaskFactorySpec, *, code_revision: str
) -> None:
    """Check all known budgets, labels and interfaces before the first author call."""
    factory_population(project, spec)
    if factory_attempt_bound(project, spec) > spec.maximum_author_actor_attempts:
        raise ValueError("factory child attempt ceilings exceed its frozen author/actor allowance")
    for family in spec.families:
        task = load_executable_task(project, family.proposal.base)
        proxy = proxy_settings(family.quality_proxy, task, spec.actors[-1])
        if proxy.deadline > spec.deadline:
            raise ValueError("factory quality-proxy deadline exceeds the outer deadline")
        if task.job.deadline > spec.deadline:
            raise ValueError("factory actor deadline exceeds the outer deadline")
        if task.job.study.training_specification is not None:
            raise ValueError("factory evaluation must not enable training admission")
        if (
            project.artifacts.read(family.proposal.base.artifact_id).manifest.code_revision
            != code_revision
        ):
            raise ValueError("factory seed source revision differs; republish its frozen task")
        CalibrationPack(
            task=family.proposal.base,
            parent=task.job.study.verifier,
            candidate=task.job.study.verifier,
            cases=family.controls,
        )
        for case in family.controls:
            witness = case.challenge.witness
            if (
                witness is None
                or artifact_input(project.artifacts.read(witness.artifact_id).manifest) != witness
            ):
                raise ValueError("factory requires exact independent control witness references")
        if saved_task_proposals(project, family.proposal, code_revision=code_revision) is None:
            if family.quality_reviews:
                raise ValueError("quality reviews require an existing exact proposal population")
            preflight_quality_proxy(project, proxy)
            preflight_task_proposals(project, family.proposal)
            preflight_actor_panel(project, ActorPanelSpec(cases=(task.job,), actors=spec.actors))
        else:
            batch = _batch(project, spec, family, code_revision)
            if saved_candidate_batch(project, batch, code_revision=code_revision) is None:
                preflight_candidate_batch(project, batch, code_revision=code_revision)


def _sources(spec: TaskFactorySpec) -> tuple[ArtifactInput, ...]:
    """Bind all public seeds and private control evidence to the frozen registration."""
    return tuple(
        sorted(
            {
                ref
                for family in spec.families
                for ref in (
                    family.proposal.base,
                    family.proposal.brief,
                    *(ref for review in family.quality_reviews for ref in review.sources()),
                    *(case.challenge.witness for case in family.controls if case.challenge.witness),
                )
            },
            key=lambda ref: ref.artifact_id,
        )
    )


def _result(
    project: ProjectStore, spec: TaskFactorySpec, cells: tuple[FactoryCell, ...], revision: str
) -> TaskFactoryResult:
    """Re-derive family coverage and yield from verified complete child inventories."""
    eligible = set()
    candidates = packaged = failures = 0
    for family, cell in zip(spec.families, cells, strict=True):
        if cell.seed != family.proposal.base:
            raise ValueError("factory lost its original seed population")
        if cell.proposals is not None:
            proposals = saved_task_proposals(project, family.proposal, code_revision=revision)
            if proposals is None or cell.proposals != artifact_input(
                project.artifacts.read(family.proposal.identity(revision)).manifest
            ):
                raise ValueError("factory proposals differ from the original author draw")
            candidates += len(proposals.variants)
        if cell.batch is not None:
            if cell.proposals is None:
                raise ValueError("factory measured candidates without their author evidence")
            batch_spec = _batch(project, spec, family, revision)
            batch = saved_candidate_batch(project, batch_spec, code_revision=revision)
            if batch is None or cell.batch != artifact_input(
                project.artifacts.read(batch_spec.identity(revision) + "-result").manifest
            ):
                raise ValueError("factory measurement lost its exact batch evidence")
            status = _batch_status(batch)
            if status != cell.status:
                raise ValueError("factory disposition does not reproduce its child evidence")
            count = sum(row.status == "packaged" for row in batch.candidates)
            packaged += count
            failures += sum(row.status == "unavailable" for row in batch.candidates)
            if status == "qualified":
                eligible.add(cell.seed.artifact_id)
        elif cell.status not in ("not_started", "operationally_incomplete"):
            raise ValueError("factory qualified a seed without complete measurements")
        if cell.status == "operationally_incomplete" and cell.batch is None:
            failures += 1
    coverage = select_distribution(
        spec.distribution, factory_population(project, spec), eligible=frozenset(eligible)
    )
    status = (
        "development_qualified"
        if not any(coverage.deficits.values())
        else "operationally_incomplete"
        if failures
        else "needs_review"
        if any(cell.status == "needs_review" for cell in cells)
        else "insufficient_evidence"
    )
    return TaskFactoryResult(
        spec=spec,
        cells=cells,
        coverage=coverage,
        status=status,
        counts={
            "seeds": len(cells),
            "source_workflow_components": len(coverage.components),
            "proposed_candidates": candidates,
            "packaged_candidates": packaged,
            "qualified_seed_families": len(eligible),
            "selected_components": sum(map(len, coverage.selected.values())),
            "operational_failures": failures,
        },
    )


def _batch_status(
    batch: CandidateBatch,
) -> Literal["qualified", "rejected", "needs_review", "operationally_incomplete"]:
    """Qualification depends on verified packages; actor difficulty is only descriptive."""
    if any(row.status == "packaged" for row in batch.candidates):
        return "qualified"
    return (
        "operationally_incomplete"
        if any(row.status == "unavailable" for row in batch.candidates)
        else "needs_review"
        if any(row.status == "needs_review" for row in batch.candidates)
        else "rejected"
    )


def saved_task_factory(
    project: ProjectStore, spec: TaskFactorySpec, *, code_revision: str
) -> TaskFactoryResult | None:
    """Verify saved results and exported bytes without credentials, imports or new inference."""
    store, identity = project.artifacts, spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    registration = artifact_input(store.read(identity).manifest)
    if TaskFactorySpec.model_validate_json(
        store.read_bytes(identity, "spec.json")
    ) != spec or store.read(identity).manifest.inputs != _sources(spec):
        raise ValueError("factory registration changed its original policy or sources")
    result = TaskFactoryResult.model_validate_json(
        store.read_bytes(identity + "-result", "report.json")
    )
    if result.spec != spec or _result(project, spec, result.cells, code_revision) != result:
        raise ValueError("factory report does not reproduce its saved evidence")
    inputs = {registration}
    for index, cell in enumerate(result.cells):
        manifest = store.read(f"{identity}-family-{index}").manifest
        if FactoryCell.model_validate_json(
            store.read_bytes(manifest.artifact_id, "cell.json")
        ) != cell or set(manifest.inputs) != {
            registration,
            cell.seed,
            *(ref for ref in (cell.proposals, cell.batch) if ref is not None),
        }:
            raise ValueError("factory cell lost its retained stage inventory")
        inputs.add(artifact_input(manifest))
    if set(store.read(identity + "-result").manifest.inputs) != inputs:
        raise ValueError("factory result lost a planned family")
    return result


def factory_requires_execution(
    project: ProjectStore, spec: TaskFactorySpec, *, code_revision: str
) -> bool:
    """Allow evidence-only assembly without asking to buy already committed child work."""
    if saved_task_factory(project, spec, code_revision=code_revision) is not None:
        return False
    planned = select_distribution(spec.distribution, factory_population(project, spec))
    if any(planned.deficits.values()):
        return False
    for index, family in enumerate(spec.families):
        if project.artifacts.exists(f"{spec.identity(code_revision)}-family-{index}"):
            continue
        if saved_task_proposals(project, family.proposal, code_revision=code_revision) is None:
            return True
        if (
            saved_candidate_batch(
                project, _batch(project, spec, family, code_revision), code_revision=code_revision
            )
            is None
        ):
            return True
    return False


def run_task_factory(
    project: ProjectStore,
    spec: TaskFactorySpec,
    *,
    code_revision: str,
    progress: ProgressHook | None = None,
) -> TaskFactoryResult:
    """Run the frozen finite population; saved and unknown outcomes are never silently redrawn."""
    spec = TaskFactorySpec.model_validate_json(spec.model_dump_json())
    store, identity = project.artifacts, spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        previous = saved_task_factory(project, spec, code_revision=code_revision)
        if previous is not None:
            return previous
        preflight_task_factory(project, spec, code_revision=code_revision)
        planned = select_distribution(spec.distribution, factory_population(project, spec))
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=_sources(spec),
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="task-factory-registration",
                envelope=envelope,
                files={"spec.json": spec},
            )
        registration = artifact_input(store.read(identity).manifest)
        cells, dependencies = [], {registration}
        for index, family in enumerate(spec.families):
            key = f"{identity}-family-{index}"
            report(progress, "Task families", completed=index, total=len(spec.families))
            if store.exists(key):
                cell = FactoryCell.model_validate_json(store.read_bytes(key, "cell.json"))
            else:
                cell = FactoryCell(
                    seed=family.proposal.base, status="not_started", phase="coverage"
                )
                if not any(planned.deficits.values()):
                    cell = cell.model_copy(update={"phase": "author"})
                    try:
                        propose_tasks(project, family.proposal, code_revision=code_revision)
                        proposals = artifact_input(
                            store.read(family.proposal.identity(code_revision)).manifest
                        )
                        cell = cell.model_copy(
                            update={"proposals": proposals, "phase": "measurement"}
                        )
                        batch_spec = _batch(project, spec, family, code_revision)
                        batch = run_candidate_batch(
                            project, batch_spec, code_revision=code_revision
                        )
                        cell = cell.model_copy(
                            update={
                                "batch": artifact_input(
                                    store.read(
                                        batch_spec.identity(code_revision) + "-result"
                                    ).manifest
                                ),
                                "status": _batch_status(batch),
                                "phase": "complete",
                            }
                        )
                    except Timeout:
                        raise  # Busy work remains pending; the owning process may still finish it.
                    except (OSError, ValueError, RuntimeError) as error:
                        cell = cell.model_copy(
                            update={
                                "status": "operationally_incomplete",
                                "error_type": type(error).__name__,
                            }
                        )
                inputs = {
                    registration,
                    cell.seed,
                    *(ref for ref in (cell.proposals, cell.batch) if ref is not None),
                }
                store.write_json(
                    artifact_id=key,
                    artifact_type="task-factory-cell",
                    envelope=envelope.model_copy(
                        update={"inputs": tuple(sorted(inputs, key=lambda ref: ref.artifact_id))}
                    ),
                    files={"cell.json": cell},
                )
            cells.append(cell)
            dependencies.add(artifact_input(store.read(key).manifest))
        result = _result(project, spec, tuple(cells), code_revision)
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="task-factory-result",
            envelope=envelope.model_copy(
                update={"inputs": tuple(sorted(dependencies, key=lambda ref: ref.artifact_id))}
            ),
            files={"report.json": result},
        )
        report(progress, "Task families", completed=len(spec.families), total=len(spec.families))
        return result

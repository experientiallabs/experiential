"""Measure every saved task prototype through existing execution and qualification owners."""

import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from filelock import Timeout
from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    FailureCode,
    JsonObject,
    StructuredFailure,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.actors import (
    ActorPanelSpec,
    PanelActor,
    preflight_actor_panel,
    run_actor_panel,
    saved_actor_panel,
)
from exp.optimize.evaluation.continuations.job import (
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.executable.deferral import (
    deferred_execution,
    record_execution_deferral,
)
from exp.optimize.evaluation.executable.package import (
    export_executable_harbor,
    verify_executable_package,
)
from exp.optimize.evaluation.executable.qualification import (
    ExecutableQualificationSpec,
    load_executable_qualification,
    qualify_executable_task,
)
from exp.optimize.evaluation.executable.task import ExecutableTask, load_executable_task
from exp.optimize.evaluation.factory.proposals import load_task_proposals
from exp.optimize.evaluation.factory.quality import (
    QualityDisposition,
    TaskQualityReview,
    quality_disposition,
    validate_quality_reviews,
)
from exp.optimize.evaluation.factory.quality_proxy import (
    QualityProxySpec,
    preflight_quality_proxy,
    proxy_settings,
    review_task_quality,
)
from exp.optimize.evaluation.factory.variants import load_task_variant
from exp.optimize.evaluation.harbor.export import verify_harbor_export
from exp.simulation.evaluation.calibration import CalibrationCase


class CandidateBatchSpec(ContractModel):
    """A preregistered evaluation of all candidates; controls never reach the author.

    Attributes:
        proposals: Complete saved proposal population, with no post-outcome candidate selection.
        controls: Independent source-backed verifier controls for this unchanged intent/verifier.
        actors: Frozen declared weak-to-strong comparison models.
        minimum_difference: Descriptive separation threshold, never correctness approval.
        destination: Optional exports after observable-contract and task-quality approval.
            Tier separation is reported independently and does not gate correctness or packaging.
        namespace: Safe Harbor organization label.
        quality_reviews: Independently attested reviews of the exact tasks and recipe.
        quality_proxy: Optional pinned override; otherwise use the last panel model as judge.
        replay_native_records: Explicitly allow native witness replay during qualification/readback.
    """

    proposals: ArtifactInput
    controls: tuple[CalibrationCase, ...] = Field(min_length=1)
    actors: tuple[PanelActor, ...] = Field(min_length=2, max_length=10)
    minimum_difference: float = Field(default=0.05, gt=0, le=1, allow_inf_nan=False)
    destination: Path | None = None
    namespace: str = Field(default="development", pattern=r"^[a-z0-9][a-z0-9.-]*$")
    quality_reviews: tuple[TaskQualityReview, ...] = ()
    quality_proxy: QualityProxySpec | None = None
    replay_native_records: bool = Field(
        default=False, strict=True, exclude_if=lambda value: not value
    )

    def proxy_for(self, task: ExecutableTask) -> QualityProxySpec:
        """Freeze the effective model/rubric policy from the declared panel and task pins."""
        return proxy_settings(self.quality_proxy, task, self.actors[-1])

    def identity(self, code_revision: str) -> str:
        """Bind all prototypes, controls, models and shipping rules before execution."""
        return stable_id("candidate-batch", {"spec": sha256_json(self), "code": code_revision})


class CandidateMeasurement(ContractModel):
    """One prototype's execution and disposition, including failed operational stages.

    Attributes:
        variant: Original proposal-linked task variant.
        task: Exact executable candidate.
        feasibility: Saved measurement under this candidate's own job.
        qualification: Successful or rejected observable-contract qualification.
        panel: Complete frozen model comparison, only after observable qualification.
        separation: Measured adjacent-tier signal, or None when no completed panel exists.
        status: Why this candidate is packaged, retained without a package, or unresolved.
        failure: Sanitized operational error, when present.
        package: Immutable output directory, when publication was requested and admitted.
        failure_detail: Operator-only phase and source references without raw exception text.
        proxy_review: Exact LLM rubric response and its request/dispatch ancestry, when used.
    """

    variant: ArtifactInput
    task: ArtifactInput
    separation: bool | None
    feasibility: ArtifactInput | None = None
    qualification: ArtifactInput | None = None
    panel: ArtifactInput | None = None
    status: Literal[
        "unavailable",
        "unqualified",
        "needs_review",
        "rejected_quality",
        "qualified_candidate",
        "packaged",
    ]
    failure: str | None = None
    package: Path | None = None
    failure_detail: StructuredFailure | None = None
    proxy_review: ArtifactInput | None = None


class CandidateBatch(ContractModel):
    """Complete candidate inventory and scoped qualification yield, never full certification.

    Attributes:
        spec: Original proposal inventory and preregistered downstream controls.
        candidates: Every original candidate in proposal order, including failures.
        counts: Counts by disposition, with the original denominator retained.
        scope: What measurement, packaging and reward qualifications do not establish.
    """

    spec: CandidateBatchSpec
    candidates: tuple[CandidateMeasurement, ...]
    counts: JsonObject
    scope: str = (
        "development candidates from one related task family; observable-contract qualification "
        "only; fixed-panel tier separation is a separate selection signal, not a shipping gate; "
        "task quality uses a frozen LLM rubric proxy or optional human attestation; "
        "no semantic-equivalence proof, independent fidelity, "
        "generalization, container-containment or full task certification; no training promotion"
    )


def _population(
    project: ProjectStore, spec: CandidateBatchSpec
) -> tuple[tuple[ArtifactInput, ArtifactInput, ExecutableTask], ...]:
    """Keep original order and every candidate, without ranking by observed difficulty."""
    report = load_task_proposals(project, spec.proposals)
    return tuple(
        (reference, variant.task, load_executable_task(project, variant.task))
        for reference in report.variants
        for variant in (load_task_variant(project, reference),)
    )


def _panel(spec: CandidateBatchSpec, task: ExecutableTask) -> ActorPanelSpec:
    """Use a separate one-case panel per variant so shared lineage is never called breadth."""
    return ActorPanelSpec(
        cases=(task.job,), actors=spec.actors, minimum_difference=spec.minimum_difference
    )


def preflight_candidate_batch(
    project: ProjectStore, spec: CandidateBatchSpec, *, code_revision: str
) -> None:
    """Check all planned runtimes/models and the complete finite allowance before any calls."""
    attempts = 0
    population = _population(project, spec)
    validate_quality_reviews(
        project,
        spec.quality_reviews,
        proposals=spec.proposals,
        tasks=frozenset(reference for _, reference, _ in population),
    )
    for _, reference, task in population:
        if project.artifacts.read(reference.artifact_id).manifest.code_revision != code_revision:
            raise ValueError("candidate task code revision differs; republish before measuring")
        preflight_actor_panel(project, _panel(spec, task))
        preflight_quality_proxy(project, spec.proxy_for(task))
        attempts += task.job.maximum_physical_attempts * (1 + len(spec.actors))
        attempts += spec.proxy_for(task).maximum_physical_attempts
    if attempts > 100_000:
        raise ValueError("candidate batch exceeds its combined actor-attempt ceiling")
    for case in spec.controls:
        if case.challenge.witness is None:
            raise ValueError("candidate controls require independent witnessed labels")
        ref = case.challenge.witness
        if artifact_input(project.artifacts.read(ref.artifact_id).manifest) != ref:
            raise ValueError("candidate control witness differs from its saved evidence")


def _separates(
    project: ProjectStore, spec: CandidateBatchSpec, task: ExecutableTask, *, code_revision: str
) -> bool:
    """Describe measured worst-case separation without granting or denying correctness."""
    panel = saved_actor_panel(project, _panel(spec, task), code_revision=code_revision)
    if panel is None:
        return False
    comparisons = panel.metrics["comparisons"]
    return (
        isinstance(comparisons, list)
        and bool(comparisons)
        and all(
            isinstance(row, dict) and row.get("separates_on_fixed_panel") is True
            for row in comparisons
        )
    )


def _package(
    project: ProjectStore,
    spec: CandidateBatchSpec,
    cell: CandidateMeasurement,
    index: int,
    *,
    existing_only: bool = False,
) -> Path:
    """Publish or verify the same exact qualified package, never overwrite changed output."""
    if spec.destination is None or cell.qualification is None:
        raise ValueError("candidate packaging needs a destination and a qualified candidate")
    quality, receipt = _quality_result(
        project,
        spec,
        cell,
        code_revision=project.artifacts.read(cell.task.artifact_id).manifest.code_revision,
        execute=False,
    )
    if quality != "accepted" or receipt != cell.proxy_review:
        raise ValueError("factory packaging requires its exact task-quality decision")
    path = spec.destination / cell.task.artifact_id
    name = f"{spec.namespace}/candidate-{index}"
    if path.exists():
        binding = verify_harbor_export(path)
        if (
            verify_executable_package(project, path) != cell.task
            or binding.get("qualification_evidence") != cell.qualification.model_dump(mode="json")
            or tomllib.loads((path / "task.toml").read_text())["task"]["name"] != name
        ):
            raise ValueError("candidate package differs from its registered export")
        return path
    if existing_only:
        raise ValueError("saved candidate package is missing")
    return export_executable_harbor(
        project, cell.task, path, name=name, qualification=cell.qualification
    )


def _quality_result(
    project: ProjectStore,
    spec: CandidateBatchSpec,
    cell: CandidateMeasurement,
    *,
    code_revision: str,
    execute: bool,
) -> tuple[QualityDisposition, ArtifactInput | None]:
    """Run the automatic rubric proxy unless an explicit human decision was supplied."""
    if any(
        row.task == cell.task and row.authority == "human" and row.approved
        for row in spec.quality_reviews
    ):
        return quality_disposition(spec.quality_reviews, cell.task), None
    assert cell.feasibility and cell.qualification and cell.panel
    reference, review = review_task_quality(
        project,
        spec.proxy_for(load_executable_task(project, cell.task)),
        task=cell.task,
        proposals=spec.proposals,
        feasibility=cell.feasibility,
        qualification=cell.qualification,
        panel=cell.panel,
        code_revision=code_revision,
        execute=execute,
    )
    return review.answer.disposition(), reference


def _measure(
    project: ProjectStore,
    spec: CandidateBatchSpec,
    cell: CandidateMeasurement,
    index: int,
    *,
    code_revision: str,
    execute: bool,
) -> CandidateMeasurement:
    """Resume only saved children after interruption; never repurchase an unknown parent cell."""
    store = project.artifacts
    task = load_executable_task(project, cell.task)
    phase = "feasibility"
    try:
        result = saved_continuation_job(project, task.job, code_revision=code_revision)
        if result is None and execute:
            result = execute_continuation_job(project, task.job, code_revision=code_revision)
        if result is None:
            return cell.model_copy(update={"failure": "UnfinishedFeasibilityIntent"})
        feasibility = continuation_result_reference(project, task.job, code_revision=code_revision)
        cell = cell.model_copy(update={"feasibility": feasibility})
        phase = "qualification"
        qualification = qualify_executable_task(
            project,
            ExecutableQualificationSpec(
                task=cell.task,
                feasibility=feasibility,
                cases=spec.controls,
                replay_native_records=spec.replay_native_records,
            ),
            created_at=datetime.now(UTC),
            code_revision=code_revision,
        )
        cell = cell.model_copy(update={"qualification": qualification})
        if not load_executable_qualification(
            project, qualification, allow_native_replay=spec.replay_native_records
        ).qualified:
            return cell.model_copy(update={"status": "unqualified"})
        phase = "actor_panel"
        panel_spec = _panel(spec, task)
        panel = saved_actor_panel(project, panel_spec, code_revision=code_revision)
        if panel is None and execute:
            panel = run_actor_panel(project, panel_spec, code_revision=code_revision)
        if panel is None:
            return cell.model_copy(update={"failure": "UnfinishedPanelIntent"})
        panel_ref = artifact_input(
            store.read(panel_spec.identity(code_revision) + "-result").manifest
        )
        cell = cell.model_copy(
            update={
                "panel": panel_ref,
                "separation": _separates(project, spec, task, code_revision=code_revision),
                "status": "qualified_candidate",
            }
        )
        phase = "quality_proxy"
        quality, proxy_review = _quality_result(
            project,
            spec,
            cell,
            code_revision=code_revision,
            execute=execute,
        )
        cell = cell.model_copy(update={"proxy_review": proxy_review})
        if quality != "accepted":
            return cell.model_copy(
                update={"status": "rejected_quality" if quality == "rejected" else "needs_review"}
            )
        if spec.destination is not None:
            phase = "package_export"
            cell = cell.model_copy(
                update={"package": _package(project, spec, cell, index), "status": "packaged"}
            )
        return cell
    except Timeout:
        raise  # Busy work remains pending; the owning process may still finish it.
    except (ValueError, OSError, RuntimeError) as exc:
        return cell.model_copy(
            update={
                "status": "unavailable",
                "failure": type(exc).__name__,
                "failure_detail": StructuredFailure(
                    code=FailureCode.VALIDATION
                    if isinstance(exc, ValueError)
                    else FailureCode.INTERNAL,
                    message=(
                        "Candidate stage failed; inspect the task and retained child receipts "
                        "before retrying. Completed stages remain immutable."
                    ),
                    exception_type=type(exc).__name__,
                    details={"phase": phase, "task": cell.task.model_dump(mode="json")},
                ),
            }
        )


def _inputs(
    registration: ArtifactInput,
    cell: CandidateMeasurement,
    deferrals: tuple[ArtifactInput, ...] = (),
) -> tuple[ArtifactInput, ...]:
    """Retain every completed stage's immutable evidence even if a later stage failed."""
    return tuple(
        sorted(
            {
                registration,
                *deferrals,
                cell.variant,
                cell.task,
                *(
                    ref
                    for ref in (cell.feasibility, cell.qualification, cell.panel, cell.proxy_review)
                    if ref is not None
                ),
            },
            key=lambda ref: ref.artifact_id,
        )
    )


def _counts(cells: tuple[CandidateMeasurement, ...]) -> JsonObject:
    """Report all dispositions against the entire proposal population."""
    return {
        "proposed": len(cells),
        **{
            status: sum(cell.status == status for cell in cells)
            for status in (
                "unavailable",
                "unqualified",
                "needs_review",
                "rejected_quality",
                "qualified_candidate",
                "packaged",
            )
        },
        "tier_separation": {
            "observed": sum(cell.separation is True for cell in cells),
            "not_observed": sum(cell.separation is False for cell in cells),
            "unavailable": sum(cell.separation is None for cell in cells),
        },
    }


def saved_candidate_batch(
    project: ProjectStore, spec: CandidateBatchSpec, *, code_revision: str
) -> CandidateBatch | None:
    """Re-derive stage identities and decisions without runtime imports, credentials or calls."""
    store, identity = project.artifacts, spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    report = CandidateBatch.model_validate_json(
        store.read_bytes(identity + "-result", "report.json")
    )
    registration = artifact_input(store.read(identity).manifest)
    sources = {
        spec.proposals,
        *(ref for review in spec.quality_reviews for ref in review.sources()),
        *(case.challenge.witness for case in spec.controls if case.challenge.witness),
    }
    if (
        report.spec != spec
        or CandidateBatchSpec.model_validate_json(store.read_bytes(identity, "spec.json")) != spec
        or set(store.read(identity).manifest.inputs) != sources
    ):
        raise ValueError("candidate batch lost its original registration")
    population = _population(project, spec)
    validate_quality_reviews(
        project,
        spec.quality_reviews,
        proposals=spec.proposals,
        tasks=frozenset(reference for _, reference, _ in population),
    )
    if len(report.candidates) != len(population):
        raise ValueError("candidate batch lost a proposed candidate")
    dependencies = {registration}
    for index, (cell, (variant, reference, task)) in enumerate(
        zip(report.candidates, population, strict=True)
    ):
        key = f"{identity}-candidate-{index}"
        if (
            cell.variant != variant
            or cell.task != reference
            or CandidateMeasurement.model_validate_json(store.read_bytes(key, "cell.json")) != cell
            or store.read(key).manifest.inputs
            != _inputs(registration, cell, deferred_execution(store, key))
        ):
            raise ValueError("candidate batch changed its planned population or stage evidence")
        if cell.feasibility is not None:
            if (
                saved_continuation_job(project, task.job, code_revision=code_revision) is None
                or continuation_result_reference(project, task.job, code_revision=code_revision)
                != cell.feasibility
            ):
                raise ValueError("candidate feasibility belongs to another runtime job")
        if cell.qualification is not None:
            qualification = load_executable_qualification(
                project, cell.qualification, allow_native_replay=spec.replay_native_records
            )
            if cell.feasibility is None:
                raise ValueError("candidate qualification has no feasible execution")
            if qualification.spec != ExecutableQualificationSpec(
                task=cell.task,
                feasibility=cell.feasibility,
                cases=spec.controls,
                replay_native_records=spec.replay_native_records,
            ):
                raise ValueError(
                    "candidate qualification changed its controls or feasible execution"
                )
        if cell.panel is not None:
            panel_spec = _panel(spec, task)
            panel = saved_actor_panel(project, panel_spec, code_revision=code_revision)
            if (
                panel is None
                or cell.separation != _separates(project, spec, task, code_revision=code_revision)
                or artifact_input(
                    store.read(panel_spec.identity(code_revision) + "-result").manifest
                )
                != cell.panel
            ):
                raise ValueError("candidate panel changed its model population")
        elif cell.separation is not None:
            raise ValueError("candidate separation has no measured panel")
        if cell.status != "unavailable":
            if cell.failure is not None or cell.qualification is None:
                raise ValueError("completed candidate lacks a qualification or has an error")
            qualified = load_executable_qualification(
                project, cell.qualification, allow_native_replay=spec.replay_native_records
            ).qualified
            quality = "needs_review"
            if qualified:
                quality, receipt = _quality_result(
                    project,
                    spec,
                    cell,
                    code_revision=code_revision,
                    execute=False,
                )
                if receipt != cell.proxy_review:
                    raise ValueError("candidate quality decision changed its original receipt")
            expected = (
                "unqualified"
                if not qualified
                else "needs_review"
                if quality == "needs_review"
                else "rejected_quality"
                if quality == "rejected"
                else "qualified_candidate"
                if spec.destination is None
                else "packaged"
            )
            if expected != cell.status or (not qualified and cell.panel is not None):
                raise ValueError("candidate disposition does not reproduce its evidence")
            if qualified and cell.panel is None:
                raise ValueError("completed qualified candidate has no measured panel")
            if cell.status == "packaged":
                if cell.package != _package(project, spec, cell, index, existing_only=True):
                    raise ValueError("candidate package location differs")
            elif cell.package is not None:
                raise ValueError("unshipped candidate unexpectedly has a package")
        elif cell.failure is None or cell.package is not None:
            raise ValueError(
                "unavailable candidate has no failure or unexpectedly exports a reward"
            )
        dependencies.add(artifact_input(store.read(key).manifest))
    if set(
        store.read(identity + "-result").manifest.inputs
    ) != dependencies or report.counts != _counts(report.candidates):
        raise ValueError("candidate batch counts or dependencies differ")
    return report


def run_candidate_batch(
    project: ProjectStore, spec: CandidateBatchSpec, *, code_revision: str
) -> CandidateBatch:
    """Measure and conditionally package every candidate; preserve rejected and unknown work."""
    spec = CandidateBatchSpec.model_validate_json(spec.model_dump_json())
    store, identity = project.artifacts, spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_candidate_batch(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        preflight_candidate_batch(project, spec, code_revision=code_revision)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(
                sorted(
                    {
                        spec.proposals,
                        *(ref for review in spec.quality_reviews for ref in review.sources()),
                        *(
                            case.challenge.witness
                            for case in spec.controls
                            if case.challenge.witness
                        ),
                    },
                    key=lambda ref: ref.artifact_id,
                )
            ),
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="candidate-batch-registration",
                envelope=envelope,
                files={"spec.json": spec},
            )
        registration = artifact_input(store.read(identity).manifest)
        cells, dependencies = [], {registration}
        for index, (variant, reference, _) in enumerate(_population(project, spec)):
            key = f"{identity}-candidate-{index}"
            if store.exists(key):
                cell = CandidateMeasurement.model_validate_json(store.read_bytes(key, "cell.json"))
            else:
                cell = CandidateMeasurement(
                    variant=variant, task=reference, status="unavailable", separation=None
                )
                execute = not store.exists(key + "-intent") or bool(deferred_execution(store, key))
                if not store.exists(key + "-intent"):
                    store.write_json(
                        artifact_id=key + "-intent",
                        artifact_type="candidate-batch-intent",
                        envelope=envelope.model_copy(
                            update={
                                "inputs": tuple(
                                    sorted(
                                        (registration, variant, reference),
                                        key=lambda ref: ref.artifact_id,
                                    )
                                )
                            }
                        ),
                        files={"cell.json": cell},
                    )
                try:
                    cell = _measure(
                        project, spec, cell, index, code_revision=code_revision, execute=execute
                    )
                except Timeout:
                    record_execution_deferral(store, key)
                    raise
                store.write_json(
                    artifact_id=key,
                    artifact_type="candidate-measurement",
                    envelope=envelope.model_copy(
                        update={
                            "inputs": _inputs(registration, cell, deferred_execution(store, key))
                        }
                    ),
                    files={"cell.json": cell},
                )
            cells.append(cell)
            dependencies.add(artifact_input(store.read(key).manifest))
        report = CandidateBatch(spec=spec, candidates=tuple(cells), counts=_counts(tuple(cells)))
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="candidate-batch",
            envelope=envelope.model_copy(
                update={"inputs": tuple(sorted(dependencies, key=lambda ref: ref.artifact_id))}
            ),
            files={"report.json": report},
        )
        return report

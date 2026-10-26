"""Paired actor execution, mechanical comparison and read-only trace council orchestration."""

from datetime import UTC, datetime
from functools import partial
from typing import Literal

from filelock import Timeout
from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    redact_secret_text,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.actor_equivalence import (
    EquivalenceRubric,
    evaluate_actor_equivalence,
)
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.failures import analyse_failures
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    preflight_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.optimize.evaluation.equivalence.calibration import (
    CaseScope,
    PolicyCalibration,
    calibrate,
)
from exp.optimize.evaluation.equivalence.dimensions import world_identity
from exp.optimize.evaluation.equivalence.evidence import (
    PairMeasurement,
    equivalence_metrics,
    measure_pair,
)
from exp.optimize.evaluation.equivalence.policy import (
    load_admission_policy,
    require_policy_precedes,
)
from exp.optimize.evaluation.equivalence.review import (
    ReviewModel,
    preflight_reviewer,
    review_call,
    review_inputs,
)
from exp.optimize.evaluation.equivalence.rubric import load_equivalence_rubric
from exp.optimize.evaluation.executable.deferral import (
    deferred_execution,
    record_execution_deferral,
)
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.evaluation.analysis import TraceAnalysis, analyze_traces
from exp.simulation.reconstruction.models import path_parts


class EquivalenceCase(ContractModel):
    """One frozen actor/task executed against two explicitly selected environments.

    Attributes:
        reference: Restorable executable reference and actor policy.
        candidate: Restorable simulator or mixed environment with the same public interface.
        initial_state_paths: Required equal private probe fields, relative to initial_state.
            A rubric with full state coverage requires at least one.
        final_state_paths: Material consequence fields to compare, relative to final_state.
            Select semantic properties, not incidental serialization or valid-method differences.
            A rubric with full state coverage requires at least one; with partial or unknown
            coverage an empty set leaves outcomes to the verifier while behavior stays judged.
    """

    reference: ContinuationJob
    candidate: ContinuationJob
    initial_state_paths: tuple[str, ...] = ()
    final_state_paths: tuple[str, ...] = ()

    @model_validator(mode="after")
    def same_policy(self) -> "EquivalenceCase":
        """Refuse changed policies and ambiguous pointers before registering any execution."""
        for name in (
            "actor_alias",
            "catalog_sha256",
            "maximum_model_calls",
            "system_prompt",
            "maximum_physical_attempts",
            "deadline",
        ):
            if getattr(self.reference, name) != getattr(self.candidate, name):
                raise ValueError(
                    f"paired actor policies differ at {name}; freeze the same actor policy"
                )
        for name in (
            "actor_model",
            "verifier",
            "repeats",
            "temperature",
            "maximum_output_tokens",
            "reasoning_effort",
        ):
            if getattr(self.reference.study, name) != getattr(self.candidate.study, name):
                raise ValueError(
                    f"paired actor studies differ at {name}; use the same evaluation contract"
                )
        for job in (self.reference, self.candidate):
            if (
                job.study.training_specification is not None
                or job.study.training_evidence is not None
            ):
                raise ValueError(
                    "equivalence measures environments; training admission must be separate"
                )
        for paths in (self.initial_state_paths, self.final_state_paths):
            if len(paths) != len(set(paths)):
                raise ValueError("equivalence state paths must be unique")
            for path in paths:
                path_parts(path)
        return self


class EquivalenceSpec(ContractModel):
    """Predeclared actor/environment pairs and bounded semantic reviewers.

    Attributes:
        rubric: Reviewed trajectory-derived judge rubric (``freeze-equivalence-rubric``),
            cited by the registration before any execution. Its development families must be
            disjoint from the evaluated ones. It shapes semantic ratings only; admission stays
            with ``policy``.
        cases: One case per actor/task binding; source families remain explicit in metrics.
        judge: Frozen semantic reviewer, separate from deterministic goal/effect checks.
        analysts: Read-only council reviewers; agreement does not establish correctness.
        maximum_review_attempts: Shared physical cap for judges, analysts and HTTP retries.
            Each planned review receives an equal integer allowance; unused capacity is not
            transferred to earlier retries at the expense of later council requests.
        deadline: Review dispatch deadline; saved evidence remains readable afterwards.
        policy: Registered admission policy, pinned before any execution, whose margins the
            report evaluates per dimension with family-clustered intervals. None keeps the
            report to measurements; it then calibrates no simulated trajectory.
    """

    rubric: ArtifactInput
    cases: tuple[EquivalenceCase, ...] = Field(min_length=1, max_length=100)
    judge: ReviewModel
    analysts: tuple[ReviewModel, ...] = Field(min_length=1, max_length=10)
    maximum_review_attempts: int = Field(ge=1, le=10000)
    deadline: AwareDatetime
    policy: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def bounded(self) -> "EquivalenceSpec":
        """Bound the full schedule and reject duplicate measurements or changing catalogs."""
        if len({row.alias for row in self.analysts}) != len(self.analysts):
            raise ValueError("equivalence council analyst aliases must be distinct")
        if len({sha256_json(row) for row in self.cases}) != len(self.cases):
            raise ValueError("equivalence cases must be distinct")
        if len({row.reference.catalog_sha256 for row in self.cases}) != 1:
            raise ValueError("equivalence study requires one frozen catalog")
        if self.maximum_review_attempts < self.planned_reviews():
            raise ValueError("review budget must allow at least one call per planned judge/analyst")
        if self.actor_attempt_bound() + self.maximum_review_attempts > 100000:
            raise ValueError("equivalence study exceeds its 100000 physical-attempt ceiling")
        return self

    def planned_reviews(self) -> int:
        """Count both pair judgments per repeat and every case's council request."""
        return sum(2 * case.reference.study.repeats + len(self.analysts) for case in self.cases)

    def review_attempt_allowance(self) -> int:
        """Keep early retries within their equal share of the frozen physical ceiling."""
        return max(1, self.maximum_review_attempts // self.planned_reviews())

    def actor_attempt_bound(self) -> int:
        """Count reference, reference-control and candidate actor provider allowances."""
        return sum(
            2 * row.reference.maximum_physical_attempts + row.candidate.maximum_physical_attempts
            for row in self.cases
        )

    def identity(self, revision: str) -> str:
        """Bind the whole comparison and implementation before dispatch."""
        return stable_id("actor-equivalence", {"spec": sha256_json(self), "code": revision})


class EquivalenceCell(ContractModel):
    """One planned execution including setup and dispatch failures.

    Attributes:
        case: Registered actor/task position.
        condition: Reference, repeated-reference control or candidate runtime.
        job: Exact execution registration, including a condition-specific identity.
        result: Saved complete study when available, including failed branches.
        failure: Sanitized setup or dispatch failure when no study could be produced.
    """

    case: int
    condition: Literal["reference", "reference_control", "candidate"]
    job: ContinuationJob
    result: ArtifactInput | None = None
    failure: str | None = None


class EquivalenceReport(ContractModel):
    """Complete paired evidence with separate measurements and diagnostic proposals.

    Attributes:
        spec: Frozen plan.
        cells: Every planned execution, including unavailable ones.
        measurements: Every branch pair for both comparison conditions.
        metrics: Mechanical agreement, unknown bounds and source-family inventory.
        analyses: Validated evidence-citing council proposals.
        analysis_failures: Explicit reviewer failures, retained without rerolling.
        calibration: Per-dimension verdicts against the registered policy's margins, absent
            without a policy.
        scope: Interpretation and qualification boundaries.
    """

    spec: EquivalenceSpec
    cells: tuple[EquivalenceCell, ...]
    measurements: tuple[PairMeasurement, ...]
    metrics: JsonObject
    analyses: dict[str, TraceAnalysis]
    analysis_failures: dict[str, str]
    calibration: PolicyCalibration | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scope: str = (
        "development paired execution against the selected reference; "
        "declared probes and verifier; "
        "no population equivalence, independent calibration, native-truth or training certificate"
    )


def equivalence_cells(spec: EquivalenceSpec, revision: str) -> tuple[EquivalenceCell, ...]:
    """Counterbalance reference/candidate order across cases and give controls fresh draws."""
    cells = []
    identity = spec.identity(revision)
    for index, case in enumerate(spec.cases):
        order = (
            ("reference", "candidate", "reference_control")
            if index % 2 == 0
            else ("candidate", "reference_control", "reference")
        )
        for condition in order:
            base = case.candidate if condition == "candidate" else case.reference
            study = base.study.model_copy(
                update={
                    "selector": f"{identity} case {index} {condition}",
                    "predicted_success": None,
                    "critic_prediction": None,
                    "selection_plan": None,
                }
            )
            cells.append(
                EquivalenceCell.model_validate(
                    {
                        "case": index,
                        "condition": condition,
                        "job": base.model_copy(update={"study": study}),
                    }
                )
            )
    return tuple(cells)


def _lineages(project: ProjectStore, spec: EquivalenceSpec) -> tuple[str, ...]:
    """Return the source lineage of every registered case, in case order."""
    return tuple(
        load_continuation_checkpoint(
            project.artifacts, case.reference.study.checkpoint
        ).task.lineage_group_id
        for case in spec.cases
    )


def _rubric(project: ProjectStore, spec: EquivalenceSpec) -> EquivalenceRubric:
    """Load the frozen rubric, refusing development families the comparison also evaluates.

    Lineages are grouped by the pinned admission policy's declared families, so a family is
    excluded as a whole; without a policy each lineage is its own family.
    """
    families = (
        {}
        if spec.policy is None
        else load_admission_policy(project.artifacts, spec.policy).families
    )
    return load_equivalence_rubric(
        project, spec.rubric, evaluation_lineages=_lineages(project, spec), families=families
    )


def _validate_cases(project: ProjectStore, spec: EquivalenceSpec) -> None:
    """Require identical public actor state while retaining distinct private worlds."""
    spec = EquivalenceSpec.model_validate(spec.model_dump(mode="json"))
    seen = set()
    for case in spec.cases:
        left = load_continuation_checkpoint(project.artifacts, case.reference.study.checkpoint)
        right = load_continuation_checkpoint(project.artifacts, case.candidate.study.checkpoint)
        if left.task != right.task or left.actor != right.actor:
            raise ValueError(
                "equivalence requires identical public task, tools and actor checkpoint"
            )
        key = (left.task.lineage_group_id, sha256_json(case.reference.study.actor_model))
        if key in seen:
            raise ValueError(
                "duplicate actor/source lineage; use repeats within its registered case"
            )
        seen.add(key)
    if _rubric(project, spec).state_coverage == "full" and any(
        not case.initial_state_paths or not case.final_state_paths for case in spec.cases
    ):
        raise ValueError(
            "a full-state rubric requires declared initial and final material probes in every case"
        )


def preflight_equivalence(project: ProjectStore, spec: EquivalenceSpec) -> None:
    """Verify the whole frozen schedule before any actor, simulator or reviewer call."""
    _validate_cases(project, spec)
    if spec.policy is not None:
        load_admission_policy(project.artifacts, spec.policy)
    for case in spec.cases:
        preflight_continuation_job(project, case.reference)
        preflight_continuation_job(project, case.candidate)
    for reviewer in (spec.judge, *spec.analysts):
        preflight_reviewer(project, reviewer, spec.cases[0].reference.catalog_sha256)


def _write(
    project: ProjectStore,
    identity: str,
    kind: str,
    filename: str,
    value: ContractModel | JsonObject,
    sources: set[ArtifactInput],
    revision: str,
) -> ArtifactInput:
    """Publish one exact stage with its evidence dependencies."""
    manifest = project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=kind,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=revision,
            inputs=tuple(sorted(sources, key=lambda ref: ref.artifact_id)),
        ),
        files={filename: value},
    )
    return artifact_input(manifest)


def _cell_branches(project: ProjectStore, cell: EquivalenceCell) -> tuple[ContinuationBranch, ...]:
    """Preserve the planned branch count even when a runtime never produced a study."""
    if cell.result is not None:
        study = load_study(project.artifacts, cell.result)
        if study.spec != cell.job.study or cell.failure is not None:
            raise ValueError("equivalence cell changed its planned study")
        return study.branches
    if not cell.failure:
        raise ValueError("equivalence cell has neither a result nor an explicit failure")
    return tuple(
        ContinuationBranch(
            index=index,
            status="outcome_unknown",
            outcome={"steps": [], "stop_reason": "not_started"},
            failure=cell.failure,
        )
        for index in range(cell.job.study.repeats)
    )


def _assemble(
    project: ProjectStore,
    spec: EquivalenceSpec,
    cells: tuple[EquivalenceCell, ...],
    revision: str,
    sources: set[ArtifactInput],
    *,
    allow_dispatch: bool,
    progress: ProgressHook | None = None,
) -> EquivalenceReport:
    """Review the complete measured inventory after actors finish, without feedback to actors."""
    identity = spec.identity(revision)
    rubric = _rubric(project, spec)
    base_sources = tuple(sorted(sources, key=lambda ref: ref.artifact_id))
    measurements, paired = [], {}
    families = set()
    scopes = []

    def review(
        reviewer: ReviewModel, body: JsonObject, key: str, consumer: Literal["analyst", "unknown"]
    ) -> JsonObject:
        """Use one durable shared review budget, retaining dependencies even on failure.

        The rubric's development sources are re-admitted under the source-access lock before
        each disclosure, so a source protected after the rubric was frozen stops later reviews.
        The lock is released before the provider call; when it is busy, its ``Timeout`` stops
        the report before anything is sent, and a later run retries the same request.
        """
        try:
            with CoordinationLock(
                project.artifacts.project_directory / "source-access.lock", timeout=0
            ):
                _rubric(project, spec)
            return review_call(
                project,
                reviewer,
                body,
                identity=key,
                panel_id=identity,
                catalog_sha256=spec.cases[0].reference.catalog_sha256,
                maximum_attempts=spec.maximum_review_attempts,
                maximum_call_attempts=spec.review_attempt_allowance(),
                deadline=spec.deadline,
                sources=base_sources,
                code_revision=revision,
                allow_dispatch=allow_dispatch,
                consumer=consumer,
            )
        finally:
            sources.update(review_inputs(project, key))

    for index, case in enumerate(spec.cases):
        checkpoint = load_continuation_checkpoint(
            project.artifacts, case.reference.study.checkpoint
        )
        families.add(checkpoint.task.lineage_group_id)
        candidate_point = load_continuation_checkpoint(
            project.artifacts, case.candidate.study.checkpoint
        )
        scopes.append(
            CaseScope(
                case=index,
                lineage=checkpoint.task.lineage_group_id,
                world=sha256_json(world_identity(candidate_point)),
                actor=sha256_json(case.candidate.study.actor_model),
            )
        )
        actor_id = sha256_json(
            {
                "checkpoint": checkpoint.actor.model_dump(mode="json"),
                "model": case.reference.study.actor_model.model_dump(mode="json"),
                "temperature": case.reference.study.temperature,
                "output_tokens": case.reference.study.maximum_output_tokens,
                **(
                    {}
                    if case.reference.study.reasoning_effort is None
                    else {"reasoning_effort": case.reference.study.reasoning_effort}
                ),
                "execution_policy": case.reference.model_dump(
                    mode="json",
                    exclude={"study", "runtime_file", "runtime_sha256", "configuration"},
                ),
                "code": revision,
            }
        )
        selected = {cell.condition: cell for cell in cells if cell.case == index}
        branches = {key: _cell_branches(project, cell) for key, cell in selected.items()}
        for condition in ("candidate", "reference_control"):
            for repeat, (reference, other) in enumerate(
                zip(branches["reference"], branches[condition], strict=True)
            ):
                key = f"{identity}-case-{index}-{condition}-{repeat}"
                measured = measure_pair(
                    reference,
                    other,
                    checkpoint=checkpoint,
                    actor_sha256=actor_id,
                    verifier=case.reference.study.verifier,
                    initial_paths=case.initial_state_paths,
                    final_paths=case.final_state_paths,
                    case=index,
                    repeat=repeat,
                    condition=condition,
                    identity=key,
                )
                # Only behavior from two completed, valid executions of the same first request
                # is judged. A known candidate failure is already a material difference with no
                # trace to rate; missing final probes leave outcomes unknown, not behavior.
                if measured.behavior_assessable:
                    if not allow_dispatch and not project.artifacts.exists(key + "-intent"):
                        raise ValueError("equivalence report lost a planned judge request")
                    report(
                        progress,
                        "Actor equivalence judgment",
                        detail=f"case {index}, {condition}, repeat {repeat}",
                    )
                    judgment = evaluate_actor_equivalence(
                        measured.comparison,
                        rubric=rubric,
                        model_identity=sha256_json(spec.judge),
                        complete=lambda body, logical: review(spec.judge, body, logical, "unknown"),
                    )
                    measured = measured.model_copy(update={"judgment": judgment})
                measurements.append(measured)
                prefix = f"pair:{index}:{condition}:{repeat}"
                paired[prefix] = {
                    "reference_execution": f"pair:{index}:execution:reference:{repeat}",
                    "other_execution": f"pair:{index}:execution:{condition}:{repeat}",
                    "behavior_assessable": measured.behavior_assessable,
                    "material_agreement": measured.material_agreement,
                    "differences": list(measured.differences),
                    "reference_checks": measured.comparison.reference_assessment.model_dump(
                        mode="json"
                    ),
                    "other_checks": measured.comparison.candidate_assessment.model_dump(
                        mode="json"
                    ),
                    "semantic_judgment": None
                    if measured.judgment is None
                    else measured.judgment.model_dump(mode="json"),
                }
                for side, branch in (("reference", reference), ("candidate", other)):
                    actual = "reference" if side == "reference" else condition
                    execution = f"pair:{index}:execution:{actual}:{repeat}"
                    if execution in paired:
                        continue
                    study_reference = selected[actual].result
                    paired[execution] = {
                        "kind": "execution",
                        "condition": actual,
                        "case": index,
                        "repeat": repeat,
                        "branch": branch.index,
                        "study": None
                        if study_reference is None
                        else study_reference.model_dump(mode="json"),
                    }
                    trace = getattr(measured.comparison, side)
                    paired[f"{execution}:execution_receipts"] = trace.audit_metadata
                    paired[f"{execution}:final"] = {
                        "answer": trace.final_answer,
                        "stop_reason": trace.stop_reason,
                    }
                    for step in trace.steps:
                        paired[f"{execution}:step:{step.step}"] = step.model_dump(mode="json")
                    calls = branch.outcome.get("model_calls", [])
                    if isinstance(calls, list):
                        for turn, call in enumerate(calls):
                            if not isinstance(call, dict) or not isinstance(
                                call.get("request"), dict
                            ):
                                continue
                            messages = call["request"].get("messages", [])
                            if isinstance(messages, list):
                                for position, message in enumerate(messages):
                                    paired[
                                        f"{execution}:context:turn-{turn}:messages[{position}]"
                                    ] = message
    # Mechanical failure routing uses the existing owner; all findings remain diagnostic.
    failures = {
        f"pair:{cell.case}:failure:{i}": {
            "condition": cell.condition,
            "study": cell.result.model_dump(mode="json"),
            "analysis": analyse_failures(project.artifacts, cell.result).model_dump(mode="json"),
        }
        for i, cell in enumerate(cells)
        if cell.result is not None
    }
    paired.update(failures)
    analyses, errors = {}, {}
    for case_index, case in enumerate(spec.cases):
        definition: JsonObject = {
            "paired_evidence": {
                key: value for key, value in paired.items() if key.startswith(f"pair:{case_index}:")
            },
            "harness": {
                "kind": "checkpointed ChatAgentRuntime",
                "case": case.model_dump(mode="json"),
            },
            "verifier": {"scope": "declared predicates; no independent calibration claim"},
            "simulator": {"scope": "selected candidate runtime; independently executed reference"},
        }
        for index, analyst in enumerate(spec.analysts):
            key = f"{identity}-case-{case_index}-analyst-{index}"
            label = f"case-{case_index}:{analyst.alias}"
            if not allow_dispatch and not project.artifacts.exists(key + "-intent"):
                raise ValueError("equivalence report lost a planned analyst request")
            report(progress, "Paired trace council", detail=label)
            try:
                analyses[label] = analyze_traces(
                    definition,
                    (),
                    logical_id=key,
                    evidence_scope="paired_execution",
                    complete=partial(review, analyst, consumer="analyst"),
                )
            except Timeout:
                raise
            except (ValueError, OSError, RuntimeError) as error:
                errors[label] = (
                    type(error).__name__ + ": " + redact_secret_text(str(error))[0][:2000]
                )
    calibration = (
        None
        if spec.policy is None
        else calibrate(
            spec.policy,
            load_admission_policy(project.artifacts, spec.policy),
            measurements,
            scopes,
            state_coverage=rubric.state_coverage,
        )
    )
    return EquivalenceReport(
        spec=spec,
        cells=cells,
        measurements=tuple(measurements),
        analyses=analyses,
        analysis_failures=errors,
        calibration=calibration,
        metrics={
            **equivalence_metrics(tuple(measurements)),
            "rubric": spec.rubric.model_dump(mode="json"),
            "rubric_revision": rubric.definition.revision,
            "state_coverage": rubric.state_coverage,
            "rubric_calibration": rubric.calibration,
            "source_families": len(families),
            "actor_task_cases": len(spec.cases),
            "actor_models": len(
                {sha256_json(row.reference.study.actor_model) for row in spec.cases}
            ),
        },
    )


def _registration(
    project: ProjectStore, spec: EquivalenceSpec, code_revision: str
) -> ArtifactInput:
    """Verify the registration and checkpoint closure before resuming any panel work."""
    store, identity = project.artifacts, spec.identity(code_revision)
    _validate_cases(project, spec)
    registration = store.read(identity).manifest
    if (
        registration.artifact_type != "actor-equivalence-registration"
        or registration.code_revision != code_revision
        or EquivalenceSpec.model_validate_json(store.read_bytes(identity, "spec.json")) != spec
    ):
        raise ValueError("equivalence registration differs")
    expected_roots = _registration_inputs(spec)
    if set(registration.inputs) != expected_roots:
        raise ValueError("equivalence registration lost its paired checkpoints, rubric or policy")
    if spec.policy is not None:
        load_admission_policy(store, spec.policy)
        require_policy_precedes(store, spec.policy, identity, subject="comparison registration")
    return artifact_input(registration)


def _registration_inputs(spec: EquivalenceSpec) -> set[ArtifactInput]:
    """Return the paired checkpoints, the frozen rubric and any pinned admission policy."""
    roots = {
        spec.rubric,
        *(job.study.checkpoint for case in spec.cases for job in (case.reference, case.candidate)),
    }
    if spec.policy is not None:
        roots.add(spec.policy)
    return roots


def saved_equivalence(
    project: ProjectStore, spec: EquivalenceSpec, *, code_revision: str
) -> EquivalenceReport | None:
    """Rebuild the report from original cells and raw reviews without credentials or execution."""
    store, identity = project.artifacts, spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    registration = _registration(project, spec, code_revision)
    sources = {registration}
    cells = []
    for index, planned in enumerate(equivalence_cells(spec, code_revision)):
        manifest = store.read(f"{identity}-cell-{index}").manifest
        cell = EquivalenceCell.model_validate_json(
            store.read_bytes(manifest.artifact_id, "cell.json")
        )
        if cell.model_copy(update={"result": None, "failure": None}) != planned:
            raise ValueError("equivalence cell changed its frozen actor or environment")
        inputs = {
            registration,
            cell.job.study.checkpoint,
            *deferred_execution(store, manifest.artifact_id),
        }
        if cell.result is not None:
            if (
                continuation_result_reference(project, cell.job, code_revision=code_revision)
                != cell.result
            ):
                raise ValueError("equivalence result belongs to a different execution")
            inputs.add(cell.result)
        if (
            manifest.artifact_type != "actor-equivalence-cell"
            or manifest.code_revision != code_revision
            or set(manifest.inputs) != inputs
        ):
            raise ValueError("equivalence cell lost its evidence dependencies")
        sources.add(artifact_input(manifest))
        cells.append(cell)
    rebuilt = _assemble(project, spec, tuple(cells), code_revision, sources, allow_dispatch=False)
    manifest = store.read(identity + "-result").manifest
    if (
        manifest.artifact_type != "actor-equivalence-report"
        or manifest.code_revision != code_revision
        or set(manifest.inputs) != sources
    ):
        raise ValueError("equivalence report lost its complete execution/review inventory")
    saved = EquivalenceReport.model_validate_json(
        store.read_bytes(identity + "-result", "report.json")
    )
    if sha256_json(saved) != sha256_json(rebuilt):
        raise ValueError("equivalence report does not reproduce from saved evidence")
    return rebuilt


def load_equivalence_report(project: ProjectStore, reference: ArtifactInput) -> EquivalenceReport:
    """Reconstruct a saved report from its executions and reviews without fresh inference.

    Args:
        project: Project holding the report.
        reference: Exact report reference ``compare-environments`` printed.

    Returns:
        The report, rebuilt from saved evidence and checked against the saved bytes.

    Raises:
        ValueError: The reference is not a saved report, lost its identity, or no longer
            reproduces from its saved evidence.
    """
    store = project.artifacts
    manifest = store.read(reference.artifact_id).manifest
    if (
        artifact_input(manifest) != reference
        or manifest.artifact_type != "actor-equivalence-report"
    ):
        raise ValueError("comparison reference differs from the saved paired report")
    report = EquivalenceReport.model_validate_json(
        store.read_bytes(reference.artifact_id, "report.json")
    )
    if reference.artifact_id != report.spec.identity(manifest.code_revision) + "-result":
        raise ValueError("comparison report lost its frozen identity")
    rebuilt = saved_equivalence(project, report.spec, code_revision=manifest.code_revision)
    # Compare canonical bytes: a rebuilt judgment holds the judge's own axis subclass.
    if rebuilt is None or sha256_json(rebuilt) != sha256_json(report):
        raise ValueError("comparison report no longer reproduces from saved evidence")
    return report


def run_equivalence(
    project: ProjectStore,
    spec: EquivalenceSpec,
    *,
    code_revision: str,
    progress: ProgressHook | None = None,
) -> EquivalenceReport:
    """Register, execute, compare and analyze under one panel lease; never silently redraw cells."""
    store, identity = project.artifacts, spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_equivalence(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        if not store.exists(identity):
            preflight_equivalence(project, spec)
            _write(
                project,
                identity,
                "actor-equivalence-registration",
                "spec.json",
                spec,
                _registration_inputs(spec),
                code_revision,
            )
        registration = _registration(project, spec, code_revision)
        cells, sources = [], {registration}
        planned = equivalence_cells(spec, code_revision)
        for index, expected in enumerate(planned):
            key = f"{identity}-cell-{index}"
            report(
                progress,
                "Paired actor execution",
                completed=index,
                total=len(planned),
                detail=f"case {expected.case}: {expected.condition}",
            )
            inputs = {registration, expected.job.study.checkpoint}
            cell = expected
            if store.exists(key):
                cell = EquivalenceCell.model_validate_json(store.read_bytes(key, "cell.json"))
                if cell.model_copy(update={"result": None, "failure": None}) != expected:
                    raise ValueError("saved equivalence cell differs from its plan")
            else:
                if (
                    saved_continuation_job(project, cell.job, code_revision=code_revision)
                    is not None
                ):
                    cell = cell.model_copy(
                        update={
                            "result": continuation_result_reference(
                                project, cell.job, code_revision=code_revision
                            )
                        }
                    )
                elif store.exists(key + "-intent") and not deferred_execution(store, key):
                    intent = store.read(key + "-intent").manifest
                    if (
                        intent.artifact_type != "actor-equivalence-cell-intent"
                        or intent.code_revision != code_revision
                        or set(intent.inputs) != inputs
                        or EquivalenceCell.model_validate_json(
                            store.read_bytes(key + "-intent", "cell.json")
                        )
                        != expected
                    ):
                        raise ValueError("unfinished equivalence intent differs from its plan")
                    cell = cell.model_copy(update={"failure": "UnfinishedDispatchIntent"})
                else:
                    if not store.exists(key + "-intent"):
                        _write(
                            project,
                            key + "-intent",
                            "actor-equivalence-cell-intent",
                            "cell.json",
                            cell,
                            inputs,
                            code_revision,
                        )
                    try:
                        execute_continuation_job(project, cell.job, code_revision=code_revision)
                        cell = cell.model_copy(
                            update={
                                "result": continuation_result_reference(
                                    project, cell.job, code_revision=code_revision
                                )
                            }
                        )
                    except Timeout:
                        record_execution_deferral(store, key)
                        raise
                    except (ValueError, OSError, RuntimeError) as error:
                        cell = cell.model_copy(
                            update={
                                "failure": type(error).__name__
                                + ": "
                                + redact_secret_text(str(error))[0][:2000]
                            }
                        )
                if cell.result is not None:
                    inputs.add(cell.result)
                inputs.update(deferred_execution(store, key))
                _write(
                    project, key, "actor-equivalence-cell", "cell.json", cell, inputs, code_revision
                )
            if cell.result is not None:
                if (
                    continuation_result_reference(project, cell.job, code_revision=code_revision)
                    != cell.result
                ):
                    raise ValueError("saved equivalence cell has a different runtime result")
                inputs.add(cell.result)
            inputs.update(deferred_execution(store, key))
            manifest = store.read(key).manifest
            if (
                manifest.artifact_type != "actor-equivalence-cell"
                or manifest.code_revision != code_revision
                or set(manifest.inputs) != inputs
            ):
                raise ValueError("saved equivalence cell lost its evidence dependencies")
            sources.add(artifact_input(manifest))
            cells.append(cell)
        result = _assemble(
            project,
            spec,
            tuple(cells),
            code_revision,
            sources,
            allow_dispatch=True,
            progress=progress,
        )
        _write(
            project,
            identity + "-result",
            "actor-equivalence-report",
            "report.json",
            result,
            sources,
            code_revision,
        )
        return saved_equivalence(project, spec, code_revision=code_revision) or result

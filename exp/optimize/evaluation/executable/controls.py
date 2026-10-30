"""Independent specifications and witnessed controls for executable qualification.

Qualification admits a control only through a saved label witness: an independent
specification graded on the control's outcome. This module registers such specifications and
records one witness per declared control, then re-verifies every witness with the same
admission check qualification runs, so a control that would be refused there (a specification
that is the task's own verifier or derives from generated evidence, or a witness that
establishes another verdict) is refused here with the same reason.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.executable.qualification import (
    ExecutableQualificationSpec,
    preflight_executable_qualification,
)
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.simulation.engines.text.generic_delta import DeltaError, apply_delta
from exp.simulation.evaluation.admission import (
    SPECIFICATION_TYPE,
    AdmissionRefusal,
    EvaluationSpecification,
    EvidenceAdmissionError,
    persist_specification,
    record_label_witness,
    verify_label_witness,
)
from exp.simulation.evaluation.calibration import CalibrationCase
from exp.simulation.evaluation.tasks import Verdict, VerifierChallenge


class SpecificationRegistration(ContractModel):
    """An independent acceptance specification and the evidence it was derived from.

    Attributes:
        specification: Data-only predicate and the subject it grades.
        sources: Saved artifacts the specification derives from. Witnessed labels stay
            independent only when every artifact in this closure is recorded or human-authored
            evidence, such as the build's trace dataset; generated sources make them generated.
    """

    specification: EvaluationSpecification
    sources: tuple[ArtifactInput, ...] = Field(default=(), max_length=64)


def register_specification(
    project: ProjectStore,
    registration: SpecificationRegistration,
    *,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Save a specification, idempotently for identical content and sources.

    Args:
        project: Project that owns the specification and its sources.
        registration: Predicate, subject and sources.
        created_at: Publication time.
        code_revision: Producing source identity.

    Returns:
        Exact ``evaluation-specification`` reference for jobs, witnesses and qualification.

    Raises:
        ValueError: A source differs from its saved digest.
    """
    return persist_specification(
        project.artifacts,
        registration.specification,
        sources=registration.sources,
        created_at=created_at,
        code_revision=code_revision,
    )


def preflight_specification(project: ProjectStore, registration: SpecificationRegistration) -> None:
    """Check every source against its saved digest without publishing.

    Args:
        project: Project that owns the sources.
        registration: Specification to check.

    Raises:
        ValueError: A source differs from its saved digest.
    """
    for source in registration.sources:
        if artifact_input(project.artifacts.read(source.artifact_id).manifest) != source:
            raise ValueError(f"specification source {source.artifact_id} differs from its digest")


class DeclaredControl(ContractModel):
    """One authored control outcome, labeled only by its independent witness.

    Attributes:
        id: Control identity, unique within the qualification.
        lineage: Exposure group; related controls share it and stay in one partition.
        partition: ``development`` or ``confirmation``; ``selection`` adds a third.
        expected: Verdict the specification must establish on the outcome.
        branch: Feasibility-study branch whose saved outcome the control starts from.
        outcome: Literal outcome, instead of a branch.
        edits: Ordered generic delta operations applied to that outcome (``set``,
            ``delete``, ``append``, ``remove_value`` or ``copy`` over list paths).
        rationale: Why the expected verdict follows from the specification.
    """

    id: str = Field(min_length=1, max_length=256)
    lineage: str = Field(min_length=1, max_length=256)
    partition: Literal["development", "selection", "confirmation"]
    expected: Verdict
    branch: int | None = Field(default=None, ge=0, strict=True)
    outcome: JsonObject | None = None
    edits: tuple[JsonValue, ...] = Field(default=(), max_length=64)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _one_base(self) -> Self:
        """A control starts from exactly one saved branch or one literal outcome."""
        if (self.branch is None) == (self.outcome is None):
            raise ValueError(f"control {self.id} needs exactly one of branch or outcome")
        return self


class ControlWitnessSpec(ContractModel):
    """Declared controls for one registered executable task and its feasible study.

    Attributes:
        task: Exact registered executable task under qualification.
        feasibility: Saved continuation study that executed the task.
        specification: Independent outcome specification; defaults to the task job's
            training specification.
        controls: Every control the qualification declares.
    """

    task: ArtifactInput
    feasibility: ArtifactInput
    specification: ArtifactInput | None = None
    controls: tuple[DeclaredControl, ...] = Field(min_length=1, max_length=64)


def preflight_controls(
    project: ProjectStore, spec: ControlWitnessSpec
) -> ExecutableQualificationSpec:
    """Resolve every control outcome and check the would-be qualification without publishing.

    Args:
        project: Project that owns the task, study and specification.
        spec: Declared controls.

    Returns:
        The qualification spec with unwitnessed cases, for checks only.

    Raises:
        ValueError: An input is missing or changed, an edit does not apply, the specification
            does not grade outcomes, or the cases break partition rules.
    """
    _specification(project, spec)
    qualification = ExecutableQualificationSpec(
        task=spec.task,
        feasibility=spec.feasibility,
        cases=tuple(_case(control, outcome, None) for control, outcome in _outcomes(project, spec)),
    )
    preflight_executable_qualification(project, qualification)
    return qualification


def witness_controls(
    project: ProjectStore, spec: ControlWitnessSpec, *, created_at: datetime, code_revision: str
) -> ExecutableQualificationSpec:
    """Record and re-verify one witness per control, returning the qualification spec.

    Args:
        project: Project that owns the task, study and specification.
        spec: Declared controls.
        created_at: Publication time of new witnesses.
        code_revision: Producing source identity.

    Returns:
        A qualification spec whose every case carries a re-verified witness.

    Raises:
        EvidenceAdmissionError: A witness would be refused by qualification; every refusal
            is named.
        ValueError: An input fails a preflight check.
    """
    preflight_controls(project, spec)
    specification = _specification(project, spec)
    predicate = load_executable_task(project, spec.task).job.study.verifier
    cases: list[CalibrationCase] = []
    refusals: list[AdmissionRefusal] = []
    for control, outcome in _outcomes(project, spec):
        witness = record_label_witness(
            project.artifacts,
            specification=specification,
            outcome=outcome,
            created_at=created_at,
            code_revision=code_revision,
        )
        case = _case(control, outcome, witness)
        refusal = verify_label_witness(
            project.artifacts, case.challenge, label=control.expected, compared=(predicate,)
        )
        if refusal is not None:
            refusals.append(refusal)
        cases.append(case)
    if refusals:
        raise EvidenceAdmissionError(refusals)
    qualification = ExecutableQualificationSpec(
        task=spec.task, feasibility=spec.feasibility, cases=tuple(cases)
    )
    preflight_executable_qualification(project, qualification)
    return qualification


def _specification(project: ProjectStore, spec: ControlWitnessSpec) -> ArtifactInput:
    """The digest-checked outcome specification the controls are witnessed against."""
    reference = spec.specification
    if reference is None:
        reference = load_executable_task(project, spec.task).job.study.training_specification
    if reference is None:
        raise ValueError(
            "the task's job names no training specification; register one with "
            "register-specification and pass it as specification"
        )
    stored = project.artifacts.read(reference.artifact_id)
    if (
        artifact_input(stored.manifest) != reference
        or stored.manifest.artifact_type != SPECIFICATION_TYPE
    ):
        raise ValueError(f"{reference.artifact_id} is not this saved evaluation specification")
    saved = EvaluationSpecification.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "specification.json")
    )
    if saved.subject != "outcome":
        raise ValueError(
            "control witnesses grade outcomes; native-observation specifications are "
            "witnessed by witness-record"
        )
    return reference


def _outcomes(
    project: ProjectStore, spec: ControlWitnessSpec
) -> Sequence[tuple[DeclaredControl, JsonObject]]:
    """Each control with its outcome: a branch's saved outcome or the literal, then edits."""
    branches = load_study(project.artifacts, spec.feasibility).branches
    resolved = []
    for control in spec.controls:
        if control.outcome is not None:
            base = control.outcome
        elif control.branch is not None and control.branch < len(branches):
            base = branches[control.branch].grading_outcome()
        else:
            raise ValueError(
                f"control {control.id} names branch {control.branch}, but the feasibility "
                f"study has {len(branches)} branches"
            )
        try:
            resolved.append((control, apply_delta(base, list(control.edits))))
        except DeltaError as error:
            raise ValueError(f"control {control.id} edit does not apply: {error}") from error
    return resolved


def _case(
    control: DeclaredControl, outcome: JsonObject, witness: ArtifactInput | None
) -> CalibrationCase:
    """One qualification case: the control's outcome, label and witness."""
    return CalibrationCase(
        challenge=VerifierChallenge(
            id=control.id,
            outcome=outcome,
            expected=control.expected,
            authority="synthetic_specification",
            witness=witness,
            rationale=control.rationale,
        ),
        lineage=control.lineage,
        partition=control.partition,
    )

"""Independent task labels and scoped verifier calibration without actor feedback."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.reconstruction import AxisOutcome, JudgeConfig, WorkflowEvidence
from exp.simulation.evaluation_tasks import TaskVerifier, Verdict, VerifierChallenge, grade


class CalibrationCase(ContractModel):
    """One independently labeled observation with an explicit exposure group.

    Attributes:
        challenge: Outcome, expected verdict and label authority.
        lineage: Source and defect family shared by all related variants.
        partition: Authoring development, independent selection, or final confirmation.
        critical: Whether a mistaken positive blocks qualification.
        expert_scores: Independently annotated rubric anchors, not model self-labels.
        expected_unassessable: Criteria that the supplied task/evidence cannot support scoring.
        expected_task: Optional task-correctness label, distinct from component validity.
        task_label_scope: Whether the label establishes task correctness or only source reward.
    """

    challenge: VerifierChallenge
    lineage: str = Field(min_length=1)
    partition: Literal["development", "selection", "confirmation"]
    critical: bool = Field(default=True, strict=True)
    expected_task: Verdict | None = None
    task_label_scope: Literal["task_correctness", "source_outcome"] = "task_correctness"
    expert_scores: dict[str, Annotated[int, Field(strict=True, ge=1, le=5)]] = Field(
        default_factory=dict
    )

    expected_unassessable: tuple[str, ...] = ()

    @model_validator(mode="after")
    def independent_scores(self) -> Self:
        """Keep expert rubric labels separate from synthetic outcome specifications."""
        if self.expert_scores and (
            not self.challenge.approved or self.challenge.authority != "human"
        ):
            raise ValueError("expert rubric labels need independently approved human witnesses")
        if len(set(self.expected_unassessable)) != len(self.expected_unassessable) or set(
            self.expected_unassessable
        ) & set(self.expert_scores):
            raise ValueError("a criterion needs exactly one assessment expectation")
        if self.expected_unassessable and not self.challenge.approved:
            raise ValueError("assessment expectations need an independently admitted witness")
        return self


class CalibrationPack(ContractModel):
    """Frozen input to one calibration, never a self-modifying rubric search.

    Attributes:
        task: Exact candidate task whose public contract supplies label meaning.
        parent: Existing mechanical predicate for paired diagnosis.
        candidate: Frozen proposed mechanical predicate.
        cases: Source-linked labels with grouped partitions.
        judge: Optional exact judge configuration to assess against expert anchors.
        score_tolerance: Registered per-axis error allowance before assessment.
        required_dimensions: Criteria required for a qualified task decision.
        minimum_scores: Predeclared pass threshold for each required criterion.
        frozen_profile_sha256: Exact maintained serving and memory profile being qualified.
        judge_completion_sha256: Exact judge completion settings required for qualification.
    """

    task: ArtifactInput
    parent: TaskVerifier
    candidate: TaskVerifier
    cases: tuple[CalibrationCase, ...] = Field(min_length=1)
    judge: JudgeConfig | None = None
    score_tolerance: int = Field(default=0, ge=0, le=4, strict=True)
    required_dimensions: tuple[str, ...] = ()
    minimum_scores: dict[str, Annotated[int, Field(strict=True, ge=1, le=5)]] = Field(
        default_factory=dict
    )
    frozen_profile_sha256: Sha256 | None = None
    judge_completion_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def partitions(self) -> Self:
        """Reject duplicate observations and related cases crossing the reserved boundary."""
        dimensions = (
            set()
            if self.judge is None
            else {dimension.dimension_id for dimension in self.judge.definition.dimensions}
        )
        if (
            len(set(self.required_dimensions)) != len(self.required_dimensions)
            or set(self.required_dimensions) - dimensions
        ):
            raise ValueError("required criteria must be unique members of the frozen rubric")
        if self.minimum_scores and set(self.minimum_scores) != set(self.required_dimensions):
            raise ValueError("task thresholds must name exactly the required criteria")
        if any(case.expected_task is not None for case in self.cases) and (
            self.judge is None or not self.minimum_scores
        ):
            raise ValueError("task labels require a frozen judge and explicit criterion thresholds")
        ids: set[str] = set()
        outcomes: dict[str, str] = {}
        groups: dict[str, str] = {}
        for case in self.cases:
            identity = case.challenge.id
            if identity in ids:
                raise ValueError("calibration case IDs must be unique")
            ids.add(identity)
            outcome_hash = sha256_json(case.challenge.outcome)
            if outcome_hash in outcomes and outcomes[outcome_hash] != case.partition:
                raise ValueError("the same outcome crosses calibration partitions")
            outcomes[outcome_hash] = case.partition
            if case.lineage in groups and groups[case.lineage] != case.partition:
                raise ValueError("related source or defect variants cross calibration partitions")
            groups[case.lineage] = case.partition
            if (case.expert_scores or case.expected_unassessable) and self.judge is None:
                raise ValueError("rubric labels need an exact judge configuration")
            if self.judge is not None and (
                set(case.expert_scores) | set(case.expected_unassessable)
            ) - {dimension.dimension_id for dimension in self.judge.definition.dimensions}:
                raise ValueError("expert labels name an unregistered rubric criterion")
        return self


class CalibrationJudgment(ContractModel):
    """One saved judge result bound to its exact blinded case and configuration.

    Attributes:
        case_id: Private case identity assigned by the calibration owner.
        outcome_sha256: Exact outcome shown to the judge.
        judge_sha256: Exact rubric, prompt and model identity.
        judge_completion_sha256: Completion-owner identity retained by the evaluator.
        report: Immutable source of these judgments, verified by the application owner.
        axes: Assessed or explicitly unavailable criteria.
    """

    case_id: str
    outcome_sha256: Sha256
    judge_sha256: Sha256
    judge_completion_sha256: Sha256 | None = None
    report: ArtifactInput
    axes: tuple[AxisOutcome, ...]


def blinded_case(case: CalibrationCase) -> JsonObject:
    """Return judge-visible observations without origin, expected scores or labels."""
    outcome = case.challenge.outcome
    return {
        "case_id": stable_id("calibration-case", {"outcome_sha256": sha256_json(outcome)}),
        "outcome": outcome.model_dump(mode="json")
        if isinstance(outcome, ContractModel)
        else outcome,
    }


def calibrate(
    pack: CalibrationPack,
    judgments: tuple[CalibrationJudgment, ...] = (),
    *,
    allow_confirmation: bool = True,
) -> JsonObject:
    """Compare frozen predicates and judge outputs against admitted independent labels.

    Args:
        pack: Immutable cases, candidate and predeclared comparison allowances.
        judgments: Saved evidence, never generated labels or provider callbacks.
        allow_confirmation: Open reserved cases only after the caller freezes selection.

    Returns:
        Counts, per-case errors and scoped statuses. No reward is supplied to an actor.
    """
    rows: list[JsonObject] = []
    index = {judgment.case_id: judgment for judgment in judgments}
    if len(index) != len(judgments) or set(index) - {c.challenge.id for c in pack.cases}:
        raise ValueError("judge results contain duplicate or unknown case IDs")
    development_ready = _partition_ready(pack, "development", index)
    has_selection = any(case.partition == "selection" for case in pack.cases)
    selection_ready = _partition_ready(pack, "selection", index) if has_selection else True
    has_confirmation = any(case.partition == "confirmation" for case in pack.cases)
    confirmation_open = allow_confirmation and development_ready and selection_ready
    confirmation_ready = confirmation_open and _partition_ready(pack, "confirmation", index)
    for case in pack.cases:
        challenge = case.challenge
        opened = case.partition != "confirmation" or confirmation_open
        parent = grade(pack.parent, challenge.outcome) if opened else None
        candidate = grade(pack.candidate, challenge.outcome) if opened else None
        admitted = challenge.approved and challenge.authority != "generated_hypothesis"
        judgment = index.get(challenge.id)
        task_status = (
            task_decision(candidate.status, _bound_axes(pack, case, judgment), pack.minimum_scores)
            if candidate is not None and case.expected_task is not None
            else None
        )
        rows.append(
            {
                "case_id": challenge.id,
                "lineage": case.lineage,
                "partition": case.partition,
                "independent": admitted,
                "critical": case.critical,
                "expected": challenge.expected if opened else None,
                "expected_task": case.expected_task if opened else None,
                "task_label_scope": case.task_label_scope,
                "task_decision": task_status,
                "task_correct": None
                if task_status is None or not admitted
                else task_status == case.expected_task,
                "task_false_acceptance": admitted
                and task_status == "pass"
                and case.expected_task != "pass",
                "task_false_rejection": admitted
                and task_status == "fail"
                and case.expected_task == "pass",
                "parent": None if parent is None else parent.model_dump(mode="json"),
                "candidate": None if candidate is None else candidate.model_dump(mode="json"),
                "correct": None
                if candidate is None or not admitted
                else candidate.status == challenge.expected,
                "false_acceptance": candidate is not None
                and admitted
                and candidate.status == "pass"
                and challenge.expected != "pass",
                "false_rejection": candidate is not None
                and admitted
                and candidate.status == "fail"
                and challenge.expected == "pass",
                "judge": _judge_comparison(pack, case, judgment)
                if opened
                else {"status": "not_run"},
            }
        )
    expert_cases = [
        row
        for row, case in zip(rows, pack.cases, strict=True)
        if case.expert_scores or case.expected_unassessable
    ]
    judge_ready = (
        bool(expert_cases)
        and pack.judge is not None
        and all(
            isinstance(row["judge"], dict) and row["judge"].get("status") == "agrees"
            for row in expert_cases
        )
    )
    all_required_judged = bool(pack.required_dimensions) and all(
        set(pack.required_dimensions) <= set(case.expert_scores) for case in pack.cases
    )
    task_cases = [
        row for row, case in zip(rows, pack.cases, strict=True) if case.expected_task is not None
    ]
    task_agreement = bool(task_cases) and all(row["task_correct"] is True for row in task_cases)
    task_coverage = (
        bool(pack.minimum_scores)
        and len(task_cases) == len(pack.cases)
        and all(
            _required_assessed(
                _bound_axes(pack, case, index.get(case.challenge.id)), pack.minimum_scores
            )
            for case in pack.cases
        )
    )
    profile_bound = pack.frozen_profile_sha256 is not None and all(
        isinstance(case.challenge.outcome, WorkflowEvidence)
        and case.challenge.outcome.sources.get("frozen_profile_sha256")
        == pack.frozen_profile_sha256
        for case in pack.cases
    )
    return {
        "pack_sha256": sha256_json(pack),
        "task": pack.task.model_dump(mode="json"),
        "parent_sha256": sha256_json(pack.parent),
        "candidate_sha256": sha256_json(pack.candidate),
        "judge_sha256": None if pack.judge is None else sha256_json(pack.judge),
        "required_dimensions": list(pack.required_dimensions),
        "minimum_scores": pack.minimum_scores,
        "frozen_profile_sha256": pack.frozen_profile_sha256,
        "judge_completion_sha256": pack.judge_completion_sha256,
        "planned": len(rows),
        "lineage_groups": len({case.lineage for case in pack.cases}),
        "label_authorities": {
            authority: sum(case.challenge.authority == authority for case in pack.cases)
            for authority in sorted({case.challenge.authority for case in pack.cases})
        },
        "independent": sum(bool(row["independent"]) for row in rows),
        "false_acceptances": sum(bool(row["false_acceptance"]) for row in rows),
        "false_rejections": sum(bool(row["false_rejection"]) for row in rows),
        "judge_criterion_disagreements": sum(
            isinstance(row["judge"], dict) and row["judge"].get("status") == "disagrees"
            for row in rows
        ),
        "task_planned": len(task_cases),
        "task_assessed": sum(row["task_decision"] in ("pass", "fail") for row in task_cases),
        "task_unassessable": sum(row["task_decision"] == "unassessable" for row in task_cases),
        "task_withheld": sum(row["task_decision"] is None for row in task_cases),
        "task_false_acceptances": sum(bool(row["task_false_acceptance"]) for row in rows),
        "task_false_rejections": sum(bool(row["task_false_rejection"]) for row in rows),
        "development_status": "passed" if development_ready else "incomplete_or_incorrect",
        "selection_status": "not_run"
        if not has_selection
        else "passed"
        if selection_ready
        else "incomplete_or_incorrect",
        "confirmation_status": "not_run"
        if not confirmation_open or not has_confirmation
        else "passed"
        if confirmation_ready
        else "incomplete_or_incorrect",
        "rubric_status": "supported_on_labeled_cases"
        if judge_ready
        else "task_decision_agreement"
        if task_agreement
        else "provisional",
        "qualification": "qualified_in_scope"
        if confirmation_ready
        and pack.judge_completion_sha256 is not None
        and profile_bound
        and bool(pack.minimum_scores)
        and ((judge_ready and all_required_judged) or (task_agreement and task_coverage))
        and all(case.challenge.authority != "synthetic_specification" for case in pack.cases)
        and all(case.task_label_scope == "task_correctness" for case in pack.cases)
        else "diagnostic_only",
        "rows": rows,
        "native_environment_accessed": False,
        "actor_feedback": False,
        "physical_attempts": 0,
        "limitations": ["Finite cases do not establish universal fidelity or training effects."],
    }


def _partition_ready(
    pack: CalibrationPack, partition: str, judgments: dict[str, CalibrationJudgment]
) -> bool:
    """Require independently admitted positive and negative cases with correct decisions."""
    cases = [case for case in pack.cases if case.partition == partition]
    admitted = [case for case in cases if case.challenge.approved]
    return (
        bool(cases)
        and len(admitted) == len(cases)
        and {"pass", "fail"} <= {case.expected_task or case.challenge.expected for case in admitted}
        and all(
            grade(pack.candidate, case.challenge.outcome).status == case.challenge.expected
            and (
                not (case.expert_scores or case.expected_unassessable)
                or _judge_comparison(pack, case, judgments.get(case.challenge.id))["status"]
                == "agrees"
            )
            and (
                case.expected_task is None
                or task_decision(
                    grade(pack.candidate, case.challenge.outcome).status,
                    _bound_axes(pack, case, judgments.get(case.challenge.id)),
                    pack.minimum_scores,
                )
                == case.expected_task
            )
            for case in admitted
        )
    )


def _judge_comparison(
    pack: CalibrationPack, case: CalibrationCase, judgment: CalibrationJudgment | None
) -> JsonObject:
    """Compare exact saved judgments with admitted numeric and assessment-status labels."""
    if judgment is None:
        return {
            "status": "not_run"
            if case.expert_scores or case.expected_unassessable or case.expected_task is not None
            else "no_expert_labels"
        }
    axes = {axis.dimension_id: axis for axis in _bound_axes(pack, case, judgment)}
    if not case.expert_scores and not case.expected_unassessable:
        return {"status": "task_decision_only", "report": judgment.report.model_dump(mode="json")}
    if any(
        axis.status != "assessed" or axis.score is None
        for axis in (axes[key] for key in case.expert_scores)
    ):
        return {"status": "unavailable", "report": judgment.report.model_dump(mode="json")}
    errors = {
        key: abs(case.expert_scores[key] - int(axes[key].score or 0)) for key in case.expert_scores
    }
    if any(
        axes[key].status not in ("assessed", "unassessable") for key in case.expected_unassessable
    ):
        return {"status": "unavailable", "report": judgment.report.model_dump(mode="json")}
    # Status errors are categorical and exceed every supported numeric tolerance.
    assessment_errors = {
        key: 0 if axes[key].status == "unassessable" else 5 for key in case.expected_unassessable
    }
    return {
        "status": "agrees"
        if all(error <= pack.score_tolerance for error in errors.values())
        and not any(assessment_errors.values())
        else "disagrees",
        "absolute_errors": errors,
        "assessment_errors": assessment_errors,
        "report": judgment.report.model_dump(mode="json"),
    }


def _bound_axes(
    pack: CalibrationPack, case: CalibrationCase, judgment: CalibrationJudgment | None
) -> tuple[AxisOutcome, ...]:
    """Resolve only results bound to this exact input and registered criterion inventory."""
    if judgment is None:
        return ()
    if judgment.outcome_sha256 != sha256_json(
        case.challenge.outcome
    ) or judgment.judge_sha256 != sha256_json(pack.judge):
        raise ValueError("judge result differs from the frozen case or judge binding")
    if pack.judge_completion_sha256 is not None and (
        judgment.judge_completion_sha256 != pack.judge_completion_sha256
    ):
        raise ValueError("judge result differs from the frozen completion profile")
    axes = {axis.dimension_id: axis for axis in judgment.axes}
    expected_axes = (
        set()
        if pack.judge is None
        else {dimension.dimension_id for dimension in pack.judge.definition.dimensions}
    )
    if len(axes) != len(judgment.axes) or set(axes) != expected_axes:
        raise ValueError("judge criteria differ from the independently labeled inventory")
    return judgment.axes


def _required_assessed(axes: tuple[AxisOutcome, ...], minimum_scores: dict[str, int]) -> bool:
    """Require actual assessments rather than treating unavailable criteria as low scores."""
    index = {axis.dimension_id: axis for axis in axes}
    return bool(minimum_scores) and all(
        key in index and index[key].status == "assessed" and index[key].score is not None
        for key in minimum_scores
    )


def task_decision(
    mechanical: Verdict, axes: tuple[AxisOutcome, ...], minimum_scores: dict[str, int]
) -> Verdict:
    """Gate semantic thresholds on correctness; efficiency cannot compensate for a failure."""
    if mechanical == "fail":
        return "fail"
    if mechanical != "pass" or not _required_assessed(axes, minimum_scores):
        return "unassessable"
    if any(axis.critical_flags for axis in axes):
        return "fail"
    index = {axis.dimension_id: axis for axis in axes}
    return (
        "pass"
        if all((index[key].score or 0) >= value for key, value in minimum_scores.items())
        else "fail"
    )

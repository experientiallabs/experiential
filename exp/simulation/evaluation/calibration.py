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
from exp.common.evaluations.uncertainty import proportion_summary
from exp.simulation.evaluation.admission import (
    MINIMUM_ADMITTED_PER_VERDICT,
    AdmissionRefusal,
    EvidenceAdmission,
    adequacy_refusals,
)
from exp.simulation.evaluation.tasks import (
    TaskVerifier,
    Verdict,
    VerifierChallenge,
    grade,
    label_admitted,
)


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
        category: Defect family this case represents or controls for, such as ``wrong_cell``
            or ``unauthorized_write``; judge errors are reported per category. None reports
            under ``uncategorized``.
        sample: How the case entered the pack. ``representative`` cases estimate deployment
            prevalence; ``failure_enriched`` cases are chosen to contain defects and only
            estimate class-conditional rates. The two are never pooled; None reports under
            ``unspecified``.
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
    category: str | None = Field(
        default=None, min_length=1, max_length=128, exclude_if=lambda value: value is None
    )
    sample: Literal["representative", "failure_enriched"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def independent_scores(self) -> Self:
        """Keep expert rubric labels separate from synthetic outcome specifications.

        Ordinal expert scores are the one deliberate human-only label: a native or
        specification witness establishes a verdict, not a 1 to 5 rubric anchor. Witnessed
        verdicts qualify through the task-correctness path instead.
        """
        if self.expert_scores and (
            not self.challenge.approved or self.challenge.authority != "human"
        ):
            raise ValueError("expert rubric labels need independently approved human witnesses")
        if len(set(self.expected_unassessable)) != len(self.expected_unassessable) or set(
            self.expected_unassessable
        ) & set(self.expert_scores):
            raise ValueError("a criterion needs exactly one assessment expectation")
        if self.expected_unassessable and not (
            self.challenge.approved or self.challenge.witness is not None
        ):
            raise ValueError("assessment expectations need an independently admitted witness")
        if (
            self.challenge.witness is not None
            and self.expected_task is not None
            and self.expected_task != self.challenge.expected
        ):
            raise ValueError(
                "a witness establishes one verdict: a witnessed case's task label must equal "
                "its verifier label"
            )
        return self

    @property
    def label(self) -> Verdict:
        """The primary label: task correctness when declared, else the verifier verdict."""
        return self.expected_task or self.challenge.expected

    def covered_axes(self, required: tuple[str, ...]) -> frozenset[str]:
        """Rubric criteria this case's labels speak to."""
        if self.expected_task is not None:
            return frozenset(required)
        return frozenset(self.expert_scores) | frozenset(self.expected_unassessable)


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
    admission: EvidenceAdmission | None = None,
) -> JsonObject:
    """Compare frozen predicates and judge outputs against admitted independent labels.

    Args:
        pack: Immutable cases, candidate and predeclared comparison allowances.
        judgments: Saved evidence, never generated labels or provider callbacks.
        allow_confirmation: Open reserved cases only after the caller freezes selection.
        admission: Store-verified witnesses and contamination refusals. Without it, witnessed
            labels stay unadmitted and the result cannot qualify.

    Returns:
        Counts, per-case errors and scoped statuses. No reward is supplied to an actor.
    """
    rows: list[JsonObject] = []
    index = {judgment.case_id: judgment for judgment in judgments}
    if len(index) != len(judgments) or set(index) - {c.challenge.id for c in pack.cases}:
        raise ValueError("judge results contain duplicate or unknown case IDs")
    verified = (
        frozenset()
        if admission is None
        else admission.verified_ids(case.challenge for case in pack.cases)
    )
    development_ready = _partition_ready(pack, "development", index, verified)
    has_selection = any(case.partition == "selection" for case in pack.cases)
    selection_ready = (
        _partition_ready(pack, "selection", index, verified) if has_selection else True
    )
    has_confirmation = any(case.partition == "confirmation" for case in pack.cases)
    confirmation_open = allow_confirmation and development_ready and selection_ready
    confirmation_ready = confirmation_open and _partition_ready(
        pack, "confirmation", index, verified
    )
    refusals = _qualification_refusals(pack, admission, verified)
    for case in pack.cases:
        challenge = case.challenge
        opened = case.partition != "confirmation" or confirmation_open
        parent = grade(pack.parent, challenge.outcome) if opened else None
        candidate = grade(pack.candidate, challenge.outcome) if opened else None
        admitted = label_admitted(challenge, verified)
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
    judge_decisions_agree = all(
        _threshold_agreement(pack, case, index.get(case.challenge.id)) for case in pack.cases
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
        "witnessed_verified": len(verified),
        "qualification_refusals": [row.model_dump(mode="json") for row in refusals],
        "exposure_ledger": None if admission is None else admission.ledger,
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
        "judge_validation": _judge_validation(pack, index, verified, confirmation_open),
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
        and (
            (judge_ready and all_required_judged and judge_decisions_agree)
            or (task_agreement and task_coverage)
        )
        and not refusals
        and all(case.task_label_scope == "task_correctness" for case in pack.cases)
        else "diagnostic_only",
        "rows": rows,
        "native_environment_accessed": False,
        "actor_feedback": False,
        "physical_attempts": 0,
        "limitations": ["Finite cases do not establish universal fidelity or training effects."],
    }


def _judge_validation(
    pack: CalibrationPack,
    judgments: dict[str, CalibrationJudgment],
    verified: frozenset[str],
    confirmation_open: bool,
) -> JsonObject:
    """Report both error directions, abstention and threshold errors per sample and category.

    Defect is the positive class: a ``fail`` label marks a defect and a ``pass`` label a valid
    outcome. Representative and failure-enriched cases are reported separately and never
    pooled; prevalence-dependent rates are reported only for representative cases.

    Args:
        pack: Frozen cases, predicates, judge and thresholds.
        judgments: Saved judge results by case ID.
        verified: Store-verified witnessed label IDs.
        confirmation_open: Whether reserved confirmation cases may be read.

    Returns:
        The admission-floor interpretation and, per sample set and category, verifier and
        task-decision confusion counts with Wilson intervals at the lineage count, judge
        failures kept apart from abstentions, plus criterion-level abstention,
        unsupported-confidence and threshold-crossing counts.
    """
    grouped: dict[str, dict[str, list[tuple[CalibrationCase, CalibrationJudgment | None]]]] = {}
    withheld = unadmitted = 0
    for case in pack.cases:
        if case.partition == "confirmation" and not confirmation_open:
            withheld += 1
            continue
        if not label_admitted(case.challenge, verified):
            unadmitted += 1
            continue
        sample = case.sample or "unspecified"
        category = case.category or "uncategorized"
        rows = grouped.setdefault(sample, {}).setdefault(category, [])
        rows.append((case, judgments.get(case.challenge.id)))
    samples: JsonObject = {}
    for sample, categories in sorted(grouped.items()):
        everything = [row for rows in categories.values() for row in rows]
        samples[sample] = {
            "overall": _category_validation(pack, everything, sample),
            "categories": {
                name: _category_validation(pack, rows, sample)
                for name, rows in sorted(categories.items())
            },
        }
    return {
        "task": pack.task.artifact_id,
        "convention": (
            "defect is the positive class: a fail label is a defect, a pass label is valid; "
            "a false acceptance passes a defect, a false rejection fails a valid outcome"
        ),
        "admission_floor": {
            "minimum_admitted_per_verdict": MINIMUM_ADMITTED_PER_VERDICT,
            "interpretation": (
                "a structural repeated-contrast floor for admission; it is not a statistical "
                "certification that the judge detects rare or consequential errors"
            ),
        },
        "withheld_confirmation_cases": withheld,
        "unadmitted_cases": unadmitted,
        "samples": samples,
    }


def _category_validation(
    pack: CalibrationPack,
    rows: list[tuple[CalibrationCase, CalibrationJudgment | None]],
    sample: str,
) -> JsonObject:
    """Measure one sample set and category for the verifier, task decision and criteria."""
    verifier = [
        (case.challenge.expected, grade(pack.candidate, case.challenge.outcome).status, case)
        for case, _ in rows
    ]
    task: list[tuple[Verdict, Verdict, CalibrationCase]] = []
    task_not_judged = task_judge_failures = 0
    abstentions = operational = unsupported = assessed = expected_assessed = 0
    correct_abstentions = optimistic = pessimistic = 0
    for case, judgment in rows:
        if case.expected_task is not None:
            if judgment is None:
                task_not_judged += 1
            elif required_judge_failed(judgment.axes, pack.minimum_scores):
                # A rejected or undispatched judge output is an operational failure, not a
                # judge abstention; it stays out of both error directions.
                task_judge_failures += 1
            else:
                mechanical = grade(pack.candidate, case.challenge.outcome).status
                decision = task_decision(
                    mechanical, _bound_axes(pack, case, judgment), pack.minimum_scores
                )
                task.append((case.expected_task, decision, case))
        if judgment is None:
            continue
        axes = {axis.dimension_id: axis for axis in _bound_axes(pack, case, judgment)}
        for key, expected in case.expert_scores.items():
            expected_assessed += 1
            axis = axes[key]
            if axis.status == "unassessable":
                abstentions += 1
            elif axis.status != "assessed" or axis.score is None:
                operational += 1
            else:
                assessed += 1
                threshold = pack.minimum_scores.get(key)
                if threshold is not None and (axis.score >= threshold) != (expected >= threshold):
                    optimistic += axis.score >= threshold
                    pessimistic += axis.score < threshold
        for key in case.expected_unassessable:
            status = axes[key].status
            unsupported += status == "assessed"
            correct_abstentions += status == "unassessable"
            operational += status not in ("assessed", "unassessable")
    return {
        "cases": len(rows),
        "verifier": _confusion(verifier, sample),
        "task_decision": {
            **_confusion(task, sample),
            "not_judged": task_not_judged,
            "judge_failures": task_judge_failures,
        },
        "criteria": {
            "expected_assessable": expected_assessed,
            "assessed": assessed,
            "unnecessary_abstentions": abstentions,
            "confident_without_evidence": unsupported,
            "correct_abstentions": correct_abstentions,
            "operationally_unavailable": operational,
            "threshold_crossings": {
                "optimistic": optimistic,
                "pessimistic": pessimistic,
                "scope": "assessed required criteria whose judge and expert scores fall on "
                "opposite sides of the registered minimum",
            },
        },
    }


def _confusion(rows: list[tuple[Verdict, Verdict, CalibrationCase]], sample: str) -> JsonObject:
    """Count both error directions and abstentions for ``(label, decision, case)`` rows."""

    def count(label: Verdict, decision: Verdict) -> int:
        """Rows with one label and one decision."""
        return sum(row[0] == label and row[1] == decision for row in rows)

    def units(*, label: Verdict | None = None, decision: Verdict | None = None) -> int | None:
        """Distinct lineages among decided pass/fail rows matching a label or decision."""
        found = {
            row[2].lineage
            for row in rows
            if row[0] in ("pass", "fail")
            and row[1] in ("pass", "fail")
            and (label is None or row[0] == label)
            and (decision is None or row[1] == decision)
        }
        return len(found) or None

    flagged, accepted_defects = count("fail", "fail"), count("fail", "pass")
    rejected_valid, accepted_valid = count("pass", "fail"), count("pass", "pass")
    representative = sample == "representative"
    prevalence_note = (
        None
        if representative
        else "prevalence-dependent; not estimable from a failure-enriched or unspecified sample"
    )
    return {
        "labeled": len(rows),
        "flagged_defects": flagged,
        "false_acceptances": accepted_defects,
        "false_rejections": rejected_valid,
        "correct_acceptances": accepted_valid,
        "unnecessary_abstentions": count("fail", "unassessable") + count("pass", "unassessable"),
        "confident_without_evidence": count("unassessable", "pass") + count("unassessable", "fail"),
        "correct_abstentions": count("unassessable", "unassessable"),
        "defect_sensitivity": proportion_summary(
            flagged, flagged + accepted_defects, units=units(label="fail")
        ),
        "specificity": proportion_summary(
            accepted_valid, accepted_valid + rejected_valid, units=units(label="pass")
        ),
        "defect_precision": proportion_summary(
            flagged, flagged + rejected_valid, units=units(decision="fail")
        )
        if representative
        else prevalence_note,
        "accepted_defective_share": proportion_summary(
            accepted_defects, accepted_defects + accepted_valid, units=units(decision="pass")
        )
        if representative
        else prevalence_note,
        "interval_unit": "lineage",
    }


def _qualification_refusals(
    pack: CalibrationPack, admission: EvidenceAdmission | None, verified: frozenset[str]
) -> tuple[AdmissionRefusal, ...]:
    """Every named reason the pack cannot qualify: admission refusals and thin evidence."""
    missing = (
        AdmissionRefusal(
            reason="unverifiable_witness",
            subject="evidence admission",
            detail=(
                "witnesses, exposure and partitions were not checked against the artifact "
                "store; calibrate through exp eval PROJECT --calibrate"
            ),
        ),
    )
    partitions = tuple(
        name
        for name in ("development", "selection", "confirmation")
        if name != "selection" or any(case.partition == name for case in pack.cases)
    )
    return (missing if admission is None else admission.refusals) + adequacy_refusals(
        (
            (
                case.partition,
                case.label,
                label_admitted(case.challenge, verified),
                case.covered_axes(pack.required_dimensions),
            )
            for case in pack.cases
        ),
        partitions=partitions,
        axes=pack.required_dimensions,
    )


def _partition_ready(
    pack: CalibrationPack,
    partition: str,
    judgments: dict[str, CalibrationJudgment],
    verified: frozenset[str],
) -> bool:
    """Require independently admitted positive and negative cases with correct decisions."""
    cases = [case for case in pack.cases if case.partition == partition]
    admitted = [case for case in cases if label_admitted(case.challenge, verified)]
    return (
        bool(cases)
        and len(admitted) == len(cases)
        and {"pass", "fail"} <= {case.label for case in admitted}
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


def _threshold_agreement(
    pack: CalibrationPack, case: CalibrationCase, judgment: CalibrationJudgment | None
) -> bool:
    """Whether the judge reaches the task decision the expert scores imply.

    Scores within tolerance can still fall on opposite sides of a registered minimum, and a
    critical flag fails a task the expert scores pass; either flips the issued decision.
    """
    axes = {axis.dimension_id: axis for axis in _bound_axes(pack, case, judgment)}
    for key, minimum in pack.minimum_scores.items():
        axis, expected = axes.get(key), case.expert_scores.get(key)
        if axis is None or expected is None or axis.status != "assessed" or axis.score is None:
            return False
        if (axis.score >= minimum) != (expected >= minimum):
            return False
    expert_passes = all(
        case.expert_scores[key] >= minimum for key, minimum in pack.minimum_scores.items()
    )
    return not (expert_passes and any(axis.critical_flags for axis in axes.values()))


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


def required_judge_failed(axes: tuple[AxisOutcome, ...], minimum_scores: dict[str, int]) -> bool:
    """Whether the evaluator obtained no judgment of a required criterion.

    Args:
        axes: Criterion outcomes of one judgment.
        minimum_scores: Registered thresholds; their keys are the required criteria.

    Returns:
        True when a required criterion failed, was unknown or was not run. An optional
        criterion's failure cannot block the decision, so it is not an operational failure.
    """
    return any(
        axis.status in ("judge_failure", "judge_unknown", "not_run")
        for axis in axes
        if axis.dimension_id in minimum_scores
    )


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

"""Bounded expert-anchored rubric development with confirmation after selection."""

from collections.abc import Callable

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, sha256_json
from exp.common.evaluations.reconstruction import JudgeConfig
from exp.common.judging.definition import JudgeDefinition
from exp.simulation.evaluation.admission import EvidenceAdmission, require_admitted
from exp.simulation.evaluation.calibration import (
    CalibrationCase,
    CalibrationJudgment,
    CalibrationPack,
    blinded_case,
    calibrate,
)
from exp.simulation.evaluation.tasks import label_admitted


class RubricEdit(ContractModel):
    """An authored semantic revision with no permission to change labels or model.

    Attributes:
        definition: Same five criterion identities with revised anchors or descriptions.
        prompt: Proposed judge instructions and examples.
        evidence_refs: Development case identities supporting the revision.
        reason: Explanation linked to actual grading failures.
    """

    definition: JudgeDefinition
    prompt: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)


def refine_rubric(
    pack: CalibrationPack,
    *,
    propose: Callable[[JsonObject, str], RubricEdit],
    judge: Callable[[JudgeConfig, JsonObject, str], CalibrationJudgment],
    maximum_revisions: int,
    admission: EvidenceAdmission | None = None,
) -> JsonObject:
    """Keep revisions that improve development errors without regressing selection errors.

    Args:
        pack: Frozen expert labels, partitions and initial judge binding.
        propose: Existing metered structured-completion owner, receiving development only.
        judge: Existing evaluator owner, receiving blinded outcomes without expected scores.
        maximum_revisions: Registered finite proposal allowance, separate from provider caps.
        admission: Store-verified evidence admission; any refusal stops before a callback.
            Without it, witnessed labels stay unadmitted.

    Returns:
        Every proposed version, selection evidence and final confirmation assessment.
        Callbacks retain their normal attempt receipts; this function performs no HTTP.
    """
    if type(maximum_revisions) is not int or maximum_revisions < 0:
        raise ValueError("a nonnegative revision allowance is required")
    if pack.judge is None:
        raise ValueError("rubric development requires an exact initial judge")
    if admission is not None:
        require_admitted(admission)
    verified = (
        frozenset()
        if admission is None
        else admission.verified_ids(case.challenge for case in pack.cases)
    )
    development = tuple(case for case in pack.cases if case.partition == "development")
    selection = tuple(case for case in pack.cases if case.partition == "selection")
    if (
        not development
        or not pack.required_dimensions
        or any(not _labeled(pack, case, verified) for case in development)
    ):
        raise ValueError(
            "development needs admitted task labels or required expert criterion labels"
        )
    if not selection or any(not _labeled(pack, case, verified) for case in selection):
        raise ValueError("rubric selection requires a separate independently labeled partition")
    allowed = {case.challenge.id for case in development}
    run = sha256_json(pack)

    def assess(
        config: JudgeConfig, cases: tuple[CalibrationCase, ...]
    ) -> tuple[CalibrationJudgment, ...]:
        """Hide label authority and bind every returned assessment to its actual input."""
        values = []
        for case in cases:
            value = judge(
                config,
                blinded_case(case),
                run + ":" + sha256_json(config) + ":" + case.challenge.id,
            )
            if value.outcome_sha256 != sha256_json(
                case.challenge.outcome
            ) or value.judge_sha256 != sha256_json(config):
                raise ValueError("judge callback returned a different outcome or configuration")
            values.append(value.model_copy(update={"case_id": case.challenge.id}))
        return tuple(values)

    selected = pack.judge
    selected_judgments = assess(selected, development) + assess(selected, selection)
    report = calibrate(
        pack.model_copy(update={"cases": development + selection}),
        selected_judgments,
        admission=admission,
    )
    errors = _errors(report, "development")
    selection_errors = _errors(report, "selection")
    history: list[JsonObject] = []
    for index in range(maximum_revisions):
        if errors and all(error <= pack.score_tolerance for error in errors.values()):
            break
        payload: JsonObject = {
            "judge": selected.model_dump(mode="json"),
            "development": [
                {
                    **blinded_case(case),
                    "case_id": case.challenge.id,
                    "expected_scores": case.expert_scores,
                    "expected_unassessable": list(case.expected_unassessable),
                    "expected_task": case.expected_task,
                    "rationale": case.challenge.rationale,
                    "source_refs": [
                        ref.model_dump(mode="json") for ref in case.challenge.source_refs
                    ],
                }
                for case in development
            ],
            "observed_errors": errors,
            "instructions": (
                "Revise only the rubric/prompt using the supplied independently labeled errors. "
                "Keep the same criterion identities and score bounds. Cite exact development "
                "case IDs. Do not prefer human authorship, add hidden task obligations or "
                "replace evidence with confidence. Preserve creative correct solutions."
            ),
        }
        edit = propose(payload, run + f":proposal:{index}")
        identities = tuple(d.dimension_id for d in selected.definition.dimensions)
        if (
            tuple(d.dimension_id for d in edit.definition.dimensions) != identities
            or not set(edit.evidence_refs) <= allowed
        ):
            raise ValueError("rubric edit changed criteria or cited unavailable evidence")
        candidate = JudgeConfig.model_validate(
            {**selected.model_dump(), "definition": edit.definition, "prompt": edit.prompt}
        )
        candidate_judgments = assess(candidate, development)
        candidate_report = calibrate(
            pack.model_copy(update={"cases": development, "judge": candidate}),
            candidate_judgments,
            admission=admission,
        )
        candidate_errors = _errors(candidate_report, "development")
        authoring_improved = (
            bool(errors)
            and set(candidate_errors) == set(errors)
            and all(candidate_errors[key] <= error for key, error in errors.items())
            and any(candidate_errors[key] < error for key, error in errors.items())
        )
        candidate_selection_errors: dict[str, int] = {}
        if authoring_improved:
            candidate_judgments += assess(candidate, selection)
            candidate_report = calibrate(
                pack.model_copy(update={"cases": development + selection, "judge": candidate}),
                candidate_judgments,
                admission=admission,
            )
            candidate_selection_errors = _errors(candidate_report, "selection")
        improved = (
            authoring_improved
            and set(candidate_selection_errors) == set(selection_errors)
            and all(
                candidate_selection_errors[key] <= error for key, error in selection_errors.items()
            )
        )
        history.append(
            {
                "revision": index,
                "edit": edit.model_dump(mode="json"),
                "judge_sha256": sha256_json(candidate),
                "selected": improved,
                "errors": candidate_errors,
                "selection_errors": candidate_selection_errors,
                "reports": [row.report.model_dump(mode="json") for row in candidate_judgments],
            }
        )
        if improved:
            selected, selected_judgments, errors = candidate, candidate_judgments, candidate_errors
            selection_errors, report = candidate_selection_errors, candidate_report
    targets_met = (
        bool(errors)
        and all(error <= pack.score_tolerance for error in errors.values())
        and report["development_status"] == "passed"
    )
    selection_met = (
        bool(selection_errors)
        and all(error <= pack.score_tolerance for error in selection_errors.values())
        and report["selection_status"] == "passed"
    )
    confirmation = tuple(case for case in pack.cases if case.partition == "confirmation")
    confirmation_ready = (
        targets_met
        and selection_met
        and bool(confirmation)
        and all(_labeled(pack, case, verified) for case in confirmation)
    )
    confirmation_judgments = assess(selected, confirmation) if confirmation_ready else ()
    final = calibrate(
        pack.model_copy(update={"judge": selected}),
        selected_judgments + confirmation_judgments,
        allow_confirmation=confirmation_ready,
        admission=admission,
    )
    if not targets_met or not selection_met:
        final["confirmation_status"] = "not_run"
        final["qualification"] = "diagnostic_only"
    return {
        "selected_judge": selected.model_dump(mode="json"),
        "history": history,
        "development_targets_met": targets_met,
        "selection_targets_met": selection_met,
        "final": final,
        "limitations": [
            "No detected gap on this finite panel does not establish an optimal rubric."
        ],
    }


def _errors(report: JsonObject, partition: str) -> dict[str, int]:
    """Extract actual labeled criterion errors; unavailable judgments remain incomplete."""
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("calibration report lacks case results")
    errors = {}
    for row in rows:
        if isinstance(row, dict) and row.get("partition") != partition:
            continue
        if not isinstance(row, dict) or not isinstance(row.get("judge"), dict):
            raise ValueError("calibration report contains an invalid case")
        assessment = row["judge"]
        absolute = assessment.get("absolute_errors")
        if (
            row.get("expected_task") in {"pass", "fail"}
            and row.get("task_decision") in {"pass", "fail"}
            and assessment.get("status") in {"task_decision_only", "agrees", "disagrees"}
        ):
            errors[str(row["case_id"]) + ":task"] = 0 if row.get("task_correct") is True else 5
            if absolute is None:
                continue
        if not isinstance(absolute, dict):
            raise ValueError(
                "required judge evidence is unavailable; do not optimize on missing scores"
            )
        assessment_errors = assessment.get("assessment_errors", {})
        if not isinstance(assessment_errors, dict):
            raise ValueError("calibration assessment errors are invalid")
        for key, value in assessment_errors.items():
            if type(value) is not int:
                raise ValueError("calibration assessment error is not an integer")
            errors[str(row["case_id"]) + ":assessment:" + key] = value
        for key, value in absolute.items():
            if type(value) is not int:
                raise ValueError("calibration error is not an integer")
            errors[str(row["case_id"]) + ":" + key] = value
    return errors


def _labeled(pack: CalibrationPack, case: CalibrationCase, verified: frozenset[str]) -> bool:
    """Accept scoped task judgments without inventing ordinal expert labels."""
    return label_admitted(case.challenge, verified) and (
        (case.expected_task in {"pass", "fail"} and case.task_label_scope == "task_correctness")
        or set(pack.required_dimensions)
        <= (set(case.expert_scores) | set(case.expected_unassessable))
    )

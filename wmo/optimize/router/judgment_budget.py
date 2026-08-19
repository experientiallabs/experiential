"""Durable exact-plan judgment dispatch reservations for composed router workflows."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    Sha256,
    sorted_unique_inputs,
    stable_id,
)
from wmo.common.evaluations import EvaluationCell, EvaluationCellEvidence, EvaluationProtocol
from wmo.common.evaluations.evidence import (
    evaluation_protocol_digest,
    read_calibration,
    read_evaluation_plan,
    read_judgment,
    read_rollout,
)
from wmo.common.judging import Judge, Judgment
from wmo.common.progress import ProgressHook, report
from wmo.common.project import ArtifactAlreadyExistsError, ProjectStore, artifact_input
from wmo.common.rollouts import RolloutArtifact, StopReason, unknown_spend_failure
from wmo.optimize.router.errors import (
    JudgeDispatchExhaustedError,
    JudgeTranscriptAdmissionError,
    RouterCompositionError,
)

if TYPE_CHECKING:
    from wmo.optimize.router.composition import ApprovedRouterReview, RouterEvaluationSetup

logger = logging.getLogger(__name__)


class JudgmentBudgetError(ValueError):
    """Stored dispatch or judgment evidence cannot safely count toward the workflow budget."""


class JudgmentDispatchReceipt(ArtifactEnvelope):
    """Durable reservation for one exact plan-cell judgment dispatch."""

    dispatch_id: str
    plan: ArtifactInput
    cell_id: str
    rollout: ArtifactInput
    rubric: ArtifactInput
    calibration: ArtifactInput
    protocol_sha256: Sha256


JudgmentExclusionReason = Literal[
    "transcript_exceeds_judge_admission_ceiling",
    "judge_dispatch_failed",
]
"""Structured causes that durably exclude one plan-cell rollout from judging evidence."""


class JudgmentExclusionRecord(ArtifactEnvelope):
    """Durable structured decision excluding one exact plan-cell rollout from judging.

    The record makes a per-cell judge failure replayable: a resumed run reads the exclusion
    and skips the dispatch instead of repeating the failing provider call or aborting.
    """

    exclusion_id: str
    plan: ArtifactInput
    cell_id: str
    rollout: ArtifactInput
    rubric: ArtifactInput
    calibration: ArtifactInput
    protocol_sha256: Sha256
    reason: JudgmentExclusionReason
    detail: str
    conservative_cost_usd: float = 0.0


def find_verified_judgment(
    project: ProjectStore,
    rollout_id: str,
    rubric_id: str,
    calibration_id: str,
    protocol: EvaluationProtocol,
) -> Judgment | None:
    """Find one fully verified judgment so a resume never repeats its dispatch.

    Args:
        project: Project whose immutable evidence is being resumed.
        rollout_id: Exact plan-bound rollout whose judgment is required.
        rubric_id: Approved rubric identity for the workflow.
        calibration_id: Approved calibration identity for the workflow.
        protocol: Frozen evaluation protocol governing the rollout.

    Returns:
        The unique exact judgment, or ``None`` when no matching evidence exists.

    Raises:
        JudgmentBudgetError: Matching evidence is duplicated or differs from frozen pins.
    """
    return find_verified_judgments(
        project,
        protocols_by_rollout={rollout_id: protocol},
        rubric_id=rubric_id,
        calibration_id=calibration_id,
    ).get(rollout_id)


def find_verified_judgments(
    project: ProjectStore,
    *,
    protocols_by_rollout: Mapping[str, EvaluationProtocol],
    rubric_id: str,
    calibration_id: str,
) -> dict[str, Judgment]:
    """Index fully verified judgments for a set of exact plan-bound rollouts.

    Args:
        project: Project whose immutable evidence is being resumed.
        protocols_by_rollout: Frozen protocol for each relevant rollout identity.
        rubric_id: Approved rubric identity for the workflow.
        calibration_id: Approved calibration identity for the workflow.

    Returns:
        Unique exact judgments keyed by relevant rollout identity.

    Raises:
        JudgmentBudgetError: Matching evidence is duplicated or differs from frozen pins.
    """
    matches: dict[str, Judgment] = {}
    for artifact_id in project.artifacts.list_ids():
        stored = project.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "judgment":
            continue
        judgment, _input = read_judgment(project.artifacts, artifact_id)
        protocol = protocols_by_rollout.get(judgment.rollout_id)
        if (
            protocol is None
            or judgment.rubric_id != rubric_id
            or judgment.calibration_id != calibration_id
        ):
            continue
        _require_judgment(
            project,
            judgment,
            judgment.rollout_id,
            rubric_id,
            calibration_id,
            protocol,
        )
        if judgment.rollout_id in matches:
            raise JudgmentBudgetError("multiple judgments bind the same rollout and review")
        matches[judgment.rollout_id] = judgment
    return matches


def read_dispatch_reservation(
    project: ProjectStore,
    plan_input: ArtifactInput,
    cell: EvaluationCell,
    rollout_id: str,
    rubric_id: str,
    calibration_id: str,
    protocol: EvaluationProtocol,
) -> JudgmentDispatchReceipt | None:
    """Load and verify an existing exact dispatch reservation, if present.

    Args:
        project: Project containing the immutable dispatch ledger.
        plan_input: Exact evaluation-plan artifact input.
        cell: Plan cell whose judgment dispatch is being resumed.
        rollout_id: Exact rollout assigned to the cell.
        rubric_id: Approved rubric identity for the workflow.
        calibration_id: Approved calibration identity for the workflow.
        protocol: Frozen evaluation protocol governing the cell.

    Returns:
        The verified reservation, or ``None`` before the first dispatch attempt.

    Raises:
        JudgmentBudgetError: Stored reservation type or bindings have drifted.
    """
    material = _dispatch_material(
        project, plan_input, cell, rollout_id, rubric_id, calibration_id, protocol
    )
    dispatch_id, rollout_input, rubric_input, calibration_input, protocol_sha256 = material
    destination = project.artifacts.project_directory / "artifacts" / dispatch_id
    if not destination.exists():
        return None
    stored = project.artifacts.read(dispatch_id)
    if stored.manifest.artifact_type != "judgment-dispatch":
        raise JudgmentBudgetError("judgment dispatch reservation has the wrong artifact type")
    receipt = JudgmentDispatchReceipt.model_validate_json(
        project.artifacts.read_bytes(dispatch_id, "dispatch.json")
    )
    expected_inputs = sorted_unique_inputs(
        plan_input, rollout_input, rubric_input, calibration_input
    )
    if (
        receipt.dispatch_id != dispatch_id
        or receipt.plan != plan_input
        or receipt.cell_id != cell.cell_id
        or receipt.rollout != rollout_input
        or receipt.rubric != rubric_input
        or receipt.calibration != calibration_input
        or receipt.protocol_sha256 != protocol_sha256
        or sorted_unique_inputs(*receipt.inputs) != expected_inputs
    ):
        raise JudgmentBudgetError("judgment dispatch reservation binding has drifted")
    return receipt


def persist_dispatch_reservation(
    project: ProjectStore,
    plan_input: ArtifactInput,
    cell: EvaluationCell,
    rollout_id: str,
    rubric_id: str,
    calibration_id: str,
    protocol: EvaluationProtocol,
) -> JudgmentDispatchReceipt:
    """Reserve one plan-cell dispatch immutably before calling the injected judge.

    Args:
        project: Project receiving the immutable dispatch reservation.
        plan_input: Exact evaluation-plan artifact input.
        cell: Plan cell whose judgment dispatch is being reserved.
        rollout_id: Exact rollout assigned to the cell.
        rubric_id: Approved rubric identity for the workflow.
        calibration_id: Approved calibration identity for the workflow.
        protocol: Frozen evaluation protocol governing the cell.

    Returns:
        The newly persisted exact dispatch reservation.
    """
    material = _dispatch_material(
        project, plan_input, cell, rollout_id, rubric_id, calibration_id, protocol
    )
    dispatch_id, rollout_input, rubric_input, calibration_input, protocol_sha256 = material
    plan, _plan_input = read_evaluation_plan(project.artifacts, plan_input.artifact_id)
    receipt = JudgmentDispatchReceipt(
        schema_version=1,
        created_at=plan.created_at,
        inputs=sorted_unique_inputs(plan_input, rollout_input, rubric_input, calibration_input),
        code_revision=plan.code_revision,
        dispatch_id=dispatch_id,
        plan=plan_input,
        cell_id=cell.cell_id,
        rollout=rollout_input,
        rubric=rubric_input,
        calibration=calibration_input,
        protocol_sha256=protocol_sha256,
    )
    project.artifacts.write_json(
        artifact_id=dispatch_id,
        artifact_type="judgment-dispatch",
        envelope=receipt,
        files={"dispatch.json": receipt},
    )
    return receipt


def read_judgment_exclusion(
    project: ProjectStore,
    plan_input: ArtifactInput,
    cell: EvaluationCell,
    rollout_id: str,
    rubric_id: str,
    calibration_id: str,
    protocol: EvaluationProtocol,
) -> JudgmentExclusionRecord | None:
    """Load and verify an existing exact judgment exclusion, if present.

    Args:
        project: Project containing the immutable exclusion ledger.
        plan_input: Exact evaluation-plan artifact input.
        cell: Plan cell whose judgment outcome is being resumed.
        rollout_id: Exact rollout assigned to the cell.
        rubric_id: Approved rubric identity for the workflow.
        calibration_id: Approved calibration identity for the workflow.
        protocol: Frozen evaluation protocol governing the cell.

    Returns:
        The verified exclusion, or ``None`` when the cell has never been excluded.

    Raises:
        JudgmentBudgetError: Stored exclusion type or bindings have drifted.
    """
    material = _exclusion_material(
        project, plan_input, cell, rollout_id, rubric_id, calibration_id, protocol
    )
    exclusion_id, rollout_input, rubric_input, calibration_input, protocol_sha256 = material
    destination = project.artifacts.project_directory / "artifacts" / exclusion_id
    if not destination.exists():
        return None
    stored = project.artifacts.read(exclusion_id)
    if stored.manifest.artifact_type != "judgment-exclusion":
        raise JudgmentBudgetError("judgment exclusion record has the wrong artifact type")
    record = JudgmentExclusionRecord.model_validate_json(
        project.artifacts.read_bytes(exclusion_id, "exclusion.json")
    )
    expected_inputs = sorted_unique_inputs(
        plan_input, rollout_input, rubric_input, calibration_input
    )
    if (
        record.exclusion_id != exclusion_id
        or record.plan != plan_input
        or record.cell_id != cell.cell_id
        or record.rollout != rollout_input
        or record.rubric != rubric_input
        or record.calibration != calibration_input
        or record.protocol_sha256 != protocol_sha256
        or sorted_unique_inputs(*record.inputs) != expected_inputs
    ):
        raise JudgmentBudgetError("judgment exclusion record binding has drifted")
    return record


def persist_judgment_exclusion(
    project: ProjectStore,
    plan_input: ArtifactInput,
    cell: EvaluationCell,
    rollout_id: str,
    rubric_id: str,
    calibration_id: str,
    protocol: EvaluationProtocol,
    *,
    reason: JudgmentExclusionReason,
    detail: str,
    conservative_cost_usd: float = 0.0,
) -> JudgmentExclusionRecord:
    """Record one plan-cell judgment exclusion immutably so replay stays deterministic.

    Args:
        project: Project receiving the immutable exclusion record.
        plan_input: Exact evaluation-plan artifact input.
        cell: Plan cell whose judgment is being excluded.
        rollout_id: Exact rollout assigned to the cell.
        rubric_id: Approved rubric identity for the workflow.
        calibration_id: Approved calibration identity for the workflow.
        protocol: Frozen evaluation protocol governing the cell.
        reason: Structured cause of the exclusion.
        detail: Concise operator-facing failure description without secrets.
        conservative_cost_usd: Retry-bound worst-case billed spend of the excluded dispatch.

    Returns:
        The newly persisted exact exclusion record.
    """
    material = _exclusion_material(
        project, plan_input, cell, rollout_id, rubric_id, calibration_id, protocol
    )
    exclusion_id, rollout_input, rubric_input, calibration_input, protocol_sha256 = material
    plan, _plan_input = read_evaluation_plan(project.artifacts, plan_input.artifact_id)
    record = JudgmentExclusionRecord(
        schema_version=1,
        created_at=plan.created_at,
        inputs=sorted_unique_inputs(plan_input, rollout_input, rubric_input, calibration_input),
        code_revision=plan.code_revision,
        exclusion_id=exclusion_id,
        plan=plan_input,
        cell_id=cell.cell_id,
        rollout=rollout_input,
        rubric=rubric_input,
        calibration=calibration_input,
        protocol_sha256=protocol_sha256,
        reason=reason,
        detail=detail,
        conservative_cost_usd=conservative_cost_usd,
    )
    project.artifacts.write_json(
        artifact_id=exclusion_id,
        artifact_type="judgment-exclusion",
        envelope=record,
        files={"exclusion.json": record},
    )
    return record


def _exclusion_material(
    project: ProjectStore,
    plan_input: ArtifactInput,
    cell: EvaluationCell,
    rollout_id: str,
    rubric_id: str,
    calibration_id: str,
    protocol: EvaluationProtocol,
) -> tuple[str, ArtifactInput, ArtifactInput, ArtifactInput, str]:
    """Return one deterministic exclusion identity and its verified artifact inputs."""
    rollout_input = artifact_input(project.artifacts.read(rollout_id).manifest)
    rubric_input = _rubric_input(project, rubric_id)
    calibration, calibration_input = read_calibration(project.artifacts, calibration_id)
    protocol_sha256 = evaluation_protocol_digest(protocol)
    if (
        protocol.rubric_id != rubric_id
        or protocol.judge_calibration_id != calibration_id
        or calibration.rubric_id != rubric_id
    ):
        raise JudgmentBudgetError("judgment exclusion differs from approved review pins")
    exclusion_id = stable_id(
        "judgment-exclusion",
        {
            "plan": plan_input.model_dump(mode="json"),
            "cell_id": cell.cell_id,
            "rollout": rollout_input.model_dump(mode="json"),
            "rubric": rubric_input.model_dump(mode="json"),
            "calibration": calibration_input.model_dump(mode="json"),
            "protocol_sha256": protocol_sha256,
        },
    )
    return exclusion_id, rollout_input, rubric_input, calibration_input, protocol_sha256


def _require_judgment(
    project: ProjectStore,
    judgment: Judgment,
    rollout_id: str,
    rubric_id: str,
    calibration_id: str,
    protocol: EvaluationProtocol,
) -> None:
    """Require exact rollout, review, protocol, calibration, model, prompt, and input pins."""
    rollout_input = artifact_input(project.artifacts.read(rollout_id).manifest)
    rubric_input = _rubric_input(project, rubric_id)
    calibration, calibration_input = read_calibration(project.artifacts, calibration_id)
    if (
        protocol.rubric_id != rubric_id
        or protocol.judge_calibration_id != calibration_id
        or calibration.rubric_id != rubric_id
        or judgment.judgment_id == rollout_id
        or judgment.rubric_id != rubric_id
        or judgment.calibration_id != calibration_id
        or judgment.judge_model != calibration.judge_model
        or judgment.judge_prompt_id != calibration.judge_prompt_id
        or judgment.judge_prompt_sha256 != calibration.judge_prompt_sha256
        or sorted_unique_inputs(*judgment.inputs)
        != sorted_unique_inputs(rollout_input, rubric_input, calibration_input)
    ):
        raise JudgmentBudgetError("persisted judgment differs from its exact plan review pins")


def _dispatch_material(
    project: ProjectStore,
    plan_input: ArtifactInput,
    cell: EvaluationCell,
    rollout_id: str,
    rubric_id: str,
    calibration_id: str,
    protocol: EvaluationProtocol,
) -> tuple[str, ArtifactInput, ArtifactInput, ArtifactInput, str]:
    """Return one deterministic dispatch identity and its verified artifact inputs."""
    rollout_input = artifact_input(project.artifacts.read(rollout_id).manifest)
    rubric_input = _rubric_input(project, rubric_id)
    calibration, calibration_input = read_calibration(project.artifacts, calibration_id)
    protocol_sha256 = evaluation_protocol_digest(protocol)
    if (
        protocol.rubric_id != rubric_id
        or protocol.judge_calibration_id != calibration_id
        or calibration.rubric_id != rubric_id
    ):
        raise JudgmentBudgetError("judgment dispatch differs from approved review pins")
    dispatch_id = stable_id(
        "judgment-dispatch",
        {
            "plan": plan_input.model_dump(mode="json"),
            "cell_id": cell.cell_id,
            "rollout": rollout_input.model_dump(mode="json"),
            "rubric": rubric_input.model_dump(mode="json"),
            "calibration": calibration_input.model_dump(mode="json"),
            "protocol_sha256": protocol_sha256,
        },
    )
    return dispatch_id, rollout_input, rubric_input, calibration_input, protocol_sha256


def _rubric_input(project: ProjectStore, rubric_id: str) -> ArtifactInput:
    """Return one typed rubric input without parsing its unrelated body schema."""
    stored = project.artifacts.read(rubric_id)
    if stored.manifest.artifact_type != "rubric":
        raise JudgmentBudgetError("approved review rubric has the wrong artifact type")
    return artifact_input(stored.manifest)


def complete_cell_evidence(
    project: ProjectStore,
    plan_input: ArtifactInput,
    cells: tuple[EvaluationCell, ...],
    simulated_rollout_ids: tuple[str, ...],
    setup: RouterEvaluationSetup,
    review: ApprovedRouterReview,
    judge: Judge,
    maximum_judgments: int,
    *,
    remaining_cost_usd: float,
    stop_on_overspend: bool,
    spend_ceiling_crossed: Callable[[bool, str, str], None],
    progress: ProgressHook | None = None,
    progress_detail: str | None = None,
) -> tuple[tuple[EvaluationCellEvidence, ...], int, float]:
    """Verify evidence and reserve each bounded judgment dispatch durably before calling it.

    A persisted reservation without a completed judgment marks an interrupted dispatch; the
    judgment is dispatched again under that same consumed reservation, so a judge failure never
    strands the project and never widens the finite judgment budget. A per-cell judge failure
    that cannot complete, an over-ceiling transcript or a dispatch whose bounded retries were
    exhausted, becomes one durable structured exclusion so the run and every replay continue
    past that cell with its rollout excluded from judged evidence.

    Judgments draw from the shared provider pool as reconciled actual spend, never a planning
    estimate. An exhausted dispatch may have billed every bounded attempt without returning
    usable output, so its exclusion charges the conservative retry-bound request cost against
    the same ledger, on the live run and on every replay. Once accumulated judge spend reaches
    ``remaining_cost_usd``, ``stop_on_overspend`` blocks the next dispatch; by default the
    authorized run logs one warning and keeps judging. The returned total covers every judgment
    and exclusion bound to the evidence so later phases subtract actual, not estimated, judge
    cost.

    Args:
        project: Project containing the immutable evidence and dispatch ledger.
        plan_input: Exact evaluation-plan artifact input.
        cells: Phase-specific plan cells requiring bound evidence.
        simulated_rollout_ids: Completed simulated rollouts keyed by their plan cells.
        setup: Reviewed protocols and observed production bindings.
        review: Approved rubric and calibration identifiers.
        judge: Injected judge completing missing judgments.
        maximum_judgments: Finite whole-workflow judgment dispatch ceiling.
        remaining_cost_usd: Shared provider-spend remainder available to judging.
        stop_on_overspend: Whether crossing the shared ceiling blocks the next dispatch.
        spend_ceiling_crossed: Fail-closed or warn-once handler for a crossed spend ceiling.
        progress: Optional progress hook for judgment counting.
        progress_detail: Optional stable progress label.

    Returns:
        Bound cell evidence, consumed dispatch count, and reconciled judge spend.

    Raises:
        RouterCompositionError: Evidence bindings, budgets, or ledger state are invalid.
    """
    rollouts_by_cell = {}
    for rollout_id in simulated_rollout_ids:
        rollout, _input = read_rollout(project.artifacts, rollout_id)
        if rollout.cell_id is None or rollout.cell_id in rollouts_by_cell:
            raise RouterCompositionError("simulator output lacks unique evaluation cell bindings")
        rollouts_by_cell[rollout.cell_id] = rollout
    observed = {
        (item.task_id, item.candidate_alias, item.repeat): item.rollout_artifact_id
        for item in setup.observed_cells
    }
    bound_cells = []
    protocols_by_rollout: dict[str, EvaluationProtocol] = {}
    rollouts_by_id: dict[str, RolloutArtifact] = {}
    for cell in cells:
        if cell.execution != "observed":
            simulated = rollouts_by_cell.get(cell.cell_id)
            if simulated is not None and unknown_spend_failure(simulated.failure):
                continue
        rollout_id = (
            cell.observed_rollout_id
            if cell.execution == "observed"
            else getattr(rollouts_by_cell.get(cell.cell_id), "rollout_id", None)
        )
        if rollout_id is None:
            raise RouterCompositionError(
                f"no completed rollout exists for planned cell {cell.cell_id}"
            )
        if (
            cell.execution == "observed"
            and observed.get((cell.task_id, cell.candidate_alias, cell.repeat)) != rollout_id
        ):
            raise RouterCompositionError("observed rollout binding changed after planning")
        protocol = (
            setup.production_protocol if cell.execution == "observed" else setup.simulation_protocol
        )
        existing_protocol = protocols_by_rollout.setdefault(rollout_id, protocol)
        if existing_protocol != protocol:
            raise RouterCompositionError("one rollout is bound to conflicting evaluation protocols")
        if rollout_id not in rollouts_by_id:
            rollouts_by_id[rollout_id] = read_rollout(project.artifacts, rollout_id)[0]
        bound_cells.append((cell, rollout_id, protocol))
    judgeable_protocols = {
        rollout_id: protocol
        for rollout_id, protocol in protocols_by_rollout.items()
        if not _rollout_failed(rollouts_by_id[rollout_id])
    }
    try:
        judgments_by_rollout = find_verified_judgments(
            project,
            protocols_by_rollout=judgeable_protocols,
            rubric_id=review.rubric_id,
            calibration_id=review.calibration_id,
        )
    except JudgmentBudgetError as exc:
        raise RouterCompositionError(str(exc)) from exc

    evidence: list[EvaluationCellEvidence] = []
    consumed = 0
    overspend_warned = False
    judge_spend_usd = math.fsum(
        _known_judgment_spend(judgment) for judgment in judgments_by_rollout.values()
    )

    def _report_judgments() -> None:
        """Report judgment progress after appending one evidence row."""
        report(
            progress,
            "judgments",
            completed=len(evidence),
            total=len(bound_cells),
            detail=progress_detail,
        )

    report(progress, "judgments", completed=0, total=len(bound_cells), detail=progress_detail)
    for cell, rollout_id, protocol in bound_cells:
        rollout = rollouts_by_id[rollout_id]
        if _rollout_failed(rollout):
            evidence.append(_unjudged_cell_evidence(cell, protocol, rollout))
            continue
        try:
            judgment = judgments_by_rollout.get(rollout_id)
            receipt = read_dispatch_reservation(
                project,
                plan_input,
                cell,
                rollout_id,
                review.rubric_id,
                review.calibration_id,
                protocol,
            )
            exclusion = read_judgment_exclusion(
                project,
                plan_input,
                cell,
                rollout_id,
                review.rubric_id,
                review.calibration_id,
                protocol,
            )
            if judgment is None and receipt is not None:
                judgment = find_verified_judgment(
                    project,
                    rollout_id,
                    review.rubric_id,
                    review.calibration_id,
                    protocol,
                )
                if judgment is not None:
                    judgments_by_rollout[rollout_id] = judgment
                    judge_spend_usd = math.fsum((judge_spend_usd, _known_judgment_spend(judgment)))
        except JudgmentBudgetError as exc:
            raise RouterCompositionError(str(exc)) from exc
        if judgment is not None or receipt is not None:
            consumed += 1
        if consumed > maximum_judgments:
            raise RouterCompositionError("judgment dispatch budget exhausted")
        if judgment is None and exclusion is not None:
            judge_spend_usd = math.fsum((judge_spend_usd, exclusion.conservative_cost_usd))
            evidence.append(_unjudged_cell_evidence(cell, protocol, rollout))
            _report_judgments()
            continue
        if judgment is None:
            if judge_spend_usd >= remaining_cost_usd:
                if stop_on_overspend or not overspend_warned:
                    spend_ceiling_crossed(
                        stop_on_overspend,
                        "reconciled provider spend reached the shared ceiling before judgment "
                        "dispatch; increase --maximum-simulation-cost-usd and rerun to resume",
                        f"reconciled judge spend ${judge_spend_usd:.4f} reached the shared "
                        f"authorized remainder ${remaining_cost_usd:.4f}",
                    )
                    overspend_warned = True
            if receipt is None:
                if consumed >= maximum_judgments:
                    raise RouterCompositionError("judgment dispatch budget exhausted")
                try:
                    persist_dispatch_reservation(
                        project,
                        plan_input,
                        cell,
                        rollout_id,
                        review.rubric_id,
                        review.calibration_id,
                        protocol,
                    )
                except JudgmentBudgetError as exc:
                    raise RouterCompositionError(str(exc)) from exc
                consumed += 1
            try:
                judgment = judge.judge_persisted(
                    project,
                    rollout_artifact_id=rollout_id,
                    rubric_artifact_id=review.rubric_id,
                    calibration_artifact_id=review.calibration_id,
                )
            except (JudgeTranscriptAdmissionError, JudgeDispatchExhaustedError) as exc:
                exhausted_cost_usd = (
                    exc.conservative_cost_usd
                    if isinstance(exc, JudgeDispatchExhaustedError)
                    else 0.0
                )
                _record_judgment_exclusion(
                    project,
                    plan_input,
                    cell,
                    rollout_id,
                    review,
                    protocol,
                    reason=(
                        "transcript_exceeds_judge_admission_ceiling"
                        if isinstance(exc, JudgeTranscriptAdmissionError)
                        else "judge_dispatch_failed"
                    ),
                    error=exc,
                    conservative_cost_usd=exhausted_cost_usd,
                )
                judge_spend_usd = math.fsum((judge_spend_usd, exhausted_cost_usd))
                evidence.append(_unjudged_cell_evidence(cell, protocol, rollout))
                _report_judgments()
                continue
            _persist_judgment(project, judgment)
            judgments_by_rollout[rollout_id] = judgment
            judge_spend_usd = math.fsum((judge_spend_usd, _known_judgment_spend(judgment)))
        evidence.append(
            EvaluationCellEvidence(
                cell_id=cell.cell_id,
                protocol_id=protocol.protocol_id,
                rollout_artifact_id=rollout_id,
                judgment_artifact_id=judgment.judgment_id,
                source_run_id=rollout.source_run_id,
            )
        )
        _report_judgments()
    return tuple(evidence), consumed, judge_spend_usd


def _record_judgment_exclusion(
    project: ProjectStore,
    plan_input: ArtifactInput,
    cell: EvaluationCell,
    rollout_id: str,
    review: ApprovedRouterReview,
    protocol: EvaluationProtocol,
    *,
    reason: JudgmentExclusionReason,
    error: Exception,
    conservative_cost_usd: float = 0.0,
) -> None:
    """Persist one durable structured judgment exclusion and warn the operator.

    Args:
        project: Project receiving the immutable exclusion record.
        plan_input: Exact evaluation-plan artifact input.
        cell: Plan cell whose judgment is being excluded.
        rollout_id: Exact rollout assigned to the cell.
        review: Approved rubric and calibration identifiers.
        protocol: Frozen evaluation protocol governing the cell.
        reason: Structured cause of the exclusion.
        error: Per-cell judge failure that triggered the exclusion.
        conservative_cost_usd: Retry-bound worst-case billed spend of the excluded dispatch.

    Raises:
        RouterCompositionError: The exclusion cannot bind to the approved review pins.
    """
    try:
        persist_judgment_exclusion(
            project,
            plan_input,
            cell,
            rollout_id,
            review.rubric_id,
            review.calibration_id,
            protocol,
            reason=reason,
            detail=str(error),
            conservative_cost_usd=conservative_cost_usd,
        )
    except JudgmentBudgetError as exc:
        raise RouterCompositionError(str(exc)) from exc
    logger.warning(
        "cell %s excluded from judging evidence (%s): %s",
        cell.cell_id,
        reason,
        error,
    )


def _unjudged_cell_evidence(
    cell: EvaluationCell,
    protocol: EvaluationProtocol,
    rollout: RolloutArtifact,
) -> EvaluationCellEvidence:
    """Return one evidence row binding a rollout that carries no judgment artifact.

    Args:
        cell: Plan cell whose rollout stays unjudged.
        protocol: Frozen evaluation protocol governing the cell.
        rollout: Completed rollout retained as spend evidence without a judgment.

    Returns:
        Evidence row binding the rollout with no judgment artifact.
    """
    return EvaluationCellEvidence(
        cell_id=cell.cell_id,
        protocol_id=protocol.protocol_id,
        rollout_artifact_id=rollout.rollout_id,
        judgment_artifact_id=None,
        source_run_id=rollout.source_run_id,
    )


def _known_judgment_spend(judgment: Judgment) -> float:
    """Return one judgment's reconciled judge dispatch cost.

    Args:
        judgment: Persisted or freshly dispatched judgment.

    Returns:
        Known judge spend in USD, or zero when the judge reported no economics.
    """
    economics = judgment.judge_economics
    if economics is None or economics.cost_usd is None:
        return 0.0
    return economics.cost_usd.value


def _rollout_failed(rollout: RolloutArtifact) -> bool:
    """Return whether one persisted rollout terminated as failed evidence.

    Failed rollouts never receive a judgment: the evaluation builder scores them as failed rows
    directly and rejects any judgment bound to them, so dispatching a judge call against one
    would waste real spend on an episode that produced no gradable output.

    Args:
        rollout: Verified persisted rollout evidence.

    Returns:
        True when the rollout carries a structured failure or a failed stop reason.
    """
    return rollout.failure is not None or rollout.stop_reason == StopReason.FAILURE


def _persist_judgment(project: ProjectStore, judgment: Judgment) -> None:
    """Persist or exactly verify one deterministic injected-judge result."""
    try:
        project.artifacts.write_json(
            artifact_id=judgment.judgment_id,
            artifact_type="judgment",
            envelope=judgment,
            files={"judgment.json": judgment},
        )
    except ArtifactAlreadyExistsError:
        existing, _input = read_judgment(project.artifacts, judgment.judgment_id)
        if existing != judgment:
            raise RouterCompositionError(
                "existing judgment differs from injected judge result"
            ) from None

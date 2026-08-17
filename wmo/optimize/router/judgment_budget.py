"""Durable exact-plan judgment dispatch reservations for composed router workflows."""

from __future__ import annotations

from collections.abc import Mapping

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    Sha256,
    sorted_unique_inputs,
    stable_id,
)
from wmo.common.evaluations import EvaluationCell, EvaluationProtocol
from wmo.common.evaluations.evidence import (
    evaluation_protocol_digest,
    read_calibration,
    read_evaluation_plan,
    read_judgment,
)
from wmo.common.judging import Judgment
from wmo.common.project import ProjectStore, artifact_input


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

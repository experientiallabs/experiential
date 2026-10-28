"""Simulated record-world trajectories are admitted only inside an applicable calibration.

Every trajectory is a real continuation study: scripted actor responses against the native
SQLite ticket service and the same service run as a simulated owner. The calibrations are
computed by ``calibrate`` from pairs of those real branches across distinct task families.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.project import ArtifactStore, ProjectStore
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.optimize.evaluation.continuations.traces_test import (
    ANSWERED,
    MODEL,
    capture,
    simulated_tickets,
    study,
    ticket_fork,
    ticket_task,
)
from exp.optimize.evaluation.equivalence.calibration import CaseScope, calibrate
from exp.optimize.evaluation.equivalence.dimensions import pair_dimensions, world_identity
from exp.optimize.evaluation.equivalence.evidence import PairMeasurement
from exp.optimize.evaluation.equivalence.policy import AdmissionPolicy, Dimension, Gate
from exp.optimize.evaluation.equivalence.policy_test import fixture_policy, registered_policy
from exp.optimize.evaluation.equivalence.verdicts import (
    CalibrationEvidence,
    TrajectoryAdmission,
    admit_trajectory,
)
from exp.runtime.environments.tests.world_conformance_test import ticket_case
from exp.simulation.continuations import ContinuationCheckpoint
from exp.simulation.evaluation.tasks import TaskVerifier

REPORT = ArtifactInput(artifact_id="actor-equivalence-fixture-result", sha256="1" * 64)
VERIFIER = ANSWERED.model_copy(update={"allowed_tools": ("open_ticket", "close_ticket")})


def run(
    store: ArtifactStore, index: int, *, simulated: bool, variant: str = "trajectory"
) -> tuple[ContinuationCheckpoint, ContinuationBranch]:
    """Capture one task of family ``index`` in a native or simulated world and continue it once.

    Args:
        store: Project artifact store.
        index: Task family; every variant shares its lineage.
        simulated: Run the simulated ticket owner instead of the native service.
        variant: Distinct task within the family, so each capture has its own source.

    Returns:
        The captured checkpoint and the continued branch.
    """
    world = simulated_tickets() if simulated else ticket_case().runtime
    task = ticket_task(index).model_copy(update={"task_id": f"ticket-task-{index}-{variant}"})
    reference, checkpoint = capture(store, world, task)
    result = study(store, reference, world, [ticket_fork(index, 0)], verifier=VERIFIER)
    return checkpoint, load_study(store, result).branches[0]


def calibration_of(
    policy_ref: ArtifactInput,
    policy: AdmissionPolicy,
    store: ArtifactStore,
    families: Sequence[int],
    *,
    agreement: Literal["match", "different"] = "match",
    tag: str = "",
) -> CalibrationEvidence:
    """Pair native and simulated runs of each family, with a native repeat as the control."""
    measurements, scopes = [], []
    for case, index in enumerate(families):
        _, native = run(store, index, simulated=False, variant="reference" + tag)
        _, repeat = run(store, index, simulated=False, variant="repeat" + tag)
        simulated_point, simulated = run(store, index, simulated=True, variant="candidate" + tag)
        scopes.append(
            CaseScope(
                case=case,
                lineage=ticket_task(index).lineage_group_id,
                world=sha256_json(world_identity(simulated_point)),
                actor=sha256_json(MODEL),
            )
        )
        pairs: tuple[
            tuple[
                Literal["candidate", "reference_control"],
                ContinuationBranch,
                Literal["match", "different"],
            ],
            ...,
        ] = (
            ("candidate", simulated, agreement),
            ("reference_control", repeat, "match"),
        )
        for condition, other, material in pairs:
            measurements.append(
                PairMeasurement.model_construct(
                    case=case,
                    repeat=0,
                    condition=condition,
                    material_agreement=material,
                    dimensions=pair_dimensions(
                        native,
                        other,
                        verifier=VERIFIER,
                        tools=ticket_task(index).tools,
                        material_agreement=material,
                    ),
                )
            )
    return CalibrationEvidence(
        report=REPORT,
        calibration=calibrate(policy_ref, policy, measurements, scopes, state_coverage="full"),
    )


def admitted(
    policy_ref: ArtifactInput,
    policy: AdmissionPolicy,
    point: ContinuationCheckpoint,
    branch: ContinuationBranch,
    calibrations: Sequence[CalibrationEvidence] = (),
    verifier: TaskVerifier = VERIFIER,
) -> TrajectoryAdmission:
    """Admit one saved trajectory under a policy."""
    return admit_trajectory(
        policy_ref,
        policy,
        branch,
        checkpoint=point,
        verifier=verifier,
        actor_model=MODEL,
        calibrations=calibrations,
    )


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    """A fresh project store."""
    return ProjectStore(tmp_path, "case").artifacts


def test_native_record_trajectory_is_admitted_without_calibration(store: ArtifactStore) -> None:
    """Native receipts establish fidelity; no calibration is consulted."""
    policy = fixture_policy()
    reference = registered_policy(store, policy)
    point, branch = run(store, 0, simulated=False)
    verdict = admitted(reference, policy, point, branch)
    assert verdict.status == "admitted" and not verdict.simulated
    assert all(row.calibration is None for row in verdict.dimensions)


def test_simulated_trajectory_needs_an_in_scope_admitting_calibration(
    store: ArtifactStore,
) -> None:
    """Calibration over three families admits the same world, actor and family only."""
    policy = fixture_policy()
    reference = registered_policy(store, policy)
    evidence = calibration_of(reference, policy, store, (0, 1, 2))
    assert evidence.calibration.dimension("goal_success").status == "admitted"
    assert evidence.calibration.dimension("state_fidelity").status == "admitted"

    point, branch = run(store, 1, simulated=True)
    verdict = admitted(reference, policy, point, branch, (evidence,))
    assert verdict.status == "admitted" and verdict.simulated, verdict.reasons
    rows = {row.dimension: row for row in verdict.dimensions}
    assert rows["state_fidelity"].observed == "unassessable"
    assert rows["state_fidelity"].calibration == REPORT.artifact_id
    assert "calibrated" in rows["goal_success"].reason

    alone = admitted(reference, policy, point, branch)
    assert alone.status == "unassessable"
    assert any("none was supplied" in reason for reason in alone.reasons)

    new_point, new_branch = run(store, 9, simulated=True)
    outside = admitted(reference, policy, new_point, new_branch, (evidence,))
    assert outside.status == "unassessable"
    assert any("start outside the calibrated scope" in reason for reason in outside.reasons)


def test_a_refusing_calibration_refuses_fidelity_and_never_rescues_a_failure(
    store: ArtifactStore,
) -> None:
    """Demonstrated state disagreement refuses; an observed goal failure stays refused."""
    policy = fixture_policy(margin=0.1)
    reference = registered_policy(store, policy)
    evidence = calibration_of(reference, policy, store, range(5), agreement="different")
    assert evidence.calibration.dimension("state_fidelity").status == "refused"
    point, branch = run(store, 2, simulated=True)
    verdict = admitted(reference, policy, point, branch, (evidence,))
    assert verdict.status == "refused"
    assert any(
        reason.startswith("state_fidelity refused: calibration refused")
        for reason in verdict.reasons
    )

    lenient = fixture_policy()
    lenient_ref = registered_policy(store, lenient)
    calibrated = calibration_of(lenient_ref, lenient, store, (0, 1, 2), tag="-lenient")
    wrong = VERIFIER.model_copy(
        update={"rules": (VERIFIER.rules[0].model_copy(update={"value": "other"}),)}
    )
    failed = admitted(lenient_ref, lenient, point, branch, (calibrated,), verifier=wrong)
    goal = next(row for row in failed.dimensions if row.dimension == "goal_success")
    assert (goal.observed, goal.status, failed.status) == ("refused", "refused", "refused")


def test_an_authored_world_policy_trusts_observations_and_records_fidelity(
    store: ArtifactStore,
) -> None:
    """A synthetic-contract policy admits simulated trajectories without a native claim."""
    gates: dict[Dimension, Gate] = {"state_fidelity": "recorded"}
    base = fixture_policy(gates=gates)
    authored = base.model_copy(
        update={
            "description": "Authored ticket-world contract; no deployment equivalence claimed.",
            "dimensions": tuple(
                rule
                if rule.dimension == "state_fidelity"
                else rule.model_copy(update={"simulated": "observed"})
                for rule in base.dimensions
            ),
        }
    )
    reference = registered_policy(store, authored)
    point, branch = run(store, 3, simulated=True)
    verdict = admitted(reference, authored, point, branch)
    assert verdict.status == "admitted" and verdict.simulated
    fidelity = next(row for row in verdict.dimensions if row.dimension == "state_fidelity")
    assert (fidelity.gate, fidelity.status) == ("recorded", "unassessable")


def test_a_covering_refusal_wins_over_an_admission_in_either_order(
    store: ArtifactStore,
) -> None:
    """One favorable report never masks a demonstrated disagreement, whatever the list order."""
    policy = fixture_policy()
    reference = registered_policy(store, policy)
    admitting = calibration_of(reference, policy, store, (0, 1, 2))
    refused = tuple(
        row.model_copy(update={"status": "refused", "reason": "demonstrated disagreement"})
        if row.dimension in ("state_fidelity", "goal_success")
        else row
        for row in admitting.calibration.dimensions
    )
    refusing = CalibrationEvidence(
        report=REPORT.model_copy(update={"artifact_id": "actor-equivalence-refusing-result"}),
        calibration=admitting.calibration.model_copy(update={"dimensions": refused}),
    )
    point, branch = run(store, 1, simulated=True)
    for calibrations in ((admitting, refusing), (refusing, admitting)):
        verdict = admitted(reference, policy, point, branch, calibrations)
        rows = {row.dimension: row for row in verdict.dimensions}
        assert verdict.status == "refused"
        assert rows["state_fidelity"].status == "refused"
        assert rows["goal_success"].status == "unassessable"
        assert {rows[name].calibration for name in ("state_fidelity", "goal_success")} == {
            refusing.report.artifact_id
        }

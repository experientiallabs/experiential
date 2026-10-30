"""Store admission verifies native and specification witnesses and names every refusal."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.common.project.database import project_connection
from exp.common.tasks import TaskCase
from exp.common.tasks.store import load_task_set
from exp.optimize.evaluation.audit.exposure import held_out_members
from exp.optimize.evaluation.audit.lifecycle_test import audit
from exp.optimize.evaluation.audit.panel_test import (
    USERS,
    SimulatedProvider,
    build_project,
    episode,
    held_out_lineages,
)
from exp.optimize.evaluation.native_artifacts import materialize_native_artifacts
from exp.optimize.evaluation.native_artifacts_test import TextAdapter, native_fixture
from exp.optimize.evaluation.tasks.admission import (
    admit_calibration_evidence,
    knowledge_refusals,
    require_refinement_admission,
)
from exp.optimize.evaluation.tasks.calibration import CalibrationSpec, execute_calibration
from exp.optimize.evaluation.tests.knowledge_exposure_test import OWN, bound
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.evaluation import admission as witness_admission
from exp.simulation.evaluation.admission import (
    NATIVE_RECEIPT_TYPE,
    EvaluationSpecification,
    EvidenceAdmissionError,
    persist_specification,
    record_label_witness,
    verify_label_witness,
)
from exp.simulation.evaluation.calibration import CalibrationCase, CalibrationPack
from exp.simulation.evaluation.tasks import (
    EvaluationTask,
    OutcomeRule,
    TaskVerifier,
    VerifierChallenge,
    persist_evaluation_task,
)
from exp.simulation.mining.exposure_ledger import ExposureLedger, LedgerExposure
from exp.simulation.world_model.checkpoints import load_world_checkpoint

NOW = datetime(2026, 10, 17, tzinfo=UTC)
PARENT = TaskVerifier(description="Weak", outcome_schema={"type": "object"})
CANDIDATE = TaskVerifier(
    description="Candidate: the order is done.",
    outcome_schema={"type": "object"},
    rules=(OutcomeRule(path="/final_state/status", operator="equals", value="done"),),
)
REFERENCE = TaskVerifier(
    description="Reference: the reported answer is done.",
    outcome_schema={"type": "object"},
    rules=(OutcomeRule(path="/final_answer", operator="equals", value="done"),),
)
NATIVE = TaskVerifier(
    description="Native contract: the produced file reads 'initial changed'.",
    outcome_schema={"type": "object"},
    rules=(
        OutcomeRule(path="/native_projection/text", operator="equals", value="initial changed"),
    ),
)

MAINTAINED_ADAPTERS = witness_admission.maintained_native_adapters


@pytest.fixture(autouse=True)
def trusted_text_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trust the text fixture adapter so native receipts re-derive without a workbook."""
    monkeypatch.setattr(witness_admission, "maintained_native_adapters", lambda: (TextAdapter(),))


def outcome(status: str, marker: str, **state: object) -> JsonObject:
    """A complete normalized outcome whose answer and final status agree."""
    return {
        "initial_state": {},
        "final_state": {"status": status, **state},
        "final_answer": status,
        "steps": [],
        "stop_reason": "completed",
        "marker": marker,
    }


def specification(
    project: ProjectStore,
    predicate: TaskVerifier = REFERENCE,
    *,
    subject: str = "outcome",
    sources: tuple[ArtifactInput, ...] = (),
) -> ArtifactInput:
    """Save one independent specification."""
    return persist_specification(
        project.artifacts,
        EvaluationSpecification.model_validate(
            {
                "description": "Independent acceptance specification.",
                "predicate": predicate.model_dump(mode="json"),
                "subject": subject,
            }
        ),
        sources=sources,
        created_at=NOW,
        code_revision="test",
    )


def label(
    project: ProjectStore,
    spec: ArtifactInput,
    identity: str,
    value: JsonObject,
    *,
    receipt: ArtifactInput | None = None,
) -> VerifierChallenge:
    """A witnessed label: no approval flag, only a saved re-derivable witness."""
    witness = record_label_witness(
        project.artifacts,
        specification=spec,
        outcome=value,
        receipt=receipt,
        created_at=NOW,
        code_revision="test",
    )
    stored = json.loads(project.artifacts.read_bytes(witness.artifact_id, "witness.json"))
    return VerifierChallenge(
        id=identity,
        outcome=value,
        expected=stored["verdict"],
        authority="synthetic_specification" if receipt is None else "native_execution",
        rationale="Saved witness decides the verdict.",
        witness=witness,
    )


def native_label(
    tmp_path: Path, *, predicted: str = "initial changed"
) -> tuple[ProjectStore, VerifierChallenge, ArtifactInput]:
    """A native-execution label over a real materialization receipt and its source outcome."""
    project, spec, adapter, _ = native_fixture(tmp_path, predicted=predicted)
    receipt = materialize_native_artifacts(project.artifacts, spec, adapter, code_revision="test")
    assert receipt is not None
    state = load_world_checkpoint(project.artifacts, spec.checkpoint).current.business_state
    value = outcome("done", "native", **state)
    native_spec = specification(project, NATIVE, subject="native_observation")
    return project, label(project, native_spec, "native", value, receipt=receipt), receipt


@pytest.mark.parametrize("predicted", ["initial changed", "wrong prediction"])
def test_native_receipt_witnesses_a_label_from_the_actual_observation(
    tmp_path: Path, predicted: str
) -> None:
    """The native observation, not the simulator's prediction, decides the witnessed verdict."""
    project, challenge, receipt = native_label(tmp_path, predicted=predicted)
    assert challenge.expected == "pass" and challenge.approved is False
    assert (
        verify_label_witness(project.artifacts, challenge, label="pass", compared=(CANDIDATE,))
        is None
    )
    status = json.loads(project.artifacts.read_bytes(receipt.artifact_id, "receipt.json"))
    assert status["status"] == ("matched" if predicted == "initial changed" else "diverged")


def test_native_witness_must_reproduce_input_adapter_plan_and_observation(
    tmp_path: Path,
) -> None:
    """Another plan, a forged receipt digest or changed native bytes are each refused."""
    project, challenge, receipt = native_label(tmp_path)
    assert isinstance(challenge.outcome, dict)
    final_state = challenge.outcome["final_state"]
    assert isinstance(final_state, dict)
    plan = dict(final_state["native_artifact_plan"])
    plan["expected_projection"] = {"text": "another plan"}
    other = outcome("done", "other", native_artifact_plan=plan)
    native_spec = specification(project, NATIVE, subject="native_observation")
    with pytest.raises(ValueError, match="different plan"):
        record_label_witness(
            project.artifacts,
            specification=native_spec,
            outcome=other,
            receipt=receipt,
            created_at=NOW,
            code_revision="test",
        )
    mismatched = challenge.model_copy(update={"authority": "synthetic_specification"})
    refusal = verify_label_witness(project.artifacts, mismatched, label="pass", compared=())
    assert refusal is not None and "needs a native_execution" in refusal.detail
    with project_connection(project.paths.root, write=True) as connection:
        connection.execute(
            "UPDATE project_artifact_files SET payload=? WHERE project_id=? AND artifact_id=? "
            "AND path='outputs/output.txt'",
            (b"tampered native bytes", project.paths.project_id, receipt.artifact_id),
        )
    refusal = verify_label_witness(project.artifacts, challenge, label="pass", compared=())
    assert refusal is not None
    assert refusal.reason == "unverifiable_witness"
    assert "does not re-verify" in refusal.detail


def test_native_witness_rederives_the_receipt_projection_from_its_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt written straight to the store with an invented projection is refused."""
    project, challenge, receipt = native_label(tmp_path)
    stored = project.artifacts.read(receipt.artifact_id).manifest
    files = {
        row.path: project.artifacts.read_bytes(receipt.artifact_id, row.path)
        for row in stored.files
    }
    forged_receipt = json.loads(files["receipt.json"])
    forged_receipt["native_projection"] = {"text": "invented"}
    forged = artifact_input(
        project.artifacts.write(
            artifact_id="forged-native-receipt",
            artifact_type=NATIVE_RECEIPT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=NOW, inputs=stored.inputs, code_revision="test"
            ),
            files={**files, "receipt.json": json.dumps(forged_receipt).encode()},
        )
    )
    native_spec = specification(project, NATIVE, subject="native_observation")
    with pytest.raises(ValueError, match="projection differs from what the adapter derives"):
        record_label_witness(
            project.artifacts,
            specification=native_spec,
            outcome=challenge.outcome,
            receipt=forged,
            created_at=NOW,
            code_revision="test",
        )
    monkeypatch.setattr(witness_admission, "maintained_native_adapters", MAINTAINED_ADAPTERS)
    refusal = verify_label_witness(project.artifacts, challenge, label="pass", compared=())
    assert refusal is not None and "no maintained native adapter" in refusal.detail


def calibration(
    project: ProjectStore,
    cases: list[CalibrationCase],
    *,
    shown: tuple[VerifierChallenge, ...] = (),
) -> CalibrationSpec:
    """Persist the candidate task and a frozen calibration spec over these cases."""
    world = load_world_checkpoint(
        project.artifacts,
        artifact_input(
            project.artifacts.read(
                project.artifacts.list_ids(artifact_type="world-checkpoint")[0]
            ).manifest
        ),
    ).world_model
    task = persist_evaluation_task(
        project.artifacts,
        EvaluationTask(
            intent="Complete the order",
            task=TaskCase(
                task_id="order",
                origin="authored",
                lineage_group_id="orders",
                partition="fit",
                instruction="Complete the order.",
                workload_weight=1,
            ),
            world_model=world,
            checkpoint=world,
            verifier=CANDIDATE,
            challenges=shown,
        ),
        created_at=NOW,
        code_revision="test",
    )
    return CalibrationSpec(
        run_id="witnessed",
        pack=CalibrationPack(task=task, parent=PARENT, candidate=CANDIDATE, cases=tuple(cases)),
    )


def witnessed_cases(project: ProjectStore, spec: ArtifactInput) -> list[CalibrationCase]:
    """Two positive and two negative specification-witnessed labels per partition."""
    return [
        CalibrationCase(
            challenge=label(project, spec, f"{partition}-{status}-{n}", outcome(status, f"{n}")),
            lineage=partition,
            partition=partition,
        )
        for partition in ("development", "confirmation")
        for status in ("done", "pending")
        for n in (partition + "-1", partition + "-2")
    ]


def test_calibration_admits_witnessed_labels_without_any_approval(tmp_path: Path) -> None:
    """Store admission re-derives every witness; no label carries an approval."""
    project, challenge, _ = native_label(tmp_path)
    cases = witnessed_cases(project, specification(project))
    cases.append(CalibrationCase(challenge=challenge, lineage="native", partition="confirmation"))
    spec = calibration(project, cases)
    assert not any(case.challenge.approved for case in spec.pack.cases)
    reference = execute_calibration(project, spec, code_revision="test")
    assert reference is not None
    report = json.loads(project.artifacts.read_bytes(reference.artifact_id, "report.json"))
    assert report["witnessed_verified"] == 9
    assert report["independent"] == 9
    assert report["label_authorities"] == {"native_execution": 1, "synthetic_specification": 8}
    assert report["development_status"] == "passed"
    assert report["confirmation_status"] == "passed"
    assert report["qualification_refusals"] == []
    inputs = project.artifacts.read(reference.artifact_id).manifest.inputs
    assert {case.challenge.witness for case in spec.pack.cases} <= set(inputs)


def test_generated_specification_chain_cannot_approve_its_labels(tmp_path: Path) -> None:
    """A specification derived from a role proposal witnesses nothing: self-approval by name."""
    project, _, _ = native_label(tmp_path)
    proposal = artifact_input(
        project.artifacts.write_json(
            artifact_id="role-proposal",
            artifact_type="task-refinement-proposal",
            envelope=ArtifactEnvelope(schema_version=1, created_at=NOW, code_revision="test"),
            files={"proposal.json": {"verifier": None}},
        )
    )
    spec = calibration(
        project, witnessed_cases(project, specification(project, sources=(proposal,)))
    )
    admission = admit_calibration_evidence(project, spec.pack)
    assert admission.verified == {}
    assert {row.reason for row in admission.refusals} == {"self_approval"}
    assert len(admission.refusals) == 8


def test_candidate_cannot_witness_itself(tmp_path: Path) -> None:
    """A specification identical to the candidate predicate is self-approval."""
    project, _, _ = native_label(tmp_path)
    spec = calibration(project, witnessed_cases(project, specification(project, CANDIDATE)))
    admission = admit_calibration_evidence(project, spec.pack)
    assert {row.reason for row in admission.refusals} == {"self_approval"}


def test_labels_shown_to_task_authoring_breach_the_sealed_partition(tmp_path: Path) -> None:
    """A confirmation outcome that was a development challenge of the task is refused."""
    project, _, _ = native_label(tmp_path)
    cases = witnessed_cases(project, specification(project))
    shown = cases[-1].challenge
    spec = calibration(project, cases, shown=(shown,))
    refusals = admit_calibration_evidence(project, spec.pack).refusals
    assert [(row.reason, row.subject) for row in refusals] == [
        ("sealed_partition_breach", shown.id)
    ]


def test_knowledge_built_from_the_task_contaminates_it(tmp_path: Path) -> None:
    """The simulator's pinned knowledge exposing the task's own trace is named contamination."""
    exposed = bound(tmp_path / "exposed", SimulationKnowledge(source_trace_ids=(OWN,)))
    project = ProjectStore(tmp_path / "exposed", "fidelity-test")
    refusals = knowledge_refusals(project, exposed.world_model, exposed.task)
    assert [row.reason for row in refusals] == ["contaminated_lineage"]
    assert "own recorded evidence" in refusals[0].detail
    clean = bound(tmp_path / "clean")
    project = ProjectStore(tmp_path / "clean", "fidelity-test")
    assert knowledge_refusals(project, clean.world_model, clean.task) == ()


def test_reserved_or_sealed_held_out_lineages_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Held-out lineages: never in development, and blocked in confirmation while reserved."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    build = store.load_project().build
    assert build is not None
    fit = next(
        task
        for task in load_task_set(store.artifacts, build.task_set.artifact_id).tasks
        if task.partition == "fit"
    )
    task = persist_evaluation_task(
        store.artifacts,
        EvaluationTask(
            intent="Serve the customer",
            task=fit.model_copy(update={"reference_answer": None}),
            world_model=build.world_model,
            checkpoint=build.world_model,
            verifier=CANDIDATE,
            challenges=(),
        ),
        created_at=NOW,
        code_revision="test",
    )
    held_out = sorted(held_out_lineages(store))
    spec_ref = specification(store)

    def pack(development_lineage: str, confirmed: list[str]) -> CalibrationPack:
        """Development on one lineage, confirmation cases spread over held-out lineages."""
        cases = [
            CalibrationCase(
                challenge=label(store, spec_ref, f"dev-{status}-{n}", outcome(status, f"d{n}")),
                lineage=development_lineage,
                partition="development",
            )
            for status in ("done", "pending")
            for n in (1, 2)
        ] + [
            CalibrationCase(
                challenge=label(store, spec_ref, f"conf-{n}", outcome(status, f"c{n}")),
                lineage=lineage,
                partition="confirmation",
            )
            for n, (lineage, status) in enumerate(
                zip(confirmed, ("done", "pending", "done", "pending"), strict=False)
            )
        ]
        return CalibrationPack(task=task, parent=PARENT, candidate=CANDIDATE, cases=tuple(cases))

    clean = admit_calibration_evidence(store, pack("fit-family", held_out))
    assert clean.refusals == ()
    assert clean.ledger is not None
    breached = admit_calibration_evidence(store, pack(held_out[0], held_out[1:]))
    assert {row.reason for row in breached.refusals} == {"sealed_partition_breach"}
    with audit(store, 1) as opened:
        reserved = set(opened.receipt.lineages)
        refusals = admit_calibration_evidence(store, pack("fit-family", held_out)).refusals
    assert reserved and reserved <= set(held_out)
    assert {(row.reason, row.subject) for row in refusals} == {
        ("contaminated_lineage", lineage) for lineage in reserved
    }
    assert all("reserved" in row.detail for row in refusals)
    # The audit's reservation stays until released; exposure is recorded on another lineage.
    shown = next(lineage for lineage in held_out if lineage not in reserved)
    ExposureLedger(store.paths.root).record_exposure(
        LedgerExposure(
            vertices=(held_out_members(store)[shown].vertices[0],),
            how="inspected",
            panel="held-out-panel-elsewhere",
            receipt="held-out-audit-receipt-elsewhere",
            report="exp eval --audit --verbose",
            project="elsewhere",
            at=NOW,
        )
    )
    exposed = admit_calibration_evidence(store, pack("fit-family", held_out)).refusals
    assert {(row.reason, row.subject, row.detail.split(" in ")[0]) for row in exposed} == {
        ("contaminated_lineage", shown, f"held-out lineage {shown} is exposed"),
        *(
            ("contaminated_lineage", lineage, f"held-out lineage {lineage} is reserved")
            for lineage in reserved
        ),
    }


def test_refinement_refuses_inadequate_or_self_approving_labels_before_spend(
    tmp_path: Path,
) -> None:
    """Paid refinement stops with named reasons when it could not qualify."""
    project, _, _ = native_label(tmp_path)
    world = load_world_checkpoint(
        project.artifacts,
        artifact_input(
            project.artifacts.read(
                project.artifacts.list_ids(artifact_type="world-checkpoint")[0]
            ).manifest
        ),
    ).world_model
    task = TaskCase(
        task_id="order",
        origin="authored",
        lineage_group_id="orders",
        partition="fit",
        instruction="Complete the order.",
        workload_weight=1,
    )
    cases = [case.challenge for case in witnessed_cases(project, specification(project))]
    verified = require_refinement_admission(
        project,
        verifier=PARENT,
        development=cases[:4],
        confirmation=cases[4:],
        world_model=world,
        task=task,
    )
    assert verified == {case.id for case in cases}
    with pytest.raises(EvidenceAdmissionError, match=r"\[inadequate_evidence\] development"):
        require_refinement_admission(
            project,
            verifier=PARENT,
            development=cases[1:4],
            confirmation=cases[4:],
            world_model=world,
            task=task,
        )
    with pytest.raises(EvidenceAdmissionError, match=r"\[self_approval\]"):
        require_refinement_admission(
            project,
            verifier=REFERENCE,
            development=cases[:4],
            confirmation=cases[4:],
            world_model=world,
            task=task,
        )


def test_refinement_refuses_development_labels_on_a_held_out_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refinement shows development labels to proposers, so a held-out task is sealed."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    build = store.load_project().build
    assert build is not None
    tasks = load_task_set(store.artifacts, build.task_set.artifact_id).tasks
    spec_ref = specification(store)
    development, confirmation = (
        [
            label(store, spec_ref, f"{name}-{status}-{n}", outcome(status, f"{name}{n}"))
            for status in ("done", "pending")
            for n in (1, 2)
        ]
        for name in ("dev", "conf")
    )

    def refine(task: TaskCase) -> frozenset[str]:
        """Run refinement admission for one built task over the same witnessed labels."""
        return require_refinement_admission(
            store,
            verifier=PARENT,
            development=development,
            confirmation=confirmation,
            world_model=build.world_model,
            task=task,
        )

    fit = next(task for task in tasks if task.partition == "fit")
    assert len(refine(fit)) == 8
    held_out = next(task for task in tasks if task.partition == "held_out")
    with pytest.raises(EvidenceAdmissionError, match=r"\[sealed_partition_breach\]") as error:
        refine(held_out)
    assert f"held-out evaluation lineage {held_out.lineage_group_id}" in str(error.value)


def test_rerunning_a_calibration_spec_refuses_a_stale_qualified_report(tmp_path: Path) -> None:
    """A refusal that appears after the first run is not hidden behind the saved report."""
    project, _, _ = native_label(tmp_path)
    spec = calibration(project, witnessed_cases(project, specification(project)))
    reference = execute_calibration(project, spec, code_revision="test")
    assert reference is not None
    report = json.loads(project.artifacts.read_bytes(reference.artifact_id, "report.json"))
    assert report["qualification_refusals"] == []
    assert execute_calibration(project, spec, code_revision="test") == reference
    project.artifacts.write_json(
        artifact_id="role-proposal",
        artifact_type="task-refinement-proposal",
        envelope=ArtifactEnvelope(schema_version=1, created_at=NOW, code_revision="test"),
        files={
            "proposal.json": {
                "verifier": {
                    "description": "A proposal that matches the reference specification.",
                    "rules": [{"path": "/final_answer", "operator": "equals", "value": "done"}],
                }
            }
        },
    )
    refused = admit_calibration_evidence(project, spec.pack).refusals
    assert refused and {row.reason for row in refused} == {"self_approval"}
    with pytest.raises(ValueError, match="was saved as .* re-run under a fresh run ID"):
        execute_calibration(project, spec, code_revision="test")

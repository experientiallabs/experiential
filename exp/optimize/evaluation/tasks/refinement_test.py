"""Task refinement keeps labels private and runs child actors through the existing owner."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.council import runner as trace_council
from exp.optimize.evaluation.council.runner_test import sequence, spec_project
from exp.optimize.evaluation.fidelity.runner import FidelitySpec
from exp.optimize.evaluation.tasks import refinement as task_refinement
from exp.optimize.evaluation.tests.fidelity_memory_test import structured_reply
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
    EvidenceAdmissionError,
    persist_specification,
    record_label_witness,
)
from exp.simulation.evaluation.tasks import (
    FieldRelation,
    OutcomeRule,
    RuleVerifierDraft,
    TaskVerifier,
    VerifierChallenge,
    load_evaluation_task,
)


def specification(project: ProjectStore, *, copied: bool = False) -> ArtifactInput:
    """Compare the observed current value with the result; optionally supply a deliberate copy."""
    return persist_specification(
        project.artifacts,
        EvaluationSpecification(
            description="Reference: the last value read must be the current one.",
            predicate=TaskVerifier(
                description="Reference specification for the current value.",
                outcome_schema={"type": "object"},
                rules=(OutcomeRule(path="/final_state/last", operator="equals", value="current"),)
                if copied
                else (),
                relations=()
                if copied
                else (FieldRelation(left="/final_state/last", right="/initial_state/current"),),
            ),
            subject="outcome",
        ),
        created_at=datetime.fromisoformat("2026-10-17T00:00:00Z"),
        code_revision="fixture-revision",
    )


def verifier(*, strict: bool) -> TaskVerifier:
    """Supply a weak parent and intent-aligned replacement without native claims."""
    return TaskVerifier(
        description="Synthetic specification: final last value must be current",
        outcome_schema={
            "type": "object",
            **(
                {
                    "properties": {
                        "final_state": {
                            "type": "object",
                            "properties": {"last": {"const": "current"}},
                            "required": ["last"],
                        }
                    },
                    "required": ["final_state"],
                }
                if strict
                else {}
            ),
        },
    )


def fit_council_project(
    tmp_path: Path,
) -> tuple[ProjectStore, trace_council.TraceCouncilSpec, FidelitySpec]:
    """Council over the fixture task as a fit task: refinement never sees held-out labels."""
    project, council, fidelity = spec_project(tmp_path)
    task = council.tasks[0].model_copy(update={"partition": "fit"})
    return project, council.model_copy(update={"tasks": (task,)}), fidelity


@pytest.mark.parametrize(
    "change_prompt,failed_judge,bad_verifier,hacker_succeeds,copied_reference",
    [
        (False, False, False, False, False),
        (True, False, False, False, False),
        (False, True, False, False, False),
        (False, False, True, False, False),
        (False, False, False, True, False),
        (False, False, False, False, True),
    ],
)
def test_refinement_keeps_confirmation_private_and_replays_without_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_prompt: bool,
    failed_judge: bool,
    bad_verifier: bool,
    hacker_succeeds: bool,
    copied_reference: bool,
) -> None:
    """Exercise source capture, proposing roles, child actors, grading and zero-call replay."""
    project, council, fidelity = fit_council_project(tmp_path)
    source_wire = ScriptedJsonTransport(sequence(fidelity))
    monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: source_wire)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    prepared_source = trace_council.prepare_trace_council(
        project, council, code_revision="fixture-revision"
    )
    trace_council.execute_trace_council(project, prepared_source)
    reference = specification(project, copied=copied_reference)

    def challenge(identity: str, value: str) -> VerifierChallenge:
        """Label with saved checks of an independent specification, not generated truth."""
        outcome: JsonObject = {
            "initial_state": {"current": "current"},
            "final_state": {"last": value},
            "partition_marker": identity,
        }
        return VerifierChallenge(
            id=identity,
            outcome=outcome,
            expected="pass" if value == "current" else "fail",
            authority="synthetic_specification",
            rationale="Declared synthetic specification",
            witness=record_label_witness(
                project.artifacts,
                specification=reference,
                outcome=outcome,
                created_at=fidelity.deadline,
                code_revision="fixture-revision",
            ),
        )

    spec = task_refinement.TaskRefinementSpec(
        run_id="refine-once",
        source_council=artifact_input(project.artifacts.read(prepared_source.id).manifest),
        intent="Read the current value accurately",
        verifier=verifier(strict=False),
        development_challenges=(
            challenge("dev-positive", "current"),
            challenge("dev-positive-2", "current"),
            challenge("dev-negative", "wrong"),
            challenge("dev-negative-2", "wrong"),
        ),
        confirmation_challenges=(
            challenge("SECRET_CONFIRM_POS", "current"),
            challenge("SECRET_CONFIRM_POS_2", "current"),
            challenge("SECRET_CONFIRM_NEG", "wrong"),
            challenge("SECRET_CONFIRM_NEG_2", "wrong"),
        ),
        author_alias="model",
        verifier_alias="model",
        hacker_alias="model",
        deadline=fidelity.deadline,
        maximum_physical_attempts=3,
        child_maximum_physical_attempts=12,
    )
    witness_ref = spec.development_challenges[2].witness
    assert witness_ref is not None
    negative_witness = witness_ref.artifact_id
    before = len(source_wire.requests)
    prepared = task_refinement.prepare_task_refinement(
        project, spec, code_revision="fixture-revision"
    )
    assert len(source_wire.requests) == before
    replies = [
        structured_reply(
            {
                "prompt": "Inspect current, then answer accurately." if change_prompt else None,
                "evidence_refs": ["task"],
                "reason": "Clarify the intended read without disclosing a target answer.",
            }
        ),
        structured_reply(
            {
                "verifier": RuleVerifierDraft(
                    description="Check the evidenced final value",
                    rules=(
                        OutcomeRule(
                            path="/final_state" if bad_verifier else "/final_state/last",
                            operator="exists" if bad_verifier else "equals",
                            value=None if bad_verifier else "current",
                        ),
                    ),
                ).model_dump(mode="json"),
                "evidence_refs": ["verifier", "challenge:dev-negative", negative_witness],
                "reason": "Reject the independently labeled wrong value.",
            }
        ),
        structured_reply(
            {
                "outcome": {"final_state": {"last": "current" if hacker_succeeds else "wrong"}},
                "rationale": "Wrong value must not earn reward.",
                "evidence_refs": ["verifier"],
            }
        ),
    ]
    role_wire = ScriptedJsonTransport(replies)
    child_responses = sequence(fidelity)
    if failed_judge:
        child_responses[3] = structured_reply({})
    child_wire = ScriptedJsonTransport(child_responses)
    monkeypatch.setattr(task_refinement, "HttpxJsonTransport", lambda: role_wire)
    monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: child_wire)
    select = project.select_evaluation_task
    if (
        not change_prompt
        and not failed_judge
        and not bad_verifier
        and not hacker_succeeds
        and not copied_reference
    ):

        def interrupted(reference: ArtifactInput) -> None:
            """Interrupt final publication after the durable report, before its selection."""
            raise RuntimeError("injected publication interruption")

        monkeypatch.setattr(project, "select_evaluation_task", interrupted)
        with pytest.raises(RuntimeError, match="injected publication"):
            task_refinement.execute_task_refinement(project, prepared)
        assert project.artifacts.exists(prepared.id + "-report")
        assert not project.load_project().evaluation_tasks
        monkeypatch.setattr(project, "select_evaluation_task", select)
        monkeypatch.delenv("FIDELITY_TEST_KEY")
        path = tmp_path / "refinement.json"
        path.write_text(spec.model_dump_json())
        monkeypatch.setattr(
            "exp.cli.evaluation.task_refinement.installed_release_revision",
            lambda: "fixture-revision",
        )
        replay = CliRunner().invoke(
            app,
            [
                "eval",
                "fidelity-test",
                "--root",
                str(tmp_path),
                "--refine-task",
                str(path),
                "--non-interactive",
            ],
        )
        assert replay.exit_code == 0, replay.output
        assert "Saved task refinement. No provider calls." in replay.output
        assert project.load_project().evaluation_tasks
    result = task_refinement.execute_task_refinement(project, prepared)
    if copied_reference:
        assert result["development_status"] == "incomplete"
        assert result["confirmation_status"] == "not_run"
        assert result["child_execution_status"] == "not_dispatched"
        assert result["child_physical_attempts"] == 0 and not child_wire.requests
        assert "self_approval" in json.dumps(result["role_failures"])
        calls = len(role_wire.requests)
        monkeypatch.delenv("FIDELITY_TEST_KEY", raising=False)
        assert task_refinement.execute_task_refinement(project, prepared) == result
        assert len(role_wire.requests) == calls
        monkeypatch.setattr(task_refinement, "_protocol", lambda: "0" * 64)
        with pytest.raises(ValueError, match="protocol changed"):
            task_refinement.prepare_task_refinement(project, spec, code_revision="fixture-revision")
        return
    assert result["development_status"] == (
        "rejected"
        if bad_verifier
        else "incomplete"
        if failed_judge
        else "needs_review"
        if change_prompt or hacker_succeeds
        else "accepted_development"
    )
    assert result["verifier_development_status"] == (
        "rejected" if bad_verifier else "accepted_development"
    )
    assert result["confirmation_status"] == (
        "not_run" if failed_judge or bad_verifier or hacker_succeeds else "passed"
    )
    assert result["gains"] == ([] if bad_verifier else ["dev-negative", "dev-negative-2"])
    assert result["losses"] == []
    assert result["role_physical_attempts"] == 3
    assert result["physical_attempts"] == (3 if bad_verifier else 12)
    assert result["child_physical_attempts"] == (0 if bad_verifier else 9)
    assert result["role_failures"] == {}
    for request in role_wire.requests:
        assert "SECRET_CONFIRM" not in json.dumps(request.payload)
    for index in () if bad_verifier else (0, 2, 4, 6):
        request = json.dumps(child_wire.requests[index].payload)
        assert "SECRET_CONFIRM" not in request
        assert "challenge:dev-negative" not in request
        assert "label-witness" not in request
        assert "evaluation-specification" not in request
    child_ref = artifact_input(project.artifacts.read(str(result["child_artifact_id"])).manifest)
    assert any(
        ref.artifact_id == prepared.id + "-report"
        for ref in project.load_project().evaluation_tasks
    )
    child = load_evaluation_task(project.artifacts, child_ref)
    assert child.intent == spec.intent and child.world_model == council.world_model
    assert child.parent == prepared.parent
    monkeypatch.delenv("FIDELITY_TEST_KEY", raising=False)
    assert task_refinement.execute_task_refinement(project, prepared) == result
    assert len(role_wire.requests) == 3
    assert len(child_wire.requests) == (0 if bad_verifier else 9)
    if (
        not change_prompt
        and not failed_judge
        and not bad_verifier
        and not hacker_succeeds
        and not copied_reference
    ):
        reused_spec = spec.model_copy(
            update={
                "run_id": "reuse-qualified-proposals",
                "prior_proposals": artifact_input(
                    project.artifacts.read(prepared.id + "-report").manifest
                ),
            }
        )
        monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
        reused_plan = task_refinement.prepare_task_refinement(
            project,
            reused_spec,
            code_revision="fixture-revision",
        )
        reused_role_wire = ScriptedJsonTransport([replies[-1]])
        reused_child_wire = ScriptedJsonTransport(sequence(fidelity))
        monkeypatch.setattr(task_refinement, "HttpxJsonTransport", lambda: reused_role_wire)
        monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: reused_child_wire)
        reused_result = task_refinement.execute_task_refinement(project, reused_plan)
        assert reused_result["development_status"] == "accepted_development"
        assert reused_result["role_physical_attempts"] == 1
        assert reused_result["physical_attempts"] == 10
        assert len(reused_role_wire.requests) == 1
        incompatible = reused_spec.model_copy(
            update={"run_id": "bad-reuse", "intent": "Other intent"}
        )
        with pytest.raises(ValueError, match="reused proposals differ"):
            task_refinement.prepare_task_refinement(
                project, incompatible, code_revision="fixture-revision"
            )
        project.model_catalog_path.unlink()
        restored_replay = CliRunner().invoke(
            app,
            [
                "eval",
                "fidelity-test",
                "--root",
                str(tmp_path),
                "--refine-task",
                str(path),
                "--non-interactive",
            ],
        )
        assert restored_replay.exit_code == 0, restored_replay.output
        assert "Saved task refinement. No provider calls." in restored_replay.output
        assert len(reused_role_wire.requests) == 1
        assert len(reused_child_wire.requests) == 9


def test_refinement_rejects_partition_overlap_before_dispatch() -> None:
    """The same labeled outcome cannot be exposed as development and hidden confirmation."""
    challenge = VerifierChallenge(
        id="one",
        outcome={"value": 1},
        expected="pass",
        authority="generated_hypothesis",
        rationale="Diagnostic only",
    )
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="overlap"):
        task_refinement.TaskRefinementSpec(
            run_id="bad",
            source_council=ArtifactInput(artifact_id="source", sha256="0" * 64),
            intent="intent",
            verifier=verifier(strict=False),
            development_challenges=(challenge,),
            confirmation_challenges=(challenge.model_copy(update={"id": "two"}),),
            author_alias="model",
            verifier_alias="model",
            hacker_alias="model",
            deadline=datetime.fromisoformat("2099-01-01T00:00:00Z"),
        )


@pytest.mark.parametrize("defect", ["self_approval", "inadequate_evidence"])
def test_refinement_refuses_unqualifiable_labels_before_any_paid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    """A generated-label chain or too few witnessed labels stop before registration or spend."""
    project, council, fidelity = fit_council_project(tmp_path)
    source_wire = ScriptedJsonTransport(sequence(fidelity))
    monkeypatch.setattr(trace_council, "HttpxJsonTransport", lambda: source_wire)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    prepared_source = trace_council.prepare_trace_council(
        project, council, code_revision="fixture-revision"
    )
    trace_council.execute_trace_council(project, prepared_source)
    reference = specification(project)
    if defect == "self_approval":
        proposal = artifact_input(
            project.artifacts.write_json(
                artifact_id="earlier-role-proposal",
                artifact_type="task-refinement-proposal",
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=fidelity.deadline, code_revision="fixture"
                ),
                files={"proposal.json": {"verifier": None}},
            )
        )
        stored = json.loads(
            project.artifacts.read_bytes(reference.artifact_id, "specification.json")
        )
        reference = persist_specification(
            project.artifacts,
            EvaluationSpecification.model_validate(stored),
            sources=(proposal,),
            created_at=fidelity.deadline,
            code_revision="fixture-revision",
        )

    def challenge(identity: str, value: str) -> VerifierChallenge:
        """One specification-witnessed label."""
        outcome: JsonObject = {
            "initial_state": {"current": "current"},
            "final_state": {"last": value},
            "partition_marker": identity,
        }
        return VerifierChallenge(
            id=identity,
            outcome=outcome,
            expected="pass" if value == "current" else "fail",
            authority="synthetic_specification",
            rationale="Declared synthetic specification",
            witness=record_label_witness(
                project.artifacts,
                specification=reference,
                outcome=outcome,
                created_at=fidelity.deadline,
                code_revision="fixture-revision",
            ),
        )

    copies = 1 if defect == "inadequate_evidence" else 2
    spec = task_refinement.TaskRefinementSpec(
        run_id="refused",
        source_council=artifact_input(project.artifacts.read(prepared_source.id).manifest),
        intent="Read the current value accurately",
        verifier=verifier(strict=False),
        development_challenges=tuple(
            challenge(f"dev-{value}-{n}", value)
            for value in ("current", "wrong")
            for n in range(copies)
        ),
        confirmation_challenges=tuple(
            challenge(f"confirm-{value}-{n}", value)
            for value in ("current", "wrong")
            for n in (0, 1)
        ),
        author_alias="model",
        verifier_alias="model",
        hacker_alias="model",
        deadline=fidelity.deadline,
    )
    before = len(source_wire.requests)
    with pytest.raises(EvidenceAdmissionError, match=rf"\[{defect}\]"):
        task_refinement.prepare_task_refinement(project, spec, code_revision="fixture-revision")
    assert len(source_wire.requests) == before
    assert not project.artifacts.list_ids(artifact_type="task-refinement-plan")

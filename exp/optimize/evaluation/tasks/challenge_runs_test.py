"""Challenge runs bind one pinned model per arm; the CLI run is in ``judge_challenges_test``."""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.evaluations.actor_rubric import actor_judge
from exp.common.evaluations.reconstruction import JudgeConfig, WorkflowEvidence, WorkflowStep
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.tasks import challenge_runs
from exp.optimize.evaluation.tasks.challenge_runs import (
    ChallengeArmModel,
    ChallengeRunSpec,
    challenge_attempts,
)
from exp.optimize.evaluation.tasks.challenges import (
    ChallengeJudge,
    ChallengeJudgment,
    DeclaredDefect,
    DeclaredInvariance,
    JudgeChallengeStudy,
    OutputEdit,
    Perturbation,
    record_challenge_study,
    run_challenge_study,
    saved_challenge_judgments,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.evaluation.calibration import (
    CalibrationCase,
    CalibrationJudgment,
    CalibrationPack,
)
from exp.simulation.evaluation.tasks import (
    EvaluationTask,
    FieldRelation,
    TaskVerifier,
    VerifierChallenge,
    persist_evaluation_task,
)
from exp.simulation.evaluation.tasks_test import challenges

CANDIDATE = TaskVerifier(
    description="Answer agrees with the observed reply",
    outcome_schema={"type": "object"},
    relations=(FieldRelation(left="/final_answer/status", right="/steps/0/reply/status"),),
)
JUDGE = actor_judge("fixture-v1", per_turn=False)


def evidence(case: str) -> WorkflowEvidence:
    """A saved workflow that read the status and reported it."""
    return WorkflowEvidence(
        workflow_id=case,
        task="Report the observed status",
        task_context={"actor": "model-a"},
        initial_state={},
        steps=(
            WorkflowStep(step=0, action={"name": "read"}, status="valid", reply={"status": "done"}),
        ),
        final_answer={"status": "done"},
        stop_reason="completed",
    )


def labeled(case: str, partition: str) -> CalibrationCase:
    """A human-approved valid saved workflow on its own lineage."""
    return CalibrationCase.model_validate(
        {
            "challenge": VerifierChallenge(
                id=case,
                outcome=evidence(case),
                expected="pass",
                authority="human",
                approved=True,
                rationale="Reviewed: the reported status is the observed one.",
                source_refs=(ArtifactInput(artifact_id="review-" + case, sha256="1" * 64),),
            ),
            "lineage": case,
            "partition": partition,
            "expected_task": "pass",
        }
    )


def run_spec(project: ProjectStore) -> ChallengeRunSpec:
    """One arm judging a saved output, an irrelevant variant and a declared defect."""
    witness = challenges(project)[0].witness
    assert witness is not None
    task = persist_evaluation_task(
        project.artifacts,
        EvaluationTask(
            intent="Report the observed status",
            task=TaskCase(
                task_id="status",
                origin="authored",
                lineage_group_id="status",
                partition="fit",
                instruction="Report the status.",
                workload_weight=1,
            ),
            world_model=witness,
            checkpoint=witness,
            verifier=CANDIDATE,
            challenges=challenges(project),
        ),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    pack = CalibrationPack(
        task=task,
        parent=TaskVerifier(description="Weak", outcome_schema={"type": "object"}),
        candidate=CANDIDATE,
        cases=(labeled("case-a", "development"), labeled("case-b", "confirmation")),
        judge=JUDGE,
        required_dimensions=("outcome",),
        minimum_scores={"outcome": 4},
    )
    study = JudgeChallengeStudy(
        pack=pack,
        invariances=(
            DeclaredInvariance(
                factor="actor_identity",
                pointers=("/task_context/actor",),
                rationale="Behavior is judged, not its author.",
            ),
        ),
        defects=(
            DeclaredDefect(
                kind="wrong_status",
                check="changed_value",
                pointers=("/final_answer",),
                rationale="The reported status must be the observed one.",
            ),
        ),
        perturbations=(
            Perturbation(
                perturbation_id="case-a-author",
                base_case_id="case-a",
                factor="actor_identity",
                edits=(OutputEdit(pointer="/task_context/actor", value="model-b"),),
                description="Another actor produced the same workflow.",
            ),
            Perturbation(
                perturbation_id="case-a-wrong",
                base_case_id="case-a",
                factor="wrong_status",
                edits=(OutputEdit(pointer="/final_answer/status", value="pending"),),
                description="A status the workflow never observed.",
            ),
        ),
        judges=(ChallengeJudge(arm_id="fixture", config=JUDGE),),
    )
    catalog = load_model_catalog(project.model_catalog_path)
    model, _ = RuntimeModelCatalog(catalog, environment={}).snapshot("model")
    return ChallengeRunSpec(
        study=study,
        models=(ChallengeArmModel(arm_id="fixture", alias="model", model=model),),
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )


def judged(score: int) -> JsonHttpResponse:
    """A judge reply that scores every criterion ``score`` with visible evidence."""
    axes: JsonObject = {
        dimension.dimension_id: {
            "status": "assessed",
            "score": score,
            "rationale": "Compared the answer with the observed reply.",
            "evidence_refs": ["workflow-0"],
            "critical_flags": [],
        }
        for dimension in JUDGE.definition.dimensions
    }
    return JsonHttpResponse(
        status_code=200,
        body={
            "id": f"judge-{score}",
            "model": "fixture-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "answer",
                                "type": "function",
                                "function": {
                                    "name": "submit_proposal",
                                    "arguments": json.dumps({"axes": axes}),
                                },
                            }
                        ],
                    },
                }
            ],
        },
    )


def test_every_arm_needs_exactly_its_judge_model(tmp_path: Path) -> None:
    """A missing arm, an extra binding or another model identity refuses at registration."""
    project, _ = source_project(tmp_path)
    spec = run_spec(project)
    assert spec.maximum_physical_attempts == 6 and challenge_attempts(spec) == {"model": 6}
    (cost,) = evaluation_costs(project, spec)
    assert (cost.name, cost.attempts) == ("model", 6)
    with pytest.raises(ValidationError, match="exactly one model to every judge arm"):
        ChallengeRunSpec.model_validate({**spec.model_dump(), "models": spec.models * 2})
    other = spec.models[0].model.model_copy(update={"model_id": "other-v1"})
    with pytest.raises(ValidationError, match="judges with fixture-v1, but model serves other-v1"):
        ChallengeRunSpec.model_validate(
            {**spec.model_dump(), "models": ({**spec.models[0].model_dump(), "model": other},)}
        )
    assert spec.identity != spec.model_copy(update={"output_tokens": 1024}).identity


def test_completed_replay_requires_the_exact_run_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed study cannot certify another run, catalog, snapshot or allowance."""
    project, _ = source_project(tmp_path)
    spec = run_spec(project)
    transport = ScriptedJsonTransport([judged(5), judged(5), judged(1)])
    monkeypatch.setattr(challenge_runs, "HttpxJsonTransport", lambda: transport)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    reference = challenge_runs.run_challenge(project, spec, code_revision="test")
    changed_model = spec.models[0].model.model_copy(update={"revision": "another-revision"})
    variants = (
        spec.model_copy(update={"output_tokens": 1024}),
        spec.model_copy(update={"attempts_per_judgment": 1}),
        spec.model_copy(update={"deadline": spec.deadline + timedelta(minutes=1)}),
        spec.model_copy(update={"catalog_sha256": "9" * 64}),
        spec.model_copy(
            update={"models": (spec.models[0].model_copy(update={"model": changed_model}),)}
        ),
    )
    for changed in variants:
        assert changed.identity != spec.identity
        assert challenge_runs.saved_challenge_run(project, changed) is None
    rows = saved_challenge_judgments(project, spec.study, artifact_id=reference.artifact_id)
    assert rows is not None
    record_challenge_study(project, spec.study, rows, code_revision="test")
    assert challenge_runs.saved_challenge_run(project, variants[0]) is None
    monkeypatch.delenv("FIDELITY_TEST_KEY")
    assert challenge_runs.run_challenge(project, spec, code_revision="test") == reference
    assert len(transport.requests) == 3


def test_interrupted_judgments_replay_only_within_the_exact_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another allowance judges afresh; resuming the same run keeps its saved judgment."""
    project, _ = source_project(tmp_path)
    spec = run_spec(project)
    transport = ScriptedJsonTransport([judged(5)] * 6)
    monkeypatch.setattr(challenge_runs, "HttpxJsonTransport", lambda: transport)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")

    def interrupted(
        study: JudgeChallengeStudy,
        judge: Callable[[JudgeConfig, JsonObject, str], CalibrationJudgment],
    ) -> tuple[ChallengeJudgment, ...]:
        """Stop after saving the first physical judgment, before publishing the study."""

        def stop_after_first(
            config: JudgeConfig, blinded: JsonObject, key: str
        ) -> CalibrationJudgment:
            """Save one normal evaluator result and interrupt its owning run."""
            registered = project.artifacts.read_bytes(spec.identity + "-registration", "run.json")
            assert ChallengeRunSpec.model_validate_json(registered) == spec
            judge(config, blinded, key)
            raise InterruptedError("after the first saved judgment")

        return run_challenge_study(study, stop_after_first)

    with monkeypatch.context() as patch:
        patch.setattr(challenge_runs, "run_challenge_study", interrupted)
        with pytest.raises(InterruptedError, match="after the first saved judgment"):
            challenge_runs.run_challenge(project, spec, code_revision="test")
    assert len(transport.requests) == 1
    assert challenge_runs.saved_challenge_run(project, spec) is None
    changed = spec.model_copy(update={"output_tokens": 1024})
    alternate = challenge_runs.run_challenge(project, changed, code_revision="test")
    assert len(transport.requests) == 4
    original = challenge_runs.run_challenge(project, spec, code_revision="test")
    assert original != alternate
    assert len(transport.requests) == 6

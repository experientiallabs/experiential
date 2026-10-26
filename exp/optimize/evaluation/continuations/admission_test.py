"""Independent acceptance refuses self-approval and retains failed native workbook branches.

Every admission applies a registered policy; per-dimension verdicts stay with each branch.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.models import AssistantAction
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.admission import (
    ContinuationAdmission,
    admit_continuations,
    require_pinned_policy,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import (
    ContinuationStudySpec,
    run_continuation_study,
)
from exp.optimize.evaluation.continuations.study_test import probe, workbook_runtime
from exp.optimize.evaluation.equivalence import policy as policy_module
from exp.optimize.evaluation.equivalence.policy import AdmissionPolicy
from exp.optimize.evaluation.equivalence.policy_test import fixture_policy, registered_policy
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.simulation.continuations import persist_continuation_checkpoint
from exp.simulation.continuations_test import capsule
from exp.simulation.evaluation.admission import EvaluationSpecification, persist_specification
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier


def admitted_workbook_study(
    project: ProjectStore,
    *,
    self_approval: bool = False,
    generated: bool = False,
    registered: bool = True,
    narrated_action: bool = False,
    followup_action: bool = False,
    actor_actions: tuple[AssistantAction, ...] | None = None,
    policy_only_schema: JsonObject | None = None,
    policy: AdmissionPolicy | None = None,
) -> ArtifactInput:
    """Execute two native branches against independently specified workbook acceptance.

    The admission policy is registered first and pinned in the study it judges.
    """
    store = project.artifacts
    policy_ref = registered_policy(store, policy)
    checkpoint, actor = capsule(store)
    reference = persist_continuation_checkpoint(
        store, checkpoint, created_at=datetime.now(UTC), code_revision="test"
    )
    verifier = TaskVerifier(
        description="Delete one data row.",
        outcome_schema={},
        rules=(OutcomeRule(path="/final_state/rows", operator="equals", value=6),),
        allowed_tools=("edit_workbook",),
        preserved_paths=("/input_digest",),
    )
    independent = (
        verifier
        if self_approval
        else TaskVerifier(
            description="Authored XLSX acceptance with independent schema and preservation checks.",
            outcome_schema={
                "properties": {
                    "final_state": {
                        "type": "object",
                        "properties": {"rows": {"const": 6}},
                        "required": ["rows"],
                    }
                },
                "required": ["final_state"],
            },
            allowed_tools=("edit_workbook",),
            preserved_paths=("/input_digest",),
        )
    )
    if policy_only_schema is not None:
        independent = TaskVerifier(
            description="Tool policy alone supplies no task success evidence.",
            outcome_schema=policy_only_schema,
            allowed_tools=("edit_workbook",),
        )
    specification = persist_specification(
        store,
        EvaluationSpecification(
            description="Conformance input has seven rows; permitted deletion leaves six.",
            predicate=independent,
            subject="outcome",
        ),
        sources=(reference,) if generated else checkpoint.sources,
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    spec = ContinuationStudySpec(
        checkpoint=reference,
        actor_model=_response(AssistantAction(content="done")).model,
        verifier=verifier,
        probe_sha256=sha256_json({"probe": "native-row-count-v1"}),
        repeats=2,
        selector="all initial checkpoints",
        training_specification=specification if registered else None,
        admission_policy=policy_ref,
    )
    actions = actor_actions or (
        AssistantAction(
            content="The output has zero rows." if narrated_action else None,
            tool_calls=(workbook_case().first,),
        ),
        *(
            (
                AssistantAction(
                    tool_calls=(workbook_case().first.model_copy(update={"call_id": "repeat"}),)
                ),
            )
            if followup_action
            else ()
        ),
    )
    run_continuation_study(
        store,
        spec,
        actor=actor,
        model=_Model(
            (
                *(_response(action) for action in actions),
                _response(AssistantAction(content="The output has zero rows.")),
                _response(AssistantAction(content="unsupported completion")),
            )
        ),
        world=workbook_runtime(checkpoint.world),
        probe=probe,
        probe_sha256=spec.probe_sha256,
        code_revision="test",
    )
    result = next(
        key for key in store.list_ids(artifact_type="continuation-study") if key.endswith("-result")
    )
    return admit_continuations(
        store,
        artifact_input(store.read(result).manifest),
        policy=policy_ref,
        created_at=datetime.now(UTC),
        code_revision="test",
    )


@pytest.mark.parametrize(
    "self_approval,generated,registered,accepted",
    [
        (False, False, True, 1),
        (True, False, True, 0),
        (False, True, True, 0),
        (False, False, False, 0),
    ],
)
def test_only_independent_registered_acceptance_admits_branches(
    tmp_path: Path, self_approval: bool, generated: bool, registered: bool, accepted: int
) -> None:
    """A pass, copied predicate or generated source cannot manufacture its own authority."""
    project = ProjectStore(tmp_path, "case")
    ref = admitted_workbook_study(
        project, self_approval=self_approval, generated=generated, registered=registered
    )
    report = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(ref.artifact_id, "admission.json")
    )
    assert len(report.branches) == 2
    assert sum(row.admitted for row in report.branches) == accepted
    assert not report.branches[1].admitted
    assert "goal or policy" in report.branches[1].reason


@pytest.mark.parametrize("schema", [{}, {"type": "object", "description": "success"}])
def test_policy_only_specification_cannot_admit_a_passing_trajectory(
    tmp_path: Path, schema: JsonObject
) -> None:
    """An independently saved allowlist does not independently check task accomplishment."""
    project = ProjectStore(tmp_path, "case")
    reference = admitted_workbook_study(project, policy_only_schema=schema)
    report = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "admission.json")
    )
    assert not any(branch.admitted for branch in report.branches)
    assert "outcome predicate" in report.branches[0].reason


def test_branch_admission_records_every_dimension_under_the_registered_policy(
    tmp_path: Path,
) -> None:
    """The passing native branch is admitted on every gating dimension; the failure is kept."""
    project = ProjectStore(tmp_path, "case")
    report = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(admitted_workbook_study(project).artifact_id, "admission.json")
    )
    assert report.policy == report.branches[0].policy.policy
    passed, failed = (row.policy for row in report.branches)
    assert passed.status == "admitted" and not passed.simulated
    assert {row.dimension: row.status for row in passed.dimensions} == {
        "goal_success": "admitted",
        "policy_compliance": "admitted",
        "forbidden_effects": "admitted",
        "state_fidelity": "admitted",
        "tool_validity": "admitted",
        "recovery": "unassessable",
    }
    assert failed.status == "refused" and failed.reasons[0].startswith("goal_success refused")


def test_a_required_dimension_that_does_not_apply_is_never_silently_passed(
    tmp_path: Path,
) -> None:
    """Requiring recovery on a trajectory without errors withholds admission."""
    project = ProjectStore(tmp_path, "case")
    reference = admitted_workbook_study(
        project, policy=fixture_policy(gates={"recovery": "required"})
    )
    report = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "admission.json")
    )
    assert not any(row.admitted for row in report.branches)
    assert report.branches[0].reason.startswith("admission policy unassessable: recovery")


def test_a_policy_registered_after_the_study_cannot_admit_it(tmp_path: Path) -> None:
    """Margins chosen after sampling refuse, even when the late policy's clock claims earlier."""
    project = ProjectStore(tmp_path, "case")
    admission = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(admitted_workbook_study(project).artifact_id, "admission.json")
    )
    late = registered_policy(
        project.artifacts, fixture_policy(margin=0.4), at=datetime.now(UTC) - timedelta(days=1)
    )
    with pytest.raises(ValueError, match="only the policy a study pins"):
        admit_continuations(
            project.artifacts,
            admission.study,
            policy=late,
            created_at=datetime.now(UTC),
            code_revision="test",
        )


def test_only_the_pinned_policy_admits_even_when_another_is_reached_by_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A policy the registration reaches through other evidence never stands in for the pin."""
    project = ProjectStore(tmp_path, "case")
    admission = ContinuationAdmission.model_validate_json(
        project.artifacts.read_bytes(admitted_workbook_study(project).artifact_id, "admission.json")
    )
    other = registered_policy(project.artifacts, fixture_policy(margin=0.4))
    # As if a comparison cited by the study's training evidence were registered under ``other``.
    monkeypatch.setattr(policy_module, "require_cites", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match=f"pins {admission.policy.artifact_id}, not "):
        admit_continuations(
            project.artifacts,
            admission.study,
            policy=other,
            created_at=datetime.now(UTC),
            code_revision="test",
        )
    study = load_study(project.artifacts, admission.study)
    unpinned = study.model_copy(
        update={"spec": study.spec.model_copy(update={"admission_policy": None})}
    )
    with pytest.raises(ValueError, match="pins no admission policy"):
        require_pinned_policy(project.artifacts, admission.study, unpinned, other)

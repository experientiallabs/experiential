"""Independent acceptance refuses self-approval and retains failed native workbook branches."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.models import AssistantAction, ModelMessage
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuation_admission import (
    ContinuationAdmission,
    admit_continuations,
)
from exp.optimize.evaluation.continuation_study import ContinuationStudySpec, run_continuation_study
from exp.optimize.evaluation.continuation_study_test import probe, workbook_runtime
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
) -> ArtifactInput:
    """Execute two native branches against independently specified workbook acceptance."""
    store = project.artifacts
    checkpoint, actor = capsule(store)
    checkpoint = checkpoint.model_copy(
        update={
            "actor": checkpoint.actor.model_copy(
                update={
                    "messages": (
                        *checkpoint.actor.messages,
                        ModelMessage(role="assistant", content="Earlier mistaken hypothesis."),
                        ModelMessage(role="user", content="Recheck before acting."),
                    ),
                }
            )
        }
    )
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

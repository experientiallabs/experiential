"""Provider recovery cannot masquerade as a semantic reconstruction improvement."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.evaluations.reconstruction import (
    EvaluationReferences,
    ReferenceStep,
    WorkflowEvidence,
    WorkflowStep,
    default_judge,
    evaluate,
)
from exp.common.project import ProjectConfig, ProjectStore
from exp.simulation.reconstruction.models import RawEpisode, RawInputs, ReconstructionBundle
from exp.simulation.reconstruction.revision import (
    ProtectedCase,
    RevisionEvaluation,
    RevisionPlan,
    _gate,
    _outcomes,
)


@pytest.mark.parametrize("status", ["failure", "valid"])
def test_semantic_gain_requires_a_valid_parent_prediction(
    tmp_path: Path,
    status: Literal["failure", "valid"],
) -> None:
    """Keep both failures and wrong replies in scores, but only wrong predictions in gain tests."""
    project = ProjectStore(tmp_path, "revision")
    project.initialize(ProjectConfig(project_id="revision"))
    raw = RawInputs(episodes=(RawEpisode(id="fixture", records=({},)),))
    bundle = ReconstructionBundle.model_validate(
        {
            "artifact_id": "parent",
            "sources": raw.model_dump(mode="json"),
            "inputs_sha256": sha256_json(raw.model_dump(mode="json")),
            "interface": {
                "tools": [
                    {
                        "name": "read",
                        "description": "Read a value",
                        "parameters": {"type": "object"},
                        "reply_schema": {"type": "integer"},
                    }
                ],
                "state_schema": {"type": "object"},
            },
            "text_rules": "Fixture hypothesis.",
            "claims": [
                {
                    "statement": "Behavior remains unknown",
                    "status": "unknown",
                    "scope": "fixture",
                    "origin": "model_prior",
                }
            ],
        }
    )
    action: JsonObject = {"name": "read", "arguments": {}}
    evidence = WorkflowEvidence(
        workflow_id="case",
        knowledge_sha256=bundle.digest,
        task="Read a value",
        initial_state={},
        steps=(WorkflowStep(step=0, action=action, status=status, before={}, after={}, reply=1),),
    )
    references = EvaluationReferences(
        capability="recorded_reference_only",
        steps=(
            ReferenceStep(
                step=0,
                action=action,
                before={},
                reply_available=True,
                reply=0,
                state_available=True,
                after={},
            ),
        ),
    )
    judge = default_judge("scripted")

    def complete(_: JsonObject, identity: str) -> JsonObject:
        """Retain explicit unassessable ratings rather than asserting synthetic calibration."""
        del identity
        return {
            "axes": [
                {
                    "dimension_id": axis.dimension_id,
                    "status": "unassessable",
                    "score": None,
                    "rationale": "Synthetic selection fixture.",
                    "evidence_refs": [],
                    "critical_flags": [],
                }
                for axis in judge.definition.dimensions
            ]
        }

    report = evaluate(
        evidence,
        references,
        judge=judge,
        complete=complete,
        artifacts=project.artifacts,
        artifact_id="parent-evaluation",
        clock=lambda: datetime.now(UTC),
        code_revision="fixture",
    )
    case = ProtectedCase(
        case_id="case",
        planned_steps=1,
        references_sha256=sha256_json(references.model_dump(mode="json")),
        actions_sha256=sha256_json([action]),
    )
    before = _outcomes(
        RevisionEvaluation(
            knowledge_sha256=bundle.digest, evidence=evidence, references=references, report=report
        ),
        bundle,
        case,
    )
    assert report.mechanical["workflow_joint_exact"] is False
    plan = RevisionPlan(
        run_id="fixture",
        parent_sha256=bundle.digest,
        inputs_sha256=bundle.inputs_sha256,
        cases=(case,),
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )
    after: dict[str, bool | None] = dict.fromkeys(before, True)
    selected, gained, _ = _gate(before, after, plan, judge_ok=True)
    if status == "failure":
        assert before["case/0/trajectory_reply_exact"] is None
        assert not selected and not gained
    else:
        assert before["case/0/trajectory_reply_exact"] is False
        assert selected and gained


def test_claim_replacement_is_bound_to_exact_parent_claim() -> None:
    """A semantic correction replaces a contradicted claim instead of retaining both."""
    from exp.simulation.reconstruction.models import Claim
    from exp.simulation.reconstruction.revision import (
        ClaimReplacement,
        RevisionPatch,
        _revised_claims,
    )

    old = Claim(
        statement="Every invalid reason wins.",
        status="inferred",
        scope="cancel",
        origin="model_prior",
    )
    new = old.model_copy(update={"statement": "Ordering outside observed cases is unknown."})
    patch = RevisionPatch(
        old_text="old",
        new_text="new",
        replace_claims=(
            ClaimReplacement(
                index=0,
                expected_sha256=sha256_json(old.model_dump(mode="json")),
                replacement=new,
            ),
        ),
    )
    assert _revised_claims((old,), patch) == (new,)
    with pytest.raises(ValueError, match="parent claim"):
        _revised_claims((new,), patch)

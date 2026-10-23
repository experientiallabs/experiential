"""Dual evaluation replays saved judgments and preserves unavailable measurements."""

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, JsonValue, sha256_json
from exp.common.evaluations.reconstruction import (
    EvaluationReferences,
    ReferenceStep,
    WorkflowEvidence,
    WorkflowStep,
    default_judge,
    evaluate,
    field_differences,
    judge_wire_schema,
    mechanical_checks,
    reply_fields,
    turn_outcomes,
)
from exp.common.project import ArtifactManifest, ProjectConfig, ProjectStore

NOW = datetime(2030, 1, 1, tzinfo=UTC)


def evidence() -> WorkflowEvidence:
    """Provide one complete synthetic read, independent of a provider response."""
    return WorkflowEvidence(
        workflow_id="read",
        task="Read items",
        initial_state={"items": []},
        steps=(
            WorkflowStep(
                step=0,
                action={"name": "read"},
                status="valid",
                reply=[],
                before={"items": []},
                after={"items": []},
            ),
        ),
    )


def references() -> EvaluationReferences:
    """Provide a separate saved reference without claiming benchmark qualification."""
    return EvaluationReferences(
        capability="recorded_reference_only",
        steps=(
            ReferenceStep(
                step=0,
                action={"name": "read"},
                before={"items": []},
                reply_available=True,
                reply=[],
                state_available=True,
                after={"items": []},
            ),
        ),
    )


def test_saved_judge_response_recovers_after_publication_crash_without_repurchase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crash after paid output retention, then publish and replay the same result once."""
    project = ProjectStore(tmp_path, "evaluation")
    project.initialize(ProjectConfig(project_id="evaluation"))
    judge = default_judge("scripted")
    calls: list[str] = []

    def complete(_: JsonObject, identity: str) -> JsonObject:
        """Return only protocol-valid ratings; this test establishes persistence, not fidelity."""
        calls.append(identity)
        return {
            "axes": [
                {
                    "dimension_id": axis.dimension_id,
                    "status": "unassessable",
                    "score": None,
                    "rationale": "Synthetic persistence fixture, no human calibration.",
                    "evidence_refs": [],
                    "critical_flags": [],
                }
                for axis in judge.definition.dimensions
            ]
        }

    original = project.artifacts.write_json

    def crash(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, JsonValue],
    ) -> ArtifactManifest:
        """Fail only final publication, after the response artifact has been saved."""
        if artifact_type == "workflow-dual-evaluation":
            raise RuntimeError("publication interrupted")
        return original(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            envelope=envelope,
            files=files,
        )

    store = project.artifacts
    monkeypatch.setattr(store, "write_json", crash)
    with pytest.raises(RuntimeError, match="publication interrupted"):
        evaluate(
            evidence(),
            references(),
            judge=judge,
            complete=complete,
            artifacts=store,
            artifact_id="evaluation",
            clock=lambda: NOW,
            code_revision="fixture",
        )
    monkeypatch.setattr(store, "write_json", original)
    result = evaluate(
        evidence(),
        references(),
        judge=judge,
        complete=complete,
        artifacts=store,
        artifact_id="evaluation",
        clock=lambda: NOW,
        code_revision="fixture",
    )
    assert (
        evaluate(
            evidence(),
            references(),
            judge=judge,
            complete=complete,
            artifacts=store,
            artifact_id="evaluation",
            clock=lambda: NOW,
            code_revision="fixture",
        )
        == result
    )
    assert calls == ["evaluation"]
    assert result.mechanical["workflow_joint_exact"] is True
    assert result.judge_status == "completed"
    changed = evidence().model_copy(update={"task": "Changed task"})
    with pytest.raises(ValueError, match="identity differs"):
        evaluate(
            changed,
            references(),
            judge=judge,
            complete=complete,
            artifacts=store,
            artifact_id="evaluation",
            clock=lambda: NOW,
            code_revision="fixture",
        )
    assert calls == ["evaluation"]


def test_unexecuted_continuity_is_unavailable() -> None:
    """Zero observations cannot establish state continuity."""
    blocked = evidence().model_copy(
        update={
            "steps": (WorkflowStep(step=0, action={"name": "read"}, status="blocked"),),
        }
    )
    measured = mechanical_checks(blocked, references())
    assert measured["carried_continuity"] is None
    assert measured["continuity_checked_steps"] == 0


def test_interrupted_judge_dispatch_stays_unknown_without_a_second_purchase(tmp_path: Path) -> None:
    """A crash after intent cannot be called undispatched or resumed with a fresh request."""
    project = ProjectStore(tmp_path, "interrupted")
    project.initialize(ProjectConfig(project_id="interrupted"))
    calls: list[str] = []

    def interrupted(_: JsonObject, identity: str) -> JsonObject:
        """Simulate process interruption after dispatch may have reached a provider."""
        calls.append(identity)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluate(
            evidence(),
            references(),
            judge=default_judge("scripted"),
            complete=interrupted,
            artifacts=project.artifacts,
            artifact_id="interrupted",
            clock=lambda: NOW,
            code_revision="fixture",
        )
    report = evaluate(
        evidence(),
        references(),
        judge=default_judge("scripted"),
        complete=interrupted,
        artifacts=project.artifacts,
        artifact_id="interrupted",
        clock=lambda: NOW,
        code_revision="fixture",
    )
    assert calls == ["interrupted"]
    assert report.judge_status == "unknown"
    assert all(axis.status == "judge_unknown" for axis in report.axes)


def test_keyed_judge_protocol_carries_mechanical_findings_and_fixed_inventory(
    tmp_path: Path,
) -> None:
    """A new wire profile owns criterion IDs and exposes actual saved discrepancies."""
    import json

    project = ProjectStore(tmp_path, "keyed")
    project.initialize(ProjectConfig(project_id="keyed"))
    judge = default_judge("scripted", protocol="keyed-v2", mechanical_feedback=True)
    bad = evidence().model_copy(
        update={"steps": (evidence().steps[0].model_copy(update={"reply": ["invented"]}),)}
    )

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Use a fixed keyed response, independent of ordering and repeated ID copying."""
        messages = body["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        payload = json.loads(str(messages[1]["content"]))
        assert payload["workflow"]["mechanical_findings"]["first_semantic_divergence"] == 0
        return {
            "axes": {
                axis.dimension_id: {
                    "status": "assessed",
                    "score": 2,
                    "rationale": "The actual reply differs from the recorded reference.",
                    "evidence_refs": ["step-0", "reference-step-0"],
                    "critical_flags": ["wrong_reply"],
                }
                for axis in judge.definition.dimensions
            }
        }

    report = evaluate(
        bad,
        references(),
        judge=judge,
        complete=complete,
        artifacts=project.artifacts,
        artifact_id="keyed",
        clock=lambda: NOW,
        code_revision="test",
    )
    assert report.judge_status == "completed"
    assert report.mechanical["workflow_joint_exact"] is False
    assert tuple(axis.dimension_id for axis in report.axes) == tuple(
        axis.dimension_id for axis in judge.definition.dimensions
    )


def test_keyed_judge_cannot_drop_an_axis(tmp_path: Path) -> None:
    """A keyed envelope does not excuse missing criteria or synthesize successful ratings."""
    project = ProjectStore(tmp_path, "missing-axis")
    project.initialize(ProjectConfig(project_id="missing-axis"))
    report = evaluate(
        evidence(),
        references(),
        judge=default_judge("scripted", protocol="keyed-v2"),
        complete=lambda *_: {"axes": {}},
        artifacts=project.artifacts,
        artifact_id="missing-axis",
        clock=lambda: NOW,
        code_revision="test",
    )
    assert report.judge_status == "failed"
    assert len(report.axes) == 5
    assert all(axis.status == "judge_failure" for axis in report.axes)


def test_capacity_refusal_is_not_a_dispatched_judge_failure(tmp_path: Path) -> None:
    """Direct and actor evaluation retain the same explicit pre-dispatch status."""
    from exp.common.models.client import RequestCapacityExceeded

    project = ProjectStore(tmp_path, "capacity")
    project.initialize(ProjectConfig(project_id="capacity"))

    def refused(body: JsonObject, identity: str) -> JsonObject:
        """Refuse before calling a provider; no hidden capacity retry."""
        raise RequestCapacityExceeded("declared context capacity exceeded before dispatch")

    report = evaluate(
        evidence(),
        references(),
        judge=default_judge("scripted"),
        complete=refused,
        artifacts=project.artifacts,
        artifact_id="capacity",
        clock=lambda: NOW,
        code_revision="test",
    )
    assert report.judge_status == "not_run"
    assert len(report.axes) == 5 and all(axis.status == "not_run" for axis in report.axes)


@pytest.mark.parametrize("protocol", ["ordered-v1", "keyed-v2"])
def test_judge_wire_format_does_not_enable_mechanical_feedback(
    tmp_path: Path,
    protocol: str,
) -> None:
    """An envelope change preserves input authority and blinded provenance."""
    import json

    from exp.common.evaluations.reconstruction import IndependentCheck, JudgeConfig

    project = ProjectStore(tmp_path, "blinding")
    project.initialize(ProjectConfig(project_id="blinding"))
    config = default_judge("scripted").model_dump(mode="json")
    config["protocol"] = protocol
    judge = JudgeConfig.model_validate(config)
    refs = references().model_copy(
        update={
            "independent_checks": (
                IndependentCheck(name="goal", passed=False, reference_id="secret-provider-route"),
            )
        }
    )
    requests: list[JsonObject] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Retain input even though this synthetic reviewer declines assessment."""
        requests.append(body)
        return {"axes": {}}

    evaluate(
        evidence(),
        refs,
        judge=judge,
        complete=complete,
        artifacts=project.artifacts,
        artifact_id="no-feedback",
        clock=lambda: NOW,
        code_revision="test",
    )
    first = json.dumps(requests[0])
    assert "mechanical_findings" not in first
    assert "secret-provider-route" not in first
    evaluate(
        evidence(),
        refs,
        judge=judge.model_copy(update={"mechanical_feedback": True}),
        complete=complete,
        artifacts=project.artifacts,
        artifact_id="feedback",
        clock=lambda: NOW,
        code_revision="test",
    )
    second = json.dumps(requests[1])
    assert "mechanical_findings" in second and "secret-provider-route" not in second


def test_reply_only_reference_does_not_invent_hidden_state() -> None:
    """Recorded observations support carried reply checks without a full native prestate."""
    evidence = WorkflowEvidence(
        workflow_id="reply-only",
        task="Read a record",
        initial_state={},
        steps=(WorkflowStep(step=0, action={"name": "read"}, status="valid", reply={"id": "a"}),),
    )
    references = EvaluationReferences(
        capability="recorded_reference_only",
        description="Archived reply only",
        steps=(
            ReferenceStep(step=0, action={"name": "read"}, reply_available=True, reply={"id": "a"}),
        ),
    )
    report = mechanical_checks(evidence, references)
    rows = report["steps"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    row = rows[0]
    assert row["trajectory_reply_exact"] is True
    assert row["reply_exact"] is None and row["state_exact"] is None
    assert report["workflow_joint_exact"] is None


def test_strict_keyed_judge_requires_the_score_its_parser_needs() -> None:
    """Schema and parser agree that assessed criteria cannot omit numeric ratings."""
    config = default_judge("glm", protocol="keyed-v3")
    schema = judge_wire_schema(config)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    axis = properties["axes"]
    assert isinstance(axis, dict)
    axes = axis["properties"]
    assert isinstance(axes, dict)
    for value in axes.values():
        assert isinstance(value, dict) and isinstance(value["required"], list)
        assert "score" in value["required"]
    assert sha256_json(config) != sha256_json(default_judge("glm", protocol="keyed-v2"))


def test_reply_fields_count_leaf_agreement_over_both_replies() -> None:
    """Missing, extra and retyped leaves all count as divergent fields."""
    actual: JsonValue = {"ok": True, "items": [{"color": "black", "price": 1}], "note": []}
    expected: JsonValue = {"ok": True, "items": [{"color": "red", "price": 1.0, "size": "L"}]}
    assert reply_fields(actual, expected) == (1, 5)
    assert reply_fields(expected, expected) == (4, 4)
    assert reply_fields(None, None) == (1, 1)


def test_per_turn_scores_come_from_the_same_single_judge_call(tmp_path: Path) -> None:
    """One request carries the per-step inventory; scores land beside field divergence."""
    import json

    project = ProjectStore(tmp_path, "turns")
    project.initialize(ProjectConfig(project_id="turns"))
    judge = default_judge("scripted", protocol="keyed-v3").model_copy(update={"per_turn": True})
    ids = [axis.dimension_id for axis in judge.definition.dimensions]
    bad = evidence().model_copy(
        update={"steps": (evidence().steps[0].model_copy(update={"reply": ["invented"]}),)}
    )
    calls: list[JsonObject] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Score the workflow and its single step in one response."""
        calls.append(body)
        messages = body["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        payload = json.loads(str(messages[1]["content"]))
        assert "per_turn_instructions" in payload
        assert body["response_schema"] == judge_wire_schema(judge, 1)
        axis = {
            "status": "assessed",
            "score": 2,
            "rationale": "Reply differs.",
            "evidence_refs": ["step-0", "reference-step-0"],
            "critical_flags": [],
        }
        return {
            "axes": {key: axis for key in ids},
            "turns": {
                "step-0": {
                    "scores": {key: (2 if key == ids[0] else None) for key in ids},
                    "rationale": "The read invents an item.",
                }
            },
        }

    report = evaluate(
        bad,
        references(),
        judge=judge,
        complete=complete,
        artifacts=project.artifacts,
        artifact_id="turns",
        clock=lambda: NOW,
        code_revision="test",
    )
    assert len(calls) == 1 and report.judge_status == "completed"
    (turn,) = report.turns
    assert (turn.fields_matched, turn.fields_total, turn.divergence) == (0, 2, 1.0)
    assert turn.scores[ids[0]] == 2 and turn.scores[ids[1]] is None
    assert turn.rationale == "The read invents an item."
    assert "per_turn" not in default_judge("scripted").model_dump(mode="json")


def test_malformed_per_turn_scores_fail_the_judge_but_keep_divergence(tmp_path: Path) -> None:
    """A response missing a planned step is refused; deterministic divergence remains."""
    project = ProjectStore(tmp_path, "turns-bad")
    project.initialize(ProjectConfig(project_id="turns-bad"))
    judge = default_judge("scripted", protocol="keyed-v3").model_copy(update={"per_turn": True})
    axis = {
        "status": "assessed",
        "score": 5,
        "rationale": "Exact.",
        "evidence_refs": ["step-0"],
        "critical_flags": [],
    }
    report = evaluate(
        evidence(),
        references(),
        judge=judge,
        complete=lambda body, identity: {
            "axes": {a.dimension_id: axis for a in judge.definition.dimensions},
            "turns": {},
        },
        artifacts=project.artifacts,
        artifact_id="turns-bad",
        clock=lambda: NOW,
        code_revision="test",
    )
    assert report.judge_status == "failed"
    (turn,) = report.turns
    assert (turn.fields_matched, turn.fields_total, turn.scores) == (1, 1, {})


@pytest.mark.parametrize("bad", [True, "4", 2.0])
def test_per_turn_scores_are_strict_integers(tmp_path: Path, bad: JsonValue) -> None:
    """A boolean, string or float score is refused, never shown as a coerced integer."""
    project = ProjectStore(tmp_path, "strict-turns")
    project.initialize(ProjectConfig(project_id="strict-turns"))
    judge = default_judge("scripted", protocol="keyed-v3").model_copy(update={"per_turn": True})
    ids = [axis.dimension_id for axis in judge.definition.dimensions]
    axis = {
        "status": "assessed",
        "score": 5,
        "rationale": "Exact.",
        "evidence_refs": ["step-0"],
        "critical_flags": [],
    }
    report = evaluate(
        evidence(),
        references(),
        judge=judge,
        complete=lambda body, identity: {
            "axes": {key: axis for key in ids},
            "turns": {"step-0": {"scores": dict.fromkeys(ids, bad), "rationale": "Scored."}},
        },
        artifacts=project.artifacts,
        artifact_id="strict-turns",
        clock=lambda: NOW,
        code_revision="test",
    )
    assert report.judge_status == "failed"
    assert all(not turn.scores for turn in report.turns)


def test_a_turn_without_a_simulated_reply_reproduces_none_of_the_recorded_fields() -> None:
    """Failed and blocked steps with a recorded reply count as divergent, not incomparable."""
    recorded: JsonValue = {"ok": True, "items": [1, 2]}
    steps = (
        WorkflowStep(step=0, action={"name": "read"}, status="failure", before={"items": []}),
        WorkflowStep(step=1, action={"name": "read"}, status="blocked"),
    )
    refs = EvaluationReferences(
        capability="recorded_reference_only",
        steps=tuple(
            ReferenceStep(step=i, action={"name": "read"}, reply_available=True, reply=recorded)
            for i in range(2)
        ),
    )
    turns = turn_outcomes(evidence().model_copy(update={"steps": steps}), refs)
    assert [(t.fields_matched, t.fields_total, t.divergence) for t in turns] == [(0, 3, 1.0)] * 2
    unreferenced = turn_outcomes(
        evidence().model_copy(update={"steps": steps}),
        EvaluationReferences(capability="description_only"),
    )
    assert all(t.divergence is None for t in unreferenced)


def test_difference_paths_are_unambiguous_and_sorted_numerically() -> None:
    """Dotted keys are quoted, indices sort numerically, and containers are not shown absent."""
    actual: JsonValue = {"a.b": 1, "a": {"b": 2}, "rows": list(range(12)), "x": {"y": None}}
    expected: JsonValue = {"a.b": 0, "a": {"b": 3}, "rows": [*range(10), 0, 0], "x": {}}
    rows = field_differences(actual, expected)
    assert [row["path"] for row in rows] == [
        "a.b",
        '["a.b"]',
        "rows[10]",
        "rows[11]",
        "x",
        "x.y",
    ]
    container = next(row for row in rows if row["path"] == "x")
    assert container["actual"] == {"y": None} and not container["actual_missing"]

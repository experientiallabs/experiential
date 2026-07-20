"""Tests for benchmark-neutral harness scoring and search evidence."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmh.harness.delta import FailureSignature
from wmh.harness.scoring import (
    MAX_RENDERED_SCORE_EVIDENCE_CHARS,
    MAX_TASK_DESCRIPTION_CHARS,
    MAX_TASK_EVIDENCE_CHARS,
    HarnessScoreReport,
    PassCriterion,
    ScoreObjective,
    ScoreProvenance,
    ScoreRequest,
    TaskScore,
    cluster_score_failures,
    render_score_evidence,
    suite_score,
    suite_secondary_score,
)

_CANDIDATE_EXECUTION_HASH = "a" * 32
_REQUEST = ScoreRequest(purpose="seed")
_PROVENANCE = ScoreProvenance(
    task_set={"tasks": [{"task_id": "fixture"}]},
    evaluator={"kind": "fixture", "version": 1},
    backend={"kind": "in_process", "config": {}},
)
_PASS_CRITERION = PassCriterion(score_at_least=1.0)


def _task(
    task_id: str,
    *,
    score: float,
    secondary: float | None = None,
    aggregate_weight: float = 1.0,
    mechanisms: tuple[str, ...] = (),
    evidence: str = "",
) -> TaskScore:
    return TaskScore(
        task_id=task_id,
        score=score,
        secondary_score=secondary,
        aggregate_weight=aggregate_weight,
        passed=score == 1.0,
        description=f"instruction for {task_id}",
        mechanisms=mechanisms,
        evidence=evidence,
    )


def _report(*tasks: TaskScore) -> HarnessScoreReport:
    per_task = {task.task_id: task for task in tasks}
    total_weight = sum(task.aggregate_weight for task in tasks)
    secondary_values = [task.secondary_score for task in tasks]
    has_secondary = any(value is not None for value in secondary_values)
    return HarnessScoreReport(
        candidate_execution_hash=_CANDIDATE_EXECUTION_HASH,
        request=_REQUEST,
        provenance=_PROVENANCE,
        pass_criterion=_PASS_CRITERION,
        label="candidate",
        score=sum(task.score * task.aggregate_weight for task in tasks) / total_weight,
        secondary_objective=(
            ScoreObjective(objective_id="dense_progress") if has_secondary else None
        ),
        secondary_score=(
            sum(
                value * task.aggregate_weight
                for task, value in zip(tasks, secondary_values, strict=True)
                if value is not None
            )
            / total_weight
            if has_secondary
            else None
        ),
        attempts=2,
        per_task=per_task,
    )


def test_score_report_rejects_mismatched_task_key() -> None:
    with pytest.raises(ValidationError, match="key 'wrong'.*task_id 'task'"):
        HarnessScoreReport(
            candidate_execution_hash=_CANDIDATE_EXECUTION_HASH,
            request=_REQUEST,
            provenance=_PROVENANCE,
            pass_criterion=_PASS_CRITERION,
            score=0.0,
            attempts=1,
            per_task={"wrong": _task("task", score=0.0)},
        )


def test_score_report_requires_an_explicit_consistent_secondary_objective() -> None:
    with pytest.raises(ValidationError, match="secondary objective"):
        HarnessScoreReport(
            candidate_execution_hash=_CANDIDATE_EXECUTION_HASH,
            request=_REQUEST,
            provenance=_PROVENANCE,
            pass_criterion=_PASS_CRITERION,
            score=0.0,
            secondary_score=0.0,
            attempts=1,
            per_task={"task": _task("task", score=0.0)},
        )
    with pytest.raises(ValidationError, match="every task"):
        HarnessScoreReport(
            candidate_execution_hash=_CANDIDATE_EXECUTION_HASH,
            request=_REQUEST,
            provenance=_PROVENANCE,
            pass_criterion=_PASS_CRITERION,
            score=0.0,
            secondary_objective=ScoreObjective(objective_id="dense_progress"),
            secondary_score=0.0,
            attempts=1,
            per_task={"task": _task("task", score=0.0)},
        )


def test_score_report_enforces_declared_weighted_aggregate() -> None:
    light = _task("light", score=0.0, aggregate_weight=1.0)
    heavy = _task("heavy", score=1.0, aggregate_weight=3.0)

    report = _report(light, heavy)

    assert report.score == 0.75
    with pytest.raises(ValidationError, match="weighted task aggregate"):
        HarnessScoreReport(
            candidate_execution_hash=_CANDIDATE_EXECUTION_HASH,
            request=_REQUEST,
            provenance=_PROVENANCE,
            pass_criterion=_PASS_CRITERION,
            score=0.5,
            attempts=1,
            per_task={"light": light, "heavy": heavy},
        )


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_task_score_rejects_nonpositive_or_nonfinite_aggregate_weight(weight: float) -> None:
    with pytest.raises(ValidationError, match="aggregate_weight"):
        _task("task", score=0.0, aggregate_weight=weight)


def test_score_report_rejects_an_empty_task_matrix() -> None:
    with pytest.raises(ValidationError, match="per_task"):
        HarnessScoreReport(
            candidate_execution_hash=_CANDIDATE_EXECUTION_HASH,
            request=_REQUEST,
            provenance=_PROVENANCE,
            pass_criterion=_PASS_CRITERION,
            score=0.0,
            attempts=1,
            per_task={},
        )


def test_score_report_enforces_its_explicit_pass_criterion() -> None:
    criterion = PassCriterion(score_at_least=0.5)
    passing = TaskScore(task_id="task", score=0.5, passed=True)

    report = HarnessScoreReport(
        candidate_execution_hash=_CANDIDATE_EXECUTION_HASH,
        request=_REQUEST,
        provenance=_PROVENANCE,
        pass_criterion=criterion,
        score=0.5,
        attempts=1,
        per_task={"task": passing},
    )

    assert report.per_task["task"].passed is True
    with pytest.raises(ValidationError, match="frozen"):
        criterion.score_at_least = 0.25
    with pytest.raises(ValidationError, match="pass criterion"):
        HarnessScoreReport.model_validate(
            {
                **report.model_dump(mode="json", exclude={"evaluation_id"}),
                "per_task": {
                    "task": passing.model_copy(update={"passed": False}).model_dump(mode="json")
                },
            }
        )


def test_evaluation_identity_is_canonical_and_excludes_display_label() -> None:
    report = _report(_task("task", score=0.0, evidence="trace one"))

    assert report.evaluation_id.startswith("score-sha256:")
    assert report.model_copy(update={"label": "renamed display label"}).evaluation_id == (
        report.evaluation_id
    )
    assert report.model_copy(update={"candidate_execution_hash": "b" * 32}).evaluation_id != (
        report.evaluation_id
    )
    assert (
        report.model_copy(update={"request": ScoreRequest(purpose="full")}).evaluation_id
        != report.evaluation_id
    )
    changed_contexts = {
        "task_set": {"tasks": [{"task_id": "other"}]},
        "evaluator": {"kind": "other", "version": 1},
        "backend": {"kind": "isolated", "config": {}},
    }
    for field, value in changed_contexts.items():
        changed = report.model_copy(
            update={"provenance": _PROVENANCE.model_copy(update={field: value})}
        )
        assert changed.evaluation_id != report.evaluation_id
    changed_evidence = _report(_task("task", score=0.0, evidence="trace two"))
    assert changed_evidence.evaluation_id != report.evaluation_id


def test_evaluation_identity_is_centrally_derived_and_serialization_round_trips() -> None:
    report = _report(_task("task", score=0.0, evidence="trace"))
    payload = report.model_dump(mode="json")
    payload["evaluation_id"] = "caller-chosen"

    restored = HarnessScoreReport.model_validate(payload)

    assert restored.evaluation_id == report.evaluation_id
    assert HarnessScoreReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize("field", ["task_set", "evaluator", "backend"])
def test_score_provenance_requires_each_frozen_context(field: str) -> None:
    payload = _PROVENANCE.model_dump(mode="json")
    payload[field] = {}
    with pytest.raises(ValidationError, match=field):
        ScoreProvenance.model_validate(payload)


@pytest.mark.parametrize("task_ids", [(), ("task", "task")])
def test_score_request_rejects_empty_or_duplicate_subset(task_ids: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="task_ids"):
        ScoreRequest(purpose="screen", task_ids=task_ids)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_score_report_rejects_non_normalized_scores(value: float) -> None:
    with pytest.raises(ValidationError):
        _report(_task("task", score=value))


def test_score_report_requires_identity_and_bounds_proposer_evidence() -> None:
    with pytest.raises(ValidationError, match="candidate_execution_hash"):
        HarnessScoreReport(
            candidate_execution_hash="",
            request=_REQUEST,
            provenance=_PROVENANCE,
            pass_criterion=_PASS_CRITERION,
            attempts=1,
            per_task={"task": _task("task", score=0.0)},
        )
    with pytest.raises(ValidationError, match="evidence"):
        _report(_task("task", score=0.0, evidence="x" * 64_001))
    report = _report(_task("task", score=0.0))
    with pytest.raises((AttributeError, ValidationError), match="evaluation_id|frozen|property"):
        report.evaluation_id = "changed"  # ty: ignore[invalid-assignment]


def test_cluster_score_failures_uses_shared_mechanisms_and_singletons() -> None:
    report = _report(
        _task("t1", score=0.0, mechanisms=("a", "b")),
        _task("t2", score=0.0, mechanisms=("b", "c")),
        _task("t3", score=0.0, mechanisms=("z",)),
        _task("t4", score=1.0),
        _task("t5", score=0.0),
    )

    clusters = cluster_score_failures(report)

    assert [cluster.task_ids for cluster in clusters] == [["t1", "t2"], ["t5"], ["t3"]]
    assert clusters[0].mechanism == "b"
    assert clusters[0].mechanism_labels == ["a", "b", "c"]
    assert clusters[1].mechanism == "run failed without mechanism details"
    assert clusters[2].mechanism == "z"


def test_render_score_evidence_is_benchmark_neutral() -> None:
    report = _report(
        _task("pass", score=1.0, secondary=1.0, evidence="pass evidence"),
        _task(
            "fail",
            score=0.0,
            secondary=0.5,
            mechanisms=("missing verification",),
            evidence="attempt trace and verifier feedback",
        ),
    )
    trigger = FailureSignature(
        mechanism="missing verification",
        task_ids=["fail"],
        mechanism_labels=["missing verification"],
    )

    rendered = render_score_evidence(trigger, report)

    assert "[other] pass: score=1.00, secondary_score=1.00" in rendered
    assert "[TARGET] fail: score=0.00, secondary_score=0.50" in rendered
    assert "Instruction: instruction for fail" in rendered
    assert "attempt trace and verifier feedback" in rendered
    assert "gold" not in rendered.lower()
    assert "world model" not in rendered.lower()


def test_render_score_evidence_handles_all_pass_parent() -> None:
    report = _report(_task("pass", score=1.0))

    rendered = render_score_evidence(FailureSignature(mechanism="none: all tasks pass"), report)

    assert "passed every task" in rendered
    assert "There are no failures to fix" in rendered


def test_render_score_evidence_is_invariant_to_task_and_label_insertion_order() -> None:
    first = _task(
        "first",
        score=0.0,
        mechanisms=("shared", "alpha"),
        evidence="first evidence",
    )
    second = _task(
        "second",
        score=0.0,
        mechanisms=("shared", "beta"),
        evidence="second evidence",
    )
    forward = _report(first, second)
    reverse = HarnessScoreReport(
        candidate_execution_hash=_CANDIDATE_EXECUTION_HASH,
        request=_REQUEST,
        provenance=_PROVENANCE,
        pass_criterion=_PASS_CRITERION,
        score=forward.score,
        secondary_objective=forward.secondary_objective,
        secondary_score=forward.secondary_score,
        attempts=forward.attempts,
        per_task={"second": second, "first": first},
    )

    forward_text = render_score_evidence(
        FailureSignature(
            mechanism="shared",
            task_ids=["first", "second"],
            mechanism_labels=["alpha", "beta", "shared"],
        ),
        forward,
    )
    reverse_text = render_score_evidence(
        FailureSignature(
            mechanism="shared",
            task_ids=["second", "first"],
            mechanism_labels=["shared", "beta", "alpha"],
        ),
        reverse,
    )

    assert forward_text == reverse_text
    assert forward_text.index("### Task first") < forward_text.index("### Task second")


def test_render_score_evidence_bounds_a_maximal_connected_cluster() -> None:
    tasks = tuple(
        TaskScore(
            task_id=f"task-{index:02}",
            score=0.0,
            secondary_score=0.0,
            passed=False,
            description=chr(65 + index % 26) * MAX_TASK_DESCRIPTION_CHARS,
            mechanisms=("shared", f"mechanism-{index:02}"),
            evidence=f"evidence-{index:02}:" + "x" * (MAX_TASK_EVIDENCE_CHARS - 12),
        )
        for index in range(30)
    )
    report = _report(*tasks)
    task_ids = [task.task_id for task in reversed(tasks)]
    trigger = FailureSignature(
        mechanism="shared",
        task_ids=task_ids,
        mechanism_labels=["shared", *(f"mechanism-{index:02}" for index in range(30))],
    )

    rendered = render_score_evidence(trigger, report)

    assert len(rendered) <= MAX_RENDERED_SCORE_EVIDENCE_CHARS
    assert "truncated" in rendered
    for task in tasks:
        assert f"### Task {task.task_id}" in rendered


def test_suite_scores_use_weights_and_reject_missing_tasks() -> None:
    report = _report(
        _task("t1", score=0.25, secondary=0.5, aggregate_weight=1.0),
        _task("t2", score=0.75, secondary=1.0, aggregate_weight=3.0),
    )

    assert suite_score(report, ["t1", "t2"]) == pytest.approx(0.625)
    assert suite_secondary_score(report, ["t1", "t2"]) == pytest.approx(0.875)
    assert suite_score(report, []) == 1.0
    assert suite_secondary_score(report, []) == 1.0
    with pytest.raises(ValueError, match="missing task"):
        suite_score(report, ["t1", "missing"])


def test_suite_secondary_score_is_absent_without_a_declared_objective() -> None:
    report = _report(_task("task", score=0.5))

    assert suite_secondary_score(report, ["task"]) is None
    assert suite_secondary_score(report, []) is None

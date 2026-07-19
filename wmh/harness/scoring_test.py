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
    ScoreRequest,
    ScoreRunHealth,
    TaskScore,
    cluster_score_failures,
    render_score_evidence,
    suite_score,
    suite_secondary_score,
)


def _task(
    task_id: str,
    *,
    score: float,
    secondary: float | None = None,
    mechanisms: tuple[str, ...] = (),
    evidence: str = "",
) -> TaskScore:
    return TaskScore(
        task_id=task_id,
        score=score,
        secondary_score=score if secondary is None else secondary,
        passed=score == 1.0,
        description=f"instruction for {task_id}",
        mechanisms=mechanisms,
        evidence=evidence,
    )


def _report(*tasks: TaskScore) -> HarnessScoreReport:
    per_task = {task.task_id: task for task in tasks}
    return HarnessScoreReport(
        evaluation_id="eval-1",
        label="candidate",
        score=sum(task.score for task in tasks) / len(tasks),
        secondary_score=sum(task.secondary_score for task in tasks) / len(tasks),
        attempts=2,
        run_health=ScoreRunHealth.VALID,
        per_task=per_task,
    )


def test_score_report_rejects_mismatched_task_key() -> None:
    with pytest.raises(ValidationError, match="key 'wrong'.*task_id 'task'"):
        HarnessScoreReport(
            evaluation_id="eval-1",
            score=0.0,
            secondary_score=0.0,
            attempts=1,
            run_health=ScoreRunHealth.VALID,
            per_task={"wrong": _task("task", score=0.0)},
        )


@pytest.mark.parametrize("task_ids", [(), ("task", "task")])
def test_score_request_rejects_empty_or_duplicate_subset(task_ids: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="task_ids"):
        ScoreRequest(purpose="screen", task_ids=task_ids)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_score_report_rejects_non_normalized_scores(value: float) -> None:
    with pytest.raises(ValidationError):
        HarnessScoreReport(
            evaluation_id="eval-1",
            score=value,
            secondary_score=0.0,
            attempts=1,
            run_health=ScoreRunHealth.VALID,
        )


def test_score_report_requires_identity_and_bounds_proposer_evidence() -> None:
    with pytest.raises(ValidationError, match="evaluation_id"):
        HarnessScoreReport(
            evaluation_id="",
            attempts=1,
            run_health=ScoreRunHealth.VALID,
        )
    with pytest.raises(ValidationError, match="evidence"):
        _report(_task("task", score=0.0, evidence="x" * 64_001))
    report = _report(_task("task", score=0.0))
    with pytest.raises(ValidationError, match="frozen"):
        report.evaluation_id = "changed"


def test_score_report_requires_explicit_run_health() -> None:
    with pytest.raises(ValidationError, match="run_health"):
        HarnessScoreReport.model_validate({"evaluation_id": "eval-1", "attempts": 1})


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
        _task("pass", score=1.0, evidence="pass evidence"),
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
        evaluation_id="eval-2",
        score=forward.score,
        secondary_score=forward.secondary_score,
        attempts=forward.attempts,
        run_health=ScoreRunHealth.VALID,
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


def test_suite_scores_count_tasks_absent_from_a_subset_report_as_zero() -> None:
    report = _report(
        _task("t1", score=0.25, secondary=0.5),
        _task("t2", score=0.75, secondary=1.0),
    )

    assert suite_score(report, ["t1", "t2", "missing"]) == pytest.approx(1 / 3)
    assert suite_secondary_score(report, ["t1", "t2", "missing"]) == 0.5
    assert suite_score(report, []) == 1.0
    assert suite_secondary_score(report, []) == 1.0

"""Tests for the held-out eval rubric author, judge and report, with a fake model client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.models import AssistantAction, ModelRequest, ModelResponse, ToolCall
from exp.optimize.evaluation.variance import eval_rubric as ev
from exp.optimize.evaluation.variance.episodes import EpisodeEvent, EpisodeEvidence

_LEVELS = {str(level): f"level {level}" for level in range(5)}
_RUBRIC = {
    "summary": "A good attempt makes the suite green without weakening it.",
    "criteria": [
        {
            "criterion_id": "suite_green",
            "dimension": "outcome",
            "question": "Green?",
            "weight": 3,
            "levels": _LEVELS,
        },
        {
            "criterion_id": "harness_intact",
            "dimension": "final_state",
            "question": "Intact?",
            "weight": 2,
            "levels": _LEVELS,
        },
        {
            "criterion_id": "verified_report",
            "dimension": "approach",
            "question": "Verified?",
            "weight": 1,
            "levels": _LEVELS,
        },
    ],
    "overall_rule": "Weighted mean of score/4.",
}
_TASK = {"task_id": "task-x", "instruction": "Fix it.", "tools": [], "starting_environment": {}}


class FakeCaller:
    """Returns queued answers as forced tool calls, recording every request."""

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.requests: list[tuple[ModelRequest, str]] = []

    def __call__(self, request: ModelRequest, identity: str) -> ModelResponse:
        self.requests.append((request, identity))
        answer = self.answers.pop(0)
        tool = request.tools[0].name
        if isinstance(answer, str):
            output = AssistantAction(content=answer)
        else:
            output = AssistantAction(
                tool_calls=(ToolCall(call_id="c1", name=tool, arguments=answer),)
            )
        return ModelResponse.model_construct(output=output)


def _evidence() -> EpisodeEvidence:
    def event(index: int, kind: str, scope: str, content: object, event_id: str) -> EpisodeEvent:
        owner = {
            "tool_call": "actor",
            "assistant_message": "actor",
            "final_answer": "actor",
            "tool_result": "world",
            "receipt": "harness",
            "state_snapshot": "world",
            "user_message": "user",
        }[kind]
        return EpisodeEvent(
            event_id=event_id,
            sequence=index,
            kind=kind,
            owner=owner,
            scope=scope,
            content=content,
            source=f"s#{index}",
        )

    return EpisodeEvidence(
        episode_id="e1",
        task_sha256="0" * 64,
        world_sha256="1" * 64,
        complete=True,
        events=(
            event(0, "user_message", "context", "Fix it.", "prefix.0"),
            event(1, "state_snapshot", "context", {"v": 0}, "state.0"),
            event(
                2,
                "tool_call",
                "scored",
                {"name": "bash", "arguments": {"command": "ls"}},
                "call.0.tool.0",
            ),
            event(
                3,
                "receipt",
                "scored",
                {"effects": "refused", "error_type": "bad"},
                "call.0.tool.0.receipt",
            ),
            event(
                4,
                "tool_result",
                "scored",
                {"content": "boom", "is_error": True},
                "call.0.tool.0.reply",
            ),
            event(5, "state_snapshot", "scored", {"v": 1}, "state.1"),
            event(6, "final_answer", "scored", "Done.", "call.1.final"),
        ),
    )


def _rubric(tmp_path: Path, caller: FakeCaller) -> tuple[ev.EvalRubric, Path, bool]:
    return ev.freeze_rubric(
        tmp_path,
        task=_TASK,
        error_context={"training_rubric_failure_patterns": [], "observed_training_findings": []},
        context_sources=[
            ev.ContextSource(
                artifact_id="episode-rubric-x", sha256="a" * 64, role="training rubric"
            )
        ],
        caller=caller,
        author_alias="glm",
    )


def test_rubric_is_frozen_once_and_reused(tmp_path: Path) -> None:
    caller = FakeCaller(_RUBRIC)
    rubric, path, authored = _rubric(tmp_path, caller)
    assert authored and path.name.endswith(f"{rubric.sha256}.json")
    assert rubric.context_sources[0].artifact_id == "episode-rubric-x"
    again, same, authored_again = _rubric(tmp_path, FakeCaller())
    assert not authored_again and same == path and again == rubric


def test_tampered_rubric_is_refused(tmp_path: Path) -> None:
    _, path, _ = _rubric(tmp_path, FakeCaller(_RUBRIC))
    body = json.loads(path.read_text())
    body["body"]["overall_rule"] = "changed"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="digest"):
        _rubric(tmp_path, FakeCaller())


def test_malformed_rubric_is_redrawn_then_accepted(tmp_path: Path) -> None:
    missing_dimension = {**_RUBRIC, "criteria": _RUBRIC["criteria"][:2] * 1}
    text_answer = "```json\n" + json.dumps(_RUBRIC) + "\n```"
    caller = FakeCaller(missing_dimension, text_answer)
    rubric, _, _ = _rubric(tmp_path, caller)
    assert len(rubric.body.criteria) == 3
    assert [identity[-2:] for _, identity in caller.requests] == ["d0", "d1"]
    assert "rejected" in str(caller.requests[1][0].messages[-1].content)


def test_rubric_never_authored_after_every_draw_is_malformed(tmp_path: Path) -> None:
    with pytest.raises(ev.MalformedAnswer):
        _rubric(tmp_path, FakeCaller("no json", "no json", "no json"))
    assert ev.rubric_path(tmp_path, "task-x") is None


def _answer(scores: tuple[int, int, int], overall: float = 0.8) -> dict[str, object]:
    ids = ("suite_green", "harness_intact", "verified_report")
    return {
        "scores": [
            {"criterion_id": c, "score": s, "reason": "r"} for c, s in zip(ids, scores, strict=True)
        ],
        "overall_score": overall,
        "summary": "Fine.",
    }


def test_judge_scores_and_retries_malformed(tmp_path: Path) -> None:
    rubric, _, _ = _rubric(tmp_path, FakeCaller(_RUBRIC))
    incomplete = {**_answer((4, 4, 4)), "scores": _answer((4, 4, 4))["scores"][:2]}
    caller = FakeCaller(incomplete, _answer((4, 4, 0), 0.83))
    judgment = ev.judge_attempt(
        rubric, _TASK, _evidence(), tier="t", study="s", branch=0, caller=caller
    )
    assert judgment.status == "judged" and judgment.draws == 2
    assert judgment.overall_score == 0.83
    assert judgment.weighted_overall == pytest.approx(5 / 6, abs=1e-3)
    payload = json.loads(caller.requests[0][0].messages[1].content)
    assert set(payload) == {"rubric", "task", "attempt"}


def test_judge_records_malformed_after_bounded_retries(tmp_path: Path) -> None:
    rubric, _, _ = _rubric(tmp_path, FakeCaller(_RUBRIC))
    bad = _answer((4, 4, 4), 1.5)
    judgment = ev.judge_attempt(
        rubric, _TASK, _evidence(), tier="t", study="s", branch=0, caller=FakeCaller(bad, bad, bad)
    )
    assert judgment.status == "malformed" and judgment.draws == 3
    assert judgment.overall_score is None


def test_trajectory_keeps_actions_replies_and_final_state() -> None:
    attempt = ev.render_trajectory(_evidence())
    assert attempt["final_environment_state"] == {"v": 1}
    assert attempt["final_answer"] == "Done."
    reply = attempt["steps"][1]
    assert reply["is_error"] and reply["harness"]["error_type"] == "bad"
    assert attempt["steps"][0] == {"tool_call": "bash", "arguments": {"command": "ls"}}


def test_error_context_dedupes_and_balances_criteria() -> None:
    rubric = {
        "body": {
            "criteria": [
                {
                    "criterion_id": "a",
                    "question": "q",
                    "not_met_when": "n",
                    "failure_patterns": ["p"],
                }
            ]
        }
    }
    findings = [
        {"criterion_id": "a", "status": "not_met", "rationale": "x"},
        {"criterion_id": "a", "status": "not_met", "rationale": "x"},
        {"criterion_id": "a", "status": "not_met", "rationale": "y"},
        {"criterion_id": "b", "status": "insufficient_evidence", "rationale": "z"},
        {"criterion_id": "b", "status": "met", "rationale": "ok"},
    ]
    context = ev.training_error_context(rubric, findings, maximum_findings=2)
    notes = [row["note"] for row in context["observed_training_findings"]]
    assert notes == ["x", "z"]
    assert context["training_rubric_failure_patterns"][0]["failure_patterns"] == ["p"]


def test_aggregate_reports_tiers_and_agreement(tmp_path: Path) -> None:
    rubric, _, _ = _rubric(tmp_path, FakeCaller(_RUBRIC))

    def row(tier: str, branch: int, overall: float | None) -> ev.EvalJudgment:
        if overall is None:
            return ev.EvalJudgment(
                rubric_sha256=rubric.sha256,
                tier=tier,
                study=tier,
                branch=branch,
                status="malformed",
                failure="x",
            )
        return ev.EvalJudgment(
            rubric_sha256=rubric.sha256,
            tier=tier,
            study=tier,
            branch=branch,
            status="judged",
            scores=tuple(
                ev.CriterionScore(
                    criterion_id=c.criterion_id, score=4 if overall > 0.5 else 0, reason="r"
                )
                for c in rubric.body.criteria
            ),
            overall_score=overall,
            weighted_overall=overall,
            summary="s",
            draws=1,
        )

    judgments = [row("a", 0, 0.9), row("a", 1, 0.2), row("a", 2, None), row("b", 0, 0.7)]
    verdict = ev.TrainingVerdict(verdict="pass", judged_study="j", judge="0" * 64)
    training = {("a", 0): verdict, ("a", 1): verdict, ("a", 2): verdict, ("b", 0): None}
    report = ev.aggregate(rubric, judgments, training)
    tier = report["tiers"]["a"]
    assert tier["overall"]["n"] == 2 and tier["overall"]["mean"] == 0.55
    assert tier["per_criterion_mean"]["suite_green"] == 2.0
    assert tier["agreement_at_threshold"] == {
        "agree": 1,
        "disagree": 1,
        "training_pass_eval_pass": 1,
        "training_pass_eval_fail": 1,
    }
    assert report["tiers"]["b"]["training_verdicts"] == {"not judged yet": 1}
    assert "| a |" in ev.markdown_report(report, rubric)

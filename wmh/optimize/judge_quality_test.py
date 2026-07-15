"""Tests for the judge-quality meta-eval harness mechanics."""

from __future__ import annotations

import json

from wmh.core.types import Action, ActionKind, Observation, Step
from wmh.optimize.judge import JudgeResult
from wmh.optimize.judge_quality import (
    JUDGE_QUALITY_CASES,
    JudgeCase,
    ScoreBand,
    run_judge_quality,
)


class StubJudge:
    """Returns a fixed JudgeResult regardless of input; records calls."""

    def __init__(self, result: JudgeResult) -> None:
        self._result = result
        self.calls: list[Step] = []

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        self.calls.append(context)
        return self._result


def _case(
    *,
    id: str = "c1",  # noqa: A002 - mirrors the JudgeCase field name
    expected: ScoreBand | None = None,
    expected_dimensions: dict[str, ScoreBand] | None = None,
) -> JudgeCase:
    return JudgeCase(
        id=id,
        defect="control",
        rationale="r",
        action=Action(kind=ActionKind.TOOL_CALL, name="t", arguments={}),
        actual=Observation(content="a"),
        predicted=Observation(content="p"),
        expected=expected or ScoreBand(lo=0.5, hi=1.0),
        expected_dimensions=expected_dimensions or {},
    )


def test_case_passes_when_score_in_band() -> None:
    judge = StubJudge(JudgeResult(score=0.8, critique="ok"))
    report = run_judge_quality(judge, [_case()])
    assert report.n_total == 1
    assert report.n_passed == 1
    assert report.verdicts[0].passed
    assert report.verdicts[0].failures == []


def test_case_fails_outside_band_with_readable_reason() -> None:
    judge = StubJudge(JudgeResult(score=0.2, critique="bad"))
    report = run_judge_quality(judge, [_case()])
    assert report.n_passed == 0
    verdict = report.verdicts[0]
    assert not verdict.passed
    assert "score 0.200 outside [0.5, 1.0]" in verdict.failures[0]
    assert report.failed() == [verdict]


def test_dimension_bands_are_checked_and_missing_dims_fail() -> None:
    case = _case(
        expected=ScoreBand(lo=0.0, hi=1.0),
        expected_dimensions={"factuality": ScoreBand(lo=0.0, hi=0.3)},
    )
    # In band on the headline, out of band on the dimension.
    out_of_band = StubJudge(JudgeResult(score=0.5, critique="", dimensions={"factuality": 0.9}))
    assert run_judge_quality(out_of_band, [case]).n_passed == 0
    # A judge that reports no dimensions at all cannot pass a dimension band.
    missing = StubJudge(JudgeResult(score=0.5, critique=""))
    verdict = run_judge_quality(missing, [case]).verdicts[0]
    assert not verdict.passed
    assert "factuality missing" in verdict.failures[0]


def test_invalid_judgement_fails_the_case_even_if_score_is_in_band() -> None:
    # A judge that errors out (valid=False, score 0.0) must not vacuously pass a low-band case.
    case = _case(expected=ScoreBand(lo=0.0, hi=0.3))
    judge = StubJudge(JudgeResult(score=0.0, critique="judge broke", valid=False))
    verdict = run_judge_quality(judge, [case]).verdicts[0]
    assert not verdict.passed
    assert any("invalid" in failure for failure in verdict.failures)


def test_concurrency_preserves_case_order() -> None:
    judge = StubJudge(JudgeResult(score=0.9, critique=""))
    cases = [_case(id=f"c{i}") for i in range(5)]
    report = run_judge_quality(judge, cases, concurrency=4)
    assert [v.case_id for v in report.verdicts] == [f"c{i}" for i in range(5)]


def test_case_step_carries_action_and_actual() -> None:
    case = _case()
    step = case.step()
    assert step.action.name == "t"
    assert step.observation.content == "a"


def test_builtin_suite_is_default_and_well_formed() -> None:
    ids = [case.id for case in JUDGE_QUALITY_CASES]
    assert len(ids) == len(set(ids))  # unique ids
    assert any(case.defect == "control" for case in JUDGE_QUALITY_CASES)
    for case in JUDGE_QUALITY_CASES:
        assert case.expected.lo <= case.expected.hi
    # The cosmetic-reordering control is only valid while its two hand-written JSON literals
    # stay functionally identical; an edit to one twin must not silently break the case.
    by_id = {case.id: case for case in JUDGE_QUALITY_CASES}
    reordered = by_id["cosmetic-reordering"]
    assert json.loads(reordered.predicted.content) == json.loads(reordered.actual.content)
    # Default suite is used when no cases are passed.
    judge = StubJudge(JudgeResult(score=0.5, critique=""))
    report = run_judge_quality(judge)
    assert report.n_total == len(JUDGE_QUALITY_CASES)
    assert report.summary().startswith("judge quality:")

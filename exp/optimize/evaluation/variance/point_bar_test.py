"""The qualification bar decides points over the whole trajectory, with no fixed count."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from exp.optimize.evaluation.variance.blast_radius import BlastRadius
from exp.optimize.evaluation.variance.point_bar import (
    DEFAULT_BAR,
    PickJustification,
    QualificationBar,
    RuleCitation,
    advance,
    bar_instruction,
    check_picks,
    qualify_points,
    schedule_points,
)
from exp.optimize.evaluation.variance.reward_states import POINT_MACHINE, PointState
from exp.simulation.mining.questions import Question, QuestionBank
from exp.simulation.mining.tagging import TagMatrix, TagRow
from exp.simulation.mining.traceml import TraceStep, TraceTrajectory, step_units
from exp.simulation.mining.views import Unit

REQUIRED = [rule.rule_id for rule in DEFAULT_BAR.rules if rule.required]


def trajectory(length: int) -> TraceTrajectory:
    """A long attempt, one tool step per boundary."""
    return TraceTrajectory(
        trace_id="actor/attempt:0",
        tier="actor",
        task="Fix the parser",
        steps=tuple(
            TraceStep(
                index=i,
                locator=f"e{i}",
                boundary=i,
                kind="tool",
                tool="edit_file",
                arguments={"step": i},
            )
            for i in range(length)
        ),
    )


class BarTagger:
    """Answers the bar per step: qualifying, rejected or doubtful steps by index."""

    def __init__(self, good: set[int], doubtful: set[int] | None = None) -> None:
        """Steps in ``good`` clear the bar; ``doubtful`` ones sit inside the band."""
        self.good, self.doubtful = good, doubtful or set()

    def tag(self, units: Sequence[Unit], bank: QuestionBank) -> TagMatrix:
        """Answer every rule for every step."""
        rows = []
        for unit in units:
            index = int(str(unit.meta["locator"])[1:])
            value = 0.95 if index in self.good else 0.5 if index in self.doubtful else 0.05
            rows.append(
                TagRow(
                    trace_id=unit.trace_id,
                    unit_hash=unit.unit_hash,
                    values=tuple(value for _ in bank.question_ids()),
                )
            )
        return TagMatrix(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            question_ids=bank.question_ids(),
            view="step",
            rows=tuple(rows),
        )


def matrix(trace: TraceTrajectory, good: set[int], doubtful: set[int] | None = None) -> TagMatrix:
    """Bar answers for the trajectory."""
    return BarTagger(good, doubtful or set()).tag(step_units(trace), DEFAULT_BAR.bank())


def radius(point: int, rank: int, score: float = 0.5) -> BlastRadius:
    """A ranked blast radius."""
    return BlastRadius(
        point_id=f"point-{point}",
        point=point,
        rank=rank,
        tiers_graded=5,
        tiers_failing=3,
        ever_passed=True,
        score=score,
    )


def justify(
    point: int, rules: Sequence[str] = REQUIRED, refs: tuple[str, ...] = ("e1",)
) -> PickJustification:
    """A pick justified against the named rules."""
    return PickJustification(
        point=point,
        rules=tuple(RuleCitation(rule_id=r, holds=True, event_refs=refs) for r in rules),
        rationale="clears the bar",
    )


def test_points_late_in_a_long_trajectory_qualify_with_no_fixed_count() -> None:
    """Seven qualifying points, the last at step 57 of 60; nothing caps them at four or 20."""
    trace = trajectory(60)
    good = {3, 22, 31, 40, 47, 52, 57}

    record = qualify_points(trace, matrix(trace, good, {10}))

    assert record.qualified == tuple(sorted(good))
    states = {p.point: p.state for p in record.points}
    assert states[10] == PointState.UNCERTAIN and states[11] == PointState.REJECTED
    assert record.refusals == ()
    path = [t.target for t in record.transitions if t.subject == "point-57"]
    assert path == ["tagged", "qualified"]
    assert "no fixed number" in bar_instruction()
    assert all(rule.rule_id in bar_instruction() for rule in DEFAULT_BAR.rules)


def test_the_bar_refuses_a_pick_that_fails_it_or_is_not_justified() -> None:
    """Non-qualifying, unjustified or uncited picks are refused; a good pick is accepted."""
    trace = trajectory(30)
    record = qualify_points(trace, matrix(trace, {5, 25}))
    events = [s.locator for s in trace.steps]

    accepted, refused = check_picks(
        [
            justify(25),
            justify(6),
            justify(5, REQUIRED[:-1]),
            justify(5, refs=("e99",)),
            justify(25),
        ],
        record,
        events,
    )

    assert accepted == (25,)
    problems = {i: r.problems for i, r in enumerate(refused)}
    assert "did not clear the bar" in problems[0][0]
    assert any("bar.not_trivial_retry is not justified" in p for p in problems[1])
    assert any("unknown events" in p for p in problems[2])


def test_uncertain_answers_and_zero_blast_radius_never_qualify() -> None:
    """Doubt and a refused tagging pass stay unqualified; zero blast radius rejects."""
    trace = trajectory(8)
    assert qualify_points(trace, None).qualified == ()
    assert {p.state for p in qualify_points(trace, None).points} == {PointState.UNCERTAIN}
    record = qualify_points(trace, matrix(trace, {2, 4}), blast={4: radius(4, 1, score=0.0)})
    assert record.qualified == (2,)


def test_qualified_points_continue_in_blast_order_and_defer_at_budget() -> None:
    """Order follows blast radius; a point that does not fit is deferred, then continued."""
    trace = trajectory(50)
    record = qualify_points(trace, matrix(trace, {4, 18, 33, 45}))
    ranked = [radius(33, 1), radius(45, 2), radius(4, 3)]
    cost = {4: 40, 18: 40, 33: 40, 45: 70}

    first = schedule_points(record, ranked, cost=cost, remaining=100)

    assert first.continued == (33, 4) and first.deferred == (45, 18)
    assert first.reserved == 80 and first.remaining == 20 and first.refusals == ()
    record = advance(record, first)
    tight = schedule_points(record, ranked, cost=cost, remaining=30)
    assert tight.continued == () and tight.deferred == (45, 18) and tight.transitions == ()
    second = schedule_points(record, ranked, cost=cost, remaining=200)
    assert second.continued == (45, 18) and second.deferred == ()
    assert {t.source for t in second.transitions} == {"deferred"}
    record = advance(record, second)
    assert schedule_points(record, ranked, cost=cost, remaining=100).continued == ()


@pytest.mark.parametrize(
    ("path", "illegal"),
    [
        ((PointState.TAGGED, PointState.REJECTED), PointState.CONTINUED),
        ((PointState.UNCERTAIN,), PointState.QUALIFIED),
        ((), PointState.CONTINUED),
        ((PointState.TAGGED, PointState.QUALIFIED, PointState.CONTINUED), PointState.DEFERRED),
    ],
)
def test_illegal_point_moves_are_refused(path: tuple[PointState, ...], illegal: PointState) -> None:
    """Only a qualified point is continued; uncertain never becomes qualified."""
    states = POINT_MACHINE.lifecycle().start("p")
    for target in path:
        assert states.move("p", target)
    assert not states.move("p", illegal) and states.refusals[0].target == illegal.value


def test_an_authored_bank_becomes_a_versioned_bar() -> None:
    """The tag author's bank is adopted as is, with its required rules named."""
    bank = QuestionBank(
        bank_id="authored-bar",
        version=3,
        questions=(
            Question(question_id="bar.choice", family="bar.meta", text="A real choice?"),
            Question(question_id="bar.signal", family="bar.meta", text="Tiers diverge?"),
        ),
    )
    bar = QualificationBar.from_bank(bank, ["bar.choice"])
    assert bar.version == 3 and [r.required for r in bar.rules] == [True, False]
    assert bar.bank() == bank
    with pytest.raises(ValueError, match="not questions"):
        QualificationBar.from_bank(bank, ["bar.missing"])

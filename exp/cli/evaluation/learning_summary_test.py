"""The short learning and retest lines name every denominator and any trace re-attempt."""

from exp.cli.evaluation.learning_summary import LearningSummary, learning_lines
from exp.cli.evaluation.miss_retest import RetestOutcome
from exp.optimize.evaluation.fidelity.recovery import QUALIFICATION, Qualification
from exp.optimize.evaluation.fidelity.replay import ReplayAggregate
from exp.optimize.evaluation.learning.retest import TurnChanges
from exp.optimize.evaluation.learning.traces import LearningOutcome


def _retest(qualification: Qualification | None) -> RetestOutcome:
    """A one-trace retest that fixed its only missed turn."""
    total = ReplayAggregate(
        traces=1, exact=3, steps=3, fields_matched=3, fields_total=3, judge_mean=None
    )
    changes = TurnChanges(missed=1, fixed=1, regressed=0, before_exact=2, after_exact=3, steps=3)
    return RetestOutcome(total=total, planned=1, changes=changes, qualification=qualification)


def test_a_retest_with_a_trace_reattempt_says_so_on_its_line() -> None:
    """The terminal line keeps the qualification; the report file holds the subsets."""
    outcome = LearningOutcome.model_construct(
        traces=4, learned=4, incomplete=0, head_version=4, attempts_used=12, stopped=None
    )
    qualification = Qualification(
        prediction=1,
        judge=0,
        investigation=0,
        retrieval=0,
        affected=1,
        traces=1,
        exact_unaffected=0,
        steps_unaffected=0,
        exact_affected=3,
        steps_affected=3,
    )
    plain = learning_lines(LearningSummary("clef", outcome, None, _retest(None)))
    qualified = learning_lines(LearningSummary("clef", outcome, None, _retest(qualification)))
    assert plain[0].endswith("learned 4/4 traces · memory v4 · 12 model attempts")
    assert plain[-1].endswith("fixed 1 of 1 missed turns · 0 regressed")
    assert qualified[-1] == f"{plain[-1]} · {QUALIFICATION}"

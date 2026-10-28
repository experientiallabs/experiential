"""The held-out aggregate line states every denominator."""

from exp.cli.evaluation.replay_view import aggregate_line
from exp.optimize.evaluation.fidelity.replay import ReplayAggregate


def test_aggregate_line_reports_replies_fields_judge_and_coverage() -> None:
    """One line, with explicit counts and a dash when nothing was judged."""
    total = ReplayAggregate(
        traces=2,
        exact=6,
        steps=9,
        fields_matched=40,
        fields_total=50,
        judge_mean=3.75,
        judged_traces=2,
    )
    assert aggregate_line(total, 3) == (
        "Held-out fidelity  6/9 replies exact (67%) · 80% reply fields match (40/50) · "
        "judge 3.8/5 (2/2 traces judged, 0 judge failures) · 2/3 traces"
    )
    empty = total.model_copy(
        update={"judge_mean": None, "judged_traces": 0, "fields_total": 0, "fields_matched": 0}
    )
    assert "- reply fields match (0/0) · judge - (0/2 traces judged" in aggregate_line(empty, 3)

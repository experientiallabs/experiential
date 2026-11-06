"""Checkpoint 4 statistics: degenerate bar meta-rules and a blast ranking one tag dominates."""

from __future__ import annotations

from exp.optimize.evaluation.variance.tag_monitor_contracts import MonitorSpec
from exp.optimize.evaluation.variance.tag_monitor_reward_stats import (
    BarRule,
    MinedPoint,
    MiningInputs,
    mining_stats,
)


def test_a_required_rule_that_never_fires_halts_and_one_tag_dominates_the_top() -> None:
    """``bar.observable_facts`` never fires (halt); ``error.timeout`` carries the top three."""
    inputs = MiningInputs(
        bar=(
            BarRule(rule_id="bar.consequential_choice"),
            BarRule(rule_id="bar.observable_facts"),
            BarRule(rule_id="bar.outcomes_diverge", required=False),
        ),
        steps=tuple(
            {"bar.consequential_choice": 0.9 if i % 2 else 0.2, "bar.observable_facts": 0.1}
            for i in range(6)
        ),
        points=(
            MinedPoint(point_id="p1", qualified=True, tags=("error.timeout",), rank=1, score=0.9),
            MinedPoint(point_id="p2", qualified=True, tags=("error.timeout",), rank=2, score=0.8),
            MinedPoint(point_id="p3", qualified=True, tags=("error.timeout", "x"), rank=3),
            MinedPoint(point_id="p4", qualified=False, tags=("x",)),
        ),
    )
    subjects, halts, evidence = mining_stats(inputs, MonitorSpec(top_points=3))
    rows = {row.subject: row for row in subjects}
    assert rows["bar_rule:bar.observable_facts"].flags == ("never_fires",)
    assert rows["bar_rule:bar.consequential_choice"].values["fire_rate"] == 0.5
    (halt,) = halts
    assert halt.condition == "bar_rule_degenerate" and "observable_facts" in halt.subject
    assert rows["point_tag:error.timeout"].flags == ("dominates_ranking",)
    assert rows["point_tag:x"].values["qualified_points"] == 1.0
    top = evidence["evidence:blast-top"]
    assert isinstance(top, list) and len(top) == 3

"""Monitor contracts: every proposal kind has a route at each checkpoint that makes it."""

from __future__ import annotations

from exp.optimize.evaluation.variance.tag_monitor_contracts import (
    PROPOSAL_KINDS,
    ROUTES,
    SubjectStats,
)


def test_every_allowed_proposal_kind_is_routed() -> None:
    """No checkpoint can make a proposal no path owns."""
    for checkpoint, kinds in PROPOSAL_KINDS.items():
        for kind in kinds:
            assert (checkpoint, kind) in ROUTES


def test_a_citation_names_a_value_or_a_flag() -> None:
    """``flag:<name>`` reads flags; anything else reads values."""
    row = SubjectStats(subject="question:b/q", kind="question", values={"x": 1.0}, flags=("f",))
    assert row.has("x") and row.has("flag:f") and not row.has("f") and not row.has("flag:x")

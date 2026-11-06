"""The host's checks on Astra's answer: citations, kinds and targets, never meaning."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.optimize.evaluation.variance.tag_monitor_contracts import CheckpointStats, SubjectStats
from exp.optimize.evaluation.variance.tag_monitor_review import check_review

STATS = CheckpointStats(
    checkpoint="corpus",
    scope="x",
    subjects=(
        SubjectStats(
            subject="question:b/q",
            kind="question",
            values={"fire_rate": 1.0},
            flags=("always_yes",),
        ),
        SubjectStats(subject="question:b/ok", kind="question", values={"fire_rate": 0.5}),
    ),
)


def proposal(**changes: JsonValue) -> JsonObject:
    """A well-cited retirement of the always-yes question, with ``changes`` applied."""
    row: JsonObject = {
        "kind": "retire_question",
        "targets": ["question:b/q"],
        "rationale": "always fires",
        "cites": [{"subject": "question:b/q", "stat": "flag:always_yes"}],
    }
    row.update(changes)
    return row


def finding(stat: str, severity: str = "warn") -> JsonObject:
    """A finding on the always-yes question citing ``stat``."""
    return {
        "subject": "question:b/q",
        "statement": "s",
        "severity": severity,
        "cites": [{"subject": "question:b/q", "stat": stat}],
    }


def test_unknown_citations_wrong_kinds_and_unflagged_targets_are_refused() -> None:
    """Each problem is named; a well-cited retirement passes."""
    assert check_review({"proposals": [proposal()]}, STATS) is None
    bad = check_review({"findings": [finding("spread")]}, STATS)
    assert "no statistic or flag spread" in (bad or "")
    wrong = check_review({"proposals": [proposal(kind="split_family")]}, STATS)
    assert "not a proposal kind at corpus" in (wrong or "")
    quiet = check_review({"proposals": [proposal(targets=["question:b/ok"])]}, STATS)
    assert "carries no flag" in (quiet or "")
    halt = check_review({"findings": [finding("fire_rate", "halt")]}, STATS)
    assert "cites no flag" in (halt or "")

"""Corpus tag-health statistics: flags, near-duplicate pairs and bank coverage."""

from __future__ import annotations

from exp.simulation.mining.tag_health import HealthThresholds, corpus_health
from exp.simulation.mining.tagging import TagMatrix, TagRow


def _matrix(columns: dict[str, list[float]]) -> TagMatrix:
    """A matrix over traces t0, t1, ... from per-question columns."""
    names = list(columns)
    return TagMatrix(
        bank_id="b",
        bank_version=1,
        question_ids=tuple(names),
        view="task",
        rows=tuple(
            TagRow(trace_id=f"t{i}", unit_hash=f"h{i}", values=tuple(columns[q][i] for q in names))
            for i in range(len(columns[names[0]]))
        ),
    )


def test_rare_questions_are_not_duplicates_of_each_other() -> None:
    """Two never-firing questions agree everywhere, yet only a real shared fire set duplicates."""
    half = [0.9] * 5 + [0.1] * 5
    health = corpus_health(
        [
            _matrix(
                {
                    "never_a": [0.0] * 10,
                    "never_b": [0.1] * 10,
                    "half": half,
                    "half_again": half,
                    "unsure": [0.5] * 10,
                }
            )
        ],
        excluded={"task": 3},
    )
    flags = {q.question_id: q.flags for q in health.questions}
    assert flags["never_a"] == ("always_no", "near_constant")
    assert "low_confidence" in flags["unsure"]
    assert flags["half"] == ()
    (pair,) = health.pairs
    assert (pair.left, pair.right, pair.agreement, pair.jaccard) == ("half", "half_again", 1.0, 1.0)
    (bank,) = health.banks
    assert bank.coverage == 0.5 and bank.excluded == 3 and bank.flagged == 3
    assert bank.duplicates == 1


def test_an_always_yes_question_is_flagged_at_the_configured_rate() -> None:
    """``always_rate`` decides always-yes."""
    matrix = _matrix({"mostly": [0.9] * 9 + [0.1], "other": [0.9, 0.1] * 5})
    strict = corpus_health([matrix])
    loose = corpus_health([matrix], HealthThresholds(always_rate=0.9))
    assert "always_yes" not in strict.questions[0].flags
    assert "always_yes" in loose.questions[0].flags

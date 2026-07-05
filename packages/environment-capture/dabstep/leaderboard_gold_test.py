"""Tests for recovering DABstep gold answers from official leaderboard submissions."""

from __future__ import annotations

import json
from pathlib import Path

from leaderboard_gold import canonical_gold, clean_answer, verified_answers


def test_verified_answers_keeps_only_officially_correct_rows(tmp_path: Path) -> None:
    (tmp_path / "sub_a.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"task_id": "1", "score": True, "agent_answer": "NL"},
                {"task_id": "1", "score": False, "agent_answer": "DE"},  # wrong -> dropped
                {"task_id": "2", "score": True, "agent_answer": "42"},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "sub_b.jsonl").write_text(
        json.dumps({"task_id": "1", "score": True, "agent_answer": "nl"}), encoding="utf-8"
    )
    assert verified_answers(tmp_path) == {"1": ["NL", "nl"], "2": ["42"]}


def test_clean_answer_rejects_reasoning_traces_and_long_text() -> None:
    assert clean_answer("  NL  ") == "NL"
    assert clean_answer("`91.852`") == "91.852"
    assert clean_answer("The answer is\n91.852") is None  # multi-line reasoning
    assert clean_answer("x" * 100) is None  # too long to be a bare answer


def test_canonical_gold_numeric_majority_with_tolerant_variants() -> None:
    # A dominant numeric value survives noisy reasoning-trace answers; equal-within-tolerance
    # string reps ("73.150" vs "73.15") reinforce it rather than splitting the vote.
    answers = ["73.150"] * 5 + ["73.15"] * 4 + ["The percentage is 73.15%.\nDetails:\n..."] * 3
    gold = canonical_gold(answers)
    assert gold is not None
    assert gold["numeric"] == 73.150
    assert gold["answer"] == "73.150"


def test_canonical_gold_string_majority() -> None:
    answers = ["NL"] * 6 + ["nl"] * 2 + ["The country is NL because ...\nlong trace"] * 3
    gold = canonical_gold(answers)
    assert gold is not None
    assert gold["answer"] == "NL"
    assert "numeric" not in gold
    assert gold["accept"] == ["NL"]


def test_canonical_gold_drops_low_confidence() -> None:
    # No clean answer clears the confidence bar -> the task is not answerable with confidence.
    assert canonical_gold(["a", "b", "c", "d"]) is None  # every clean answer disagrees
    assert canonical_gold(["reasoning\ntrace only"]) is None  # nothing clean at all
    assert canonical_gold([]) is None


def test_recovered_gold_self_grades_through_the_adapter(tmp_path: Path) -> None:
    # A recovered gold, written as a sidecar, must score 1.0 when its own answer is submitted.
    from environment_capture.benchmarks.dabstep import DabstepAdapter
    from environment_capture.trajectory import Task

    (tmp_path / "gold").mkdir()
    for name, answers in (("num", ["12.50"] * 5), ("str", ["Ecommerce"] * 5)):
        gold = canonical_gold(answers)
        assert gold is not None
        (tmp_path / "gold" / f"dab-train-{name}.json").write_text(
            json.dumps(gold), encoding="utf-8"
        )
        adapter = DabstepAdapter(data_root=tmp_path)
        task = Task(task_id=f"dab-train-{name}", prompt="q", data={"file_ids": []})
        assert adapter.grade(task, str(gold["answer"])) == 1.0

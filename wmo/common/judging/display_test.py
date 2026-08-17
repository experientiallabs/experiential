"""Tests for narrow-terminal rendering of the canonical rubric table."""

from __future__ import annotations

from wmo.common.judging import (
    RubricDimension,
    ScoreAnchor,
    default_task_success_axis,
    render_rubric_table,
)


def test_default_axis_table_shows_range_meaning_and_score_anchors() -> None:
    """The 0-1 default axis prints ID, range, product meaning, and both scores."""
    table = render_rubric_table((default_task_success_axis(),), width=80)

    assert table.startswith("Rubric")
    assert "1. task-success  Task success" in table
    assert "Range: 0-1" in table
    assert (
        "The agent successfully completed the task requested in the original user prompt" in table
    )
    assert "0: The agent did not complete the requested task." in table
    assert "1: The agent successfully completed the requested task." in table


def test_zero_to_four_table_wraps_on_a_narrow_terminal() -> None:
    """A 0-4 axis stays readable at 40 columns and keeps every supplied anchor."""
    axis = RubricDimension(
        dimension_id="quality",
        name="Quality",
        description=("How completely the agent solved the requested work without leaving gaps."),
        min_score=0,
        max_score=4,
        anchors=(
            ScoreAnchor(score=0, description="No useful progress on the requested work."),
            ScoreAnchor(score=2, description="Partial completion with remaining gaps."),
            ScoreAnchor(score=4, description="Complete and correct resolution."),
        ),
    )

    table = render_rubric_table((axis,), width=40)
    lines = table.splitlines()

    assert "1. quality  Quality" in table
    assert "Range: 0-4" in table
    assert "0: No useful progress" in table
    assert "2: Partial completion" in table
    assert "4: Complete and correct" in table
    assert all(len(line) <= 40 for line in lines)

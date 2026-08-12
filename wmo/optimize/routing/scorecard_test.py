"""Tests for direct imports from the public scorecard facade."""

from __future__ import annotations

from wmo.optimize.routing import scorecard, scorecard_core, scorecard_ladder
from wmo.optimize.routing.scorecard import (
    Arm,
    build_ladder,
    build_scorecard,
    rows_for_model,
    rows_for_policy,
)


def test_scorecard_facade_preserves_direct_core_and_ladder_imports() -> None:
    """The supported module remains a direct import path for both split responsibilities."""
    assert Arm is scorecard_core.Arm
    assert build_scorecard is scorecard_core.build_scorecard
    assert rows_for_model is scorecard_core.rows_for_model
    assert build_ladder is scorecard_ladder.build_ladder
    assert rows_for_policy is scorecard_ladder.rows_for_policy
    assert {
        "Arm",
        "build_ladder",
        "build_scorecard",
        "rows_for_model",
        "rows_for_policy",
    } <= set(scorecard.__all__)

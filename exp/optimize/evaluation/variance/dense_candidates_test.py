"""Candidates from definitions, error tags and linters are hypotheses mapped through sources."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.behavior_profiles import BehaviorGuidance
from exp.optimize.evaluation.variance.dense_candidates import (
    definition_candidates,
    error_tag_candidates,
    linter_candidates,
    merge_candidates,
)
from exp.optimize.evaluation.variance.dense_judge_test import fixture
from exp.simulation.mining.error_tags import ErrorTraceTags


def test_candidates_map_tagger_and_linter_sources_to_frozen_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fired and uncertain propositions and cited linters propose the tags that list them."""
    _, judge, _ = fixture(tmp_path, monkeypatch, [])
    definition = judge.definition
    summary = ErrorTraceTags(
        trace_id="episode",
        fired=("modified_evaluation", "error.unrelated"),
        uncertain=("verified_result",),
        expected_errors=1.5,
        errors_low=1,
        errors_high=2,
    )
    errors = error_tag_candidates(definition, summary, hint_refs=("e1",))
    assert [(c.tag_id, c.tagger_status, c.hint_refs) for c in errors] == [
        ("modified_evaluation", "fired", ("e1",)),
        ("verified_result", "uncertain", ("e1",)),
    ]
    guidance = BehaviorGuidance(
        locator="linter:test-tampering",
        kind="linter",
        revision=1,
        title="Test tampering",
        statement="Edits tests to pass.",
        detection="A test edit before a passing run.",
        false_positive_notes=("Fixing a test that contradicts the task.",),
    )
    (linted,) = linter_candidates(definition, (guidance,))
    assert linted.tag_id == "modified_evaluation" and "contradicts" in (linted.note or "")
    merged = merge_candidates(definition_candidates(definition), errors, errors, (linted,))
    assert len(merged) == len(definition.tags) + 3

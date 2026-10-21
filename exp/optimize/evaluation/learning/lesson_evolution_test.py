"""Lesson-library evolution from the repair-learning command: accept on gain, reject otherwise.

Offline through :class:`exp.optimize.evaluation.learning.lessons_test.LessonProvider`: the
proposer returns scripted replacement rules, and rules naming the learnable tool teach the
simulator, so a useful edit gains on held-out fit traces and a useless one regresses.
"""

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.optimize.evaluation.learning.lesson_evolution import RESULT_TYPE, EvolutionResult
from exp.optimize.evaluation.learning.lessons import load_lesson_set, load_library
from exp.optimize.evaluation.learning.lessons_test import continuation, screened_project


def evolve(
    reference: ArtifactInput, run_id: str, library: ArtifactInput | None = None
) -> JsonObject:
    """A repair-learning evolution operation over a lesson set."""
    request: JsonObject = {
        "resolution": "evolve-lessons",
        "lessons": reference.model_dump(mode="json"),
        "run_id": run_id,
        "held_out": 2,
        "protected": 1,
    }
    if library is not None:
        request["library"] = library.model_dump(mode="json")
    return {"action": "repair-learning", "learning_repair": request}


def result(output: str) -> EvolutionResult:
    """The JSON result printed after the preflight line."""
    return EvolutionResult.model_validate(json.loads(output[output.index("{") :]))


def test_an_edit_is_kept_only_on_held_out_gain_and_rejections_stay_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry-run plans paired replays; a gain accepts; a regression is retained as research."""
    root, store, provider, reference = screened_project(tmp_path, monkeypatch)
    before = len(provider.requests)
    preview = continuation(root, tmp_path, evolve(reference, "edit-one"), "--dry-run")
    assert "1 screened lessons train one proposal; 2 held-out and 1 protected" in preview
    assert "6 paired replays" in preview
    assert len(provider.requests) == before and not store.artifacts.list_ids(
        artifact_type=RESULT_TYPE
    )

    accepted = result(
        continuation(root, tmp_path, evolve(reference, "edit-one"), "--yes", "--non-interactive")
    )
    assert accepted.accepted and (accepted.held_out_gain or 0) > 0
    assert accepted.protected_regressions == ()
    child = load_library(store, ArtifactInput.model_validate(accepted.library))
    entries = {entry.entry_id: entry for entry in child.entries}
    assert entries[accepted.rules_entry].disposition == "accepted"
    assert entries[accepted.research_entry].kind == "research"
    lessons = load_lesson_set(store, reference)
    lesson = lessons.screened_in[0].lesson
    assert lesson is not None and entries[lesson].disposition == "unresolved"

    purchased = len(provider.requests)
    replay = result(
        continuation(root, tmp_path, evolve(reference, "edit-one"), "--yes", "--non-interactive")
    )
    assert replay == accepted and len(provider.requests) == purchased

    provider.edit_rules = "Replies follow the recorded format."
    rejected = result(
        continuation(
            root,
            tmp_path,
            evolve(reference, "edit-two", ArtifactInput.model_validate(accepted.library)),
            "--yes",
            "--non-interactive",
        )
    )
    assert rejected.accepted is False
    grown = load_library(store, ArtifactInput.model_validate(rejected.library))
    found = {entry.entry_id: entry for entry in grown.entries}
    assert found[rejected.rules_entry].disposition == "rejected"
    assert found[rejected.rules_entry].supersedes is None
    assert found[rejected.research_entry].kind == "research"
    assert found[accepted.rules_entry].disposition == "accepted", "the accepted edit stays"

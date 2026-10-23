"""Lesson-library evolution from the repair-learning command: accept on gain, reject otherwise.

Offline through :class:`exp.optimize.evaluation.learning.lessons_test.LessonProvider`: the
proposer returns scripted replacement rules, and rules naming the learnable tool teach the
simulator, so a useful edit gains on held-out fit traces and a useless one regresses.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, JsonValue
from exp.common.project import ArtifactStore
from exp.optimize.evaluation.learning import lesson_evolution
from exp.optimize.evaluation.learning.lesson_evolution import (
    RESULT_TYPE,
    EvolutionResult,
    preflight_evolution,
)
from exp.optimize.evaluation.learning.lessons import (
    LessonEvolutionRequest,
    load_lesson_set,
    load_library,
)
from exp.optimize.evaluation.learning.lessons_test import continuation, screened_project
from exp.optimize.evaluation.learning.traces import fit_cases
from exp.simulation.reconstruction.library import LibraryEntry
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings


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
    return {
        "action": "repair-learning",
        "learning_repair": request,
        # Lesson evolution has no cost inventory, so the operation names its ceiling.
        "spend": {"ceiling_usd": 1.0, "unpriced_attempts": 1000},
    }


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


def test_lineages_the_parent_rules_were_learned_from_never_select_the_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepted rules learned from more traces than the lessons expose those traces' lineages."""
    root, store, _, reference = screened_project(tmp_path, monkeypatch)
    accepted = result(
        continuation(root, tmp_path, evolve(reference, "edit-one"), "--yes", "--non-interactive")
    )
    request = LessonEvolutionRequest(
        resolution="evolve-lessons",
        lessons=reference,
        library=ArtifactInput.model_validate(accepted.library),
        run_id="edit-two",
        held_out=2,
        protected=1,
    )
    first = preflight_evolution(store, request)
    selected = {case.task.task_id: case for case in fit_cases(store)}
    learned = selected[first.held_out[0]]
    resolve = lesson_evolution.resolve_library_entry
    extra: list[str] = [learned.trace_id]

    def wider(artifacts: ArtifactStore, entry: LibraryEntry) -> JsonValue:
        """The accepted rules, as if also learned from the ``extra`` traces."""
        payload = resolve(artifacts, entry)
        assert isinstance(payload, dict) and isinstance(payload["knowledge"], dict)
        sources = payload["knowledge"]["source_trace_ids"]
        assert isinstance(sources, list)
        payload["knowledge"]["source_trace_ids"] = sorted([*sources, *extra])
        return payload

    monkeypatch.setattr(lesson_evolution, "resolve_library_entry", wider)
    widened = preflight_evolution(store, request)
    chosen = {
        selected[case].task.lineage_group_id for case in (*widened.held_out, *widened.protected)
    }
    assert learned.task.lineage_group_id not in chosen

    # Every other fit lineage taught the rules too: nothing is left to select on.
    extra[:] = [case.trace_id for case in selected.values()]
    with pytest.raises(ValueError, match="selection needs 3 fit lineages"):
        preflight_evolution(store, request)
    # A source the build never bound has no known lineage.
    extra[:] = ["unbound-trace"]
    with pytest.raises(ValueError, match="cite traces outside this build"):
        preflight_evolution(store, request)
    # A held-out trace is evidence this fit-only plan reserves, never a parent source.
    build = store.load_project().build
    assert build is not None
    extra[:] = [
        next(
            row.trace_id
            for row in load_completed_build_rag_lineage_bindings(store.artifacts, build)
            if row.partition == "held_out"
        )
    ]
    with pytest.raises(ValueError, match="held-out traces this fit-only plan reserves"):
        preflight_evolution(store, request)


def test_a_completed_run_id_refuses_a_different_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reusing a run ID with other settings is refused instead of replaying another result."""
    root, store, provider, reference = screened_project(tmp_path, monkeypatch)
    operation = evolve(reference, "edit-one")
    continuation(root, tmp_path, operation, "--yes", "--non-interactive")
    purchased = len(provider.requests)
    request = LessonEvolutionRequest.model_validate(operation["learning_repair"])
    changed = request.model_copy(update={"minimum_gain": 0.5})
    with pytest.raises(ValueError, match="already records a different request"):
        preflight_evolution(store, changed)
    with pytest.raises(ValueError, match="already records a different request"):
        lesson_evolution.evolve_lessons(
            store, changed, code_revision="c", clock=lambda: datetime.now(UTC)
        )
    assert len(provider.requests) == purchased

    # A result saved without its request cannot prove which request it answered.
    legacy = request.model_copy(update={"run_id": "legacy"})
    saved = store.artifacts.read(lesson_evolution._result_id(request))
    store.artifacts.write_json(
        artifact_id=lesson_evolution._result_id(legacy),
        artifact_type=RESULT_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="c"
        ),
        files={
            "result.json": json.loads(
                store.artifacts.read_bytes(saved.manifest.artifact_id, "result.json")
            )
        },
    )
    with pytest.raises(ValueError, match="without its request binding"):
        preflight_evolution(store, legacy)

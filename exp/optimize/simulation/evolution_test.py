"""Behavioral edits publish only after real paired selection, without exposing selection logs."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_bytes
from exp.common.project import ProjectConfig, ProjectStore, artifact_input
from exp.optimize.simulation.evolution import LibraryTrainingCase, evolve_library
from exp.optimize.simulation.library import LibraryCase, LibraryEvaluation
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.reconstruction.library import LibraryEntry, extend_library, retrieve_library
from exp.simulation.world_model.runtime_test import _trace


@pytest.mark.parametrize("regress", [False, True])
def test_selected_edit_is_retrievable_and_rejected_edit_is_research_only(
    tmp_path: Path, regress: bool
) -> None:
    """A second construction sees accepted rules, never a failed proposal promoted as fact."""
    now = datetime(2026, 10, 14, tzinfo=UTC)
    project = ProjectStore(tmp_path, "evolution")
    project.initialize(ProjectConfig(project_id="evolution"))
    store = project.artifacts
    manifest = store.write_json(
        artifact_id="originals",
        artifact_type="observations",
        envelope=ArtifactEnvelope(schema_version=1, created_at=now, code_revision="test"),
        files={"source.json": {"trace_id": "source", "reply": "observed"}},
    )
    library = extend_library(
        store,
        (
            LibraryEntry(
                entry_id="source",
                kind="trajectory",
                artifact=artifact_input(manifest),
                path="source.json",
                payload_sha256=sha256_bytes(store.read_bytes("originals", "source.json")),
                disposition="source",
            ),
        ),
        created_at=now,
        code_revision="test",
    )
    parent = SimulationKnowledge(source_trace_ids=("source",), rules="original rule")
    training = (
        LibraryTrainingCase(
            case_id="training", lineage_id="training-world", input={}, source_trace_ids=("source",)
        ),
    )
    held = (LibraryCase(case_id="secret-selection", lineage_id="selection-world", input={}),)
    protected = (LibraryCase(case_id="protected", lineage_id="protected-world", input={}),)
    events: list[JsonObject] = []

    def complete(body: JsonObject, logical_id: str) -> JsonObject:
        """Inspect exact proposed inputs to prevent development selection leakage."""
        sent = json.dumps(body)
        assert "secret-selection" not in sent and "protected-world" not in sent
        assert "training-world" in sent and "training failure" in sent
        return {
            "rules": "corrected general rule",
            "rationale": "Distinguish observed conditions",
            "source_trace_ids": ["source"],
        }

    def evaluate(knowledge: SimulationKnowledge, case: LibraryCase) -> LibraryEvaluation:
        """Give the child a held-out gain, optionally with a disqualifying regression."""
        edited = knowledge.rules != parent.rules
        score = 0.5 + (0.5 if edited else 0)
        if case.case_id == "protected":
            score = 0.5 if regress and edited else 1
        return LibraryEvaluation(
            score=score, status="completed", output={}, feedback="training failure"
        )

    result = evolve_library(
        parent,
        library=library,
        store=store,
        run_id="attempt-1",
        training=training,
        held_out=held,
        protected=protected,
        evaluator_sha256="a" * 64,
        complete=complete,
        evaluate=evaluate,
        record=events.append,
        created_at=now,
        code_revision="test",
    )
    assert result.decision.accepted is not regress
    assert result.decision.active.rules == (parent.rules if regress else "corrected general rule")
    assert len(library.entries) == 1 and len(result.library.entries) == 3
    seen: list[LibraryEntry] = []

    def select(query: JsonObject, entries: tuple[LibraryEntry, ...]) -> tuple[str, ...]:
        """Simulate the next construction's normal prior-library access seam."""
        seen.extend(entries)
        return tuple(entry.entry_id for entry in entries)

    context = retrieve_library(store, result.library, (_trace(now),), select=select, top_k=3)
    assert any(entry.kind == "rule" for entry in seen) is not regress
    assert any(entry.kind == "research" for entry in seen)
    assert result.note.disposition == ("rejected" if regress else "accepted")
    assert context["selected"] and events
    with pytest.raises(ValueError, match="already registered"):
        evolve_library(
            parent,
            library=library,
            store=store,
            run_id="attempt-1",
            training=training,
            held_out=held,
            protected=protected,
            evaluator_sha256="a" * 64,
            complete=complete,
            evaluate=evaluate,
            record=events.append,
            created_at=now,
            code_revision="test",
        )

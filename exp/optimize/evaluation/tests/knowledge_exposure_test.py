"""Knowledge pinned in a simulator never exposes the evaluated task's own evidence."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_bytes
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.tests.fidelity_memory_test import RoutedTransport, memory_project
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.reconstruction.library import LibraryEntry
from exp.simulation.reconstruction.revision import ResearchNote
from exp.simulation.retrieval import load_rag_index
from exp.simulation.world_model.artifact import (
    GroundedWorldModelArtifact,
    persist_grounded_world_model,
)
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

SECRET = "HELDOUT_REPLY_SECRET: reply-0"
OWN = "held-out-trace"


def bound(
    root: Path,
    knowledge: SimulationKnowledge | None = None,
    case: str = "",
    *,
    library_source: JsonObject | None = None,
) -> FidelitySpec:
    """A held-out task (sourced from ``OWN``) on the fixture simulator with this knowledge.

    ``same-lineage`` puts the task in the lineage of the indexed trace ``observed`` and binds
    a simulator without recorded-evidence retrieval, so only knowledge could expose it.
    """
    project, base = memory_project(root)
    artifact = load_grounded_world_model_artifact(project.artifacts, base.world_model)
    if library_source is not None:
        knowledge = SimulationKnowledge(transfer_context=_selected(project, source=library_source))
    if case in ("library", "dependency-only"):
        knowledge = SimulationKnowledge(
            source_trace_ids=("observed",),
            transfer_context=_selected(project, own_source=case == "library"),
        )
    model = persist_grounded_world_model(
        project.artifacts,
        artifact.serving_rag,
        model_alias=artifact.model_alias,
        model=artifact.model,
        created_at=artifact.created_at,
        code_revision="test",
        top_k=artifact.top_k,
        knowledge=knowledge,
        original_evidence="none" if case == "same-lineage" else artifact.original_evidence,
    )
    data = base.model_dump(mode="json")
    data["world_model"] = artifact_input(model.manifest).model_dump(mode="json")
    data["task"]["source_trace_ids"] = [OWN]
    data["task"]["origin"] = "mined"
    data["output_protocol"] = "json-tool"
    if case == "same-lineage":
        data["task"]["lineage_group_id"] = _indexed_lineage(project, artifact)
    spec = FidelitySpec.model_validate(data)
    assert spec.task.partition == "held_out"
    return spec


def _indexed_lineage(project: ProjectStore, artifact: GroundedWorldModelArtifact) -> str:
    """Task lineage of the fixture's indexed trace ``observed``."""
    index = load_rag_index(project.artifacts, artifact.serving_rag.artifact_id)
    return next(row.lineage_id for row in index.transitions if row.trace_id == "observed")


def _selected(
    project: ProjectStore, *, own_source: bool = True, source: JsonObject | None = None
) -> JsonObject:
    """A prior-library selection whose entry derives from (or only depends on) held-out data."""
    payload = {
        "trajectory": {
            "task": "Inspect current entries",
            "task_lineage": "other" if own_source else "independent",
            "source_trace_ids": [OWN if own_source else "observed"],
            "actions": [{"name": "inspect", "arguments": {"key": "current"}}],
            "observed_replies": [SECRET],
        }
    }
    if source is not None:
        payload = {"trajectory": source}
    manifest = project.artifacts.write_json(
        artifact_id="held-out-source",
        artifact_type="consumed-trajectory",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"trajectory.json": payload},
    )
    ref = artifact_input(manifest)
    stored = project.artifacts.read_bytes(ref.artifact_id, "trajectory.json")
    entry = LibraryEntry(
        entry_id="held-out-source",
        kind="trajectory",
        artifact=ref,
        path="trajectory.json",
        payload_sha256=sha256_bytes(stored),
        disposition="source",
        derived_from=ref if own_source else None,
        depends_on=() if own_source else (ref,),
    )
    return {"selected": [{"entry": entry.model_dump(mode="json"), "payload": payload}]}


NOTE = ResearchNote(
    id="leak", content=SECRET, investigation_sha256="a" * 64, source_trace_ids=(OWN,)
)


@pytest.mark.parametrize(
    ("knowledge", "case", "message"),
    [
        (SimulationKnowledge(source_trace_ids=(OWN,), rules=SECRET), "", "own recorded"),
        (SimulationKnowledge(rules=SECRET), "", "no recorded provenance"),
        (
            SimulationKnowledge(source_trace_ids=("observed",), research_history=(NOTE,)),
            "",
            "own recorded",
        ),
        (None, "library", "own recorded"),
        (
            SimulationKnowledge(
                source_trace_ids=("observed",),
                research_history=(NOTE.model_copy(update={"source_trace_ids": ()}),),
            ),
            "",
            "research note leak",
        ),
    ],
    ids=["constructor", "unknown", "research-note", "library", "unprovenanced-note"],
)
def test_knowledge_exposing_the_task_is_refused_before_any_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    knowledge: SimulationKnowledge | None,
    case: str,
    message: str,
) -> None:
    """Every predictor-visible knowledge channel is checked; refusal precedes credentials."""
    monkeypatch.delenv("FIDELITY_TEST_KEY", raising=False)
    spec = bound(tmp_path, knowledge, case)
    project = ProjectStore(tmp_path, "fidelity-test")
    with pytest.raises(ValueError, match=message):
        prepare_fidelity_evaluation(project, spec, code_revision="test")


@pytest.mark.parametrize(
    "knowledge",
    [
        SimulationKnowledge(rules="Orders ship in two days.", authored=True),
        SimulationKnowledge(source_trace_ids=("observed",), rules="Orders ship in two days."),
    ],
    ids=["authored", "other-traces"],
)
def test_authored_or_independent_knowledge_is_accepted(
    tmp_path: Path, knowledge: SimulationKnowledge
) -> None:
    """Hand-written rules and knowledge from other traces remain valid held-out simulators."""
    spec = bound(tmp_path, knowledge)
    project = ProjectStore(tmp_path, "fidelity-test")
    assert prepare_fidelity_evaluation(project, spec, code_revision="test").spec == spec


@pytest.mark.parametrize(
    ("knowledge", "case", "message"),
    [
        (SimulationKnowledge(source_trace_ids=("observed",), rules=SECRET), "same-lineage", "own"),
        (SimulationKnowledge(source_trace_ids=("ghost",), rules=SECRET), "", "trace ghost"),
        (None, "dependency-only", "prior-library entry held-out-source"),
    ],
    ids=["same-lineage-trace", "nonexistent-trace", "dependency-only-library"],
)
def test_lineages_resolve_and_unknown_sources_are_never_verified(
    tmp_path: Path, knowledge: SimulationKnowledge | None, case: str, message: str
) -> None:
    """Knowledge trace IDs resolve to lineages; unresolvable or borrowed provenance refuses."""
    spec = bound(tmp_path, knowledge, case)
    project = ProjectStore(tmp_path, "fidelity-test")
    with pytest.raises(ValueError, match=message):
        prepare_fidelity_evaluation(project, spec, code_revision="test")


def test_a_registration_made_before_the_guard_is_refused_on_reuse_and_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved registrations are rechecked; none can buy a prediction past the guard."""
    spec = bound(tmp_path, SimulationKnowledge(rules=SECRET))
    project = ProjectStore(tmp_path, "fidelity-test")
    with monkeypatch.context() as unguarded:
        unguarded.setattr(fidelity, "require_unexposed_task", lambda *_args: None)
        prepared = prepare_fidelity_evaluation(project, spec, code_revision="test")
    with pytest.raises(ValueError, match="no recorded provenance"):
        prepare_fidelity_evaluation(project, spec, code_revision="test")
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    wire = RoutedTransport([])
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    with pytest.raises(ValueError, match="no recorded provenance"):
        execute_fidelity_evaluation(project, prepared)
    assert not wire.requests


@pytest.mark.parametrize(
    ("source", "case", "message"),
    [
        ({}, "", "no recorded provenance"),
        ({"source_trace_ids": ["observed"]}, "", "no recorded provenance"),
        ({"source_trace_ids": ["observed"]}, "same-lineage", "own recorded"),
        (
            {"task_lineage": "independent", "source_trace_ids": ["ghost"]},
            "",
            "trace ghost",
        ),
    ],
    ids=["empty-source", "missing-lineage", "sibling-trace", "unknown-trace"],
)
def test_library_source_contents_must_establish_independence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: JsonObject, case: str, message: str
) -> None:
    """A valid artifact digest cannot substitute for its source's missing provenance."""
    monkeypatch.delenv("FIDELITY_TEST_KEY", raising=False)
    spec = bound(tmp_path, case=case, library_source=source)
    project = ProjectStore(tmp_path, "fidelity-test")
    with pytest.raises(ValueError, match=message):
        prepare_fidelity_evaluation(project, spec, code_revision="test")


def test_resolved_independent_library_source_is_accepted(tmp_path: Path) -> None:
    """A library entry with known independent source evidence remains usable."""
    spec = bound(
        tmp_path,
        library_source={"task_lineage": "source", "source_trace_ids": ["observed"]},
    )
    project = ProjectStore(tmp_path, "fidelity-test")
    assert prepare_fidelity_evaluation(project, spec, code_revision="test").spec == spec

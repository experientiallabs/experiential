"""Library selection cannot promote a same-name but inapplicable prior record."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_bytes
from exp.common.project import ProjectConfig, ProjectStore, artifact_input
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.reconstruction.library import LibraryEntry
from exp.simulation.retrieval.library_jev import JEVLibrarySelector, LibraryJEVPolicy


@pytest.mark.parametrize("size", [1, 20])
def test_only_directly_relevant_admitted_sources_are_selected(tmp_path: Path, size: int) -> None:
    """An uncertain/irrelevant prior source is excluded even though capacity remains."""
    project = ProjectStore(tmp_path, "library-selection")
    project.initialize(ProjectConfig(project_id="library-selection"))
    store = project.artifacts
    manifest = store.write_json(
        artifact_id="prior",
        artifact_type="prior-observations",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"prior.json": {"tool": "get_user_details", "observation": "actual"}},
    )
    entry = LibraryEntry(
        entry_id="prior",
        kind="trajectory",
        artifact=artifact_input(manifest),
        path="prior.json",
        payload_sha256=sha256_bytes(store.read_bytes("prior", "prior.json")),
        disposition="source",
    )
    events: list[JsonObject] = []

    def decide(request: DecisionRequest) -> JsonObject:
        """Return the native wire shape with a deliberately indirect relevance score."""
        assert isinstance(request.state, dict)
        assert request.state["prior_records"]
        answers: JsonObject = {"dependency": {"type": "choice", "choice": "tool behavior"}}
        for key in request.questions:
            if key != "dependency":
                answers[key] = {"type": "score", "score": 0 if key.startswith("history") else 1}
        assert len(request.questions) <= 31
        return {"answers": answers}

    selector = JEVLibrarySelector(
        store=store, policy=LibraryJEVPolicy(), decide=decide, record=events.append
    )
    entries = tuple(entry.model_copy(update={"entry_id": f"prior-{i}"}) for i in range(size))
    assert selector({"target_traces": []}, entries) == ()
    assert len(events) == (3 if size == 1 else 4) and events[-1]["selected"] == []
    if size > 15:
        assert events[0]["eligible_count"] == size
        shortlist = events[0]["shortlist"]
        assert isinstance(shortlist, list) and len(shortlist) == 15

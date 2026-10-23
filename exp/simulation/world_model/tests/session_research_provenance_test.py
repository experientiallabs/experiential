"""Dependencies come from committed prediction evidence, so they survive a restart."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from exp.common.core.artifacts import JsonObject, JsonValue, sha256_json
from exp.common.project import ProjectConfig, ProjectStore
from exp.runtime.sessions.store import SessionBinding, SessionStore
from exp.simulation.retrieval.selection import AllOriginals
from exp.simulation.world_model.memory_tree_test import source
from exp.simulation.world_model.session_research import OnlineResearchPolicy, SessionResearch


def test_revealed_overview_sources_are_inherited_from_committed_evidence(tmp_path: Path) -> None:
    """A restarted session knows only stored state; recorded overview sources still count."""
    project = ProjectStore(tmp_path, "provenance")
    project.initialize(ProjectConfig(project_id="provenance"))
    store = project.artifacts
    ref_a, ref_c = source(store, "a"), source(store, "c")
    now = datetime.now(UTC)
    sessions = SessionStore(tmp_path / "sessions.sqlite")
    binding = SessionBinding(
        deadline=now + timedelta(hours=1),
        target_id="fixture@1",
        initialization_receipt_sha256=sha256_json("init"),
        knowledge_sha256=sha256_json("knowledge"),
        corpus_sha256=sha256_json("corpus"),
        representation_sha256=sha256_json("representation"),
        execution_policy_sha256=sha256_json("policy"),
    )
    transitions: list[JsonValue] = []
    sessions.create_session("B", binding, {"transitions": transitions})
    selector = AllOriginals()
    answers: list[JsonObject] = [
        {
            "discrepancy": "wrong",
            "hypothesis": "C retired",
            "scope": "read",
            "uncertainty": "test",
            "supersedes": ["C-note"],
            "transferable": True,
            "observable_cue": "the same tool and arguments",
        },
        {
            "discrepancy": "wrong",
            "hypothesis": "A performed 0/3 in the history now shown",
            "scope": "read",
            "uncertainty": "test",
            "supersedes": [],
            "transferable": True,
            "observable_cue": "the same tool and arguments",
        },
    ]

    def restarted() -> SessionResearch:
        """A fresh research owner holding no overview resolver or initial sources."""
        return SessionResearch(
            session_id="B",
            policy=OnlineResearchPolicy(
                feedback_source="feedback", selection_policy_sha256=selector.identity
            ),
            store=sessions,
            artifacts=store,
            complete=lambda body, identity: answers.pop(0),
            lock_path=tmp_path / "research.lock",
            clock=lambda: datetime.now(UTC),
            code_revision="fixture",
            selection=selector,
            prior={"C-note": {"id": "C-note", "investigation": {"hypothesis": "C"}}},
            prior_identity={"version": 1},
            prior_sources={"C-note": (ref_c,)},
        )

    delivered = [[ref_c], [ref_a, ref_c]]
    records: list[JsonObject] = []
    for step in range(2):
        transitions.append(
            {
                "operation_id": f"op-{step}",
                "actions": [{"name": "read"}],
                "replies": ["wrong"],
                "evidence": {
                    "session_research_selection": {"selected_ids": []},
                    "memory_overview_sources": [r.model_dump(mode="json") for r in delivered[step]],
                },
            }
        )
        decision = sessions.begin_step("B", f"op-{step}", sha256_json(step), step, binding)
        claim = decision.claim
        assert claim is not None
        sessions.mark_provider_started(claim, 10)
        sessions.record_proposal(claim, {})
        sessions.complete_step(claim, {"transitions": transitions}, {})
        records.append(restarted().observe(f"op-{step}", "right", "feedback"))
    first, second = (_dependency_ids(record) for record in records)
    assert ref_a.artifact_id not in first
    assert ref_a.artifact_id in second


def _dependency_ids(record: JsonObject) -> set[str]:
    """Artifact identities a research record declared as dependencies."""
    rows = record["depends_on"]
    assert isinstance(rows, list)
    return {str(row["artifact_id"]) for row in rows if isinstance(row, dict)}

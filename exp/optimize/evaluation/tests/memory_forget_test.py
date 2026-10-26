"""Forgetting summaries: generations, checkpoint-stable views, rebuilds and recovery."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.project import ArtifactStore, ProjectStore
from exp.optimize.evaluation.tests.fidelity_memory_test import memory_project, trajectory
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    MemoryVersion,
    forget_summary,
    memory_head,
    memory_view,
    publish_trajectory,
    retired_lines,
    summary_generations,
)
from exp.simulation.world_model.memory_tree import (
    build_pending,
    overview,
    read_summary,
    summary_id,
)

NOW = datetime(2026, 10, 15, tzinfo=UTC)
LINEAGE = "forget-memory"


class Compressor:
    """Deterministic compression: joins the supplied texts, and counts calls."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls: list[str] = []

    def __call__(self, request: JsonObject, logical_id: str) -> JsonObject:
        """Summarize the request's lines or halves by joining their texts."""
        self.calls.append(logical_id)
        messages = request["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        content = json.loads(str(messages[1]["content"]))
        rows = content.get("lines") or content.get("halves") or []
        text = " | ".join(str(row["text"]) for row in rows)
        return {"summary": ("S: " + text)[:200]}


class Memory:
    """A memory lineage grown by direct publication, as the learning lifecycle does it."""

    def __init__(self, tmp_path: Path) -> None:
        """Bind the fixture simulator and memory selector."""
        project, base = memory_project(tmp_path)
        spec = trajectory(base, "seed", "seed", "learn")
        self.project: ProjectStore = project
        self.store: ArtifactStore = project.artifacts
        self.world_model: ArtifactInput = spec.world_model
        self.selector = str(spec.memory_selection_identity)

    def research(self, identity: str, supersedes: tuple[str, ...] = ()) -> str:
        """Persist one session research record with its investigation."""
        self.store.write_json(
            artifact_id=identity,
            artifact_type="session-research",
            envelope=ArtifactEnvelope(schema_version=1, created_at=NOW, code_revision="fixture"),
            files={
                "record.json": {
                    "id": identity,
                    "investigation": {
                        "hypothesis": f"{identity} hypothesis",
                        "supersedes": list(supersedes),
                    },
                    "depends_on": [],
                }
            },
        )
        return identity

    def learn(self, task: str, hypothesis: str, supersedes: tuple[str, ...] = ()) -> MemoryVersion:
        """Consume one trajectory with one observed line and one hypothesis line."""
        rid = self.research(f"research-{task}", supersedes)
        version, consumed = publish_trajectory(
            self.store,
            lineage=LINEAGE,
            parent=memory_head(self.store, LINEAGE),
            world_model=self.world_model,
            selection_policy_sha256=self.selector,
            trajectory={
                "task_lineage": task,
                "actions": [{"name": "inspect", "arguments": {"key": task}}],
                "observed_replies": [f"{task}-reply"],
            },
            research=[rid],
            lines=[("observed", f"[observed] {task}", task), ("hypothesis", hypothesis, rid)],
            created_at=NOW,
            code_revision="fixture",
            publisher=f"run-{task}",
        )
        assert consumed
        return version

    def maintain(self, complete: Compressor) -> list[str]:
        """Rebuild pending blocks at the head's generations, as maintenance does."""
        head = memory_head(self.store, LINEAGE)
        assert head is not None
        built, _ = build_pending(
            self.store,
            LINEAGE,
            head.log_length or 0,
            complete,
            limit=16,
            created_at=NOW,
            code_revision="fixture",
            generations=summary_generations(head),
            retired=retired_lines(self.store, head),
        )
        return built

    def view(self, version: MemoryVersion, budget: int = 1) -> str:
        """The overview a reader pinned to ``version`` sees, as canonical text."""
        retired = set(memory_view(self.store, version, Eligibility()).retired)
        rows = overview(
            self.store,
            LINEAGE,
            version.log_length or 0,
            budget,
            admits_line=lambda line: line.origin not in retired,
            generations=summary_generations(version),
        )
        return json.dumps(rows, sort_keys=True)


def test_superseded_hypothesis_leaves_every_current_summary_and_old_views_stay(
    tmp_path: Path,
) -> None:
    """Retiring a line forgets its block and ancestors; checkpoints keep their exact view."""
    memory = Memory(tmp_path)
    memory.learn("alpha", "[unresolved] alpha WRONG-CLAIM")
    v2 = memory.learn("beta", "[unresolved] beta claim")
    compressor = Compressor()
    assert len(memory.maintain(compressor)) == 3  # #0-1, #2-3, then #0-3
    before = memory.view(v2)
    assert "WRONG-CLAIM" in before and '"#0-3"' in before

    v3 = memory.learn("gamma", "[unresolved] gamma claim", supersedes=("research-alpha",))
    assert v3.forgotten == ("0-2", "0-4")  # the retired line's block and its ancestor only
    assert summary_generations(v3) == {(0, 2): 1, (0, 4): 1}
    assert retired_lines(memory.store, v3) == frozenset({1})
    # Until rebuilt, a forgotten block opens into its children; the retired line is hidden.
    assert "WRONG-CLAIM" not in memory.view(v3, budget=4)

    rebuilt = memory.maintain(compressor)
    assert summary_id(LINEAGE, 0, 2, 1) in rebuilt and summary_id(LINEAGE, 0, 4, 1) in rebuilt
    fresh = read_summary(memory.store, LINEAGE, 0, 2, 1)
    assert fresh is not None and fresh.omitted == (1,) and "WRONG-CLAIM" not in fresh.text
    current = memory.view(v3, budget=2)
    assert "WRONG-CLAIM" not in current and '"#0-3"' in current
    # The original generation is untouched, and the older checkpoint renders identically.
    old = read_summary(memory.store, LINEAGE, 0, 2)
    assert old is not None and "WRONG-CLAIM" in old.text
    assert memory.view(v2) == before


def test_a_reader_that_still_admits_an_omitted_line_opens_the_block(tmp_path: Path) -> None:
    """A rebuilt summary is shown only to readers that also treat its omitted line as retired."""
    memory = Memory(tmp_path)
    memory.learn("alpha", "[unresolved] alpha WRONG-CLAIM")
    memory.learn("beta", "[unresolved] beta claim")
    v3 = memory.learn("gamma", "[unresolved] gamma claim", supersedes=("research-alpha",))
    memory.maintain(Compressor())
    rows = overview(
        memory.store,
        LINEAGE,
        v3.log_length or 0,
        6,
        generations=summary_generations(v3),
    )
    assert not [row for row in rows if row["kind"] == "summary" and row["id"] in ("#0-1", "#0-3")]
    assert any(row.get("text") == "[unresolved] alpha WRONG-CLAIM" for row in rows)


def test_explicit_forget_applies_from_the_next_version_with_ancestors(tmp_path: Path) -> None:
    """An operator forget is recorded, folded into the next version, and rebuilt once."""
    memory = Memory(tmp_path)
    memory.learn("alpha", "[unresolved] alpha claim")
    v2 = memory.learn("beta", "[unresolved] beta BAD-SUMMARY")
    compressor = Compressor()
    memory.maintain(compressor)
    with pytest.raises(ValueError, match="aligned block"):
        forget_summary(
            memory.store, LINEAGE, (1, 3), reason="unaligned", created_at=NOW, code_revision="x"
        )
    record = forget_summary(
        memory.store, LINEAGE, (2, 4), reason="bad summary", created_at=NOW, code_revision="x"
    )
    assert record == forget_summary(
        memory.store, LINEAGE, (2, 4), reason="bad summary", created_at=NOW, code_revision="x"
    )
    assert memory_head(memory.store, LINEAGE) == v2, "a published checkpoint never changes"
    v3 = memory.learn("gamma", "[unresolved] gamma claim")
    assert v3.forgotten == ("2-4", "0-4")
    calls = len(compressor.calls)
    rebuilt = memory.maintain(compressor)
    assert summary_id(LINEAGE, 2, 4, 1) in rebuilt and summary_id(LINEAGE, 0, 4, 1) in rebuilt
    assert read_summary(memory.store, LINEAGE, 2, 4) is not None  # generation 0 kept
    assert len(compressor.calls) - calls == 3  # #2-3 and #0-3 again, plus the new #4-5
    v4 = memory.learn("delta", "[unresolved] delta claim")
    assert v4.forgotten is None and summary_generations(v4) == summary_generations(v3)


def test_crash_between_forget_and_rebuild_recovers_exactly_once(tmp_path: Path) -> None:
    """A version that forgot blocks is durable; the next maintenance rebuilds them once."""
    memory = Memory(tmp_path)
    memory.learn("alpha", "[unresolved] alpha WRONG-CLAIM")
    memory.learn("beta", "[unresolved] beta claim")
    memory.maintain(Compressor())

    v3 = memory.learn("gamma", "[unresolved] gamma claim", supersedes=("research-alpha",))
    # The forget is committed with the version; the process dies before any maintenance.
    assert v3.forgotten
    assert read_summary(memory.store, LINEAGE, 0, 2, 1) is None

    recovering = Compressor()
    memory.maintain(recovering)
    rebuilt = [call for call in recovering.calls if summary_id(LINEAGE, 0, 2, 1) in call]
    assert len(rebuilt) == 1
    again = Compressor()
    assert memory.maintain(again) == [] and again.calls == []
    assert "WRONG-CLAIM" not in memory.view(v3, budget=2)

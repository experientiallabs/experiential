"""Frozen actor memory filters both original records and overview rows by provenance."""

import json
from datetime import UTC, datetime
from pathlib import Path

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.council.memory import CouncilMemory, read_council_memory
from exp.optimize.evaluation.tests.fidelity_memory_test import memory_project
from exp.simulation.world_model.experience_memory import memory_head, publish_trajectory
from exp.simulation.world_model.memory_tree import build_pending


def test_excluded_owner_cannot_enter_records_or_overview(tmp_path: Path) -> None:
    """The same immutable checkpoint exposes a lesson only to an eligible task lineage."""
    project, base = memory_project(tmp_path)
    publish_trajectory(
        project.artifacts,
        lineage="bank",
        parent=None,
        world_model=base.world_model,
        selection_policy_sha256="0" * 64,
        trajectory={
            "task_lineage": "owner-a",
            "actions": [{"name": "inspect", "arguments": {"key": "a"}}],
            "observed_replies": ["private-observation-a"],
        },
        research=(),
        lines=(("observed", "private-lesson-a", "owner-a"),),
        created_at=datetime.now(UTC),
        code_revision="fixture",
        publisher="source-a",
    )
    before = memory_head(project.artifacts, "bank")
    config = CouncilMemory(lineage="bank", checkpoint=1, overview_lines=2)
    allowed = read_council_memory(project.artifacts, config, base.world_model, ("owner-b",))
    withheld = read_council_memory(project.artifacts, config, base.world_model, ("owner-a",))
    assert "private-observation-a" in json.dumps(allowed.records)
    assert "private-lesson-a" in json.dumps(allowed.overview)
    assert withheld.records == {}
    assert "private-lesson-a" not in json.dumps(withheld.overview)
    assert allowed.snapshot == withheld.snapshot
    assert memory_head(project.artifacts, "bank") == before


def test_council_overview_is_frozen_before_later_compression(tmp_path: Path) -> None:
    """A saved council must not receive a different summary as background compression finishes."""
    project, base = memory_project(tmp_path)
    now = datetime.now(UTC)
    version, _ = publish_trajectory(
        project.artifacts,
        lineage="bank",
        parent=None,
        world_model=base.world_model,
        selection_policy_sha256="0" * 64,
        trajectory={"task_lineage": "source", "actions": [], "observed_replies": []},
        research=(),
        lines=(
            ("observed", "first observation", "source"),
            ("observed", "second observation", "source"),
        ),
        created_at=now,
        code_revision="fixture",
        publisher="source",
    )
    config = CouncilMemory(lineage="bank", checkpoint=1, overview_lines=1)
    before = read_council_memory(project.artifacts, config, base.world_model, ())

    def complete(body: JsonObject, logical_id: str) -> JsonObject:
        """Return a late summary without changing the consumed memory version."""
        return {"summary": "compressed observations"}

    built, _ = build_pending(
        project.artifacts,
        "bank",
        version.log_length or 0,
        complete,
        limit=1,
        created_at=now,
        code_revision="fixture",
    )
    assert built
    after = read_council_memory(project.artifacts, config, base.world_model, ())
    assert after == before
    assert memory_head(project.artifacts, "bank") == version

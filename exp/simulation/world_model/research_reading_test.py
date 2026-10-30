"""Reader archives survive a crash after packet publication without changing evidence."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.project import ArtifactStore, ProjectPaths
from exp.simulation.retrieval.evidence_reading import ReadingResult
from exp.simulation.world_model.memory_tree_test import source
from exp.simulation.world_model.research_reading import archive_reading


def test_archive_replays_exact_packet_and_refuses_changed_coverage(tmp_path: Path) -> None:
    """Publication time may change on resume; source coverage and dependencies may not."""
    store = ArtifactStore(ProjectPaths(tmp_path, "reading"))
    ref = source(store, "a")
    result = ReadingResult(
        proposal={},
        inspected_ids=("a",),
        unread_ids=(),
        passages=(),
        objectives=("Support?",),
        skipped={},
        stop_reason="catalog",
    )
    now = datetime.now(UTC)
    receipt = archive_reading(
        store, "investigation", result, (ref,), created_at=now, code_revision="test"
    )
    assert (
        archive_reading(
            store,
            "investigation",
            result,
            (ref,),
            created_at=now + timedelta(hours=1),
            code_revision="test",
        )
        == receipt
    )
    with pytest.raises(ValueError, match="changed"):
        archive_reading(
            store,
            "investigation",
            result.model_copy(update={"inspected_ids": ()}),
            (ref,),
            created_at=now,
            code_revision="test",
        )
    with pytest.raises(ValueError, match="changed"):
        archive_reading(store, "investigation", result, (), created_at=now, code_revision="test")

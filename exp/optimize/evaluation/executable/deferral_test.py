"""Contention receipts cannot authorize retry for another orchestration intent."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.executable.deferral import (
    deferred_execution,
    record_execution_deferral,
)


@pytest.mark.parametrize("wrong_parent", [False, True])
def test_deferral_binds_exact_parent_and_remains_immutable(
    tmp_path: Path, wrong_parent: bool
) -> None:
    """A verified marker is idempotent; a marker attached to another intent is refused."""
    store = ProjectStore(tmp_path, "case").artifacts
    envelope = ArtifactEnvelope(
        schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
    )
    for cell in ("first", "second"):
        store.write_json(
            artifact_id=cell + "-intent",
            artifact_type="fixture-intent",
            envelope=envelope,
            files={"intent.json": {"cell": cell}},
        )
    if wrong_parent:
        store.write(
            artifact_id="first-deferred",
            artifact_type="execution-deferral",
            envelope=envelope.model_copy(
                update={"inputs": (artifact_input(store.read("second-intent").manifest),)}
            ),
            files={"reason.txt": b"child lease busy"},
        )
        with pytest.raises(ValueError, match="exact parent intent"):
            deferred_execution(store, "first")
    else:
        assert not deferred_execution(store, "first")
        record_execution_deferral(store, "first")
        original = deferred_execution(store, "first")
        record_execution_deferral(store, "first")
        assert deferred_execution(store, "first") == original
        assert not deferred_execution(store, "second")

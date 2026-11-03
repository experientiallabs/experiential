"""Tests for causal order between saved artifacts."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.common.project.causality import cites, require_cites

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def saved(
    store: ArtifactStore, name: str, *inputs: ArtifactInput, at: datetime = NOW
) -> ArtifactInput:
    """Save one artifact citing ``inputs`` and return its exact reference."""
    manifest = store.write_json(
        artifact_id=name,
        artifact_type="fixture",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=at,
            code_revision="test",
            inputs=tuple(sorted(inputs, key=lambda item: item.artifact_id)),
        ),
        files={"value.json": {"name": name}},
    )
    return artifact_input(manifest)


def test_a_citation_orders_artifacts_whatever_their_clocks_say(tmp_path: Path) -> None:
    """A later artifact with an earlier clock still follows what it cites, and never the reverse."""
    store = ProjectStore(tmp_path, "case").artifacts
    first = saved(store, "first", at=NOW + timedelta(days=1))
    second = saved(store, "second", first, at=NOW - timedelta(days=1))
    assert cites(store, "second", first)
    assert not cites(store, "first", second)


def test_citations_are_followed_transitively_and_only_while_they_verify(tmp_path: Path) -> None:
    """A chain of exact citations orders its ends; a stale digest breaks the chain."""
    store = ProjectStore(tmp_path, "case").artifacts
    root = saved(store, "root")
    middle = saved(store, "middle", root)
    saved(store, "leaf", middle)
    assert cites(store, "leaf", root)
    stale = ArtifactInput(artifact_id="middle", sha256="0" * 64)
    saved(store, "orphan", stale)
    assert not cites(store, "orphan", root)
    assert not cites(store, "leaf", ArtifactInput(artifact_id="root", sha256="1" * 64))
    assert not cites(store, "missing", root)


def test_require_cites_refuses_with_the_callers_message(tmp_path: Path) -> None:
    """An uncited predecessor refuses with the actionable message the caller supplied."""
    store = ProjectStore(tmp_path, "case").artifacts
    earlier = saved(store, "earlier")
    saved(store, "unrelated")
    saved(store, "follower", earlier)
    require_cites(store, "follower", earlier, refusal="unused")
    with pytest.raises(ValueError, match="cite it first"):
        require_cites(store, "unrelated", earlier, refusal="cite it first")

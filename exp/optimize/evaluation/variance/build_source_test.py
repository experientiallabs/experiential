"""A variance build is keyed on its inputs and its own sources, not on the repository revision."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance import build_source
from exp.optimize.evaluation.variance.build_records import VarianceBuildSpec


def test_an_unrelated_commit_replays_the_build_but_a_protocol_or_source_change_rekeys_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: a new revision alone keeps the saved build; build semantics re-key it.

    The revision is recorded metadata only. Bumping the build protocol or editing a module in
    the build's semantic closure changes the identity, so a stale build is never replayed.
    """
    spec = VarianceBuildSpec()
    before = spec.identity("revision-a")
    assert spec.identity("revision-b") == before
    sources = build_source.build_sources()
    assert "exp.optimize.evaluation.variance.family_scopes" in sources
    assert "exp.simulation.mining.environment_families" in sources
    assert not any(m.startswith("exp.runtime.") for m in sources)

    build_source.build_source_identity.cache_clear()
    monkeypatch.setattr(build_source, "VARIANCE_BUILD_PROTOCOL", "variance-build-v2")
    try:
        assert spec.identity("revision-a") != before
    finally:
        build_source.build_source_identity.cache_clear()
    monkeypatch.undo()
    real = build_source.build_sources
    edited = dict(real())
    edited["exp.simulation.mining.tag_clusters"] = Path(build_source.__file__)
    monkeypatch.setattr(build_source, "build_sources", lambda: edited)
    build_source.build_source_identity.cache_clear()
    try:
        assert spec.identity("revision-a") != before
    finally:
        monkeypatch.undo()
        build_source.build_source_identity.cache_clear()
    assert spec.identity("revision-c") == before

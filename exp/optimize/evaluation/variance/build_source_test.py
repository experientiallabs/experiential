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


PIN = Path(__file__).with_name("build_source_pin.json")
"""The build's semantic sources as of the last deliberate re-key: protocol, identity, digests.

Regenerate it with :func:`write_pin`, only together with a deliberate
:data:`build_source.VARIANCE_BUILD_PROTOCOL` bump.
"""


def current_pin() -> dict[str, object]:
    """Return the build's protocol, source identity and per-module source digests now."""
    from exp.common.core.artifacts import sha256_bytes

    build_source.build_source_identity.cache_clear()
    return {
        "protocol": build_source.VARIANCE_BUILD_PROTOCOL,
        "identity": build_source.build_source_identity(),
        "sources": {
            module: sha256_bytes(path.read_bytes())
            for module, path in sorted(build_source.build_sources().items())
        },
    }


def write_pin() -> None:
    """Record the current build sources as the deliberate pin (after a protocol bump)."""
    import json

    PIN.write_text(json.dumps(current_pin(), indent=2, sort_keys=True) + "\n")


def test_no_build_source_changes_without_a_deliberate_protocol_bump() -> None:
    """Intent: editing a module in the build's import closure never re-keys saved builds silently.

    An edit to any module in :func:`build_source.build_sources` (traceml.py, views.py, ...)
    changes :func:`build_source.build_source_identity`, so every saved variance build stops
    replaying and the next front door plans a brand-new build (tens of thousands of decision and
    review calls). This happened once by accident (a turn-numbering edit to traceml.py re-keyed
    f0f0951a to 1fc34904). The pin makes it a reviewed decision: move new code to a module the
    build does not import, revert the edit, or bump the protocol and regenerate the pin.
    """
    import json

    pinned = json.loads(PIN.read_text())
    now = current_pin()
    if now == pinned:
        return
    old, new = pinned["sources"], now["sources"]
    changed = sorted(m for m in old.keys() & new.keys() if old[m] != new[m])
    added = sorted(new.keys() - old.keys())
    removed = sorted(old.keys() - new.keys())
    detail = f"changed {changed}, added {added}, removed {removed}"
    if now["protocol"] == pinned["protocol"]:
        pytest.fail(
            f"the variance build's semantic sources changed without a protocol bump ({detail}); "
            f"identity {pinned['identity'][:8]} -> {now['identity'][:8]} would re-key every saved "
            "build. Revert the edit or move it to a module outside build_sources(); if the build's "
            "semantics really changed, bump VARIANCE_BUILD_PROTOCOL and regenerate "
            f"{PIN.name} with write_pin()."
        )
    pytest.fail(
        f"VARIANCE_BUILD_PROTOCOL is now {now['protocol']!r} (pinned {pinned['protocol']!r}; "
        f"{detail}); regenerate {PIN.name} with write_pin() to record the deliberate re-key."
    )

"""Diagnostic admission refuses protected, unknown, and incomplete source ancestry."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.project import ArtifactStore, artifact_input
from exp.optimize.evaluation.analysis.source_access import (
    declare_source_access,
    exclude_source_from_qualification,
    refuse_known_ineligible_sources,
    require_development_sources,
)
from exp.optimize.evaluation.fidelity.runner_test import source_project


def _source(
    store: ArtifactStore, name: str, inputs: tuple[ArtifactInput, ...] = ()
) -> ArtifactInput:
    """Persist one immutable source for the dependency admission checks."""
    return artifact_input(
        store.write_json(
            artifact_id=name,
            artifact_type="fixture-source",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision="fixture",
                inputs=inputs,
            ),
            files={"source.json": {"value": "private fixture"}},
        )
    )


@pytest.mark.parametrize("scope", ["protected", "unknown", "development"])
def test_admitted_root_cannot_hide_ancestor_classification(tmp_path: Path, scope: str) -> None:
    """Every ancestor needs exact development admission, with protection taking precedence."""
    project, _ = source_project(tmp_path)
    store = project.artifacts
    ancestor = _source(store, "ancestor")
    root = _source(store, "root-source", (ancestor,))
    root_grant = declare_source_access(
        store, root, scope="development", rationale="Reviewed root", code_revision="fixture"
    )
    grants = {root_grant}
    if scope != "unknown":
        grants.add(
            declare_source_access(
                store,
                ancestor,
                scope="protected" if scope == "protected" else "development",
                rationale="Reviewed ancestor",
                code_revision="fixture",
            )
        )
    if scope == "development":
        assert set(require_development_sources(store, (root,), maximum_artifacts=2)) == grants
        with pytest.raises(ValueError, match="complete dependency bound"):
            require_development_sources(store, (root,), maximum_artifacts=1)
        declare_source_access(
            store,
            ancestor,
            scope="protected",
            rationale="Reserved for confirmation",
            code_revision="fixture",
        )
        with pytest.raises(ValueError, match="protected"):
            require_development_sources(store, (root,), maximum_artifacts=2)
        with pytest.raises(ValueError, match="cannot be admitted"):
            declare_source_access(
                store,
                ancestor,
                scope="development",
                rationale="Cannot override protection",
                code_revision="fixture",
            )
    else:
        with pytest.raises(
            ValueError, match="protected" if scope == "protected" else "unclassified"
        ):
            require_development_sources(store, (root,), maximum_artifacts=2)


@pytest.mark.parametrize("excluded", ["ancestor", "root"])
def test_qualification_exclusion_preserves_diagnostic_access(tmp_path: Path, excluded: str) -> None:
    """Development declarations permit triage but cannot launder an excluded dependency."""
    project, _ = source_project(tmp_path)
    store = project.artifacts
    ancestor = _source(store, "exclusion-ancestor")
    root = _source(store, "exclusion-root", (ancestor,))
    audit = _source(store, "exclusion-audit")
    grants = {
        declare_source_access(
            store,
            reference,
            scope="development",
            rationale="Diagnostic triage",
            code_revision="test",
        )
        for reference in (root, ancestor)
    }
    source = ancestor if excluded == "ancestor" else root
    original = store.read_bytes(source.artifact_id, "source.json")
    exclusion = exclude_source_from_qualification(
        store, source, audit=audit, rationale="Lost keyed output was replaced", code_revision="test"
    )
    assert store.read(exclusion.artifact_id).manifest.inputs == tuple(
        sorted((source, audit), key=lambda ref: ref.artifact_id)
    )
    assert set(require_development_sources(store, (root,), maximum_artifacts=2)) == grants
    assert store.read_bytes(source.artifact_id, "source.json") == original
    with pytest.raises(ValueError, match="excluded from qualification"):
        refuse_known_ineligible_sources(store, (root,), maximum_artifacts=2)
    declare_source_access(
        store,
        source,
        scope="development",
        rationale="Still available for triage",
        code_revision="test",
    )
    with pytest.raises(ValueError, match="excluded from qualification"):
        refuse_known_ineligible_sources(store, (root,), maximum_artifacts=2)


def test_exclusion_is_immutable_and_unknown_sources_are_not_admitted(tmp_path: Path) -> None:
    """Exclusion replay keeps its audit, and a blacklist check never grants development access."""
    project, _ = source_project(tmp_path)
    store = project.artifacts
    source = _source(store, "excluded-source")
    audit = _source(store, "initial-audit")
    other_audit = _source(store, "later-audit")
    safe = _source(store, "unclassified-source", (audit,))
    refuse_known_ineligible_sources(store, (safe,), maximum_artifacts=2)
    with pytest.raises(ValueError, match="unclassified"):
        require_development_sources(store, (safe,), maximum_artifacts=2)
    with pytest.raises(ValueError, match="dependency bound"):
        refuse_known_ineligible_sources(store, (safe,), maximum_artifacts=1)
    marker = exclude_source_from_qualification(
        store, source, audit=audit, rationale="Audited duplicate draw", code_revision="test"
    )
    original = store.read_bytes(marker.artifact_id, "exclusion.json")
    assert (
        exclude_source_from_qualification(
            store, source, audit=audit, rationale="Audited duplicate draw", code_revision="test"
        )
        == marker
    )
    with pytest.raises(ValueError, match="different immutable audit"):
        exclude_source_from_qualification(
            store, source, audit=other_audit, rationale="Another audit", code_revision="test"
        )
    assert store.read_bytes(marker.artifact_id, "exclusion.json") == original
    with pytest.raises(ValueError, match="reference identity differs"):
        exclude_source_from_qualification(
            store,
            safe.model_copy(update={"sha256": "0" * 64}),
            audit=audit,
            rationale="Must verify exact source",
            code_revision="test",
        )

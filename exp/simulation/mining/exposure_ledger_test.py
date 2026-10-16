"""The root exposure ledger: content nodes, grouping edges, exposures, and component queries."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.simulation.mining.exposure_ledger import (
    FINGERPRINT_SCHEME,
    ExposureLedger,
    LedgerBuild,
    LedgerEdge,
    LedgerExposure,
    LedgerNode,
)

A, B, C, D = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
AT = datetime(2026, 10, 16, tzinfo=UTC)


def node(fingerprint: str, project: str, build: str) -> LedgerNode:
    """One observed trace content under the current scheme."""
    return LedgerNode(
        fingerprint=fingerprint,
        scheme=FINGERPRINT_SCHEME,
        trace_id=f"trace-{fingerprint[0]}",
        task_lineage=None,
        project=project,
        build=build,
    )


def declared(left: str, right: str, project: str, build: str) -> LedgerEdge:
    """A grouping edge: both traces declare the same task identity."""
    return LedgerEdge(
        left=min(left, right),
        right=max(left, right),
        reason="declared-task-identity",
        task_lineage="user_01",
        project=project,
        build=build,
    )


def near(left: str, right: str, project: str, build: str) -> LedgerEdge:
    """A grouping edge: the build's own near-duplicate union."""
    return LedgerEdge(
        left=min(left, right),
        right=max(left, right),
        reason="near-duplicate",
        duplicate_kind="semantic",
        similarity=0.97,
        project=project,
        build=build,
    )


def expose(ledger: ExposureLedger, *fingerprints: str) -> LedgerExposure:
    """Mark content exposed through a per-trace report."""
    exposure = LedgerExposure(
        fingerprints=tuple(sorted(fingerprints)),
        how="inspected",
        panel="panel-1",
        receipt="held-out-audit-receipt-1",
        report="exp eval --held-out-fidelity --verbose",
        project="one",
        at=AT,
    )
    ledger.record_exposure(exposure)
    return exposure


def test_exposure_flows_transitively_and_backward_through_recorded_grouping(
    tmp_path: Path,
) -> None:
    """Exposed A grouped with B, then C grouped only with B: C is exposed, with its path."""
    ledger = ExposureLedger(tmp_path)
    ledger.record_grouping(
        LedgerBuild(project="one", build="task-set-1", scheme=FINGERPRINT_SCHEME),
        nodes=(node(A, "one", "task-set-1"), node(B, "one", "task-set-1")),
        edges=(declared(A, B, "one", "task-set-1"),),
    )
    # A later import drops A entirely; C groups only with B, and is recorded before A is exposed.
    ledger.record_grouping(
        LedgerBuild(project="two", build="task-set-2", scheme=FINGERPRINT_SCHEME),
        nodes=(node(B, "two", "task-set-2"), node(C, "two", "task-set-2")),
        edges=(near(B, C, "two", "task-set-2"),),
    )
    before = ledger.snapshot()
    assert before.exposure((C,)) is None
    exposure = expose(ledger, A)

    snapshot = ledger.snapshot()
    assert snapshot.identity != before.identity
    found = snapshot.exposure((C,))
    assert found is not None
    assert found.exposed_node == A and found.component_size == 3
    assert [(edge.reason, edge.build) for edge in found.path] == [
        ("near-duplicate", "task-set-2"),
        ("declared-task-identity", "task-set-1"),
    ]
    assert found.exposures == (exposure.record_id,)
    # The same content seen anywhere is itself exposed, with no path.
    direct = snapshot.exposure((A,))
    assert direct is not None and direct.path == () and direct.component_size == 3
    assert snapshot.exposure((D,)) is None
    assert snapshot.has_build("two", "task-set-2") and not snapshot.has_build("two", "task-set-1")


def test_ledger_is_append_only_exportable_and_mergeable(tmp_path: Path) -> None:
    """Records are content-addressed and never removed; a merged export reproduces the snapshot."""
    ledger = ExposureLedger(tmp_path / "here")
    grouping = (
        LedgerBuild(project="one", build="task-set-1", scheme=FINGERPRINT_SCHEME),
        (node(A, "one", "task-set-1"), node(B, "one", "task-set-1")),
        (declared(A, B, "one", "task-set-1"),),
    )
    ledger.record_grouping(grouping[0], nodes=grouping[1], edges=grouping[2])
    first = ledger.snapshot().identity
    ledger.record_grouping(grouping[0], nodes=grouping[1], edges=grouping[2])
    assert ledger.snapshot().identity == first, "recording the same build again adds nothing"
    expose(ledger, A)
    exported = ledger.export()
    assert exported.identity == ledger.snapshot().identity
    assert len(exported.edges) == 1 and len(exported.exposures) == 1

    elsewhere = ExposureLedger(tmp_path / "elsewhere")
    assert elsewhere.snapshot().exposure((B,)) is None, "an unshared ledger sees no exposure"
    elsewhere.merge(exported)
    assert elsewhere.snapshot().identity == exported.identity
    assert elsewhere.snapshot().exposure((B,)) is not None


def test_only_the_two_grouping_reasons_exist() -> None:
    """Declared task identity and near-duplicate unions are the only edges, each with evidence."""
    with pytest.raises(ValidationError):
        LedgerEdge(left=A, right=B, reason="near-duplicate", project="one", build="b")
    with pytest.raises(ValidationError):
        LedgerEdge(left=A, right=B, reason="declared-task-identity", project="one", build="b")
    with pytest.raises(ValidationError):
        LedgerEdge.model_validate(
            {"left": A, "right": B, "reason": "same-customer", "project": "one", "build": "b"}
        )
    with pytest.raises(ValidationError, match="sorted"):
        LedgerEdge.model_validate(
            {**declared(A, B, "one", "b").model_dump(), "left": B, "right": A}
        )

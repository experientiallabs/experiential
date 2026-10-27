"""The root exposure ledger: propagation, scheme continuity, reservations and leases."""

import multiprocessing
from datetime import UTC, datetime
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Barrier, Event
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, JsonValue, sha256_json
from exp.common.traces import Trace
from exp.simulation.mining.exposure_ledger import (
    DECLARED_PREFIX,
    ExposureLedger,
    LedgerEdge,
    LedgerExposure,
    LedgerRelease,
    LedgerReservation,
    PanelInUseError,
    SchemeConflictError,
)
from exp.simulation.mining.fingerprint import (
    TRACE_CONTENT,
    TRACE_CONTENT_SPECIFICATION,
    GoldenVector,
    Scheme,
    paired_exchanges,
    trace_content_fingerprint,
    vector_trace,
)
from exp.simulation.mining.lineage import assign_source_lineages

AT = datetime(2026, 10, 16, tzinfo=UTC)
DATASET = ArtifactInput(artifact_id="trace-dataset-fixture", sha256="a" * 64)


def trace(index: int, *, lineage: str | None = None) -> Trace:
    """A distinct fingerprintable trace, optionally declaring a task identity."""
    context: dict[str, JsonValue] = {} if lineage is None else {"task_lineage": lineage}
    return vector_trace(
        f"Look up order #W{index}.",
        ((f"call-{index}", "get_order", {"order_id": f"#W{index}"}, f'{{"total": {index}}}'),),
        trace_id=f"trace-{index}",
        context=context,
    )


def declaring(item: Trace, lineage: str) -> Trace:
    """The same content declaring another task identity (not part of the fingerprint)."""
    return item.model_copy(update={"initial_context": {"task_lineage": lineage}})


def customer(item: Trace, customer_id: str, *, source_id: str | None = None) -> Trace:
    """The same trace captured for one customer, optionally from another source."""
    spans = tuple(
        span.model_copy(update={"attributes": {**span.attributes, "exp.customer.id": customer_id}})
        for span in item.spans
    )
    source = item.source
    if source_id is not None:
        identity = source.identity.model_copy(update={"source_id": source_id})
        source = source.model_copy(update={"identity": identity})
    return item.model_copy(update={"spans": spans, "source": source})


def ordered_fingerprint(item: Trace) -> str | None:
    """An older, order-sensitive scheme used to test continuity."""
    exchanges = paired_exchanges(item)
    return None if exchanges is None else sha256_json({"task": item.task, "calls": exchanges})


OLD = Scheme(
    specification={"name": "trace-order", "version": 1, "hash": "task and calls in order"},
    compute=ordered_fingerprint,
    vectors=(GoldenVector(trace=trace(0), fingerprint=ordered_fingerprint(trace(0))),),
)
TASK_ONLY = Scheme(
    specification={"name": "trace-task", "version": 1, "hash": "task text only"},
    compute=lambda item: sha256_json({"task": item.task}),
    vectors=(GoldenVector(trace=trace(0), fingerprint=sha256_json({"task": trace(0).task})),),
)
EDITED = Scheme(
    specification={**TRACE_CONTENT_SPECIFICATION, "unicode": "NFC before hashing"},
    compute=trace_content_fingerprint,
    vectors=TRACE_CONTENT.vectors,
)


def build(ledger: ExposureLedger, project: str, *traces: Trace) -> None:
    """Record one build of ``traces`` with its partition's declarations, no near-duplicates."""
    ledger.record_build(
        project=project,
        build=f"task-set-{project}",
        dataset=DATASET,
        traces=traces,
        lineages=assign_source_lineages(traces),
        near_duplicates=(),
    )


def vertex(ledger: ExposureLedger, item: Trace) -> str:
    """The trace's vertex under the ledger's newest computable scheme."""
    scheme = ledger.computable[-1].identity
    return f"{scheme.key}:{ledger.vertices(item)[scheme]}"


def expose(ledger: ExposureLedger, *vertices: str, at: datetime = AT) -> LedgerExposure:
    """Mark content exposed through a per-trace report."""
    exposure = LedgerExposure(
        vertices=tuple(sorted(vertices)),
        how="inspected",
        panel="panel-1",
        receipt="held-out-audit-receipt-1",
        report="exp eval --audit --verbose",
        project="one",
        at=at,
    )
    ledger.record_exposure(exposure)
    return exposure


def test_exposure_flows_transitively_and_backward_through_recorded_grouping(
    tmp_path: Path,
) -> None:
    """A3: exposed A grouped with B, then C grouped only with B: C is exposed, with its path."""
    ledger = ExposureLedger(tmp_path)
    a, b, c, d = trace(1, lineage="x"), trace(2, lineage="x"), trace(3), trace(4)
    build(ledger, "one", a, b)
    # A later import drops A; C groups with B only, recorded before A is exposed.
    build(ledger, "two", declaring(b, "y"), declaring(c, "y"))
    before = ledger.snapshot()
    assert before.exposure((vertex(ledger, c),)) is None
    exposure = expose(ledger, vertex(ledger, a))

    snapshot = ledger.snapshot()
    assert snapshot.identity != before.identity
    found = snapshot.exposure((vertex(ledger, c),))
    assert found is not None
    assert found.exposed_node == vertex(ledger, a) and found.component_size == 3
    assert [(edge.task_lineage, edge.build) for edge in found.path] == [
        ("y", "task-set-two"),
        ("x", "task-set-one"),
    ]
    assert found.by == (exposure.record_id,)
    direct = snapshot.exposure((vertex(ledger, a),))
    assert direct is not None and direct.path == ()
    assert snapshot.exposure((vertex(ledger, d),)) is None
    # Late-link reassessment counts only the exposure records its caller's predicate admits.
    assert snapshot.exposure((vertex(ledger, c),), counts=lambda item: False) is None
    counted = snapshot.exposure((vertex(ledger, c),), counts=lambda item: item == exposure)
    assert counted is not None and counted.by == (exposure.record_id,)


def test_declared_task_edges_stay_within_one_source_and_customer(tmp_path: Path) -> None:
    """A declared name links one customer's traces; another customer or source reusing it is not."""
    ledger = ExposureLedger(tmp_path)
    a = customer(trace(1, lineage="ticket-1"), "c1")
    b = customer(trace(2, lineage="ticket-1"), "c1")
    other_customer = customer(trace(3, lineage="ticket-1"), "c2")
    other_source = customer(trace(4, lineage="ticket-1"), "c1", source_id="other-export")
    build(ledger, "one", a, b, other_customer, other_source)

    edges = ledger.export().edges
    identities = {edge.left for edge in edges if edge.left.startswith(DECLARED_PREFIX)}
    assert len(identities) == 3, "each source and customer scope has its own identity vertex"
    assert [
        (edge.left, edge.right, edge.task_lineage) for edge in edges if edge.left not in identities
    ] == [(*sorted((vertex(ledger, a), vertex(ledger, b))), "ticket-1")]
    expose(ledger, vertex(ledger, a))
    snapshot = ledger.snapshot()
    found = snapshot.exposure((vertex(ledger, b),))
    assert found is not None and found.exposed_node == vertex(ledger, a)
    assert snapshot.exposure((vertex(ledger, other_customer),)) is None
    assert snapshot.exposure((vertex(ledger, other_source),)) is None
    # The scope comes from the build's own partition: a trace without its assignment is refused.
    with pytest.raises(ValueError, match="trace-1 has no source lineage assignment"):
        ledger.record_build(
            project="two",
            build="task-set-two",
            dataset=DATASET,
            traces=(a,),
            lineages=(),
            near_duplicates=(),
        )
    assert ledger.snapshot().identity == snapshot.identity


def test_a_declared_task_identity_links_traces_across_builds(tmp_path: Path) -> None:
    """A task seen in one build stays exposed when a later import holds only new content of it."""
    ledger = ExposureLedger(tmp_path)
    seen = customer(trace(1, lineage="user_17"), "c1")
    fresh = customer(trace(2, lineage="user_17"), "c1")
    other_customer = customer(trace(3, lineage="user_17"), "c2")
    other_source = customer(trace(4, lineage="user_17"), "c1", source_id="other-export")
    build(ledger, "one", seen)
    build(ledger, "two", fresh, other_customer, other_source)
    exposure = expose(ledger, vertex(ledger, seen))
    snapshot = ledger.snapshot()
    found = snapshot.exposure((vertex(ledger, fresh),))
    assert found is not None and found.by == (exposure.record_id,)
    assert found.component_size == 2, "the task identity itself is not counted as content"
    assert [(edge.task_lineage, edge.build) for edge in found.path] == [
        ("user_17", "task-set-two"),
        ("user_17", "task-set-one"),
    ]
    assert snapshot.exposure((vertex(ledger, other_customer),)) is None
    assert snapshot.exposure((vertex(ledger, other_source),)) is None


def test_matching_runs_under_every_computable_scheme_with_equivalence_links(
    tmp_path: Path,
) -> None:
    """A4: an old scheme still matches; every source links its fingerprints across schemes."""
    old = ExposureLedger(tmp_path, schemes=(OLD,))
    a, b = trace(1, lineage="x"), trace(2, lineage="x")
    build(old, "one", a, b)
    expose(old, vertex(old, a))

    both = ExposureLedger(tmp_path, schemes=(OLD, TRACE_CONTENT))
    build(both, "two", trace(9))
    links = both.snapshot().records.equivalences
    assert [len(link.vertices) for link in links] == [2], "unexposed traces are linked too"
    # A reimport of B under a new trace ID matches only through the old scheme's edge.
    reimported = b.model_copy(update={"trace_id": "reimported", "initial_context": {}})
    candidates = [
        f"{scheme.key}:{fingerprint}" for scheme, fingerprint in both.vertices(reimported).items()
    ]
    found = both.snapshot().exposure(candidates)
    assert found is not None and found.exposed_node == vertex(old, a)

    # A machine that implements only the new scheme cannot compute the old one.
    new = ExposureLedger(tmp_path, schemes=(TRACE_CONTENT,))
    [unknown] = new.snapshot().unknown_schemes([TRACE_CONTENT.identity])
    assert (unknown.scheme, unknown.nodes) == (OLD.identity, 2)
    assert new.resolve_sources(lambda source: None) == 0, "unavailable sources resolve nothing"
    sources = {item.trace_id: item for item in (a, b)}
    tampered = {**sources, "trace-2": b.model_copy(update={"task": "edited"})}
    assert new.resolve_sources(lambda source: tampered.get(source.trace_id)) == 1
    assert new.resolve_sources(lambda source: sources.get(source.trace_id)) == 1
    assert new.snapshot().unknown_schemes([TRACE_CONTENT.identity]) == ()
    resolved = new.snapshot().exposure((vertex(new, b),))
    assert resolved is not None, "full equivalence lets the new scheme see the old exposure"


def test_a_link_to_an_unfingerprintable_vertex_leaves_the_scheme_unknown(
    tmp_path: Path,
) -> None:
    """An unresolved vertex names a source, not content, so linking to it never resolves."""
    ambiguous = vector_trace(
        "Look up order #W5.",
        ((None, "get_order", {"order_id": "#W5"}, "{}"),),
        trace_id="ambiguous",
    )
    build(ExposureLedger(tmp_path, schemes=(TASK_ONLY,)), "one", ambiguous)
    new = ExposureLedger(tmp_path, schemes=(TRACE_CONTENT,))
    assert new.resolve_sources(lambda source: ambiguous) == 0
    [node] = [
        node for node in new.snapshot().records.nodes if node.scheme == TRACE_CONTENT.identity
    ]
    assert node.fingerprint is None, "the current scheme cannot pair its calls"
    assert len(new.snapshot().records.equivalences) == 1, "the link still groups the two"
    [unknown] = new.snapshot().unknown_schemes([TRACE_CONTENT.identity])
    assert (unknown.scheme, unknown.nodes) == (TASK_ONLY.identity, 1)


def test_unknown_schemes_linked_only_to_each_other_still_resolve_from_sources(
    tmp_path: Path,
) -> None:
    """An equivalence between two unknown schemes never blocks recovery from the source."""
    item = trace(1)
    build(ExposureLedger(tmp_path, schemes=(OLD, TASK_ONLY)), "one", item)
    new = ExposureLedger(tmp_path, schemes=(TRACE_CONTENT,))
    unknown = new.snapshot().unknown_schemes([TRACE_CONTENT.identity])
    assert {row.scheme for row in unknown} == {OLD.identity, TASK_ONLY.identity}
    assert new.resolve_sources(lambda source: item) == 2
    assert new.snapshot().unknown_schemes([TRACE_CONTENT.identity]) == ()


def test_resolution_follows_an_equivalence_chain_from_one_recoverable_endpoint(
    tmp_path: Path,
) -> None:
    """Two sources share a task-only vertex; recovering one source resolves the whole chain."""
    recoverable, lost = trace(1), trace(2).model_copy(update={"task": trace(1).task})
    build(ExposureLedger(tmp_path, schemes=(OLD, TASK_ONLY)), "one", recoverable, lost)
    new = ExposureLedger(tmp_path, schemes=(TRACE_CONTENT,))
    snapshot = new.snapshot()
    assert len(snapshot.records.equivalences) == 2, "one per source, joined at the task vertex"
    unknown = {row.scheme: row.nodes for row in snapshot.unknown_schemes([TRACE_CONTENT.identity])}
    assert unknown == {OLD.identity: 2, TASK_ONLY.identity: 2}
    assert new.resolve_sources(lambda source: recoverable if source.trace_id == "trace-1" else None)
    resolved = new.snapshot()
    assert resolved.unknown_schemes([TRACE_CONTENT.identity]) == ()
    [far] = [
        node
        for node in resolved.records.nodes
        if node.scheme == OLD.identity and node.source.trace_id == "trace-2"
    ]
    assert far.vertex in resolved.resolved_vertices([TRACE_CONTENT.identity])


def test_scheme_identity_conflicts_are_refused(tmp_path: Path) -> None:
    """A4: a spec edit without a version bump is refused; so is a conflicting import."""
    ledger = ExposureLedger(tmp_path / "here")
    build(ledger, "one", trace(1))
    with pytest.raises(SchemeConflictError, match="needs a new scheme version"):
        build(ExposureLedger(tmp_path / "here", schemes=(EDITED,)), "two", trace(2))

    elsewhere = ExposureLedger(tmp_path / "elsewhere", schemes=(EDITED,))
    build(elsewhere, "far", trace(3))
    assert ledger.merge(elsewhere.export()) == (EDITED.identity,)
    assert EDITED.identity not in ledger.snapshot().records.schemes
    [quarantined] = ledger.snapshot().unknown_schemes([TRACE_CONTENT.identity])
    assert (quarantined.scheme, quarantined.nodes) == (EDITED.identity, 1)

    broken = Scheme(
        specification=TRACE_CONTENT_SPECIFICATION,
        compute=trace_content_fingerprint,
        vectors=(GoldenVector(trace=trace(0), fingerprint="0" * 64),),
    )
    assert ExposureLedger(tmp_path / "here", schemes=(broken,)).computable == ()
    with pytest.raises(ValueError, match="reproduces its golden vectors"):
        build(ExposureLedger(tmp_path / "here", schemes=(broken,)), "three", trace(4))


def test_unfingerprintable_traces_are_unresolved_nodes_still_linked_by_grouping(
    tmp_path: Path,
) -> None:
    """A trace with ambiguous call IDs keeps an unresolved node that grouping edges reach."""
    ledger = ExposureLedger(tmp_path)
    ambiguous = vector_trace(
        "Look up order #W5.",
        ((None, "get_order", {"order_id": "#W5"}, "{}"),),
        trace_id="ambiguous",
        context={"task_lineage": "x"},
    )
    build(ledger, "one", trace(1, lineage="x"), ambiguous)
    [unresolved] = [node for node in ledger.snapshot().records.nodes if node.fingerprint is None]
    assert unresolved.vertex.startswith("unresolved:")
    assert unresolved.source.trace_id == "ambiguous"
    expose(ledger, unresolved.vertex)
    assert ledger.snapshot().exposure((vertex(ledger, trace(1)),)) is not None


def test_ledger_rows_are_append_only_and_exports_merge(tmp_path: Path) -> None:
    """Rows are never edited or deleted; a merged export reproduces the snapshot, not provenance."""
    ledger = ExposureLedger(tmp_path / "here")
    build(ledger, "one", trace(1, lineage="x"), trace(2, lineage="x"))
    first = ledger.snapshot().identity
    build(ledger, "one", trace(1, lineage="x"), trace(2, lineage="x"))
    assert ledger.snapshot().identity == first, "recording the same build again adds nothing"
    exposure = expose(ledger, vertex(ledger, trace(1)))
    exported = ledger.export()
    assert exported.recorded_here == (exposure.record_id,)
    elsewhere = ExposureLedger(tmp_path / "elsewhere")
    assert elsewhere.snapshot().exposure((vertex(ledger, trace(2)),)) is None
    assert elsewhere.merge(exported) == ()
    assert elsewhere.snapshot().identity == exported.identity
    assert elsewhere.export().recorded_here == (), "a merge never claims it recorded a row"
    elsewhere.record_exposure(exposure)
    assert elsewhere.export().recorded_here == (), "replaying a merged row keeps it foreign"
    with pytest.raises(Exception, match="append-only"):
        with ledger.transaction() as transaction:
            transaction._connection.execute("DELETE FROM records")  # noqa: SLF001
    with pytest.raises(ValidationError):
        LedgerEdge(left="a", right="b", reason="near-duplicate", project="one", build="b")
    with pytest.raises(ValidationError, match="sorted"):
        LedgerEdge(
            left="b",
            right="a",
            reason="declared-task-identity",
            task_lineage="x",
            project="one",
            build="b",
        )


def reservation(receipt: str, *vertices: str, snapshot: str = "ledger-x") -> LedgerReservation:
    """A reservation of ``vertices`` by one receipt."""
    return LedgerReservation(
        receipt=receipt,
        panel=f"panel-{receipt}",
        project="one",
        lineages=("lineage-a",),
        vertices=tuple(sorted(vertices)),
        snapshot=snapshot,
        at=AT,
    )


def claim(root: Path, receipt: str, *vertices: str) -> bool:
    """Reserve ``vertices`` for ``receipt`` unless another receipt holds their component."""
    with ExposureLedger(root).transaction() as transaction:
        snapshot = transaction.snapshot()
        if snapshot.taken(vertices, receipt) is not None:
            return False
        transaction.reserve(reservation(receipt, *vertices, snapshot=snapshot.identity))
        return True


def test_reservations_take_components_until_their_owner_releases(tmp_path: Path) -> None:
    """A reserved component is taken for every other receipt, never for its owner."""
    ledger = ExposureLedger(tmp_path)
    a, b = trace(1, lineage="x"), trace(2, lineage="x")
    build(ledger, "one", a, b)
    assert claim(tmp_path, "receipt-1", vertex(ledger, a))
    snapshot = ledger.snapshot()
    assert snapshot.active("receipt-1")
    assert snapshot.taken((vertex(ledger, a),), "receipt-1") is None
    taken = snapshot.taken((vertex(ledger, b),), "receipt-2")
    assert taken is not None and taken.by == ("receipt-1",) and len(taken.path) == 1
    assert snapshot.exposure((vertex(ledger, b),)) is None, "a reservation is not an exposure"
    assert not claim(tmp_path, "receipt-2", vertex(ledger, b))
    with ledger.transaction() as transaction:
        transaction.release(LedgerRelease(receipt="receipt-1", at=AT))
    assert not ledger.snapshot().active("receipt-1")
    assert claim(tmp_path, "receipt-2", vertex(ledger, b))


def _contend(
    root: str, receipt: str, item: str, start: Barrier, outcomes: "DictProxy[str, bool]"
) -> None:
    """Child process: wait for every contender, then try to reserve."""
    start.wait()
    outcomes[receipt] = claim(Path(root), receipt, item)


def test_concurrent_claims_from_separate_processes_admit_exactly_one(tmp_path: Path) -> None:
    """A7: racing reservations of one component from separate processes leave one owner."""
    ledger = ExposureLedger(tmp_path)
    a, b, c = trace(1, lineage="x"), trace(2, lineage="x"), trace(3, lineage="x")
    build(ledger, "one", a, b, c)
    targets = [vertex(ledger, item) for item in (a, b, c)]
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        start = manager.Barrier(4)
        outcomes = manager.dict()
        processes = [
            context.Process(
                target=_contend,
                args=(str(tmp_path), f"receipt-{index}", targets[index % 3], start, outcomes),
            )
            for index in range(4)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(120)
        assert all(process.exitcode == 0 for process in processes)
        assert sum(outcomes.values()) == 1
    assert len(ledger.snapshot().reservations) == 1


def _hold(root: str, panel: str, holding: Event, done: Event) -> None:
    """Child process: hold a panel lease until told to stop."""
    with ExposureLedger(Path(root)).lease(panel):
        holding.set()
        done.wait(120)


def test_panel_lease_is_exclusive_across_processes_and_dies_with_its_holder(
    tmp_path: Path,
) -> None:
    """A8: a second process fails at once with "panel in use"; a killed holder frees it."""
    context = multiprocessing.get_context("spawn")
    holding, done = context.Event(), context.Event()
    holder = context.Process(target=_hold, args=(str(tmp_path), "panel-1", holding, done))
    holder.start()
    try:
        assert holding.wait(120)
        with pytest.raises(PanelInUseError, match="in use"):
            with ExposureLedger(tmp_path).lease("panel-1"):
                pass
        with ExposureLedger(tmp_path).lease("panel-2"):
            pass
    finally:
        holder.kill()
        holder.join(120)
    with ExposureLedger(tmp_path).lease("panel-1"):
        pass

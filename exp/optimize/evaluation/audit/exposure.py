"""Which held-out lineages of a build are still independent, under the root exposure ledger.

A lineage here is the build's own leakage group. Its members are every source trace the build
bound to it, and each member is identified by its content under every computable fingerprint
scheme, so exposure recorded in another project, another import, under another grouping or under
an older scheme still applies. A held-out lineage is not independent when any member:

- shares a connected component of the ledger with exposed content (``exposed``);
- shares one with content another audit receipt reserved and did not release (``reserved``);
- cannot be fingerprinted under the current scheme, so its exposure history is unresolved
  (``unresolved``); the whole group is ineligible even when the selected trace is fingerprintable.

Nodes under a scheme this machine cannot compute, with no verified equivalence link, block every
independent audit, since they could match any content; development re-audits may proceed. See
:mod:`exp.simulation.mining.exposure_ledger`.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.project import (
    ArtifactStoreError,
    ProjectBuildArtifacts,
    ProjectStore,
    artifact_input,
)
from exp.common.traces import Trace
from exp.common.traces.store import load_trace_dataset
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.mining.exposure_ledger import (
    SAME_HOST_LIMIT,
    ExposureLedger,
    ExposurePath,
    LedgerExposure,
    LedgerSnapshot,
    SourceRecord,
    UnknownScheme,
    node_vertex,
    record_sha256,
)
from exp.simulation.mining.fingerprint import SchemeIdentity

LIMITS = (
    "A ledger on another machine that was never merged here cannot show its exposure.",
    "Paraphrased or regenerated variants that near-duplicate grouping misses are not caught.",
    "Knowledge a person carries over by hand is not tracked.",
    SAME_HOST_LIMIT,
    "Exposure matching is conservative: the fingerprint ignores call order, so two different "
    "execution sequences of the same exchanges match.",
)


class LineageExposure(ContractModel):
    """Why one held-out lineage is not independent.

    Attributes:
        lineage: The build's leakage lineage.
        kind: ``exposed`` (content was shown), ``reserved`` (another receipt holds it) or
            ``unresolved`` (a member cannot be fingerprinted).
        path: Nearest marked content, the component size, and the edges leading there; None
            for an unresolved lineage.
        unresolved: Member traces the current scheme cannot fingerprint.
    """

    lineage: str
    kind: Literal["exposed", "reserved", "unresolved"] = "exposed"
    path: ExposurePath | None = None
    unresolved: tuple[str, ...] = ()


class LineageMembers(ContractModel):
    """One held-out lineage's member content.

    Attributes:
        vertices: Member vertices under every computable scheme.
        unresolved: Members the current scheme cannot fingerprint.
    """

    vertices: tuple[str, ...]
    unresolved: tuple[str, ...] = ()


class HeldOutExposure(ContractModel):
    """The exposure decision for one build's held-out lineages, and the ledger it used.

    Attributes:
        schemes: Every computable scheme the match used, with its full identity.
        ledger: Snapshot identity of the ledger the decision used.
        exposed: Exposed, reserved or unresolved held-out lineages, in lineage order.
        unknown: Nodes under schemes not computable here, without verified equivalence; any
            blocks every independent audit.
    """

    schemes: tuple[SchemeIdentity, ...] = Field(min_length=1)
    ledger: str
    exposed: tuple[LineageExposure, ...] = ()
    unknown: tuple[UnknownScheme, ...] = ()

    @property
    def lineages(self) -> frozenset[str]:
        """Lineage IDs that are not independent."""
        return frozenset(item.lineage for item in self.exposed)


def held_out_members(
    project: ProjectStore, ledger: ExposureLedger | None = None
) -> dict[str, LineageMembers]:
    """Every held-out lineage's member content under every computable scheme.

    Args:
        project: Project with a completed build.
        ledger: Ledger whose schemes to use; the project's root ledger when omitted.

    Returns:
        Member content keyed by lineage, in lineage order.
    """
    ledger = ledger or ExposureLedger(project.paths.root)
    build = _build(project)
    primary = ledger.computable[-1].identity
    members: dict[str, set[str]] = defaultdict(set)
    unresolved: dict[str, set[str]] = defaultdict(set)
    for trace, lineage, source in _held_out_traces(project, build):
        for scheme, fingerprint in ledger.vertices(trace).items():
            members[lineage].add(node_vertex(scheme, fingerprint, source))
            if scheme == primary and fingerprint is None:
                unresolved[lineage].add(trace.trace_id)
    return {
        lineage: LineageMembers(
            vertices=tuple(sorted(vertices)), unresolved=tuple(sorted(unresolved[lineage]))
        )
        for lineage, vertices in sorted(members.items())
    }


def refresh_ledger(project: ProjectStore, ledger: ExposureLedger | None = None) -> None:
    """Fingerprint the build under every computable scheme and resolve sources where possible.

    A scheme added after the build only adds nodes and equivalence links. Unknown-scheme nodes
    whose verified source record is still available here are re-fingerprinted and linked.

    Args:
        project: Project with a completed build.
        ledger: Ledger to refresh; the project's root ledger when omitted.
    """
    ledger = ledger or ExposureLedger(project.paths.root)
    build = _build(project)
    traces = load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces
    ledger.record_nodes(
        project=project.paths.project_id,
        build=build.task_set.artifact_id,
        dataset=build.trace_dataset,
        traces=traces,
    )
    ledger.resolve_sources(lambda source: _load_source(project, source))


def held_out_exposure(
    project: ProjectStore,
    snapshot: LedgerSnapshot | None = None,
    owner: str | None = None,
    *,
    ledger: ExposureLedger | None = None,
    counts: Callable[[LedgerExposure], bool] | None = None,
) -> HeldOutExposure:
    """Decide which held-out lineages of the build are not independent.

    Args:
        project: Project with a completed build.
        snapshot: Ledger snapshot to decide with; the current root ledger when omitted.
        owner: Receipt whose own reservation does not count against it.
        ledger: Ledger whose schemes to use; the project's root ledger when omitted.
        counts: Which exposure records count (late-link reassessment counts only those not
            provably recorded after the receipt); every record when omitted.

    Returns:
        The decision with the snapshot identity and schemes it used.

    Raises:
        ValueError: The build never recorded its grouping in the ledger.
    """
    ledger = ledger or ExposureLedger(project.paths.root)
    build = _build(project)
    snapshot = snapshot or ledger.snapshot()
    if not snapshot.has_build(project.paths.project_id, build.task_set.artifact_id):
        raise ValueError(
            "this build never recorded its traces and grouping in the exposure ledger, so "
            "held-out independence cannot be decided; rerun exp build PROJECT on the same import "
            "to record them"
        )
    computable = tuple(scheme.identity for scheme in ledger.computable)
    exposed = []
    for lineage, members in held_out_members(project, ledger).items():
        path = snapshot.exposure(members.vertices, counts=counts)
        if path is not None:
            exposed.append(LineageExposure(lineage=lineage, kind="exposed", path=path))
            continue
        taken = snapshot.taken(members.vertices, owner)
        if taken is not None:
            exposed.append(LineageExposure(lineage=lineage, kind="reserved", path=taken))
        elif members.unresolved:
            exposed.append(
                LineageExposure(lineage=lineage, kind="unresolved", unresolved=members.unresolved)
            )
    return HeldOutExposure(
        schemes=computable,
        ledger=snapshot.identity,
        exposed=tuple(exposed),
        unknown=snapshot.unknown_schemes(computable),
    )


def unknown_scheme_refusal(unknown: tuple[UnknownScheme, ...]) -> str:
    """Why no independent audit can run while unknown-scheme nodes remain unresolved."""
    counts = "; ".join(
        f"{item.nodes} under {item.scheme.name} v{item.scheme.version} "
        f"(definition {item.scheme.definition_sha256[:12]})"
        for item in unknown
    )
    return (
        f"the exposure ledger holds nodes under fingerprint schemes this machine cannot compute, "
        f"with no verified equivalence link: {counts}. They could match any held-out content, "
        "so an independent audit is refused; --include-inspected runs a development re-audit"
    )


def _held_out_traces(
    project: ProjectStore, build: ProjectBuildArtifacts
) -> list[tuple[Trace, str, SourceRecord]]:
    """Held-out member traces with their lineage and verified source record."""
    bindings = load_task_set_lineage_bindings(project.artifacts, build.task_set.artifact_id)
    lineage_of = {
        binding.trace_id: binding.lineage_id
        for binding in bindings.bindings
        if binding.partition == "held_out"
    }
    return [
        (
            trace,
            lineage_of[trace.trace_id],
            SourceRecord(
                project=project.paths.project_id,
                dataset=build.trace_dataset,
                trace_id=trace.trace_id,
                record_sha256=record_sha256(trace),
            ),
        )
        for trace in load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces
        if trace.trace_id in lineage_of
    ]


def _load_source(project: ProjectStore, source: SourceRecord) -> Trace | None:
    """The trace a source record points to, from any project under the same root."""
    store = ProjectStore(project.paths.root, source.project)
    try:
        if artifact_input(store.artifacts.read(source.dataset.artifact_id).manifest) != (
            source.dataset
        ):
            return None
        traces = load_trace_dataset(store.artifacts, source.dataset.artifact_id).traces
    except (OSError, ValueError, ArtifactStoreError):
        return None
    return next((trace for trace in traces if trace.trace_id == source.trace_id), None)


def _build(project: ProjectStore) -> ProjectBuildArtifacts:
    """The completed build, or a clear error."""
    build = project.load_project().build
    if build is None:
        raise ValueError("held-out fidelity needs a completed build; run exp build first")
    return build

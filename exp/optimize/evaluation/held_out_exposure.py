"""Which held-out lineages of a build are still independent, under the root exposure ledger.

A lineage here is the build's own leakage group. Its members are every source trace the build
bound to it, and each member is identified by its content fingerprint, so exposure recorded in
another project, another import, or under another grouping still applies. A held-out lineage is
exposed when any member shares a connected component of the ledger with exposed content; see
:mod:`exp.simulation.mining.exposure_ledger`.
"""

from collections import defaultdict

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.project import ProjectBuildArtifacts, ProjectStore
from exp.common.traces.store import load_trace_dataset
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.mining.exposure_ledger import (
    FINGERPRINT_SCHEME,
    ExposureLedger,
    ExposurePath,
    LedgerSnapshot,
    trace_fingerprint,
)

LIMITS = (
    "A ledger on another machine that was never merged here cannot show its exposure.",
    "Paraphrased or regenerated variants that near-duplicate grouping misses are not caught.",
    "Knowledge a person carries over by hand is not tracked.",
)


class LineageExposure(ContractModel):
    """Why one held-out lineage is no longer independent.

    Attributes:
        lineage: The build's leakage lineage.
        path: Nearest exposed content, the component size, and the edges leading there.
    """

    lineage: str
    path: ExposurePath


class HeldOutExposure(ContractModel):
    """The exposure decision for one build's held-out lineages, and the ledger it used.

    Attributes:
        scheme: Fingerprint scheme and version.
        ledger: Snapshot identity of the ledger the decision used.
        exposed: Exposed held-out lineages, in lineage order.
    """

    scheme: str = FINGERPRINT_SCHEME
    ledger: str
    exposed: tuple[LineageExposure, ...] = Field(default=())

    @property
    def lineages(self) -> frozenset[str]:
        """Exposed lineage IDs."""
        return frozenset(item.lineage for item in self.exposed)


def held_out_members(project: ProjectStore) -> dict[str, tuple[str, ...]]:
    """Content fingerprints of every held-out lineage's member traces.

    Args:
        project: Project with a completed build.

    Returns:
        Sorted member fingerprints keyed by lineage.
    """
    build = _build(project)
    bindings = load_task_set_lineage_bindings(project.artifacts, build.task_set.artifact_id)
    lineage_of = {
        binding.trace_id: binding.lineage_id
        for binding in bindings.bindings
        if binding.partition == "held_out"
    }
    members: dict[str, set[str]] = defaultdict(set)
    for trace in load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces:
        if trace.trace_id in lineage_of:
            members[lineage_of[trace.trace_id]].add(trace_fingerprint(trace))
    return {lineage: tuple(sorted(found)) for lineage, found in sorted(members.items())}


def held_out_exposure(
    project: ProjectStore, snapshot: LedgerSnapshot | None = None
) -> HeldOutExposure:
    """Decide which held-out lineages of the build are exposed.

    Args:
        project: Project with a completed build.
        snapshot: Ledger snapshot to decide with; the current root ledger when omitted.

    Returns:
        The decision with the snapshot identity it used.

    Raises:
        ValueError: The build never recorded its grouping in the ledger.
    """
    build = _build(project)
    snapshot = snapshot or ExposureLedger(project.paths.root).snapshot()
    if not snapshot.has_build(project.paths.project_id, build.task_set.artifact_id):
        raise ValueError(
            "this build never recorded its traces and grouping in the exposure ledger, so "
            "held-out independence cannot be decided; rerun exp build PROJECT on the same import "
            "to record them"
        )
    exposed = []
    for lineage, fingerprints in held_out_members(project).items():
        path = snapshot.exposure(fingerprints)
        if path is not None:
            exposed.append(LineageExposure(lineage=lineage, path=path))
    return HeldOutExposure(ledger=snapshot.identity, exposed=tuple(exposed))


def _build(project: ProjectStore) -> ProjectBuildArtifacts:
    """The completed build, or a clear error."""
    build = project.load_project().build
    if build is None:
        raise ValueError("held-out fidelity needs a completed build; run exp build first")
    return build

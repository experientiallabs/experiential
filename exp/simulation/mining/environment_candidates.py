"""Complete bounded evidence candidates for independent environment review.

Contract equality is a partitioning convenience, never evidence of shared state or semantics.
Tag clusters intersect these candidates and each intersection receives its own native review.
"""

from collections import defaultdict
from collections.abc import Sequence

from exp.common.core.artifacts import ContractModel, sha256_json, stable_id
from exp.common.traces import Trace
from exp.simulation.mining.environment_review import (
    CandidateCapacityError,
    CompatibilitySpec,
    checked_group_packet,
)
from exp.simulation.rules.contracts import parse_tool_contracts


class EnvironmentCandidate(ContractModel):
    """One complete evidence shard, with no compatibility verdict.

    Attributes:
        candidate_id: Identity of members, contracts and complete source evidence.
        contract_sha256: Digest of exact declared and observed contract distinctions.
        source_sha256: Digest of complete member trace content, including provenance.
        members: All traces assigned to this shard, exactly once across the partition.
    """

    candidate_id: str
    contract_sha256: str
    source_sha256: str
    members: tuple[str, ...]


def environment_candidates(
    traces: Sequence[Trace], spec: CompatibilitySpec
) -> tuple[EnvironmentCandidate, ...]:
    """Partition complete evidence deterministically under registered candidate bounds.

    Unset bounds preserve cluster-level review. Registered bounds separate exact contract
    fingerprints, then pack sorted members against the actual native inventory serializer.
    An oversized singleton refuses instead of omitting its tools or evidence.
    """
    ordered = sorted(traces, key=lambda trace: trace.trace_id)
    if len({trace.trace_id for trace in ordered}) != len(ordered):
        raise ValueError("environment candidates require distinct source trace identities")
    if not ordered:
        return ()
    if spec.maximum_candidate_members is None:
        return (_candidate(ordered, sha256_json({"unpartitioned": True})),)
    buckets: dict[str, list[Trace]] = defaultdict(list)
    for trace in ordered:
        contracts = parse_tool_contracts((trace,))
        fingerprint = sha256_json(
            {
                "declared": sorted(
                    (tool.model_dump(mode="json") for tool in trace.tools),
                    key=lambda tool: str(tool["name"]),
                ),
                "observed": {
                    name: row.model_dump(mode="json", exclude={"calls", "failures", "examples"})
                    for name, row in contracts.items()
                },
            }
        )
        buckets[fingerprint].append(trace)
    candidates: list[EnvironmentCandidate] = []
    for fingerprint, members in sorted(buckets.items()):
        pending: list[Trace] = []
        for trace in members:
            try:
                checked_group_packet((*pending, trace), spec)
            except CandidateCapacityError:
                if not pending:
                    raise
                candidates.append(_candidate(pending, fingerprint))
                pending = []
                checked_group_packet((trace,), spec)
            pending.append(trace)
        if pending:
            candidates.append(_candidate(pending, fingerprint))
    return tuple(candidates)


def candidate_root_bound(traces: Sequence[Trace], spec: CompatibilitySpec, *, clusters: int) -> int:
    """Bound cluster intersections with the exact registered evidence partition."""
    if clusters < 1:
        raise ValueError("environment review needs a positive cluster bound")
    return min(len(traces), clusters * len(environment_candidates(traces, spec)))


def _candidate(traces: Sequence[Trace], contract: str) -> EnvironmentCandidate:
    """Bind every member and its complete provenance to one candidate identity."""
    source = sha256_json({trace.trace_id: sha256_json(trace) for trace in traces})
    return EnvironmentCandidate(
        candidate_id=stable_id("environment-candidate", {"contract": contract, "source": source}),
        contract_sha256=contract,
        source_sha256=source,
        members=tuple(trace.trace_id for trace in traces),
    )

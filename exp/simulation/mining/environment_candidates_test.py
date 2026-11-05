"""Complete bounded candidates exercise native inventory, review and cost owners."""

import json

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.tasks import ToolSchema
from exp.simulation.mining import environment_candidates as candidate_owner
from exp.simulation.mining.environment_candidates import (
    candidate_root_bound,
    environment_candidates,
)
from exp.simulation.mining.environment_families import (
    families_identity,
    review_environment_families,
)
from exp.simulation.mining.environment_families_test import (
    answer,
    clustering,
    file_trace,
    order_trace,
    reviewer,
)
from exp.simulation.mining.environment_review import (
    CandidateCapacityError,
    CompatibilitySpec,
    checked_group_packet,
    group_packet,
    review_protocol_sha256,
)


def test_partition_preserves_complete_members_contracts_and_source_identity() -> None:
    """Stable shards retain every source and separate differing exact contracts."""
    traces = [order_trace(f"o{i:02}") for i in range(7)] + [file_trace("f0")]
    spec = CompatibilitySpec(maximum_candidate_members=3, maximum_candidate_packet_bytes=48_000)
    candidates = environment_candidates(traces, spec)
    assert candidates == environment_candidates(tuple(reversed(traces)), spec)
    assert len(candidates) == 4
    members = [member for candidate in candidates for member in candidate.members]
    assert sorted(members) == sorted(trace.trace_id for trace in traces)
    assert len(set(members)) == len(members)
    by_id = {trace.trace_id: trace for trace in traces}
    for candidate in candidates:
        shown = [by_id[member] for member in candidate.members]
        packet, locators = checked_group_packet(shown, spec)
        assert packet == group_packet(shown)[0]
        assert len(json.dumps(packet, sort_keys=True).encode("utf-8")) <= 48_000
        assert {f"trace:{member}" for member in candidate.members} <= locators
        assert candidate.source_sha256 == sha256_json(
            {trace.trace_id: sha256_json(trace) for trace in shown}
        )
    changed = traces[0].model_copy(update={"task": "Changed complete source task"})
    revised = environment_candidates([changed, *traces[1:]], spec)
    assert {row.candidate_id for row in revised} != {row.candidate_id for row in candidates}
    changed_schema = traces[0].model_copy(
        update={
            "tools": (
                ToolSchema(name="lookup", description="lookup", input_schema={"type": "array"}),
            )
        }
    )
    revised = environment_candidates([changed_schema, *traces[1:]], spec)
    assert next(row for row in revised if "o00" in row.members).members == ("o00",)


def test_byte_partition_uses_actual_packet_and_refuses_irreducible_inventory() -> None:
    """Byte bounds split members, without abbreviating oversized singleton evidence."""
    traces = [order_trace(f"o{i}") for i in range(3)]
    size = len(json.dumps(group_packet(traces[:1])[0], sort_keys=True).encode("utf-8"))
    spec = CompatibilitySpec(maximum_candidate_members=64, maximum_candidate_packet_bytes=size)
    assert len(environment_candidates(traces, spec)) == 3
    script, bound = reviewer(lambda members, _variant: answer("compatible", members))
    tiny = spec.model_copy(update={"maximum_candidate_packet_bytes": size - 1})
    with pytest.raises(CandidateCapacityError, match="above registered"):
        review_environment_families(
            clustering({"c00": tuple(trace.trace_id for trace in traces)}),
            bound,
            traces={trace.trace_id: trace for trace in traces},
            eligible=tuple(trace.trace_id for trace in traces),
            held_out=(),
            spec=tiny,
        )
    assert not script.requests


def test_each_cluster_candidate_intersection_is_independently_reviewed() -> None:
    """One cohort yields multiple bases, and every allowed trace has exactly one base."""
    traces = [order_trace(f"o{i}") for i in range(7)]
    spec = CompatibilitySpec(
        max_depth=0,
        maximum_candidate_members=3,
        maximum_candidate_packet_bytes=48_000,
    )
    script, bound = reviewer(lambda members, _variant: answer("compatible", members))
    groups = {"c00": ("o0", "o1", "o3", "o4"), "c01": ("o2", "o5", "o6")}
    record = review_environment_families(
        clustering(groups),
        bound,
        traces={trace.trace_id: trace for trace in traces},
        eligible=tuple(trace.trace_id for trace in traces),
        held_out=(),
        spec=spec,
    )
    members = [member for family in record.families for member in family.members]
    assert sorted(members) == sorted(trace.trace_id for trace in traces)
    assert len(set(members)) == len(members)
    assert len(record.families) == 5
    assert {family.cluster_id for family in record.families} == set(groups)
    assert all("environment-candidate" in family.path for family in record.families)
    assert len({family.family_id for family in record.families}) == 5
    roots = candidate_root_bound(traces, spec, clusters=2)
    assert len(script.requests) <= spec.calls(len(traces), roots=roots)
    assert not record.uncertain
    changed = traces[0].model_copy(update={"task": "Different source content"})
    revised = review_environment_families(
        clustering(groups),
        bound,
        traces={trace.trace_id: trace for trace in [changed, *traces[1:]]},
        eligible=tuple(trace.trace_id for trace in traces),
        held_out=(),
        spec=spec,
    )
    original = next(family for family in record.families if "o0" in family.members)
    updated = next(family for family in revised.families if "o0" in family.members)
    assert updated.family_id != original.family_id


def test_depth_bound_uses_roots_fanout_and_preserves_unset_serialization() -> None:
    """Registered recursion bounds are truthful and unset canonical fields stay unchanged."""
    spec = CompatibilitySpec(max_depth=0, max_turns=2, repairs=1)
    assert spec.calls_per_group() == 11
    assert spec.calls(2_875, roots=12) == 132
    assert spec.model_copy(update={"max_depth": 1}).calls(100, roots=1) == 17 * 11
    assert spec.model_copy(update={"max_depth": 4}).calls(100, roots=100) == 199 * 11
    assert spec.calls(0, roots=0) == 0
    dumped = spec.model_dump(mode="json")
    assert "maximum_candidate_members" not in dumped
    assert "maximum_candidate_packet_bytes" not in dumped
    assert sha256_json(spec) == sha256_json(dumped)
    with pytest.raises(ValueError, match="both be registered"):
        CompatibilitySpec(maximum_candidate_members=64)
    with pytest.raises(ValueError, match="nonnegative"):
        spec.calls(10, roots=-1)


def test_candidate_protocol_rekeys_only_explicitly_bounded_family_reviews(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partition changes rekey bounded results while preserving original review semantics."""
    traces = {"o0": order_trace("o0")}
    clustered = clustering({"c00": ("o0",)})
    _, bound = reviewer(lambda members, _variant: answer("compatible", members))
    original = CompatibilitySpec(max_depth=0)
    bounded = original.model_copy(
        update={
            "maximum_candidate_members": 64,
            "maximum_candidate_packet_bytes": 48_000,
        }
    )
    legacy = families_identity(clustered, bound, original, traces, ())
    before = families_identity(clustered, bound, bounded, traces, ())
    candidates = environment_candidates(tuple(traces.values()), bounded)
    review_protocol = review_protocol_sha256()
    monkeypatch.setattr(candidate_owner, "ENVIRONMENT_CANDIDATE_PROTOCOL", "changed-partition")
    assert families_identity(clustered, bound, original, traces, ()) == legacy
    assert families_identity(clustered, bound, bounded, traces, ()) != before
    assert environment_candidates(tuple(traces.values()), bounded) != candidates
    assert review_protocol_sha256() == review_protocol

"""Tests for duplicate-union boundary checks on selection geometry."""

from __future__ import annotations

import pytest

from exp.simulation.mining.deduplicate import DeduplicatedTrace, validate_selection_geometry
from exp.simulation.mining.descriptors import CoverageDescriptor, RoutingDescriptor


def _coverage(trace_id: str) -> CoverageDescriptor:
    """Return an empty mining-only coverage descriptor."""
    return CoverageDescriptor(
        trace_id=trace_id,
        tool_transitions=(),
        outcome_statuses=(),
        is_escalation=False,
        span_counts=(1,),
        domains=(),
    )


def _candidate(
    trace_id: str, vector: tuple[float, ...], selection: tuple[float, ...] = ()
) -> DeduplicatedTrace:
    """Return one candidate positioned in the named geometries."""
    return DeduplicatedTrace(
        representative_trace_id=trace_id,
        lineage_group_id="leakage-1",
        source_trace_ids=(trace_id,),
        workload_mass=1,
        routing_descriptor=RoutingDescriptor(
            trace_id=trace_id, intent="do", initial_context={}, tools=(), tags=()
        ),
        coverage_descriptor=_coverage(trace_id),
        source_coverage_descriptors=(_coverage(trace_id),),
        vector=vector,
        selection_vector=selection,
    )


def test_one_shared_width_is_accepted() -> None:
    """The ordinary case: every candidate positioned by the same geometry."""
    validate_selection_geometry(
        [_candidate("trace-1", (1.0, 0.0)), _candidate("trace-2", (0.0, 1.0))]
    )
    validate_selection_geometry([])


def test_the_descriptor_vector_is_the_default_geometry() -> None:
    """Provider-free mining keeps one geometry, which is what makes the default path intact."""
    candidate = _candidate("trace-1", (1.0, 0.0))

    assert candidate.selection_vector == candidate.vector


def test_two_widths_fail_at_the_boundary_naming_both_traces() -> None:
    """Without this the mismatch surfaces as a bare zip() error inside the metric."""
    candidates = [
        _candidate("trace-1", (1.0, 0.0), selection=(1.0, 0.0)),
        _candidate("trace-2", (0.0, 1.0), selection=(0.0, 1.0, 0.0)),
    ]

    with pytest.raises(ValueError, match="share one selection geometry") as caught:
        validate_selection_geometry(candidates)

    message = str(caught.value)
    assert "trace-1 carries 2 coordinates" in message
    assert "trace-2 carries 3" in message


def test_a_candidate_with_no_position_is_named() -> None:
    """An unpositioned candidate is nearest to nothing, so it must not pass silently."""
    candidates = [_candidate("trace-9", ())]

    with pytest.raises(ValueError, match="trace-9 has an empty selection vector"):
        validate_selection_geometry(candidates)

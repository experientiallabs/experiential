"""Deterministic behavior tests for leakage-safe representative task mining."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import FailureCode, JsonObject, SourceIdentity, StructuredFailure
from exp.common.project import ArtifactStore
from exp.common.project.paths import ProjectPaths
from exp.common.tasks import TaskSet, ToolSchema
from exp.common.traces import Trace, TraceOutcome, TraceSource, TraceSpan
from exp.simulation.mining.descriptors import routing_descriptor
from exp.simulation.mining.lineage import assign_source_lineages
from exp.simulation.mining.service import MiningSpec, mine_tasks, persist_task_set


class IndexEmbedder:
    """Returns deterministic orthogonal vectors in source order for mining fixtures."""

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(1.0 if index == coordinate else 0.0 for coordinate in range(len(texts)))
            for index, _text in enumerate(texts)
        )


class ContentStableEmbedder:
    """Embeds descriptor content with a deterministic hash rather than input position."""

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(
                1.0 if byte & (1 << bit) else -1.0
                for byte in hashlib.sha256(text.encode("utf-8")).digest()
                for bit in range(8)
            )
            for text in texts
        )


class ProposalModel:
    """Returns one deterministic cleanup proposal without provider calls."""

    def __init__(self, proposal: str) -> None:
        self._proposal = proposal

    def cleanup_instruction(
        self,
        *,
        instruction: str,
        initial_context: JsonObject,
        tools: tuple[ToolSchema, ...],
    ) -> str:
        return self._proposal


class SameVectorEmbedder:
    """Makes a test pair semantically identical without changing its source requests."""

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, float], ...]:
        return tuple((1.0, 0.0) for _text in texts)


def _trace(
    index: int,
    *,
    task: str | None = None,
    conversation_id: str | None = None,
    outcome: str = "success",
    escalation: bool = False,
    span_count: int = 1,
    completion: str | None = None,
) -> Trace:
    started = datetime(2026, 8, 11, tzinfo=UTC) + timedelta(minutes=index)
    attributes: JsonObject = {
        "exp.customer.id": "customer-1",
        "exp.request.tags": ["domain:support"],
        "gen_ai.tool.name": "lookup_order",
    }
    if escalation:
        attributes["exp.outcome.escalated"] = True
    if completion is not None:
        attributes["gen_ai.completion"] = completion
    spans = tuple(
        TraceSpan(
            span_id=f"span-{index}-{step}",
            name="agent.model_call" if step == 0 else "agent.tool_call",
            started_at=started + timedelta(seconds=step),
            ended_at=started + timedelta(seconds=step + 1),
            attributes=attributes if step == 0 else {"gen_ai.tool.name": "lookup_order"},
        )
        for step in range(span_count)
    )
    trace_outcome = (
        TraceOutcome(status="success")
        if outcome == "success"
        else TraceOutcome(
            status="failure",
            failure=StructuredFailure(code=FailureCode.INTERNAL, message="recorded failure"),
        )
    )
    return Trace(
        trace_id=f"trace-{index}",
        conversation_id=conversation_id or f"conversation-{index}",
        task=task or f"Handle customer request {index}",
        initial_context={"channel": "email", "case": index},
        tools=(
            ToolSchema(
                name="lookup_order",
                description="Look up one customer order.",
                input_schema={"type": "object"},
            ),
        ),
        spans=spans,
        outcome=trace_outcome,
        source=TraceSource(
            identity=SourceIdentity(kind="production", source_id="fixture", sha256="a" * 64),
            semantic_convention_version="1.37.0",
        ),
    )


def _coverage_trace(
    index: int,
    *,
    domain: str,
    tool_name: str,
    outcome: str,
    span_count: int,
) -> Trace:
    """Return a source trace with mining-only facets distinct from its router descriptor."""
    trace = _trace(
        index,
        task="Handle one customer request",
        outcome=outcome,
        span_count=span_count,
    )
    spans = tuple(
        span.model_copy(update={"attributes": {"gen_ai.tool.name": tool_name}})
        for span in trace.spans
    )
    return trace.model_copy(update={"initial_context": {"domain": domain}, "spans": spans})


def test_router_descriptor_ignores_later_action_length_and_outcome_evidence() -> None:
    trace = _trace(1)
    later = TraceSpan(
        span_id="later",
        name="agent.tool_call",
        started_at=datetime(2026, 8, 11, 1, tzinfo=UTC),
        ended_at=datetime(2026, 8, 11, 1, 1, tzinfo=UTC),
        attributes={"gen_ai.tool.name": "escalate_case"},
    )
    changed = trace.model_copy(
        update={
            "spans": (*trace.spans, later),
            "outcome": TraceOutcome(
                status="failure",
                failure=StructuredFailure(code=FailureCode.INTERNAL, message="later failure"),
            ),
        }
    )

    assert (
        routing_descriptor(changed).canonical_payload()
        == routing_descriptor(trace).canonical_payload()
    )


def test_default_100_trace_path_selects_50_fit_and_20_held_out_tasks() -> None:
    result = mine_tasks(tuple(_trace(index) for index in range(100)), embedder=IndexEmbedder())

    assert len(result.tasks) == 70
    assert sum(task.partition == "fit" for task in result.tasks) == 50
    assert sum(task.partition == "held_out" for task in result.tasks) == 20
    assert sum(task.workload_weight for task in result.tasks) == pytest.approx(1.0)
    assert result.coverage.operating_range == "within"
    assert result.coverage.split_separation_verified
    assert not set(result.partition.fit_lineage_group_ids).intersection(
        result.partition.held_out_lineage_group_ids
    )


def test_over_70_lineage_core_selection_is_input_order_independent() -> None:
    traces = tuple(
        _trace(
            index,
            task=f"Resolve specialty customer workflow {index * index}",
            conversation_id=f"lineage-{index}",
            outcome="failure" if index % 11 == 0 else "success",
            escalation=index % 13 == 0,
            span_count=8 if index % 7 == 0 else 1,
        )
        for index in range(80)
    )

    result = mine_tasks(traces, embedder=ContentStableEmbedder())
    reversed_result = mine_tasks(tuple(reversed(traces)), embedder=ContentStableEmbedder())

    assert len(result.analysis.leakage_groups) == 80
    assert len(result.tasks) == 70
    assert tuple(task.task_id for task in result.tasks) == tuple(
        task.task_id for task in reversed_result.tasks
    )
    assert {task.task_id: task.workload_weight for task in result.tasks} == {
        task.task_id: task.workload_weight for task in reversed_result.tasks
    }
    assert result.partition.fit_lineage_group_ids == reversed_result.partition.fit_lineage_group_ids
    assert (
        result.partition.held_out_lineage_group_ids
        == reversed_result.partition.held_out_lineage_group_ids
    )
    assert result.coverage.fit == reversed_result.coverage.fit
    assert result.coverage.held_out == reversed_result.coverage.held_out
    assert result.coverage.selections == reversed_result.coverage.selections
    assert result.coverage.facets == reversed_result.coverage.facets
    assert result.coverage.distances == reversed_result.coverage.distances


def test_partition_targets_survive_extreme_workload_skew() -> None:
    """Lineage workload mass cannot displace requested fit and held-out counts.

    Explicit small budgets exercise the partition invariant; the separate default-path test owns
    the production 50/20 cardinalities.
    """
    light = tuple(_trace(index, conversation_id=f"light-{index}") for index in range(6))
    heavy = tuple(_trace(1_000 + index, conversation_id="heavy-lineage") for index in range(20))

    result = mine_tasks(
        (*light, *heavy),
        MiningSpec(fit_task_budget=5, held_out_task_budget=2),
        embedder=IndexEmbedder(),
    )

    assert len(result.analysis.leakage_groups) == 7
    assert len(result.partition.fit_lineage_group_ids) == 5
    assert len(result.partition.held_out_lineage_group_ids) == 2
    assert sum(task.partition == "fit" for task in result.tasks) == 5
    assert sum(task.partition == "held_out" for task in result.tasks) == 2


def test_under_70_lineages_are_all_retained_with_deterministic_underfill_evidence() -> None:
    traces = tuple(_trace(index, conversation_id=f"lineage-{index}") for index in range(69))

    result = mine_tasks(traces, embedder=IndexEmbedder())
    reversed_result = mine_tasks(tuple(reversed(traces)), embedder=IndexEmbedder())

    selected_lineages = {task.lineage_group_id for task in result.tasks}
    all_lineages = {
        *result.partition.fit_lineage_group_ids,
        *result.partition.held_out_lineage_group_ids,
    }
    assert selected_lineages == all_lineages
    assert len(result.tasks) == 69
    assert result.partition.underfilled_reason is not None
    assert result.coverage.partition_underfilled_reason == result.partition.underfilled_reason
    assert result.coverage.fit.underfilled
    assert result.partition.fit_lineage_group_ids == reversed_result.partition.fit_lineage_group_ids
    assert (
        result.partition.held_out_lineage_group_ids
        == reversed_result.partition.held_out_lineage_group_ids
    )


def test_mining_requires_an_explicit_embedding_interface() -> None:
    with pytest.raises(ValueError, match="explicit DescriptorEmbedder"):
        mine_tasks((_trace(1),))


def test_small_lineage_sets_keep_every_lineage_even_with_multiple_source_traces() -> None:
    """Underfilled selection retains all lineages when each has multiple source traces."""
    traces = tuple(
        _trace(index, conversation_id=f"conversation-{index // 2}") for index in range(10)
    )

    result = mine_tasks(
        traces,
        MiningSpec(fit_task_budget=4, held_out_task_budget=2),
        embedder=IndexEmbedder(),
    )

    selected_lineages = {task.lineage_group_id for task in result.tasks}
    all_lineages = {
        *result.partition.fit_lineage_group_ids,
        *result.partition.held_out_lineage_group_ids,
    }
    assert len(result.analysis.leakage_groups) == 5
    assert selected_lineages == all_lineages


def test_exact_duplicate_lineages_are_unioned_before_partition_and_keep_workload_mass() -> None:
    first = _trace(1, task="Cancel reservation R-17", conversation_id="conversation-a")
    second = _trace(
        2,
        task="Cancel reservation R-17",
        conversation_id="conversation-b",
    ).model_copy(update={"initial_context": first.initial_context})

    result = mine_tasks((first, second), embedder=IndexEmbedder())

    assert len(result.analysis.edges) == 1
    assert result.analysis.edges[0].kind == "exact"
    assert len(result.analysis.leakage_groups) == 1
    assert len(result.tasks) == 1
    assert result.tasks[0].source_trace_ids == ("trace-1", "trace-2")
    assert result.tasks[0].workload_weight == pytest.approx(1.0)
    assert result.coverage.duplicate_trace_count == 1
    assert result.coverage.held_out.underfilled


def test_semantic_duplicate_lineages_are_unioned_before_partition() -> None:
    first = _trace(1, task="Cancel a reservation", conversation_id="conversation-a")
    second = _trace(2, task="Please cancel this booking", conversation_id="conversation-b")

    result = mine_tasks((first, second), embedder=SameVectorEmbedder())

    assert len(result.analysis.edges) == 1
    assert result.analysis.edges[0].kind == "semantic"
    assert len(result.analysis.leakage_groups) == 1
    assert result.analysis.candidates[0].source_trace_ids == ("trace-1", "trace-2")


def test_source_lineages_use_conversation_then_stable_customer_time_buckets() -> None:
    first = _trace(1).model_copy(update={"conversation_id": None})
    second = _trace(2).model_copy(update={"conversation_id": None})
    later = _trace(2_000).model_copy(update={"conversation_id": None})

    assignments = assign_source_lineages((first, second, later))
    reverse_assignments = assign_source_lineages((later, second, first))
    by_trace = {assignment.trace_id: assignment for assignment in assignments}

    assert by_trace["trace-1"].lineage_group_id == by_trace["trace-2"].lineage_group_id
    assert by_trace["trace-1"].lineage_group_id != by_trace["trace-2000"].lineage_group_id
    assert {item.trace_id: item.lineage_group_id for item in assignments} == {
        item.trace_id: item.lineage_group_id for item in reverse_assignments
    }


def test_reserved_failure_tail_is_selected_and_missing_slots_remain_reported() -> None:
    traces = tuple(
        _trace(
            index,
            conversation_id="one-conversation",
            outcome="failure" if index == 3 else "success",
            escalation=index == 4,
            span_count=8 if index == 5 else 1,
        )
        for index in range(6)
    )

    result = mine_tasks(
        traces,
        MiningSpec(fit_task_budget=5, held_out_task_budget=0),
        embedder=IndexEmbedder(),
    )

    reasons = [
        reason for selection in result.coverage.selections for reason in selection.selection_reasons
    ]
    assert "reserved:failure" in reasons
    assert "reserved:escalation" in reasons
    assert "reserved:long" in reasons
    assert "reserved:boundary" in reasons
    assert any(item.startswith("rare_tool:") for item in result.coverage.fit.missing_reserved_slots)


def test_cleanup_rejects_answer_leakage_and_invented_tools_without_a_paid_call() -> None:
    trace = _trace(
        1,
        task="Look up the order status",
        conversation_id="cleanup-case",
        completion="The order status is 42.",
    )
    leak = mine_tasks(
        (trace,),
        MiningSpec(fit_task_budget=1, held_out_task_budget=0),
        embedder=IndexEmbedder(),
        instruction_cleanup_model=ProposalModel("Tell the customer the order status is 42."),
    )
    invented_tool = mine_tasks(
        (trace,),
        MiningSpec(fit_task_budget=1, held_out_task_budget=0),
        embedder=IndexEmbedder(),
        instruction_cleanup_model=ProposalModel(
            "Use tool delete_account to look up the order status."
        ),
    )
    invented_requirement = mine_tasks(
        (trace,),
        MiningSpec(fit_task_budget=1, held_out_task_budget=0),
        embedder=IndexEmbedder(),
        instruction_cleanup_model=ProposalModel(
            "Ignore the customer and look up the order status."
        ),
    )

    assert leak.tasks[0].instruction == trace.task
    assert "leaks" in leak.cleanup_results[0][1].reason
    assert invented_tool.tasks[0].instruction == trace.task
    assert "unavailable tool" in invented_tool.cleanup_results[0][1].reason
    assert invented_requirement.tasks[0].instruction == trace.task
    assert "invent" in invented_requirement.cleanup_results[0][1].reason


def test_cleanup_rejects_json_encoded_standard_output_message_answer_leakage() -> None:
    trace = _trace(1, task="Look up the order status", conversation_id="cleanup-output-message")
    first_span = trace.spans[0].model_copy(
        update={
            "attributes": {
                **trace.spans[0].attributes,
                "gen_ai.output.messages": json.dumps(
                    [
                        {
                            "role": "assistant",
                            "content": "The order status is shipped.",
                        }
                    ]
                ),
            }
        }
    )
    trace = trace.model_copy(update={"spans": (first_span, *trace.spans[1:])})

    result = mine_tasks(
        (trace,),
        MiningSpec(fit_task_budget=1, held_out_task_budget=0),
        embedder=IndexEmbedder(),
        instruction_cleanup_model=ProposalModel("Tell the customer the order status is shipped."),
    )

    assert result.tasks[0].instruction == trace.task
    assert "leaks" in result.cleanup_results[0][1].reason


def test_cleanup_accepts_a_source_preserving_faked_proposal() -> None:
    trace = _trace(1, task="Look up the order status", conversation_id="cleanup-accepted")

    result = mine_tasks(
        (trace,),
        MiningSpec(fit_task_budget=1, held_out_task_budget=0),
        embedder=IndexEmbedder(),
        instruction_cleanup_model=ProposalModel("Please look up the order status."),
    )

    assert result.tasks[0].instruction == "Please look up the order status."
    assert result.cleanup_results[0][1].accepted


def test_duplicate_coverage_retains_mixed_source_facet_mass() -> None:
    first = _coverage_trace(
        1,
        domain="travel",
        tool_name="lookup_reservation",
        outcome="success",
        span_count=1,
    )
    second = _coverage_trace(
        2,
        domain="billing",
        tool_name="cancel_reservation",
        outcome="failure",
        span_count=8,
    )

    result = mine_tasks(
        (first, second),
        MiningSpec(fit_task_budget=1, held_out_task_budget=0),
        embedder=SameVectorEmbedder(),
    )

    facet_mass = {
        (facet.dimension, facet.value): (
            facet.input_workload_mass,
            facet.directly_selected_workload_mass,
        )
        for facet in result.coverage.facets
    }
    assert len(result.analysis.candidates) == 1
    assert facet_mass[("tool", "lookup_reservation")] == (1, 1)
    assert facet_mass[("tool", "cancel_reservation")] == (1, 1)
    assert facet_mass[("domain", "travel")] == (1, 1)
    assert facet_mass[("domain", "billing")] == (1, 1)
    assert facet_mass[("outcome", "success")] == (1, 1)
    assert facet_mass[("outcome", "failure")] == (1, 1)
    assert facet_mass[("complexity", "short")] == (1, 1)
    assert facet_mass[("complexity", "long")] == (1, 1)


def test_persisted_task_set_reuses_the_w2_task_and_artifact_contracts(tmp_path: Path) -> None:
    result = mine_tasks(
        (_trace(1), _trace(2)),
        MiningSpec(fit_task_budget=1, held_out_task_budget=1),
        embedder=IndexEmbedder(),
    )
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))

    task_set = persist_task_set(
        result,
        store,
        task_set_id="tasks-a",
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )

    assert task_set.task_ids == tuple(task.task_id for task in result.tasks)
    assert store.read("tasks-a").manifest.artifact_type == "task-set"
    assert TaskSet.model_validate_json(store.read_bytes("tasks-a", "task-set.json")) == task_set

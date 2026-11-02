"""Deterministic behavior tests for leakage-safe representative task mining."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import FailureCode, JsonObject, SourceIdentity, StructuredFailure
from exp.common.project import ArtifactStore
from exp.common.project.paths import ProjectPaths
from exp.common.tasks import TaskSet, ToolSchema
from exp.common.traces import Trace, TraceOutcome, TraceSource, TraceSpan
from exp.common.traces.ingest.vendor_trace import SYNTHETIC_TIME_ATTRIBUTE
from exp.simulation.mining.clustering import (
    AtomsHierarchyClusterSpec,
    BernoulliMixtureClusterSpec,
    ClusterSpec,
    DescriptorClusterSpec,
    KMeansClusterSpec,
    PatternHierarchyClusterSpec,
)
from exp.simulation.mining.descriptors import HashingDescriptorEmbedder, routing_descriptor
from exp.simulation.mining.features import DecisionTagSelectionSpec, HashingSelectionSpec
from exp.simulation.mining.lineage import assign_source_lineages
from exp.simulation.mining.questions import (
    EXAMPLE_BANK_ID,
    QuestionBank,
    packaged_question_bank,
)
from exp.simulation.mining.sampling import (
    DiversitySamplingSpec,
    HierarchicalSamplingSpec,
    MedoidSamplingSpec,
    ProportionalSamplingSpec,
    RepresentativeSamplingSpec,
    SamplingSpec,
    StratifiedSamplingSpec,
)
from exp.simulation.mining.service import MiningSpec, mine_tasks, persist_task_set
from exp.simulation.mining.tagging import TagMatrix, TagRow, tagged_view
from exp.simulation.mining.views import Unit


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
    """Similar requests share a leakage boundary while retaining distinct scenarios."""
    first = _trace(1, task="Cancel a reservation", conversation_id="conversation-a")
    second = _trace(2, task="Please cancel this booking", conversation_id="conversation-b")

    result = mine_tasks((first, second), embedder=SameVectorEmbedder())

    assert len(result.analysis.edges) == 1
    assert result.analysis.edges[0].kind == "semantic"
    assert len(result.analysis.leakage_groups) == 1
    assert [candidate.source_trace_ids for candidate in result.analysis.candidates] == [
        ("trace-1",),
        ("trace-2",),
    ]
    assert len(result.tasks) == 2
    assert len({task.partition for task in result.tasks}) == 1


@pytest.mark.parametrize("exact", [False, True])
def test_synthetic_trace_boundaries_preserve_duplicate_leakage_exclusions(exact: bool) -> None:
    """Missing measured timing never separates exact or semantic duplicate evidence."""
    first = _trace(1, task="Cancel a reservation")
    second = _trace(2, task=first.task if exact else "Please cancel this booking").model_copy(
        update={"initial_context": first.initial_context}
    )
    traces = tuple(
        trace.model_copy(
            update={
                "conversation_id": None,
                "spans": tuple(
                    span.model_copy(
                        update={"attributes": {**span.attributes, SYNTHETIC_TIME_ATTRIBUTE: True}}
                    )
                    for span in trace.spans
                ),
            }
        )
        for trace in (first, second)
    )

    result = mine_tasks(traces, embedder=SameVectorEmbedder())

    assert len({item.lineage_group_id for item in result.lineage_assignments}) == 2
    assert len(result.analysis.leakage_groups) == 1
    assert [edge.kind for edge in result.analysis.edges] == ["exact" if exact else "semantic"]
    assert len(result.tasks) == (1 if exact else 2)
    assert len({task.lineage_group_id for task in result.tasks}) == 1
    assert len({task.partition for task in result.tasks}) == 1


@pytest.mark.parametrize("case_in_context", [False, True])
def test_twenty_distinct_company_cases_survive_shared_instructions(
    case_in_context: bool,
) -> None:
    """Shared prompt text cannot erase entity differences or hide an exact repeat."""
    context: JsonObject = {"instructions": "Research the company using reliable sources. " * 200}
    traces = tuple(
        _trace(
            index,
            task="Research this company" if case_in_context else f"Research company {index}",
        ).model_copy(
            update={
                "initial_context": {
                    **context,
                    **({"company": index} if case_in_context else {}),
                }
            }
        )
        for index in range(20)
    )
    repeat = traces[0].model_copy(
        update={"trace_id": "trace-repeat", "conversation_id": "conversation-repeat"}
    )
    corpus = (*traces, repeat)

    result = mine_tasks(corpus, embedder=HashingDescriptorEmbedder())
    replay = mine_tasks(tuple(reversed(corpus)), embedder=HashingDescriptorEmbedder())

    assert len(result.tasks) == 20
    assert len(result.analysis.candidates) == 20
    assert result.coverage.duplicate_trace_count == 1
    assert any(edge.kind == "semantic" for edge in result.analysis.edges)
    duplicated = next(task for task in result.tasks if "trace-repeat" in task.source_trace_ids)
    assert duplicated.source_trace_ids == ("trace-0", "trace-repeat")
    assert duplicated.workload_weight == pytest.approx(2 / 21)
    assert sum(task.workload_weight for task in result.tasks) == pytest.approx(1)
    assert {source for task in result.tasks for source in task.source_trace_ids} == {
        trace.trace_id for trace in corpus
    }
    assert result.coverage.split_separation_verified
    assert not set(result.partition.fit_lineage_group_ids).intersection(
        result.partition.held_out_lineage_group_ids
    )
    assert result.tasks == replay.tasks
    assert result.coverage == replay.coverage


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


def test_similar_cases_retain_mixed_source_facet_mass_when_both_selected() -> None:
    """Keeping two similar scenarios preserves each source's direct coverage."""
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
        MiningSpec(fit_task_budget=2, held_out_task_budget=0),
        embedder=SameVectorEmbedder(),
    )

    facet_mass = {
        (facet.dimension, facet.value): (
            facet.input_workload_mass,
            facet.directly_selected_workload_mass,
        )
        for facet in result.coverage.facets
    }
    assert len(result.analysis.candidates) == 2
    assert facet_mass[("tool", "lookup_reservation")] == (1, 1)
    assert facet_mass[("tool", "cancel_reservation")] == (1, 1)
    assert facet_mass[("domain", "travel")] == (1, 1)
    assert facet_mass[("domain", "billing")] == (1, 1)
    assert facet_mass[("outcome", "success")] == (1, 1)
    assert facet_mass[("outcome", "failure")] == (1, 1)
    assert facet_mass[("complexity", "short")] == (1, 1)
    assert facet_mass[("complexity", "long")] == (1, 1)


def test_exact_duplicates_retain_each_source_outcome_and_tool_facet() -> None:
    """Exact requests collapse while their different observed outcomes remain counted."""
    first = _coverage_trace(
        1, domain="travel", tool_name="lookup_reservation", outcome="success", span_count=1
    )
    second = _coverage_trace(
        2, domain="travel", tool_name="cancel_reservation", outcome="failure", span_count=8
    )
    result = mine_tasks((first, second), embedder=SameVectorEmbedder())

    assert len(result.analysis.candidates) == 1
    assert result.tasks[0].source_trace_ids == ("trace-1", "trace-2")
    facet_mass = {
        (facet.dimension, facet.value): (
            facet.input_workload_mass,
            facet.directly_selected_workload_mass,
        )
        for facet in result.coverage.facets
    }
    assert facet_mass[("domain", "travel")] == (2, 2)
    for facet in (
        ("tool", "lookup_reservation"),
        ("tool", "cancel_reservation"),
        ("outcome", "success"),
        ("outcome", "failure"),
        ("complexity", "short"),
        ("complexity", "long"),
    ):
        assert facet_mass[facet] == (1, 1)


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


class _ProfileTagger:
    """A no-network tagger whose answers describe the fixture corpus rather than hashing it.

    Traces split into two readable profiles by index parity, with a third group of long episodes.
    Unlike the hashing tagger this carries real signal, so a test can check that the geometry
    changed the selection for a reason rather than by accident.
    """

    def tag(self, units: Sequence[Unit], bank: QuestionBank) -> TagMatrix:
        """Answer every question about every unit from the unit's own trace index."""
        rows = []
        for unit in units:
            index = int(unit.trace_id.rsplit("-", 1)[1])
            profile = index % 3
            rows.append(
                TagRow(
                    trace_id=unit.trace_id,
                    unit_hash=unit.unit_hash,
                    values=tuple(
                        0.9 if position % 3 == profile else 0.05
                        for position in range(len(bank.questions))
                    ),
                )
            )
        return TagMatrix(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            question_ids=bank.question_ids(),
            view=tagged_view(units),
            rows=tuple(rows),
        )


def _geometry_corpus(count: int = 40) -> tuple[Trace, ...]:
    """Return a corpus of distinct lineages whose descriptor text carries no profile signal."""
    return tuple(
        _trace(index, task=f"Handle workflow {index}", conversation_id=f"lineage-{index}")
        for index in range(count)
    )


def _shared_lineage_corpus(count: int = 24, lineages: int = 4) -> tuple[Trace, ...]:
    """Return a corpus whose cases share a few lineages, leaving budget for a strategy to spend.

    One anchor per lineage is paid for before the discretionary half of selection begins, so a
    fixture of distinct lineages leaves nothing for a sampling strategy to decide.
    """
    return tuple(
        _trace(
            index, task=f"Handle workflow {index}", conversation_id=f"lineage-{index % lineages}"
        )
        for index in range(count)
    )


def _tag_spec(
    cluster: ClusterSpec | None = None, sampling: SamplingSpec | None = None
) -> MiningSpec:
    """Return a mining spec selecting on the packaged bank with the fixture tagger."""
    return MiningSpec(
        fit_task_budget=8,
        held_out_task_budget=3,
        selection=DecisionTagSelectionSpec(bank_id=EXAMPLE_BANK_ID),
        cluster=cluster or DescriptorClusterSpec(),
        sampling=sampling or MedoidSamplingSpec(),
    )


def test_a_decision_tag_geometry_selects_differently_from_the_hashing_one() -> None:
    """If the tags did not move the selection, paying for them bought nothing."""
    corpus = _geometry_corpus()

    hashing = mine_tasks(
        corpus,
        MiningSpec(fit_task_budget=8, held_out_task_budget=3),
        embedder=ContentStableEmbedder(),
    )
    tagged = mine_tasks(
        corpus, _tag_spec(), embedder=ContentStableEmbedder(), tagger=_ProfileTagger()
    )

    assert {task.task_id for task in tagged.tasks} != {task.task_id for task in hashing.tasks}
    assert len(tagged.tasks) == len(hashing.tasks)
    assert tagged.coverage.split_separation_verified


def test_the_tag_geometry_reaches_the_persisted_coverage_distances() -> None:
    """Coverage has to describe the geometry that selected, or the audit is of something else."""
    corpus = _geometry_corpus()

    hashing = mine_tasks(
        corpus,
        MiningSpec(fit_task_budget=8, held_out_task_budget=3),
        embedder=ContentStableEmbedder(),
    )
    tagged = mine_tasks(
        corpus, _tag_spec(), embedder=ContentStableEmbedder(), tagger=_ProfileTagger()
    )

    hashing_similarity = {item.trace_id: item.similarity for item in hashing.coverage.distances}
    tagged_similarity = {item.trace_id: item.similarity for item in tagged.coverage.distances}
    assert set(hashing_similarity) == set(tagged_similarity)
    assert hashing_similarity != tagged_similarity


def test_the_duplicate_and_partition_evidence_is_untouched_by_the_geometry() -> None:
    """Selection geometry must never move the leakage boundary."""
    corpus = _geometry_corpus()

    hashing = mine_tasks(
        corpus,
        MiningSpec(fit_task_budget=8, held_out_task_budget=3),
        embedder=ContentStableEmbedder(),
    )
    tagged = mine_tasks(
        corpus, _tag_spec(), embedder=ContentStableEmbedder(), tagger=_ProfileTagger()
    )

    assert tagged.analysis.edges == hashing.analysis.edges
    assert tagged.analysis.leakage_groups == hashing.analysis.leakage_groups
    assert tagged.partition == hashing.partition
    assert [candidate.vector for candidate in tagged.analysis.candidates] == [
        candidate.vector for candidate in hashing.analysis.candidates
    ]


def test_tagged_candidates_carry_the_tag_vector_not_the_descriptor_vector() -> None:
    """A defaulted selection vector would silently restore the hashing geometry."""
    result = mine_tasks(
        _geometry_corpus(), _tag_spec(), embedder=ContentStableEmbedder(), tagger=_ProfileTagger()
    )

    for candidate in result.analysis.candidates:
        assert candidate.selection_vector != candidate.vector
        assert len(candidate.selection_vector) == len(packaged_question_bank().questions)


def test_the_bernoulli_mixture_runs_end_to_end_on_tagged_answers() -> None:
    """The one backend that needs untransformed answers has to receive them through mining."""
    result = mine_tasks(
        _geometry_corpus(),
        _tag_spec(BernoulliMixtureClusterSpec(k=3, seed=1)),
        embedder=ContentStableEmbedder(),
        tagger=_ProfileTagger(),
    )

    assert len(result.tasks) == 11
    assert all(item.cluster_id >= 0 for item in result.coverage.selections)
    assert all(item.candidate_count > 0 for item in result.coverage.fit.clusters)


@pytest.mark.parametrize(
    ("cluster", "sampling"),
    [
        (KMeansClusterSpec(k=4, seed=1), None),
        (AtomsHierarchyClusterSpec(atom_count=8, groups=3, seed=1), None),
        (None, ProportionalSamplingSpec()),
        (None, DiversitySamplingSpec()),
        (None, StratifiedSamplingSpec()),
        (
            AtomsHierarchyClusterSpec(atom_count=8, groups=3, seed=1),
            StratifiedSamplingSpec(),
        ),
        (PatternHierarchyClusterSpec(), HierarchicalSamplingSpec()),
        (
            AtomsHierarchyClusterSpec(atom_count=8, groups=3, seed=1),
            HierarchicalSamplingSpec(),
        ),
    ],
)
def test_every_alternative_stage_satisfies_the_downstream_validators(
    cluster: ClusterSpec | None, sampling: SamplingSpec | None
) -> None:
    """The coverage contract is strict, so an alternative stage has to honour all of it."""
    corpus = _geometry_corpus()

    result = mine_tasks(
        corpus,
        _tag_spec(cluster, sampling),
        embedder=ContentStableEmbedder(),
        tagger=_ProfileTagger(),
    )

    fit_ids = {task.task_id for task in result.tasks if task.partition == "fit"}
    held_out_ids = {task.task_id for task in result.tasks if task.partition == "held_out"}
    assert not fit_ids & held_out_ids
    assert sum(task.workload_weight for task in result.tasks) == pytest.approx(1.0)
    assert sum(item.workload_mass for item in result.coverage.selections) == len(corpus)
    assert result.coverage.split_separation_verified
    assert all(item.workload_mass > 0 for item in result.coverage.selections)


def test_an_alternative_stage_is_still_input_order_independent() -> None:
    """A seeded backend must not let corpus order into the result."""
    corpus = _geometry_corpus()
    spec = _tag_spec(cluster=KMeansClusterSpec(k=4, seed=7), sampling=ProportionalSamplingSpec())

    forward = mine_tasks(corpus, spec, embedder=ContentStableEmbedder(), tagger=_ProfileTagger())
    backward = mine_tasks(
        tuple(reversed(corpus)), spec, embedder=ContentStableEmbedder(), tagger=_ProfileTagger()
    )

    assert tuple(task.task_id for task in forward.tasks) == tuple(
        task.task_id for task in backward.tasks
    )
    assert forward.coverage == backward.coverage


def test_a_decision_tag_geometry_without_a_tagger_is_refused() -> None:
    """A hidden fallback to hashing would be indistinguishable from success."""
    with pytest.raises(ValueError, match="needs an explicit tagger"):
        mine_tasks(_geometry_corpus(), _tag_spec(), embedder=ContentStableEmbedder())


def test_a_tagger_that_would_go_unused_is_refused() -> None:
    """Passing a paid tagger and silently ignoring it is the expensive version of the same bug."""
    with pytest.raises(ValueError, match="would go unused"):
        mine_tasks(
            _geometry_corpus(),
            MiningSpec(fit_task_budget=8, held_out_task_budget=3),
            embedder=ContentStableEmbedder(),
            tagger=_ProfileTagger(),
        )


def test_the_mixture_backend_needs_a_tag_geometry() -> None:
    """Caught when the spec is built, not after a corpus has been tagged."""
    with pytest.raises(ValidationError, match="only the decision_tag selection geometry"):
        MiningSpec(cluster=BernoulliMixtureClusterSpec())


def test_an_untransformed_selection_geometry_is_refused() -> None:
    """Raw probability rows saturate the core-set metric and collapse every medoid."""
    with pytest.raises(ValidationError, match="saturate it"):
        MiningSpec(selection=DecisionTagSelectionSpec(bank_id=EXAMPLE_BANK_ID, transform="none"))


def test_the_default_spec_names_the_provider_free_path() -> None:
    """Everything above rests on the default being today's geometry."""
    spec = MiningSpec()

    assert spec.selection == HashingSelectionSpec()
    assert spec.cluster == DescriptorClusterSpec()
    assert spec.sampling == MedoidSamplingSpec()


def test_the_tagged_pattern_pipeline_mines_the_same_tasks_in_any_order() -> None:
    """The decision-tag route's clustering has to be order independent end to end.

    This is the tagged route's clustering with the facility-location fill over flat clusters:
    fired-proposition sets and Jaccard average linkage over the distinct ones, spent by cluster
    mass. Every tie inside it is broken by identity, so the corpus arriving reversed must mine the
    same task set.
    """
    corpus = _geometry_corpus()
    spec = _tag_spec(cluster=PatternHierarchyClusterSpec(), sampling=RepresentativeSamplingSpec())

    forward = mine_tasks(corpus, spec, embedder=ContentStableEmbedder(), tagger=_ProfileTagger())
    backward = mine_tasks(
        tuple(reversed(corpus)), spec, embedder=ContentStableEmbedder(), tagger=_ProfileTagger()
    )

    assert {task.task_id for task in forward.tasks} == {task.task_id for task in backward.tasks}
    assert forward.coverage.split_separation_verified
    assert all(selection.cluster_id >= 0 for selection in forward.coverage.selections)


def test_the_tagged_pattern_pipeline_keeps_a_continuous_selection_geometry() -> None:
    """The backend thresholds the answers, so the vectors selection measures stay continuous.

    This is the whole point of thresholding inside the clusterer. Every member of a cluster cut at
    zero fired the same propositions, so a thresholded selection vector would put them at the same
    position and leave the medoid and facility-location phases nothing but the identity tie-break.
    """
    corpus = _geometry_corpus(12)
    spec = _tag_spec(cluster=PatternHierarchyClusterSpec(cut=0.0))

    result = mine_tasks(corpus, spec, embedder=ContentStableEmbedder(), tagger=_ProfileTagger())

    values = {
        round(value, 6)
        for candidate in result.analysis.candidates
        for value in candidate.selection_vector
    }
    assert not values <= {0.0, 1.0}
    assert len({selection.cluster_id for selection in result.coverage.selections}) > 1


def test_the_pattern_backend_needs_a_tagged_selection_at_the_spec() -> None:
    """Catching the pairing when the spec is written beats discovering it after a paid tagging."""
    with pytest.raises(ValidationError, match="only the decision_tag selection geometry"):
        MiningSpec(cluster=PatternHierarchyClusterSpec())
    tagged = MiningSpec(
        selection=DecisionTagSelectionSpec(bank_id=EXAMPLE_BANK_ID),
        cluster=PatternHierarchyClusterSpec(),
    )
    assert isinstance(tagged.selection, DecisionTagSelectionSpec)
    assert tagged.selection.transform == "l2"


def test_the_hierarchical_sampler_descends_the_pattern_tree_through_mining() -> None:
    """The tag route's default pipeline has to reach the tree the clusterer published.

    Every discretionary slot is bought by a node of the agglomeration, so the audit reasons name
    tree nodes rather than flat clusters. The reserved tail slots and the lineage anchors are
    untouched by the strategy and keep their own reasons, which is the phase boundary this check
    is really protecting.

    The corpus shares four lineages between twenty-four cases on purpose. One case per lineage is
    anchored before any strategy runs, so a corpus of distinct lineages spends the whole budget on
    anchors and leaves a sampling strategy nothing to decide.
    """
    corpus = _shared_lineage_corpus()
    spec = _tag_spec(cluster=PatternHierarchyClusterSpec(), sampling=HierarchicalSamplingSpec())

    result = mine_tasks(corpus, spec, embedder=ContentStableEmbedder(), tagger=_ProfileTagger())

    reasons = [
        reason for selection in result.coverage.selections for reason in selection.selection_reasons
    ]
    assert len(result.tasks) == 11
    assert "hierarchy-representative" in reasons
    assert not any(reason.startswith("cluster-representative:") for reason in reasons)
    assert any(reason.startswith("reserved:") for reason in reasons)
    assert all(
        reason.startswith(("reserved:", "lineage-anchor", "hierarchy-representative"))
        for reason in reasons
    )
    assert result.coverage.split_separation_verified
    assert sum(item.workload_mass for item in result.coverage.selections) == len(corpus)


def test_the_tagged_hierarchical_pipeline_mines_the_same_tasks_in_any_order() -> None:
    """Descending a tree adds two more tie-breaks, both of which have to be order independent.

    Children are visited in ascending node order and the apportionment breaks its ties by node id,
    so the corpus arriving reversed must mine the identical task set in the identical order.
    """
    corpus = _shared_lineage_corpus()
    spec = _tag_spec(cluster=PatternHierarchyClusterSpec(), sampling=HierarchicalSamplingSpec())

    forward = mine_tasks(corpus, spec, embedder=ContentStableEmbedder(), tagger=_ProfileTagger())
    backward = mine_tasks(
        tuple(reversed(corpus)), spec, embedder=ContentStableEmbedder(), tagger=_ProfileTagger()
    )

    assert tuple(task.task_id for task in forward.tasks) == tuple(
        task.task_id for task in backward.tasks
    )
    assert forward.coverage == backward.coverage


def test_the_hierarchical_sampler_needs_a_backend_that_publishes_a_tree() -> None:
    """Caught when the spec is written, long before a corpus has been tagged."""
    with pytest.raises(ValidationError, match="descriptor backend publishes none"):
        MiningSpec(sampling=HierarchicalSamplingSpec())
    with pytest.raises(ValidationError, match="bernoulli_mixture backend publishes none"):
        MiningSpec(
            selection=DecisionTagSelectionSpec(bank_id=EXAMPLE_BANK_ID),
            cluster=BernoulliMixtureClusterSpec(),
            sampling=HierarchicalSamplingSpec(),
        )
    paired = MiningSpec(
        selection=DecisionTagSelectionSpec(bank_id=EXAMPLE_BANK_ID),
        cluster=PatternHierarchyClusterSpec(),
        sampling=HierarchicalSamplingSpec(),
    )
    assert paired.sampling == HierarchicalSamplingSpec()

"""Captured request identities and measured usage determine expected workload."""

import pytest

from exp.common.models import Usage
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.usage_estimate import model_turns, task_usage, token_estimate
from exp.optimize.router.automatic.service_test import _trace
from exp.optimize.router.composition_test import _completion_reservation
from exp.simulation.retrieval.tests.retrieval_test import _tool_trace


def test_parallel_tool_spans_count_one_original_request() -> None:
    """Normalization may emit multiple tool-action spans for a single billed assistant turn."""
    trace = _trace(1, _completion_reservation("candidate").model)
    first, second = trace.spans
    first = first.model_copy(
        update={
            "usage": Usage(input_tokens=400, output_tokens=50),
            "attributes": {**first.attributes, "exp.source.span.id": "request-1"},
        }
    )
    duplicate = first.model_copy(update={"span_id": "parallel-tool-2"})
    second = second.model_copy(update={"usage": Usage(input_tokens=600, output_tokens=100)})
    trace = trace.model_copy(update={"spans": (first, duplicate, second)})
    task = TaskCase(
        task_id="case",
        lineage_group_id="lineage",
        partition="fit",
        instruction=trace.task,
        source_trace_ids=(trace.trace_id,),
        workload_weight=1,
    )
    assert len(model_turns(trace)) == 2
    usage = task_usage(task, (trace,), (), top_k=2, maximum_steps=100, maximum_query_tokens=32_768)
    assert usage.assistant_input == 1_000
    assert usage.assistant_output == 150
    assert usage.turns == usage.measured_turns == 2
    enlarged = task_usage(
        task, (trace,), (), top_k=2, maximum_steps=1_000, maximum_query_tokens=32_768
    )
    assert enlarged == usage
    shorter = task_usage(task, (trace,), (), top_k=2, maximum_steps=1, maximum_query_tokens=32_768)
    assert shorter.assistant_input == 400


def test_identical_messages_without_source_identity_are_distinct_requests() -> None:
    """Identical retry or repeated turns are not deduplicated merely because their text matches."""
    trace = _trace(1, _completion_reservation("candidate").model)
    span = trace.spans[0]
    trace = trace.model_copy(
        update={
            "spans": (
                span,
                span.model_copy(update={"span_id": "another-request"}),
            )
        }
    )
    assert len(model_turns(trace)) == 2


def test_vendor_tool_arguments_count_without_full_output_messages() -> None:
    """Grouped normalized tool output contributes to assistant, world and judge estimates."""
    trace = _trace(1, _completion_reservation("candidate").model)
    first = trace.spans[0].model_copy(
        update={
            "usage": None,
            "attributes": {
                "exp.source.span.id": "request-1",
                "gen_ai.tool.call.id": "search",
                "gen_ai.tool.name": "search",
                "gen_ai.tool.call.arguments": {"query": "x" * 8_000},
            },
        }
    )
    second = first.model_copy(
        update={
            "span_id": "other-call",
            "attributes": {
                **first.attributes,
                "gen_ai.tool.call.id": "extract",
                "gen_ai.tool.name": "extract",
                "gen_ai.tool.call.arguments": '{"url":"https://example.com"}',
            },
        }
    )
    trace = trace.model_copy(update={"spans": (first, second)})
    task = TaskCase(
        task_id="case",
        lineage_group_id="lineage",
        partition="fit",
        instruction=trace.task,
        source_trace_ids=(trace.trace_id,),
        workload_weight=1,
    )
    usage = task_usage(task, (trace,), (), top_k=2, maximum_steps=100, maximum_query_tokens=32_768)
    assert usage.assistant_output == token_estimate(
        {
            "content": None,
            "tool_calls": [
                {"name": "search", "arguments": {"query": "x" * 8_000}},
                {"name": "extract", "arguments": {"url": "https://example.com"}},
            ],
        }
    )
    assert usage.assistant_output > 2_000
    assert usage.world_input > usage.assistant_output
    assert usage.judge_input > usage.assistant_output
    assert usage.turns == 1
    assert usage.measured_turns == 0


@pytest.mark.parametrize("call_ids", [True, False])
def test_otlp_observations_use_canonical_pairing_for_arbitrary_span_names(call_ids: bool) -> None:
    """Distinct results pair once by ID or tool-name order and enter later request estimates."""
    first, second = (_tool_trace(with_result=True, index=index) for index in (1, 2))
    spans = first.spans + second.spans
    if not call_ids:
        spans = tuple(
            span.model_copy(
                update={
                    "attributes": {
                        key: value
                        for key, value in span.attributes.items()
                        if key != "gen_ai.tool.call.id"
                    }
                }
            )
            for span in spans
        )
    trace = first.model_copy(update={"spans": spans})
    task = TaskCase(
        task_id="case",
        lineage_group_id="lineage",
        partition="fit",
        instruction=trace.task,
        source_trace_ids=(trace.trace_id,),
        workload_weight=1,
    )
    baseline = task_usage(
        task, (trace,), (), top_k=2, maximum_steps=100, maximum_query_tokens=32_768
    )
    spans = tuple(
        span.model_copy(
            update={
                "attributes": {
                    **span.attributes,
                    "gen_ai.tool.message": "x" * (8_000 if index == 1 else 4_000),
                }
            }
        )
        if index in (1, 3)
        else span
        for index, span in enumerate(spans)
    )
    trace = trace.model_copy(update={"spans": spans})
    usage = task_usage(task, (trace,), (), top_k=2, maximum_steps=100, maximum_query_tokens=32_768)
    first_increase = 2_000 - token_estimate("account found")
    total_increase = 3_000 - 2 * token_estimate("account found")
    assert usage.turns == 2
    assert usage.world_output - baseline.world_output == total_increase
    assert usage.judge_input - baseline.judge_input == total_increase
    assert usage.assistant_input - baseline.assistant_input == first_increase

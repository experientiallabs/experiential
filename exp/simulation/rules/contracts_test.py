"""Tests for tool contracts and trajectory views parsed from canonical traces."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from exp.common.core.artifacts import FailureCode, JsonObject, SourceIdentity, StructuredFailure
from exp.common.tasks import ToolSchema
from exp.common.traces import Trace, TraceSpan
from exp.common.traces.trace import TraceOutcome, TraceSource
from exp.simulation.rules.contracts import parse_tool_contracts, result_shape, trajectory_view

_START = datetime(2026, 10, 1, tzinfo=UTC)
_LOOKUP = ToolSchema(
    name="lookup",
    description="Find an order.",
    input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
)


def _span(
    span_id: str,
    second: int,
    attributes: JsonObject,
    failure: StructuredFailure | None = None,
) -> TraceSpan:
    """One span starting at a fixed second offset and lasting under a second."""
    started = _START + timedelta(seconds=second)
    return TraceSpan(
        span_id=span_id,
        name="span",
        started_at=started,
        ended_at=started + timedelta(milliseconds=500),
        attributes=attributes,
        failure=failure,
    )


def call(span_id: str, second: int, name: str, arguments: JsonObject, call_id: str) -> TraceSpan:
    """One tool-call span.

    Args:
        span_id: Span identity.
        second: Start offset in seconds.
        name: Tool name.
        arguments: Call arguments.
        call_id: Call identity shared with its result.

    Returns:
        The span.
    """
    return _span(
        span_id,
        second,
        {
            "gen_ai.tool.name": name,
            "gen_ai.tool.call.arguments": arguments,
            "gen_ai.tool.call.id": call_id,
        },
    )


def result(
    span_id: str,
    second: int,
    name: str,
    message: JsonObject | str | None,
    call_id: str,
    *,
    failure: StructuredFailure | None = None,
    is_error: bool = False,
) -> TraceSpan:
    """One tool-result span.

    Args:
        span_id: Span identity.
        second: Start offset in seconds.
        name: Tool name.
        message: Result content, or None for a result that recorded none.
        call_id: Identity of the call it answers.
        failure: Structured failure, when the call failed.
        is_error: Whether the result is a tool error.

    Returns:
        The span.
    """
    attributes: JsonObject = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": name,
        "gen_ai.tool.call.id": call_id,
        "gen_ai.tool.is_error": is_error,
    }
    if message is not None:
        attributes["gen_ai.tool.message"] = message
    return _span(span_id, second, attributes, failure)


def trace(trace_id: str, spans: tuple[TraceSpan, ...], *, declare: bool = True) -> Trace:
    """One canonical trace with the lookup tool declared.

    Args:
        trace_id: Trace identity.
        spans: Ordered spans.
        declare: Whether the trace declares the lookup tool.

    Returns:
        The trace.
    """
    return Trace(
        trace_id=trace_id,
        task="Refund order 7",
        tools=(_LOOKUP,) if declare else (),
        spans=spans,
        outcome=TraceOutcome(status="success"),
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="fixture.otlp", sha256="a" * 64),
            semantic_convention_version="1.37.0",
        ),
    )


def refund_trace(trace_id: str = "t1") -> Trace:
    """A lookup, a refused refund and a successful refund.

    Args:
        trace_id: Trace identity.

    Returns:
        The trace.
    """
    denied = StructuredFailure(code=FailureCode.INTERNAL, message="order   not paid")
    return trace(
        trace_id,
        (
            call("c1", 0, "lookup", {"order_id": "7"}, "k1"),
            result("r1", 1, "lookup", {"status": "paid", "amount": 40}, "k1"),
            call("c2", 2, "refund", {"order_id": "7", "amount": 400}, "k2"),
            result("r2", 3, "refund", None, "k2", failure=denied),
            call("c3", 4, "refund", {"order_id": "7"}, "k3"),
            result("r3", 5, "refund", '{"ok": true}', "k3"),
        ),
    )


def test_contracts_collect_types_requirements_shapes_and_failures() -> None:
    """Each tool's calls fold into one deterministic contract."""
    contracts = parse_tool_contracts((refund_trace(),))

    assert list(contracts) == ["lookup", "refund"]
    lookup, refund = contracts["lookup"], contracts["refund"]
    assert lookup.declared_schema == _LOOKUP.input_schema
    assert lookup.argument_types == {"order_id": ("string",)}
    assert lookup.result_shapes == ("object[amount,status]",)
    assert lookup.examples == ("trace:t1:span:c1",)
    assert refund.declared_schema is None
    assert refund.argument_types == {"amount": ("integer",), "order_id": ("string",)}
    assert refund.required_arguments == ("order_id",)
    assert refund.result_shapes == ("object[ok]",)
    assert refund.error_forms == ("internal: order not paid",)
    assert (refund.calls, refund.failures) == (2, 1)
    assert refund.examples == ("trace:t1:span:c2", "trace:t1:span:c3")


def test_contracts_merge_traces_cap_examples_and_keep_declared_only_tools() -> None:
    """Calls across traces merge, examples stop at three, and declared tools always appear."""
    traces = tuple(refund_trace(f"t{n}") for n in range(3))
    idle = trace("idle", (_span("s", 0, {}),))
    contracts = parse_tool_contracts((*traces, idle))

    assert contracts["refund"].calls == 6
    assert len(contracts["refund"].examples) == 3
    assert contracts["lookup"].calls == 3
    assert parse_tool_contracts((idle,))["lookup"].calls == 0


def test_tool_errors_count_as_failures() -> None:
    """A result flagged as a tool error is a failure form, not a result shape."""
    flagged = trace(
        "t",
        (
            call("c", 0, "lookup", {"order_id": "9"}, "k"),
            result("r", 1, "lookup", "no such order", "k", is_error=True),
        ),
    )
    contract = parse_tool_contracts((flagged,))["lookup"]

    assert contract.error_forms == ("tool_error: no such order",)
    assert contract.result_shapes == ()
    assert contract.failures == 1


def test_repeated_traces_are_refused() -> None:
    """Contract counts would double count a repeated trace."""
    with pytest.raises(ValueError, match="repeats"):
        parse_tool_contracts((refund_trace(), refund_trace()))


def test_trajectory_view_orders_steps_and_lists_locators() -> None:
    """Steps pair each call with its observation or failure, and every locator is citable."""
    view, locators = trajectory_view(refund_trace())

    steps = view["steps"]
    assert isinstance(steps, list)
    assert [step["step"] for step in steps if isinstance(step, dict)] == [0, 1, 2]
    assert steps[0] == {
        "step": 0,
        "locator": "trace:t1:span:c1",
        "action": {"name": "lookup", "arguments": {"order_id": "7"}},
        "observation": {"status": "paid", "amount": 40},
        "observation_locator": "trace:t1:span:r1",
        "failure": None,
    }
    assert isinstance(steps[1], dict) and steps[1]["failure"] == "internal: order not paid"
    assert steps[1]["observation_locator"] == "trace:t1:span:r2"
    assert isinstance(steps[2], dict) and steps[2]["observation"] == {"ok": True}
    assert view["outcome"] == "success"
    assert locators == {
        "trace:t1",
        *(f"trace:t1:span:{kind}{n}" for kind in "cr" for n in (1, 2, 3)),
    }


def test_unanswered_calls_appear_without_an_observation() -> None:
    """A call with no recorded result is still a step, with nothing observed."""
    view, locators = trajectory_view(
        trace("t", (_span("c", 0, {"gen_ai.tool.name": "lookup"}),), declare=False)
    )

    assert view["steps"] == [
        {
            "step": 0,
            "locator": "trace:t:span:c",
            "action": {"name": "lookup", "arguments": {}},
            "observation": None,
            "observation_locator": None,
            "failure": None,
        }
    ]
    assert locators == {"trace:t", "trace:t:span:c"}


def test_result_shapes_name_types_keys_and_elements() -> None:
    """Shapes are stable strings over types and sorted keys."""
    assert result_shape({"b": 1, "a": 2}) == "object[a,b]"
    assert result_shape([{"x": 1}, 2, {"x": 3}]) == "array<integer|object[x]>"
    assert result_shape("text") == "string"
    assert result_shape(None) == "null"

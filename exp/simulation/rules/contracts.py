"""Tool contracts and compact trajectory views parsed from canonical traces, with no model call.

A tool contract records what traces show a tool accepting and returning: its declared input
schema, the JSON types of the arguments it was called with, the arguments every call carried, the
distinct shapes of its results and failures, and a few locators of observed calls. A trajectory
view is one trace's ordered tool steps, each pairing an action with what the agent was shown, plus
the inventory of locators a model reading that view may cite.

Tool calls are paired with their results by the retrieval transition extractor, the single owner
of action-to-observation pairing. A call that transition extraction leaves unpaired, such as a
failed call whose result span carries no message, is matched to a result span by its explicit call
identifier when one exists, and is otherwise shown without an observation.

Locators have two forms: ``trace:<trace_id>`` for a whole trajectory and
``trace:<trace_id>:span:<span_id>`` for one span.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, stable_id
from exp.common.tasks import ToolSchema
from exp.common.traces import Trace, TraceSpan
from exp.common.traces.attributes import is_tool_result, tool_arguments, tool_name, tool_result
from exp.simulation.retrieval.contracts import RAGLineageBinding
from exp.simulation.retrieval.transitions import extract_real_transitions
from exp.simulation.rules.model import ToolContract


class ToolSchemaCoverage(ContractModel):
    """Declared-schema coverage only, without semantic, state or task-start validation.

    Attributes:
        status: Whether every requested tool has the exact saved input declaration.
        issues: Missing, absent or unequal declarations by tool name.
    """

    status: Literal["schema_matches", "schema_unknown", "schema_mismatch", "missing_tools"]
    issues: dict[str, str]


def tool_schema_coverage(
    tools: Sequence[ToolSchema],
    contracts: Mapping[str, ToolContract],
) -> ToolSchemaCoverage:
    """Compare exact declarations without treating names or schemas as runnable validity."""
    issues: dict[str, str] = {}
    for tool in tools:
        contract = contracts.get(tool.name)
        if contract is None:
            issues[tool.name] = "missing tool"
        elif contract.declared_schema is None:
            issues[tool.name] = "no saved declaration"
        elif contract.declared_schema != tool.input_schema:
            issues[tool.name] = "declaration differs"
    status: Literal["schema_matches", "schema_unknown", "schema_mismatch", "missing_tools"]
    if "missing tool" in issues.values():
        status = "missing_tools"
    elif "declaration differs" in issues.values():
        status = "schema_mismatch"
    else:
        status = "schema_unknown" if issues else "schema_matches"
    return ToolSchemaCoverage(status=status, issues=issues)


MAXIMUM_EXAMPLES = 3
"""Most example locators a contract keeps per tool."""

_FORM_CHARACTERS = 160
"""Longest rendered error form."""

_IS_ERROR = "gen_ai.tool.is_error"
_CALL_ID = "gen_ai.tool.call.id"


def trace_locator(trace_id: str) -> str:
    """Return the locator of a whole trajectory.

    Args:
        trace_id: Trace identity.

    Returns:
        ``trace:<trace_id>``.
    """
    return f"trace:{trace_id}"


def span_locator(trace_id: str, span_id: str) -> str:
    """Return the locator of one span of a trajectory.

    Args:
        trace_id: Trace identity.
        span_id: Span identity within that trace.

    Returns:
        ``trace:<trace_id>:span:<span_id>``.
    """
    return f"trace:{trace_id}:span:{span_id}"


@dataclass(frozen=True)
class _Step:
    """One tool call of a trajectory with its paired result.

    Attributes:
        index: Position among the trajectory's tool calls.
        call: The span that requested the call.
        result: The span that reported its result, when one was found.
        name: Tool name.
        arguments: Decoded call arguments.
        observation: Decoded result the agent was shown, when recorded.
        failure: Short failure form, when the call failed.
    """

    index: int
    call: TraceSpan
    result: TraceSpan | None
    name: str
    arguments: JsonValue
    observation: JsonValue
    failure: str | None


def parse_tool_contracts(traces: Sequence[Trace]) -> dict[str, ToolContract]:
    """Collect one contract per tool from declared tools and observed calls.

    Args:
        traces: Canonical traces, in the order their declarations take precedence.

    Returns:
        Contracts keyed by tool name, in name order. A tool declared but never called has a
        contract with zero calls; the first trace declaring a tool supplies its schema.

    Raises:
        ValueError: Trace identities repeat, or a trace's tool results cannot be paired
            unambiguously with their calls.
    """
    ids = [trace.trace_id for trace in traces]
    if len(set(ids)) != len(ids):
        raise ValueError("tool contracts need distinct traces; a trace ID repeats")
    schemas: dict[str, JsonObject] = {}
    observed: dict[str, list[tuple[Trace, _Step]]] = {}
    for trace in traces:
        for declared in trace.tools:
            schemas.setdefault(declared.name, declared.input_schema)
        for step in _tool_steps(trace):
            observed.setdefault(step.name, []).append((trace, step))
    return {
        name: _contract(name, schemas.get(name), observed.get(name, []))
        for name in sorted(set(schemas) | set(observed))
    }


def trajectory_view(trace: Trace) -> tuple[JsonObject, frozenset[str]]:
    """Render one trajectory's ordered tool steps and the locators a reader may cite.

    Args:
        trace: One canonical trace.

    Returns:
        The view, with the trace identity, task, terminal outcome and ordered steps, and the
        locator inventory: the trajectory locator plus each step's call and result locators.

    Raises:
        ValueError: The trace's tool results cannot be paired unambiguously with their calls.
    """
    locators = {trace_locator(trace.trace_id)}
    steps: list[JsonValue] = []
    for step in _tool_steps(trace):
        locator = span_locator(trace.trace_id, step.call.span_id)
        result = None if step.result is None else span_locator(trace.trace_id, step.result.span_id)
        locators.add(locator)
        if result is not None:
            locators.add(result)
        steps.append(
            {
                "step": step.index,
                "locator": locator,
                "action": {"name": step.name, "arguments": step.arguments},
                "observation": step.observation,
                "observation_locator": result,
                "failure": step.failure,
            }
        )
    view: JsonObject = {
        "trace_id": trace.trace_id,
        "locator": trace_locator(trace.trace_id),
        "task": trace.task,
        "outcome": None if trace.outcome is None else trace.outcome.status,
        "steps": steps,
    }
    return view, frozenset(locators)


def json_type(value: JsonValue) -> str:
    """Return the JSON type name of a value.

    Args:
        value: Any JSON value.

    Returns:
        One of ``object``, ``array``, ``string``, ``integer``, ``number``, ``boolean``, ``null``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def result_shape(value: JsonValue) -> str:
    """Describe a result by its type and sorted top-level keys.

    Args:
        value: Decoded tool result.

    Returns:
        ``object[a,b]`` for an object, ``array<...>`` naming the distinct element shapes for an
        array, otherwise the JSON type name.
    """
    if isinstance(value, dict):
        return "object[" + ",".join(sorted(value)) + "]"
    if isinstance(value, list):
        return "array<" + "|".join(sorted({result_shape(item) for item in value})) + ">"
    return json_type(value)


def _contract(
    name: str, schema: JsonObject | None, calls: list[tuple[Trace, _Step]]
) -> ToolContract:
    """Fold one tool's observed calls into its contract."""
    types: dict[str, set[str]] = {}
    required: set[str] | None = None
    shapes: set[str] = set()
    errors: set[str] = set()
    failures = 0
    for _, step in calls:
        arguments = step.arguments if isinstance(step.arguments, dict) else {}
        for key, value in arguments.items():
            types.setdefault(key, set()).add(json_type(value))
        required = set(arguments) if required is None else required & set(arguments)
        if step.failure is not None:
            failures += 1
            errors.add(step.failure)
        elif step.result is not None:
            shapes.add(result_shape(step.observation))
    return ToolContract(
        name=name,
        declared_schema=schema,
        argument_types={key: tuple(sorted(found)) for key, found in sorted(types.items())},
        required_arguments=tuple(sorted(required or ())),
        result_shapes=tuple(sorted(shapes)),
        error_forms=tuple(sorted(errors)),
        calls=len(calls),
        failures=failures,
        examples=tuple(
            span_locator(trace.trace_id, step.call.span_id)
            for trace, step in calls[:MAXIMUM_EXAMPLES]
        ),
    )


def _tool_steps(trace: Trace) -> list[_Step]:
    """Return one trace's tool calls in span order, each with its paired result."""
    binding = RAGLineageBinding(
        trace_id=trace.trace_id,
        lineage_id=stable_id("rule-lineage", trace.trace_id),
        partition="fit",
    )
    paired = {
        transition.action_span_id: transition.observation_span_id
        for transition in extract_real_transitions(
            (trace,), (binding,), included_partitions=frozenset({"fit"})
        )
        if transition.action.kind == "tool_call"
    }
    spans = sorted(trace.spans, key=lambda item: (item.started_at, item.span_id))
    by_id = {span.span_id: span for span in spans}
    results_by_call = {
        span.attributes[_CALL_ID]: span
        for span in spans
        if is_tool_result(span.attributes) and isinstance(span.attributes.get(_CALL_ID), str)
    }
    steps: list[_Step] = []
    for span in spans:
        name = tool_name(span.attributes)
        if name is None or is_tool_result(span.attributes):
            continue
        result = by_id.get(paired[span.span_id]) if span.span_id in paired else None
        call_id = span.attributes.get(_CALL_ID)
        if result is None and isinstance(call_id, str):
            result = results_by_call.get(call_id)
        observation = None if result is None else tool_result(result.attributes)
        arguments = tool_arguments(span.attributes)
        steps.append(
            _Step(
                index=len(steps),
                call=span,
                result=result,
                name=name,
                arguments={} if arguments is None else arguments,
                observation=observation,
                failure=_failure(span, result, observation),
            )
        )
    return steps


def _failure(call: TraceSpan, result: TraceSpan | None, observation: JsonValue) -> str | None:
    """Return a short failure form when the call or its result reports one."""
    for span in (call, result):
        if span is not None and span.failure is not None:
            return _short(f"{span.failure.code.value}: {span.failure.message}")
    if result is not None and result.attributes.get(_IS_ERROR) is True:
        rendered = (
            observation
            if isinstance(observation, str)
            else json.dumps(observation, sort_keys=True, ensure_ascii=False)
        )
        return _short(f"tool_error: {rendered}")
    return None


def _short(text: str) -> str:
    """Collapse whitespace and bound one error form."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _FORM_CHARACTERS:
        return collapsed
    return collapsed[: _FORM_CHARACTERS - 3] + "..."

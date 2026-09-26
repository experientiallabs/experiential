"""Trace-sized evaluation planning, separate from hard execution reservations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from statistics import mean

from exp.common.core.artifacts import JsonValue, canonical_json_bytes
from exp.common.models import CompletionCostReservation
from exp.common.tasks import TaskCase
from exp.common.traces import Trace, TraceSpan
from exp.simulation.engines.text.prompt import WORLD_MODEL_TEXT_SYSTEM_PROMPT
from exp.simulation.retrieval.contracts import RAGLineageBinding, RAGTransition
from exp.simulation.retrieval.transitions import extract_real_transitions


@dataclass(frozen=True)
class ExpectedUsage:
    """Expected tokens for one captured episode, without multiplying execution ceilings.

    Attributes:
        assistant_input: Input tokens summed once per captured model request.
        assistant_output: Generated tokens summed once per captured model request.
        world_input: Estimated framed world-model input across captured turns.
        world_output: Estimated observations and JSON framing across captured turns.
        query_input: Estimated retrieval query tokens across actions.
        judge_input: Estimated final transcript and rubric input.
        turns: Distinct captured assistant requests.
        measured_turns: Requests with recorded provider token usage.
    """

    assistant_input: float
    assistant_output: float
    world_input: float
    world_output: float
    query_input: float
    judge_input: float
    turns: float
    measured_turns: float


def token_estimate(value: JsonValue) -> int:
    """Approximate visible text at four UTF-8 bytes per token, never for admission."""
    if value is None or value == "":
        return 0
    text = value if isinstance(value, str) else canonical_json_bytes(value).decode("utf-8")
    return math.ceil(len(text.encode("utf-8")) / 4)


def _decoded(value: JsonValue) -> JsonValue:
    """Decode message attributes that importers may retain as JSON strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def model_turns(trace: Trace) -> tuple[tuple[TraceSpan, ...], ...]:
    """Group normalized parallel-tool spans by their explicit original request identity.

    Spans without a source request identity remain distinct, even if their text matches.
    Tool-result spans are observations, never additional assistant calls.
    """
    groups: dict[str, list[TraceSpan]] = {}
    for span in trace.spans:
        if span.name == "agent.tool_call" or span.attributes.get("gen_ai.operation.name") in {
            "execute_tool",
            "tool_call",
            "tool",
        }:
            continue
        if span.model is None and not any(
            key in span.attributes
            for key in (
                "gen_ai.output.messages",
                "gen_ai.completion",
                "gen_ai.tool.name",
                "gen_ai.tool.call.arguments",
            )
        ):
            continue
        source_id = span.attributes.get("exp.source.span.id")
        key = str(source_id) if source_id is not None else span.span_id
        groups.setdefault(key, []).append(span)
    return tuple(tuple(group) for group in groups.values())


def task_usage(
    task: TaskCase,
    traces: tuple[Trace, ...],
    transitions: tuple[RAGTransition, ...],
    *,
    top_k: int,
    maximum_steps: int,
    maximum_query_tokens: int,
) -> ExpectedUsage:
    """Average this scenario's source episodes using visible messages and real observations.

    Args:
        task: Selected scenario, with exact source trace IDs and declared tools.
        traces: Its immutable source traces, not all serialized dataset bytes.
        transitions: Fit-only examples eligible for this scenario's grounding.
        top_k: Configured retrieved examples per action.
        maximum_steps: Execution ceiling, used only to clip longer observed episodes.
        maximum_query_tokens: Existing per-query embedding admission ceiling.

    Returns:
        Trace-derived usage. Missing token counts use a visible-text heuristic. World state,
        future reasoning, changed behavior and retries remain uncertain planning inputs.
    """
    evidence = [
        token_estimate(
            {
                "transition_id": item.transition_id,
                "task": item.task,
                "initial_context": item.initial_context,
                "action": item.action.model_dump(mode="json"),
                "observation": item.observation.model_dump(mode="json"),
            }
        )
        for item in transitions
        if item.lineage_id != task.lineage_group_id
    ]
    example_size = mean(evidence) if evidence else 0.0
    task_size = token_estimate(
        {
            "task": task.instruction,
            "initial_context": task.initial_context,
            "tools": [tool.model_dump(mode="json") for tool in task.tools],
        }
    )
    observations = {
        (item.trace_id, item.action_span_id): item.observation.content
        for item in extract_real_transitions(
            traces,
            tuple(
                RAGLineageBinding(
                    trace_id=trace.trace_id,
                    lineage_id=task.lineage_group_id,
                    partition=task.partition,
                )
                for trace in traces
            ),
            included_partitions=frozenset({task.partition}),
        )
    }
    episodes = []
    for trace in traces:
        groups = model_turns(trace)[:maximum_steps]
        if not groups:
            continue
        inputs = outputs = world_inputs = world_outputs = queries = 0.0
        measured = 0
        transcript = 0.0
        for group in groups:
            first = group[0]
            attrs = first.attributes
            messages = _decoded(attrs.get("gen_ai.input.messages"))
            input_tokens = token_estimate(messages) if messages else task_size + transcript
            input_tokens += token_estimate([tool.model_dump(mode="json") for tool in task.tools])
            output_tokens = _output_tokens(group)
            usage = next((span.usage for span in group if span.usage is not None), None)
            if usage is not None:
                measured += 1
                input_tokens, output_tokens = usage.input_tokens, usage.output_tokens
            calls = {
                span.attributes.get("gen_ai.tool.call.id") or span.span_id
                for span in group
                if span.attributes.get("gen_ai.tool.name") is not None
                or span.attributes.get("gen_ai.tool.call.id") is not None
            }
            observation_tokens = sum(
                token_estimate(observations.get((trace.trace_id, span.span_id))) for span in group
            )
            # One world response per assistant turn, including all parallel tool results.
            query_count = len(calls) or int(output_tokens > 0)
            queries += query_count * min(
                maximum_query_tokens, task_size + output_tokens / max(1, query_count)
            )
            world_inputs += (
                input_tokens
                + output_tokens
                + task_size
                + token_estimate(WORLD_MODEL_TEXT_SYSTEM_PROMPT)
                + example_size * min(len(evidence), top_k if query_count else 0)
            )
            world_outputs += observation_tokens + 64 + 16 * len(calls)
            inputs += input_tokens
            outputs += output_tokens
            transcript += output_tokens + observation_tokens
        episodes.append(
            ExpectedUsage(
                inputs,
                outputs,
                world_inputs,
                world_outputs,
                queries,
                task_size + transcript + 1024,
                len(groups),
                measured,
            )
        )
    if not episodes:
        raise ValueError(f"scenario {task.task_id} has no captured model requests to estimate")
    return ExpectedUsage(
        **{
            field: mean(getattr(item, field) for item in episodes)
            for field in ExpectedUsage.__dataclass_fields__
        }
    )


def expected_completion_cost(
    request: CompletionCostReservation, input_tokens: float, output_tokens: float
) -> float:
    """Price expected usage at ordinary catalog rates, without assuming retries or cache hits."""
    return (
        input_tokens * request.input_usd_per_million_tokens
        + output_tokens * request.output_usd_per_million_tokens
    ) / 1_000_000


def _output_tokens(group: tuple[TraceSpan, ...]) -> int:
    """Count one reply, reconstructing normalized tool calls when full messages are absent."""
    attributes = group[0].attributes
    messages = _decoded(attributes.get("gen_ai.output.messages"))
    if messages:
        return token_estimate(messages)
    calls: dict[str, JsonValue] = {}
    for span in group:
        name = span.attributes.get("gen_ai.tool.name")
        arguments = span.attributes.get("gen_ai.tool.call.arguments")
        if name is not None or arguments is not None:
            identity = str(span.attributes.get("gen_ai.tool.call.id") or span.span_id)
            calls[identity] = {"name": name, "arguments": _decoded(arguments)}
    completion = attributes.get("gen_ai.completion")
    if not calls:
        return token_estimate(completion)
    return token_estimate({"content": completion, "tool_calls": list(calls.values())})

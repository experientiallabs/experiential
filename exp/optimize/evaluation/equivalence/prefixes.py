"""Admit complete retained actor inputs without reconstructing an unavailable initial world."""

import json
from datetime import datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    stable_id,
    strict_equal,
)
from exp.common.evaluations.reconstruction import EvaluationReferences, WorkflowEvidence
from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ToolCall
from exp.common.project import ArtifactStore, artifact_input
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.analysis.actor_decisions import ActorDecisionCase


class ActorPrefixSpec(ContractModel):
    """Source selectors for one next-decision probe, with no caller-authored message copy.

    Attributes:
        dataset: Immutable canonical trace dataset from build.
        trace_id: Exact canonical trace identity within that dataset.
        model_span_id: Span whose recorded input ends at the selected completed tool turn.
        observation_call_id: Call ID of the observation to replace, including parallel turns.
        evaluation: Saved workflow evaluation intent containing evidence and references.
        step: Position in that evaluation; both action and recorded reply must match the prefix.
        output_tokens: Explicit new probe allowance, not a reconstruction of historical settings.
    """

    dataset: ArtifactInput
    trace_id: str = Field(min_length=1)
    model_span_id: str = Field(min_length=1)
    observation_call_id: str = Field(min_length=1)
    evaluation: ArtifactInput
    step: int = Field(ge=0, strict=True)
    output_tokens: int = Field(default=4096, ge=1, le=32768, strict=True)


class _Function(ContractModel):
    """Supported recorded function payload, without silently discarding extensions.

    Attributes:
        name: Recorded tool name.
        arguments: Exact JSON argument string.
    """

    name: str
    arguments: str


class _Call(ContractModel):
    """Recorded OpenAI-style call retained by canonical trace ingestion.

    Attributes:
        id: Pairing identity.
        type: Only function tools are representable here.
        function: Name and complete argument string.
    """

    id: str
    type: Literal["function"]
    function: _Function


class _Message(ContractModel):
    """Strict text-only source subset; unrepresented media or extensions fail closed.

    Attributes:
        role: Recorded message authority.
        content: Exact text; structured/media content requires a different admission path.
        tool_calls: Structured assistant calls.
        tool_call_id: Result pairing identity.
        name: Optional tool-result name, checked against its paired call before dropping it.
    """

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[_Call, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


def _messages(raw: object) -> tuple[ModelMessage, ...]:
    """Preserve every supported message; reject rather than flatten unsupported inputs."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "recorded model input messages unavailable; retain a complete public input"
        )
    output: list[ModelMessage] = []
    names: dict[str, str] = {}
    for value in raw:
        message = _Message.model_validate(value)
        if message.tool_call_id is not None and message.role != "tool":
            raise ValueError("recorded result linkage requires a tool message; repair the export")
        if message.name is not None and (
            message.role != "tool" or names.get(message.tool_call_id or "") != message.name
        ):
            raise ValueError(
                "recorded message name is not a paired tool name; repair source export"
            )
        if message.tool_calls:
            if message.role != "assistant":
                raise ValueError("recorded tool calls require an assistant message")
            calls = tuple(
                ToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=json.loads(call.function.arguments),
                )
                for call in message.tool_calls
            )
            names.update({call.call_id: call.name for call in calls})
            output.append(
                ModelMessage(
                    role=message.role,
                    assistant_action=AssistantAction(content=message.content, tool_calls=calls),
                )
            )
        else:
            output.append(
                ModelMessage(
                    role=message.role, content=message.content, tool_call_id=message.tool_call_id
                )
            )
    return tuple(output)


def admit_actor_prefix(store: ArtifactStore, spec: ActorPrefixSpec) -> ActorDecisionCase:
    """Bind the retained full prefix to a matching saved action and reference observation.

    This establishes reproducible extraction from the export, not that the exporter retained
    all original context. No private state, future response or judge fields enter the request.
    """
    for ref in (spec.dataset, spec.evaluation):
        if artifact_input(store.read(ref.artifact_id).manifest) != ref:
            raise ValueError("actor prefix source digest changed; select the exact saved artifact")
    traces = load_trace_dataset(store, spec.dataset.artifact_id).traces
    trace = next((row for row in traces if row.trace_id == spec.trace_id), None)
    if trace is None:
        raise ValueError("actor prefix trace absent from dataset; select a retained trace ID")
    span = next((row for row in trace.spans if row.span_id == spec.model_span_id), None)
    if span is None:
        raise ValueError("actor prefix span absent; select the recorded next-decision model span")
    messages = _messages(trace.input_messages(span))
    indexes = [
        i
        for i, row in enumerate(messages)
        if row.role == "tool" and row.tool_call_id == spec.observation_call_id
    ]
    if len(indexes) != 1:
        raise ValueError("actor prefix needs one selected reply; use its unambiguous call ID")
    actions = [
        call
        for message in messages
        if message.assistant_action is not None
        for call in message.assistant_action.tool_calls
        if call.call_id == spec.observation_call_id
    ]
    if len(actions) != 1:
        raise ValueError("actor prefix call pairing is ambiguous; repair the source export")
    if (
        store.read(spec.evaluation.artifact_id).manifest.artifact_type
        != "workflow-evaluation-intent"
    ):
        raise ValueError("actor prefix needs a saved workflow evaluation intent")
    evidence = WorkflowEvidence.model_validate_json(
        store.read_bytes(spec.evaluation.artifact_id, "evidence.json")
    )
    references = EvaluationReferences.model_validate_json(
        store.read_bytes(spec.evaluation.artifact_id, "references.json")
    )
    reference = [row for row in references.steps if row.step == spec.step]
    if spec.step >= len(evidence.steps) or len(reference) != 1:
        raise ValueError(
            "actor prefix step lacks a unique saved reference; choose an observed step"
        )
    candidate, recorded = evidence.steps[spec.step], reference[0]
    action = {"name": actions[0].name, "arguments": actions[0].arguments}
    if (
        evidence.task != trace.task
        or not strict_equal(candidate.action, action)
        or not strict_equal(recorded.action, action)
        or not recorded.reply_available
        or not strict_equal(recorded.reply, messages[indexes[0]].content)
        or candidate.status != "valid"
        or not isinstance(candidate.reply, str)
    ):
        raise ValueError(
            "actor prefix task/action/reply differs from saved evaluation; check selectors"
        )
    # Canonical datasets can merge tool catalogs. Require the selected turn's actual declaration
    # rather than silently borrowing a schema from another turn.
    raw_tools = span.attributes.get("exp.request.tools")
    if not isinstance(raw_tools, list):
        raise ValueError("turn tool declarations unavailable; retain the recorded tool interface")
    declared = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in trace.tools
    ]
    if not strict_equal(raw_tools, declared):
        raise ValueError(
            "turn tools differ from canonical catalog; dynamic interfaces need admission"
        )
    return ActorDecisionCase(
        name=f"{spec.trace_id}:{spec.model_span_id}:{spec.step}",
        request=ModelRequest(
            messages=messages,
            tools=trace.tools,
            tool_choice="auto",
            maximum_output_tokens=spec.output_tokens,
        ),
        observation=indexes[0],
        simulated_reply=candidate.reply,
        sources=tuple(sorted((spec.dataset, spec.evaluation), key=lambda ref: ref.artifact_id)),
        admission=(
            "Complete retained model input, paired by call ID and checked against saved task, "
            "action and recorded reply. No original-context completeness claim. Export metadata: "
            + json.dumps(span.attributes.get("exp.trace.metadata", {}), sort_keys=True)
            + ". Probe sampling settings are new; historical actor settings are not reconstructed."
        ),
    )


def persist_actor_prefix(
    store: ArtifactStore, spec: ActorPrefixSpec, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Recompute on replay and save an immutable case usable by the existing probe command."""
    case = admit_actor_prefix(store, spec)
    identity = stable_id(
        "actor-prefix", {"spec": spec.model_dump(mode="json"), "code": code_revision}
    )
    if store.exists(identity):
        if (
            ActorDecisionCase.model_validate_json(store.read_bytes(identity, "case.json")) != case
            or store.read(identity).manifest.inputs != case.sources
        ):
            raise ValueError("saved actor prefix differs; inspect its source evidence")
        return artifact_input(store.read(identity).manifest)
    return artifact_input(
        store.write_json(
            artifact_id=identity,
            artifact_type="actor-prefix",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=created_at,
                code_revision=code_revision,
                inputs=case.sources,
            ),
            files={"spec.json": spec, "case.json": case},
        )
    )

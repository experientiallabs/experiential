"""Normalize LangSmith run exports into canonical trace evidence.

LangSmith models an agent execution as runs: each run declares ``id``, ``trace_id``,
``parent_run_id``, ``run_type``, timing, inputs, outputs, and any error. Exports arrive as a run
array, a ``runs`` envelope, or JSONL with one run per line, and runs of one trace may be spread
across records.

Run types map to canonical evidence by what they observe:

- ``llm`` and ``chat_model`` become model calls, including the tool calls their output requests,
- ``tool`` becomes the tool result paired with the earlier requesting call,
- ``chain``, ``retriever``, ``embedding``, ``prompt``, and ``parser`` observe orchestration rather
  than a visible agent step and are not converted.

Model identity is retained only when LangSmith declares a provider in run metadata, normally
``extra.metadata.ls_provider`` next to ``ls_model_name``. A model name declared without a provider
is retained as ``gen_ai.request.model`` evidence instead of being resolved into model identity.
"""

from __future__ import annotations

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.simulation.ingest.vendor_observations import (
    VendorModelIdentity,
    VendorObservation,
    VendorTokenUsage,
    declared_completion_text,
    declared_error_message,
    declared_model_identity,
    declared_tool_calls,
    declared_usage,
)
from exp.simulation.ingest.vendor_records import (
    first_text,
    first_user_text,
    json_text,
    json_value,
    required_text,
    source_interval,
)
from exp.simulation.ingest.vendor_source import VendorSource, record_flattener
from exp.simulation.ingest.vendor_trace import approved_extensions

VENDOR = "langsmith"

_MODEL_RUN_TYPES = frozenset({"llm", "chat_model"})
_TOOL_RUN_TYPES = frozenset({"tool"})
_MODEL_KEYS = ("ls_model_name", "model_name", "model", "model_id")
_PROVIDER_KEYS = ("ls_provider", "provider")
_ERROR_KEYS = ("error",)


def _run_observation(run: JsonObject, ordinal: int) -> tuple[VendorObservation, ...]:
    """Convert one LangSmith run to a declared model or tool-result observation.

    Args:
        run: LangSmith run object.
        ordinal: Source order position for the emitted observation.

    Returns:
        Declared observation, or nothing for orchestration-only run types.

    Raises:
        VendorTraceFormatError: The run lacks identity, timing, or tool evidence.
    """
    run_type = (first_text(run, ("run_type", "runType")) or "").casefold()
    if run_type not in _MODEL_RUN_TYPES | _TOOL_RUN_TYPES:
        return ()
    source_span_id = required_text(run.get("id"), "LangSmith run id")
    source_trace_id = first_text(run, ("trace_id", "traceId")) or source_span_id
    started_at, ended_at = source_interval(
        run.get("start_time", run.get("startTime")),
        run.get("end_time", run.get("endTime")),
        start_label="LangSmith run start_time",
        end_label="LangSmith run end_time",
    )
    parent = first_text(run, ("parent_run_id", "parentRunId"))
    failure = declared_error_message(run, keys=_ERROR_KEYS, label="LangSmith run")
    inputs = json_value(run.get("inputs"))
    outputs = json_value(run.get("outputs"))
    extensions = _extensions(run)
    if run_type in _TOOL_RUN_TYPES:
        return (
            VendorObservation(
                source_trace_id=source_trace_id,
                source_span_id=source_span_id,
                ordinal=ordinal,
                started_at=started_at,
                ended_at=ended_at,
                kind="tool_result",
                source_parent_span_id=parent,
                request_text=first_user_text(inputs),
                tool_name=required_text(run.get("name"), "LangSmith tool run name"),
                tool_arguments=json_text(inputs),
                tool_message=declared_completion_text(outputs),
                failure_message=failure,
                extensions=extensions,
            ),
        )
    model, declared_model = _model_identity(run)
    return (
        VendorObservation(
            source_trace_id=source_trace_id,
            source_span_id=source_span_id,
            ordinal=ordinal,
            started_at=started_at,
            ended_at=ended_at,
            kind="model",
            source_parent_span_id=parent,
            request_text=first_user_text(inputs),
            input_messages=_input_messages(inputs),
            completion_text=declared_completion_text(outputs) or None,
            tool_calls=declared_tool_calls(outputs),
            model=model,
            usage=_usage(run, outputs),
            failure_message=failure,
            declared_attributes=(
                {} if declared_model is None else {"gen_ai.request.model": declared_model}
            ),
            extensions=extensions,
        ),
    )


def _input_messages(inputs: JsonValue | None) -> JsonValue | None:
    """Return the declared input messages for one model run.

    Args:
        inputs: Decoded run inputs.

    Returns:
        The declared message list, or the raw inputs when the run declares no list.
    """
    if isinstance(inputs, dict):
        for key in ("messages", "input", "prompts"):
            value = inputs.get(key)
            if isinstance(value, list):
                return value
    return inputs


def _usage(run: JsonObject, outputs: JsonValue | None) -> VendorTokenUsage | None:
    """Read declared token accounting from run metadata or model output.

    Args:
        run: LangSmith run object.
        outputs: Decoded run outputs.

    Returns:
        Declared usage, or ``None`` when the run declares no complete accounting.

    Raises:
        VendorTraceFormatError: A declared token count is not a non-negative integer.
    """
    candidates: list[JsonValue] = []
    for key in ("usage_metadata", "usage"):
        if key in run:
            candidates.append(run[key])
    if isinstance(outputs, dict):
        for key in ("usage_metadata", "llm_output", "usage"):
            value = outputs.get(key)
            if isinstance(value, dict):
                candidates.append(value)
                token_usage = value.get("token_usage")
                if token_usage is not None:
                    candidates.append(token_usage)
    for candidate in candidates:
        usage = declared_usage(candidate)
        if usage is not None:
            return usage
    return None


def _model_identity(run: JsonObject) -> tuple[VendorModelIdentity | None, str | None]:
    """Read declared model identity from LangSmith run metadata and invocation parameters.

    Args:
        run: LangSmith run object.

    Returns:
        Retained model identity when a provider is declared, and the declared model name.
    """
    merged: JsonObject = {}
    extra = run.get("extra")
    if isinstance(extra, dict):
        for key in ("invocation_params", "metadata"):
            nested = extra.get(key)
            if isinstance(nested, dict):
                merged.update(nested)
    for key in (*_MODEL_KEYS, *_PROVIDER_KEYS):
        if key in run:
            merged[key] = run[key]
    return declared_model_identity(
        merged,
        model_keys=_MODEL_KEYS,
        provider_keys=_PROVIDER_KEYS,
        revision_keys=("ls_model_revision", "model_revision", "revision"),
    )


def _extensions(run: JsonObject) -> JsonObject:
    """Read approved EXP extensions, session identity, and metadata from one run.

    Args:
        run: LangSmith run object.

    Returns:
        Approved extension attributes for the spans of this run.
    """
    extensions = approved_extensions(run)
    extra = run.get("extra")
    metadata = extra.get("metadata") if isinstance(extra, dict) else None
    if isinstance(metadata, dict):
        extensions.update(approved_extensions(metadata))
        if "exp.trace.metadata" not in extensions:
            extensions["exp.trace.metadata"] = metadata
    session_id = first_text(run, ("session_id", "sessionId", "thread_id"))
    if session_id is not None and "exp.conversation.id" not in extensions:
        extensions["exp.conversation.id"] = session_id
    return extensions


LANGSMITH_SOURCE: VendorSource[JsonObject] = VendorSource(
    vendor=VENDOR,
    records=record_flattener(
        vendor=VENDOR,
        wrapper_keys=("runs", "data", "results", "items"),
        record_keys=("run_type", "runType"),
    ),
    convert=_run_observation,
)

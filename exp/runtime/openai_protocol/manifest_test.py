"""Tests for the executable closed OpenAI compatibility manifests."""

from __future__ import annotations

import typing

import pytest

from exp.common.models.model import ReasoningEffort
from exp.runtime.gateway.compatibility import (
    CompatibilityDisposition,
    CompatibilityManifest,
)
from exp.runtime.openai_protocol.manifest import (
    CHAT_MANIFEST,
    RESPONSES_INCLUDE_PATHS_ACCEPTED,
    RESPONSES_INCLUDE_PATHS_REJECTED,
    RESPONSES_INPUT_ITEM_FIELDS_ACCEPTED,
    RESPONSES_INPUT_ITEM_FIELDS_REJECTED,
    RESPONSES_MANIFEST,
    RESPONSES_REASONING_CONTEXTS_ACCEPTED,
    RESPONSES_REASONING_EFFORTS_ACCEPTED,
    RESPONSES_REASONING_FIELDS_ACCEPTED,
    RESPONSES_REASONING_FIELDS_REJECTED,
    RESPONSES_REASONING_SUMMARIES_ACCEPTED,
    disposition_map,
)


@pytest.mark.parametrize(
    ("manifest", "field"),
    tuple(
        (manifest, item.field_path)
        for manifest in (CHAT_MANIFEST, RESPONSES_MANIFEST)
        for item in manifest.fields
        if item.disposition != CompatibilityDisposition.UNSUPPORTED
    ),
)
def test_every_accepted_field_has_one_executable_manifest_decision(
    manifest: CompatibilityManifest, field: str
) -> None:
    """Every accepted field is explicit and unique rather than SDK-version widened."""
    assert field in disposition_map(manifest)


def test_manifests_classify_explicit_exclusions() -> None:
    """Unsupported fields stay excluded and lossy controls are rejected."""
    chat = disposition_map(CHAT_MANIFEST)
    responses = disposition_map(RESPONSES_MANIFEST)
    assert chat["audio"] == CompatibilityDisposition.UNSUPPORTED
    # Value-constrained acceptances: the wire models admit only the no-op
    # values Copilot hardcodes (n:1, truncation:"disabled",
    # prompt_cache_options:{"mode":"implicit"}).
    assert chat["n"] == CompatibilityDisposition.SUPPORTED
    assert responses["truncation"] == CompatibilityDisposition.SUPPORTED
    assert responses["prompt_cache_options"] == CompatibilityDisposition.SUPPORTED
    assert chat["logprobs"] == CompatibilityDisposition.CONDITIONALLY_SUPPORTED
    assert chat["top_logprobs"] == CompatibilityDisposition.UNSUPPORTED
    assert chat["top_k"] == CompatibilityDisposition.CONDITIONALLY_SUPPORTED
    assert chat["top_p"] == CompatibilityDisposition.SUPPORTED
    assert chat["service_tier"] == CompatibilityDisposition.CONDITIONALLY_SUPPORTED
    assert responses["service_tier"] == CompatibilityDisposition.CONDITIONALLY_SUPPORTED
    assert responses["background"] == CompatibilityDisposition.UNSUPPORTED
    assert responses["conversation"] == CompatibilityDisposition.UNSUPPORTED
    assert responses["include"] == CompatibilityDisposition.CONDITIONALLY_SUPPORTED
    assert responses["store"] == CompatibilityDisposition.SUPPORTED
    assert responses["top_p"] == CompatibilityDisposition.SUPPORTED
    assert responses["top_k"] == CompatibilityDisposition.CONDITIONALLY_SUPPORTED
    assert responses["top_logprobs"] == CompatibilityDisposition.UNSUPPORTED


def _sdk_literal_values(annotation: object) -> frozenset[str]:
    """Collect every string literal reachable inside one SDK type annotation."""
    values: set[str] = set()

    def walk(node: object) -> None:
        """Accumulate string literals depth-first across unions and wrappers."""
        origin = typing.get_origin(node)
        if origin is typing.Literal:
            values.update(arg for arg in typing.get_args(node) if isinstance(arg, str))
            return
        for argument in typing.get_args(node):
            walk(argument)

    walk(annotation)
    return frozenset(values)


def _decided(new_fields: frozenset[str] | set[str], surface: str) -> str:
    """Render the drift-gate failure text naming exactly what must be decided."""
    listed = ", ".join(sorted(new_fields))
    return (
        f"The installed openai SDK carries {surface} fields this gateway has never "
        f"classified: {listed}. Decide each one now: add it to the manifest (or the "
        "reasoning/include decision tables in exp/runtime/openai_protocol/manifest.py) as "
        "SUPPORTED, CONDITIONALLY_SUPPORTED, or UNSUPPORTED with a written rationale. "
        "Silent drift here is how paying customers become the first detector."
    )


def test_every_official_sdk_request_field_is_consciously_classified() -> None:
    """The SDK-surface drift gate: an ``openai`` bump may add request fields.

    Every top-level field the official request types carry must appear in the
    matching manifest, in exactly one disposition bucket, so a dependency bump
    turns new OpenAI surface into a loud decision instead of a silent
    customer-facing 400.
    """
    from openai.types.chat.completion_create_params import CompletionCreateParamsBase
    from openai.types.responses.response_create_params import ResponseCreateParamsBase

    responses_fields = set(ResponseCreateParamsBase.__annotations__) | {"stream"}
    chat_fields = set(CompletionCreateParamsBase.__annotations__) | {"stream"}
    responses_decided = set(disposition_map(RESPONSES_MANIFEST))
    chat_decided = set(disposition_map(CHAT_MANIFEST))

    undecided_responses = responses_fields - responses_decided
    assert not undecided_responses, _decided(undecided_responses, "Responses")
    undecided_chat = chat_fields - chat_decided
    assert not undecided_chat, _decided(undecided_chat, "Chat Completions")


def test_every_official_sdk_reasoning_field_and_value_is_decided() -> None:
    """Nested ``reasoning`` fields and enum values are part of the drift gate."""
    from openai.types.shared_params.reasoning import Reasoning

    hints = typing.get_type_hints(Reasoning)
    undecided = (
        set(hints) - RESPONSES_REASONING_FIELDS_ACCEPTED - RESPONSES_REASONING_FIELDS_REJECTED
    )
    assert not undecided, _decided(undecided, "reasoning.*")
    # Accepted and rejected buckets must stay disjoint decisions.
    assert not RESPONSES_REASONING_FIELDS_ACCEPTED & RESPONSES_REASONING_FIELDS_REJECTED

    new_efforts = _sdk_literal_values(hints["effort"]) - RESPONSES_REASONING_EFFORTS_ACCEPTED
    assert not new_efforts, _decided(new_efforts, "reasoning.effort value")
    new_contexts = _sdk_literal_values(hints["context"]) - RESPONSES_REASONING_CONTEXTS_ACCEPTED
    assert not new_contexts, _decided(new_contexts, "reasoning.context value")
    for field in ("summary", "generate_summary"):
        new_summaries = _sdk_literal_values(hints[field]) - RESPONSES_REASONING_SUMMARIES_ACCEPTED
        assert not new_summaries, _decided(new_summaries, f"reasoning.{field} value")

    # The engine ladder must cover every SDK effort so decode never narrows it.
    engine_efforts = set(typing.get_args(ReasoningEffort))
    assert _sdk_literal_values(hints["effort"]) <= engine_efforts


def test_every_official_sdk_include_selector_is_decided() -> None:
    """``include[]`` members are decided: honored or rejected by name."""
    from openai.types.responses.response_includable import ResponseIncludable

    sdk_selectors = _sdk_literal_values(ResponseIncludable)
    undecided = sdk_selectors - RESPONSES_INCLUDE_PATHS_ACCEPTED - RESPONSES_INCLUDE_PATHS_REJECTED
    assert not undecided, _decided(undecided, "include selector")
    assert not RESPONSES_INCLUDE_PATHS_ACCEPTED & RESPONSES_INCLUDE_PATHS_REJECTED


def test_every_official_sdk_echoable_input_item_field_is_decided() -> None:
    """Echoed output items are part of the drift gate.

    A stateless caller resends prior output items verbatim as input, so a
    new SDK field on any echoable item type must be a conscious decision:
    accepted-and-dropped or rejected by name, never a silent 400.
    """
    from openai.types.responses.response_function_tool_call_param import (
        ResponseFunctionToolCallParam,
    )
    from openai.types.responses.response_input_param import FunctionCallOutput
    from openai.types.responses.response_output_message_param import ResponseOutputMessageParam
    from openai.types.responses.response_reasoning_item_param import ResponseReasoningItemParam

    sdk_items = {
        "message": ResponseOutputMessageParam,
        "function_call": ResponseFunctionToolCallParam,
        "function_call_output": FunctionCallOutput,
        "reasoning": ResponseReasoningItemParam,
    }
    for item_type, sdk_type in sdk_items.items():
        accepted = RESPONSES_INPUT_ITEM_FIELDS_ACCEPTED[item_type]
        rejected = RESPONSES_INPUT_ITEM_FIELDS_REJECTED[item_type]
        assert not accepted & rejected, item_type
        undecided = set(sdk_type.__annotations__) - accepted - rejected
        assert not undecided, _decided(undecided, f"echoed {item_type} input item")

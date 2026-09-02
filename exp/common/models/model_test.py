"""Tests for canonical model data contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import sha256_json
from exp.common.models import (
    AssistantAction,
    BillingSource,
    Embedding,
    ModelCapabilities,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    RawEmbedding,
    RawEmbeddingBatch,
    RoutedCandidateSnapshot,
    ToolCall,
    ToolChoice,
    Usage,
    combine_economics,
)
from exp.common.tasks import ToolSchema

_CAPABILITIES_DIGEST = "a" * 64


def test_actions_need_payload_and_measurements_are_finite() -> None:
    """Invalid empty actions and non-finite economics fail at the shared boundary."""
    with pytest.raises(ValidationError, match="content or at least one tool"):
        AssistantAction()
    with pytest.raises(ValidationError, match="finite"):
        NumericMeasurement(value=float("inf"), provenance="observed")


def test_current_model_snapshot_requires_explicit_billing_source() -> None:
    """A newly frozen model identity cannot silently infer the credential owner."""
    with pytest.raises(ValidationError, match="billing_source"):
        ModelSnapshot(  # ty: ignore[missing-argument]
            provider="openai",
            model_id="fixture",
            capabilities_sha256=_CAPABILITIES_DIGEST,
            connection_sha256="b" * 64,
        )


def test_completed_factory_prefers_served_identity_and_maps_the_length_limit() -> None:
    """The shared factory keeps served identity, observed latency, and the finish reason."""
    configured = ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="configured-model",
        capabilities_sha256=_CAPABILITIES_DIGEST,
        connection_sha256="b" * 64,
    )
    action = AssistantAction(content="Done.")

    served = ModelResponse.completed(
        output=action,
        configured_model=configured,
        served_model_id="served-model",
        usage=None,
        latency_seconds=0.25,
        hit_length_limit=True,
    )
    kept = ModelResponse.completed(
        output=action,
        configured_model=configured,
        served_model_id="",
        usage=None,
        latency_seconds=0.25,
    )

    assert served.model.model_id == "served-model"
    assert served.finish_reason is ModelFinishReason.LENGTH
    assert kept.model.model_id == "configured-model"
    assert kept.finish_reason is ModelFinishReason.COMPLETED
    assert kept.economics.latency_seconds == NumericMeasurement(value=0.25, provenance="observed")


def test_model_request_keeps_tool_contract_and_capabilities_deterministic() -> None:
    """A request retains typed tool behavior and rejects incoherent choices.

    The regression also verifies deterministic capability identity hashing.
    """
    tool = ToolSchema(
        name="create_ticket",
        description="Create one support ticket.",
        input_schema={"type": "object"},
    )
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="Please help."),),
        tools=(tool,),
        tool_choice=ToolChoice(name="create_ticket"),
        maximum_output_tokens=512,
    )

    assert request.tools == (tool,)
    assert ModelCapabilities(supports_tools=True).model_dump(mode="json") == {
        "supports_tools": True,
        "supports_embeddings": None,
        "supports_structured_output": False,
        "supports_completions": None,
        "supports_temperature": True,
        "supports_top_p": None,
        "supports_top_k": None,
        "supports_logprobs": None,
        "supports_reasoning": False,
        "reasoning_effort": None,
        "sampling_requires_reasoning_none": False,
        "chat_max_tokens_field": None,
        "minimum_temperature": None,
        "maximum_temperature": None,
        "minimum_top_p": None,
        "maximum_top_p": None,
        "minimum_top_k": None,
        "maximum_top_k": None,
        "context_window_tokens": None,
        "maximum_output_tokens": None,
        "input_cost_per_million_tokens_usd": None,
        "output_cost_per_million_tokens_usd": None,
        "cached_input_cost_per_million_tokens_usd": None,
        "cache_write_cost_per_million_tokens_usd": None,
    }
    with pytest.raises(ValidationError, match="named tool_choice"):
        ModelRequest(
            messages=(ModelMessage(role="user", content="Please help."),),
            tool_choice=ToolChoice(name="missing"),
        )


def test_completion_support_preserves_provider_identity_for_existing_traces() -> None:
    """Completion eligibility is frozen separately without orphaning old trace snapshots."""
    identity_payload = {
        "supports_tools": None,
        "supports_embeddings": None,
        "context_window_tokens": None,
        "maximum_output_tokens": None,
    }
    unknown = ModelCapabilities()
    supported = ModelCapabilities(supports_completions=True)
    unsupported = ModelCapabilities(supports_completions=False)

    assert unknown.identity_sha256() == sha256_json(identity_payload)
    assert supported.identity_sha256() == unsupported.identity_sha256()
    assert supported.identity_sha256() == unknown.identity_sha256()
    pinned_sampling = ModelCapabilities(
        supports_temperature=False,
        supports_reasoning=True,
        reasoning_effort="xhigh",
    )
    assert pinned_sampling.identity_sha256() == unknown.identity_sha256()
    reasoning_route = ModelCapabilities(supports_reasoning=True)
    assert reasoning_route.identity_sha256() == unknown.identity_sha256()
    with pytest.raises(ValidationError, match="required tool_choice"):
        ModelRequest(
            messages=(ModelMessage(role="user", content="Please help."),),
            tool_choice="required",
        )


def test_reasoning_effort_requires_explicit_reasoning_support() -> None:
    """A pinned effort cannot create reasoning support by implication."""
    with pytest.raises(ValidationError, match="reasoning_effort requires reasoning support"):
        ModelCapabilities(reasoning_effort="high")


def test_routed_candidate_payload_serializes_explicit_billing_source() -> None:
    """Automatic capability freezing retains the model billing source byte for byte."""
    payload = {
        "alias": "candidate-a",
        "model": {
            "provider": "openai",
            "model_id": "model-a",
            "revision": None,
            "billing_source": "host_managed",
            "capabilities_sha256": "a" * 64,
            "connection_sha256": "b" * 64,
        },
    }

    candidate = RoutedCandidateSnapshot.model_validate(payload)

    assert candidate.model_dump(mode="json") == payload
    assert isinstance(candidate.model, ModelSnapshot)


def test_model_messages_reject_tool_and_assistant_fields_on_the_wrong_roles() -> None:
    """Request-visible messages retain tool linkage without role-crossing payloads."""
    with pytest.raises(ValidationError, match="assistant_action is valid only"):
        ModelMessage(role="user", assistant_action=AssistantAction(content="wrong"))
    with pytest.raises(ValidationError, match="tool messages require tool_call_id"):
        ModelMessage(role="tool", content="missing linkage")


def test_tool_call_preserves_optional_raw_arguments_without_changing_legacy_payloads() -> None:
    """Raw provider JSON replays exactly but stays outside semantic artifact payloads."""
    legacy = ToolCall(call_id="call-1", name="lookup", arguments={"a": 1, "b": 2})
    retained = ToolCall(
        call_id="call-1",
        name="lookup",
        arguments={"a": 1, "b": 2},
        raw_arguments='{ "b": 2, "a": 1 }',
    )

    assert legacy.model_dump(mode="json") == {
        "call_id": "call-1",
        "name": "lookup",
        "arguments": {"a": 1, "b": 2},
    }
    assert legacy.arguments_json(sort_keys=True, compact=True) == '{"a":1,"b":2}'
    assert retained.arguments_json() == '{ "b": 2, "a": 1 }'
    assert retained.arguments_json(sort_keys=True, compact=True) == '{"a":1,"b":2}'
    assert retained.model_dump(mode="json") == legacy.model_dump(mode="json")
    with pytest.raises(ValidationError, match="must match parsed"):
        ToolCall(
            call_id="call-1",
            name="lookup",
            arguments={"a": 1},
            raw_arguments='{"a":2}',
        )


def test_combine_economics_reports_partial_totals_only_when_asked() -> None:
    """Aggregation never presents a partial usage, cost, or provenance as complete."""
    priced = OperationEconomics(
        usage=Usage(input_tokens=10, output_tokens=2, cached_input_tokens=4),
        cost_usd=NumericMeasurement(value=0.5, provenance="observed"),
        latency_seconds=NumericMeasurement(value=1.0, provenance="observed"),
    )
    estimated = OperationEconomics(
        usage=Usage(input_tokens=5, output_tokens=1),
        cost_usd=NumericMeasurement(value=0.25, provenance="estimated"),
    )
    unmetered = OperationEconomics()

    complete = combine_economics((priced, estimated))
    partial = combine_economics((priced, unmetered), require_complete_usage=False)
    strict = combine_economics((priced, unmetered))

    assert combine_economics(()) == OperationEconomics()
    assert complete.usage == Usage(input_tokens=15, output_tokens=3, cached_input_tokens=None)
    assert complete.cost_usd == NumericMeasurement(value=0.75, provenance="estimated")
    assert complete.latency_seconds is None
    assert partial.usage == priced.usage
    assert partial.cost_usd is None
    assert strict.usage is None
    assert combine_economics((unmetered,), require_complete_usage=False).usage is None


def test_embeddings_require_nonempty_finite_vectors() -> None:
    """Embedding output cannot hide empty or non-finite provider data."""
    assert Embedding(values=(1.0, 0.0)).values == (1.0, 0.0)
    with pytest.raises(ValidationError, match="at least 1 item"):
        Embedding(values=())
    with pytest.raises(ValidationError, match="finite"):
        Embedding(values=(float("nan"),))
    with pytest.raises(ValidationError, match="unit norm"):
        Embedding(values=(0.1, -0.2))


def test_raw_embeddings_preserve_magnitude_but_still_reject_bad_vectors() -> None:
    """The public-surface carrier keeps raw magnitude yet enforces finiteness."""
    # A non-unit vector that Embedding would reject is valid raw output.
    assert RawEmbedding(values=(0.1, -0.2)).values == (0.1, -0.2)
    with pytest.raises(ValidationError, match="at least 1 item"):
        RawEmbedding(values=())
    with pytest.raises(ValidationError, match="finite"):
        RawEmbedding(values=(float("inf"),))
    batch = RawEmbeddingBatch(
        embeddings=(RawEmbedding(values=(0.1, -0.2)),),
        prompt_tokens=5,
        served_model_id="text-embedding-3-small",
    )
    assert batch.prompt_tokens == 5
    with pytest.raises(ValidationError, match="at least 1 item"):
        RawEmbeddingBatch(embeddings=(), prompt_tokens=0)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        RawEmbeddingBatch(embeddings=(RawEmbedding(values=(1.0,)),), prompt_tokens=-1)


def test_combine_economics_sums_present_usage_and_complete_measurements() -> None:
    """Usage sums when present. Cost and latency stay unknown unless every call reports them."""
    observed = NumericMeasurement(value=0.4, provenance="observed")
    estimated = NumericMeasurement(value=0.1, provenance="estimated")
    combined = combine_economics(
        (
            OperationEconomics(
                usage=Usage(input_tokens=3, output_tokens=1, cached_input_tokens=1),
                cost_usd=observed,
                latency_seconds=observed,
            ),
            OperationEconomics(
                usage=Usage(input_tokens=2, output_tokens=4),
                cost_usd=estimated,
            ),
        )
    )

    assert combined.usage == Usage(input_tokens=5, output_tokens=5, cached_input_tokens=None)
    assert combined.cost_usd == NumericMeasurement(value=0.5, provenance="estimated")
    assert combined.latency_seconds is None
    assert combine_economics(()) == OperationEconomics()


def test_combine_economics_preserves_cache_write_when_every_record_reports_it() -> None:
    """Aggregated usage keeps cache-write counts only when every record carries them."""
    first = OperationEconomics(
        usage=Usage(
            input_tokens=100,
            output_tokens=10,
            cached_input_tokens=20,
            cache_write_input_tokens=30,
        ),
        cost_usd=NumericMeasurement(value=0.2, provenance="observed"),
        latency_seconds=NumericMeasurement(value=0.4, provenance="observed"),
    )
    second = OperationEconomics(
        usage=Usage(
            input_tokens=50,
            output_tokens=5,
            cached_input_tokens=10,
            cache_write_input_tokens=15,
        ),
        cost_usd=NumericMeasurement(value=0.1, provenance="observed"),
        latency_seconds=NumericMeasurement(value=0.2, provenance="observed"),
    )
    partial = OperationEconomics(
        usage=Usage(input_tokens=20, output_tokens=2, cached_input_tokens=5),
        cost_usd=NumericMeasurement(value=0.05, provenance="observed"),
    )

    complete = combine_economics((first, second))
    mixed = combine_economics((first, partial))
    zero_write = combine_economics(
        (
            OperationEconomics(
                usage=Usage(
                    input_tokens=10,
                    output_tokens=1,
                    cached_input_tokens=2,
                    cache_write_input_tokens=0,
                ),
            ),
            OperationEconomics(
                usage=Usage(
                    input_tokens=10,
                    output_tokens=1,
                    cached_input_tokens=3,
                    cache_write_input_tokens=0,
                ),
            ),
        )
    )

    assert complete.usage == Usage(
        input_tokens=150,
        output_tokens=15,
        cached_input_tokens=30,
        cache_write_input_tokens=45,
    )
    assert mixed.usage == Usage(
        input_tokens=120,
        output_tokens=12,
        cached_input_tokens=25,
        cache_write_input_tokens=None,
    )
    assert zero_write.usage == Usage(
        input_tokens=20,
        output_tokens=2,
        cached_input_tokens=5,
        cache_write_input_tokens=0,
    )
    assert zero_write.usage is not None
    assert zero_write.usage.cache_write_input_tokens == 0


def test_combine_economics_cache_write_partial_mode_keeps_only_complete_subset() -> None:
    """Partial aggregation keeps cache-write totals only from the present subset."""
    priced = OperationEconomics(
        usage=Usage(
            input_tokens=10,
            output_tokens=2,
            cached_input_tokens=4,
            cache_write_input_tokens=2,
        ),
        cost_usd=NumericMeasurement(value=0.5, provenance="observed"),
    )
    unmetered = OperationEconomics()

    strict = combine_economics((priced, unmetered))
    relaxed = combine_economics((priced, unmetered), require_complete_usage=False)
    empty_relaxed = combine_economics((unmetered,), require_complete_usage=False)

    assert strict.usage is None
    assert relaxed.usage == priced.usage
    assert relaxed.usage is not None
    assert relaxed.usage.cache_write_input_tokens == 2
    assert empty_relaxed.usage is None


def test_unknown_support_flags_change_the_frozen_capability_identity_digest() -> None:
    """Unknown tool and embedding support hash as their own tri-state value.

    Runtime dispatch treats unknown support permissively and an explicit denial as a hard
    block, so a drift between them must invalidate a frozen capability identity.
    """
    assert (
        ModelCapabilities().identity_sha256()
        != ModelCapabilities(supports_tools=False, supports_embeddings=False).identity_sha256()
    )
    assert (
        ModelCapabilities(supports_tools=True).identity_sha256()
        != ModelCapabilities().identity_sha256()
    )


def test_generation_parameter_contract_rejects_inverted_ranges() -> None:
    """Catalog metadata cannot publish an empty numeric parameter interval."""
    with pytest.raises(ValidationError, match="minimum_temperature"):
        ModelCapabilities(minimum_temperature=1.0, maximum_temperature=0.5)
    with pytest.raises(ValidationError, match="minimum_top_k"):
        ModelCapabilities(minimum_top_k=10, maximum_top_k=5)
    with pytest.raises(ValidationError, match="conditional sampling requires reasoning support"):
        ModelCapabilities(sampling_requires_reasoning_none=True)

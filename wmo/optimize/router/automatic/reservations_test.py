"""Tests for automatic router provider reservations."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wmo.common.core.artifacts import (
    ArtifactInput,
    SourceIdentity,
    canonical_json_bytes,
    sha256_json,
)
from wmo.common.models import (
    BillingSource,
    ConnectionConfig,
    EmbeddingCostReservation,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelSnapshot,
    completion_cost_reservation,
)
from wmo.common.traces import Trace, TraceSource, TraceSpan
from wmo.optimize.router.automatic.reservations import (
    completion_reservation_from_catalog,
    median_trace_token_estimate,
    simulation_input_token_estimate,
)
from wmo.runtime.models import RuntimeModelCatalog
from wmo.simulation.engines.text.grounding import episode_reservation_failure
from wmo.simulation.specs import (
    CandidateCompletionReservation,
    SimulationCompletionContract,
    WorldModelSettings,
)

_LARGE_CONTEXT_TOKENS = 1_050_000
_OUTPUT_TOKENS = 16_000
_QUERY_TOKENS = 32_768


def _trace(trace_id: str, task: str) -> Trace:
    """Build one minimal normalized trace whose serialized size tracks its task text.

    Args:
        trace_id: Unique trace identifier.
        task: Task text controlling the canonical serialized length.

    Returns:
        One single-span normalized trace.
    """
    moment = datetime(2026, 1, 1, tzinfo=UTC)
    return Trace(
        trace_id=trace_id,
        task=task,
        spans=(
            TraceSpan(
                span_id=f"{trace_id}-span",
                name="turn",
                started_at=moment,
                ended_at=moment,
            ),
        ),
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="fixture"),
            semantic_convention_version="1",
        ),
    )


def _catalog() -> ModelCatalog:
    """Return one large-context world model, one candidate, and one embedder."""
    world = ModelCapabilities(
        supports_completions=True,
        context_window_tokens=_LARGE_CONTEXT_TOKENS,
        maximum_output_tokens=128_000,
        input_cost_per_million_tokens_usd=1.25,
        output_cost_per_million_tokens_usd=6.0,
        cached_input_cost_per_million_tokens_usd=0.125,
        cache_write_cost_per_million_tokens_usd=1.25,
    )
    candidate = ModelCapabilities(
        supports_completions=True,
        context_window_tokens=400_000,
        maximum_output_tokens=128_000,
        input_cost_per_million_tokens_usd=0.25,
        output_cost_per_million_tokens_usd=2.0,
        cached_input_cost_per_million_tokens_usd=0.025,
        cache_write_cost_per_million_tokens_usd=0.25,
    )
    embedder = ModelCapabilities(
        supports_embeddings=True,
        input_cost_per_million_tokens_usd=0.02,
    )
    return ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")},
        models={
            "world": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="openai",
                model="world",
                capabilities=world,
            ),
            "candidate": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="openai",
                model="candidate",
                capabilities=candidate,
            ),
            "candidate-b": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="openai",
                model="candidate-b",
                capabilities=candidate,
            ),
            "embedder": ModelRecord(
                billing_source=BillingSource.CUSTOMER_MANAGED,
                connection="openai",
                model="embedder",
                capabilities=embedder,
            ),
        },
    )


def _snapshot(alias: str) -> ModelSnapshot:
    """Return one exact credential-free snapshot from the fixture catalog.

    Args:
        alias: Fixture catalog alias to resolve.

    Returns:
        Frozen model identity.
    """
    return RuntimeModelCatalog(_catalog(), environment={}).snapshot(alias)[0]


def test_median_trace_token_estimate_is_lower_median_of_canonical_bytes() -> None:
    """The estimate is the deterministic lower median of canonical serialized lengths."""
    traces = (
        _trace("t-long", "x" * 9_000),
        _trace("t-short", "x" * 10),
        _trace("t-mid", "x" * 500),
    )

    estimate = median_trace_token_estimate(traces)

    assert estimate == len(canonical_json_bytes(_trace("t-mid", "x" * 500)))
    assert median_trace_token_estimate(()) is None


def test_simulation_input_estimate_sums_explicit_deterministic_components() -> None:
    """The per-call input reservation adds transcript, retrieval, echo, and framing budgets."""
    traces = (_trace("t-a", "x" * 100), _trace("t-b", "x" * 100), _trace("t-c", "x" * 100))
    median = median_trace_token_estimate(traces)
    assert median is not None

    estimate = simulation_input_token_estimate(
        traces,
        retrieved_transition_count=5,
        maximum_retrieval_query_tokens=_QUERY_TOKENS,
        maximum_output_tokens=_OUTPUT_TOKENS,
    )

    assert estimate == 6 * median + _QUERY_TOKENS + _OUTPUT_TOKENS + 4_096
    assert (
        simulation_input_token_estimate(
            (),
            retrieved_transition_count=5,
            maximum_retrieval_query_tokens=_QUERY_TOKENS,
            maximum_output_tokens=_OUTPUT_TOKENS,
        )
        is None
    )
    with pytest.raises(ValueError, match="retrieved transition count"):
        simulation_input_token_estimate(
            traces,
            retrieved_transition_count=0,
            maximum_retrieval_query_tokens=_QUERY_TOKENS,
            maximum_output_tokens=_OUTPUT_TOKENS,
        )


def test_completion_reservation_prices_from_trace_estimate_and_admits_to_context() -> None:
    """The trace estimate prices the reservation while context capacity bounds admission."""
    traces = tuple(_trace(f"t-{index}", "x" * 2_000) for index in range(5))
    estimate = simulation_input_token_estimate(
        traces,
        retrieved_transition_count=5,
        maximum_retrieval_query_tokens=_QUERY_TOKENS,
        maximum_output_tokens=_OUTPUT_TOKENS,
    )
    assert estimate is not None
    problems: list[str] = []

    reservation = completion_reservation_from_catalog(
        problems,
        catalog=_catalog(),
        alias="world",
        model=_snapshot("world"),
        label="world model",
        maximum_attempts=3,
        estimated_input_tokens=estimate,
        maximum_output_tokens=_OUTPUT_TOKENS,
    )

    assert problems == []
    assert reservation is not None
    assert reservation.estimated_input_tokens == estimate
    assert reservation.maximum_input_tokens == _LARGE_CONTEXT_TOKENS - _OUTPUT_TOKENS
    assert reservation.planning_input_tokens() == estimate
    assert reservation.planning_input_tokens() < _LARGE_CONTEXT_TOKENS // 10
    assert (
        reservation.estimated_maximum_call_cost_usd
        < reservation.absolute_maximum_call_cost_usd() / 5
    )


def test_completion_reservation_rejects_estimates_above_the_context_window() -> None:
    """An input estimate that cannot fit the model context fails closed with a problem."""
    problems: list[str] = []

    reservation = completion_reservation_from_catalog(
        problems,
        catalog=_catalog(),
        alias="world",
        model=_snapshot("world"),
        label="world model",
        maximum_attempts=3,
        estimated_input_tokens=_LARGE_CONTEXT_TOKENS,
        maximum_output_tokens=_OUTPUT_TOKENS,
    )

    assert reservation is None
    assert problems == [
        f"world model alias 'world' cannot fit the estimated {_LARGE_CONTEXT_TOKENS} input plus "
        f"{_OUTPUT_TOKENS} output tokens inside its {_LARGE_CONTEXT_TOKENS}-token context window"
    ]


def _completion_contract(estimated_input_tokens: int) -> SimulationCompletionContract:
    """Build one immutable completion contract for the fixture world and candidate.

    Args:
        estimated_input_tokens: Realistic per-call input planning size for both models.

    Returns:
        Frozen candidate and world reservations under retry ceiling three.
    """
    problems: list[str] = []
    world = completion_reservation_from_catalog(
        problems,
        catalog=_catalog(),
        alias="world",
        model=_snapshot("world"),
        label="world model",
        maximum_attempts=3,
        estimated_input_tokens=estimated_input_tokens,
        maximum_output_tokens=_OUTPUT_TOKENS,
    )
    candidate = completion_reservation_from_catalog(
        problems,
        catalog=_catalog(),
        alias="candidate",
        model=_snapshot("candidate"),
        label="candidate",
        maximum_attempts=3,
        estimated_input_tokens=estimated_input_tokens,
        maximum_output_tokens=_OUTPUT_TOKENS,
    )
    candidate_b = completion_reservation_from_catalog(
        problems,
        catalog=_catalog(),
        alias="candidate-b",
        model=_snapshot("candidate-b"),
        label="candidate",
        maximum_attempts=3,
        estimated_input_tokens=estimated_input_tokens,
        maximum_output_tokens=_OUTPUT_TOKENS,
    )
    assert problems == []
    assert world is not None and candidate is not None and candidate_b is not None
    return SimulationCompletionContract(
        schema_version=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        inputs=(),
        code_revision="f" * 40,
        completion_contract_id="simulation-completion-fixture",
        candidate_requests=(
            CandidateCompletionReservation(candidate_alias="candidate", request=candidate),
            CandidateCompletionReservation(candidate_alias="candidate-b", request=candidate_b),
        ),
        world_model_alias="world",
        world_model_request=world,
        maximum_attempts=3,
    )


def _world_settings() -> WorldModelSettings:
    """Return grounded world settings with an explicit query-embedding reservation."""
    return WorldModelSettings(
        world_model_alias="world",
        grounded_world_model_input=ArtifactInput(
            artifact_id="grounded-world-model-fixture",
            sha256=sha256_json({"fixture": "grounded"}),
        ),
        prompt_version="v1",
        query_embedding=EmbeddingCostReservation(
            model=_snapshot("embedder"),
            input_usd_per_million_tokens=0.02,
            maximum_attempts=3,
            maximum_input_tokens=_QUERY_TOKENS,
        ),
        maximum_output_tokens=_OUTPUT_TOKENS,
    )


def test_episode_admission_ignores_estimates_and_gates_on_actual_spend() -> None:
    """An oversized planning estimate never rejects an episode with real spend remaining."""
    traces = tuple(_trace(f"t-{index}", "x" * 4_000) for index in range(5))
    estimate = simulation_input_token_estimate(
        traces,
        retrieved_transition_count=5,
        maximum_retrieval_query_tokens=_QUERY_TOKENS,
        maximum_output_tokens=_OUTPUT_TOKENS,
    )
    assert estimate is not None

    admitted = episode_reservation_failure(
        _world_settings(),
        completion_contract=_completion_contract(estimate),
        remaining_cost_usd=30.0,
    )

    assert admitted is None

    full_context_world = completion_cost_reservation(
        model=_snapshot("world"),
        input_usd_per_million_tokens=1.25,
        output_usd_per_million_tokens=6.0,
        cached_input_usd_per_million_tokens=0.125,
        cache_write_usd_per_million_tokens=1.25,
        maximum_attempts=3,
        maximum_input_tokens=_LARGE_CONTEXT_TOKENS - _OUTPUT_TOKENS,
        maximum_output_tokens=_OUTPUT_TOKENS,
    )
    expensive_estimate = episode_reservation_failure(
        _world_settings(),
        completion_contract=_completion_contract(estimate).model_copy(
            update={"world_model_request": full_context_world}
        ),
        remaining_cost_usd=0.01,
    )

    assert expensive_estimate is None

    exhausted = episode_reservation_failure(
        _world_settings(),
        completion_contract=_completion_contract(estimate),
        remaining_cost_usd=0.0,
    )

    assert exhausted is not None
    assert exhausted.details is not None
    assert exhausted.details["phase"] == "episode_provider_spend"

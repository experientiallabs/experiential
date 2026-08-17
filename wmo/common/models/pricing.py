"""Canonical immutable candidate-pricing snapshots for router economics."""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    canonical_json_bytes,
    stable_id,
)
from wmo.common.models.model import (
    ModelAlias,
    ModelCapabilities,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
)
from wmo.common.project import ArtifactStore, artifact_input


class CandidateTokenPrice(ContractModel):
    """USD price units per one million candidate-model tokens."""

    candidate_alias: ModelAlias
    input_usd_per_million_tokens: float = Field(ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)
    cached_input_usd_per_million_tokens: float | None = Field(default=None, ge=0)
    cache_write_usd_per_million_tokens: float | None = Field(default=None, ge=0)

    @field_validator(
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "cached_input_usd_per_million_tokens",
        "cache_write_usd_per_million_tokens",
    )
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("candidate token prices must be finite")
        return value


class EmbeddingCostReservation(ContractModel):
    """Exact model, price, and retry bound reserved for one embedding request."""

    model: ModelSnapshot
    input_usd_per_million_tokens: float = Field(ge=0)
    maximum_attempts: int = Field(gt=0)
    maximum_input_tokens: int = Field(gt=0)

    @field_validator("input_usd_per_million_tokens")
    @classmethod
    def _finite_input_price(cls, value: float) -> float:
        """Reject a non-finite embedding input price.

        Args:
            value: Nonnegative catalog price supplied for validation.

        Returns:
            The unchanged finite price.

        Raises:
            ValueError: The price is infinite or NaN.
        """
        if not math.isfinite(value):
            raise ValueError("embedding input price must be finite")
        return value


class CompletionCostReservation(ContractModel):
    """Conservative retry-bound ceiling for one completion provider request.

    ``maximum_input_tokens`` is the hard per-request admission ceiling, sized from the model's
    real context capacity. ``estimated_input_tokens`` is the realistic per-call planning size
    used only to price ``estimated_maximum_call_cost_usd``; an actual request may exceed the
    estimate as long as it fits the hard ceiling and the caller's remaining spend budget.
    An absent estimate prices the reservation from the hard ceiling itself.
    """

    model: ModelSnapshot
    input_usd_per_million_tokens: float = Field(ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)
    cached_input_usd_per_million_tokens: float = Field(ge=0)
    cache_write_usd_per_million_tokens: float = Field(ge=0)
    maximum_attempts: int = Field(gt=0)
    maximum_input_tokens: int = Field(gt=0)
    maximum_output_tokens: int = Field(gt=0)
    estimated_input_tokens: int | None = Field(default=None, gt=0)
    estimated_maximum_call_cost_usd: float = Field(ge=0)

    @field_validator(
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "cached_input_usd_per_million_tokens",
        "cache_write_usd_per_million_tokens",
        "estimated_maximum_call_cost_usd",
    )
    @classmethod
    def _require_finite_prices(cls, value: float) -> float:
        """Reject non-finite completion prices and reservations.

        Args:
            value: Nonnegative price or estimated ceiling.

        Returns:
            The unchanged finite value.

        Raises:
            ValueError: The value is infinite or NaN.
        """
        if not math.isfinite(value):
            raise ValueError("completion reservation economics must be finite")
        return value

    @staticmethod
    def _input_price_ceiling(
        input_price: float,
        cached_input_price: float,
        cache_write_price: float,
    ) -> float:
        """Return the highest mutually exclusive provider input billing rate.

        Args:
            input_price: Ordinary input price.
            cached_input_price: Cached-read input price.
            cache_write_price: Total cache-write input rate.

        Returns:
            Maximum possible price applied to one input token.
        """
        return max(input_price, cached_input_price, cache_write_price)

    def planning_input_tokens(self) -> int:
        """Return the input size used to price this reservation.

        Returns:
            The realistic per-call estimate, or the hard ceiling without one.
        """
        if self.estimated_input_tokens is not None:
            return self.estimated_input_tokens
        return self.maximum_input_tokens

    def expected_maximum_call_cost_usd(self) -> float:
        """Calculate the retry-bound planning cost for one realistic request.

        Returns:
            Conservative planning cost in USD priced from the realistic input estimate.
        """
        input_price = self._input_price_ceiling(
            self.input_usd_per_million_tokens,
            self.cached_input_usd_per_million_tokens,
            self.cache_write_usd_per_million_tokens,
        )
        return (
            self.maximum_attempts
            * (
                self.planning_input_tokens() * input_price
                + self.maximum_output_tokens * self.output_usd_per_million_tokens
            )
            / 1_000_000
        )

    def absolute_maximum_call_cost_usd(self) -> float:
        """Calculate the retry-bound cost of one request at the hard admission ceiling.

        Returns:
            Absolute maximum cost in USD for one admitted request.
        """
        input_price = self._input_price_ceiling(
            self.input_usd_per_million_tokens,
            self.cached_input_usd_per_million_tokens,
            self.cache_write_usd_per_million_tokens,
        )
        return (
            self.maximum_attempts
            * (
                self.maximum_input_tokens * input_price
                + self.maximum_output_tokens * self.output_usd_per_million_tokens
            )
            / 1_000_000
        )

    @model_validator(mode="after")
    def _require_complete_ceiling(self) -> CompletionCostReservation:
        """Require the persisted total to equal its retry-bound planning cost.

        Returns:
            The unchanged validated reservation.

        Raises:
            ValueError: The estimate exceeds the hard ceiling or the total omits a price,
                token estimate, or retry factor.
        """
        if (
            self.estimated_input_tokens is not None
            and self.estimated_input_tokens > self.maximum_input_tokens
        ):
            raise ValueError("completion input estimate exceeds its hard admission ceiling")
        if not math.isclose(
            self.estimated_maximum_call_cost_usd,
            self.expected_maximum_call_cost_usd(),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("completion maximum call cost differs from its reservation")
        return self


def completion_cost_reservation(
    *,
    model: ModelSnapshot,
    input_usd_per_million_tokens: float,
    output_usd_per_million_tokens: float,
    cached_input_usd_per_million_tokens: float,
    cache_write_usd_per_million_tokens: float,
    maximum_attempts: int,
    maximum_input_tokens: int,
    maximum_output_tokens: int,
    estimated_input_tokens: int | None = None,
) -> CompletionCostReservation:
    """Create one exact conservative completion-call reservation.

    Args:
        model: Exact provider model identity.
        input_usd_per_million_tokens: Ordinary request-input price.
        output_usd_per_million_tokens: Generated-output price.
        cached_input_usd_per_million_tokens: Cached-read input price.
        cache_write_usd_per_million_tokens: Total cache-write input rate.
        maximum_attempts: Runtime request-attempt ceiling.
        maximum_input_tokens: Hard per-request input admission ceiling.
        maximum_output_tokens: Provider output ceiling.
        estimated_input_tokens: Realistic per-call input size used only for cost planning,
            or ``None`` to plan at the hard ceiling.

    Returns:
        Validated retry-bound planning call cost.
    """
    input_price = CompletionCostReservation._input_price_ceiling(
        input_usd_per_million_tokens,
        cached_input_usd_per_million_tokens,
        cache_write_usd_per_million_tokens,
    )
    planning_input_tokens = (
        estimated_input_tokens if estimated_input_tokens is not None else maximum_input_tokens
    )
    estimated = (
        maximum_attempts
        * (
            planning_input_tokens * input_price
            + maximum_output_tokens * output_usd_per_million_tokens
        )
        / 1_000_000
    )
    return CompletionCostReservation(
        model=model,
        input_usd_per_million_tokens=input_usd_per_million_tokens,
        output_usd_per_million_tokens=output_usd_per_million_tokens,
        cached_input_usd_per_million_tokens=cached_input_usd_per_million_tokens,
        cache_write_usd_per_million_tokens=cache_write_usd_per_million_tokens,
        maximum_attempts=maximum_attempts,
        maximum_input_tokens=maximum_input_tokens,
        maximum_output_tokens=maximum_output_tokens,
        estimated_input_tokens=estimated_input_tokens,
        estimated_maximum_call_cost_usd=estimated,
    )


def completion_request_cost_usd(
    reservation: CompletionCostReservation,
    *,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Price one exact pending request under its conservative retry ceiling.

    Args:
        reservation: Frozen model, price, retry, and per-request bounds.
        input_tokens: Full serialized request token upper bound.
        output_tokens: Requested provider output ceiling.

    Returns:
        Conservative retry-inclusive pending request cost.

    Raises:
        ValueError: Actual request input or output exceeds the frozen reservation.
    """
    if input_tokens > reservation.maximum_input_tokens:
        raise ValueError("completion request exceeds its reserved input-token ceiling")
    if output_tokens > reservation.maximum_output_tokens:
        raise ValueError("completion request exceeds its reserved output-token ceiling")
    input_price = CompletionCostReservation._input_price_ceiling(
        reservation.input_usd_per_million_tokens,
        reservation.cached_input_usd_per_million_tokens,
        reservation.cache_write_usd_per_million_tokens,
    )
    return (
        reservation.maximum_attempts
        * (input_tokens * input_price + output_tokens * reservation.output_usd_per_million_tokens)
        / 1_000_000
    )


def reconcile_completion_economics(
    reservation: CompletionCostReservation,
    economics: OperationEconomics,
) -> OperationEconomics:
    """Derive a conservative retry-inclusive charge from response economics.

    A successful response exposes usage for its completed attempt but provider adapters do not
    expose whether earlier retry attempts were billed. The derived charge therefore prices the
    successful usage exactly under the frozen mutually exclusive rates and reserves, for every
    possible earlier attempt, the observed request input at the highest input rate plus the full
    reserved output budget. Earlier attempts of the same call sent the same request, so the
    observed input size bounds them without charging the context-sized admission ceiling. A
    provider cost measurement has no retry-coverage marker, so it is treated as successful-attempt
    evidence and never as proof that earlier attempts were free.

    Args:
        reservation: Frozen model prices, token bounds, and retry ceiling.
        economics: Provider response usage and optional exact cost measurement.

    Returns:
        Economics with the provider measurement for a one-attempt operation, or an estimated
        conservative retry-inclusive cost.

    Raises:
        ValueError: Usage is absent, inconsistent, outside the reservation, or measured spend
            exceeds the frozen request ceiling.
    """
    usage = economics.usage
    measured = economics.cost_usd
    if usage is None:
        raise ValueError("completion provider returned unknown usage and spend")
    if measured is not None and measured.value < 0:
        raise ValueError("completion provider spend cannot be negative")
    if (
        usage.input_tokens > reservation.maximum_input_tokens
        or usage.output_tokens > reservation.maximum_output_tokens
    ):
        raise ValueError("completion provider usage exceeds its request reservation")
    cached = usage.cached_input_tokens
    written = usage.cache_write_input_tokens
    if cached is not None and cached > usage.input_tokens:
        raise ValueError("cached input usage exceeds total input usage")
    if written is not None and written > usage.input_tokens:
        raise ValueError("cache-write usage exceeds total input usage")
    if cached is not None and written is not None:
        if cached + written > usage.input_tokens:
            raise ValueError("cache token counts must be subsets of input_tokens")
        successful_input_cost = (
            (usage.input_tokens - cached - written) * reservation.input_usd_per_million_tokens
            + cached * reservation.cached_input_usd_per_million_tokens
            + written * reservation.cache_write_usd_per_million_tokens
        )
    elif cached is None:
        successful_input_cost = usage.input_tokens * CompletionCostReservation._input_price_ceiling(
            reservation.input_usd_per_million_tokens,
            reservation.cached_input_usd_per_million_tokens,
            reservation.cache_write_usd_per_million_tokens,
        )
    else:
        uncached_price = max(
            reservation.input_usd_per_million_tokens,
            reservation.cache_write_usd_per_million_tokens,
        )
        successful_input_cost = (
            cached * reservation.cached_input_usd_per_million_tokens
            + (usage.input_tokens - cached) * uncached_price
        )
    successful_cost = (
        successful_input_cost + usage.output_tokens * reservation.output_usd_per_million_tokens
    ) / 1_000_000
    maximum_attempt_cost = (
        usage.input_tokens
        * CompletionCostReservation._input_price_ceiling(
            reservation.input_usd_per_million_tokens,
            reservation.cached_input_usd_per_million_tokens,
            reservation.cache_write_usd_per_million_tokens,
        )
        + reservation.maximum_output_tokens * reservation.output_usd_per_million_tokens
    ) / 1_000_000
    retry_inclusive_cost = (
        successful_cost + (reservation.maximum_attempts - 1) * maximum_attempt_cost
    )
    derived_cost = max(
        retry_inclusive_cost,
        measured.value if measured is not None else 0.0,
    )
    if derived_cost > reservation.absolute_maximum_call_cost_usd():
        raise ValueError("derived completion spend exceeds its request reservation")
    if (
        measured is not None
        and reservation.maximum_attempts == 1
        and measured.value >= successful_cost
    ):
        return economics
    return economics.model_copy(
        update={"cost_usd": NumericMeasurement(value=derived_cost, provenance="estimated")}
    )


def verify_completion_reservation(
    reservation: CompletionCostReservation,
    *,
    model: ModelSnapshot,
    capabilities: ModelCapabilities,
    maximum_attempts: int,
) -> None:
    """Verify one frozen reservation against the exact active runtime metadata.

    Args:
        reservation: Frozen model, price, retry, and token bounds.
        model: Active exact model identity.
        capabilities: Active explicit capability and pricing declaration.
        maximum_attempts: Active provider retry ceiling.

    Raises:
        ValueError: Model, pricing, capacity, or retry metadata drifted or is unknown.
    """
    expected_prices = (
        capabilities.input_cost_per_million_tokens_usd,
        capabilities.output_cost_per_million_tokens_usd,
        capabilities.cached_input_cost_per_million_tokens_usd,
        capabilities.cache_write_cost_per_million_tokens_usd,
    )
    if reservation.model != model:
        raise ValueError("completion reservation model differs from the active model")
    if capabilities.supports_completions is not True:
        raise ValueError("active model does not explicitly support completions")
    if None in expected_prices:
        raise ValueError("active completion pricing is incomplete")
    if (
        reservation.input_usd_per_million_tokens,
        reservation.output_usd_per_million_tokens,
        reservation.cached_input_usd_per_million_tokens,
        reservation.cache_write_usd_per_million_tokens,
    ) != expected_prices:
        raise ValueError("completion reservation pricing differs from the active catalog")
    if reservation.maximum_attempts != maximum_attempts:
        raise ValueError("completion reservation retry bound differs from the active client")
    if (
        capabilities.context_window_tokens is None
        or reservation.maximum_input_tokens + reservation.maximum_output_tokens
        > capabilities.context_window_tokens
    ):
        raise ValueError("completion reservation exceeds the active context capacity")
    if (
        capabilities.maximum_output_tokens is None
        or reservation.maximum_output_tokens > capabilities.maximum_output_tokens
    ):
        raise ValueError("completion reservation exceeds the active output capacity")


class PricingSnapshot(ArtifactEnvelope):
    """Frozen candidate aliases and token-price units used by evaluation and routing."""

    pricing_snapshot_id: ArtifactId
    candidate_prices: tuple[CandidateTokenPrice, ...]

    @field_validator("candidate_prices")
    @classmethod
    def _unique_candidates(
        cls, values: tuple[CandidateTokenPrice, ...]
    ) -> tuple[CandidateTokenPrice, ...]:
        aliases = tuple(value.candidate_alias for value in values)
        if not aliases or len(set(aliases)) != len(aliases):
            raise ValueError("pricing snapshot needs unique candidate aliases")
        return values


def load_pricing_snapshot(
    store: ArtifactStore, artifact_id: ArtifactId
) -> tuple[PricingSnapshot, str]:
    """Load a manifest-verified pricing artifact and its exact manifest digest.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Pricing-snapshot artifact identity.

    Returns:
        Parsed pricing snapshot and exact artifact-manifest digest.

    Raises:
        ValueError: The artifact type, payload, or bound identity is invalid.
    """
    stored = store.read(artifact_id)
    if stored.manifest.artifact_type != "pricing-snapshot":
        raise ValueError(f"artifact {artifact_id} is not a pricing snapshot")
    value = PricingSnapshot.model_validate_json(store.read_bytes(artifact_id, "pricing.json"))
    if value.pricing_snapshot_id != artifact_id:
        raise ValueError("pricing snapshot identity differs from its artifact")
    return value, artifact_input(stored.manifest).sha256


def persist_pricing_snapshot(
    store: ArtifactStore,
    prices: tuple[CandidateTokenPrice, ...],
    *,
    created_at: datetime,
    code_revision: str,
) -> PricingSnapshot:
    """Persist or exactly replay one candidate pricing snapshot.

    Args:
        store: Project-local immutable artifact store.
        prices: Complete explicit prices in selected candidate order.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Persisted pricing snapshot.

    Raises:
        ValueError: Existing immutable content differs from deterministic replay.
    """
    pricing_snapshot_id = stable_id(
        "pricing",
        {
            "version": "candidate-pricing-v1",
            "prices": [price.model_dump(mode="json") for price in prices],
        },
    )
    snapshot = PricingSnapshot(
        schema_version=1,
        created_at=created_at,
        inputs=(),
        code_revision=code_revision,
        pricing_snapshot_id=pricing_snapshot_id,
        candidate_prices=prices,
    )
    try:
        stored, _ = store.write_or_replay(
            artifact_id=pricing_snapshot_id,
            artifact_type="pricing-snapshot",
            envelope=snapshot,
            envelope_path="pricing.json",
            envelope_type=PricingSnapshot,
            files={"pricing.json": canonical_json_bytes(snapshot)},
        )
    except ValueError as exc:
        raise ValueError("existing pricing snapshot differs from deterministic replay") from exc
    return stored

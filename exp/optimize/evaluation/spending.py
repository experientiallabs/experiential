"""Evaluation provider wrappers enforcing an approved allowance with durable replay."""

from collections.abc import Sequence
from typing import Self

from pydantic import TypeAdapter, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, sha256_json
from exp.common.models import (
    CompletionCostReservation,
    Embedding,
    EmbeddingClient,
    EmbeddingCostReservation,
    ModelClient,
    ModelRequest,
    ModelResponse,
    completion_request_cost_usd,
    reconcile_completion_economics,
)
from exp.common.models.token_cost import schedule_prices_complete
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.errors import (
    ProviderPricingUnavailableError,
    ProviderTruncatedResponseError,
    retain_unbounded_response_liability,
)
from exp.simulation.engines.text.tokens import Utf8UpperBoundTokenCounter


class _RecordedCompletion(ContractModel):
    """Keep a paid response replayable even when its frozen tariff cannot price it.

    Attributes:
        response: Complete paid provider result, or None for truncated tool JSON.
        raw_response: Exact decoded HTTP body of a truncated response, mutually exclusive
            with response. The owning provider client supplies it before parsing rejects it.
        pricing_error: Frozen valuation error re-raised after durable save or exact replay.
        charge_usd: Known reconciled charge, or None for explicitly unbounded liability.
    """

    response: ModelResponse | None = None
    raw_response: JsonObject | None = None
    pricing_error: str | None = None
    charge_usd: float | None

    @model_validator(mode="after")
    def _require_one_response(self) -> Self:
        """Keep saved usable and truncated responses distinct, with unknown truncated cost.

        Returns:
            This envelope after validating its mutually exclusive response representations.

        Raises:
            ValueError: The envelope has no response, or combines a raw truncated body with
                a parsed response, pricing error or known charge.
        """
        if self.raw_response is not None:
            if (
                self.response is not None
                or self.pricing_error is not None
                or self.charge_usd is not None
            ):
                raise ValueError(
                    "a saved truncated response cannot contain a parsed result or cost"
                )
        elif self.response is None:
            raise ValueError("a saved completion requires its provider response")
        return self


class BudgetedCompletion:
    """Reserve each complete provider call and replay exactly saved successful responses."""

    def __init__(
        self,
        client: ModelClient,
        budget: RequestBudget,
        reservation: CompletionCostReservation,
        *,
        role: str,
        served_model_id: str | None = None,
    ) -> None:
        """Bind one role to its immutable prices and the evaluation-wide allowance."""
        self._client = client
        self._budget = budget
        self._reservation = reservation
        self._role = role
        self._served_model = (
            reservation.model.model_copy(update={"model_id": served_model_id})
            if served_model_id is not None
            else reservation.model
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Admit the exact pending request before dispatch, including all permitted retries.

        Args:
            request: Full visible conversation and explicit output allowance.

        Returns:
            The provider's successful response, or an exact saved response on resume.

        Raises:
            SpendLimitReached: The next request cannot fit the approved allowance.
            ValueError: Request identity, usage, pricing or saved evidence is invalid.
        """
        if request.maximum_output_tokens is None:
            raise ValueError("evaluation request is missing its output reservation")
        # Reconciliation retains the frozen output ceiling for any unobserved retries.
        input_tokens = Utf8UpperBoundTokenCounter().count(request)
        maximum = completion_request_cost_usd(
            self._reservation,
            input_tokens=input_tokens,
            output_tokens=self._reservation.maximum_output_tokens,
        )
        cost_is_upper_bound = self._reservation.token_prices is None or schedule_prices_complete(
            self._reservation.token_prices, maximum_input_tokens=input_tokens
        )
        identity: JsonObject = {
            "request": request.model_dump(mode="json"),
            "reservation": self._reservation.model_dump(mode="json"),
            "served_model": self._served_model.model_dump(mode="json"),
            "response_contract": "priced-completion-v1",
        }
        fingerprint = sha256_json(identity)

        def dispatch() -> _RecordedCompletion:
            """Retain the raw response; recorders independently reconcile its economics."""
            try:
                response = self._client.complete(request)
            except ProviderTruncatedResponseError as error:
                if error.response_body is None:
                    raise
                return _RecordedCompletion(raw_response=error.response_body, charge_usd=None)
            if response.model not in (self._reservation.model, self._served_model):
                raise ValueError("provider response identity differs from the request reservation")
            try:
                economics = reconcile_completion_economics(self._reservation, response.economics)
                assert economics.cost_usd is not None
                if cost_is_upper_bound and economics.cost_usd.value > maximum + 1e-9:
                    raise ValueError("provider charge exceeds the admitted request reservation")
                return _RecordedCompletion(response=response, charge_usd=economics.cost_usd.value)
            except ValueError as error:
                # This is retained liability, not a measured price. Save the paid
                # response before rejecting it, including across lower-cap replay.
                return _RecordedCompletion(
                    response=response, pricing_error=str(error), charge_usd=None
                )

        recorded = self._budget.call(
            role=self._role,
            fingerprint=fingerprint,
            maximum_cost_usd=maximum,
            operation=dispatch,
            # One response contract covers all current tariffs. Old unwrapped
            # receipts fail the fingerprint check without migration or dispatch.
            encode=lambda result: result.model_dump_json(),
            decode=_RecordedCompletion.model_validate_json,
            charge=lambda result: result.charge_usd,
            cost_is_upper_bound=cost_is_upper_bound,
        )
        if recorded.raw_response is not None:
            truncated = ProviderTruncatedResponseError(
                "saved provider response ended inside tool JSON; the paid body is retained",
                response_body=recorded.raw_response,
            )
            retain_unbounded_response_liability(truncated)
            raise truncated
        assert recorded.response is not None
        if recorded.pricing_error is not None:
            # Recover the type from the exact saved economics and frozen reservation,
            # never from error-message matching or by dispatching the request again.
            failure: ValueError = ValueError(recorded.pricing_error)
            try:
                reconcile_completion_economics(self._reservation, recorded.response.economics)
            except ProviderPricingUnavailableError:
                failure = ProviderPricingUnavailableError(recorded.pricing_error)
            except ValueError:
                pass
            if recorded.charge_usd is None:
                retain_unbounded_response_liability(failure)
            raise failure from None
        return recorded.response


class BudgetedEmbedding:
    """Include live retrieval queries in the same durable evaluation allowance."""

    def __init__(
        self,
        client: EmbeddingClient,
        budget: RequestBudget,
        reservation: EmbeddingCostReservation,
    ) -> None:
        """Bind the retrieval embedder without making a provider call."""
        self._client = client
        self._budget = budget
        self._reservation = reservation
        self._adapter = TypeAdapter(tuple[Embedding, ...])

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Reserve serialized query bytes and retain vectors before releasing the request.

        Args:
            texts: Full retrieval queries already admitted by the retriever.

        Returns:
            New or replayed embeddings in the original query order.

        Raises:
            SpendLimitReached: The queries cannot fit the approved allowance.
            ValueError: Saved request identity or response evidence is invalid.
        """
        tokens = sum(len(text.encode("utf-8")) for text in texts)
        maximum = (
            tokens
            * self._reservation.maximum_attempts
            * self._reservation.input_usd_per_million_tokens
            / 1_000_000
        )
        fingerprint = sha256_json(
            {
                "texts": list(texts),
                "reservation": self._reservation.model_dump(mode="json"),
            }
        )
        return self._budget.call(
            role="retrieval",
            fingerprint=fingerprint,
            maximum_cost_usd=maximum,
            operation=lambda: self._client.embed(texts),
            encode=lambda result: self._adapter.dump_json(result).decode("utf-8"),
            decode=self._adapter.validate_json,
            charge=lambda result: maximum,
        )

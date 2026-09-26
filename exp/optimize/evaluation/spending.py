"""Evaluation provider wrappers enforcing an approved allowance with durable replay."""

from collections.abc import Sequence

from pydantic import TypeAdapter

from exp.common.core.artifacts import sha256_json
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
from exp.runtime.models.budget import RequestBudget
from exp.simulation.engines.text.tokens import Utf8UpperBoundTokenCounter


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
        maximum = completion_request_cost_usd(
            self._reservation,
            input_tokens=Utf8UpperBoundTokenCounter().count(request),
            output_tokens=self._reservation.maximum_output_tokens,
        )
        fingerprint = sha256_json(
            {
                "request": request.model_dump(mode="json"),
                "reservation": self._reservation.model_dump(mode="json"),
                "served_model": self._served_model.model_dump(mode="json"),
            }
        )

        def dispatch() -> ModelResponse:
            """Retain the raw response; recorders independently reconcile its economics."""
            response = self._client.complete(request)
            if response.model not in (self._reservation.model, self._served_model):
                raise ValueError("provider response identity differs from the request reservation")
            return response

        def charge(response: ModelResponse) -> float:
            """Price observed usage and unresolved retry attempts with the frozen market rates."""
            economics = reconcile_completion_economics(self._reservation, response.economics)
            assert economics.cost_usd is not None
            return economics.cost_usd.value

        return self._budget.call(
            role=self._role,
            fingerprint=fingerprint,
            maximum_cost_usd=maximum,
            operation=dispatch,
            encode=lambda response: response.model_dump_json(),
            decode=ModelResponse.model_validate_json,
            charge=charge,
        )


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

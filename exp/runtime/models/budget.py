"""Durable request admission and exact response replay with an optional aggregate limit."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from exp.common.core.artifacts import sha256_json
from exp.common.project import ProjectStore
from exp.common.project.request_budget import RequestBudgetStore, RequestReceipt
from exp.runtime.models.providers.transport import (
    known_unbilled_attempts,
    retain_request_attempt_evidence,
)


class SpendLimitReached(BaseException):
    """Pause execution before an unaffordable request, without producing a failed rollout.

    Attributes:
        limit_usd: Total amount explicitly approved for this evaluation.
        accounted_usd: Completed charges and unresolved request reservations.
        required_usd: Minimum total allowance that would admit the pending request.
    """

    def __init__(self, limit_usd: float, accounted_usd: float, required_usd: float) -> None:
        """Retain safe numeric diagnostics for the operator's increase-and-resume choice."""
        self.limit_usd = limit_usd
        self.accounted_usd = accounted_usd
        self.required_usd = required_usd
        super().__init__("Evaluation paused at its spending limit; completed calls are saved.")


class RequestBudget:
    """Persist request reservations and successful responses before releasing their allowance.

    Every request belongs to a deterministic cell/attempt/role/ordinal coordinate. Replaying
    that coordinate returns its saved response only if the exact request digest matches.
    Unknown crash or failure charges retain their reservation. A response saved with unknown
    pricing replays its exact payload, including the owning wrapper's pricing error, while an
    unknown dispatch without saved output never replays silently.
    Certified unpaid failures replay their saved failure proof without dispatching again.
    No credentials or request bodies are saved. Response payloads are immutable project artifacts,
    with large files referenced from SQLite.
    """

    def __init__(
        self, project: ProjectStore, *, identity: str, maximum_cost_usd: float | None = None
    ) -> None:
        """Bind a local ledger to immutable execution identity and explicit authorization.

        Args:
            project: Owner of the shared SQLite accounting records and response artifacts.
            identity: Digest of immutable models, tasks, prompts, prices and execution settings.
            maximum_cost_usd: Optional total allowance, including completed and unknown calls.
                None disables the aggregate cap. Every request retains a finite reservation;
                incomplete tariffs are explicitly estimates and cannot enforce a numeric cap.

        Raises:
            ValueError: Authorization is invalid or the ledger belongs to another execution.
        """
        if maximum_cost_usd is not None and (
            not math.isfinite(maximum_cost_usd) or maximum_cost_usd <= 0
        ):
            raise ValueError("spending limit must be finite and positive")
        self._store = RequestBudgetStore(project, identity)
        self._limit = maximum_cost_usd
        self._condition = threading.Condition()
        self._active: set[str] = set()
        self._scope: ContextVar[tuple[str, dict[str, int]] | None] = ContextVar(
            f"request-budget-{identity}", default=None
        )
        self._store.authorize(self._limit)

    @contextmanager
    def scope(self, identity: str) -> Iterator[None]:
        """Reset per-role ordinals for one exact episode attempt or one rollout judgment."""
        token = self._scope.set((identity, {}))
        try:
            yield
        finally:
            self._scope.reset(token)

    @property
    def is_uncapped(self) -> bool:
        """Return whether this ledger's current authorization has no aggregate spend cap."""
        return self._limit is None

    @property
    def accounted_usd(self) -> float:
        """Return completed charges plus conservative reservations for unknown dispatches."""
        return self._store.total()

    def accounted_requests(self, coordinates: Sequence[tuple[str, str, int]]) -> float:
        """Sum distinct saved charges for exact scope, role, and ordinal coordinates.

        Unknown dispatches retain their full admission charge. Coordinates without a saved
        dispatch contribute nothing. Reading several scopes never repeats a provider call.

        Args:
            coordinates: Exact logical request scopes, roles, and zero-based ordinals.

        Returns:
            Total retained USD charge across the distinct matching request rows.
        """
        keys = sorted(
            {
                sha256_json({"scope": scope, "role": role, "ordinal": ordinal})
                for scope, role, ordinal in coordinates
            }
        )
        with self._store.transaction():
            return math.fsum(
                receipt.charge for key in keys if (receipt := self._store.read(key)) is not None
            )

    def call[ResultT](
        self,
        *,
        role: str,
        fingerprint: str,
        maximum_cost_usd: float,
        operation: Callable[[], ResultT],
        encode: Callable[[ResultT], str],
        decode: Callable[[str], ResultT],
        charge: Callable[[ResultT], float | None],
        cost_is_upper_bound: bool = True,
    ) -> ResultT:
        """Reserve an exact request, replay a saved answer, or pause before dispatch.

        Args:
            role: Distinguishes assistant aliases, world model, embedder and judge.
            fingerprint: Exact request, model, reservation and execution digest.
            maximum_cost_usd: Retry-inclusive known-rate reservation for this pending call.
                It is a strict bound only when ``cost_is_upper_bound`` is true.
            operation: Provider call, executed outside the transaction.
            encode: Serialize the successful result for exact replay.
            decode: Restore the successful result without contacting a provider.
            charge: Reconcile actual usage and any unresolved retry charges. None saves
                the response but retains explicitly unbounded liability.
            cost_is_upper_bound: False when the reservation only prices known tariff
                dimensions. Such requests require an uncapped aggregate policy; exact
                completed replay remains available under a subsequently supplied cap.

        Returns:
            A new or exactly replayed response. Replays spend no additional allowance.

        Raises:
            SpendLimitReached: No in-flight request can release enough allowance.
            ValueError: A saved coordinate drifted, previously failed, or violates its bound.
            Exception: The provider failed. Unknown dispatch retains its full reservation;
                certified wholly unpaid refusal retains a zero-charge failed receipt.
        """
        if not math.isfinite(maximum_cost_usd) or maximum_cost_usd < 0:
            raise ValueError("request reservation must be finite and nonnegative")
        if not isinstance(cost_is_upper_bound, bool):
            raise ValueError("cost_is_upper_bound must be an explicit boolean")
        scope = self._scope.get()
        if scope is None:
            raise ValueError("paid request requires an explicit execution scope")
        identity, ordinals = scope
        ordinal = ordinals.get(role, 0)
        ordinals[role] = ordinal + 1
        key = sha256_json({"scope": identity, "role": role, "ordinal": ordinal})
        cached = self._reserve(
            key, fingerprint, maximum_cost_usd, cost_is_upper_bound=cost_is_upper_bound
        )
        if cached is not None:
            return decode(cached)
        try:
            result = operation()
            cost = charge(result)
            if cost is not None and (
                not math.isfinite(cost)
                or cost < 0
                or (cost_is_upper_bound and cost > maximum_cost_usd + 1e-9)
            ):
                raise ValueError("provider charge exceeds the admitted request reservation")
            payload = encode(result)
            with self._condition:
                self._store.complete(key, cost, payload)
            return result
        except BaseException as error:
            with self._store.transaction():
                receipt = self._store.read(key)
                if receipt is not None and receipt.state == "pending":
                    unbilled_attempts = known_unbilled_attempts(error)
                    self._store.write(
                        key,
                        receipt.model_copy(
                            update={
                                "state": "unbilled" if unbilled_attempts else "unknown",
                                "charge": 0.0 if unbilled_attempts else receipt.charge,
                                "charge_is_upper_bound": True
                                if unbilled_attempts
                                else receipt.charge_is_upper_bound,
                                "unbilled_attempts": unbilled_attempts,
                            }
                        ),
                    )
            raise
        finally:
            with self._condition:
                self._active.discard(key)
                self._condition.notify_all()

    def _reserve(
        self, key: str, fingerprint: str, maximum: float, *, cost_is_upper_bound: bool
    ) -> str | None:
        """Atomically admit concurrent requests without multiplying the approved allowance.

        Args:
            key: Cell, role and ordinal digest within this evaluation.
            fingerprint: Exact request and pricing digest to verify on replay.
            maximum: Conservative pending request reservation in dollars.

        Returns:
            Serialized saved response, or None after reserving a new dispatch.

        Raises:
            SpendLimitReached: Settled spend leaves insufficient room for this request.
            ValueError: Replay identity changed or an earlier dispatch has unknown results.
        """
        with self._condition:
            while True:
                with self._store.transaction():
                    row = self._store.read(key)
                    if row is not None:
                        if row.fingerprint != fingerprint:
                            raise ValueError(
                                "saved provider request changed; start a new evaluation"
                            )
                        if row.state == "unbilled":
                            failure = ValueError(
                                "saved provider request was certified unpaid; "
                                "start a fresh attempt to retry"
                            )
                            retain_request_attempt_evidence(
                                failure,
                                attempts=row.unbilled_attempts,
                                unbilled_attempts=row.unbilled_attempts,
                            )
                            raise failure
                        if row.state != "complete" and not (
                            row.state == "unknown" and row.response is not None
                        ):
                            raise ValueError(
                                "saved provider dispatch has unresolved spend; not replayed"
                            )
                        return self._store.response(row)
                    if self._limit is not None and (
                        not cost_is_upper_bound or self._store.has_unbounded_liability()
                    ):
                        raise ValueError(
                            "a spending limit requires complete applicable token prices and "
                            "resolved earlier charges; reconcile pricing before dispatch"
                        )
                    spent = self._store.total()
                    if self._limit is None or spent + maximum <= self._limit + 1e-9:
                        self._store.write(
                            key,
                            RequestReceipt(
                                fingerprint=fingerprint,
                                charge=maximum,
                                state="pending",
                                charge_is_upper_bound=cost_is_upper_bound,
                            ),
                        )
                        self._active.add(key)
                        return None
                if not self._active and self._limit is not None:
                    raise SpendLimitReached(self._limit, spent, spent + maximum)
                self._condition.wait(timeout=1)

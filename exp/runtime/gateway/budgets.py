"""UTC-month gateway budget authority and atomic reservation checks."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from exp.common.config.settings import GatewayResourceSettings
from exp.common.core.artifacts import ContractModel, stable_id
from exp.runtime.gateway.attempt_costs import (
    LONG_CONTEXT_TIER_MARGIN_PERCENT as LONG_CONTEXT_TIER_MARGIN_PERCENT,
)
from exp.runtime.gateway.attempt_costs import (
    maximum_attempt_cost_nano_usd as maximum_attempt_cost_nano_usd,
)
from exp.runtime.gateway.auth import utc_text
from exp.runtime.gateway.budget_authority import (
    BudgetAliasRevision,
    active_budget_revision,
    validate_budget_revision,
)
from exp.runtime.gateway.interfaces import GatewayClock
from exp.runtime.gateway.ledger_valuation import (
    MAXIMUM_NANO_USD,
)
from exp.runtime.gateway.sqlite.migrations import initialize_database, persistent_connection
from exp.runtime.gateway.sqlite.store import SystemGatewayClock

__all__ = ["MAXIMUM_NANO_USD"]


class BudgetScopeKind(StrEnum):
    """Supported hard-limit scopes inside one UTC month."""

    TEAM = "team"
    IDENTITY = "identity"
    POOL = "pool"
    DEPLOYMENT = "deployment"


class BudgetScope(ContractModel):
    """One explicit team, identity, pool, or deployment allocation target."""

    kind: BudgetScopeKind
    identity_id: str | None = Field(default=None, min_length=1, max_length=256)
    alias_id: str | None = Field(default=None, min_length=1, max_length=256)
    pool_id: str | None = Field(default=None, min_length=1, max_length=256)
    deployment_id: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def _require_scope_shape(self) -> BudgetScope:
        """Require only the identifiers owned by the selected scope."""
        present = (
            self.identity_id is not None,
            self.alias_id is not None,
            self.pool_id is not None,
            self.deployment_id is not None,
        )
        expected = {
            BudgetScopeKind.TEAM: (False, False, False, False),
            BudgetScopeKind.IDENTITY: (True, False, False, False),
            BudgetScopeKind.POOL: (False, True, True, False),
            BudgetScopeKind.DEPLOYMENT: (False, True, True, True),
        }[self.kind]
        if present != expected:
            raise ValueError(f"{self.kind.value} budget scope has invalid identifiers")
        return self

    def key(self) -> str:
        """Return a concise operator-facing key for this scope."""
        parts = [self.kind.value]
        parts.extend(
            value
            for value in (self.identity_id, self.alias_id, self.pool_id, self.deployment_id)
            if value is not None
        )
        return ":".join(parts)

    def storage_key(self) -> str:
        """Return a collision-free content address for SQLite uniqueness."""
        return stable_id(
            "gateway-monthly-budget-scope",
            self.model_dump(mode="json", exclude_none=False),
        )


class MonthlyBudgetLimit(ContractModel):
    """One hard integer nano-USD allocation for an immutable UTC month bucket."""

    budget_id: str
    organization_id: str
    period: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    scope: BudgetScope
    limit_nano_usd: int = Field(ge=0, le=MAXIMUM_NANO_USD)
    strict_unknown_cost: bool = False
    created_at: datetime
    updated_at: datetime


class MonthlyBudgetRemaining(ContractModel):
    """Content-free remaining allocation for one configured monthly limit."""

    budget: MonthlyBudgetLimit
    charged_nano_usd: int = Field(ge=0)
    reserved_nano_usd: int = Field(ge=0)
    settled_nano_usd: int = Field(ge=0)
    remaining_nano_usd: int = Field(ge=0)
    unknown_cost_attempts: int = Field(ge=0)
    unknown_cost_input_tokens: int = Field(ge=0)
    unknown_cost_output_tokens: int = Field(ge=0)
    exhausted: bool


BudgetApplication = Literal["root", "destination", "shared"]


class BudgetRefusalBinding(ContractModel):
    """Exact request and budget scope selected by the atomic refusal decision.

    Attributes:
        request_id: Logical request whose atomic reservation was refused.
        organization_id: Tenant owning the refused request.
        alias_revision_id: Exact authorized alias revision used by the reservation.
        scope: Budget scope selected by the authoritative refusal.
        application: Whether that scope applies to the root, destination, or shared request.
    """

    request_id: str
    organization_id: str
    alias_revision_id: str
    scope: BudgetScope
    application: BudgetApplication


class BudgetReservationRejected(ValueError):
    """One physical route cannot reserve beneath every applicable hard limit."""

    def __init__(
        self,
        *,
        scope_kind: BudgetScopeKind,
        reason: str,
        binding: BudgetRefusalBinding | None = None,
    ) -> None:
        """Create a rejection; only a matching atomic binding permits scoped continuation."""
        if binding is not None and binding.scope.kind != scope_kind:
            raise ValueError("budget refusal binding kind differs from its rejection")
        self.scope_kind = scope_kind
        self.binding = binding
        super().__init__(reason)


class SQLiteBudgetStore:
    """Manage and report hard monthly limits in the shared gateway database."""

    def __init__(
        self,
        database_path: Path,
        *,
        clock: GatewayClock | None = None,
        busy_timeout_ms: int = 5_000,
        snapshot_max_bytes: int | None = None,
    ) -> None:
        """Bind storage, time, and an optional authoring-only snapshot resource budget."""
        self.database_path = database_path
        self._clock = SystemGatewayClock() if clock is None else clock
        self._busy_timeout_ms = busy_timeout_ms
        self._snapshot_max_bytes = (
            GatewayResourceSettings()
            if snapshot_max_bytes is None
            else GatewayResourceSettings(budget_snapshot_max_bytes=snapshot_max_bytes)
        ).budget_snapshot_max_bytes
        initialize_database(database_path, busy_timeout_ms=busy_timeout_ms)

    def set_limit(
        self,
        *,
        organization_id: str,
        period: str,
        scope: BudgetScope,
        limit_nano_usd: int,
        replace: bool = False,
        strict_unknown_cost: bool = False,
    ) -> tuple[bool, MonthlyBudgetLimit]:
        """Create or explicitly replace one monthly hard limit.

        Args:
            organization_id: Tenant whose attempts consume the allocation.
            period: Immutable UTC month in ``YYYY-MM`` form.
            scope: Exact allocation target.
            limit_nano_usd: Nonnegative integer hard limit.
            replace: Whether a different existing limit may be replaced.
            strict_unknown_cost: Whether unpriced attempts fail closed instead of
                being admitted and tracked as unknown cost with token volume.

        Returns:
            Change flag and current typed allocation.
        """
        if not 0 <= limit_nano_usd <= MAXIMUM_NANO_USD:
            raise ValueError("budget limit must fit a nonnegative SQLite integer")
        period_start = budget_period_start(period)
        now = utc_text(self._clock.now())
        budget_id = stable_id(
            "gateway-monthly-budget",
            {
                "organization_id": organization_id,
                "period_start": period_start,
                "scope_key": scope.storage_key(),
            },
        )
        revision = self._prepare_pool_scope(organization_id, scope)
        with self._transaction() as connection:
            self._require_scope_authority(
                connection,
                organization_id=organization_id,
                scope=scope,
            )
            if revision is not None and scope.alias_id is not None:
                current = active_budget_revision(connection, organization_id, scope.alias_id)
                if current != revision:
                    raise ValueError("budget alias revision changed; reload its scope and retry")
            row = connection.execute(
                """
                SELECT limit_nano_usd, strict_unknown_cost FROM gateway_monthly_budgets
                WHERE organization_id = ? AND period_start = ? AND scope_key = ?
                """,
                (organization_id, period_start, scope.storage_key()),
            ).fetchone()
            if row is not None:
                unchanged = (
                    int(row["limit_nano_usd"]) == limit_nano_usd
                    and bool(row["strict_unknown_cost"]) == strict_unknown_cost
                )
                if unchanged:
                    return False, self._read_limit(connection, budget_id)
                if not replace:
                    raise ValueError("monthly budget exists with another limit; pass --replace")
                connection.execute(
                    """
                    UPDATE gateway_monthly_budgets
                    SET limit_nano_usd = ?, strict_unknown_cost = ?, updated_at = ?
                    WHERE budget_id = ?
                    """,
                    (limit_nano_usd, int(strict_unknown_cost), now, budget_id),
                )
                return True, self._read_limit(connection, budget_id)
            connection.execute(
                """
                INSERT INTO gateway_monthly_budgets (
                    budget_id, organization_id, period_start, scope_kind, scope_key,
                    identity_id, alias_id, pool_id, deployment_id, limit_nano_usd,
                    strict_unknown_cost, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    budget_id,
                    organization_id,
                    period_start,
                    scope.kind.value,
                    scope.storage_key(),
                    scope.identity_id,
                    scope.alias_id,
                    scope.pool_id,
                    scope.deployment_id,
                    limit_nano_usd,
                    int(strict_unknown_cost),
                    now,
                    now,
                ),
            )
            _backfill_budget(connection, budget_id=budget_id)
            return True, self._read_limit(connection, budget_id)

    def limits(
        self,
        *,
        organization_id: str,
        period: str,
    ) -> tuple[MonthlyBudgetLimit, ...]:
        """List configured limits for one immutable UTC month."""
        period_start = budget_period_start(period)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM gateway_monthly_budgets
                WHERE organization_id = ? AND period_start = ?
                ORDER BY scope_kind, scope_key
                """,
                (organization_id, period_start),
            ).fetchall()
        return tuple(_limit_from_row(row) for row in rows)

    def remaining(
        self,
        *,
        organization_id: str,
        period: str,
    ) -> tuple[MonthlyBudgetRemaining, ...]:
        """Read every configured allocation from one consistent SQLite snapshot."""
        period_start = budget_period_start(period)
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                rows = connection.execute(
                    """
                    SELECT * FROM gateway_monthly_budgets
                    WHERE organization_id = ? AND period_start = ?
                    ORDER BY scope_kind, scope_key
                    """,
                    (organization_id, period_start),
                ).fetchall()
                results = tuple(_remaining_from_row(connection, row) for row in rows)
            finally:
                connection.rollback()
        return results

    def reconcile_unknown_costs(
        self,
        *,
        organization_id: str,
        period: str,
        scope: BudgetScope,
        assigned_cost_nano_usd: int,
    ) -> tuple[int, MonthlyBudgetRemaining]:
        """Settle every unknown-cost attempt on one limit at an explicit assigned cost.

        Unknown-cost attempts stay visible on the allocation until an operator
        deliberately assigns an exact integer nano-USD cost to each of them here,
        and under a strict limit they block every new reservation until then. Each
        reconciled attempt keeps its own charge row settled at the assigned cost, so
        per-attempt attribution stays exact, and a later natural settlement of a
        reconciled attempt is skipped rather than double-counted.

        Args:
            organization_id: Tenant whose allocation is reconciled.
            period: Immutable UTC month in ``YYYY-MM`` form.
            scope: Exact allocation target holding the unknown-cost attempts.
            assigned_cost_nano_usd: Nonnegative integer nano-USD charged per attempt.

        Returns:
            Count of reconciled attempts and the resulting allocation balance.

        Raises:
            ValueError: The assigned cost is unrepresentable, the limit does not
                exist, or settling would exceed SQLite integer capacity.
        """
        if not 0 <= assigned_cost_nano_usd <= MAXIMUM_NANO_USD:
            raise ValueError("assigned cost must fit a nonnegative SQLite integer")
        period_start = budget_period_start(period)
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM gateway_monthly_budgets
                WHERE organization_id = ? AND period_start = ? AND scope_key = ?
                """,
                (organization_id, period_start, scope.storage_key()),
            ).fetchone()
            if row is None:
                raise ValueError("monthly budget does not exist for this scope and period")
            budget_id = str(row["budget_id"])
            charges = connection.execute(
                """
                SELECT attempt_id FROM gateway_attempt_budget_charges
                WHERE budget_id = ?
                  AND reserved_nano_usd IS NULL AND settled_nano_usd IS NULL
                ORDER BY attempt_id
                """,
                (budget_id,),
            ).fetchall()
            if len(charges) != int(row["unknown_cost_attempts"]):
                raise RuntimeError("monthly budget counters are inconsistent")
            if not charges:
                return 0, _remaining_from_row(connection, row)
            added = assigned_cost_nano_usd * len(charges)
            if int(row["settled_nano_usd"]) + added > MAXIMUM_NANO_USD:
                raise ValueError("settled monthly gateway cost exceeds SQLite integer capacity")
            for charge in charges:
                updated = connection.execute(
                    """
                    UPDATE gateway_attempt_budget_charges SET settled_nano_usd = ?
                    WHERE budget_id = ? AND attempt_id = ?
                      AND reserved_nano_usd IS NULL AND settled_nano_usd IS NULL
                    """,
                    (assigned_cost_nano_usd, budget_id, str(charge["attempt_id"])),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("monthly budget counters are inconsistent")
            updated = connection.execute(
                """
                UPDATE gateway_monthly_budgets
                SET unknown_cost_attempts = unknown_cost_attempts - ?,
                    settled_nano_usd = settled_nano_usd + ?
                WHERE budget_id = ? AND unknown_cost_attempts >= ?
                """,
                (len(charges), added, budget_id, len(charges)),
            )
            if updated.rowcount != 1:
                raise RuntimeError("monthly budget counters are inconsistent")
            fresh = connection.execute(
                "SELECT * FROM gateway_monthly_budgets WHERE budget_id = ?",
                (budget_id,),
            ).fetchone()
            if fresh is None:
                raise RuntimeError("monthly budget disappeared inside its transaction")
            return len(charges), _remaining_from_row(connection, fresh)

    def _read_limit(
        self,
        connection: sqlite3.Connection,
        budget_id: str,
    ) -> MonthlyBudgetLimit:
        """Read one limit by stable ID inside the caller's transaction."""
        row = connection.execute(
            "SELECT * FROM gateway_monthly_budgets WHERE budget_id = ?",
            (budget_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("monthly budget disappeared inside its transaction")
        return _limit_from_row(row)

    def _require_scope_authority(
        self,
        connection: sqlite3.Connection,
        *,
        organization_id: str,
        scope: BudgetScope,
    ) -> None:
        """Require every referenced allocation target before storing a limit."""
        organization = connection.execute(
            "SELECT 1 FROM organizations WHERE organization_id = ? AND active = 1",
            (organization_id,),
        ).fetchone()
        if organization is None:
            raise ValueError("budget organization is not active")
        if scope.identity_id is not None:
            identity = connection.execute(
                """
                SELECT 1 FROM identities
                WHERE organization_id = ? AND identity_id = ? AND active = 1
                """,
                (organization_id, scope.identity_id),
            ).fetchone()
            if identity is None:
                raise ValueError("budget identity is not active")
        if scope.alias_id is not None:
            alias = connection.execute(
                """
                SELECT 1 FROM gateway_aliases
                WHERE organization_id = ? AND alias_id = ? AND active = 1
                """,
                (organization_id, scope.alias_id),
            ).fetchone()
            if alias is None:
                raise ValueError("budget alias is not active")

    def _prepare_pool_scope(
        self, organization_id: str, scope: BudgetScope
    ) -> BudgetAliasRevision | None:
        """Validate pinned graph outside the write transaction and return its revision fence."""
        if scope.pool_id is None or scope.alias_id is None:
            return None
        with self._connect() as connection:
            revision = active_budget_revision(connection, organization_id, scope.alias_id)
        return validate_budget_revision(
            self.database_path,
            revision,
            scope.pool_id,
            scope.deployment_id,
            self._snapshot_max_bytes,
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Check out one reusable configured connection."""
        with persistent_connection(
            self.database_path, busy_timeout_ms=self._busy_timeout_ms
        ) as connection:
            yield connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run one immediate mutation transaction."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()


def current_budget_period(now: datetime) -> str:
    """Return the UTC ``YYYY-MM`` bucket containing one aware instant."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("budget clock must return a timezone-aware instant")
    return now.astimezone(UTC).strftime("%Y-%m")


def budget_period_start(period: str) -> str:
    """Normalize one ``YYYY-MM`` value to its immutable UTC bucket start."""
    try:
        parsed = datetime.strptime(period, "%Y-%m").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("budget period must use YYYY-MM") from exc
    canonical = parsed.strftime("%Y-%m")
    if canonical != period:
        raise ValueError("budget period must use zero-padded YYYY-MM")
    return f"{period}-01T00:00:00+00:00"


def require_attempt_budget(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    identity_id: str,
    alias_id: str,
    pool_id: str,
    deployment_id: str,
    attempt_id: str,
    period_start: str,
    maximum_cost_nano_usd: int | None,
    root_pool_id: str | None = None,
    request_id: str | None = None,
    alias_revision_id: str | None = None,
) -> None:
    """Atomically require room beneath every limit applicable to one attempt.

    A limit with ``strict_unknown_cost`` fails closed: it rejects an unpriced
    attempt outright and rejects every attempt while unknown-cost attempts remain
    unresolved. A default limit admits an unpriced attempt, records it as one
    unknown-cost charge for later reconciliation, and keeps enforcing the hard
    cap over every known-cost reservation and settlement. Request-wide gates run
    before any skippable child or deployment refusal, regardless of scope-key order.
    """
    rows = connection.execute(
        """
        SELECT * FROM gateway_monthly_budgets
        WHERE organization_id = ? AND period_start = ? AND (
            scope_kind = 'team'
            OR (scope_kind = 'identity' AND identity_id = ?)
            OR (scope_kind = 'pool' AND alias_id = ? AND pool_id IN (?, ?))
            OR (scope_kind = 'deployment' AND alias_id = ? AND pool_id = ?
                AND deployment_id = ?)
        )
        ORDER BY CASE WHEN scope_kind IN ('team', 'identity')
            OR (scope_kind = 'pool' AND pool_id = ?) THEN 0 ELSE 1 END,
            scope_kind, scope_key
        """,
        (
            organization_id,
            period_start,
            identity_id,
            alias_id,
            pool_id,
            root_pool_id or pool_id,
            alias_id,
            pool_id,
            deployment_id,
            root_pool_id or pool_id,
        ),
    ).fetchall()
    if not rows:
        return
    for row in rows:
        scope_kind = BudgetScopeKind(str(row["scope_kind"]))
        refusal = None
        if request_id is not None and alias_revision_id is not None:
            scope = _limit_from_row(row).scope
            application: BudgetApplication = "shared"
            if scope_kind in (BudgetScopeKind.POOL, BudgetScopeKind.DEPLOYMENT):
                application = (
                    "root" if scope.pool_id == (root_pool_id or pool_id) else "destination"
                )
                if pool_id == (root_pool_id or pool_id):
                    application = "shared"
            refusal = BudgetRefusalBinding(
                request_id=request_id,
                organization_id=organization_id,
                alias_revision_id=alias_revision_id,
                scope=scope,
                application=application,
            )
        strict = bool(row["strict_unknown_cost"])
        unknown = int(row["unknown_cost_attempts"])
        charged = int(row["reserved_nano_usd"]) + int(row["settled_nano_usd"])
        limit = int(row["limit_nano_usd"])
        if strict and unknown:
            raise BudgetReservationRejected(
                scope_kind=scope_kind,
                binding=refusal,
                reason="monthly hard limit has prior attempts with unknown cost",
            )
        if maximum_cost_nano_usd is None:
            if strict:
                raise BudgetReservationRejected(
                    scope_kind=scope_kind,
                    binding=refusal,
                    reason="monthly hard limit requires a known maximum attempt cost",
                )
            continue
        if maximum_cost_nano_usd > max(0, limit - charged):
            raise BudgetReservationRejected(
                scope_kind=scope_kind,
                binding=refusal,
                reason=f"monthly {scope_kind.value} allocation is exhausted",
            )
    for row in rows:
        if maximum_cost_nano_usd is None:
            connection.execute(
                """
                UPDATE gateway_monthly_budgets
                SET unknown_cost_attempts = unknown_cost_attempts + 1
                WHERE budget_id = ?
                """,
                (str(row["budget_id"]),),
            )
            connection.execute(
                """
                INSERT INTO gateway_attempt_budget_charges (
                    budget_id, attempt_id
                ) VALUES (?, ?)
                """,
                (str(row["budget_id"]), attempt_id),
            )
            continue
        connection.execute(
            """
            UPDATE gateway_monthly_budgets
            SET reserved_nano_usd = reserved_nano_usd + ?
            WHERE budget_id = ?
            """,
            (maximum_cost_nano_usd, str(row["budget_id"])),
        )
        connection.execute(
            """
            INSERT INTO gateway_attempt_budget_charges (
                budget_id, attempt_id, reserved_nano_usd
            ) VALUES (?, ?, ?)
            """,
            (str(row["budget_id"]), attempt_id, maximum_cost_nano_usd),
        )


def settle_attempt_budgets(
    connection: sqlite3.Connection,
    *,
    attempt_id: str,
    settled_nano_usd: int | None,
) -> None:
    """Move every applicable attempt charge from reserved or unknown to settled."""
    rows = connection.execute(
        """
        SELECT c.budget_id, c.reserved_nano_usd, c.settled_nano_usd,
               b.settled_nano_usd AS budget_settled_nano_usd
        FROM gateway_attempt_budget_charges AS c
        JOIN gateway_monthly_budgets AS b ON b.budget_id = c.budget_id
        WHERE c.attempt_id = ?
        """,
        (attempt_id,),
    ).fetchall()
    for row in rows:
        if row["settled_nano_usd"] is not None or settled_nano_usd is None:
            continue
        budget_id = str(row["budget_id"])
        if int(row["budget_settled_nano_usd"]) + settled_nano_usd > MAXIMUM_NANO_USD:
            raise ValueError("settled monthly gateway cost exceeds SQLite integer capacity")
        reserved = None if row["reserved_nano_usd"] is None else int(row["reserved_nano_usd"])
        if reserved is None:
            update = connection.execute(
                """
                UPDATE gateway_monthly_budgets
                SET unknown_cost_attempts = unknown_cost_attempts - 1,
                    settled_nano_usd = settled_nano_usd + ?
                WHERE budget_id = ? AND unknown_cost_attempts > 0
                """,
                (settled_nano_usd, budget_id),
            )
        else:
            update = connection.execute(
                """
                UPDATE gateway_monthly_budgets
                SET reserved_nano_usd = reserved_nano_usd - ?,
                    settled_nano_usd = settled_nano_usd + ?
                WHERE budget_id = ? AND reserved_nano_usd >= ?
                """,
                (
                    reserved,
                    settled_nano_usd,
                    budget_id,
                    reserved,
                ),
            )
        if update.rowcount != 1:
            raise RuntimeError("monthly budget counters are inconsistent")
        charge = connection.execute(
            """
            UPDATE gateway_attempt_budget_charges SET settled_nano_usd = ?
            WHERE budget_id = ? AND attempt_id = ? AND settled_nano_usd IS NULL
            """,
            (settled_nano_usd, budget_id, attempt_id),
        )
        if charge.rowcount != 1:
            raise RuntimeError("attempt budget charge is already settled")


def _remaining_from_row(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> MonthlyBudgetRemaining:
    """Decode one materialized allocation balance and its unknown-cost token volume."""
    budget = _limit_from_row(row)
    reserved = int(row["reserved_nano_usd"])
    settled = int(row["settled_nano_usd"])
    charged = reserved + settled
    unknown = int(row["unknown_cost_attempts"])
    strict = budget.strict_unknown_cost
    remaining = 0 if strict and unknown else max(0, budget.limit_nano_usd - charged)
    input_tokens, output_tokens = _unknown_token_volume(connection, str(row["budget_id"]))
    return MonthlyBudgetRemaining(
        budget=budget,
        charged_nano_usd=charged,
        reserved_nano_usd=reserved,
        settled_nano_usd=settled,
        remaining_nano_usd=remaining,
        unknown_cost_attempts=unknown,
        unknown_cost_input_tokens=input_tokens,
        unknown_cost_output_tokens=output_tokens,
        exhausted=(strict and unknown > 0) or charged >= budget.limit_nano_usd,
    )


def _unknown_token_volume(
    connection: sqlite3.Connection,
    budget_id: str,
) -> tuple[int, int]:
    """Sum observed token volume across one limit's unresolved unknown-cost attempts.

    Cached-input and reasoning counts are subsets of the input and output totals
    (``GatewayUsage``), so the totals alone gauge the real traffic behind attempts
    whose price is still unknown.
    """
    row = connection.execute(
        """
        SELECT
            COALESCE(SUM(COALESCE(a.input_tokens, 0)), 0) AS input_volume,
            COALESCE(SUM(COALESCE(a.output_tokens, 0)), 0) AS output_volume
        FROM gateway_attempt_budget_charges AS c
        JOIN gateway_attempts AS a ON a.attempt_id = c.attempt_id
        WHERE c.budget_id = ?
          AND c.reserved_nano_usd IS NULL AND c.settled_nano_usd IS NULL
        """,
        (budget_id,),
    ).fetchone()
    return int(row["input_volume"]), int(row["output_volume"])


def _backfill_budget(connection: sqlite3.Connection, *, budget_id: str) -> None:
    """Bind prior attempts to a newly configured limit and materialize exact counters."""
    row = connection.execute(
        "SELECT * FROM gateway_monthly_budgets WHERE budget_id = ?",
        (budget_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("monthly budget disappeared before historical backfill")
    period_start = str(row["period_start"])
    predicate, parameters = _attempt_scope_predicate(row, period_start=period_start)
    attempts = connection.execute(
        f"""
        SELECT a.attempt_id, a.budget_reserved_nano_usd,
               a.budget_settled_nano_usd, a.estimated_cost_nano_usd
        FROM gateway_attempts AS a
        JOIN gateway_requests AS r ON r.request_id = a.request_id
        WHERE {predicate} ORDER BY a.attempt_id
        """,
        parameters,
    )
    reserved_total = 0
    settled_total = 0
    unknown_total = 0
    for attempt in attempts:
        settled = _historical_settlement(attempt)
        reserved = (
            int(attempt["budget_reserved_nano_usd"])
            if settled is None and attempt["budget_reserved_nano_usd"] is not None
            else None
        )
        unknown = settled is None and reserved is None
        reserved_total += reserved or 0
        settled_total += settled or 0
        unknown_total += int(unknown)
        if reserved_total > MAXIMUM_NANO_USD or settled_total > MAXIMUM_NANO_USD:
            raise ValueError("historical monthly gateway cost exceeds SQLite integer capacity")
        connection.execute(
            """
            INSERT INTO gateway_attempt_budget_charges (
                budget_id, attempt_id, reserved_nano_usd, settled_nano_usd
            ) VALUES (?, ?, ?, ?)
            """,
            (budget_id, str(attempt["attempt_id"]), reserved, settled),
        )
    connection.execute(
        """
        UPDATE gateway_monthly_budgets
        SET reserved_nano_usd = ?, settled_nano_usd = ?, unknown_cost_attempts = ?
        WHERE budget_id = ?
        """,
        (reserved_total, settled_total, unknown_total, budget_id),
    )


def _historical_settlement(row: sqlite3.Row) -> int | None:
    """Return one prior attempt's settled cost while preserving active reservations."""
    if row["budget_settled_nano_usd"] is not None:
        return int(row["budget_settled_nano_usd"])
    if row["budget_reserved_nano_usd"] is not None:
        return None
    if row["estimated_cost_nano_usd"] is not None:
        return int(row["estimated_cost_nano_usd"])
    return None


def _attempt_scope_predicate(
    row: sqlite3.Row,
    *,
    period_start: str,
) -> tuple[str, tuple[str, ...]]:
    """Return one fixed SQL predicate and parameters for a stored scope."""
    base = "a.organization_id = ? AND a.budget_period_start = ?"
    organization_id = str(row["organization_id"])
    kind = BudgetScopeKind(str(row["scope_kind"]))
    if kind is BudgetScopeKind.TEAM:
        return base, (organization_id, period_start)
    if kind is BudgetScopeKind.IDENTITY:
        return f"{base} AND r.identity_id = ?", (
            organization_id,
            period_start,
            str(row["identity_id"]),
        )
    if kind is BudgetScopeKind.POOL:
        # Charge the destination or the accepted revision's root, never the active alias.
        return (
            f"""{base} AND r.alias_id = ? AND (a.pool_id = ? OR EXISTS (
            SELECT 1 FROM alias_revisions AS revision
            WHERE revision.organization_id = r.organization_id
              AND revision.alias_id = r.alias_id
              AND revision.revision_id = r.alias_revision_id
              AND revision.pool_id = ?
        ))""",
            (
                organization_id,
                period_start,
                str(row["alias_id"]),
                str(row["pool_id"]),
                str(row["pool_id"]),
            ),
        )
    return f"{base} AND r.alias_id = ? AND a.pool_id = ? AND a.deployment_id = ?", (
        organization_id,
        period_start,
        str(row["alias_id"]),
        str(row["pool_id"]),
        str(row["deployment_id"]),
    )


def _limit_from_row(row: sqlite3.Row) -> MonthlyBudgetLimit:
    """Decode one strict SQLite budget row."""
    return MonthlyBudgetLimit(
        budget_id=str(row["budget_id"]),
        organization_id=str(row["organization_id"]),
        period=str(row["period_start"])[:7],
        scope=BudgetScope(
            kind=BudgetScopeKind(str(row["scope_kind"])),
            identity_id=(None if row["identity_id"] is None else str(row["identity_id"])),
            alias_id=None if row["alias_id"] is None else str(row["alias_id"]),
            pool_id=None if row["pool_id"] is None else str(row["pool_id"]),
            deployment_id=(None if row["deployment_id"] is None else str(row["deployment_id"])),
        ),
        limit_nano_usd=int(row["limit_nano_usd"]),
        strict_unknown_cost=bool(row["strict_unknown_cost"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )

"""One operation's dollar ceiling across every paid role, enforced at physical dispatch.

An operation (one consented CLI invocation and its resumes) owns one durable SQLite ledger. Every
physical provider POST sent through the admitted transport reserves its worst-case cost before
its intent is written: catalog prices applied to a byte upper bound of the exact request body
and its reserved output allowance. Metadata GETs carry no body and are not priced. A returned
response settles the reservation at its observed usage, freeing the difference. A failure
whose outcome is unknown keeps its full reservation and is never refunded. A request whose
tariff, alias or output allowance is unknown cannot be bounded, so it is refused unless the
operator explicitly allowed unpriced attempts; then it counts against that explicit attempt
ceiling and is reported as unpriced.

Reservations live in the ledger rather than in process memory, so every process on the host that
opens the same operation shares one ceiling. Each in-flight row records its holder process. A
row still ``reserved`` when its holder died was provably never sent (the dispatch mark commits
before network I/O) and is released by the next reader; a ``dispatched`` row whose holder died
has an unknown outcome and keeps its full reservation. Resume therefore neither loses nor
double counts a reservation.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue
from exp.common.core.host_scope import current_host
from exp.common.models.catalog_prices import GatewayTokenPrices
from exp.common.models.client import RequestAdmissionRefused
from exp.common.models.token_cost import schedule_maximum_cost_nano_usd, schedule_prices_complete
from exp.common.sqlite.connection import connect_database
from exp.runtime.models.providers.backpressure import holder_alive, process_start
from exp.runtime.models.providers.transport import JsonHttpResponse

SCHEMA_VERSION: Final = 1
"""Ledger schema identity stored in ``PRAGMA user_version``."""

OPERATION_DOLLAR_AUTHORIZATION: Final = "operation-explicit-spend"
"""Reserved authorization identity whose dollar ceiling overrides every stage estimate."""

OPERATION_UNPRICED_AUTHORIZATION: Final = "operation-unpriced-allowance"
"""Reserved authorization identity whose unpriced allowance overrides every stage grant."""

MAXIMUM_UNPRICED_ATTEMPTS: Final = 100_000
"""Largest explicit unpriced-attempt ceiling one authorization may grant."""

_TOLERANCE_USD: Final = 1e-9
_MEDIA_PART_TYPES: Final = frozenset(
    {
        "image",
        "image_url",
        "input_image",
        "input_audio",
        "audio",
        "input_file",
        "file",
        "document",
        "video_url",
    }
)
_MEDIA_KEYS: Final = frozenset({"inline_data", "inlineData", "file_data", "fileData"})
_OUTPUT_FIELDS: Final = ("max_tokens", "max_completion_tokens", "max_output_tokens")
_IN_FLIGHT: Final = ("reserved", "dispatched")

ReservationState = Literal["reserved", "dispatched", "settled", "unknown", "released"]


class SpendCeilingReached(RequestAdmissionRefused):
    """The next reservation does not fit the operation's ceiling, so nothing was sent.

    This is an infrastructure stop, never a scored budget exhaustion: the stopped work stays
    incomplete and resumes in place once the ceiling is raised.

    Attributes:
        ceiling_usd: The operation's authorized dollar ceiling.
        accounted_usd: Settled charges plus every retained reservation.
        required_usd: Smallest ceiling that would admit the refused request.
    """

    def __init__(self, ceiling_usd: float, accounted_usd: float, required_usd: float) -> None:
        """Keep the numbers an operator needs to raise the ceiling and resume."""
        self.ceiling_usd = ceiling_usd
        self.accounted_usd = accounted_usd
        self.required_usd = required_usd
        super().__init__(
            f"the operation's spend ceiling ${ceiling_usd:.6f} is reached: ${accounted_usd:.6f} "
            f"is settled or reserved and the next request needs ${required_usd:.6f}; it was "
            "not sent"
        )


class UnpricedDispatchRefused(RequestAdmissionRefused):
    """A request without a provable worst-case cost was stopped before its intent was written.

    Attributes:
        alias: Catalog alias the request was priced as, or None when no alias was bound.
    """

    def __init__(self, alias: str | None, reason: str) -> None:
        """Explain what is unpriced and how the operator can authorize or price it."""
        self.alias = alias
        super().__init__(
            f"{reason}; the operation's spend ceiling cannot bound it, so it was not sent. "
            "Add complete prices for the alias to the model catalog, or authorize an explicit "
            "unpriced-attempt ceiling for this operation, then run the same command again."
        )


@dataclass(frozen=True)
class AliasTariff:
    """Credential-free worst-case prices for requests priced as one catalog alias.

    Attributes:
        schedule: Complete authored token schedule; takes precedence over the flat rates.
        input_usd_per_million: Highest input-side rate (fresh, cached or cache-write), or None.
        output_usd_per_million: Highest output-side rate (output or reasoning), or None. A
            request with no output allowance (an embedding) is priced without it.
        context_window_tokens: Provider context capacity, bounding billed input per request.
        default_output_tokens: Output allowance of a request whose body sets none (the
            decision wire, or a model's published maximum), or None when unbounded.
        output_uses_context_bound: Reserve the trusted total model window for generation,
            including reasoning; a missing window leaves the request unpriced.
        repeated_input: Times the provider may bill the whole input in one request, such as
            every question of a native decision.
        input_framing_tokens: Extra input tokens per repetition added by the provider.
    """

    schedule: GatewayTokenPrices | None = None
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    context_window_tokens: int | None = None
    # Hand-authored tariffs retain the caller's explicit output-bound contract. Catalog
    # consumers opt into the conservative joint-window contract below.
    default_output_tokens: int | None = None
    output_uses_context_bound: bool = False
    repeated_input: int = 1
    input_framing_tokens: int = 0

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        """Return the highest known charge for these token counts, or None when unpriced.

        Args:
            input_tokens: Billed input tokens, including cache reads and writes.
            output_tokens: Billed output tokens, including reasoning.

        Returns:
            A strict upper bound in USD, or None when a needed rate is unknown.
        """
        if self.schedule is not None:
            if not schedule_prices_complete(self.schedule, maximum_input_tokens=input_tokens):
                return None
            nano = schedule_maximum_cost_nano_usd(
                self.schedule, input_tokens=input_tokens, output_tokens=output_tokens
            )
            return nano / 1_000_000_000
        if self.input_usd_per_million is None or (
            output_tokens and self.output_usd_per_million is None
        ):
            return None
        return (
            input_tokens * self.input_usd_per_million
            + output_tokens * (self.output_usd_per_million or 0.0)
        ) / 1_000_000


@dataclass(frozen=True)
class Reservation:
    """One physical request's reservation in the operation ledger.

    Attributes:
        id: Ledger row identity.
        priced: Whether a dollar amount is reserved; False counts one unpriced attempt.
        reserved_usd: Worst-case amount held until settlement; zero when unpriced.
        tariff: Prices settling the observed usage, or None when unpriced.
        embedding: Whether the dispatched endpoint has no output-token charge.
    """

    id: str
    priced: bool
    reserved_usd: float
    tariff: AliasTariff | None
    embedding: bool = False


class RoleSpend(ContractModel):
    """Spend of one priced alias (or unbound route) in an operation.

    Attributes:
        role: Catalog alias the requests were priced as, or ``unbound``.
        attempts: Physical requests reserved, excluding provably unsent releases.
        settled_usd: Charges settled from returned usage.
        unknown_usd: Full reservations kept for unknown outcomes.
        in_flight_usd: Reservations of requests still running.
        unpriced_attempts: Requests sent without a dollar bound under an explicit allowance.
    """

    role: str
    attempts: int = Field(ge=0)
    settled_usd: float = Field(ge=0)
    unknown_usd: float = Field(ge=0)
    in_flight_usd: float = Field(ge=0)
    unpriced_attempts: int = Field(ge=0)


class SpendAuthorization(ContractModel):
    """One consented stage of an operation's ceiling.

    Attributes:
        stage: Identity of the consented cost inventory.
        ceiling_usd: Dollar ceiling this stage authorized.
        unpriced_attempts: Explicit unpriced-attempt ceiling this stage authorized.
        authorized_at: Informational time of the consent.
    """

    stage: str
    ceiling_usd: float = Field(ge=0)
    unpriced_attempts: int = Field(ge=0)
    authorized_at: str


class SpendSummary(ContractModel):
    """The ledger's accounting against the authorized ceiling.

    Attributes:
        ceiling_usd: Explicit operation ceiling, or the sum of current stage estimates.
        unpriced_attempt_ceiling: Current explicit unpriced-attempt ceiling.
        accounted_usd: Settled charges plus every retained reservation.
        settled_usd: Charges settled from returned usage.
        unknown_usd: Full reservations kept for unknown outcomes; never refunded.
        in_flight_usd: Reservations of requests still running.
        unpriced_attempts: Unpriced requests counted against their attempt ceiling.
        released_attempts: Reservations released because the request was provably not sent.
        overrun_attempts: Settlements whose observed usage exceeded the reservation.
        authorizations: Every consent appended to this operation, in order.
        roles: Spend by priced alias.
    """

    ceiling_usd: float = Field(ge=0)
    unpriced_attempt_ceiling: int = Field(ge=0)
    accounted_usd: float = Field(ge=0)
    settled_usd: float = Field(ge=0)
    unknown_usd: float = Field(ge=0)
    in_flight_usd: float = Field(ge=0)
    unpriced_attempts: int = Field(ge=0)
    released_attempts: int = Field(ge=0)
    overrun_attempts: int = Field(ge=0)
    authorizations: tuple[SpendAuthorization, ...]
    roles: tuple[RoleSpend, ...]


class OperationSpendCeiling:
    """Durable, cross-process dollar ceiling shared by every paid role of one operation.

    Args:
        path: SQLite ledger of the operation, created with its schema when absent.
        tariffs: Worst-case prices by catalog alias. An alias outside it is unpriced.
        sleep: Wait between admission polls while another request's settlement is pending.
        maximum_wait_seconds: Longest wait for in-flight settlements before refusing.

    Raises:
        ValueError: The file holds another schema.
    """

    def __init__(
        self,
        path: Path,
        *,
        tariffs: Mapping[str, AliasTariff],
        sleep: Callable[[float], None] = time.sleep,
        maximum_wait_seconds: float = 600.0,
    ) -> None:
        """Create or open the ledger without authorizing any spend."""
        self._path, self._tariffs = path, dict(tariffs)
        self._sleep, self._maximum_wait = sleep, maximum_wait_seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                if connection.execute(
                    "SELECT COUNT(*) FROM sqlite_master"
                    " WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchone()[0]:
                    raise ValueError("spend ledger path holds another database; use a fresh path")
                connection.execute(
                    """CREATE TABLE authorizations (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        stage TEXT NOT NULL,
                        ceiling_usd REAL NOT NULL CHECK (ceiling_usd >= 0),
                        unpriced_attempts INTEGER NOT NULL CHECK (unpriced_attempts >= 0),
                        authorized_at TEXT NOT NULL
                    ) STRICT"""
                )
                connection.execute(
                    """CREATE TRIGGER authorizations_append_only BEFORE UPDATE ON authorizations
                    BEGIN SELECT RAISE(ABORT, 'authorizations are append-only'); END"""
                )
                connection.execute(
                    """CREATE TABLE reservations (
                        id TEXT PRIMARY KEY,
                        seq INTEGER NOT NULL UNIQUE,
                        role TEXT NOT NULL,
                        priced INTEGER NOT NULL CHECK (priced IN (0, 1)),
                        reserved_usd REAL NOT NULL CHECK (reserved_usd >= 0),
                        charge_usd REAL NOT NULL CHECK (charge_usd >= 0),
                        state TEXT NOT NULL CHECK (state IN
                            ('reserved', 'dispatched', 'settled', 'unknown', 'released')),
                        overrun INTEGER NOT NULL DEFAULT 0 CHECK (overrun IN (0, 1)),
                        attempt_id TEXT,
                        machine TEXT NOT NULL,
                        pid INTEGER NOT NULL,
                        start TEXT,
                        reserved_at TEXT NOT NULL
                    ) STRICT"""
                )
                connection.execute(
                    """CREATE TRIGGER reservations_kept BEFORE DELETE ON reservations
                    BEGIN SELECT RAISE(ABORT, 'reservations are never deleted'); END"""
                )
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            elif version != SCHEMA_VERSION:
                raise ValueError(
                    f"spend ledger schema {version} is not {SCHEMA_VERSION}; use a fresh path"
                )

    @property
    def path(self) -> Path:
        """Return the operation's ledger file."""
        return self._path

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run one ``BEGIN IMMEDIATE`` transaction on a fresh connection, then close it."""
        connection = connect_database(self._path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
        finally:
            connection.close()

    def authorize(self, stage: str, *, ceiling_usd: float, unpriced_attempts: int = 0) -> None:
        """Append one consented stage; its latest authorization replaces its earlier ones.

        A resumed operation re-consents the same stage, so the ceiling is not multiplied by
        resumes, while spend already accounted keeps counting against it. The reserved
        operation identities override stage grants without rewriting their evidence rows.

        Args:
            stage: Identity of the consented cost inventory.
            ceiling_usd: Displayed and authorized dollar ceiling of the stage.
            unpriced_attempts: Explicit ceiling on unpriced requests; zero refuses them.

        Raises:
            ValueError: The ceiling or allowance is negative, non-finite or too large.
        """
        _validate_authorization(ceiling_usd, unpriced_attempts)
        with self._transaction() as connection:
            self._append_authorization(connection, stage, ceiling_usd, unpriced_attempts)

    def authorize_policy(self, *, ceiling_usd: float | None, unpriced_attempts: int) -> None:
        """Atomically replace explicit operation limits without changing historical grants.

        Args:
            ceiling_usd: Explicit operation dollar cap, or None to retain stage estimates or
                an already explicit cap.
            unpriced_attempts: Whole-operation allowance replacing every earlier grant.

        Raises:
            ValueError: A limit is negative, non-finite or outside the allowed range.
        """
        _validate_authorization(0.0 if ceiling_usd is None else ceiling_usd, unpriced_attempts)
        with self._transaction() as connection:
            if ceiling_usd is not None:
                self._append_authorization(
                    connection, OPERATION_DOLLAR_AUTHORIZATION, ceiling_usd, 0
                )
            self._append_authorization(
                connection, OPERATION_UNPRICED_AUTHORIZATION, 0.0, unpriced_attempts
            )

    def _append_authorization(
        self, connection: sqlite3.Connection, stage: str, ceiling_usd: float, unpriced_attempts: int
    ) -> None:
        """Append a validated authorization inside the caller's transaction."""
        connection.execute(
            "INSERT INTO authorizations (stage, ceiling_usd, unpriced_attempts, authorized_at)"
            " VALUES (?, ?, ?, ?)",
            (stage, ceiling_usd, unpriced_attempts, datetime.now(UTC).isoformat()),
        )

    def reserve(self, *, alias: str | None, url: str, payload: JsonObject) -> Reservation:
        """Reserve one request's worst-case cost, or refuse it before its intent is written.

        Args:
            alias: Catalog alias the request is priced as, or None when unbound.
            url: Absolute provider endpoint, which distinguishes embedding requests.
            payload: Exact JSON request body.

        Returns:
            The durable reservation to mark, settle, keep or release.

        Raises:
            SpendCeilingReached: The reservation does not fit the remaining ceiling.
            UnpricedDispatchRefused: The cost is unbounded and no unpriced attempt remains.
            ValueError: No ceiling was authorized for this operation.
        """
        tariff = None if alias is None else self._tariffs.get(alias)
        bounds = None if tariff is None else request_token_bounds(url, payload, tariff)
        cost = None if tariff is None or bounds is None else tariff.cost_usd(*bounds)
        if alias is None:
            reason = "the request is not bound to a catalog alias"
        elif tariff is None:
            reason = f"alias {alias!r} has no complete catalog prices"
        else:
            reason = f"alias {alias!r} request has no bounded token or price schedule"
        waited = 0.0
        delay = 0.05
        while True:
            with self._transaction() as connection:
                self._recover(connection)
                ceiling, allowance = _authorized(connection)
                if ceiling is None:
                    raise ValueError(
                        "no spend ceiling is authorized for this operation; "
                        "consent to its cost inventory first"
                    )
                if cost is None:
                    used = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM reservations"
                            " WHERE priced = 0 AND state != 'released'"
                        ).fetchone()[0]
                    )
                    if used >= allowance:
                        raise UnpricedDispatchRefused(alias, reason)
                    return self._insert(
                        connection, alias, None, 0.0, embedding=is_embedding_endpoint(url)
                    )
                accounted = _accounted(connection)
                if accounted + cost <= ceiling + _TOLERANCE_USD:
                    return self._insert(
                        connection, alias, tariff, cost, embedding=is_embedding_endpoint(url)
                    )
                pending = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM reservations WHERE priced = 1"
                        " AND state IN ('reserved', 'dispatched')"
                    ).fetchone()[0]
                )
            if not pending or waited >= self._maximum_wait:
                raise SpendCeilingReached(ceiling, accounted, accounted + cost)
            self._sleep(delay)
            waited += delay
            delay = min(delay * 2, 1.0)

    def _insert(
        self,
        connection: sqlite3.Connection,
        alias: str | None,
        tariff: AliasTariff | None,
        cost: float,
        *,
        embedding: bool,
    ) -> Reservation:
        """Write a ``reserved`` row owned by this process inside the admission transaction."""
        identity = uuid4().hex
        sequence = int(
            connection.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM reservations").fetchone()[0]
        )
        pid = os.getpid()
        connection.execute(
            "INSERT INTO reservations (id, seq, role, priced, reserved_usd, charge_usd, state,"
            " machine, pid, start, reserved_at) VALUES (?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?)",
            (
                identity,
                sequence,
                alias or "unbound",
                int(tariff is not None),
                cost,
                cost,
                current_host().machine,
                pid,
                process_start(pid),
                datetime.now(UTC).isoformat(),
            ),
        )
        return Reservation(identity, tariff is not None, cost, tariff, embedding)

    def _recover(self, connection: sqlite3.Connection) -> None:
        """Resolve in-flight rows of dead holders on this machine: unsent are released."""
        machine = current_host().machine
        rows = connection.execute(
            "SELECT id, state, pid, start FROM reservations"
            " WHERE state IN ('reserved', 'dispatched') AND machine = ?",
            (machine,),
        ).fetchall()
        for identity, state, pid, start in rows:
            if holder_alive(int(pid), start):
                continue
            if state == "reserved":
                connection.execute(
                    "UPDATE reservations SET state = 'released', charge_usd = 0 WHERE id = ?",
                    (identity,),
                )
            else:
                connection.execute(
                    "UPDATE reservations SET state = 'unknown' WHERE id = ?", (identity,)
                )

    def _transition(
        self,
        reservation: Reservation,
        *,
        source: tuple[ReservationState, ...],
        target: ReservationState,
        charge_usd: float | None = None,
        overrun: bool = False,
        attempt_id: str | None = None,
    ) -> None:
        """Move one row between states, refusing an unexpected source state."""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state, charge_usd FROM reservations WHERE id = ?", (reservation.id,)
            ).fetchone()
            if row is None or row[0] not in source:
                raise ValueError(
                    f"spend reservation is {row[0] if row else 'missing'}, not {'/'.join(source)}"
                )
            connection.execute(
                "UPDATE reservations SET state = ?, charge_usd = ?, overrun = ?,"
                " attempt_id = COALESCE(?, attempt_id) WHERE id = ?",
                (
                    target,
                    float(row[1]) if charge_usd is None else charge_usd,
                    int(overrun),
                    attempt_id,
                    reservation.id,
                ),
            )

    def mark_dispatched(self, reservation: Reservation, attempt_id: str) -> None:
        """Commit that network I/O is about to start; a crash after this is an unknown outcome."""
        self._transition(
            reservation, source=("reserved",), target="dispatched", attempt_id=attempt_id
        )

    def release(self, reservation: Reservation) -> None:
        """Free a reservation whose request was provably never sent."""
        self._transition(reservation, source=("reserved",), target="released", charge_usd=0.0)

    def retain_unknown(self, reservation: Reservation) -> None:
        """Keep the full reservation of a request whose outcome and billing are unknown."""
        self._transition(reservation, source=("dispatched",), target="unknown")

    def settle(self, reservation: Reservation, response: JsonHttpResponse) -> float:
        """Settle a returned request at its observed usage, freeing the reserved difference.

        A certified unbilled refusal settles at zero. A body without readable usage keeps
        the full reservation. Observed usage above the reservation is recorded as charged
        and flagged as an overrun rather than hidden.

        Args:
            reservation: The dispatched reservation.
            response: Exact decoded provider response.

        Returns:
            The settled charge in USD.
        """
        charge = reservation.reserved_usd
        if response.known_unbilled:
            charge = 0.0
        elif reservation.tariff is not None:
            observed = observed_token_usage(response.body, embedding=reservation.embedding)
            if observed is not None:
                priced = reservation.tariff.cost_usd(*observed)
                if priced is not None:
                    charge = priced
        overrun = reservation.priced and charge > reservation.reserved_usd + _TOLERANCE_USD
        self._transition(
            reservation,
            source=("dispatched",),
            target="settled",
            charge_usd=charge if reservation.priced else 0.0,
            overrun=overrun,
        )
        return charge

    def summary(self) -> SpendSummary:
        """Return the ledger's accounting after resolving dead holders' in-flight rows."""
        with self._transaction() as connection:
            self._recover(connection)
            ceiling, allowance = _authorized(connection)
            authorizations = tuple(
                SpendAuthorization(
                    stage=row[0],
                    ceiling_usd=row[1],
                    unpriced_attempts=row[2],
                    authorized_at=row[3],
                )
                for row in connection.execute(
                    "SELECT stage, ceiling_usd, unpriced_attempts, authorized_at"
                    " FROM authorizations ORDER BY seq"
                )
            )
            rows = connection.execute(
                "SELECT role, priced, state, charge_usd, overrun FROM reservations ORDER BY seq"
            ).fetchall()
        roles: dict[str, dict[str, float]] = {}
        released = overruns = 0
        for role, priced, state, charge, overrun in rows:
            if state == "released":
                released += 1
                continue
            line = roles.setdefault(
                role,
                {"attempts": 0, "settled": 0.0, "unknown": 0.0, "in_flight": 0.0, "unpriced": 0},
            )
            line["attempts"] += 1
            overruns += int(overrun)
            if not priced:
                line["unpriced"] += 1
            elif state == "settled":
                line["settled"] += charge
            elif state == "unknown":
                line["unknown"] += charge
            else:
                line["in_flight"] += charge
        spend = tuple(
            RoleSpend(
                role=role,
                attempts=int(line["attempts"]),
                settled_usd=line["settled"],
                unknown_usd=line["unknown"],
                in_flight_usd=line["in_flight"],
                unpriced_attempts=int(line["unpriced"]),
            )
            for role, line in sorted(roles.items())
        )
        settled = math.fsum(row.settled_usd for row in spend)
        unknown = math.fsum(row.unknown_usd for row in spend)
        in_flight = math.fsum(row.in_flight_usd for row in spend)
        return SpendSummary(
            ceiling_usd=ceiling or 0.0,
            unpriced_attempt_ceiling=allowance,
            accounted_usd=settled + unknown + in_flight,
            settled_usd=settled,
            unknown_usd=unknown,
            in_flight_usd=in_flight,
            unpriced_attempts=sum(row.unpriced_attempts for row in spend),
            released_attempts=released,
            overrun_attempts=overruns,
            authorizations=authorizations,
            roles=spend,
        )


def _validate_authorization(ceiling_usd: float, unpriced_attempts: int) -> None:
    """Reject invalid authorization limits before a transaction can change any evidence."""
    if not math.isfinite(ceiling_usd) or ceiling_usd < 0:
        raise ValueError("spend ceiling must be finite and nonnegative")
    if not 0 <= unpriced_attempts <= MAXIMUM_UNPRICED_ATTEMPTS:
        raise ValueError(
            f"unpriced-attempt ceiling must be between 0 and {MAXIMUM_UNPRICED_ATTEMPTS}"
        )


def _authorized(connection: sqlite3.Connection) -> tuple[float | None, int]:
    """Apply explicit operation limits over stage grants, preserving every historical row."""
    rows = connection.execute(
        "SELECT stage, ceiling_usd, unpriced_attempts FROM authorizations"
        " WHERE seq IN (SELECT MAX(seq) FROM authorizations GROUP BY stage)"
    ).fetchall()
    if not rows:
        return None, 0
    latest = {str(row[0]): (float(row[1]), int(row[2])) for row in rows}
    ceiling = (
        latest[OPERATION_DOLLAR_AUTHORIZATION][0]
        if OPERATION_DOLLAR_AUTHORIZATION in latest
        else math.fsum(row[0] for row in latest.values())
    )
    allowance = (
        latest[OPERATION_UNPRICED_AUTHORIZATION][1]
        if OPERATION_UNPRICED_AUTHORIZATION in latest
        else sum(row[1] for row in latest.values())
    )
    return ceiling, allowance


def _accounted(connection: sqlite3.Connection) -> float:
    """Return settled charges plus every retained priced reservation."""
    return float(
        connection.execute(
            "SELECT COALESCE(SUM(charge_usd), 0) FROM reservations"
            " WHERE priced = 1 AND state != 'released'"
        ).fetchone()[0]
    )


def _carries_media(value: JsonValue) -> bool:
    """Return whether a body holds an image, audio, file or document part."""
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            kind = current.get("type")
            if (
                isinstance(kind, str) and kind in _MEDIA_PART_TYPES
            ) or _MEDIA_KEYS & current.keys():
                return True
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return False


def _positive_int(value: JsonValue) -> int | None:
    """Return a strictly positive JSON integer, or None."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def is_embedding_endpoint(url: str) -> bool:
    """Return whether the endpoint is an embedding route, whose requests bill no output."""
    path = urlsplit(url).path
    return path.endswith("/embeddings") or path.endswith((":embedContent", ":batchEmbedContents"))


def request_token_bounds(
    url: str, payload: JsonObject, tariff: AliasTariff
) -> tuple[int, int] | None:
    """Bound the billable input and output tokens of one exact request body.

    Input is at most the UTF-8 bytes of the serialized body, the same provider-neutral bound
    the text engine uses, capped by the context window. A body with media parts is bounded
    only by the context window. An embedding route applies its window to each input of a
    batch, not to the whole body, so its input is the uncapped body size and a media
    embedding body is unbounded. Output is the body's explicit allowance (times its requested
    choices), the tariff's default allowance, or zero for an embedding route.

    Args:
        url: Absolute provider endpoint.
        payload: Exact JSON request body.
        tariff: Prices and capacities of the alias.

    Returns:
        ``(input_tokens, output_tokens)``, or None when either is unbounded.
    """
    size = len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    context = tariff.context_window_tokens
    if is_embedding_endpoint(url):
        if _carries_media(payload):
            return None
        return tariff.repeated_input * (size + tariff.input_framing_tokens), 0
    if _carries_media(payload):
        if context is None:
            return None
        size = context
    elif context is not None:
        size = min(size, context)
    input_tokens = tariff.repeated_input * (size + tariff.input_framing_tokens)
    generation = payload.get("generationConfig")
    settings = generation if isinstance(generation, dict) else {}
    explicit = next(
        (
            allowance
            for allowance in (
                *(_positive_int(payload.get(name)) for name in _OUTPUT_FIELDS),
                _positive_int(settings.get("maxOutputTokens")),
            )
            if allowance is not None
        ),
        None,
    )
    output = explicit if explicit is not None else tariff.default_output_tokens
    if output is None:
        output = context if tariff.output_uses_context_bound else None
    if output is None:
        return None
    thinking = settings.get("thinkingConfig")
    if isinstance(thinking, dict):
        output += _positive_int(thinking.get("thinkingBudget")) or 0
    if tariff.output_uses_context_bound:
        # The authored model window bounds the entire generation, including hidden reasoning.
        # A requested visible-output limit is not evidence of a billed-total limit.
        if context is None:
            return None
        output = max(output, context)
    choices = _positive_int(payload.get("n")) or _positive_int(settings.get("candidateCount")) or 1
    return input_tokens, output * choices


def observed_token_usage(body: JsonObject, *, embedding: bool = False) -> tuple[int, int] | None:
    """Read billed input and output tokens from a provider body, or None when absent.

    Cache reads and writes count as input and reasoning as output, so settlement at the
    tariff's highest rates never undercounts a charge. Missing completion counters keep the
    reservation; only embedding endpoints can omit output counters and still settle usage.

    Args:
        body: Exact decoded provider response.
        embedding: The request used an embedding endpoint, so omitted output usage is zero.

    Returns:
        ``(input_tokens, output_tokens)``, or None when usage is missing or malformed.
    """

    def count(container: Mapping[str, JsonValue], *names: str) -> int | None:
        """Sum the named nonnegative integers; None when any present value is malformed."""
        total = 0
        for name in names:
            value = container.get(name)
            if name not in container:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return None
            total += value
        return total

    usage = body.get("usage")
    if isinstance(usage, dict):
        if "prompt_tokens" in usage:
            first = count(usage, "prompt_tokens")
            second = (
                count(usage, "completion_tokens")
                if "completion_tokens" in usage or embedding
                else None
            )
        elif "input_tokens" in usage:
            first = count(
                usage, "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"
            )
            second = (
                count(usage, "output_tokens") if "output_tokens" in usage or embedding else None
            )
        else:
            return None
        return None if first is None or second is None else (first, second)
    metadata = body.get("usageMetadata")
    if isinstance(metadata, dict) and "promptTokenCount" in metadata:
        first = count(metadata, "promptTokenCount", "toolUsePromptTokenCount")
        second = (
            count(metadata, "candidatesTokenCount", "thoughtsTokenCount")
            if "candidatesTokenCount" in metadata or embedding
            else None
        )
        return None if first is None or second is None else (first, second)
    return None


_SCOPE: ContextVar[list[OperationSpendCeiling | None] | None] = ContextVar(
    "operation_spend_scope", default=None
)


@contextmanager
def operation_spend_scope() -> Iterator[None]:
    """Open one operation's spend scope; consent installs its ceiling inside it.

    Admitted transports constructed inside the scope after installation enforce the
    installed ceiling. Outside any scope no ceiling is ambient.
    """
    token = _SCOPE.set([None])
    try:
        yield
    finally:
        _SCOPE.reset(token)


def install_spend_ceiling(ceiling: OperationSpendCeiling) -> None:
    """Make ``ceiling`` the operation's ambient ceiling for the rest of its scope.

    Raises:
        ValueError: No operation scope is open, or another ceiling is already installed.
    """
    holder = _SCOPE.get()
    if holder is None:
        raise ValueError("a spend ceiling needs an open operation scope")
    if holder[0] is not None and holder[0].path != ceiling.path:
        raise ValueError("an operation has exactly one spend ceiling")
    holder[0] = ceiling


def current_spend_ceiling() -> OperationSpendCeiling | None:
    """Return the ceiling installed in the current operation scope, if any."""
    holder = _SCOPE.get()
    return None if holder is None else holder[0]

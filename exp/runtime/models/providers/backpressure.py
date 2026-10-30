"""Host-wide provider admission: shared concurrency slots and per-provider circuit breakers.

Every process on one host that dispatches through the admitted transport shares one SQLite file
under the user-data directory (``provider-admission.sqlite`` beside ``auth.json``). It holds two
kinds of state:

* In-flight slots. A slot is claimed in a ``BEGIN IMMEDIATE`` transaction only when both the
  provider's and the alias's registered limits have room, held while one physical request is in
  flight, and deleted when it returns or fails. A slot whose holder process is gone (no such pid,
  a zombie, or a pid reused by a process with a different start time) is reclaimed by the next
  claimant, so a crash never leaks capacity.
* One breaker per provider. It opens after the registered number of consecutive overload
  responses (HTTP 429, 503 and 529), stays open for the cool-down or the server's Retry-After,
  whichever is longer, then admits exactly one probe. The probe's definite non-overload response
  closes the breaker; anything else reopens it. A 503 that refuses one exact request (see
  :data:`REQUEST_REFUSAL_CODES`) is definite, not overload.

A refusal, whether by an open breaker or by a slot wait that outlives the attempt's timeout,
raises :class:`ProviderAdmissionRefused` before any ledger claim or network I/O. It is a
:class:`~exp.common.models.client.RequestAdmissionRefused`, so durable callers record the
attempt as not dispatched and workflows stop resumably instead of scoring a miss.

Limits are registered in ``provider-admission.toml`` in the same directory. The file is optional;
absent settings use :data:`DEFAULT_ADMISSION_LIMITS`.
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys
import time
import tomllib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from exp.common.auth.paths import provider_data_dir
from exp.common.models.client import RequestAdmissionRefused
from exp.common.sqlite.connection import connect_database

ADMISSION_DATABASE_NAME: Final = "provider-admission.sqlite"
"""Shared host-wide admission database, beside the provider credential file."""

ADMISSION_LIMITS_NAME: Final = "provider-admission.toml"
"""Optional host-wide registered limits file, beside the admission database."""

SCHEMA_VERSION: Final = 1
"""Admission schema identity stored in ``PRAGMA user_version``."""

OVERLOAD_STATUS_CODES: Final = frozenset({429, 503, 529})
"""Statuses that mean the provider is shedding load: rate limited, unavailable, overloaded."""

REQUEST_REFUSAL_CODES: Final = frozenset({"invalid_provider_stream"})
"""Gateway ``error.code`` values that refuse one exact request rather than shed load.

An Experiential gateway answers HTTP 503 with ``invalid_provider_stream`` when the provider's
stream for this request cannot be encoded; resending it repeats the refusal, and other requests
are unaffected. Such a response is definite and is not overload evidence for the breaker.
"""


def is_overload(status_code: int | None, error_code: str | None = None) -> bool:
    """Return whether one definite response is evidence the provider is shedding load.

    Args:
        status_code: HTTP status, or None for an ambiguous outcome (never overload).
        error_code: Sanitized OpenAI-style ``error.code`` of the response body, if any.
    """
    return status_code in OVERLOAD_STATUS_CODES and error_code not in REQUEST_REFUSAL_CODES


_POLL_INITIAL_SECONDS: Final = 0.02
_POLL_MAXIMUM_SECONDS: Final = 0.5


class ProviderAdmissionRefused(RequestAdmissionRefused):
    """A provider request was refused on this host before any dispatch or ledger claim.

    Attributes:
        provider: Provider admission scope that refused.
        reason: ``circuit_open`` while the provider's breaker refuses new dispatches, or
            ``capacity_wait`` when no in-flight slot freed within the attempt's timeout.
        retry_after_seconds: Earliest useful wait before a resumed attempt, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        reason: Literal["circuit_open", "capacity_wait"],
        retry_after_seconds: float | None = None,
    ) -> None:
        """Keep the refusal's scope, reason and earliest useful retry time."""
        super().__init__(message)
        self.provider = provider
        self.reason = reason
        self.retry_after_seconds = retry_after_seconds


class AliasLimit(BaseModel):
    """Registered concurrency limit of one scope.

    Attributes:
        max_in_flight: Maximum concurrent physical requests on this host, at least one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_in_flight: int = Field(ge=1, strict=True)


class AdmissionLimits(BaseModel):
    """Registered host-wide provider admission limits.

    Attributes:
        provider_max_in_flight: Default in-flight limit of a provider without an override.
            Defaults to 8.
        alias_max_in_flight: Default in-flight limit of a model alias without an override.
            Defaults to 4, so one alias cannot take a provider's whole budget.
        breaker_threshold: Consecutive overload responses that open a provider's breaker.
            Defaults to 5.
        breaker_cooldown_seconds: Minimum time an open breaker refuses before one probe.
            Defaults to 30 seconds; a longer server Retry-After extends it.
        providers: Per-provider overrides keyed by provider name.
        aliases: Per-alias overrides keyed by catalog alias.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_max_in_flight: int = Field(default=8, ge=1, strict=True)
    alias_max_in_flight: int = Field(default=4, ge=1, strict=True)
    breaker_threshold: int = Field(default=5, ge=1, strict=True)
    breaker_cooldown_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    providers: Mapping[str, AliasLimit] = Field(default_factory=dict)
    aliases: Mapping[str, AliasLimit] = Field(default_factory=dict)

    def provider_limit(self, provider: str) -> int:
        """Return the registered in-flight limit of one provider."""
        override = self.providers.get(provider)
        return self.provider_max_in_flight if override is None else override.max_in_flight

    def alias_limit(self, alias: str) -> int:
        """Return the registered in-flight limit of one alias."""
        override = self.aliases.get(alias)
        return self.alias_max_in_flight if override is None else override.max_in_flight


DEFAULT_ADMISSION_LIMITS: Final = AdmissionLimits()
"""Conservative limits used when the host registers none."""


def load_admission_limits(path: Path) -> AdmissionLimits:
    """Read registered limits from TOML, or the defaults when the file is absent.

    Args:
        path: ``provider-admission.toml`` path.

    Returns:
        The validated limits.

    Raises:
        ValueError: The file is not valid TOML or names an unknown or invalid setting.
    """
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        return DEFAULT_ADMISSION_LIMITS
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{path} is not valid TOML; fix or delete it") from error
    return AdmissionLimits.model_validate(data)


@dataclass(frozen=True)
class ProviderRoute:
    """The admission scopes of one physical request.

    Attributes:
        provider: Registry provider name, such as ``openrouter``; one breaker and one limit.
        alias: Catalog alias, limited separately inside its provider's budget, or None.
    """

    provider: str
    alias: str | None = None


@dataclass
class AdmissionSlot:
    """One held in-flight slot and the outcome its request reported.

    Attributes:
        waited_seconds: Time spent waiting for the slot; exactly zero when the first claim
            succeeded, so an uncontended request keeps its whole timeout.
        recorded: Whether :meth:`record` was called.
        status_code: Recorded definite HTTP status, or None for an ambiguous failure.
        retry_after_seconds: Recorded sanitized server minimum wait.
        error_code: Recorded sanitized ``error.code`` of the response body.
    """

    waited_seconds: float
    recorded: bool = False
    status_code: int | None = None
    retry_after_seconds: float | None = None
    error_code: str | None = None

    def record(
        self,
        *,
        status_code: int | None,
        retry_after_seconds: float | None,
        error_code: str | None = None,
    ) -> None:
        """Report the request's outcome, applied to the breaker when the slot is released."""
        self.recorded = True
        self.status_code, self.retry_after_seconds = status_code, retry_after_seconds
        self.error_code = error_code


def default_admission_path() -> Path:
    """Return the host-wide admission database path in the user-data directory."""
    return provider_data_dir() / ADMISSION_DATABASE_NAME


def process_start(pid: int) -> str | None:
    """Return a Linux process start-time token, or None where ``/proc`` is unavailable."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name is parenthesized and may contain spaces; fields follow its last ")".
    fields = stat[stat.rfind(")") + 2 :].split()
    return fields[19] if len(fields) > 19 else None


def _process_state(pid: int) -> str | None:
    """Return the Linux scheduler state letter of a process, or None when unknown."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat[stat.rfind(")") + 2 :].split()
    return fields[0] if fields else None


def holder_alive(pid: int, start: str | None) -> bool:
    """Return whether the process that took a slot is still the live holder.

    Args:
        pid: Holder process id.
        start: Holder start-time token recorded at claim, when the platform has one.

    Returns:
        False for a missing pid, a zombie, or a pid now owned by a later process.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if _process_state(pid) in {"Z", "X"}:
        return False
    return start is None or process_start(pid) in {None, start}


class HostProviderAdmission:
    """Shared, crash-safe concurrency slots and breakers for one host.

    Args:
        path: SQLite file shared by every process on the host.
        limits: Registered limits this process applies.
        clock: Wall clock shared across processes for breaker timing.
        sleep: Wait between slot polls, injectable for tests.
    """

    def __init__(
        self,
        path: Path,
        limits: AdmissionLimits = DEFAULT_ADMISSION_LIMITS,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create or open the admission database and validate its schema.

        Raises:
            ValueError: The file exists with tables of another schema.
        """
        self._path, self._limits = path, limits
        self._clock, self._sleep = clock, sleep
        self._pid = os.getpid()
        self._start = process_start(self._pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = connect_database(path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version == 0:
                    connection.execute(
                        """CREATE TABLE IF NOT EXISTS slots (
                            token TEXT PRIMARY KEY,
                            provider TEXT NOT NULL,
                            alias TEXT,
                            pid INTEGER NOT NULL,
                            start TEXT,
                            acquired_at REAL NOT NULL
                        ) STRICT"""
                    )
                    connection.execute(
                        """CREATE TABLE IF NOT EXISTS breakers (
                            provider TEXT PRIMARY KEY,
                            state TEXT NOT NULL CHECK (state IN ('closed', 'open', 'probing')),
                            consecutive INTEGER NOT NULL,
                            open_until REAL NOT NULL,
                            probe_pid INTEGER,
                            probe_start TEXT
                        ) STRICT"""
                    )
                    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                elif version != SCHEMA_VERSION:
                    raise ValueError(
                        f"provider admission database {path} has schema {version}, not "
                        f"{SCHEMA_VERSION}; delete it while no exp process runs"
                    )
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        """Open the shared database without a per-commit fsync.

        Slots and breaker rows describe in-flight work only, so losing the last commit to a
        power failure costs nothing a reboot does not already reset; WAL keeps it consistent.
        """
        connection = connect_database(self._path)
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @classmethod
    def for_host(cls) -> HostProviderAdmission:
        """Open the host-wide database with the host's registered limits."""
        directory = provider_data_dir()
        return cls(
            directory / ADMISSION_DATABASE_NAME,
            load_admission_limits(directory / ADMISSION_LIMITS_NAME),
        )

    @property
    def limits(self) -> AdmissionLimits:
        """Return the registered limits this process applies."""
        return self._limits

    def in_flight(self, route: ProviderRoute) -> tuple[int, int]:
        """Return live provider and alias slot counts, reclaiming dead holders first."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._reclaim(connection)
                counts = self._counts(connection, route)
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
            return counts
        finally:
            connection.close()

    def breaker_state(self, provider: str) -> str:
        """Return ``closed``, ``open`` or ``probing`` for one provider."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT state FROM breakers WHERE provider = ?", (provider,)
            ).fetchone()
            return "closed" if row is None else str(row[0])
        finally:
            connection.close()

    @contextmanager
    def admit(self, route: ProviderRoute, *, wait_seconds: float) -> Iterator[AdmissionSlot]:
        """Hold one in-flight slot for the duration of a single physical request.

        Args:
            route: Provider and alias scopes of the request.
            wait_seconds: Longest time to wait for a slot; the attempt's own timeout.

        Yields:
            The held slot. Its outcome, when recorded, updates the breaker in the same
            transaction that releases the slot.

        Raises:
            ProviderAdmissionRefused: The breaker is open or no slot freed in time.
        """
        if not math.isfinite(wait_seconds) or wait_seconds <= 0:
            raise ValueError("admission wait must be finite and positive")
        started = time.monotonic()
        poll = _POLL_INITIAL_SECONDS
        contended = False
        while True:
            token, probing = self._try_claim(route)
            if token is not None:
                break
            contended = True
            waited = time.monotonic() - started
            if waited + poll > wait_seconds:
                raise ProviderAdmissionRefused(
                    f"no {route.provider} in-flight slot freed within {wait_seconds:.1f}s on "
                    "this host; nothing was dispatched, so resume the run once load drops or "
                    f"raise the limit in {ADMISSION_LIMITS_NAME}",
                    provider=route.provider,
                    reason="capacity_wait",
                )
            self._sleep(poll)
            poll = min(poll * 2, _POLL_MAXIMUM_SECONDS)
        slot = AdmissionSlot(waited_seconds=time.monotonic() - started if contended else 0.0)
        try:
            yield slot
        finally:
            self._release(token, route.provider, slot, probing=probing)

    def observe(
        self,
        provider: str,
        *,
        status_code: int | None,
        retry_after_seconds: float | None = None,
        error_code: str | None = None,
    ) -> None:
        """Update one provider's breaker with a physical attempt's outcome.

        Args:
            provider: Provider admission scope.
            status_code: Definite HTTP status, or None for an ambiguous transport failure.
            retry_after_seconds: Sanitized server minimum wait from an overload response.
            error_code: Sanitized ``error.code`` of the response body, if any.
        """
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._apply(connection, provider, status_code, retry_after_seconds, error_code)
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
        finally:
            connection.close()

    def _apply(
        self,
        connection: sqlite3.Connection,
        provider: str,
        status_code: int | None,
        retry_after_seconds: float | None,
        error_code: str | None = None,
    ) -> None:
        """Advance one breaker: definite non-overload closes, overload counts, probes decide.

        An ambiguous outcome (no status) leaves a closed breaker untouched and reopens a probe.
        """
        row = connection.execute(
            "SELECT state, consecutive FROM breakers WHERE provider = ?", (provider,)
        ).fetchone()
        state = "closed" if row is None else str(row[0])
        consecutive = 0 if row is None else int(row[1])
        if status_code is not None and not is_overload(status_code, error_code):
            self._write_breaker(connection, provider, "closed", 0, 0.0)
        elif status_code is not None or state == "probing":
            consecutive += 1
            if state == "probing" or consecutive >= self._limits.breaker_threshold:
                cooldown = max(self._limits.breaker_cooldown_seconds, retry_after_seconds or 0.0)
                until = self._clock() + cooldown
                self._write_breaker(connection, provider, "open", consecutive, until)
            else:
                self._write_breaker(connection, provider, state, consecutive, 0.0)

    def _try_claim(self, route: ProviderRoute) -> tuple[str | None, bool]:
        """Claim one slot when the breaker admits and both limits have room.

        Returns:
            The slot token, or None when the caller should wait, and whether this claim is the
            breaker's single cool-down probe.

        Raises:
            ProviderAdmissionRefused: The provider's breaker refuses new dispatches.
        """
        now = self._clock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._reclaim(connection)
                probing = self._breaker_admits(connection, route.provider, now)
                provider_count, alias_count = self._counts(connection, route)
                room = provider_count < self._limits.provider_limit(route.provider) and (
                    route.alias is None or alias_count < self._limits.alias_limit(route.alias)
                )
                token = None
                if room:
                    token = uuid4().hex
                    connection.execute(
                        "INSERT INTO slots VALUES (?, ?, ?, ?, ?, ?)",
                        (token, route.provider, route.alias, self._pid, self._start, now),
                    )
                    if probing:
                        connection.execute(
                            "UPDATE breakers SET state = 'probing', probe_pid = ?,"
                            " probe_start = ? WHERE provider = ?",
                            (self._pid, self._start, route.provider),
                        )
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
            return token, probing and token is not None
        finally:
            connection.close()

    def _breaker_admits(self, connection: sqlite3.Connection, provider: str, now: float) -> bool:
        """Refuse while open or probing; return True when this claim becomes the probe."""
        row = connection.execute(
            "SELECT state, open_until, probe_pid, probe_start FROM breakers WHERE provider = ?",
            (provider,),
        ).fetchone()
        if row is None or row[0] == "closed":
            return False
        state, open_until = str(row[0]), float(row[1])
        if state == "probing" and not holder_alive(int(row[2]), row[3]):
            state = "open"
        if state == "open" and now >= open_until:
            return True
        raise ProviderAdmissionRefused(
            f"{provider} circuit breaker is open after repeated overload responses; nothing "
            "was dispatched, so resume the run after the cool-down",
            provider=provider,
            reason="circuit_open",
            retry_after_seconds=max(open_until - now, 0.0),
        )

    def _release(self, token: str, provider: str, slot: AdmissionSlot, *, probing: bool) -> None:
        """Delete one held slot and apply its outcome; an unobserved probe reopens."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("DELETE FROM slots WHERE token = ?", (token,))
                if slot.recorded:
                    self._apply(
                        connection,
                        provider,
                        slot.status_code,
                        slot.retry_after_seconds,
                        slot.error_code,
                    )
                elif probing:
                    self._apply(connection, provider, None, None)
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
        finally:
            connection.close()

    @staticmethod
    def _write_breaker(
        connection: sqlite3.Connection,
        provider: str,
        state: str,
        consecutive: int,
        open_until: float,
    ) -> None:
        """Replace one provider's breaker row, clearing any probe holder."""
        connection.execute(
            "INSERT OR REPLACE INTO breakers VALUES (?, ?, ?, ?, NULL, NULL)",
            (provider, state, consecutive, open_until),
        )

    @staticmethod
    def _reclaim(connection: sqlite3.Connection) -> None:
        """Delete slots whose holder process is no longer alive."""
        rows = connection.execute("SELECT token, pid, start FROM slots").fetchall()
        for token, pid, start in rows:
            if not holder_alive(int(pid), start):
                connection.execute("DELETE FROM slots WHERE token = ?", (token,))

    @staticmethod
    def _counts(connection: sqlite3.Connection, route: ProviderRoute) -> tuple[int, int]:
        """Count live slots of the route's provider and alias."""
        provider = int(
            connection.execute(
                "SELECT COUNT(*) FROM slots WHERE provider = ?", (route.provider,)
            ).fetchone()[0]
        )
        alias = int(
            connection.execute(
                "SELECT COUNT(*) FROM slots WHERE provider = ? AND alias IS ?",
                (route.provider, route.alias),
            ).fetchone()[0]
        )
        return provider, alias

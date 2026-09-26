"""Durable request admission and exact response replay under an approved spend limit."""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from exp.common.core.artifacts import sha256_json


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
    Unknown crash or failure charges retain their full reservation and never replay silently.
    No credentials or request bodies are saved. Response payloads stay in the local run directory.
    """

    def __init__(self, directory: Path, *, identity: str, maximum_cost_usd: float) -> None:
        """Bind a local ledger to immutable execution identity and explicit authorization.

        Args:
            directory: Private local runtime directory owning response receipts.
            identity: Digest of immutable models, tasks, prompts, prices and execution settings.
            maximum_cost_usd: Total approved allowance, including completed and unknown calls.

        Raises:
            ValueError: Authorization is invalid or the ledger belongs to another execution.
        """
        if not math.isfinite(maximum_cost_usd) or maximum_cost_usd <= 0:
            raise ValueError("spending limit must be finite and positive")
        directory.mkdir(parents=True, exist_ok=True)
        self._path = directory / "requests.sqlite3"
        self._limit = maximum_cost_usd
        self._condition = threading.Condition()
        self._active: set[str] = set()
        self._scope: ContextVar[tuple[str, dict[str, int]] | None] = ContextVar(
            f"request-budget-{identity}", default=None
        )
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS identity (digest TEXT NOT NULL)")
            row = db.execute("SELECT digest FROM identity").fetchone()
            if row is None:
                db.execute("INSERT INTO identity VALUES (?)", (identity,))
            elif row[0] != identity:
                raise ValueError("saved request ledger belongs to different execution settings")
            db.execute(
                "CREATE TABLE IF NOT EXISTS requests ("
                "key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, "
                "charge REAL NOT NULL, response TEXT, state TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS authorizations ("
                "created_at TEXT DEFAULT CURRENT_TIMESTAMP, limit_usd REAL NOT NULL)"
            )
            db.execute("INSERT INTO authorizations (limit_usd) VALUES (?)", (self._limit,))
        self._path.chmod(0o600)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Serialize accounting transactions; release the database before provider I/O."""
        db = sqlite3.connect(self._path, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def scope(self, identity: str) -> Iterator[None]:
        """Reset per-role ordinals for one exact episode attempt or one rollout judgment."""
        token = self._scope.set((identity, {}))
        try:
            yield
        finally:
            self._scope.reset(token)

    @property
    def accounted_usd(self) -> float:
        """Return completed charges plus conservative reservations for unknown dispatches."""
        with self._connect() as db:
            return float(db.execute("SELECT COALESCE(SUM(charge), 0) FROM requests").fetchone()[0])

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
        charges = []
        with self._connect() as db:
            for offset in range(0, len(keys), 500):
                batch = keys[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                charges.extend(
                    row[0]
                    for row in db.execute(
                        f"SELECT charge FROM requests WHERE key IN ({placeholders})", batch
                    )
                )
        return math.fsum(charges)

    def call[ResultT](
        self,
        *,
        role: str,
        fingerprint: str,
        maximum_cost_usd: float,
        operation: Callable[[], ResultT],
        encode: Callable[[ResultT], str],
        decode: Callable[[str], ResultT],
        charge: Callable[[ResultT], float],
    ) -> ResultT:
        """Reserve an exact request, replay a saved answer, or pause before dispatch.

        Args:
            role: Distinguishes assistant aliases, world model, embedder and judge.
            fingerprint: Exact request, model, reservation and execution digest.
            maximum_cost_usd: Retry-inclusive bound for this pending provider call.
            operation: Provider call, executed outside the transaction.
            encode: Serialize the successful result for exact replay.
            decode: Restore the successful result without contacting a provider.
            charge: Reconcile actual usage and any unresolved retry charges.

        Returns:
            A new or exactly replayed response. Replays spend no additional allowance.

        Raises:
            SpendLimitReached: No in-flight request can release enough allowance.
            ValueError: A saved coordinate drifted, has unknown spend, or violates its bound.
            Exception: The provider failed; its full reservation remains durably charged.
        """
        if not math.isfinite(maximum_cost_usd) or maximum_cost_usd < 0:
            raise ValueError("request reservation must be finite and nonnegative")
        scope = self._scope.get()
        if scope is None:
            raise ValueError("paid request requires an explicit execution scope")
        identity, ordinals = scope
        ordinal = ordinals.get(role, 0)
        ordinals[role] = ordinal + 1
        key = sha256_json({"scope": identity, "role": role, "ordinal": ordinal})
        cached = self._reserve(key, fingerprint, maximum_cost_usd)
        if cached is not None:
            return decode(cached)
        try:
            result = operation()
            cost = charge(result)
            if not math.isfinite(cost) or cost < 0 or cost > maximum_cost_usd + 1e-9:
                raise ValueError("provider charge exceeds the admitted request reservation")
            payload = encode(result)
            with self._condition, self._connect() as db:
                db.execute(
                    "UPDATE requests SET charge=?, response=?, state='complete' WHERE key=?",
                    (cost, payload, key),
                )
            return result
        except BaseException:
            with self._connect() as db:
                db.execute("UPDATE requests SET state='unknown' WHERE key=?", (key,))
            raise
        finally:
            with self._condition:
                self._active.discard(key)
                self._condition.notify_all()

    def _reserve(self, key: str, fingerprint: str, maximum: float) -> str | None:
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
                with self._connect() as db:
                    row = db.execute(
                        "SELECT fingerprint, response, state FROM requests WHERE key=?", (key,)
                    ).fetchone()
                    if row is not None:
                        if row[0] != fingerprint:
                            raise ValueError(
                                "saved provider request changed; start a new evaluation"
                            )
                        if row[2] != "complete":
                            raise ValueError(
                                "saved provider dispatch has unresolved spend; not replayed"
                            )
                        return str(row[1])
                    spent = float(
                        db.execute("SELECT COALESCE(SUM(charge), 0) FROM requests").fetchone()[0]
                    )
                    if spent + maximum <= self._limit + 1e-9:
                        db.execute(
                            "INSERT INTO requests VALUES (?, ?, ?, NULL, 'pending')",
                            (key, fingerprint, maximum),
                        )
                        self._active.add(key)
                        return None
                if not self._active:
                    raise SpendLimitReached(self._limit, spent, spent + maximum)
                self._condition.wait(timeout=1)

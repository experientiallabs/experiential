"""Additive parent envelopes over one physical spend ledger, preserving legacy authority.

Existing claims stay untouched. New scoped claims are associated in an append-only side table
in the same SQLite transaction as physical reservation. This low-level budget primitive does
not qualify a preparation, runtime, study, or completion: those native domain checks must bind
the authorization and inventory references before registration.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256, sha256_json
from exp.common.models.client import RequestAdmissionRefused
from exp.common.project import ProjectStore, artifact_input

if TYPE_CHECKING:
    from exp.runtime.models.spend_ceiling import OperationSpendCeiling

MAXIMUM_PARENT_ATTEMPTS = 100_000


class ScopeBudgetRefused(RequestAdmissionRefused):
    """A parent, child, envelope, or operational scope cannot admit a physical reservation."""


class ParentSpendPolicy(ContractModel):
    """Exact host-consented parent policy, separate from existing legacy authorizations.

    Attributes:
        operation: Original operation, never a replacement root.
        legacy_attempt_limit: Unchanged legacy physical-request cap.
        parent_attempt_limit: Whole-operation physical cap, including every legacy claim.
        maximum_known_usd: Original known-price/account ceiling, never raised by this policy.
        deadline: Finite scoped dispatch cutoff.
        aliases: Exact configured native role aliases allowed to use scoped headroom.
        authorization: Immutable typed host approval; domain registration validates its meaning.
        preparation: Genuine completed preparation/required-QA provenance, validated by domain.
        execution: Exact reviewed study/source/protocol registration, validated by domain.
        producer_revision: Actual reviewed executor allowed to certify native completion.
    """

    protocol: Literal["scoped-parent-spend-v1"] = "scoped-parent-spend-v1"
    operation: str = Field(min_length=1)
    legacy_attempt_limit: int = Field(ge=0, le=MAXIMUM_PARENT_ATTEMPTS, strict=True)
    parent_attempt_limit: int = Field(ge=1, le=MAXIMUM_PARENT_ATTEMPTS, strict=True)
    maximum_known_usd: float = Field(ge=0)
    deadline: AwareDatetime
    aliases: tuple[str, ...] = Field(min_length=1)
    authorization: ArtifactInput
    preparation: ArtifactInput
    execution: ArtifactInput
    producer_revision: str = Field(min_length=1)


class _ParentRecord(ContractModel):
    """Registration-time immutable audit of original authority and retained liabilities."""

    protocol: Literal["scoped-parent-audit-v1"] = "scoped-parent-audit-v1"
    policy: ParentSpendPolicy
    ledger_sha256: Sha256
    legacy_authorizations: tuple[JsonObject, ...]
    legacy_cutoff_seq: int
    legacy_retained_claims: int
    legacy_claims_sha256: Sha256


def _grants(connection: sqlite3.Connection) -> tuple[JsonObject, ...]:
    """Credential-free immutable original consent prefix, including exact native sequence."""
    return tuple(
        {
            "seq": row[0],
            "stage": row[1],
            "ceiling_usd": row[2],
            "unpriced_attempts": row[3],
            "authorized_at": row[4],
        }
        for row in connection.execute(
            "SELECT seq,stage,ceiling_usd,unpriced_attempts,authorized_at "
            "FROM authorizations ORDER BY seq"
        )
    )


class ParentSpendRef(ContractModel):
    """Immutable parent policy identity in this exact original physical ledger."""

    identity: Sha256
    ledger_sha256: Sha256


class SpendEnvelopePlan(ContractModel):
    """Full native current-wave bound plus mandatory future logical request floor.

    Bounds must come from the actual native planner, not model output or private arithmetic.
    The budget primitive deliberately cannot decide domain admission or task correctness.
    """

    protocol: Literal["native-spend-envelope-v1"] = "native-spend-envelope-v1"
    policy: ParentSpendRef
    scope: str = Field(min_length=1, max_length=256)
    inventory: ArtifactInput
    maximum_attempts: int = Field(ge=1, le=MAXIMUM_PARENT_ATTEMPTS, strict=True)
    maximum_known_usd: float = Field(ge=0)
    remaining_minimum_attempts: int = Field(ge=0, le=MAXIMUM_PARENT_ATTEMPTS, strict=True)
    requirements_sha256: Sha256


class SpendEnvelopeRef(ContractModel):
    """Exact registered envelope; no request can choose another policy, scope, or inventory."""

    identity: Sha256
    policy: ParentSpendRef


def _ledger(ceiling: OperationSpendCeiling) -> Sha256:
    """Bind authority to the actual existing ledger path, without copying claims."""
    return sha256_json({"ledger_path": str(ceiling.path.resolve())})


def _present(connection: sqlite3.Connection) -> bool:
    """Whether explicit parent authority was registered in this original ledger."""
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scoped_parent_policies'"
        ).fetchone()
        is not None
    )


def _schema(connection: sqlite3.Connection) -> None:
    """Create additive append-only authority tables without editing any original row."""
    for name, columns in (
        ("scoped_parent_policies", "identity TEXT PRIMARY KEY, body TEXT NOT NULL"),
        (
            "scoped_envelopes",
            "identity TEXT PRIMARY KEY, policy TEXT NOT NULL "
            "REFERENCES scoped_parent_policies(identity), body TEXT NOT NULL",
        ),
        (
            "scoped_claims",
            "reservation TEXT PRIMARY KEY REFERENCES reservations(id), "
            "envelope TEXT NOT NULL REFERENCES scoped_envelopes(identity)",
        ),
        (
            "scoped_envelope_settlements",
            "envelope TEXT PRIMARY KEY REFERENCES scoped_envelopes(identity), "
            "completion TEXT NOT NULL",
        ),
    ):
        connection.execute(f"CREATE TABLE IF NOT EXISTS {name} ({columns}) STRICT")
        for action in ("UPDATE", "DELETE"):
            connection.execute(
                f"CREATE TRIGGER IF NOT EXISTS {name}_{action.lower()} "
                f"BEFORE {action} ON {name} BEGIN "
                "SELECT RAISE(ABORT, 'scoped spending records are append-only'); END"
            )


def register_parent_policy(
    ceiling: OperationSpendCeiling,
    policy: ParentSpendPolicy,
) -> ParentSpendRef:
    """Append explicit parent consent while preserving every original grant and paid claim.

    The host's domain registration must validate the referenced preparation, QA, frozen study,
    producer, and user approval first. This method grants no new legacy allowance or dollars.
    """
    from exp.runtime.models.spend_ceiling import _authorized

    if policy.parent_attempt_limit < policy.legacy_attempt_limit:
        raise ValueError("parent cannot be smaller than the retained legacy grant")
    if datetime.now(UTC) >= policy.deadline:
        raise ValueError("scoped parent policy is already expired")
    if len(set(policy.aliases)) != len(policy.aliases) or any(
        not alias for alias in policy.aliases
    ):
        raise ValueError("scoped aliases must be distinct and nonempty")
    with ceiling._transaction() as connection:
        known, legacy = _authorized(connection)
        if legacy != policy.legacy_attempt_limit or known != policy.maximum_known_usd:
            raise ValueError("parent policy must retain the exact original legacy authority")
        _schema(connection)
        previous = connection.execute("SELECT identity,body FROM scoped_parent_policies").fetchall()
        if previous:
            if len(previous) != 1:
                raise ValueError("operation has conflicting parent registrations")
            record = _ParentRecord.model_validate_json(previous[0][1])
            if record.policy != policy or record.ledger_sha256 != _ledger(ceiling):
                raise ValueError("this operation already has another immutable parent policy")
            if sha256_json(record) != previous[0][0]:
                raise ValueError("parent registration audit identity changed")
            _policy(connection, previous[0][0])
            return ParentSpendRef(identity=previous[0][0], ledger_sha256=_ledger(ceiling))
        used, charged = _counts(connection)
        if used > policy.legacy_attempt_limit or charged > policy.maximum_known_usd + 1e-9:
            raise ScopeBudgetRefused("retained original claims exceed the original legacy grant")
        if connection.execute(
            "SELECT COUNT(*) FROM reservations WHERE state IN ('reserved','dispatched')"
        ).fetchone()[0]:
            raise ScopeBudgetRefused("original preparation must settle before parent registration")
        rows = connection.execute(
            "SELECT id,seq,role,priced,reserved_usd,charge_usd,state,overrun,attempt_id "
            "FROM reservations ORDER BY seq"
        ).fetchall()
        record = _ParentRecord(
            policy=policy,
            ledger_sha256=_ledger(ceiling),
            legacy_authorizations=_grants(connection),
            legacy_cutoff_seq=max((row[1] for row in rows), default=0),
            legacy_retained_claims=used,
            legacy_claims_sha256=sha256_json([tuple(row) for row in rows]),
        )
        identity = sha256_json(record)
        connection.execute(
            "INSERT INTO scoped_parent_policies VALUES (?, ?)", (identity, record.model_dump_json())
        )
    return ParentSpendRef(identity=identity, ledger_sha256=_ledger(ceiling))


def _policy(connection: sqlite3.Connection, identity: Sha256 | None = None) -> ParentSpendPolicy:
    """Read the sole immutable native parent policy."""
    rows = connection.execute("SELECT identity, body FROM scoped_parent_policies").fetchall()
    if len(rows) != 1 or (identity is not None and rows[0][0] != identity):
        raise ValueError("scope does not name this operation's exact parent policy")
    record = _ParentRecord.model_validate_json(rows[0][1])
    if sha256_json(record) != rows[0][0]:
        raise ValueError("parent policy audit identity changed")
    if _grants(connection)[: len(record.legacy_authorizations)] != record.legacy_authorizations:
        raise ValueError("original authorization history was altered")
    from exp.runtime.models.spend_ceiling import _authorized

    known, legacy = _authorized(connection)
    if legacy != record.policy.legacy_attempt_limit or known != record.policy.maximum_known_usd:
        raise ScopeBudgetRefused("scoped authority cannot alter the original legacy grant")
    return record.policy


def _plan(connection: sqlite3.Connection, reference: SpendEnvelopeRef) -> SpendEnvelopePlan:
    """Read and verify an exact existing open envelope."""
    row = connection.execute(
        "SELECT policy, body FROM scoped_envelopes WHERE identity=?", (reference.identity,)
    ).fetchone()
    if row is None or row[0] != reference.policy.identity:
        raise ValueError("scope envelope is not registered in this parent")
    plan = SpendEnvelopePlan.model_validate_json(row[1])
    if sha256_json(plan) != reference.identity or plan.policy != reference.policy:
        raise ValueError("scope envelope identity or provenance changed")
    if connection.execute(
        "SELECT 1 FROM scoped_envelope_settlements WHERE envelope=?", (reference.identity,)
    ).fetchone():
        raise ScopeBudgetRefused("scope envelope is already settled")
    return plan


def _counts(connection: sqlite3.Connection, envelope: str | None = None) -> tuple[int, float]:
    """Retained physical liabilities; released unsent reservations are excluded."""
    where = "r.state != 'released'"
    parameters: tuple[str, ...] = ()
    if envelope is not None:
        where += " AND s.envelope=?"
        parameters = (envelope,)
    return connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN r.priced=1 THEN r.charge_usd ELSE 0 END),0) "
        "FROM reservations r LEFT JOIN scoped_claims s ON s.reservation=r.id WHERE " + where,
        parameters,
    ).fetchone()


def _booked(
    connection: sqlite3.Connection,
    *,
    include_floor: bool = True,
) -> tuple[int, float]:
    """Unconsumed bounds of every open envelope, including orphaned owners."""
    attempts, known = 0, 0.0
    rows = connection.execute(
        "SELECT e.identity,e.body FROM scoped_envelopes e "
        "LEFT JOIN scoped_envelope_settlements s ON s.envelope=e.identity WHERE s.envelope IS NULL"
    ).fetchall()
    for identity, body in rows:
        plan = SpendEnvelopePlan.model_validate_json(body)
        used, charged = _counts(connection, identity)
        attempts += max(0, plan.maximum_attempts - used)
        known += max(0.0, plan.maximum_known_usd - charged)
    latest = connection.execute(
        "SELECT body FROM scoped_envelopes ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    if include_floor and latest is not None:
        attempts += SpendEnvelopePlan.model_validate_json(latest[0]).remaining_minimum_attempts
    return attempts, known


def open_spend_envelope(
    ceiling: OperationSpendCeiling,
    plan: SpendEnvelopePlan,
) -> SpendEnvelopeRef:
    """Atomically reserve one native wave against original claims and all outstanding waves."""
    if plan.policy.ledger_sha256 != _ledger(ceiling):
        raise ValueError("envelope belongs to a different original ledger")
    identity = sha256_json(plan)
    reference = SpendEnvelopeRef(identity=identity, policy=plan.policy)
    with ceiling._transaction() as connection:
        policy = _policy(connection, plan.policy.identity)
        if datetime.now(UTC) >= policy.deadline:
            raise ScopeBudgetRefused("parent scope deadline expired")
        previous = connection.execute(
            "SELECT body FROM scoped_envelopes WHERE identity=?", (identity,)
        ).fetchone()
        if previous is not None:
            _plan(connection, reference)
            return reference
        scopes = connection.execute(
            "SELECT body FROM scoped_envelopes WHERE policy=?", (plan.policy.identity,)
        ).fetchall()
        if any(SpendEnvelopePlan.model_validate_json(row[0]).scope == plan.scope for row in scopes):
            raise ValueError("a native scope already has another immutable envelope inventory")
        used, charged = _counts(connection)
        booked, held = _booked(connection, include_floor=False)
        if (
            used + booked + plan.maximum_attempts + plan.remaining_minimum_attempts
            > policy.parent_attempt_limit
        ):
            raise ScopeBudgetRefused(
                "native wave and remaining mandatory study exceed parent headroom"
            )
        if charged + held + plan.maximum_known_usd > policy.maximum_known_usd + 1e-9:
            raise ScopeBudgetRefused("native wave exceeds original known-price parent authority")
        connection.execute(
            "INSERT INTO scoped_envelopes VALUES (?, ?, ?)",
            (identity, plan.policy.identity, plan.model_dump_json()),
        )
    return reference


def scoped_spend_ceiling(
    ceiling: OperationSpendCeiling,
    envelope: SpendEnvelopeRef,
) -> OperationSpendCeiling:
    """Return the same native physical-reservation implementation with an exact scoped lease."""
    from exp.runtime.models.spend_ceiling import OperationSpendCeiling

    if envelope.policy.ledger_sha256 != _ledger(ceiling):
        raise ValueError("scoped ceiling cannot switch the original operation ledger")
    with ceiling._transaction() as connection:
        _plan(connection, envelope)
    return OperationSpendCeiling(
        ceiling.path,
        tariffs=ceiling._tariffs,
        maximum_wait_seconds=ceiling._maximum_wait,
        sleep=ceiling._sleep,
        envelope=envelope,
    )


def allowance(
    connection: sqlite3.Connection, envelope: SpendEnvelopeRef | None, legacy: int
) -> int:
    """Use parent allowance only for a genuine exact scoped lease."""
    return (
        legacy
        if envelope is None
        else _policy(connection, envelope.policy.identity).parent_attempt_limit
    )


def unpriced_used(connection: sqlite3.Connection, envelope: SpendEnvelopeRef | None) -> int:
    """Keep legacy claims in their original child scope without relabeling old reservations."""
    if not _present(connection) or envelope is not None:
        return connection.execute(
            "SELECT COUNT(*) FROM reservations WHERE priced=0 AND state!='released'"
        ).fetchone()[0]
    return connection.execute(
        "SELECT COUNT(*) FROM reservations r LEFT JOIN scoped_claims s ON s.reservation=r.id "
        "WHERE r.priced=0 AND r.state!='released' AND s.reservation IS NULL"
    ).fetchone()[0]


def check_reservation(
    connection: sqlite3.Connection,
    envelope: SpendEnvelopeRef | None,
    alias: str | None,
    cost: float | None,
) -> None:
    """Enforce physical parent/legacy/envelope capacity in the insertion transaction."""
    if not _present(connection):
        if envelope is not None:
            raise ValueError("scoped reservation has no native parent authority")
        return
    policy = _policy(connection, None if envelope is None else envelope.policy.identity)
    used, charged = _counts(connection)
    booked, held = _booked(connection)
    if envelope is None:
        legacy_used = connection.execute(
            "SELECT COUNT(*) FROM reservations r LEFT JOIN scoped_claims s ON s.reservation=r.id "
            "WHERE r.state!='released' AND s.reservation IS NULL"
        ).fetchone()[0]
        if legacy_used >= policy.legacy_attempt_limit:
            raise ScopeBudgetRefused("original legacy preparation cap is exhausted")
        if used + booked + 1 > policy.parent_attempt_limit:
            raise ScopeBudgetRefused("legacy request cannot consume encumbered parent headroom")
        if charged + held + (cost or 0) > policy.maximum_known_usd + 1e-9:
            raise ScopeBudgetRefused("legacy request cannot consume encumbered parent dollars")
    else:
        if datetime.now(UTC) >= policy.deadline:
            raise ScopeBudgetRefused("parent scope deadline expired")
        if alias not in policy.aliases:
            raise ScopeBudgetRefused("alias is outside the exact registered scope")
        plan = _plan(connection, envelope)
        owned, paid = _counts(connection, envelope.identity)
        if owned >= plan.maximum_attempts:
            raise ScopeBudgetRefused("native envelope physical attempt bound is exhausted")
        if paid + (cost or 0) > plan.maximum_known_usd + 1e-9:
            raise ScopeBudgetRefused("native envelope known-price bound is exhausted")
        if used + 1 > policy.parent_attempt_limit:
            raise ScopeBudgetRefused("parent physical attempt cap is exhausted")
        if charged + (cost or 0) > policy.maximum_known_usd + 1e-9:
            raise ScopeBudgetRefused("original known-price parent cap is exhausted")


def associate_reservation(
    connection: sqlite3.Connection,
    reservation: str,
    envelope: SpendEnvelopeRef | None,
) -> None:
    """Associate only a newly inserted scoped claim; old claims remain immutable and unscoped."""
    if envelope is not None:
        authority = connection.execute(
            "SELECT body FROM scoped_parent_policies WHERE identity=?", (envelope.policy.identity,)
        ).fetchone()
        record = _ParentRecord.model_validate_json(authority[0])
        row = connection.execute(
            "SELECT seq FROM reservations WHERE id=?", (reservation,)
        ).fetchone()
        if row is None or row[0] <= record.legacy_cutoff_seq:
            raise ValueError("original retained claims can never be relabeled into a child scope")
        connection.execute(
            "INSERT INTO scoped_claims VALUES (?, ?)", (reservation, envelope.identity)
        )


class EnvelopeCompletion(ContractModel):
    """Exact native domain certificate, never a model-generated success declaration.

    Domain code must verify every owner, count, protocol, task/world/scope and quality condition
    before saving this artifact. The physical kernel independently checks the actual ledger and
    exact saved certificate, and never manufactures one.
    """

    protocol: Literal["native-spend-envelope-completion-v1"] = "native-spend-envelope-completion-v1"
    envelope: SpendEnvelopeRef
    inventory: ArtifactInput
    owners: tuple[ArtifactInput, ...] = Field(min_length=1)
    requirements_sha256: Sha256
    qualification: Literal["native_owners_complete"] = "native_owners_complete"


def settle_spend_envelope(
    project: ProjectStore,
    ceiling: OperationSpendCeiling,
    envelope: SpendEnvelopeRef,
    completion: ArtifactInput,
) -> None:
    """Release unused headroom only after exact native completion and all physical settlement.

    An unknown outcome or live/reserved claim retains the entire open envelope. Closing after
    the deadline is permitted because this reconciles completed work and grants no new calls.
    """
    if envelope.policy.ledger_sha256 != _ledger(ceiling):
        raise ValueError("completion cannot switch the original operation ledger")
    stored = project.artifacts.read(completion.artifact_id)
    if (
        artifact_input(stored.manifest) != completion
        or stored.manifest.artifact_type != "native-spend-envelope-completion"
    ):
        raise ValueError("envelope needs its exact saved native completion certificate")
    record = EnvelopeCompletion.model_validate_json(stored.payloads["completion.json"])
    if record.envelope != envelope:
        raise ValueError("completion certifies another envelope")
    for owner in record.owners:
        if artifact_input(project.artifacts.read(owner.artifact_id).manifest) != owner:
            raise ValueError("completion lost an exact native owner")
    with ceiling._transaction() as connection:
        policy = _policy(connection, envelope.policy.identity)
        if stored.manifest.code_revision != policy.producer_revision:
            raise ValueError("completion producer differs from the reviewed executor")
        row = connection.execute(
            "SELECT body FROM scoped_envelopes WHERE identity=?", (envelope.identity,)
        ).fetchone()
        if row is None:
            raise ValueError("completion has no registered envelope")
        plan = SpendEnvelopePlan.model_validate_json(row[0])
        if (
            record.inventory != plan.inventory
            or record.requirements_sha256 != plan.requirements_sha256
        ):
            raise ValueError("completion changes the native inventory or mandatory study")
        expected = {
            plan.inventory,
            *record.owners,
            policy.authorization,
            policy.preparation,
            policy.execution,
        }
        if set(stored.manifest.inputs) != expected:
            raise ValueError("completion inputs do not preserve the exact native provenance")
        unsettled = connection.execute(
            "SELECT COUNT(*) FROM reservations r JOIN scoped_claims s ON s.reservation=r.id "
            "WHERE s.envelope=? AND r.state NOT IN ('settled','released')",
            (envelope.identity,),
        ).fetchone()[0]
        if unsettled:
            raise ScopeBudgetRefused(
                "orphaned, unknown or unsettled physical claims retain envelope headroom"
            )
        previous = connection.execute(
            "SELECT completion FROM scoped_envelope_settlements WHERE envelope=?",
            (envelope.identity,),
        ).fetchone()
        body = completion.model_dump_json()
        if previous is not None:
            if previous[0] != body:
                raise ValueError("envelope already closed under different completion evidence")
            return
        connection.execute(
            "INSERT INTO scoped_envelope_settlements VALUES (?, ?)", (envelope.identity, body)
        )


class ParentScopeSummary(ContractModel):
    """Actual shared physical claims and still-encumbered future capacity."""

    policy: ParentSpendRef
    parent_attempt_limit: int
    legacy_attempt_limit: int
    parent_attempts: int
    legacy_attempts: int
    encumbered_attempts: int
    accounted_known_usd: float
    encumbered_known_usd: float


def parent_scope_summary(ceiling: OperationSpendCeiling) -> ParentScopeSummary | None:
    """Read scoped facts without recovering, relabeling or deleting any physical liability."""
    with ceiling._transaction() as connection:
        if not _present(connection):
            return None
        row = connection.execute("SELECT identity FROM scoped_parent_policies").fetchone()
        policy = _policy(connection)
        used, charged = _counts(connection)
        booked, held = _booked(connection)
        legacy = connection.execute(
            "SELECT COUNT(*) FROM reservations r LEFT JOIN scoped_claims s ON s.reservation=r.id "
            "WHERE r.state!='released' AND s.reservation IS NULL"
        ).fetchone()[0]
    return ParentScopeSummary(
        policy=ParentSpendRef(identity=row[0], ledger_sha256=_ledger(ceiling)),
        parent_attempt_limit=policy.parent_attempt_limit,
        legacy_attempt_limit=policy.legacy_attempt_limit,
        parent_attempts=used,
        legacy_attempts=legacy,
        encumbered_attempts=booked,
        accounted_known_usd=charged,
        encumbered_known_usd=held,
    )


@contextmanager
def spend_envelope_scope(
    ceiling: OperationSpendCeiling,
    envelope: SpendEnvelopeRef,
) -> Iterator[OperationSpendCeiling]:
    """Open an isolated native context so parallel waves never share a mutable scope holder."""
    from exp.runtime.models.spend_ceiling import (
        current_spend_ceiling,
        install_spend_ceiling,
        operation_spend_scope,
    )

    current = current_spend_ceiling()
    if current is not None and current.path != ceiling.path:
        raise ValueError("an envelope cannot switch the active original operation")
    if current is not None and current.envelope not in (None, envelope):
        raise ValueError("cannot nest a different active spend envelope")
    scoped = scoped_spend_ceiling(ceiling, envelope)
    with operation_spend_scope():
        install_spend_ceiling(scoped)
        yield scoped


class VerifiedSpendEnvelope(ContractModel):
    """Exact currently open authority and native wave, read without dispatch or reconciliation."""

    reference: SpendEnvelopeRef
    policy: ParentSpendPolicy
    plan: SpendEnvelopePlan


def verify_spend_envelope(
    ceiling: OperationSpendCeiling,
    envelope: SpendEnvelopeRef,
) -> VerifiedSpendEnvelope:
    """Verify exact ledger, policy, envelope and lifecycle before operational client creation."""
    if envelope.policy.ledger_sha256 != _ledger(ceiling):
        raise ValueError("operational authority belongs to another original ledger")
    with ceiling._transaction() as connection:
        policy = _policy(connection, envelope.policy.identity)
        plan = _plan(connection, envelope)
        if datetime.now(UTC) >= policy.deadline:
            raise ScopeBudgetRefused("operational authority's native window expired")
    return VerifiedSpendEnvelope(reference=envelope, policy=policy, plan=plan)

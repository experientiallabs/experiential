"""A failed accounting write fences only the organization that owns it.

These tests drive the accounting registry the native data plane calls: one
organization's failed settlement, abandonment or request-only finish refuses
that organization's new work with ``accounting_unavailable`` while every other
organization keeps reserving and settling, and each organization recovers on
its own once the sweep lands its retained write. A deliberate ledger refusal is
a clean, retryable per-request error that never fences anyone.
"""

from __future__ import annotations

import json

import pytest

from exp.runtime.gateway.boundary import boundary_protocol_error
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.ledger_errors import AccountingRefusedError
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_accounting_errors import NativeBridgeError
from exp.runtime.gateway.native_accounting_fence import OrganizationFence
from exp.runtime.gateway.native_accounting_test import (
    _admit,
    _deployment,
    _RecordingLedger,
    _settle,
    _start,
)
from exp.runtime.gateway.native_execution import InflightRequest

_ORG_A = "organization-a"
_ORG_B = "organization-b"


class _ScriptedLedger(_RecordingLedger):
    """Recording ledger that raises one scripted exception per request-only write."""

    def __init__(self) -> None:
        """Start with no scripted request-only write exceptions."""
        super().__init__()
        self.request_errors: list[Exception] = []

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
        certify_no_effects: bool = False,
        web_search_requests: int = 0,
    ) -> bool:
        """Raise the next scripted exception, else record the request terminal."""
        if self.request_errors:
            raise self.request_errors.pop(0)
        return super().finish_request(
            authorization=authorization,
            failure=failure,
            certify_no_effects=certify_no_effects,
            web_search_requests=web_search_requests,
        )


def _fail_next_attempt_write(accounting: NativeAttemptAccounting, error: Exception) -> None:
    """Make the registry's next attempt terminal write raise ``error``, then restore it."""
    original = accounting._finish_attempt  # noqa: SLF001 - durable failure injection.

    def fail_once(**_kwargs: object) -> None:
        """Restore the real writer and raise the scripted failure."""
        accounting._finish_attempt = original  # noqa: SLF001 - durable failure injection.
        raise error

    accounting._finish_attempt = fail_once  # noqa: SLF001 - durable failure injection.


def _accounting() -> tuple[NativeAttemptAccounting, _ScriptedLedger]:
    """Compose a registry whose recovery sweeps run only when a test calls them."""
    ledger = _ScriptedLedger()
    return NativeAttemptAccounting(ledger, background_recovery=False), ledger  # type: ignore[arg-type]


def _admit_org(
    accounting: NativeAttemptAccounting, request_id: str, organization_id: str
) -> InflightRequest:
    """Admit one request for ``organization_id`` over a single-rung ladder."""
    deployments = (_deployment("lead", connection_sha256="b" * 64),)
    accounting.require_clear(organization_id)
    return _admit(accounting, deployments, request_id=request_id, organization_id=organization_id)


def _serve(accounting: NativeAttemptAccounting, request_id: str, organization_id: str) -> None:
    """Admit, reserve and settle one request end to end for ``organization_id``."""
    _admit_org(accounting, request_id, organization_id)
    attempt = _start(accounting, ordinal=0, request_id=request_id)
    _settle(
        accounting,
        attempt_id=str(attempt["attempt_id"]),
        outcome="completed",
        finalize=True,
        request_id=request_id,
    )
    assert accounting.entry(request_id) is None


def _public(error: NativeBridgeError) -> dict[str, object]:
    """Decode a boundary error's public payload."""
    return json.loads(error.public_error_json)


def _fail_settle(
    accounting: NativeAttemptAccounting, ledger: _ScriptedLedger, error: Exception
) -> NativeBridgeError:
    """Admit org A's request and fail its settlement with ``error``."""
    _admit_org(accounting, "a-1", _ORG_A)
    attempt = _start(accounting, ordinal=0, request_id="a-1")
    _fail_next_attempt_write(accounting, error)
    with pytest.raises(NativeBridgeError) as raised:
        _settle(
            accounting,
            attempt_id=str(attempt["attempt_id"]),
            outcome="completed",
            finalize=True,
            request_id="a-1",
        )
    return raised.value


def test_failed_settle_fences_only_its_organization_and_recovers_per_organization() -> None:
    """Org A's settle failure refuses org A only; org B serves; A resumes once it lands."""
    accounting, ledger = _accounting()
    _fail_settle(accounting, ledger, RuntimeError("ledger write lost"))

    assert accounting.fenced_organizations() == frozenset({_ORG_A})
    with pytest.raises(NativeBridgeError) as fenced:
        accounting.require_clear(_ORG_A)
    assert _public(fenced.value)["code"] == "accounting_unavailable"
    assert _public(fenced.value)["status_code"] == 503

    # Org B keeps admitting, reserving and settling on the same worker.
    _serve(accounting, "b-1", _ORG_B)
    _serve(accounting, "b-2", _ORG_B)
    with pytest.raises(NativeBridgeError):
        accounting.require_clear(_ORG_A)

    # The sweep lands org A's retained write: org A recovers, org B never noticed.
    accounting.sweep_expired()
    assert accounting.fenced_organizations() == frozenset()
    assert accounting.entry("a-1") is None
    accounting.require_clear(_ORG_A)
    _serve(accounting, "a-2", _ORG_A)
    assert [row["attempt_id"] for row in ledger.finished] == [
        "attempt-2",
        "attempt-3",
        "attempt-1",
        "attempt-4",
    ]


def test_two_fenced_organizations_recover_independently() -> None:
    """Each organization's fence lifts when its own retained write lands, not another's."""
    accounting, ledger = _accounting()
    _admit_org(accounting, "a-1", _ORG_A)
    _admit_org(accounting, "b-1", _ORG_B)
    attempts = {
        request_id: _start(accounting, ordinal=0, request_id=request_id)
        for request_id in ("a-1", "b-1")
    }
    for request_id in ("a-1", "b-1"):
        _fail_next_attempt_write(accounting, RuntimeError("ledger write lost"))
        with pytest.raises(NativeBridgeError):
            _settle(
                accounting,
                attempt_id=str(attempts[request_id]["attempt_id"]),
                outcome="completed",
                finalize=True,
                request_id=request_id,
            )
    assert accounting.fenced_organizations() == frozenset({_ORG_A, _ORG_B})

    # Org B's write lands on the data plane's own retry; org A's stays lost.
    _settle(
        accounting,
        attempt_id=str(attempts["b-1"]["attempt_id"]),
        outcome="completed",
        finalize=True,
        request_id="b-1",
    )
    assert accounting.fenced_organizations() == frozenset({_ORG_A})
    accounting.require_clear(_ORG_B)
    with pytest.raises(NativeBridgeError):
        accounting.require_clear(_ORG_A)


def test_failed_request_finish_and_abandon_fence_only_their_organization() -> None:
    """Request-only finishes and abandonments fence their own organization the same way."""
    accounting, ledger = _accounting()
    entry = _admit_org(accounting, "a-1", _ORG_A)
    ledger.request_errors.append(RuntimeError("ledger write lost"))
    failure = GatewayFailure(failure_class=GatewayFailureClass.GUARDRAIL, safe_message="blocked")
    assert not accounting.finish_request_quietly(entry.authorization, failure)
    assert accounting.fenced_organizations() == frozenset({_ORG_A})
    _serve(accounting, "b-1", _ORG_B)

    _admit_org(accounting, "b-2", _ORG_B)
    _start(accounting, ordinal=0, request_id="b-2")
    _fail_next_attempt_write(accounting, RuntimeError("ledger write lost"))
    with pytest.raises(NativeBridgeError):
        accounting.abandon(json.dumps({"request_id": "b-2"}))
    assert accounting.fenced_organizations() == frozenset({_ORG_A, _ORG_B})

    accounting.sweep_expired()
    assert accounting.fenced_organizations() == frozenset()
    accounting.require_clear(_ORG_A)
    accounting.require_clear(_ORG_B)


def test_deliberate_settle_refusal_is_a_clean_retryable_error_that_fences_no_one() -> None:
    """A ledger refusal answers 503 ``accounting_refused`` and is replayed without a fence."""
    accounting, ledger = _accounting()
    refused = _fail_settle(accounting, ledger, AccountingRefusedError("owned elsewhere"))

    assert _public(refused)["status_code"] == 503
    assert _public(refused)["code"] == "accounting_refused"
    assert _public(refused)["retry_after_seconds"] == 1
    assert accounting.fenced_organizations() == frozenset()
    # Still an unlanded terminal write: health reports it, admission ignores it.
    assert not accounting.accounting_healthy
    accounting.require_clear(_ORG_A)
    _serve(accounting, "a-2", _ORG_A)

    # The refused meter is still owned and lands once the ledger accepts it.
    retained = accounting.entry("a-1")
    assert retained is not None and retained.pending_settlement is not None
    accounting.sweep_expired()
    assert accounting.entry("a-1") is None
    assert accounting.accounting_healthy
    assert {row["attempt_id"] for row in ledger.finished} == {"attempt-1", "attempt-2"}


def test_deliberate_request_finish_refusal_fences_no_one_and_is_replayed() -> None:
    """A refused request-only write stays retained for replay but never gates admission."""
    accounting, ledger = _accounting()
    entry = _admit_org(accounting, "a-1", _ORG_A)
    ledger.request_errors.append(AccountingRefusedError("owned elsewhere"))
    failure = GatewayFailure(failure_class=GatewayFailureClass.GUARDRAIL, safe_message="blocked")
    assert not accounting.finish_request_quietly(entry.authorization, failure)
    assert accounting.request_settlements.pending
    assert accounting.fenced_organizations() == frozenset()
    accounting.require_clear(_ORG_A)

    # A later non-refusal failure of the same retained write does fence its organization.
    ledger.request_errors.append(RuntimeError("ledger write lost"))
    accounting.sweep_expired()
    assert accounting.fenced_organizations() == frozenset({_ORG_A})
    accounting.sweep_expired()
    assert not accounting.request_settlements.pending
    assert accounting.fenced_organizations() == frozenset()
    assert ledger.finished_requests == [failure]


def test_refusal_maps_to_its_own_public_error_and_validates_its_wait() -> None:
    """The boundary maps a refusal to a retryable 503 carrying the host's wait."""
    error = boundary_protocol_error(AccountingRefusedError("internal", retry_after_seconds=7))
    assert (error.status_code, error.detail.code) == (503, "accounting_refused")
    assert error.retry_after_seconds == 7
    assert "internal" not in error.detail.message
    with pytest.raises(ValueError, match="must be positive"):
        AccountingRefusedError("internal", retry_after_seconds=0)


def test_organization_fence_is_constant_time_and_releases_each_organization() -> None:
    """An organization's entry exists only while it holds a fencing write id."""
    fence = OrganizationFence()
    fence.mark(_ORG_A, "a-1", fencing=True)
    fence.mark(_ORG_A, "a-2", fencing=True)
    fence.mark(_ORG_B, "b-1", fencing=True)
    assert fence.fenced(_ORG_A) and fence.organizations() == frozenset({_ORG_A, _ORG_B})
    fence.mark(_ORG_A, "a-1", fencing=False)
    assert fence.fenced(_ORG_A)
    fence.mark(_ORG_A, "a-2", fencing=False)
    fence.mark(_ORG_A, "a-2", fencing=False)  # Releasing twice is a no-op.
    fence.mark(_ORG_B, "b-1", fencing=False)
    assert not fence.fenced(_ORG_A) and fence.organizations() == frozenset()
    assert fence._requests == {}  # noqa: SLF001 - no per-organization residue.


def test_a_write_failing_after_its_request_left_never_strands_a_fence() -> None:
    """Only the live registry entry can fence; a removed or replaced one releases."""
    accounting, _ledger = _accounting()
    entry = _admit_org(accounting, "a-1", _ORG_A)
    entry.settlement_write_failed = True
    fence = OrganizationFence()
    fence.sync({}, entry)
    assert not fence.fenced(_ORG_A)
    fence.sync({"a-1": entry}, entry)
    assert fence.fenced(_ORG_A)
    assert fence.drop({"a-1": entry}, "a-1") is entry
    assert not fence.fenced(_ORG_A)


def test_recovering_many_organizations_leaves_no_fence_entries() -> None:
    """Fence state follows outstanding failed writes, never every organization seen."""
    accounting, ledger = _accounting()
    failure = GatewayFailure(failure_class=GatewayFailureClass.GUARDRAIL, safe_message="blocked")
    organizations = [f"organization-{index}" for index in range(40)]
    for index, organization in enumerate(organizations):
        entry = _admit_org(accounting, f"r-{index}", organization)
        _admit_org(accounting, f"s-{index}", organization)
        attempt = _start(accounting, ordinal=0, request_id=f"s-{index}")
        ledger.request_errors.append(RuntimeError("ledger write lost"))
        assert not accounting.finish_request_quietly(entry.authorization, failure)
        _fail_next_attempt_write(accounting, RuntimeError("ledger write lost"))
        with pytest.raises(NativeBridgeError):
            _settle(
                accounting,
                attempt_id=str(attempt["attempt_id"]),
                outcome="completed",
                finalize=True,
                request_id=f"s-{index}",
            )
    assert accounting.fenced_organizations() == frozenset(organizations)
    for _ in range(50):
        if accounting.accounting_healthy:
            break
        accounting.sweep_expired()
    assert accounting.accounting_healthy
    assert accounting.fenced_organizations() == frozenset()
    assert accounting._fence._requests == {}  # noqa: SLF001 - released per organization.
    assert accounting.request_settlements._fence._requests == {}  # noqa: SLF001
    for organization in organizations:
        accounting.require_clear(organization)

"""Admission transfers financial ownership once and never selects a Python ledger."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from exp_gateway_native import CaptureCollector, NativeCallbacks

from exp.common.sqlite.native import connect
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.guardrails.contracts import GuardrailOutcome
from exp.runtime.gateway.guardrails.session import GuardrailSession
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.ledger_test import (
    FakeLedgerClock,
    _authority_fixture,
    _deployment,
    _execution,
    _request,
)
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_capture import (
    CaptureConfiguration,
    CaptureController,
    PendingCapture,
)
from exp.runtime.gateway.native_continuation_store import NativeContinuationStore
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_handoff import NativeRequestHandoff, accounting_drained, uses_native
from exp.runtime.gateway.native_sqlite import SQLiteRequestFactory
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.tests.guardrail_observation_test import _Classifier, _Engine, _policy


def _prepared(tmp_path: Path) -> tuple[SQLiteRequestFactory, InflightRequest, SQLiteAttemptLedger]:
    """Accept a real local request before transferring its resolved authority."""
    clock = FakeLedgerClock()
    clock.wall = datetime.now(UTC)
    clock.monotonic_value = time.monotonic()
    store, ledger, key = _authority_fixture(tmp_path, clock)
    request = _request("native ownership fixture").model_copy(update={"maximum_output_tokens": 100})
    authorization = store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=time.monotonic() + 60
    )
    ledger.accept_request(authorization=authorization)
    entry = InflightRequest(
        authorization=authorization,
        route=GatewayRoute(
            snapshot=_execution(authorization), deployment=_deployment(), route_reason="direct"
        ),
        request=request,
        deadline_monotonic=time.monotonic() + 60,
    )
    return ledger.native_request_factory(NativeContinuationStore()), entry, ledger


class _Factory:
    """Script only the initial handoff; every financial write uses real native SQLite."""

    def __init__(
        self,
        factory: SQLiteRequestFactory,
        *,
        failure: bool = False,
        absent: bool = False,
        release: Mock | None = None,
    ) -> None:
        """Retain the genuine native target and a deterministic admission fault."""
        self.factory = factory
        self.failure = failure
        self.absent = absent
        self.release = release
        self.admission_calls = 0

    def __call__(self, entry: InflightRequest, /) -> NativeCallbacks | NativeRequestHandoff:
        """Return native authority or fail before transferring it."""
        if self.failure:
            raise ValueError("owner unavailable")
        if self.absent:
            return cast("NativeCallbacks", None)  # Deliberately invalid host contract.
        target = self.factory(entry)
        return target if self.release is None else NativeRequestHandoff(target, self.release)

    def admission(self, authorization: AuthorizationSnapshot, /) -> NativeCallbacks:
        """Keep failed admission's financial authority independent of route construction."""
        self.admission_calls += 1
        return self.factory.admission(authorization)


@pytest.mark.parametrize("absent", [False, True])
@pytest.mark.parametrize("search_requests", [0, 1, 3])
def test_missing_native_owner_terminalizes_admission_without_a_python_fallback(
    tmp_path: Path, absent: bool, search_requests: int
) -> None:
    """Factory failure retains paid prework and never creates a dispatchable request."""
    underlying, entry, ledger = _prepared(tmp_path)
    factory = _Factory(underlying, failure=not absent, absent=absent)
    accounting = NativeAttemptAccounting(native_request_factory=factory, background_recovery=False)
    entry.web_search_requests = search_requests
    try:
        with pytest.raises(TypeError if absent else ValueError):
            accounting.register(entry)
        assert not uses_native(accounting, entry.authorization.request_id)
        assert accounting_drained(accounting)
        assert factory.admission_calls == 1
        with connect(ledger.database_path) as connection:
            assert connection.execute(
                "SELECT terminal_state,web_search_requests FROM gateway_requests"
            ).fetchall() == [("failed", search_requests)]
            assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone() == (0,)
    finally:
        assert accounting.close()


@pytest.mark.parametrize("fail_registration", [False, True])
def test_host_memos_release_only_after_native_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_registration: bool
) -> None:
    """A rejected registration leaves the original admission available for native cleanup."""
    underlying, entry, _ = _prepared(tmp_path)
    release = Mock()
    factory = _Factory(underlying, release=release)
    accounting = NativeAttemptAccounting(native_request_factory=factory, background_recovery=False)
    if fail_registration:
        monkeypatch.setattr(
            "exp.runtime.gateway.native_registration.registration_json", lambda *_a, **_k: "{}"
        )
    try:
        if fail_registration:
            with pytest.raises(ValueError):
                accounting.register(entry)
            release.assert_not_called()
            assert accounting_drained(accounting)
        else:
            accounting.register(entry)
            release.assert_called_once_with()
            assert uses_native(accounting, entry.authorization.request_id)
            assert not accounting_drained(accounting)
            accounting.abandon(json.dumps({"request_id": entry.authorization.request_id}))
    finally:
        assert accounting.close()


@pytest.mark.parametrize("fail_preparation", [False, True])
def test_failed_handoff_releases_capture_without_a_provider_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_preparation: bool
) -> None:
    """A capture started before ownership transfer cannot retain a rejected request."""
    underlying, entry, _ = _prepared(tmp_path)
    records: list[str] = []
    collector = CaptureCollector(CaptureConfiguration().model_dump_json(), records.append)
    capture = CaptureController(collector, application_for=lambda _auth: "fixture")
    accounting = NativeAttemptAccounting(
        native_request_factory=underlying, background_recovery=False, capture=capture
    )
    if not fail_preparation:
        monkeypatch.setattr(
            "exp.runtime.gateway.native_registration.registration_json", lambda *_a, **_k: "{}"
        )

    def prepare() -> None:
        """Start real native capture before the injected admission failure."""
        assert isinstance(entry.request, GatewayRequest)
        assert capture.begin(entry.authorization, entry.request, "fixture")
        if fail_preparation:
            raise ValueError("preparation failed")

    try:
        with pytest.raises(ValueError):
            accounting.register(entry, prepare=prepare)
        assert accounting_drained(accounting)
        assert collector.close(0.1)
        assert records == []
    finally:
        assert accounting.close()
        collector.close(1)


@pytest.mark.parametrize("stage", ["capture", "serialize", "register"])
@pytest.mark.parametrize("completed", [False, True])
def test_failed_handoff_retains_the_enforcing_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, completed: bool
) -> None:
    """Every pre-transfer failure keeps one enforcement report and its telemetry."""
    underlying, entry, ledger = _prepared(tmp_path)
    classifier = _Classifier(GuardrailOutcome.FLAGGED)
    policy = _policy(timeout_ms=5000).model_copy(
        update={"mode": "enforce", "input_execution": "parallel"}
    )
    engine = _Engine(classifier, policy)
    session = GuardrailSession(
        engine, (policy,), time.monotonic() + 10, request_id=str(entry.authorization.request_id)
    )
    entry.guardrails = session
    assert isinstance(entry.request, GatewayRequest)
    entry.pending_capture = PendingCapture(entry.request, None)
    collector = CaptureCollector(CaptureConfiguration().model_dump_json(), lambda _record: None)
    capture = CaptureController(collector, application_for=lambda _auth: "fixture")
    accounting = NativeAttemptAccounting(
        native_request_factory=underlying, background_recovery=False, capture=capture
    )

    def fail(*_args: object, **_kwargs: object) -> str:
        """Inject a failure before the native request owner accepts the handoff."""
        raise ValueError("handoff fixture failed")

    if stage == "capture":
        monkeypatch.setattr(capture, "prepare", fail)
    else:
        monkeypatch.setattr(
            "exp.runtime.gateway.native_registration.registration_json",
            fail if stage == "serialize" else lambda *_args, **_kwargs: "{}",
        )

    def prepare() -> None:
        """Start a genuine native inspection before reaching the failing stage."""
        assert isinstance(entry.request, GatewayRequest)
        session.prepare_dispatch((entry.request,), overlap=True)
        assert session.native_input is not None
        assert classifier.started.wait(2)
        if completed:
            classifier.release.set()
            session.native_input.wait_json()

    try:
        with pytest.raises(ValueError):
            accounting.register(entry, prepare=prepare)
        classifier.release.set()
        assert accounting.close(timeout_seconds=2)
        with connect(ledger.database_path) as connection:
            reports = connection.execute("SELECT result FROM gateway_input_moderation").fetchall()
            assert len(reports) == 1
            observation = json.loads(reports[0][0])["observations"][0]
            assert observation["mode"] == "enforce"
            assert observation["outcome"] == ("flagged" if completed else "cancelled")
            assert connection.execute("SELECT terminal_state FROM gateway_requests").fetchall() == [
                ("failed",)
            ]
            assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone() == (0,)
        counters = json.loads(accounting.native.moderation_recording_json())
        assert counters["telemetry"]["totals"]["samples"] == 1
        assert counters["failed"] == counters["dropped"] == counters["pending"] == 0
        assert session.native_input is None
    finally:
        classifier.release.set()
        accounting.close(timeout_seconds=2)
        collector.close(1)
        engine.close(timeout_seconds=2)


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("search_requests", [0, 1, 3])
def test_registered_callback_failure_stays_with_the_native_owner(
    tmp_path: Path, cleanup_fails: bool, search_requests: int
) -> None:
    """Lost handoff acknowledgement cannot re-enter host admission or lose its search meter."""
    underlying, entry, ledger = _prepared(tmp_path)
    factory = _Factory(underlying, release=Mock(side_effect=ValueError("acknowledgement failed")))
    accounting = NativeAttemptAccounting(native_request_factory=factory, background_recovery=False)
    entry.web_search_requests = search_requests
    if cleanup_fails:
        with connect(ledger.database_path) as connection:
            connection.execute(
                "CREATE TRIGGER fail_terminal BEFORE UPDATE ON gateway_requests "
                "BEGIN SELECT RAISE(ABORT,'fixture storage unavailable'); END"
            )
    try:
        with pytest.raises(ValueError, match="acknowledgement failed"):
            accounting.register(entry)
        assert factory.admission_calls == 0
        assert accounting.accounting_healthy is not cleanup_fails
        if cleanup_fails:
            with connect(ledger.database_path) as connection:
                connection.execute("DROP TRIGGER fail_terminal")
            accounting.sweep_expired()
        assert accounting_drained(accounting)
        with connect(ledger.database_path) as connection:
            assert connection.execute(
                "SELECT terminal_state,web_search_requests FROM gateway_requests"
            ).fetchall() == [("failed", search_requests)]
    finally:
        assert accounting.close()

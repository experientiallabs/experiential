"""Tests for the native attempt-accounting registry's waterfall reservations."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from typing import Literal, cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelCapabilities
from exp.common.models.catalog import (
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayRungDispatchPolicy,
    GatewayTokenPrices,
)
from exp.common.models.dispatch_policy import GatewayThrottleRedialPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment, FailoverMode
from exp.common.models.gateway_chains import ModelExecutionStage
from exp.runtime.gateway import disconnect_estimate, native_recovery
from exp.runtime.gateway.attempt_tokens import counted_input_tokens
from exp.runtime.gateway.budgets import (
    BudgetReservationRejected,
    BudgetScopeKind,
    maximum_attempt_cost_nano_usd,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.ledger import AttemptRejectedError
from exp.runtime.gateway.native_accounting import (
    NativeAttemptAccounting,
    NativeBridgeError,
)
from exp.runtime.gateway.native_components import SyncWriteLedger
from exp.runtime.gateway.native_execution import (
    InflightRequest,
    deployment_health_key,
    rung_load_key,
)
from exp.runtime.gateway.native_recovery import record_session_outcome, session_cache_key
from exp.runtime.gateway.native_recovery_test import RecoveryHostFake, recovery_entry
from exp.runtime.gateway.native_settlement import failure_from_boundary_payload, ledger_failure
from exp.runtime.gateway.recovery import (
    FrozenRecoveryBinding,
    RecoveryLease,
    RecoveryObservation,
    RecoverySnapshot,
    SessionCacheKey,
    SessionRecoveryRegistry,
)
from exp.runtime.gateway.recovery_test import Clock, eligible
from exp.runtime.gateway.reservation_tokenizer import reservation_encoder
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.rung_admission import RungLoadKey, RungLoadRegistry, RungShed
from exp.runtime.openai_protocol.errors import (
    THROTTLED_RETRY_AFTER_SECONDS,
    public_failure_error,
)

_DIGEST = "a" * 64


def _deployment(
    deployment_id: str,
    *,
    connection_sha256: str,
    dispatch: GatewayRungDispatchPolicy | None = None,
) -> ExactModelDeployment:
    """Build one deployment in the shared certified exact-model pool."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai",
        provider_model="provider-model",
        connection_sha256=connection_sha256,
        capabilities_sha256="d" * 64,
        capabilities=ModelCapabilities(maximum_output_tokens=128_000),
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            dispatch=dispatch,
        ),
    )


def _authorization(catalog_sha256: str) -> AuthorizationSnapshot:
    """Build one direct authority snapshot pinned to the test catalog."""
    return AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256=catalog_sha256,
        canonical_request_sha256=_DIGEST,
        deadline_monotonic=1.0,
    )


def _request() -> GatewayRequest:
    """Build one canonical request for physical execution tests."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )


def _route(
    deployments: tuple[ExactModelDeployment, ...],
    *,
    refusal_failover: bool = False,
) -> GatewayRoute:
    """Build one frozen certified route with a live request deadline."""
    authorization = _authorization(_DIGEST).model_copy(
        update={
            "deadline_monotonic": time.monotonic() + 30,
            "refusal_failover": refusal_failover,
        }
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def test_typed_preflight_rejection_survives_public_and_ledger_terminal() -> None:
    """Expected unavailable root preflight never becomes an internal-error settlement."""
    ledger = _RecordingLedger()
    ledger.typed_rejection = GatewayFailure(
        failure_class=GatewayFailureClass.UNAVAILABLE,
        safe_message="root funding preflight is unavailable",
    )
    accounting = NativeAttemptAccounting(ledger)
    route = _route((_deployment("first", connection_sha256="b" * 64),))
    entry = InflightRequest(
        authorization=route.snapshot.authorization,
        route=route,
        request=_request(),
        deadline_monotonic=time.monotonic() + 10,
    )
    accounting.register(entry)
    with pytest.raises(NativeBridgeError) as raised:
        accounting.start_attempt(
            json.dumps({"request_id": entry.authorization.request_id, "attempt_ordinal": 0})
        )
    assert json.loads(raised.value.public_error_json)["status_code"] == 503
    assert ledger.finished_requests == [ledger.typed_rejection]
    assert not ledger.started


class _RecordingLedger:
    """Blocking write-ledger fake recording every waterfall write."""

    def __init__(self) -> None:
        """Start with empty write logs and no scripted rejections."""
        self.started: list[JsonObject] = []
        self.finished: list[JsonObject] = []
        self.terminal_events: list[GatewayEvent | None] = []
        self.upstream_providers: list[str | None] = []
        self.first_token_times: list[datetime | None] = []
        self.web_search_requests: list[int | None] = []
        self.tool_search_requests: list[int | None] = []
        self.rate_limit_settlements: list[JsonObject] = []
        self.finished_requests: list[GatewayFailure] = []
        self.budget_rejections: dict[str, BudgetScopeKind] = {}
        self.fail_finishes = 0
        self.typed_rejection: GatewayFailure | None = None
        self._counter = 0

    def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Record one accepted request (unused by the registry itself)."""
        del authorization

    def start_attempt(
        self,
        *,
        snapshot: object,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_nano_usd: int | None = None,
        reserved_input_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
        dispatch_reason: str | None = None,
        preferred_deployment: ExactModelDeployment | None = None,
    ) -> str:
        """Reserve one recorded attempt row, honoring scripted rejections."""
        del snapshot, fallback_reason
        if self.typed_rejection is not None:
            raise AttemptRejectedError("root preflight required", failure=self.typed_rejection)
        scope = self.budget_rejections.get(deployment.deployment_id)
        if scope is not None:
            raise BudgetReservationRejected(scope_kind=scope, reason="scripted")
        self._counter += 1
        attempt_id = f"attempt-{self._counter}"
        self.started.append(
            {
                "attempt_id": attempt_id,
                "deployment_id": deployment.deployment_id,
                "attempt_ordinal": attempt_ordinal,
                "route_depth": route_depth,
                "reserved_input_tokens": reserved_input_tokens,
                "reserved_output_tokens": reserved_output_tokens,
                "maximum_cost_nano_usd": maximum_cost_nano_usd,
                "route_reason": route_reason,
                "dispatch_reason": dispatch_reason,
                "preferred_deployment_id": (
                    None if preferred_deployment is None else preferred_deployment.deployment_id
                ),
            }
        )
        return attempt_id

    def finish_attempt(
        self,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
        retry_after_seconds: int | None = None,
        ratelimit_limit_requests: int | None = None,
        ratelimit_remaining_requests: int | None = None,
        ratelimit_limit_tokens: int | None = None,
        ratelimit_remaining_tokens: int | None = None,
        upstream_provider: str | None = None,
        web_search_requests: int | None = None,
        tool_search_requests: int | None = None,
    ) -> None:
        """Record one settled attempt, tracking harvested rate-limit values apart.

        ``web_search_requests`` and ``tool_search_requests`` default to ``None``
        here (the protocol says ``0``) so a recorded ``None`` proves the
        registry withheld the keyword.
        """
        self.first_token_times.append(first_token_at)
        self.upstream_providers.append(upstream_provider)
        self.web_search_requests.append(web_search_requests)
        self.tool_search_requests.append(tool_search_requests)
        self.terminal_events.append(terminal_event)
        if self.fail_finishes > 0:
            self.fail_finishes -= 1
            raise RuntimeError("scripted terminal-write failure")
        self.finished.append(
            {
                "attempt_id": attempt_id,
                "failure_class": None if failure is None else failure.failure_class.value,
                "finalize": finalize_request,
            }
        )
        if any(
            value is not None
            for value in (
                retry_after_seconds,
                ratelimit_limit_requests,
                ratelimit_remaining_requests,
                ratelimit_limit_tokens,
                ratelimit_remaining_tokens,
            )
        ):
            self.rate_limit_settlements.append(
                {
                    "attempt_id": attempt_id,
                    "retry_after_seconds": retry_after_seconds,
                    "ratelimit_limit_requests": ratelimit_limit_requests,
                    "ratelimit_remaining_requests": ratelimit_remaining_requests,
                    "ratelimit_limit_tokens": ratelimit_limit_tokens,
                    "ratelimit_remaining_tokens": ratelimit_remaining_tokens,
                }
            )

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Record one request-only terminalization."""
        del authorization
        self.finished_requests.append(failure)


def _registry() -> tuple[NativeAttemptAccounting, _RecordingLedger, InflightRequest]:
    """Compose one registry over a two-deployment certified route."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)  # type: ignore[arg-type]
    deployments = (
        _deployment("deployment-a", connection_sha256="b" * 64),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )
    route = _route(deployments)
    entry = InflightRequest(
        authorization=route.snapshot.authorization,
        route=route,
        request=_request(),
        deadline_monotonic=time.monotonic() + 30,
    )
    registry.register(entry)
    return registry, ledger, entry


@pytest.mark.parametrize(
    ("surface", "opened", "marker", "has_usage", "expected"),
    [
        (GatewayApiSurface.DECISIONS, False, True, False, True),
        (GatewayApiSurface.DECISIONS, False, False, False, False),
        (GatewayApiSurface.DECISIONS, False, "true", False, False),
        (GatewayApiSurface.DECISIONS, True, True, False, False),
        (GatewayApiSurface.DECISIONS, False, True, True, False),
        (GatewayApiSurface.CHAT_COMPLETIONS, False, True, False, False),
        (GatewayApiSurface.RESPONSES, False, True, False, False),
    ],
)
def test_rejection_evidence_reaches_only_unopened_unmetered_decision_failures(
    surface: GatewayApiSurface,
    opened: bool,
    marker: bool | str,
    has_usage: bool,
    expected: bool,
) -> None:
    """Only explicit native evidence can release a decision's unobserved liability."""
    registry, ledger, entry = _registry()
    entry.authorization = entry.authorization.model_copy(update={"surface": surface})
    registry.settle(
        json.dumps(
            {
                "request_id": entry.authorization.request_id,
                "attempt_id": "attempt-one",
                "outcome": "failed",
                "usage": {"input_tokens": 7, "output_tokens": 3} if has_usage else None,
                "failure": {"failure_class": "provider_authentication", "safe_message": "rejected"},
                "opened": opened,
                "decision_provider_rejected": marker,
            }
        )
    )
    event = ledger.terminal_events[-1]
    assert event is not None
    assert event.decision_provider_rejected is expected
    assert (event.usage is not None) is has_usage


@pytest.mark.parametrize("frozen_bound", (2_048, 100_000))
def test_attempt_token_and_money_reservations_use_the_frozen_payload_bound(
    frozen_bound: int,
) -> None:
    """Omitted public caps cannot reserve less than the selected wire may generate."""
    registry, ledger, entry = _registry()
    deployment = entry.route.deployment.model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(
                prices=GatewayTokenPrices(
                    input_nano_usd_per_million_tokens=1_000_000,
                    output_nano_usd_per_million_tokens=2_000_000,
                )
            )
        }
    )
    entry.route = entry.route.model_copy(update={"deployment": deployment})
    entry.reserved_output_tokens_by_depth = (frozen_bound, 128_000)
    _start(registry, ordinal=0)
    assert isinstance(entry.request, GatewayRequest)
    assert entry.request.maximum_output_tokens is None
    row = ledger.started[-1]
    assert row["reserved_output_tokens"] == frozen_bound
    assert row["maximum_cost_nano_usd"] == maximum_attempt_cost_nano_usd(
        entry.request.model_copy(update={"maximum_output_tokens": frozen_bound}), deployment
    )


@pytest.mark.parametrize("partial_usage", (False, True))
@pytest.mark.parametrize("retry", ("direct", "explicit", "sweep"))
def test_disconnect_usage_evidence_survives_every_settlement_path(
    partial_usage: bool, retry: str
) -> None:
    """Observed partial meters never erase the trusted incomplete-usage hold evidence."""
    registry, ledger, entry = _registry()
    started = _start(registry, ordinal=0)
    payload: JsonObject = {
        "request_id": entry.authorization.request_id,
        "attempt_id": started["attempt_id"],
        "outcome": "failed",
        "usage": {"input_tokens": 7, "output_tokens": 3} if partial_usage else None,
        "failure": {"failure_class": "cancelled", "safe_message": "caller disconnected"},
        "finalize": True,
        "opened": partial_usage,
        "dispatched": True,
        "usage_incomplete_due_to_disconnect": True,
    }
    encoded = json.dumps(payload)
    if retry != "direct":
        ledger.fail_finishes = 1
        with pytest.raises(NativeBridgeError):
            registry.settle(encoded)
        assert entry.pending_settlement == payload
        assert ledger.terminal_events[-1] is not None
        assert ledger.terminal_events[-1].usage_incomplete_due_to_disconnect is True
    if retry == "sweep":
        registry.sweep_expired()
    else:
        registry.settle(encoded)
    terminal = ledger.terminal_events[-1]
    assert terminal is not None
    assert terminal.usage_incomplete_due_to_disconnect is True
    assert (terminal.usage is not None) is partial_usage
    if terminal.usage is not None:
        assert terminal.usage.input_tokens == 7
        assert terminal.usage.output_tokens == 3
    assert registry.entry(entry.authorization.request_id) is None
    assert len(ledger.finished) == 1


@pytest.mark.parametrize("retry", ["direct", "sweep"])
def test_opened_disconnect_estimates_cache_reads_from_the_organizations_settled_share(
    retry: str,
) -> None:
    """The org's settled cached fraction fills the cache-read leg, frozen for an exact replay."""
    registry, ledger, entry = _registry()
    started = _start(registry, ordinal=0)
    registry._loads.record_settle(  # noqa: SLF001 - seeding the EWMA the settle path reads
        rung_load_key(entry.route.deployments[0]),
        entry.authorization.organization_id,
        cached_tokens=900,
        input_tokens=1_000,
    )
    encoded = json.dumps(
        {
            "request_id": entry.authorization.request_id,
            "attempt_id": started["attempt_id"],
            "outcome": "failed",
            "usage": None,
            "failure": {"failure_class": "cancelled", "safe_message": "caller disconnected"},
            "finalize": True,
            "opened": True,
            "dispatched": True,
            "usage_incomplete_due_to_disconnect": True,
            "streamed_output": {"text": "partial", "reasoning": "", "images": 0},
        }
    )
    if retry == "sweep":
        ledger.fail_finishes = 1
        with pytest.raises(NativeBridgeError):
            registry.settle(encoded)
        # The live signal moves before the sweep lands the retained payload.
        registry._loads.record_settle(  # noqa: SLF001 - moving the EWMA the replay must ignore
            rung_load_key(entry.route.deployments[0]),
            entry.authorization.organization_id,
            cached_tokens=0,
            input_tokens=1_000_000,
        )
        registry.sweep_expired()
    else:
        registry.settle(encoded)
    terminal = ledger.terminal_events[-1]
    assert terminal is not None and terminal.usage is not None
    assert terminal.usage_estimated is True
    counted = counted_input_tokens(entry.request)
    assert terminal.usage.input_tokens == counted
    assert terminal.usage.cached_input_tokens == int(counted * 0.9)
    assert len(ledger.finished) == 1


@pytest.mark.parametrize("first_fraction", [0.0, 0.9])
def test_concurrent_disconnect_estimates_share_one_frozen_fraction(
    monkeypatch: pytest.MonkeyPatch, first_fraction: float
) -> None:
    """A delayed duplicate cannot replace the first frozen sample, including an explicit zero."""
    registry, _ledger, entry = _registry()
    started = _start(registry, ordinal=0)
    payload: JsonObject = {
        "attempt_id": started["attempt_id"],
        "outcome": "failed",
        "failure": {"failure_class": "cancelled", "safe_message": "caller disconnected"},
        "usage": None,
        "opened": True,
        "dispatched": True,
        "usage_incomplete_due_to_disconnect": True,
        "streamed_output": {"text": "partial"},
    }
    reading, release = threading.Event(), threading.Event()
    results: list[GatewayEvent] = []
    errors: list[BaseException] = []
    owner = threading.current_thread()

    def racing_fraction(
        loads: RungLoadRegistry | None, current: InflightRequest, attempt_id: object
    ) -> float:
        """Pause the older sample until the competing settlement freezes its sample."""
        assert current is entry and attempt_id == started["attempt_id"]
        if threading.current_thread() is owner:
            return first_fraction
        reading.set()
        assert release.wait(5)
        return 0.3

    def estimate() -> None:
        """Retain the delayed caller's actual terminal or surface its thread failure."""
        try:
            results.append(disconnect_estimate.settled_terminal(payload, entry)[0])
        except BaseException as error:  # noqa: BLE001 - thread failures must reach the assertion.
            errors.append(error)

    monkeypatch.setattr(disconnect_estimate, "_recent_cached_fraction", racing_fraction)
    worker = threading.Thread(target=estimate)
    worker.start()
    try:
        assert reading.wait(5)
        first = disconnect_estimate.settled_terminal(payload, entry)[0]
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not errors
    assert results == [first]
    assert entry.estimated_cache_fractions[str(started["attempt_id"])] == first_fraction


@pytest.mark.parametrize("retry", ["direct", "sweep"])
@pytest.mark.parametrize("observed_write", [False, True])
def test_estimated_cache_pricing_uses_actual_child_without_creating_observed_evidence(
    monkeypatch: pytest.MonkeyPatch, retry: str, observed_write: bool
) -> None:
    """Child pricing reads only its org/rung sample and never feeds estimates back into evidence."""
    registry, ledger, entry = _registry()
    registry.recovery_host = RecoveryHostFake()
    entry.route = entry.route.model_copy(
        update={
            "snapshot": entry.route.snapshot.model_copy(
                update={
                    "model_stages": tuple(
                        ModelExecutionStage(
                            stage_index=depth,
                            exact_model_id="child-model"
                            if depth
                            else entry.route.snapshot.exact_model_id,
                            pool_id="child-pool" if depth else entry.route.snapshot.pool_id,
                            deployment_ids=(deployment.deployment_id,),
                        )
                        for depth, deployment in enumerate(entry.route.deployments)
                    )
                }
            )
        }
    )
    first = _start(registry, ordinal=0)
    failure: JsonObject = {
        "failure_class": "provider_internal",
        "safe_message": "provider unavailable",
        "retryable_same_deployment": False,
        "failover_eligible": True,
    }
    _settle(
        registry,
        attempt_id=str(first["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=failure,
    )
    started = _start(registry, ordinal=1, current_depth=0, failure=failure)
    assert started["route_depth"] == 1
    for depth, cached in ((0, 100), (1, 900)):
        registry.loads.record_settle(
            rung_load_key(entry.route.deployments[depth]),
            entry.authorization.organization_id,
            cached_tokens=cached,
            input_tokens=1_000,
        )
    registry.loads.record_settle(
        rung_load_key(entry.route.deployments[1]),
        "other-organization",
        cached_tokens=200,
        input_tokens=1_000,
    )
    recorded = set(entry.recovery_recorded_attempts)
    sessions = dict(registry.recovery._sessions)

    def reject_estimated_sample(
        key: RungLoadKey, organization_id: str, *, cached_tokens: int, input_tokens: int
    ) -> None:
        """An estimated terminal must never reach the observed fairness/EWMA writer."""
        pytest.fail("estimated cache pricing fed the observed cache registry")

    monkeypatch.setattr(registry.loads, "record_settle", reject_estimated_sample)
    payload = json.dumps(
        {
            "request_id": entry.authorization.request_id,
            "attempt_id": started["attempt_id"],
            "outcome": "failed",
            "usage": {"input_tokens": 1_000, "output_tokens": 1, "cache_creation_input_tokens": 800}
            if observed_write
            else None,
            "failure": {"failure_class": "cancelled", "safe_message": "caller disconnected"},
            "finalize": True,
            "opened": True,
            "dispatched": True,
            "usage_incomplete_due_to_disconnect": True,
            "streamed_output": {"text": "partial"},
        }
    )
    if retry == "sweep":
        ledger.fail_finishes = 1
        with pytest.raises(NativeBridgeError):
            registry.settle(payload)
        registry.sweep_expired()
    else:
        registry.settle(payload)
    registry.settle(payload)
    terminal = ledger.terminal_events[-1]
    assert terminal is not None and terminal.usage is not None and terminal.usage_estimated
    expected_read = 200 if observed_write else int(counted_input_tokens(entry.request) * 0.9)
    assert terminal.usage.cached_input_tokens == expected_read
    assert terminal.usage.cache_creation_input_tokens == (800 if observed_write else None)
    assert terminal.usage.cache_creation_1h_input_tokens is None
    assert entry.recovery_recorded_attempts == recorded
    assert registry.recovery._sessions == sessions
    assert not entry.cache_recorded_attempts
    assert len(ledger.finished) == 2


@pytest.mark.parametrize("retry", ["direct", "sweep"])
def test_estimated_disconnect_preserves_search_charges_and_receipt_time_without_cache_proof(
    monkeypatch: pytest.MonkeyPatch, retry: str
) -> None:
    """Known request operations survive an absent meter exactly once, without estimated warmth."""
    registry, ledger, entry = _registry()
    registry.recovery_host = RecoveryHostFake()
    started = _start(registry, ordinal=0)
    attempt_id = str(started["attempt_id"])
    clock = Clock()
    clock.now = 1000
    monkeypatch.setattr(registry.recovery, "observation_time", lambda: clock.now)
    original = disconnect_estimate._text_tokens

    def delayed_tokens(text: str, overflow: int) -> int:
        """Tokenization happens only after the first validated receipt timestamp is retained."""
        assert entry.recovery_observed_at[attempt_id] == 1000
        clock.now = 1100
        return original(text, overflow)

    monkeypatch.setattr(disconnect_estimate, "_text_tokens", delayed_tokens)
    payload = json.dumps(
        {
            "request_id": entry.authorization.request_id,
            "attempt_id": attempt_id,
            "outcome": "failed",
            "failure": {"failure_class": "cancelled", "safe_message": "cut"},
            "usage": None,
            "tool_names": [],
            "dispatched": True,
            "opened": True,
            "finalize": True,
            "usage_incomplete_due_to_disconnect": True,
            "streamed_output": {"text": "visible output"},
            "web_search_requests": 2,
            "tool_search_requests": 3,
        }
    )
    if retry == "sweep":
        ledger.fail_finishes = 1
        with pytest.raises(NativeBridgeError):
            registry.settle(payload)
        registry.sweep_expired()
    else:
        registry.settle(payload)
    registry.settle(payload)
    terminal = ledger.terminal_events[-1]
    assert terminal is not None and terminal.usage_estimated and terminal.usage is not None
    assert terminal.usage.web_search_requests == 2 and terminal.usage.tool_search_requests == 3
    assert ledger.web_search_requests[-1] == 2 and ledger.tool_search_requests[-1] == 3
    assert len(ledger.finished) == 1
    assert entry.recovery_observed_at[attempt_id] == 1000
    assert not entry.recovery_recorded_attempts
    assert not registry.recovery._sessions
    assert (
        registry.loads.cached_fraction(
            rung_load_key(entry.route.deployment), entry.authorization.organization_id
        )
        == 0
    )


@pytest.mark.parametrize("retry", ["direct", "sweep"])
@pytest.mark.parametrize(
    "surface", [GatewayApiSurface.CHAT_COMPLETIONS, GatewayApiSurface.DECISIONS]
)
@pytest.mark.parametrize("opened", [False, True])
def test_opened_disconnect_settles_an_estimated_meter_on_every_path(
    opened: bool, surface: GatewayApiSurface, retry: str
) -> None:
    """An accepted request the caller abandoned prices its prompt and streamed text once."""
    registry, ledger, entry = _registry()
    entry.authorization = entry.authorization.model_copy(update={"surface": surface})
    started = _start(registry, ordinal=0)
    payload: JsonObject = {
        "request_id": entry.authorization.request_id,
        "attempt_id": started["attempt_id"],
        "outcome": "failed",
        "usage": None,
        "failure": {"failure_class": "cancelled", "safe_message": "caller disconnected"},
        "finalize": True,
        "opened": opened,
        "dispatched": True,
        "usage_incomplete_due_to_disconnect": True,
        "streamed_output": {
            "text": "The sky is blue because",
            "reasoning": "",
            "text_overflow_chars": 0,
            "reasoning_overflow_chars": 0,
        },
    }
    encoded = json.dumps(payload)
    if retry == "sweep":
        ledger.fail_finishes = 1
        with pytest.raises(NativeBridgeError):
            registry.settle(encoded)
        registry.sweep_expired()
    else:
        registry.settle(encoded)
    terminal = ledger.terminal_events[-1]
    assert terminal is not None
    assert terminal.usage_incomplete_due_to_disconnect is True
    estimated = opened and surface is not GatewayApiSurface.DECISIONS
    assert terminal.usage_estimated is estimated
    if estimated:
        assert terminal.usage is not None
        assert terminal.usage.input_tokens == counted_input_tokens(entry.request) > 0
        assert terminal.usage.output_tokens == len(
            reservation_encoder().encode_ordinary("The sky is blue because")
        )
        assert terminal.usage.reasoning_tokens == 0
    else:
        assert terminal.usage is None
    assert registry.entry(entry.authorization.request_id) is None
    assert len(ledger.finished) == 1


@pytest.mark.parametrize(
    ("outcome", "dispatched", "usage"),
    (
        ("failed", False, None),
        ("completed", True, None),
        ("completed", True, {"input_tokens": 7, "output_tokens": 3}),
    ),
)
def test_predispatch_and_provider_terminal_settlements_do_not_invent_disconnect_evidence(
    outcome: str, dispatched: bool, usage: JsonObject | None
) -> None:
    """No terminal meter is different from the native witness of an interrupted provider."""
    registry, ledger, entry = _registry()
    started = _start(registry, ordinal=0)
    registry.settle(
        json.dumps(
            {
                "request_id": entry.authorization.request_id,
                "attempt_id": started["attempt_id"],
                "outcome": outcome,
                "usage": usage,
                "failure": (
                    {"failure_class": "cancelled", "safe_message": "cancelled before dispatch"}
                    if outcome == "failed"
                    else None
                ),
                "dispatched": dispatched,
                "finalize": True,
            }
        )
    )
    terminal = ledger.terminal_events[-1]
    assert terminal is not None
    assert terminal.usage_incomplete_due_to_disconnect is False


def _start(
    registry: NativeAttemptAccounting,
    *,
    ordinal: int,
    current_depth: int | None = None,
    failure: JsonObject | None = None,
    request_id: str = "request-one",
    throttle_backoff: bool = False,
    tool_search_round: bool = False,
) -> JsonObject:
    """Call one start_attempt with the data plane's wire shape."""
    return json.loads(
        registry.start_attempt(
            json.dumps(
                {
                    "request_id": request_id,
                    "attempt_ordinal": ordinal,
                    "current_depth": current_depth,
                    "failure": failure,
                    "throttle_backoff": throttle_backoff,
                    "tool_search_round": tool_search_round,
                }
            )
        )
    )


def _settle(
    registry: NativeAttemptAccounting,
    *,
    attempt_id: str,
    outcome: str,
    finalize: bool,
    failure: JsonObject | None = None,
    request_id: str = "request-one",
) -> str:
    """Call one settle with the data plane's wire shape."""
    return registry.settle(
        json.dumps(
            {
                "request_id": request_id,
                "attempt_id": attempt_id,
                "outcome": outcome,
                "usage": None,
                "tool_names": [],
                "failure": failure,
                "finalize": finalize,
                "opened": True,
            }
        )
    )


def _retryable_failure() -> JsonObject:
    """One wire failure the executor may redial on the same deployment."""
    return {
        "failure_class": "provider_internal",
        "safe_message": "provider service failed; retry after a short delay",
        "retryable_same_deployment": True,
        "failover_eligible": True,
    }


@pytest.mark.parametrize("reject_successor", [False, True])
def test_repair_successor_owns_fresh_reservation_without_erasing_prior_unknown(
    reject_successor: bool,
) -> None:
    """A settled unknown first attempt survives both successful and refused repair reservations."""
    registry, ledger, entry = _registry()
    first = _start(registry, ordinal=0)
    _settle(
        registry,
        attempt_id=str(first["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure={"failure_class": "invalid_request", "safe_message": "encrypted item refused"},
    )
    assert ledger.terminal_events[-1] is not None and ledger.terminal_events[-1].usage is None
    assert entry.active_attempt_id is None
    payload = json.dumps(
        {
            "request_id": entry.authorization.request_id,
            "attempt_ordinal": 1,
            "current_depth": 0,
            "reasoning_repair": True,
        }
    )
    if reject_successor:
        ledger.budget_rejections[entry.route.deployment.deployment_id] = BudgetScopeKind.TEAM
        with pytest.raises(NativeBridgeError):
            registry.start_attempt(payload)
        assert len(ledger.started) == 1 and len(ledger.finished) == 1
        assert registry.entry(entry.authorization.request_id) is None
    else:
        second = json.loads(registry.start_attempt(payload))
        assert second["attempt_id"] != first["attempt_id"]
        assert len(ledger.started) == 2 and entry.total_attempts == 2
        assert entry.attempt_counts[0] == 2 and entry.ordinary_attempt_counts[0] == 1
        registry.abandon(json.dumps({"request_id": entry.authorization.request_id}))
        assert len(ledger.finished) == 2
    assert ledger.terminal_events[0] is not None and ledger.terminal_events[0].usage is None


def test_waterfall_reservations_count_every_physical_dispatch() -> None:
    """Ordinals count all dispatches; depth tracks the deployment position."""
    registry, ledger, _entry = _registry()
    first = _start(registry, ordinal=0)
    assert first == {"attempt_id": "attempt-1", "route_depth": 0}
    assert (
        _settle(
            registry,
            attempt_id="attempt-1",
            outcome="failed",
            finalize=False,
            failure=_retryable_failure(),
        )
        == "{}"
    )
    redial = _start(registry, ordinal=1, current_depth=0, failure=_retryable_failure())
    assert redial == {"attempt_id": "attempt-2", "route_depth": 0}
    assert (
        _settle(
            registry,
            attempt_id="attempt-2",
            outcome="failed",
            finalize=False,
            failure=_retryable_failure(),
        )
        == "{}"
    )
    failover = _start(registry, ordinal=2, current_depth=0, failure=_retryable_failure())
    assert failover == {"attempt_id": "attempt-3", "route_depth": 1}
    assert _settle(registry, attempt_id="attempt-3", outcome="completed", finalize=True) == "{}"
    assert [(row["attempt_ordinal"], row["route_depth"]) for row in ledger.started] == [
        (0, 0),
        (1, 0),
        (2, 1),
    ]
    # Every physical dispatch reserves a positive worst-case token window so the
    # platform's token caps bind on the in-flight burst, not only on settlement.
    for row in ledger.started:
        assert isinstance(row["reserved_input_tokens"], int) and row["reserved_input_tokens"] > 0
        assert isinstance(row["reserved_output_tokens"], int) and row["reserved_output_tokens"] > 0
    assert [row["finalize"] for row in ledger.finished] == [False, False, True]
    assert registry.entry("request-one") is None
    assert ledger.finished_requests == []


def test_deployment_budget_rejection_skips_to_the_next_route() -> None:
    """A deployment-scope budget rejection advances without a caller error."""
    registry, ledger, _entry = _registry()
    ledger.budget_rejections["deployment-a"] = BudgetScopeKind.DEPLOYMENT
    started = _start(registry, ordinal=0)
    assert started["route_depth"] == 1
    assert ledger.started[0]["deployment_id"] == "deployment-b"


def test_non_deployment_budget_rejection_finalizes_with_quota() -> None:
    """A team-scope rejection raises the public quota error and finalizes."""
    registry, ledger, _entry = _registry()
    ledger.budget_rejections["deployment-a"] = BudgetScopeKind.TEAM
    with pytest.raises(NativeBridgeError) as excinfo:
        _start(registry, ordinal=0)
    payload = json.loads(excinfo.value.public_error_json)
    assert payload["status_code"] == 429
    assert payload["code"] == "insufficient_quota"
    assert [failure.failure_class.value for failure in ledger.finished_requests] == [
        "quota_exceeded"
    ]
    assert registry.entry("request-one") is None


def test_exhaustion_finalizes_the_request_with_the_last_failure() -> None:
    """An ineligible failure class exhausts the ladder and finalizes."""
    registry, ledger, _entry = _registry()
    started = _start(registry, ordinal=0)
    assert (
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure={"failure_class": "invalid_request", "safe_message": "bad request"},
        )
        == "{}"
    )
    exhausted = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure={
            "failure_class": "invalid_request",
            "safe_message": "bad request",
            "retryable_same_deployment": False,
            "failover_eligible": False,
        },
    )
    assert exhausted["exhausted"] is True
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["failure_class"] == "invalid_request"
    assert [failure.failure_class.value for failure in ledger.finished_requests] == [
        "invalid_request"
    ]
    assert registry.entry("request-one") is None


def test_a_fully_throttled_route_exhausts_as_throttled_not_provider_internal() -> None:
    """Pre-dispatch exhaustion caused only by provider throttle windows is
    caller-facing rate limiting, never platform deadness.

    Production signal (2026-09-04): a single-rung alias whose rung sat inside
    the 30s throttle window after provider 429s reported every shadowed
    request as provider_internal "all exact-model deployments are
    unavailable", misfiling a 429 storm as an outage.
    """
    registry, ledger, entry = _registry()
    throttle = GatewayFailure(
        failure_class=GatewayFailureClass.THROTTLED,
        safe_message="provider throttled the request",
    )
    for deployment in entry.route.deployments:
        registry.health.failed(deployment_health_key(entry.authorization, deployment), throttle)

    exhausted = _start(registry, ordinal=0)

    assert exhausted["exhausted"] is True
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["failure_class"] == "throttled"
    message = str(failure_payload["safe_message"])
    assert "throttle window" in message
    retry_after = failure_payload["retry_after_seconds"]
    assert isinstance(retry_after, int)
    # The advertised Retry-After covers the whole remaining window (floored
    # at the default backoff) and the message names the same wait, so a
    # client honoring the header never retries into the window it was told
    # to sit out.
    assert THROTTLED_RETRY_AFTER_SECONDS <= retry_after <= 30
    assert f"retry in {retry_after}s" in message
    public = public_failure_error(GatewayFailure.model_validate(failure_payload))
    assert public.retry_after_seconds == retry_after
    assert [failure.failure_class.value for failure in ledger.finished_requests] == ["throttled"]
    assert registry.entry("request-one") is None


def test_an_open_circuit_route_still_dispatches_and_never_reports_throttled() -> None:
    """Circuit-open deployments stay dispatchable through forced claims, so
    the throttled exhaustion class is reserved for real throttle windows."""
    registry, ledger, entry = _registry()
    dead = GatewayFailure(
        failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
        safe_message="provider authentication failed",
    )
    for deployment in entry.route.deployments:
        registry.health.failed(deployment_health_key(entry.authorization, deployment), dead)

    started = _start(registry, ordinal=0)

    assert started["route_depth"] == 0
    assert ledger.finished_requests == []


def test_ordinal_mismatch_is_a_wire_contract_failure() -> None:
    """A desynchronized dispatch count fails closed as an internal error."""
    registry, _ledger, _entry = _registry()
    with pytest.raises(NativeBridgeError):
        _start(registry, ordinal=3)


def test_abandon_without_an_active_attempt_finalizes_the_request_row() -> None:
    """Abandoning an accepted request with no reservation closes the request."""
    registry, ledger, _entry = _registry()
    assert registry.abandon(json.dumps({"request_id": "request-one"})) == "{}"
    assert [failure.failure_class.value for failure in ledger.finished_requests] == ["cancelled"]
    assert registry.entry("request-one") is None


def test_sweep_cancels_the_active_attempt_after_the_deadline() -> None:
    """The deadline sweep closes an unsettled reservation as cancelled."""
    registry, ledger, entry = _registry()
    started = _start(registry, ordinal=0)
    entry.deadline_monotonic = time.monotonic() - 60.0
    registry.sweep_expired()
    assert ledger.finished == [
        {
            "attempt_id": started["attempt_id"],
            "failure_class": "cancelled",
            "finalize": True,
        }
    ]
    assert registry.entry("request-one") is None
    assert registry.counters()[1] == 1


@pytest.mark.parametrize("writes", [None, 0, 25])
@pytest.mark.parametrize("fault", ["raise", "provider", "exact_model_id", "organization_id"])
@pytest.mark.parametrize("delivery", ["direct", "retry", "sweep"])
@pytest.mark.parametrize("finalize", [False, True])
def test_recovery_observer_failure_cannot_block_durable_settlement_cleanup(
    writes: int | None,
    fault: Literal["raise", "provider", "exact_model_id", "organization_id"],
    delivery: Literal["direct", "retry", "sweep"],
    finalize: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both settlement paths release load and finalize despite unusable host scope."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger, recovery_host=RecoveryHostFake(fault))
    deployments = _bounded_pair(1)
    entry = _admit(registry, deployments, request_id="recovery-fault")
    entry.request = _request().model_copy(
        update={"provider_prompt_cache_key": "xpl-test-session", "prompt_cache_key": "session"}
    )
    deployment = entry.route.deployment
    invalid_scope = (
        RecoveryHostFake()
        .scope_for(deployment, entry.authorization.organization_id)
        .model_copy(update={"provider" if fault == "raise" else fault: "synthetic-private-detail"})
    )
    entry.recovery_bindings[deployment.deployment_id] = FrozenRecoveryBinding(
        deployment.deployment_id,
        deployment.connection_sha256,
        "https://test.invalid",
        deployment.provider_model,
        invalid_scope,
    )
    frozen_bindings = dict(entry.recovery_bindings)
    started = _start(registry, ordinal=0, request_id=entry.authorization.request_id)
    settlement = json.dumps(
        {
            "request_id": entry.authorization.request_id,
            "attempt_id": started["attempt_id"],
            "outcome": "completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5,
                "cached_input_tokens": 50,
                "cache_creation_input_tokens": writes,
            },
            "first_token_at": "2026-09-18T01:02:03+00:00",
            "upstream_provider": "Azure",
            "finalize": finalize,
        }
    )
    if delivery != "direct":
        ledger.fail_finishes = 1
        with pytest.raises(NativeBridgeError):
            registry.settle(settlement)
        assert entry.pending_settlement == json.loads(settlement)
        if delivery == "sweep":
            registry.sweep_expired()
            assert entry.pending_settlement is None
            assert registry.counters()[0] == 1
        else:
            assert registry.settle(settlement) == "{}"
    else:
        assert registry.settle(settlement) == "{}"
    assert len(ledger.finished) == 1 and ledger.finished[0]["finalize"] is finalize
    observed = datetime.fromisoformat("2026-09-18T01:02:03+00:00")
    calls = 1 if delivery == "direct" else 2
    assert ledger.first_token_times == [observed] * calls
    assert ledger.upstream_providers == ["Azure"] * calls
    assert len(ledger.terminal_events) == calls
    for event in ledger.terminal_events:
        assert event is not None and event.usage is not None
        assert event.usage.cache_creation_input_tokens == writes
        assert event.usage.input_tokens == 100
        assert event.usage.cached_input_tokens == 50
    assert registry.loads.inflight(("deployment-a", "b" * 64)) == 0
    assert registry.accounting_healthy
    assert entry.recovery_bindings == frozen_bindings
    assert not entry.recovery_recorded_attempts
    assert not registry.recovery._sessions  # noqa: SLF001 - scope faults must write no evidence.
    assert "synthetic-private-detail" not in caplog.text
    assert caplog.records and all(record.exc_info is None for record in caplog.records)
    if finalize:
        assert registry.entry(entry.authorization.request_id) is None
    else:
        assert registry.entry(entry.authorization.request_id) is entry
        assert entry.active_attempt_id is None
        registry.abandon(json.dumps({"request_id": entry.authorization.request_id}))
    # Another expired request still settles on the next sweep after the host fault.
    other = _admit(registry, deployments, request_id="after-recovery-fault")
    _start(registry, ordinal=0, request_id=other.authorization.request_id)
    other.deadline_monotonic = time.monotonic() - 60
    registry.sweep_expired()
    assert registry.entry(other.authorization.request_id) is None
    assert registry.counters()[1] == 1


def test_rejected_parameter_crosses_the_boundary_only_as_a_string() -> None:
    """The provider-named parameter path survives the failure payload decode."""
    registry, _ledger, _entry = _registry()
    started = _start(registry, ordinal=0)
    assert (
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure={
                "failure_class": "invalid_request",
                "safe_message": "provider rejected the request",
                "rejected_parameter": "input[1].status",
            },
        )
        == "{}"
    )
    exhausted = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure={
            "failure_class": "invalid_request",
            "safe_message": "provider rejected the request",
            "rejected_parameter": "input[1].status",
        },
    )
    assert exhausted["exhausted"] is True
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["rejected_parameter"] == "input[1].status"
    # Non-string or empty payload values decode to None, never a coerced str.
    numeric = failure_from_boundary_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "rejected_parameter": 7}
    )
    assert numeric is not None and numeric.rejected_parameter is None
    empty = failure_from_boundary_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "rejected_parameter": ""}
    )
    assert empty is not None and empty.rejected_parameter is None


def _admit(
    registry: NativeAttemptAccounting,
    deployments: tuple[ExactModelDeployment, ...],
    *,
    request_id: str,
    organization_id: str = "organization-one",
    weight: int = 1,
    failover_mode: FailoverMode = "maximize_availability",
    throttle_cache_threshold: float | None = None,
    throttle_redial: GatewayThrottleRedialPolicy | None = None,
    affinity_fingerprint: bytes | None = None,
    sticky_preferred: bool = False,
    reasoning_pinned_deployment_id: str | None = None,
    catalog_sha256: str = _DIGEST,
) -> InflightRequest:
    """Register one admitted request over the given rung ladder.

    ``reasoning_pinned_deployment_id`` admits the request as a reasoning
    continuation pinned to that rung (route reason ``reasoning_continuation``).
    ``catalog_sha256`` places the request under another catalog revision: its
    health view (circuits, throttle windows) is isolated from the default
    revision's while the physical rung load registry is shared.
    """
    authorization = _authorization(catalog_sha256).model_copy(
        update={
            "request_id": request_id,
            "organization_id": organization_id,
            "fair_share_weight": weight,
            "deadline_monotonic": time.monotonic() + 30,
        }
    )
    route = GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
            failover_mode=failover_mode,
            throttle_cache_threshold=throttle_cache_threshold,
            throttle_redial=throttle_redial,
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason=(
            "direct" if reasoning_pinned_deployment_id is None else "reasoning_continuation"
        ),
        reasoning_pinned_deployment_id=reasoning_pinned_deployment_id,
    )
    entry = InflightRequest(
        authorization=authorization,
        route=route,
        request=_request(),
        deadline_monotonic=time.monotonic() + 30,
        affinity_fingerprint=affinity_fingerprint,
        sticky_preferred=sticky_preferred,
    )
    registry.register(entry)
    return entry


def _bounded_pair(
    bound: int,
    *,
    fair_share: bool = False,
) -> tuple[ExactModelDeployment, ExactModelDeployment]:
    """Build a bounded lead rung with an unbounded spill rung behind it."""
    return (
        _deployment(
            "deployment-a",
            connection_sha256="b" * 64,
            dispatch=GatewayRungDispatchPolicy(concurrency_bound=bound, fair_share=fair_share),
        ),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )


class TestLaneSaturation:
    """The worker's default lane bound and refuse-instead-of-overflow (lane_saturation)."""

    def test_default_lane_bound_spills_an_unauthored_rung_sideways(self) -> None:
        """A rung with no authored policy still sheds at the worker's default share."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger, default_lane_bound=1)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(registry, deployments, request_id="request-1")
        _admit(registry, deployments, request_id="request-2")
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        spilled = _start(registry, ordinal=0, request_id="request-2")
        assert spilled["route_depth"] == 1
        assert ledger.started[1]["dispatch_reason"] == "queue_bound"
        assert ledger.started[1]["preferred_deployment_id"] == "deployment-a"
        assert registry.rung_admission_counters() == (1, 0, 0)

    def test_default_lane_bound_refuses_fast_instead_of_overflowing(self) -> None:
        """Every unauthored rung at its default share: a retryable 429, no dispatch."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger, default_lane_bound=1)
        only = (_deployment("deployment-a", connection_sha256="b" * 64),)
        _admit(registry, only, request_id="request-1")
        _admit(registry, only, request_id="request-2")
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        refused = _start(registry, ordinal=0, request_id="request-2")
        assert refused["exhausted"] is True
        failure = cast("JsonObject", refused["failure"])
        assert failure["failure_class"] == "throttled"
        assert failure["retry_after_seconds"] == 5
        assert "in-flight bound" in str(failure["safe_message"])
        assert len(ledger.started) == 1
        assert registry.rung_admission_counters() == (1, 0, 1)
        # The refused request is finished, so the slot it never took frees nothing
        # and the next request after a settle admits again.
        started = ledger.started[0]
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="completed",
            finalize=True,
            request_id="request-1",
        )
        _admit(registry, only, request_id="request-3")
        assert _start(registry, ordinal=0, request_id="request-3")["route_depth"] == 0

    @pytest.mark.parametrize("authored", [False, True])
    @pytest.mark.parametrize("staged", [False, True])
    def test_reasoning_pin_never_bypasses_a_refusing_lane_bound(
        self, authored: bool, staged: bool
    ) -> None:
        """Pinned reasoning cannot force a default or explicitly refusing lane past its bound."""

        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger, default_lane_bound=1)
        policy = (
            GatewayRungDispatchPolicy(concurrency_bound=1, saturation="refuse")
            if authored
            else None
        )
        only = (_deployment("deployment-a", connection_sha256="b" * 64, dispatch=policy),)
        _admit(registry, only, request_id="occupied")
        assert _start(registry, ordinal=0, request_id="occupied")["route_depth"] == 0
        pinned = _admit(
            registry, only, request_id="pinned", reasoning_pinned_deployment_id="deployment-a"
        )
        if staged:
            snapshot = pinned.route.snapshot.model_copy(
                update={
                    "model_stages": (
                        ModelExecutionStage(
                            stage_index=0,
                            exact_model_id="exact-one",
                            pool_id="pool-one",
                            deployment_ids=("deployment-a",),
                        ),
                    )
                }
            )
            pinned.route = pinned.route.model_copy(update={"snapshot": snapshot})
        refused = _start(registry, ordinal=0, request_id="pinned")
        assert refused["exhausted"] is True
        assert cast("JsonObject", refused["failure"])["failure_class"] == "throttled"
        assert registry.rung_admission_counters() == (1, 0, 1)
        assert len(ledger.started) == 1

    @pytest.mark.parametrize("authored", [False, True])
    @pytest.mark.parametrize("conditional_child", [False, True])
    def test_root_child_capacity_refusal_preserves_existing_reservations(
        self, authored: bool, conditional_child: bool
    ) -> None:
        """A full staged ladder cannot overflow or promote a failure-only child after a shed."""

        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger, default_lane_bound=1)
        policy = (
            GatewayRungDispatchPolicy(concurrency_bound=1, saturation="refuse")
            if authored
            else None
        )
        root = _deployment("deployment-a", connection_sha256="b" * 64, dispatch=policy)
        child_policy = GatewayRungDispatchPolicy(
            concurrency_bound=1 if authored else None,
            saturation="refuse" if authored else "overflow",
        )
        child = _deployment(
            "deployment-b", connection_sha256="c" * 64, dispatch=child_policy
        ).model_copy(update={"exact_model_id": "child-model"})
        if conditional_child:
            child = child.model_copy(
                update={
                    "gateway": child.gateway.model_copy(
                        update={
                            "capabilities": child.gateway.capabilities.model_copy(
                                update={
                                    "failover_only_on": frozenset({"throttled"}),
                                }
                            ),
                        }
                    )
                }
            )
        for index, deployment in enumerate((root, child)):
            _admit(registry, (deployment,), request_id=f"occupied-{index}")
            # Occupy only ordinary candidates; a conditional child is never a fresh first dial.
            if not (conditional_child and index == 1):
                assert (
                    _start(registry, ordinal=0, request_id=f"occupied-{index}")["route_depth"] == 0
                )
        before = len(ledger.started)
        entry = _admit(registry, (root, child), request_id="full-chain")
        snapshot = entry.route.snapshot.model_copy(
            update={
                "model_stages": (
                    ModelExecutionStage(
                        stage_index=0,
                        exact_model_id=root.exact_model_id,
                        pool_id="pool-one",
                        deployment_ids=(root.deployment_id,),
                    ),
                    ModelExecutionStage(
                        stage_index=1,
                        exact_model_id=child.exact_model_id,
                        pool_id="child-pool",
                        deployment_ids=(child.deployment_id,),
                    ),
                )
            }
        )
        entry.route = entry.route.model_copy(update={"snapshot": snapshot})
        entry.verified_warm_deployment_id = child.deployment_id
        entry.verified_warm_until_monotonic = time.monotonic() + 30
        entry.recovery_scoped = True
        refused = _start(registry, ordinal=0, request_id="full-chain")
        assert refused["exhausted"] is True
        assert len(ledger.started) == before
        assert registry.entry("full-chain") is None
        assert registry.loads.inflight(rung_load_key(root)) == 1
        assert registry.loads.inflight(rung_load_key(child)) == int(not conditional_child)
        assert registry.rung_admission_counters()[1:] == (0, 1)

    @pytest.mark.parametrize("competing_fill", [False, True])
    def test_ordinary_rate_overflow_uses_latest_hard_shed(
        self, monkeypatch: pytest.MonkeyPatch, competing_fill: bool
    ) -> None:
        """A competing capacity fill replaces the old rate shed and terminates without spinning."""

        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger, default_lane_bound=1)
        deployment = _deployment(
            "deployment-a",
            connection_sha256="b" * 64,
            dispatch=GatewayRungDispatchPolicy(requests_per_minute=1),
        )
        key = rung_load_key(deployment)
        _admit(registry, (deployment,), request_id="spent-rate")
        first = _start(registry, ordinal=0, request_id="spent-rate")
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="completed",
            finalize=True,
            request_id="spent-rate",
        )
        assert registry.loads.inflight(key) == 0
        entry = _admit(registry, (deployment,), request_id="ordinary-overflow")
        reserve = registry._reserve_rung_slot
        calls: list[tuple[bool, bool, str]] = []
        competitor: list[str] = []

        def interleaved_reserve(
            request_entry: InflightRequest,
            selected: ExactModelDeployment,
            *,
            reserved_tokens: int,
            force: bool,
            rate_retry: bool = False,
        ) -> str | RungShed | None:
            """Fill capacity after the initial rate decision and fail on a repeated hard refusal."""
            assert len(calls) < 2, "stale shed retried a hard-refused overflow target"
            result = reserve(
                request_entry,
                selected,
                reserved_tokens=reserved_tokens,
                force=force,
                rate_retry=rate_retry,
            )
            calls.append(
                (force, rate_retry, result.reason if isinstance(result, RungShed) else "admitted")
            )
            if len(calls) == 1 and competing_fill:
                assert isinstance(result, RungShed) and result.reason == "rate_limit"
                ticket = registry.loads.reserve(
                    key,
                    organization_id="competitor",
                    weight=1,
                    bound=1,
                    fair_share=False,
                )
                assert isinstance(ticket, str)
                competitor.append(ticket)
            return result

        monkeypatch.setattr(registry, "_reserve_rung_slot", interleaved_reserve)
        result = _start(registry, ordinal=0, request_id=entry.authorization.request_id)
        assert calls == [
            (False, False, "rate_limit"),
            (True, False, "queue_bound" if competing_fill else "admitted"),
        ]
        if competing_fill:
            assert result["exhausted"] is True
            assert cast("JsonObject", result["failure"])["failure_class"] == "throttled"
            assert len(ledger.started) == 1 and len(ledger.finished_requests) == 1
            assert registry.entry(entry.authorization.request_id) is None
            assert registry.rung_admission_counters() == (2, 0, 1)
            assert registry.loads.inflight(key) == 1
            registry.loads.release_ticket(competitor[0])
            assert registry.loads.inflight(key) == 0
            calls.clear()
            monkeypatch.setattr(registry, "_reserve_rung_slot", reserve)
            _admit(registry, (deployment,), request_id="after-release")
            accepted = _start(registry, ordinal=0, request_id="after-release")
            assert accepted["route_depth"] == 0 and len(ledger.started) == 2
            assert registry.rung_admission_counters() == (3, 1, 1)
            _settle(
                registry,
                attempt_id=str(accepted["attempt_id"]),
                outcome="completed",
                finalize=True,
                request_id="after-release",
            )
        else:
            assert result["route_depth"] == 0 and len(ledger.started) == 2
            assert registry.rung_admission_counters() == (1, 1, 0)
            _settle(
                registry,
                attempt_id=str(result["attempt_id"]),
                outcome="completed",
                finalize=True,
                request_id=entry.authorization.request_id,
            )
        assert registry.loads.inflight(key) == 0

    @pytest.mark.parametrize("interruption", ["deadline", "cancel"])
    def test_lane_reselection_observes_request_interruption(
        self, monkeypatch: pytest.MonkeyPatch, interruption: str
    ) -> None:
        """A request ending after its initial shed cannot dispatch through a later selection."""

        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployment = _deployment(
            "deployment-a",
            connection_sha256="b" * 64,
            dispatch=GatewayRungDispatchPolicy(requests_per_minute=1),
        )
        _admit(registry, (deployment,), request_id="spent-rate")
        first = _start(registry, ordinal=0, request_id="spent-rate")
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="completed",
            finalize=True,
            request_id="spent-rate",
        )
        entry = _admit(registry, (deployment,), request_id="interrupted")
        reserve = registry._reserve_rung_slot
        calls = 0

        def interrupt_after_shed(
            request_entry: InflightRequest,
            selected: ExactModelDeployment,
            *,
            reserved_tokens: int,
            force: bool,
            rate_retry: bool = False,
        ) -> str | RungShed | None:
            """Advance the deadline or durably abandon before the next selection."""
            nonlocal calls
            calls += 1
            assert calls == 1, "interrupted request retried its reservation"
            result = reserve(
                request_entry,
                selected,
                reserved_tokens=reserved_tokens,
                force=force,
                rate_retry=rate_retry,
            )
            assert isinstance(result, RungShed)
            if interruption == "deadline":
                entry.deadline_monotonic = time.monotonic() - 1
            else:
                registry.abandon(json.dumps({"request_id": entry.authorization.request_id}))
            return result

        monkeypatch.setattr(registry, "_reserve_rung_slot", interrupt_after_shed)
        result = _start(registry, ordinal=0, request_id=entry.authorization.request_id)
        assert result["exhausted"] is True
        assert cast("JsonObject", result["failure"])["failure_class"] == (
            "timeout" if interruption == "deadline" else "cancelled"
        )
        assert calls == 1 and len(ledger.started) == 1
        assert len(ledger.finished_requests) == 1
        assert registry.entry(entry.authorization.request_id) is None

    def test_authored_bound_keeps_the_default_on_its_unauthored_sibling(self) -> None:
        """The authored bound wins on its rung; the sibling gets the worker default."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger, default_lane_bound=1)
        deployments = (
            _deployment(
                "deployment-a",
                connection_sha256="b" * 64,
                dispatch=GatewayRungDispatchPolicy(concurrency_bound=2),
            ),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        for request_id in ("request-1", "request-2", "request-3"):
            _admit(registry, deployments, request_id=request_id)
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        assert _start(registry, ordinal=0, request_id="request-2")["route_depth"] == 0
        # Both slots of the authored bound are held; the third spills to the
        # sibling, whose own (default) bound of one is still free.
        assert _start(registry, ordinal=0, request_id="request-3")["route_depth"] == 1
        assert registry.rung_admission_counters() == (1, 0, 0)

    def test_authored_refuse_saturation_replaces_the_overflow(self) -> None:
        """``saturation="refuse"`` on a single authored rung refuses rather than overflows."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        only = (
            _deployment(
                "deployment-a",
                connection_sha256="b" * 64,
                dispatch=GatewayRungDispatchPolicy(concurrency_bound=1, saturation="refuse"),
            ),
        )
        _admit(registry, only, request_id="request-1")
        _admit(registry, only, request_id="request-2")
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        refused = _start(registry, ordinal=0, request_id="request-2")
        assert refused["exhausted"] is True
        assert cast("JsonObject", refused["failure"])["failure_class"] == "throttled"
        assert registry.rung_admission_counters() == (1, 0, 1)


@pytest.mark.parametrize(
    "dispatch",
    [
        None,
        GatewayRungDispatchPolicy(concurrency_bound=1),
        GatewayRungDispatchPolicy(concurrency_bound=1, saturation="refuse"),
        GatewayRungDispatchPolicy(requests_per_minute=1),
    ],
)
@pytest.mark.parametrize("reasoning_pinned", [False, True])
def test_selected_first_route_refuses_load_shed_without_spill_or_overflow(
    dispatch: GatewayRungDispatchPolicy | None, reasoning_pinned: bool
) -> None:
    """A caller selector neither moves sideways nor overrides any lane load ceiling."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger, default_lane_bound=1)
    deployments = (
        _deployment("deployment-a", connection_sha256="b" * 64, dispatch=dispatch),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )
    _admit(registry, deployments, request_id="occupied")
    selected = _admit(
        registry,
        deployments,
        request_id="selected",
        reasoning_pinned_deployment_id="deployment-a" if reasoning_pinned else None,
    )
    selected.route = selected.route.model_copy(update={"resolved_route_id": "route_" + "a" * 64})
    assert _start(registry, ordinal=0, request_id="occupied")["route_depth"] == 0
    if dispatch is not None and dispatch.requests_per_minute is not None:
        # Release concurrency so this arm exercises only the retained rate window.
        _settle(
            registry,
            attempt_id=str(ledger.started[0]["attempt_id"]),
            outcome="completed",
            finalize=True,
            request_id="occupied",
        )
    refused = _start(registry, ordinal=0, request_id="selected")
    assert refused["exhausted"] is True
    failure = refused["failure"]
    assert isinstance(failure, dict)
    assert failure["failure_class"] == "throttled"
    assert failure["retry_after_seconds"] == THROTTLED_RETRY_AFTER_SECONDS
    assert len(ledger.started) == 1
    assert registry.rung_admission_counters() == (1, 0, 1)
    assert registry.entry("selected") is None


@pytest.mark.parametrize("probe_occupied", [False, True])
def test_selected_first_route_does_not_force_an_open_health_circuit(
    probe_occupied: bool,
) -> None:
    """Selecting a suppressed lead never converts a healthy sibling into circuit override."""
    registry, ledger, selected = _registry()
    selected.route = selected.route.model_copy(update={"resolved_route_id": "route_" + "a" * 64})
    key = deployment_health_key(selected.authorization, selected.route.deployment)
    registry.health.failed(
        key,
        GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
            safe_message="provider rejected its credential",
        ),
    )
    if probe_occupied:
        assert registry.health.claim_last_resort(key)
    refused = _start(registry, ordinal=0)
    assert refused["exhausted"] is True
    failure = refused["failure"]
    assert isinstance(failure, dict)
    assert failure["failure_class"] == "provider_internal"
    assert not ledger.started
    assert registry.entry(selected.authorization.request_id) is None
    ordinary = _admit(registry, selected.route.deployments, request_id="ordinary")
    assert _start(registry, ordinal=0, request_id="ordinary")["route_depth"] == 1
    assert ordinary.total_attempts == 1


class TestRungDispatchPolicy:
    """Bounded-queue spill, fair-share sheds, overflow, and their disclosures."""

    def test_bound_spills_to_the_next_rung_with_disclosure(self) -> None:
        """The dispatch past the bound lands on the spill rung, disclosed."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(1)
        _admit(registry, deployments, request_id="request-1")
        _admit(registry, deployments, request_id="request-2")
        first = _start(registry, ordinal=0, request_id="request-1")
        assert first["route_depth"] == 0
        assert ledger.started[0]["dispatch_reason"] is None
        assert ledger.started[0]["preferred_deployment_id"] is None
        spilled = _start(registry, ordinal=0, request_id="request-2")
        assert spilled["route_depth"] == 1
        assert ledger.started[1]["dispatch_reason"] == "queue_bound"
        assert ledger.started[1]["preferred_deployment_id"] == "deployment-a"
        assert registry.rung_admission_counters() == (1, 0, 0)

    def test_settle_frees_the_bounded_slot(self) -> None:
        """A settled dispatch returns its slot so the next request is not shed."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(1)
        _admit(registry, deployments, request_id="request-1")
        _admit(registry, deployments, request_id="request-2")
        started = _start(registry, ordinal=0, request_id="request-1")
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="completed",
            finalize=True,
            request_id="request-1",
        )
        follow = _start(registry, ordinal=0, request_id="request-2")
        assert follow["route_depth"] == 0
        assert registry.rung_admission_counters() == (0, 0, 0)

    def test_saturated_overflow_never_manufactures_a_failure(self) -> None:
        """A single-rung pool at its bound still dispatches, disclosed as such."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        only = (
            _deployment(
                "deployment-a",
                connection_sha256="b" * 64,
                dispatch=GatewayRungDispatchPolicy(concurrency_bound=1),
            ),
        )
        _admit(registry, only, request_id="request-1")
        _admit(registry, only, request_id="request-2")
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        overflow = _start(registry, ordinal=0, request_id="request-2")
        assert overflow["route_depth"] == 0
        assert ledger.started[1]["dispatch_reason"] == "saturated_overflow"
        assert ledger.started[1]["preferred_deployment_id"] is None
        assert registry.rung_admission_counters() == (1, 1, 0)

    def test_fair_share_shed_discloses_and_spills(self) -> None:
        """An over-share organization spills while the under-share one admits."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(4, fair_share=True)
        for index in range(1, 4):
            _admit(registry, deployments, request_id=f"a-{index}", organization_id="org-a")
            assert _start(registry, ordinal=0, request_id=f"a-{index}")["route_depth"] == 0
        # org-b admits its first (total 4, at the bound afterwards)...
        _admit(registry, deployments, request_id="b-1", organization_id="org-b")
        assert _start(registry, ordinal=0, request_id="b-1")["route_depth"] == 0
        # ...one org-a request settles, freeing a slot reserved for org-b.
        settled = ledger.started[0]
        _settle(
            registry,
            attempt_id=str(settled["attempt_id"]),
            outcome="completed",
            finalize=True,
            request_id="a-1",
        )
        _admit(registry, deployments, request_id="a-4", organization_id="org-a")
        shed = _start(registry, ordinal=0, request_id="a-4")
        assert shed["route_depth"] == 1
        assert ledger.started[-1]["dispatch_reason"] == "fair_share_shed"
        assert ledger.started[-1]["preferred_deployment_id"] == "deployment-a"
        # The under-share organization still lands on the house rung.
        _admit(registry, deployments, request_id="b-2", organization_id="org-b")
        assert _start(registry, ordinal=0, request_id="b-2")["route_depth"] == 0

    def test_affinity_pool_discloses_every_attempt(self) -> None:
        """Affinity pools stamp the happy path and name a dead preferred rung."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache_affinity",
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        assert first["route_depth"] == 0
        assert ledger.started[0]["dispatch_reason"] == "affinity"
        assert ledger.started[0]["preferred_deployment_id"] is None
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure={
                "failure_class": "provider_internal",
                "safe_message": "provider failed",
                "failover_eligible": True,
            },
            request_id="request-1",
        )
        failover = _start(
            registry,
            ordinal=1,
            current_depth=0,
            failure={
                "failure_class": "provider_internal",
                "safe_message": "provider failed",
                "failover_eligible": True,
            },
            request_id="request-1",
        )
        assert failover["route_depth"] == 1
        assert ledger.started[1]["dispatch_reason"] == "rung_dead"
        assert ledger.started[1]["preferred_deployment_id"] == "deployment-a"

    def test_affinity_throttle_fails_over_unlike_maximize_cache(self) -> None:
        """A throttle on an affinity pool spills to the deterministic alternate."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache_affinity",
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        throttle: JsonObject = {
            "failure_class": "throttled",
            "safe_message": "provider throttled the request",
            "failover_eligible": True,
        }
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=throttle,
            request_id="request-1",
        )
        failover = _start(
            registry, ordinal=1, current_depth=0, failure=throttle, request_id="request-1"
        )
        assert failover["route_depth"] == 1

    def test_flag_off_attempts_carry_no_disclosures_or_load_state(self) -> None:
        """Untouched pools keep null disclosure fields and an empty registry."""
        registry, ledger, _entry = _registry()
        started = _start(registry, ordinal=0)
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_retryable_failure(),
        )
        _start(registry, ordinal=1, current_depth=0, failure=_retryable_failure())
        assert all(row["dispatch_reason"] is None for row in ledger.started)
        assert all(row["preferred_deployment_id"] is None for row in ledger.started)
        assert registry.loads.inflight(("deployment-a", "b" * 64)) == 0
        assert registry.rung_admission_counters() == (0, 0, 0)

    def test_budget_skip_releases_the_reserved_slot(self) -> None:
        """A deployment-budget rejection frees the rung's bounded reservation."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(1)
        ledger.budget_rejections["deployment-a"] = BudgetScopeKind.DEPLOYMENT
        _admit(registry, deployments, request_id="request-1")
        started = _start(registry, ordinal=0, request_id="request-1")
        assert started["route_depth"] == 1
        assert registry.loads.inflight(("deployment-a", "b" * 64)) == 0

    def test_abandon_releases_the_reserved_slot(self) -> None:
        """An abandoned active attempt frees its rung slot for new arrivals."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(1)
        _admit(registry, deployments, request_id="request-1")
        _admit(registry, deployments, request_id="request-2")
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        assert registry.abandon(json.dumps({"request_id": "request-1"})) == "{}"
        assert registry.loads.inflight(("deployment-a", "b" * 64)) == 0
        assert _start(registry, ordinal=0, request_id="request-2")["route_depth"] == 0


def test_provider_detail_crosses_the_boundary_only_as_a_string() -> None:
    """The provider explanation survives the failure payload decode."""
    registry, _ledger, _entry = _registry()
    _start(registry, ordinal=0)
    exhausted = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure={
            "failure_class": "invalid_request",
            "safe_message": "provider rejected the request",
            "provider_detail": "`top_p` is deprecated for this model.",
        },
    )
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["provider_detail"] == "`top_p` is deprecated for this model."
    numeric = failure_from_boundary_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "provider_detail": 7}
    )
    assert numeric is not None and numeric.provider_detail is None
    empty_detail = failure_from_boundary_payload(
        {"failure_class": "invalid_request", "safe_message": "x", "provider_detail": ""}
    )
    assert empty_detail is not None and empty_detail.provider_detail is None


def test_deployment_priced_for_service_tier_overrides_only_for_a_carried_tier() -> None:
    """A requested tier with a pass-through card reprices the deployment copy;
    no tier, or a tier the deployment lacks, returns the deployment unchanged."""
    from exp.common.models.catalog import (
        GatewayDeploymentMetadata,
        GatewayServiceTierPrices,
        GatewayTokenPrices,
    )
    from exp.common.models.gateway_catalog import ExactModelDeployment
    from exp.runtime.gateway.native_execution import deployment_priced_for_service_tier

    deployment = ExactModelDeployment(
        deployment_id="d1",
        source_alias="d1",
        exact_model_id="exact-one",
        connection="connection-d1",
        provider="openai",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(
            prices=GatewayTokenPrices(
                input_nano_usd_per_million_tokens=1_000_000,
                output_nano_usd_per_million_tokens=4_000_000,
                flex=GatewayServiceTierPrices(
                    input_nano_usd_per_million_tokens=500_000,
                    output_nano_usd_per_million_tokens=2_000_000,
                ),
            )
        ),
    )

    flex = deployment_priced_for_service_tier(deployment, "flex", forwards_tier=True)
    assert flex is not deployment
    assert flex.gateway.prices.input_nano_usd_per_million_tokens == 500_000
    assert flex.gateway.prices.output_nano_usd_per_million_tokens == 2_000_000
    # Identity and everything else is preserved on the copy.
    assert flex.deployment_id == "d1" and flex.exact_model_id == "exact-one"

    # No tier, default/auto, and a tier the deployment does not carry: unchanged.
    assert deployment_priced_for_service_tier(deployment, None, forwards_tier=False) is deployment
    assert (
        deployment_priced_for_service_tier(deployment, "default", forwards_tier=False) is deployment
    )
    assert (
        deployment_priced_for_service_tier(deployment, "priority", forwards_tier=False)
        is deployment
    )

    # A carded tier that the SELECTED depth does not forward (a card on a lane
    # whose wire would strip the tier) bills the BASE schedule, never the card:
    # forwards_tier=False returns the deployment unchanged even though the flex
    # card exists.
    assert deployment_priced_for_service_tier(deployment, "flex", forwards_tier=False) is deployment


def test_start_attempt_reprices_only_when_the_selected_depth_forwards_the_tier() -> None:
    """The reservation applies the tier card ONLY on a depth that forwards it.

    Regression for the forward/bill divergence: a flex CARD on a lane the
    selected depth does not forward reserves the BASE schedule, never the card,
    so the gateway can never reserve the tier rate while the provider runs the
    base schedule.
    """
    from exp.common.models.catalog import GatewayServiceTierPrices, GatewayTokenPrices

    class _PriceCapturingLedger(_RecordingLedger):
        """Recording ledger that also captures each reserved input rate."""

        def __init__(self) -> None:
            """Track the per-attempt reserved input rate alongside the base log."""
            super().__init__()
            self.reserved_input_micro: list[int | None] = []

        def start_attempt(
            self,
            *,
            snapshot: object,
            deployment: ExactModelDeployment,
            attempt_ordinal: int,
            route_depth: int,
            maximum_cost_nano_usd: int | None = None,
            reserved_input_tokens: int | None = None,
            reserved_output_tokens: int | None = None,
            route_reason: str | None = None,
            fallback_reason: str | None = None,
            dispatch_reason: str | None = None,
            preferred_deployment: ExactModelDeployment | None = None,
        ) -> str:
            """Record the reserved input rate, then reserve as the base fake does."""
            self.reserved_input_micro.append(
                deployment.gateway.prices.input_nano_usd_per_million_tokens
            )
            return super().start_attempt(
                snapshot=snapshot,
                deployment=deployment,
                attempt_ordinal=attempt_ordinal,
                route_depth=route_depth,
                maximum_cost_nano_usd=maximum_cost_nano_usd,
                reserved_input_tokens=reserved_input_tokens,
                reserved_output_tokens=reserved_output_tokens,
                route_reason=route_reason,
                fallback_reason=fallback_reason,
                dispatch_reason=dispatch_reason,
                preferred_deployment=preferred_deployment,
            )

    carded = _deployment("deployment-a", connection_sha256="b" * 64).model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(
                capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
                prices=GatewayTokenPrices(
                    input_nano_usd_per_million_tokens=1_000_000,
                    output_nano_usd_per_million_tokens=4_000_000,
                    flex=GatewayServiceTierPrices(
                        input_nano_usd_per_million_tokens=500_000,
                        output_nano_usd_per_million_tokens=2_000_000,
                    ),
                ),
            )
        }
    )
    route = _route((carded,))
    flex_request = _request().model_copy(update={"service_tier": "flex"})

    def _reserved_rate(*, forwards: bool) -> int | None:
        ledger = _PriceCapturingLedger()
        registry = NativeAttemptAccounting(ledger)  # type: ignore[arg-type]
        registry.register(
            InflightRequest(
                authorization=route.snapshot.authorization,
                route=route,
                request=flex_request,
                deadline_monotonic=time.monotonic() + 30,
                tier_forwarded_by_depth=(forwards,),
            )
        )
        _start(registry, ordinal=0)
        assert len(ledger.reserved_input_micro) == 1
        return ledger.reserved_input_micro[0]

    # The selected depth forwards flex -> reserve at the flex card rate.
    assert _reserved_rate(forwards=True) == 500_000
    # Same flex card, but the selected depth strips the tier -> reserve at BASE.
    assert _reserved_rate(forwards=False) == 1_000_000


def test_customer_owned_failures_round_trip_and_file_as_the_callers_invalid_request() -> None:
    """A BYOK credential failure keeps its ladder class, echoes its ownership, and
    is recorded as the caller's invalid request."""
    parsed = failure_from_boundary_payload(
        {
            "failure_class": "provider_authentication",
            "safe_message": "your connected openai credential was rejected by the provider",
            "failover_eligible": True,
            "customer_owned": True,
        }
    )
    assert parsed is not None
    assert parsed.customer_owned is True
    assert parsed.failure_class == GatewayFailureClass.PROVIDER_AUTHENTICATION
    assert ledger_failure(parsed).failure_class == GatewayFailureClass.INVALID_REQUEST
    # Only the two customer-configurable provider classes re-file; a
    # house-shaped failure (or one without the flag) is untouched.
    house = failure_from_boundary_payload(
        {
            "failure_class": "provider_authentication",
            "safe_message": "provider authentication failed",
        }
    )
    assert house is not None and ledger_failure(house).failure_class is (
        GatewayFailureClass.PROVIDER_AUTHENTICATION
    )

    registry, _ledger, _entry = _registry()
    _start(registry, ordinal=0)
    exhausted = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure={
            "failure_class": "provider_quota",
            "safe_message": "your connected openrouter account has exhausted its quota",
            "customer_owned": True,
        },
    )
    failure_payload = exhausted["failure"]
    assert isinstance(failure_payload, dict)
    assert failure_payload["customer_owned"] is True
    assert failure_payload["failure_class"] == "provider_quota"


def _rated_pair(
    *,
    requests_per_minute: int | None = None,
    tokens_per_minute: int | None = None,
) -> tuple[ExactModelDeployment, ExactModelDeployment]:
    """Build a rate-capped lead rung with an unlimited spill rung behind it."""
    return (
        _deployment(
            "deployment-a",
            connection_sha256="b" * 64,
            dispatch=GatewayRungDispatchPolicy(
                requests_per_minute=requests_per_minute,
                tokens_per_minute=tokens_per_minute,
            ),
        ),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )


class TestRateLimitSheds:
    """Rate windows spill sideways pre-429 and force-admit at exhaustion."""

    def test_request_rate_shed_spills_sideways_with_disclosure(self) -> None:
        """The over-rate dispatch lands on the next rung, disclosed verbatim."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _rated_pair(requests_per_minute=1)
        _admit(registry, deployments, request_id="request-1")
        _admit(registry, deployments, request_id="request-2")
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        spilled = _start(registry, ordinal=0, request_id="request-2")
        assert spilled["route_depth"] == 1
        assert ledger.started[1]["dispatch_reason"] == "rate_limit"
        assert ledger.started[1]["preferred_deployment_id"] == "deployment-a"
        assert registry.rung_admission_counters() == (1, 0, 0)
        assert registry.rung_rate_counters() == (1, 0)

    def test_rate_shed_force_admits_a_reasoning_pinned_rung_until_a_real_failure(self) -> None:
        """A pinned continuation never spills to a stripped fallback on a policy shed.

        The issuing rung's per-worker rate window is already used by another
        request; the continuation is still force-admitted THERE
        (``saturated_overflow``), because its fallbacks run without the
        request's thinking and a rate fact trips under ordinary load. A real
        failover-eligible throttle on that attempt then advances to the
        fallback, recorded as ``reasoning_continuation_failover``.
        """
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _rated_pair(requests_per_minute=1)
        _admit(registry, deployments, request_id="request-1")
        _admit(
            registry,
            deployments,
            request_id="request-2",
            reasoning_pinned_deployment_id="deployment-a",
        )
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        kept = _start(registry, ordinal=0, request_id="request-2")
        assert kept["route_depth"] == 0
        assert ledger.started[1]["deployment_id"] == "deployment-a"
        assert ledger.started[1]["dispatch_reason"] == "saturated_overflow"
        assert ledger.started[1]["route_reason"] == "reasoning_continuation"
        assert registry.rung_admission_counters() == (1, 1, 0)
        throttled: JsonObject = {
            "failure_class": "throttled",
            "safe_message": "provider throttled the request",
            "retryable_same_deployment": False,
            "failover_eligible": True,
        }
        _settle(
            registry,
            attempt_id=str(kept["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=throttled,
            request_id="request-2",
        )
        advanced = _start(
            registry, ordinal=1, current_depth=0, failure=throttled, request_id="request-2"
        )
        assert advanced["route_depth"] == 1
        assert ledger.started[2]["deployment_id"] == "deployment-b"
        assert ledger.started[2]["route_reason"] == "reasoning_continuation_failover"
        assert ledger.started[2]["dispatch_reason"] != "saturated_overflow"

    def test_token_rate_counts_the_worst_case_reservation(self) -> None:
        """An over-cap request bursts into an empty window; the next one spills.

        The burst allowance keeps a token cap below one request's worst case
        from becoming a permanent shed loop: the first dispatch lands on the
        rung and occupies the window, and the follow-up spills as a normal
        ``rate_limit`` shed until the window slides.
        """
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _rated_pair(tokens_per_minute=1)
        _admit(registry, deployments, request_id="request-1")
        _admit(registry, deployments, request_id="request-2")
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        spilled = _start(registry, ordinal=0, request_id="request-2")
        assert spilled["route_depth"] == 1
        assert ledger.started[1]["dispatch_reason"] == "rate_limit"

    def test_whole_ladder_rate_limited_still_force_admits(self) -> None:
        """A single rate-capped rung never manufactures a failure."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        only = (
            _deployment(
                "deployment-a",
                connection_sha256="b" * 64,
                dispatch=GatewayRungDispatchPolicy(requests_per_minute=1),
            ),
        )
        _admit(registry, only, request_id="request-1")
        _admit(registry, only, request_id="request-2")
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        overflow = _start(registry, ordinal=0, request_id="request-2")
        assert overflow["route_depth"] == 0
        assert ledger.started[1]["dispatch_reason"] == "saturated_overflow"
        assert registry.rung_admission_counters() == (1, 1, 0)

    def test_whole_ladder_fresh_spill_limited_still_force_admits(self) -> None:
        """A narrow ladder blocked only by the fresh threshold never mints a 429.

        Production showed one org taking hard 429s while its only eligible
        rung sat healthy; both new shed reasons (rate_limit above,
        fresh_session_spill here) must participate in the saturated-overflow
        force-admit so policy sheds can never manufacture a caller failure.
        """
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        only = (
            _deployment(
                "deployment-a",
                connection_sha256="b" * 64,
                dispatch=GatewayRungDispatchPolicy(
                    concurrency_bound=2,
                    fresh_session_spill_fraction=0.5,
                    sticky_spill_seconds=600,
                ),
            ),
        )
        _admit(
            registry,
            only,
            request_id="request-1",
            failover_mode="maximize_cache_affinity",
            affinity_fingerprint=b"conversation-1",
        )
        _admit(
            registry,
            only,
            request_id="request-2",
            failover_mode="maximize_cache_affinity",
            affinity_fingerprint=b"conversation-2",
        )
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        overflow = _start(registry, ordinal=0, request_id="request-2")
        assert overflow["route_depth"] == 0
        assert ledger.started[1]["dispatch_reason"] == "saturated_overflow"
        assert registry.rung_admission_counters() == (1, 1, 0)
        assert registry.rung_rate_counters() == (0, 1)

    def test_throttled_settle_teaches_the_rungs_learned_ceiling(self) -> None:
        """A provider 429 clamps the physical lane's learned request ceiling."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _rated_pair(requests_per_minute=100)
        _admit(registry, deployments, request_id="request-1")
        started = _start(registry, ordinal=0, request_id="request-1")
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="failed",
            finalize=True,
            failure={
                "failure_class": "throttled",
                "safe_message": "provider throttled the request",
                "failover_eligible": True,
            },
            request_id="request-1",
        )
        # One dispatch observed in the window: learned = 1 * 0.9 (a float; the
        # ceiling may sit below one per minute so fleet totals can undershoot).
        assert registry.loads.learned_ceilings() == {"deployment-a:bbbbbbbb": 0.9}

    def test_bound_only_rungs_calibrate_and_unpolicied_rungs_do_not(self) -> None:
        """A bound-only rung learns from its real window; unpolicied lanes never do."""
        throttle: JsonObject = {
            "failure_class": "throttled",
            "safe_message": "provider throttled the request",
            "failover_eligible": True,
        }
        # Bound-only: the reservation fed the window, so the throttle clamps
        # to the observed dispatch rate rather than an empty window's floor.
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(4)
        _admit(registry, deployments, request_id="request-1")
        started = _start(registry, ordinal=0, request_id="request-1")
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="failed",
            finalize=True,
            failure=throttle,
            request_id="request-1",
        )
        assert registry.loads.learned_ceilings() == {"deployment-a:bbbbbbbb": 0.9}
        # Unpolicied: a throttle teaches nothing (nothing would enforce it and
        # the window never observed the lane's rate).
        bare_ledger = _RecordingLedger()
        bare_registry = NativeAttemptAccounting(bare_ledger)
        bare = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(bare_registry, bare, request_id="request-1")
        bare_started = _start(bare_registry, ordinal=0, request_id="request-1")
        _settle(
            bare_registry,
            attempt_id=str(bare_started["attempt_id"]),
            outcome="failed",
            finalize=True,
            failure=throttle,
            request_id="request-1",
        )
        assert bare_registry.loads.learned_ceilings() == {}


class TestRateLimitSettlement:
    """Harvested rate-limit headers reach the ledger and the throttle window."""

    def test_settle_plumbs_harvested_headers_to_the_ledger(self) -> None:
        """The normalized header integers ride the finish_attempt kwargs."""
        registry, ledger, _entry = _registry()
        started = _start(registry, ordinal=0)
        registry.settle(
            json.dumps(
                {
                    "request_id": "request-one",
                    "attempt_id": str(started["attempt_id"]),
                    "outcome": "completed",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "tool_names": [],
                    "failure": None,
                    "finalize": True,
                    "opened": True,
                    "rate_limit_headers": {
                        "x-ratelimit-limit-requests": "10000",
                        "x-ratelimit-remaining-requests": "9999",
                        "x-ratelimit-limit-tokens": "180000000",
                        "x-ratelimit-remaining-tokens": "179000000",
                    },
                }
            )
        )
        assert ledger.rate_limit_settlements == [
            {
                "attempt_id": str(started["attempt_id"]),
                "retry_after_seconds": None,
                "ratelimit_limit_requests": 10_000,
                "ratelimit_remaining_requests": 9_999,
                "ratelimit_limit_tokens": 180_000_000,
                "ratelimit_remaining_tokens": 179_000_000,
            }
        ]

    def test_settle_without_headers_records_no_rate_limit_values(self) -> None:
        """An engine that sends no header map keeps every kwarg None."""
        registry, ledger, _entry = _registry()
        started = _start(registry, ordinal=0)
        _settle(
            registry,
            attempt_id=str(started["attempt_id"]),
            outcome="completed",
            finalize=True,
        )
        assert ledger.rate_limit_settlements == []

    def test_settle_feeds_the_cached_fraction_ewma(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settled cached and input tokens reach the load registry's EWMA."""
        recorded: list[tuple[tuple[str, str], str, int, int]] = []
        registry, _ledger, _entry = _registry()

        def _record(
            key: tuple[str, str],
            organization_id: str,
            *,
            cached_tokens: int,
            input_tokens: int,
        ) -> None:
            """Record one EWMA sample instead of folding it."""
            recorded.append((key, organization_id, cached_tokens, input_tokens))

        monkeypatch.setattr(registry.loads, "record_settle", _record)
        started = _start(registry, ordinal=0)
        settlement = json.dumps(
            {
                "request_id": "request-one",
                "attempt_id": str(started["attempt_id"]),
                "outcome": "completed",
                "usage": {
                    "input_tokens": 1_000,
                    "cached_input_tokens": 800,
                    "output_tokens": 5,
                },
                "tool_names": [],
                "failure": None,
                "finalize": False,
                "opened": True,
            }
        )
        registry.settle(settlement)
        # A redelivered settlement (the ledger write is idempotent) must not
        # fold the same attempt's sample into the EWMA a second time.
        registry.settle(settlement)
        assert recorded == [(("deployment-a", "b" * 64), "organization-one", 800, 1_000)]

    def test_cache_sample_gate_excludes_promo_funded_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hosted gate can veto samples so promo replay cannot buy weight.

        A gate answering False (the host marked the attempt promo-funded) and
        a raising gate both skip the fold; only an admitted attempt records.
        """
        for verdict, folds in (("deny", 0), ("raise", 0), ("admit", 1)):
            recorded: list[str] = []
            ledger = _RecordingLedger()

            def _gate(attempt_id: str, verdict: str = verdict) -> bool:
                """Answer the scripted verdict for every attempt."""
                del attempt_id
                if verdict == "raise":
                    raise RuntimeError("scripted gate failure")
                return verdict == "admit"

            registry = NativeAttemptAccounting(ledger, cache_sample_gate=_gate)
            deployments = (
                _deployment("deployment-a", connection_sha256="b" * 64),
                _deployment("deployment-b", connection_sha256="c" * 64),
            )
            entry = _admit(registry, deployments, request_id="request-1")
            del entry

            def _record(
                key: tuple[str, str],
                organization_id: str,
                *,
                cached_tokens: int,
                input_tokens: int,
                folds: list[str] = recorded,
            ) -> None:
                """Record the fold instead of applying it."""
                del key, organization_id, cached_tokens, input_tokens
                folds.append("fold")

            monkeypatch.setattr(registry.loads, "record_settle", _record)
            started = _start(registry, ordinal=0, request_id="request-1")
            registry.settle(
                json.dumps(
                    {
                        "request_id": "request-1",
                        "attempt_id": str(started["attempt_id"]),
                        "outcome": "completed",
                        "usage": {
                            "input_tokens": 1_000,
                            "cached_input_tokens": 800,
                            "output_tokens": 5,
                        },
                        "tool_names": [],
                        "failure": None,
                        "finalize": True,
                        "opened": True,
                    }
                )
            )
            assert len(recorded) == folds, verdict

    @pytest.mark.parametrize(
        ("surface", "marker", "opened", "expected"),
        [
            (GatewayApiSurface.DECISIONS, True, False, True),
            (GatewayApiSurface.DECISIONS, False, False, False),
            (GatewayApiSurface.DECISIONS, True, True, False),
            (GatewayApiSurface.CHAT_COMPLETIONS, True, False, False),
        ],
    )
    def test_swept_rejection_keeps_exact_scoped_liability_evidence(
        self,
        surface: GatewayApiSurface,
        marker: bool,
        opened: bool,
        expected: bool,
    ) -> None:
        """A failed ledger write must not change a rejection into unknown paid work on retry."""
        registry, ledger, entry = _registry()
        started = _start(registry, ordinal=0)
        entry.authorization = entry.authorization.model_copy(update={"surface": surface})
        ledger.fail_finishes = 1
        settlement = json.dumps(
            {
                "request_id": entry.authorization.request_id,
                "attempt_id": str(started["attempt_id"]),
                "outcome": "failed",
                "usage": None,
                "failure": {"failure_class": "provider_internal", "safe_message": "rejected"},
                "finalize": True,
                "opened": opened,
                "decision_provider_rejected": marker,
            }
        )
        with pytest.raises(NativeBridgeError):
            registry.settle(settlement)
        first = ledger.terminal_events[-1]
        assert first is not None and first.decision_provider_rejected is expected
        assert entry.pending_settlement is not None
        registry.sweep_expired()
        recovered = ledger.terminal_events[-1]
        assert recovered is not None and recovered.decision_provider_rejected is expected
        assert recovered.usage is None
        assert entry.pending_settlement is None
        assert len(ledger.finished) == 1

    def test_swept_retained_settlement_still_records_the_cache_fraction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A settlement recovered by the sweep feeds the EWMA like a direct one."""
        recorded: list[tuple[tuple[str, str], str, int, int]] = []
        registry, ledger, _entry = _registry()

        def _record(
            key: tuple[str, str],
            organization_id: str,
            *,
            cached_tokens: int,
            input_tokens: int,
        ) -> None:
            """Record one EWMA sample instead of folding it."""
            recorded.append((key, organization_id, cached_tokens, input_tokens))

        monkeypatch.setattr(registry.loads, "record_settle", _record)
        started = _start(registry, ordinal=0)
        ledger.fail_finishes = 1
        settlement = json.dumps(
            {
                "request_id": "request-one",
                "attempt_id": str(started["attempt_id"]),
                "outcome": "completed",
                "usage": {
                    "input_tokens": 1_000,
                    "cached_input_tokens": 800,
                    "output_tokens": 5,
                },
                "tool_names": [],
                "failure": None,
                "finalize": True,
                "opened": True,
                "rate_limit_headers": {"x-ratelimit-remaining-requests": "9999"},
            }
        )
        with pytest.raises(NativeBridgeError):
            registry.settle(settlement)
        assert recorded == []
        registry.sweep_expired()
        assert recorded == [(("deployment-a", "b" * 64), "organization-one", 800, 1_000)]
        # The harvested rate-limit values ride the swept write too.
        assert ledger.rate_limit_settlements == [
            {
                "attempt_id": str(started["attempt_id"]),
                "retry_after_seconds": None,
                "ratelimit_limit_requests": None,
                "ratelimit_remaining_requests": 9_999,
                "ratelimit_limit_tokens": None,
                "ratelimit_remaining_tokens": None,
            }
        ]


class TestStickySpillBindings:
    """Dispatches record conversation bindings; disclosures name sticky leads."""

    def test_affinity_dispatch_binds_the_fingerprint_to_its_rung(self) -> None:
        """A sticky-enabled rung records where the conversation's cache lives."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment(
                "deployment-a",
                connection_sha256="b" * 64,
                dispatch=GatewayRungDispatchPolicy(sticky_spill_seconds=600),
            ),
            _deployment(
                "deployment-b",
                connection_sha256="c" * 64,
                dispatch=GatewayRungDispatchPolicy(sticky_spill_seconds=600),
            ),
        )
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache_affinity",
            affinity_fingerprint=b"conversation-1",
        )
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        assert registry.sticky.bound_deployment(b"conversation-1") == "deployment-a"
        # A spilled dispatch of another conversation binds to the spill rung.
        bounded = (
            deployments[0].model_copy(
                update={
                    "gateway": deployments[0].gateway.model_copy(
                        update={
                            "dispatch": GatewayRungDispatchPolicy(
                                concurrency_bound=1, sticky_spill_seconds=600
                            )
                        }
                    )
                }
            ),
            deployments[1],
        )
        _admit(
            registry,
            bounded,
            request_id="request-2",
            failover_mode="maximize_cache_affinity",
            affinity_fingerprint=b"conversation-2",
        )
        _admit(
            registry,
            bounded,
            request_id="request-3",
            failover_mode="maximize_cache_affinity",
            affinity_fingerprint=b"conversation-3",
        )
        assert _start(registry, ordinal=0, request_id="request-2")["route_depth"] == 0
        spilled = _start(registry, ordinal=0, request_id="request-3")
        assert spilled["route_depth"] == 1
        assert registry.sticky.bound_deployment(b"conversation-3") == "deployment-b"

    def test_rung_without_sticky_lifetime_records_no_binding(self) -> None:
        """No authored lifetime means no binding, even on an affinity pool."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache_affinity",
            affinity_fingerprint=b"conversation-1",
        )
        _start(registry, ordinal=0, request_id="request-1")
        assert registry.sticky.size() == 0

    def test_sticky_lead_discloses_affinity_sticky(self) -> None:
        """A route whose depth 0 was sticky-chosen names the binding, not rendezvous."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache_affinity",
            affinity_fingerprint=b"conversation-1",
            sticky_preferred=True,
        )
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        assert ledger.started[0]["dispatch_reason"] == "affinity_sticky"
        assert ledger.started[0]["preferred_deployment_id"] is None


class TestFreshSessionSpillDispatch:
    """The early threshold spills fresh sessions and keeps warm ones home."""

    def test_fresh_session_sheds_early_while_a_warm_session_admits(self) -> None:
        """At the early threshold the fresh session spills, the bound one stays."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment(
                "deployment-a",
                connection_sha256="b" * 64,
                dispatch=GatewayRungDispatchPolicy(
                    concurrency_bound=2,
                    fresh_session_spill_fraction=0.5,
                    sticky_spill_seconds=600,
                ),
            ),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        # A first (fresh) conversation occupies the sub-threshold slot and, by
        # dispatching, becomes warm on the rung.
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache_affinity",
            affinity_fingerprint=b"warm-conversation",
        )
        assert _start(registry, ordinal=0, request_id="request-1")["route_depth"] == 0
        # A second fresh conversation hits the early threshold (1 >= 2 * 0.5)
        # and spills, disclosed as a fresh-session spill...
        _admit(
            registry,
            deployments,
            request_id="request-2",
            failover_mode="maximize_cache_affinity",
            affinity_fingerprint=b"fresh-conversation",
        )
        spilled = _start(registry, ordinal=0, request_id="request-2")
        assert spilled["route_depth"] == 1
        assert ledger.started[1]["dispatch_reason"] == "fresh_session_spill"
        assert ledger.started[1]["preferred_deployment_id"] == "deployment-a"
        assert registry.rung_rate_counters() == (0, 1)
        # ...while the warm conversation's next turn rides to the hard bound.
        _admit(
            registry,
            deployments,
            request_id="request-3",
            failover_mode="maximize_cache_affinity",
            affinity_fingerprint=b"warm-conversation",
        )
        assert _start(registry, ordinal=0, request_id="request-3")["route_depth"] == 0


_THROTTLE: JsonObject = {
    "failure_class": "throttled",
    "safe_message": "provider throttled the request",
    "failover_eligible": True,
}


def _settle_with_usage(
    registry: NativeAttemptAccounting,
    *,
    attempt_id: str,
    request_id: str,
    cached_input_tokens: int,
    input_tokens: int,
) -> None:
    """Settle one completed attempt with observed usage, finalizing its request."""
    registry.settle(
        json.dumps(
            {
                "request_id": request_id,
                "attempt_id": attempt_id,
                "outcome": "completed",
                "usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_input_tokens,
                    "output_tokens": 5,
                },
                "tool_names": [],
                "failure": None,
                "finalize": True,
                "opened": True,
            }
        )
    )


@pytest.mark.parametrize("mode", ["maximize_availability", "maximize_cache_affinity"])
def test_recovery_placement_reason_does_not_replace_throttle_redial_reason(
    mode: FailoverMode,
) -> None:
    """Initial recovery placement and a later physical backoff remain distinguishable."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    deployments = (_deployment("deployment-a", connection_sha256="b" * 64),)
    entry = _admit(
        registry,
        deployments,
        request_id="request-1",
        failover_mode=mode,
        throttle_redial=GatewayThrottleRedialPolicy(
            max_attempts=1, base_delay_ms=100, max_delay_ms=100
        ),
    )
    entry.recovery_reason = "retained_warm_fallback"
    first = _start(registry, ordinal=0, request_id="request-1")
    assert ledger.started[0]["dispatch_reason"] == "retained_warm_fallback"
    _settle(
        registry,
        attempt_id=str(first["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=_THROTTLE,
        request_id="request-1",
    )
    redial = _start(
        registry,
        ordinal=1,
        current_depth=0,
        failure=_THROTTLE,
        throttle_backoff=True,
        request_id="request-1",
    )
    assert redial["route_depth"] == 0
    assert ledger.started[1]["dispatch_reason"] == "throttle_backoff"
    assert ledger.started[1]["preferred_deployment_id"] is None
    registry.abandon(json.dumps({"request_id": "request-1"}))


def test_recovery_placement_reason_does_not_replace_forced_overflow() -> None:
    """A retained route forced past its capacity reports the real admission override."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    deployments = (_bounded_pair(1)[0],)
    _admit(registry, deployments, request_id="holder")
    _start(registry, ordinal=0, request_id="holder")
    entry = _admit(
        registry, deployments, request_id="overflow", failover_mode="maximize_cache_affinity"
    )
    entry.recovery_reason = "retained_warm_fallback"
    assert _start(registry, ordinal=0, request_id="overflow")["route_depth"] == 0
    assert ledger.started[-1]["dispatch_reason"] == "saturated_overflow"
    assert registry.rung_admission_counters() == (1, 1, 0)
    for request_id in ("holder", "overflow"):
        registry.abandon(json.dumps({"request_id": request_id}))


class TestThrottleCacheThreshold:
    """The per-request cache-stakes throttle decision and its disclosures."""

    def test_cold_failover_discloses_the_throttled_rung_on_the_next_attempt(self) -> None:
        """Below the threshold a throttle fails over, disclosed as throttle_failover_cold.

        The organization has no cache evidence on the lead rung, so the
        fraction reads 0 and the request advances; the cold attempt names the
        throttled warm rung as its bypassed preferred counterfactual.
        """
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache",
            throttle_cache_threshold=0.5,
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        assert first["route_depth"] == 0
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        failover = _start(
            registry, ordinal=1, current_depth=0, failure=_THROTTLE, request_id="request-1"
        )
        assert failover["route_depth"] == 1
        assert ledger.started[0]["dispatch_reason"] is None
        assert ledger.started[1]["dispatch_reason"] == "throttle_failover_cold"
        assert ledger.started[1]["preferred_deployment_id"] == "deployment-a"
        assert registry.throttle_cache_counters() == (0, 1, 0, 0)

    def test_warm_cache_surfaces_the_throttle_instead_of_failing_over(self) -> None:
        """At or above the threshold the ladder ends and the caller gets the throttle.

        The organization's earlier settled traffic on the lead rung taught the
        worker a cached fraction of 0.9, so under a 0.5 threshold the throttle
        surfaces even though the pool's mode (availability) would fail over.
        """
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(registry, deployments, request_id="request-warm", throttle_cache_threshold=0.5)
        warm = _start(registry, ordinal=0, request_id="request-warm")
        _settle_with_usage(
            registry,
            attempt_id=str(warm["attempt_id"]),
            request_id="request-warm",
            cached_input_tokens=900,
            input_tokens=1_000,
        )
        assert registry.loads.cached_fraction(("deployment-a", "b" * 64), "organization-one") == (
            pytest.approx(0.9)
        )

        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_availability",
            throttle_cache_threshold=0.5,
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        surfaced = _start(
            registry, ordinal=1, current_depth=0, failure=_THROTTLE, request_id="request-1"
        )
        assert surfaced["exhausted"] is True
        exhaustion = surfaced["failure"]
        assert isinstance(exhaustion, dict)
        assert exhaustion["failure_class"] == "throttled"
        # No cold attempt was reserved; the request terminalized as throttled.
        assert [row["deployment_id"] for row in ledger.started] == ["deployment-a", "deployment-a"]
        assert ledger.finished_requests[-1].failure_class == GatewayFailureClass.THROTTLED
        assert registry.throttle_cache_counters() == (1, 0, 0, 0)

    def test_another_organizations_cache_never_counts(self) -> None:
        """The fraction is scoped to the requesting organization on the throttled rung."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        registry.loads.record_settle(
            ("deployment-a", "b" * 64),
            "organization-other",
            cached_tokens=1_000,
            input_tokens=1_000,
        )
        _admit(registry, deployments, request_id="request-1", throttle_cache_threshold=0.5)
        first = _start(registry, ordinal=0, request_id="request-1")
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        failover = _start(
            registry, ordinal=1, current_depth=0, failure=_THROTTLE, request_id="request-1"
        )
        assert failover["route_depth"] == 1
        assert ledger.started[1]["dispatch_reason"] == "throttle_failover_cold"

    def test_no_threshold_keeps_legacy_decisions_and_records_nothing(self) -> None:
        """Unauthored pools decide by mode as before: no disclosure, no counters."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        # Even a fully warm cache changes nothing without a threshold.
        registry.loads.record_settle(
            ("deployment-a", "b" * 64), "organization-one", cached_tokens=1_000, input_tokens=1_000
        )
        _admit(registry, deployments, request_id="request-1", failover_mode="maximize_availability")
        first = _start(registry, ordinal=0, request_id="request-1")
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        failover = _start(
            registry, ordinal=1, current_depth=0, failure=_THROTTLE, request_id="request-1"
        )
        assert failover["route_depth"] == 1
        assert all(row["dispatch_reason"] is None for row in ledger.started)

        _admit(registry, deployments, request_id="request-2", failover_mode="maximize_cache")
        first = _start(registry, ordinal=0, request_id="request-2")
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-2",
        )
        surfaced = _start(
            registry, ordinal=1, current_depth=0, failure=_THROTTLE, request_id="request-2"
        )
        assert surfaced["exhausted"] is True
        assert registry.throttle_cache_counters() == (0, 0, 0, 0)

    def test_cold_decision_that_exhausts_the_ladder_counts_no_failover(self) -> None:
        """A below-threshold throttle with nothing claimable ends as a plain exhausted throttle.

        The decision was to fail over, but no fallback attempt was reserved,
        so neither disposition is counted and no attempt discloses a cold
        failover: the metric reports only failovers that happened.
        """
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        entry = _admit(registry, deployments, request_id="request-1", throttle_cache_threshold=0.5)
        # The only fallback rung sits inside its own provider throttle window.
        registry.health.failed(
            deployment_health_key(entry.authorization, deployments[1]),
            GatewayFailure(
                failure_class=GatewayFailureClass.THROTTLED,
                safe_message="provider throttled the request",
                retry_after_seconds=30,
            ),
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        exhausted = _start(
            registry, ordinal=1, current_depth=0, failure=_THROTTLE, request_id="request-1"
        )
        assert exhausted["exhausted"] is True
        assert len(ledger.started) == 1
        assert registry.throttle_cache_counters() == (0, 0, 0, 0)


class TestThrottleRedial:
    """Post-backoff redials of a throttled rung and their disclosures."""

    def test_backoff_redials_reserve_the_same_rung_then_the_cold_advance_is_disclosed(
        self,
    ) -> None:
        """Each redial is its own attempt row on the warm rung; the spent budget fails over."""
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache",
            throttle_redial=GatewayThrottleRedialPolicy(
                max_attempts=2, base_delay_ms=100, max_delay_ms=2_000
            ),
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        assert first["route_depth"] == 0
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        # The data plane waited the backoff: two redials of the throttled rung,
        # each reserved through the rung's own throttle window.
        for ordinal in (1, 2):
            redial = _start(
                registry,
                ordinal=ordinal,
                current_depth=0,
                failure=_THROTTLE,
                request_id="request-1",
                throttle_backoff=True,
            )
            assert redial["route_depth"] == 0
            assert ledger.started[ordinal]["dispatch_reason"] == "throttle_backoff"
            assert ledger.started[ordinal]["preferred_deployment_id"] is None
            _settle(
                registry,
                attempt_id=str(redial["attempt_id"]),
                outcome="failed",
                finalize=False,
                failure=_THROTTLE,
                request_id="request-1",
            )
        # The budget is spent: the data plane no longer asks to wait, and
        # the throttle advances cold under maximize_cache instead of surfacing.
        cold = _start(
            registry, ordinal=3, current_depth=0, failure=_THROTTLE, request_id="request-1"
        )
        assert cold["route_depth"] == 1
        assert ledger.started[3]["dispatch_reason"] == "throttle_failover_cold"
        assert ledger.started[3]["preferred_deployment_id"] == "deployment-a"
        assert [row["attempt_ordinal"] for row in ledger.started] == [0, 1, 2, 3]
        assert registry.throttle_cache_counters() == (0, 1, 2, 0)

    @pytest.mark.parametrize("saturation", ["overflow", "refuse"])
    def test_backoff_redial_is_force_admitted_past_the_warm_rungs_own_rate_shed(
        self, saturation: Literal["overflow", "refuse"]
    ) -> None:
        """A paid-for redial stays on the throttled rung when its rate window would shed it.

        The warm rung authors ``requests_per_minute: 1`` per worker and this
        request's first attempt already spent that window before the provider
        throttled it. After the data plane waited the pool's backoff, the
        redial is admitted THERE anyway, disclosed ``throttle_backoff`` with no
        counterfactual (never ``rate_limit`` on the cold rung, never
        ``saturated_overflow``): the per-minute window is pacing the redial
        already paid on the provider's 429 clock. The shed is still counted,
        and the forced redial has its own worker counter. Once the redial
        budget is spent the next throttle advances cold as
        ``throttle_failover_cold`` exactly as before.
        """
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _rated_pair(requests_per_minute=1)
        deployments = tuple(
            deployment.model_copy(
                update={
                    "gateway": deployment.gateway.model_copy(
                        update={
                            "dispatch": GatewayRungDispatchPolicy(
                                concurrency_bound=10,
                                requests_per_minute=1,
                                saturation=saturation,
                            )
                        }
                    )
                }
            )
            for deployment in deployments
        )
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache",
            throttle_redial=GatewayThrottleRedialPolicy(
                max_attempts=1, base_delay_ms=100, max_delay_ms=2_000
            ),
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        assert first["route_depth"] == 0
        assert ledger.started[0]["dispatch_reason"] is None
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        redial = _start(
            registry,
            ordinal=1,
            current_depth=0,
            failure=_THROTTLE,
            request_id="request-1",
            throttle_backoff=True,
        )
        assert redial["route_depth"] == 0
        assert ledger.started[1]["deployment_id"] == "deployment-a"
        assert ledger.started[1]["dispatch_reason"] == "throttle_backoff"
        assert ledger.started[1]["preferred_deployment_id"] is None
        # The rate shed happened and is counted as one; the forced admission is
        # a backoff redial, not a saturated overflow.
        assert registry.rung_admission_counters() == (1, 0, 0)
        assert registry.rung_rate_counters() == (1, 0)
        assert registry.throttle_cache_counters() == (0, 0, 1, 1)
        _settle(
            registry,
            attempt_id=str(redial["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        cold = _start(
            registry, ordinal=2, current_depth=0, failure=_THROTTLE, request_id="request-1"
        )
        assert cold["route_depth"] == 1
        assert ledger.started[2]["deployment_id"] == "deployment-b"
        assert ledger.started[2]["dispatch_reason"] == "throttle_failover_cold"
        assert ledger.started[2]["preferred_deployment_id"] == "deployment-a"
        assert [row["attempt_ordinal"] for row in ledger.started] == [0, 1, 2]
        assert registry.throttle_cache_counters() == (0, 1, 1, 1)

    def test_a_non_redial_rate_shed_after_a_real_failure_still_spills_sideways(self) -> None:
        """Only the redialed rung is kept; a shed elsewhere on a failed ladder spills as today.

        Rung 1 authors ``requests_per_minute: 1`` and another request already
        holds its window. This request's throttle on rung 0 advances cold with a
        spent budget; the shed on rung 1 is not a redial of rung 1, so it spills
        on to rung 2. The disclosure names the first bypass of the walk, the
        cold advance past the throttled rung 0, and the shed is counted.
        """
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment(
                "deployment-b",
                connection_sha256="c" * 64,
                dispatch=GatewayRungDispatchPolicy(requests_per_minute=1),
            ),
            _deployment("deployment-c", connection_sha256="d" * 64),
        )
        _admit(registry, (deployments[1],), request_id="request-other")
        assert _start(registry, ordinal=0, request_id="request-other")["route_depth"] == 0
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache",
            throttle_redial=GatewayThrottleRedialPolicy(
                max_attempts=1, base_delay_ms=100, max_delay_ms=2_000
            ),
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        assert first["route_depth"] == 0
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        redial = _start(
            registry,
            ordinal=1,
            current_depth=0,
            failure=_THROTTLE,
            request_id="request-1",
            throttle_backoff=True,
        )
        assert redial["route_depth"] == 0
        _settle(
            registry,
            attempt_id=str(redial["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        spilled = _start(
            registry, ordinal=2, current_depth=0, failure=_THROTTLE, request_id="request-1"
        )
        assert spilled["route_depth"] == 2
        assert ledger.started[3]["deployment_id"] == "deployment-c"
        assert ledger.started[3]["dispatch_reason"] == "throttle_failover_cold"
        assert ledger.started[3]["preferred_deployment_id"] == "deployment-a"
        assert registry.rung_admission_counters() == (1, 0, 0)
        assert registry.rung_rate_counters() == (1, 0)
        assert registry.throttle_cache_counters() == (0, 1, 1, 0)

    def test_backoff_redial_shed_by_the_concurrency_bound_spills_sideways(self) -> None:
        """The hard per-worker bound stays hard for a redial; only the rate window is pacing.

        Rung 0 authors ``concurrency_bound: 1``. This request's first attempt
        held the slot until the provider throttled it; a request under another
        catalog revision (its own health view, the same physical rung) then
        took the slot. The post-backoff redial is shed ``queue_bound`` and
        spills sideways to rung 1 exactly like any other dispatch: the bound
        protects the provider connection and the other tenants on the rung,
        so no redial force-admits past it, and nothing is counted as a backoff
        redial. Only the redial's own accounting deltas are asserted: the
        slot-holder's admission on a single-rung ladder is not under test.
        """
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = _bounded_pair(1)
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache",
            throttle_redial=GatewayThrottleRedialPolicy(
                max_attempts=1, base_delay_ms=100, max_delay_ms=2_000
            ),
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        assert first["route_depth"] == 0
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        _admit(
            registry,
            (deployments[0],),
            request_id="request-other",
            organization_id="organization-two",
            catalog_sha256="f" * 64,
        )
        assert _start(registry, ordinal=0, request_id="request-other")["route_depth"] == 0
        sheds_before, overflows_before, _ = registry.rung_admission_counters()
        redial = _start(
            registry,
            ordinal=1,
            current_depth=0,
            failure=_THROTTLE,
            request_id="request-1",
            throttle_backoff=True,
        )
        assert redial["route_depth"] == 1
        assert ledger.started[2]["deployment_id"] == "deployment-b"
        assert ledger.started[2]["dispatch_reason"] == "queue_bound"
        assert ledger.started[2]["preferred_deployment_id"] == "deployment-a"
        sheds_after, overflows_after, _ = registry.rung_admission_counters()
        assert (sheds_after - sheds_before, overflows_after - overflows_before) == (1, 0)
        assert registry.throttle_cache_counters() == (0, 0, 0, 0)

    def test_budget_rejection_of_a_forced_redial_releases_the_forced_state(self) -> None:
        """A redial forced past rung 0's rate shed, then budget-rejected there, forces nothing else.

        Rung 0 and rung 1 both author ``requests_per_minute: 1``; another
        request already holds rung 1's window, and rung 0's hard deployment
        budget rejects the redial after the shed was force-admitted. The
        ladder advances to rung 1 with the forced state cleared, so rung 1's
        own rate shed spills the request on to rung 2 (two sheds, zero
        saturated overflows, zero backoff redials) instead of rung 1 being
        forced open and disclosed ``saturated_overflow``.
        """
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment(
                "deployment-a",
                connection_sha256="b" * 64,
                dispatch=GatewayRungDispatchPolicy(requests_per_minute=1),
            ),
            _deployment(
                "deployment-b",
                connection_sha256="c" * 64,
                dispatch=GatewayRungDispatchPolicy(requests_per_minute=1),
            ),
            _deployment("deployment-c", connection_sha256="d" * 64),
        )
        _admit(registry, (deployments[1],), request_id="request-other")
        assert _start(registry, ordinal=0, request_id="request-other")["route_depth"] == 0
        _admit(
            registry,
            deployments,
            request_id="request-1",
            failover_mode="maximize_cache",
            throttle_redial=GatewayThrottleRedialPolicy(
                max_attempts=1, base_delay_ms=100, max_delay_ms=2_000
            ),
        )
        first = _start(registry, ordinal=0, request_id="request-1")
        assert first["route_depth"] == 0
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_THROTTLE,
            request_id="request-1",
        )
        ledger.budget_rejections["deployment-a"] = BudgetScopeKind.DEPLOYMENT
        redial = _start(
            registry,
            ordinal=1,
            current_depth=0,
            failure=_THROTTLE,
            request_id="request-1",
            throttle_backoff=True,
        )
        assert redial["route_depth"] == 2
        assert ledger.started[2]["deployment_id"] == "deployment-c"
        assert ledger.started[2]["dispatch_reason"] != "saturated_overflow"
        assert ledger.started[2]["dispatch_reason"] != "throttle_backoff"
        assert registry.rung_admission_counters() == (2, 0, 0)
        assert registry.rung_rate_counters() == (2, 0)
        assert registry.throttle_cache_counters() == (0, 0, 0, 0)
        assert registry.loads.inflight(("deployment-b", "c" * 64)) == 1


class _LegacySignatureLedger(_RecordingLedger):
    """A hosted ledger whose settle predates the ``upstream_provider`` keyword."""

    def finish_attempt(  # ty: ignore[invalid-method-override] - the drift under test
        self,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
        retry_after_seconds: int | None = None,
        ratelimit_limit_requests: int | None = None,
        ratelimit_remaining_requests: int | None = None,
        ratelimit_limit_tokens: int | None = None,
        ratelimit_remaining_tokens: int | None = None,
    ) -> None:
        """Record the settle exactly as the previous engine handed it over."""
        del retry_after_seconds, ratelimit_limit_requests
        del ratelimit_remaining_requests, ratelimit_limit_tokens, ratelimit_remaining_tokens
        self.first_token_times.append(first_token_at)
        self.terminal_events.append(terminal_event)
        self.finished.append(
            {"attempt_id": attempt_id, "finalize": finalize_request, "failed": failure is not None}
        )


@pytest.mark.parametrize("writes", [None, 0, 25])
@pytest.mark.parametrize("legacy", [False, True])
def test_cache_write_usage_keeps_legacy_and_current_host_signatures(
    writes: int | None, legacy: bool
) -> None:
    """Cache writes ride typed usage, never a new keyword an older host must accept."""
    ledger = _LegacySignatureLedger() if legacy else _RecordingLedger()
    registry = NativeAttemptAccounting(cast("SyncWriteLedger", ledger))
    _admit(registry, _bounded_pair(1), request_id="cache-write-host")
    started = _start(registry, ordinal=0, request_id="cache-write-host")
    observed = datetime(2026, 9, 18, 1, 2, 3, tzinfo=UTC)
    registry.settle(
        json.dumps(
            {
                "request_id": "cache-write-host",
                "attempt_id": started["attempt_id"],
                "outcome": "completed",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 5,
                    "cached_input_tokens": 50,
                    "cache_creation_input_tokens": writes,
                },
                "first_token_at": observed.isoformat(),
                "upstream_provider": "Azure",
                "finalize": True,
                "opened": True,
            }
        )
    )
    (event,) = ledger.terminal_events
    assert event is not None and event.usage is not None
    assert event.usage.cache_creation_input_tokens == writes
    assert ledger.first_token_times == [observed]
    assert ledger.upstream_providers == ([] if legacy else ["Azure"])
    assert len(ledger.finished) == 1
    assert registry.entry("cache-write-host") is None


def _settle_naming_upstream(
    registry: NativeAttemptAccounting, *, attempt_id: str, request_id: str
) -> str:
    """One completed settle whose stream named the upstream that served it."""
    return registry.settle(
        json.dumps(
            {
                "request_id": request_id,
                "attempt_id": attempt_id,
                "outcome": "completed",
                "usage": None,
                "tool_names": [],
                "failure": None,
                "finalize": True,
                "opened": True,
                "upstream_provider": "Azure",
            }
        )
    )


def test_settle_hands_the_upstream_provider_only_to_a_ledger_that_accepts_it() -> None:
    """The hosted-ledger seam: a pre-keyword ledger settles cleanly; a current one gets the value.

    The engine repins independently of the host's ledger, so the new settle
    keyword must never TypeError a host that has not learned it (the 2026-08-30
    hosted-ledger incident class); the signature is probed once at construction.
    """
    legacy = _LegacySignatureLedger()
    # The older host shape is exactly the drift under test, so the protocol
    # mismatch is asserted away at this one seam.
    registry = NativeAttemptAccounting(cast("SyncWriteLedger", legacy))
    deployments = _bounded_pair(1)
    _admit(registry, deployments, request_id="request-1")
    started = _start(registry, ordinal=0, request_id="request-1")
    _settle_naming_upstream(registry, attempt_id=str(started["attempt_id"]), request_id="request-1")
    assert legacy.finished[-1]["finalize"] is True
    assert legacy.upstream_providers == []

    current = _RecordingLedger()
    registry = NativeAttemptAccounting(current)
    _admit(registry, deployments, request_id="request-2")
    started = _start(registry, ordinal=0, request_id="request-2")
    _settle_naming_upstream(registry, attempt_id=str(started["attempt_id"]), request_id="request-2")
    assert current.upstream_providers == ["Azure"]


def _settle_billing_searches(
    registry: NativeAttemptAccounting,
    *,
    attempt_id: str,
    request_id: str,
    web_search_requests: int | None,
    tool_search_requests: int | None = None,
) -> str:
    """One completed, token-bearing settle that bills the gateway's own search meters.

    ``None`` omits the key exactly as an engine predating the field does.
    """
    payload: JsonObject = {
        "request_id": request_id,
        "attempt_id": attempt_id,
        "outcome": "completed",
        "usage": {"input_tokens": 12, "output_tokens": 4},
        "tool_names": [],
        "failure": None,
        "finalize": True,
        "opened": True,
    }
    if web_search_requests is not None:
        payload["web_search_requests"] = web_search_requests
    if tool_search_requests is not None:
        payload["tool_search_requests"] = tool_search_requests
    return registry.settle(json.dumps(payload))


def test_settle_hands_web_search_requests_only_to_a_ledger_that_accepts_it() -> None:
    """The hosted-ledger seam for the search meter mirrors ``upstream_provider``.

    A ledger predating the keyword settles cleanly with it withheld; a current
    ledger receives the settled count; zero or an absent count is withheld from
    every ledger so an attempt that never searched settles as before the field.
    """
    legacy = _LegacySignatureLedger()
    registry = NativeAttemptAccounting(cast("SyncWriteLedger", legacy))
    deployments = _bounded_pair(1)
    _admit(registry, deployments, request_id="request-1")
    started = _start(registry, ordinal=0, request_id="request-1")
    _settle_billing_searches(
        registry,
        attempt_id=str(started["attempt_id"]),
        request_id="request-1",
        web_search_requests=3,
    )
    assert legacy.finished[-1]["finalize"] is True
    assert legacy.web_search_requests == []

    current = _RecordingLedger()
    registry = NativeAttemptAccounting(current)
    for ordinal, (request_id, count) in enumerate(
        (("request-2", 3), ("request-3", 0), ("request-4", None))
    ):
        _admit(registry, deployments, request_id=request_id)
        started = _start(registry, ordinal=0, request_id=request_id)
        _settle_billing_searches(
            registry,
            attempt_id=str(started["attempt_id"]),
            request_id=request_id,
            web_search_requests=count,
        )
        assert len(current.finished) == ordinal + 1
    terminal = current.terminal_events[0]
    assert terminal is not None and terminal.usage is not None
    assert terminal.usage.web_search_requests == 3
    assert current.web_search_requests == [3, None, None]


def test_swept_retained_settlement_still_bills_its_web_searches() -> None:
    """A settlement the sweep recovers hands the ledger the same search count as a direct one."""
    registry, ledger, entry = _registry()
    started = _start(registry, ordinal=0)
    ledger.fail_finishes = 1
    with pytest.raises(NativeBridgeError):
        _settle_billing_searches(
            registry,
            attempt_id=str(started["attempt_id"]),
            request_id="request-one",
            web_search_requests=2,
        )
    assert entry.pending_settlement is not None
    registry.sweep_expired()
    assert entry.pending_settlement is None
    assert len(ledger.finished) == 1
    assert ledger.web_search_requests == [2, 2]


class _PreToolSearchLedger(_RecordingLedger):
    """A hosted ledger that learned the web-search meter but not ``tool_search_requests``."""

    def finish_attempt(  # ty: ignore[invalid-method-override] - the drift under test
        self,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        first_token_at: datetime | None = None,
        retry_after_seconds: int | None = None,
        ratelimit_limit_requests: int | None = None,
        ratelimit_remaining_requests: int | None = None,
        ratelimit_limit_tokens: int | None = None,
        ratelimit_remaining_tokens: int | None = None,
        upstream_provider: str | None = None,
        web_search_requests: int | None = None,
    ) -> None:
        """Record the settle exactly as the previous engine handed it over."""
        del first_token_at, retry_after_seconds, ratelimit_limit_requests
        del ratelimit_remaining_requests, ratelimit_limit_tokens, ratelimit_remaining_tokens
        self.upstream_providers.append(upstream_provider)
        self.web_search_requests.append(web_search_requests)
        self.terminal_events.append(terminal_event)
        self.finished.append(
            {"attempt_id": attempt_id, "finalize": finalize_request, "failed": failure is not None}
        )


def test_settle_hands_tool_search_requests_only_to_a_ledger_that_accepts_it() -> None:
    """The hosted-ledger seam for the tool-search meter mirrors ``web_search_requests``.

    A ledger that learned the web-search meter but predates the tool-search
    keyword settles cleanly with it withheld (and still receives the web-search
    count); a current ledger receives the settled count; zero or an absent
    count is withheld from every ledger.
    """
    legacy = _PreToolSearchLedger()
    registry = NativeAttemptAccounting(cast("SyncWriteLedger", legacy))
    deployments = _bounded_pair(1)
    _admit(registry, deployments, request_id="request-1")
    started = _start(registry, ordinal=0, request_id="request-1")
    _settle_billing_searches(
        registry,
        attempt_id=str(started["attempt_id"]),
        request_id="request-1",
        web_search_requests=1,
        tool_search_requests=3,
    )
    assert legacy.finished[-1]["finalize"] is True
    assert legacy.web_search_requests == [1]
    assert legacy.tool_search_requests == []

    current = _RecordingLedger()
    registry = NativeAttemptAccounting(current)
    for ordinal, (request_id, count) in enumerate(
        (("request-2", 3), ("request-3", 0), ("request-4", None))
    ):
        _admit(registry, deployments, request_id=request_id)
        started = _start(registry, ordinal=0, request_id=request_id)
        _settle_billing_searches(
            registry,
            attempt_id=str(started["attempt_id"]),
            request_id=request_id,
            web_search_requests=None,
            tool_search_requests=count,
        )
        assert len(current.finished) == ordinal + 1
    terminal = current.terminal_events[0]
    assert terminal is not None and terminal.usage is not None
    assert terminal.usage.tool_search_requests == 3
    assert terminal.usage.web_search_requests == 0
    assert current.tool_search_requests == [3, None, None]
    assert current.web_search_requests == [None, None, None]


def test_swept_retained_settlement_still_bills_its_tool_searches() -> None:
    """A settlement the sweep recovers hands the ledger the same tool-search count as a direct."""
    registry, ledger, entry = _registry()
    started = _start(registry, ordinal=0)
    ledger.fail_finishes = 1
    with pytest.raises(NativeBridgeError):
        _settle_billing_searches(
            registry,
            attempt_id=str(started["attempt_id"]),
            request_id="request-one",
            web_search_requests=1,
            tool_search_requests=2,
        )
    assert entry.pending_settlement is not None
    registry.sweep_expired()
    assert entry.pending_settlement is None
    assert len(ledger.finished) == 1
    assert ledger.web_search_requests == [1, 1]
    assert ledger.tool_search_requests == [2, 2]


class TestToolSearchRound:
    """A gateway tool-search round re-dials the serving rung as a fresh attempt."""

    def test_round_reserves_the_same_rung_with_its_own_dispatch_reason(self) -> None:
        ledger = _RecordingLedger()
        registry = NativeAttemptAccounting(ledger)
        deployments = (
            _deployment("deployment-a", connection_sha256="b" * 64),
            _deployment("deployment-b", connection_sha256="c" * 64),
        )
        _admit(registry, deployments, request_id="request-1")
        first = _start(registry, ordinal=0, request_id="request-1")
        assert first["route_depth"] == 0
        _settle(
            registry,
            attempt_id=str(first["attempt_id"]),
            outcome="completed",
            finalize=False,
            request_id="request-1",
        )
        again = _start(
            registry, ordinal=1, current_depth=0, request_id="request-1", tool_search_round=True
        )
        assert again["route_depth"] == 0
        assert ledger.started[1]["dispatch_reason"] == "tool_search_round"
        assert ledger.started[1]["route_depth"] == 0
        # Not a throttle redial: the throttle budget is untouched.
        assert registry.throttle_cache_counters() == (0, 0, 0, 0)


@pytest.mark.parametrize("failed_cleanup", [False, True])
def test_abandon_during_committed_reservation_retains_and_closes_late_attempt(
    monkeypatch: pytest.MonkeyPatch,
    failed_cleanup: bool,
) -> None:
    """A cancelled callback cannot orphan a reservation committed before its result returns."""
    monkeypatch.setattr(NativeAttemptAccounting, "_sweep_loop", lambda self: None)
    ledger = _RecordingLedger()
    accounting = NativeAttemptAccounting(ledger)
    entry = _admit(
        accounting,
        (_deployment("lead", connection_sha256="b" * 64),),
        request_id="late-reservation",
        failover_mode="maximize_availability",
    )
    original = ledger.start_attempt
    committed, release = threading.Event(), threading.Event()
    results: list[JsonObject] = []
    errors: list[BaseException] = []

    def start() -> None:
        """Enter actual accounting and wait after its durable reservation commits."""
        try:
            results.append(_start(accounting, ordinal=0, request_id="late-reservation"))
        except BaseException as error:  # noqa: BLE001 - propagate the worker's failure.
            errors.append(error)

    def pause_after_commit(
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_nano_usd: int | None = None,
        reserved_input_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        route_reason: str | None = None,
        fallback_reason: str | None = None,
        dispatch_reason: str | None = None,
        preferred_deployment: ExactModelDeployment | None = None,
    ) -> str:
        """Delay the existing typed ledger method without changing reservation semantics."""
        result = original(
            snapshot=snapshot,
            deployment=deployment,
            attempt_ordinal=attempt_ordinal,
            route_depth=route_depth,
            maximum_cost_nano_usd=maximum_cost_nano_usd,
            reserved_input_tokens=reserved_input_tokens,
            reserved_output_tokens=reserved_output_tokens,
            route_reason=route_reason,
            fallback_reason=fallback_reason,
            dispatch_reason=dispatch_reason,
            preferred_deployment=preferred_deployment,
        )
        committed.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(ledger, "start_attempt", pause_after_commit)
    worker = threading.Thread(target=start)
    worker.start()
    try:
        assert committed.wait(5)
        accounting.abandon(json.dumps({"request_id": "late-reservation"}))
        assert accounting.entry("late-reservation") is entry
        assert not ledger.finished_requests and not ledger.finished
        if failed_cleanup:
            ledger.fail_finishes = 1
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    if failed_cleanup:
        assert len(errors) == 1 and isinstance(errors[0], NativeBridgeError)
        assert not results and accounting.entry("late-reservation") is entry
        assert entry.active_attempt_id is not None and not ledger.finished
        accounting.sweep_expired()
    else:
        assert not errors and results and results[0].get("exhausted") is True
    assert len(ledger.started) == len(ledger.finished) == 1
    assert ledger.finished[0]["failure_class"] == "cancelled"
    assert ledger.terminal_events[-1] is not None
    assert ledger.terminal_events[-1].usage is None
    assert accounting.entry("late-reservation") is None
    accounting.sweep_expired()
    assert len(ledger.finished) == 1


@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("failure_class", ["provider_authentication", "provider_quota"])
def test_customer_account_failure_replaces_transport_recovery_only(
    monkeypatch: pytest.MonkeyPatch,
    swept: bool,
    failure_class: str,
) -> None:
    """Ledger and house health stay caller-owned while recovery remembers failed credentials."""
    monkeypatch.setattr(NativeAttemptAccounting, "_sweep_loop", lambda self: None)
    ledger, host, clock = _RecordingLedger(), RecoveryHostFake(), Clock()
    accounting = NativeAttemptAccounting(ledger, recovery_host=host)
    accounting.recovery = SessionRecoveryRegistry(clock=clock)
    entry = recovery_entry()
    usage = GatewayUsage(input_tokens=100, output_tokens=5, cached_input_tokens=80)
    record_session_outcome(accounting.recovery, host, entry, "attempt", usage, None)
    record_session_outcome(
        accounting.recovery,
        host,
        entry,
        "departure",
        None,
        GatewayFailure(failure_class=GatewayFailureClass.TRANSPORT, safe_message="transport"),
    )
    record_session_outcome(accounting.recovery, host, entry, "fallback", usage, None)
    entry.attempt_depths["account-failure"] = 0
    entry.active_attempt_id = "account-failure"
    accounting.register(entry)
    payload = json.dumps(
        {
            "request_id": entry.authorization.request_id,
            "attempt_id": "account-failure",
            "outcome": "failed",
            "failure": {
                "failure_class": failure_class,
                "safe_message": "account failed",
                "customer_owned": True,
            },
        }
    )
    if swept:
        ledger.fail_finishes = 1
        with pytest.raises(NativeBridgeError):
            accounting.settle(payload)
        clock.now += 1
        accounting.sweep_expired()
    else:
        accounting.settle(payload)
    assert ledger.finished[-1]["failure_class"] == "invalid_request"
    assert not accounting.health.suppressed(
        deployment_health_key(entry.authorization, entry.route.deployment)
    )
    clock.now += 10
    key = session_cache_key(entry)
    assert key is not None
    candidates = tuple(
        (d.deployment_id, entry.recovery_bindings[d.deployment_id].scope)
        for d in entry.route.deployments
    )
    scope = candidates[0][1]
    snapshot = RecoverySnapshot(
        loaded_at=clock.now,
        observations=(
            RecoveryObservation(
                scope=scope.operational(), cause="transport", observed_at=clock.now, healthy=True
            ),
        ),
        leases=(
            RecoveryLease(
                lease_id="after-account", scope=scope.operational(), expires_at=clock.now + 20
            ),
        ),
    )
    decision = accounting.recovery.choose(key, candidates, eligible=eligible, snapshot=snapshot)
    assert decision.deployment_id == entry.route.deployments[1].deployment_id
    assert decision.reason == "retained_warm_fallback"


def _retained_settlement(
    monkeypatch: pytest.MonkeyPatch, *, failing: bool = False
) -> tuple[NativeAttemptAccounting, _RecordingLedger, Clock, InflightRequest, JsonObject]:
    """Retain a real accounting write at its original arrival time for deterministic retry."""
    monkeypatch.setattr(NativeAttemptAccounting, "_sweep_loop", lambda self: None)
    ledger, host, clock = _RecordingLedger(), RecoveryHostFake(), Clock(now=990)
    accounting = NativeAttemptAccounting(ledger, recovery_host=host)
    accounting._sweeper.join(timeout=5)
    accounting.recovery = SessionRecoveryRegistry(clock=clock)
    entry = recovery_entry()
    usage = GatewayUsage(input_tokens=100, output_tokens=5, cached_input_tokens=50)
    if failing:
        for attempt_id in ("attempt", "fallback"):
            record_session_outcome(accounting.recovery, host, entry, attempt_id, usage, None)
    attempt_id = "departure" if failing else "attempt"
    entry.active_attempt_id = attempt_id
    accounting.register(entry)
    payload: JsonObject = {
        "request_id": entry.authorization.request_id,
        "attempt_id": attempt_id,
        "outcome": "failed" if failing else "completed",
        "finalize": True,
    }
    if failing:
        payload["failure"] = {"failure_class": "transport", "safe_message": "synthetic failure"}
    else:
        payload["usage"] = usage.model_dump(mode="json")
    ledger.fail_finishes = 1
    clock.now = 1000
    with pytest.raises(NativeBridgeError):
        accounting.settle(json.dumps(payload))
    assert entry.pending_settlement == payload
    return accounting, ledger, clock, entry, payload


@pytest.mark.parametrize("failing", [False, True])
def test_concurrent_settle_and_sweep_observe_recovery_once(
    monkeypatch: pytest.MonkeyPatch, failing: bool
) -> None:
    """Serialized ledger retries cannot renew cache residency or duplicate a recovery failure."""
    accounting, ledger, clock, entry, payload = _retained_settlement(monkeypatch, failing=failing)
    original_key = native_recovery.session_cache_key
    paused, release, sweep_finished_write = threading.Event(), threading.Event(), threading.Event()
    finish = accounting._finish_attempt
    ledger_lock = threading.Lock()
    successful_writes: set[str] = set()
    calls: list[str] = []

    def observed_finish(
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
        **metadata: object,
    ) -> None:
        """Serialize idempotent ledger delivery and signal after the swept write returns."""
        with ledger_lock:
            calls.append(threading.current_thread().name)
            if attempt_id not in successful_writes:
                finish(
                    attempt_id=attempt_id,
                    terminal_event=terminal_event,
                    failure=failure,
                    finalize_request=finalize_request,
                    **metadata,
                )
                successful_writes.add(attempt_id)
        if threading.current_thread().name == "swept-retry":
            sweep_finished_write.set()

    monkeypatch.setattr(accounting, "_finish_attempt", observed_finish)
    derived: list[str] = []
    errors: list[BaseException] = []

    def gated_key(current: InflightRequest) -> SessionCacheKey | None:
        """Pause one observer without blocking a ledger write on another thread."""
        derived.append(threading.current_thread().name)
        if threading.current_thread().name == "direct-retry":
            paused.set()
            assert release.wait(5)
        return original_key(current)

    def retry() -> None:
        """Retry the original terminal payload through the direct accounting path."""
        try:
            accounting.settle(json.dumps(payload))
        except BaseException as error:  # noqa: BLE001 - propagate thread failures to the test.
            errors.append(error)

    def sweep() -> None:
        """Race the retained terminal write through the actual sweep path."""
        try:
            accounting.sweep_expired()
        except BaseException as error:  # noqa: BLE001 - propagate thread failures to the test.
            errors.append(error)

    monkeypatch.setattr(native_recovery, "session_cache_key", gated_key)
    direct = threading.Thread(target=retry, name="direct-retry")
    swept = threading.Thread(target=sweep, name="swept-retry")
    direct.start()
    try:
        assert paused.wait(5)
        swept.start()
        assert sweep_finished_write.wait(5)
        clock.now = 1010
    finally:
        release.set()
        direct.join(5)
        if swept.ident is not None:
            swept.join(5)
    assert not direct.is_alive() and not swept.is_alive() and not errors
    assert len(derived) == 1
    assert accounting.entry(entry.authorization.request_id) is None
    assert calls == ["direct-retry", "swept-retry"]
    assert successful_writes == {str(payload["attempt_id"])}
    assert len(ledger.finished) == 1
    assert len(ledger.terminal_events) == 2  # Initial failed delivery and one durable terminal.
    key = original_key(entry)
    assert key is not None
    history = accounting.recovery._sessions[key]
    primary = entry.route.deployment.deployment_id
    if failing:
        departure = history.departures[primary]
        assert (departure.failed_at, departure.retry_at) == (1000, 1005)
    else:
        assert history.evidence[primary].expires_at == history.retained_until == 1100


def test_delayed_first_sweep_does_not_renew_expired_cache_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write received at 1000 cannot create fresh TTL100 warmth when retried at 1150."""
    accounting, ledger, clock, entry, _payload = _retained_settlement(monkeypatch)
    clock.now = 1150
    accounting.sweep_expired()
    key = session_cache_key(entry)
    assert key is not None
    deployment = entry.route.deployment
    scope = entry.recovery_bindings[deployment.deployment_id].scope
    decision = accounting.recovery.choose(
        key, ((deployment.deployment_id, scope),), eligible=eligible, snapshot=None
    )
    assert len(ledger.finished) == 1
    assert decision.deployment_id is None and decision.warm_remaining_seconds == 0

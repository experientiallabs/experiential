"""Observation uses the shared engine without authority over native delivery or billing."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import httpx
import pytest
from websockets.sync.client import connect

from exp.common.models import BillingSource, GatewayTokenPrices
from exp.runtime.gateway.contracts import GatewayFailureClass, GatewayRequest
from exp.runtime.gateway.guardrails.bounded import BoundedInspect
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierCoverageError,
    ClassifierUncertainError,
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailOutcome,
    GuardrailPolicy,
    GuardrailRejected,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.native import require_unguarded_surface
from exp.runtime.gateway.guardrails.native_test import _authorization
from exp.runtime.gateway.guardrails.session import GuardrailSession
from exp.runtime.gateway.guardrails.session_test import _request
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.subjects_test import _changed_subject
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.tests.guardrail_policy_integrity_test import _body, _provider
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.native_tool_search_test import _TOOLS_CHAT, _configure, _Provider
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames


class _Classifier(ScriptedClassifier):
    """Barrier-controlled input classifier exposing every bounded observation outcome."""

    def __init__(self, outcome: GuardrailOutcome) -> None:
        """Create independent synchronization and outcome state."""
        super().__init__()
        self.outcome = outcome
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.subjects: list[GatewayRequest] = []

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Wait off the request path, then return or raise one sanitized outcome."""
        del check
        self.input_calls += 1
        self.subjects.append(request)
        self.started.set()
        try:
            while not self.release.is_set():
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.outcome is GuardrailOutcome.UNSUPPORTED:
            raise ClassifierCoverageError
        if self.outcome is GuardrailOutcome.UNCERTAIN:
            raise ClassifierUncertainError
        if self.outcome is GuardrailOutcome.UNAVAILABLE:
            raise RuntimeError("private-detector-diagnostic")
        return ClassifierVerdict(flagged=self.outcome is GuardrailOutcome.FLAGGED)


def _policy(*, timeout_ms: int = 1000, max_request_bytes: int = 1_048_576) -> GuardrailPolicy:
    """Keep the blocking policy intact while choosing operator-owned observation."""
    return GuardrailPolicy(
        policy_id="platform-observer",
        protected=True,
        mode="observe",
        max_request_bytes=max_request_bytes,
        checks=(
            GuardrailCheck(
                check_id="input",
                adapter_id="detector",
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.BLOCK,
                timeout_ms=timeout_ms,
            ),
        ),
    )


class _Engine(GuardrailEngine):
    """Capture aggregate outcomes separately from actions affecting the request."""

    def __init__(
        self,
        classifier: _Classifier,
        policy: GuardrailPolicy | None = None,
        *,
        max_observations: int = 8,
        max_observation_bytes: int = 8 * 1024 * 1024,
        enforcing: GuardrailPolicy | None = None,
        inspects: BoundedInspect | None = None,
        max_observation_records: int = 128,
    ) -> None:
        """Build the actual shared executor and bounded observation owner."""
        self.observed: list[tuple[str, str | None, GuardrailOutcome]] = []
        self.enforced: list[GuardrailAction] = []
        self.recorded = threading.Event()
        policies = (_policy() if policy is None else policy,)
        if enforcing is not None:
            policies += (enforcing,)
        super().__init__(
            store=MappingGuardrailStore(policies),
            client=DirectClassifierClient(
                ClassifierRegistry(
                    {
                        "detector": classifier,
                        "blocker": ScriptedClassifier(
                            input_verdict=ClassifierVerdict(flagged=True)
                        ),
                    }
                )
            ),
            monotonic=time.monotonic,
            inspects=inspects,
            max_observation_records=max_observation_records,
            max_observations=max_observations,
            max_observation_bytes=max_observation_bytes,
        )

    def _record(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck | None,
        action: GuardrailAction,
        latency_seconds: float,
    ) -> None:
        """Capture actual customer-facing actions without changing the existing recorder seam."""
        del policy, check, latency_seconds
        self.enforced.append(action)

    def _record_observation(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck | None,
        outcome: GuardrailOutcome,
        latency_seconds: float,
    ) -> None:
        """Capture only bounded identities and outcomes, never subject content."""
        del latency_seconds
        assert policy.organization_id is not None and policy.identity_id is not None
        self.observed.append((policy.policy_id, None if check is None else check.check_id, outcome))
        self.recorded.set()


def _session(engine: GuardrailEngine, *, deadline: float | None = None) -> GuardrailSession:
    """Freeze authenticated policies and the original absolute request deadline."""
    session = engine.open_request(
        _authorization(),
        deadline_monotonic=time.monotonic() + 5 if deadline is None else deadline,
    )
    assert session is not None
    return session


@pytest.mark.parametrize(
    "outcome",
    [
        GuardrailOutcome.ALLOW,
        GuardrailOutcome.FLAGGED,
        GuardrailOutcome.UNSUPPORTED,
        GuardrailOutcome.UNCERTAIN,
        GuardrailOutcome.UNAVAILABLE,
    ],
)
def test_observation_never_controls_approval_or_settlement(outcome: GuardrailOutcome) -> None:
    """Even a protected block policy only records its observed decision."""
    classifier = _Classifier(outcome)
    engine = _Engine(classifier)
    session = _session(engine)
    request = _request()
    try:
        assert session.inspect_input(request) == request
        session.prepare_dispatch((request, request.model_copy()), overlap=False)
        assert classifier.started.wait(2)
        assert not session.input_pending
        assert session.input_decision() == {"action": "allow"}
        assert session.settlement_failure() is None
        assert session.output_policies == ()
        assert session.inspect_output(GuardrailCompletion(text="provider")) == GuardrailCompletion(
            text="provider"
        )
        session.cancel()
        assert session.settlement_failure() is None
        assert not classifier.cancelled.is_set()
        classifier.release.set()
        assert engine.recorded.wait(2)
        assert engine.observed == [("platform-observer", "input", outcome)]
        assert engine.enforced == []
        assert classifier.subjects == [request]
    finally:
        classifier.release.set()
        engine.close(timeout_seconds=2)


def test_observer_limits_deadlines_and_shutdown_do_not_admit_unbounded_work() -> None:
    """Skipped jobs and timeouts remain visible without rejection or retained serving state."""
    classifier = _Classifier(GuardrailOutcome.FLAGGED)
    engine = _Engine(classifier, _policy(timeout_ms=30), max_observations=1)
    session = _session(engine)
    try:
        session.inspect_input(_request("first"))
        assert classifier.started.wait(2)
        session.inspect_input(_request("second"))
        assert engine.recorded.wait(2)
        assert ("platform-observer", None, GuardrailOutcome.SKIPPED) in engine.observed
        assert classifier.cancelled.wait(2)
        engine.close(timeout_seconds=2)
        assert ("platform-observer", "input", GuardrailOutcome.TIMEOUT) in engine.observed
        assert classifier.input_calls == 1
        assert session.settlement_failure() is None
        session.inspect_input(_request("after-close"))
        assert engine.observation_recording_dropped == 1
    finally:
        classifier.release.set()
        engine.close(timeout_seconds=0)


@pytest.mark.parametrize("bound", ["bytes", "coverage", "deadline"])
def test_unsupported_or_unadmitted_subjects_are_recorded_without_inspection(bound: str) -> None:
    """A skipped observation cannot be counted as allow or block."""
    classifier = _Classifier(GuardrailOutcome.FLAGGED)
    engine = _Engine(
        classifier,
        _policy(max_request_bytes=1 if bound == "coverage" else 1_048_576),
        max_observation_bytes=1 if bound == "bytes" else 8 * 1024 * 1024,
    )
    session = _session(engine, deadline=time.monotonic() - 1 if bound == "deadline" else None)
    try:
        assert session.inspect_input(_request()) == _request()
        assert session.settlement_failure() is None
        assert classifier.input_calls == 0
        expected = GuardrailOutcome.UNSUPPORTED if bound == "coverage" else GuardrailOutcome.SKIPPED
        engine.close(timeout_seconds=2)
        assert engine.observed == [("platform-observer", None, expected)]
    finally:
        engine.close(timeout_seconds=0)


def test_observers_do_not_disable_an_existing_enforcing_policy() -> None:
    """A mixed session still blocks on its ordinary identity checks."""
    classifier = _Classifier(GuardrailOutcome.ALLOW)
    identity = GuardrailPolicy(
        policy_id="identity-enforcer",
        organization_id="organization-one",
        identity_id="identity-one",
        checks=(_policy().checks[0].model_copy(update={"adapter_id": "blocker"}),),
    )
    engine = _Engine(classifier, enforcing=identity)
    try:
        with pytest.raises(GuardrailRejected) as raised:
            _session(engine).inspect_input(_request())
        assert raised.value.failure.failure_class is GatewayFailureClass.GUARDRAIL
        assert engine.enforced == [GuardrailAction.BLOCK]
    finally:
        classifier.release.set()
        engine.close(timeout_seconds=2)


def test_observer_capacity_cannot_starve_an_unrelated_enforcing_check() -> None:
    """An occupied observer worker leaves even a single-slot enforcer available."""
    classifier = _Classifier(GuardrailOutcome.ALLOW)
    engine = _Engine(classifier, inspects=BoundedInspect(max_inflight=1))
    enforcer = _policy().model_copy(
        update={
            "mode": "enforce",
            "checks": (
                _policy().checks[0].model_copy(update={"adapter_id": "blocker", "timeout_ms": 100}),
            ),
        }
    )
    try:
        _session(engine).inspect_input(_request("observed"))
        assert classifier.started.wait(2)
        with pytest.raises(GuardrailRejected) as raised:
            asyncio.run(
                engine.enforce_input(
                    policy=enforcer,
                    request=_request("enforced"),
                    deadline_monotonic=time.monotonic() + 1,
                )
            )
        assert raised.value.failure.failure_class is GatewayFailureClass.GUARDRAIL
        assert engine.enforced == [GuardrailAction.BLOCK]
        assert not classifier.release.is_set()
    finally:
        classifier.release.set()
        engine.close(timeout_seconds=2)


def test_observer_quarantine_cannot_quarantine_the_same_enforcing_adapter() -> None:
    """A detector's stalled observation cannot deny a later healthy enforced subject."""

    class BlockingObservation(_Classifier):
        """Block one synthetic observed subject while promptly deciding another."""

        async def inspect_input(
            self, *, request: GatewayRequest, check: GuardrailCheck
        ) -> ClassifierVerdict:
            """Run both modes through the same adapter while only one subject stalls."""
            del check
            self.input_calls += 1
            if request.messages[0].content == "observed":
                self.started.set()
                assert self.release.wait(5)
            return ClassifierVerdict(flagged=False)

    classifier = BlockingObservation(GuardrailOutcome.ALLOW)
    engine = _Engine(classifier, _policy(timeout_ms=30), inspects=BoundedInspect(max_inflight=1))
    try:
        _session(engine).inspect_input(_request("observed"))
        assert classifier.started.wait(2)
        assert engine.recorded.wait(2)
        assert engine.observed[-1][2] is GuardrailOutcome.TIMEOUT
        assert engine._observation_inspects.quarantined_adapter_ids() == {"detector"}
        enforcer = _policy(timeout_ms=100).model_copy(update={"mode": "enforce"})
        request = _request("enforced")
        assert (
            asyncio.run(
                engine.enforce_input(
                    policy=enforcer, request=request, deadline_monotonic=time.monotonic() + 1
                )
            )
            is request
        )
        assert classifier.input_calls == 2
        assert engine.enforced == [GuardrailAction.ALLOW]
    finally:
        classifier.release.set()
        engine.close(timeout_seconds=2)


@pytest.mark.parametrize(
    "field",
    [
        "context_management",
        "provider_output_config",
        "reasoning",
        "input_examples",
        "provider_anthropic_blocks",
        "provider_text_blocks",
    ],
)
def test_observation_deduplication_preserves_changed_private_context(field: str) -> None:
    """Context recovery and provider-private fields cannot inherit an earlier observation."""
    classifier = _Classifier(GuardrailOutcome.ALLOW)
    classifier.release.set()
    engine = _Engine(classifier)
    before, after = _changed_subject(field)
    try:
        session = _session(engine)
        session.inspect_input(before)
        session.inspect_input(before.model_copy(deep=True))
        session.inspect_input(after)
        engine.close(timeout_seconds=2)
        assert len(classifier.subjects) == 2
        assert before in classifier.subjects and after in classifier.subjects
    finally:
        engine.close(timeout_seconds=2)


def test_observation_budget_counts_private_context_before_dispatch() -> None:
    """Private payload bytes exhaust the same owner budget as visible prompt bytes."""
    classifier = _Classifier(GuardrailOutcome.ALLOW)
    engine = _Engine(classifier, max_observation_bytes=10_000)
    before, _ = _changed_subject("context_management")
    request = before.model_copy(update={"context_management": {"private": "x" * 10_000}})
    try:
        assert _session(engine).inspect_input(request) is request
        assert classifier.input_calls == 0
        engine.close(timeout_seconds=2)
        assert engine.observed == [("platform-observer", None, GuardrailOutcome.SKIPPED)]
    finally:
        engine.close(timeout_seconds=2)


def test_pending_observation_revisits_do_not_report_skipped_coverage() -> None:
    """The normal admission/dispatch tuple reuses one pending subject at full capacity."""
    classifier = _Classifier(GuardrailOutcome.ALLOW)
    engine = _Engine(classifier, max_observations=1)
    request = _request("original")
    session = _session(engine)
    try:
        session.inspect_input(request)
        assert classifier.started.wait(2)
        session.prepare_dispatch(
            (request, request.model_copy(deep=True), request, request.model_copy()),
            overlap=False,
        )
        classifier.release.set()
        engine.close(timeout_seconds=2)
        assert classifier.input_calls == 1
        assert engine.observed == [("platform-observer", "input", GuardrailOutcome.ALLOW)]
    finally:
        classifier.release.set()
        engine.close(timeout_seconds=2)


@pytest.mark.parametrize("bound", ["policy", "owner"])
def test_oversized_subject_closes_session_observation_with_one_coverage_outcome(bound: str) -> None:
    """A rejected large subject records incomplete request coverage once without retaining text."""
    classifier = _Classifier(GuardrailOutcome.ALLOW)
    policy = _policy().model_copy(
        update={"max_request_bytes": 8192 if bound == "policy" else 1_000_000}
    )
    engine = _Engine(classifier, policy, max_observation_bytes=8192)
    request = _request("x" * 100_000)
    session = _session(engine)
    try:
        session.inspect_input(request)
        session.prepare_dispatch(
            (request, request.model_copy(deep=True), request, _request("later recovered text")),
            overlap=False,
        )
        engine.close(timeout_seconds=2)
        outcome = GuardrailOutcome.UNSUPPORTED if bound == "policy" else GuardrailOutcome.SKIPPED
        assert classifier.input_calls == 0
        assert engine.observed == [("platform-observer", None, outcome)]
        assert session._observed.closed
        assert session._observed.fingerprints == set()
    finally:
        engine.close(timeout_seconds=2)


def test_nested_private_mutation_remains_a_distinct_observation_subject() -> None:
    """Deduplication reads complete current values even when the request object is reused."""
    classifier = _Classifier(GuardrailOutcome.ALLOW)
    classifier.release.set()
    engine = _Engine(classifier)
    _, request = _changed_subject("context_management")
    session = _session(engine)
    try:
        session.inspect_input(request)
        assert engine._observations._idle.wait(2)
        assert request.context_management is not None
        request.context_management["private"] = "changed"
        session.inspect_input(request)
        engine.close(timeout_seconds=2)
        assert classifier.input_calls == 2
    finally:
        engine.close(timeout_seconds=2)


def test_direct_observe_output_never_enforces_a_response_bound() -> None:
    """The public engine API leaves an input-only observer's completion untouched."""
    classifier = _Classifier(GuardrailOutcome.FLAGGED)
    engine = _Engine(classifier)
    completion = GuardrailCompletion(text="oversized")
    try:
        result = asyncio.run(
            engine.enforce_output(
                policy=_policy().model_copy(update={"max_response_bytes": 1}),
                completion=completion,
                deadline_monotonic=time.monotonic() - 1,
            )
        )
        assert result is completion
        assert engine.output_invocations == 0
        assert classifier.output_calls == 0
        assert engine.enforced == engine.observed == []
    finally:
        engine.close(timeout_seconds=2)


def test_only_enforcing_policies_change_replay_identity() -> None:
    """Mixed policies preserve enforcement identity when only observation changes."""
    engine = _Engine(_Classifier(GuardrailOutcome.ALLOW))
    observer = _policy()
    revised = observer.model_copy(update={"revision": "new-detector"})
    enforcer = observer.model_copy(update={"mode": "enforce", "policy_id": "enforcer"})
    try:
        assert engine.policy_revision(()) is None
        assert engine.policy_revision((observer,)) is None
        assert engine.policy_revision((revised,)) is None
        original = engine.policy_revision((enforcer,))
        assert original is not None
        assert engine.policy_revision((observer, enforcer)) == original
        assert engine.policy_revision((enforcer, revised)) == original
        assert (
            engine.policy_revision((enforcer.model_copy(update={"revision": "changed"}),))
            != original
        )
        assert (
            engine.policy_revision((observer.model_copy(update={"mode": "enforce"}), enforcer))
            != original
        )
    finally:
        engine.close(timeout_seconds=2)


@pytest.mark.parametrize("surface", ["chat", "responses"])
@pytest.mark.parametrize("change", ["added", "revised"])
@pytest.mark.parametrize("phase", ["during_admission", "after_completion"])
def test_native_replay_survives_observation_reload_but_detects_enforcement_promotion(
    tmp_path: Path, surface: str, change: str, phase: str
) -> None:
    """The real HTTP replay boundary ignores observer-only reloads without ignoring promotion."""
    observer = _policy()
    before = () if change == "added" else (observer,)
    after = (observer.model_copy(update={"revision": "changed"}),)
    engine = GuardrailEngine(
        store=MappingGuardrailStore(before),
        client=DirectClassifierClient(ClassifierRegistry({"detector": ScriptedClassifier()})),
        monotonic=time.monotonic,
    )

    class ReloadingPlane(NativeControlPlane):
        """Reload once at the narrow claim-to-admission boundary when selected."""

        reloaded = False

        def claim_scope(self, argument: str) -> str:
            """Move only observation configuration after replay scope has been claimed."""
            scope = super().claim_scope(argument)
            if phase == "during_admission" and not self.reloaded:
                engine._store = MappingGuardrailStore(after)
                self.reloaded = True
            return scope

    try:
        with _provider(_content_chunk("allowed") + _terminal_frames()) as (base_url, received):
            _, key = _configured_gateway(tmp_path, base_url=base_url)
            components = load_gateway_components(
                tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"}
            )
            route, body = _body(surface, False)
            headers = {"authorization": f"Bearer {key}", "Idempotency-Key": "operation"}
            with _serving(ReloadingPlane(components, guardrails=engine)) as url:
                accepted = httpx.post(url + route, headers=headers, json=body, timeout=10)
                assert accepted.status_code == 200, accepted.text
                if phase == "after_completion":
                    engine._store = MappingGuardrailStore(after)
                replay = httpx.post(url + route, headers=headers, json=body, timeout=10)
                assert replay.status_code == 200, replay.text
                assert replay.content == accepted.content
                assert len(received) == 1
                engine._store = MappingGuardrailStore(
                    (after[0].model_copy(update={"mode": "enforce"}),)
                )
                conflict = httpx.post(url + route, headers=headers, json=body, timeout=10)
                assert conflict.status_code == 409, conflict.text
                assert len(received) == 1
    finally:
        engine.close(timeout_seconds=2)


def test_observation_recorder_failure_cannot_reject_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both synchronous coverage reporting and asynchronous decisions tolerate a broken recorder."""
    classifier = _Classifier(GuardrailOutcome.ALLOW)
    engine = _Engine(classifier)
    recorded = threading.Event()

    def unavailable(
        policy: GuardrailPolicy,
        check: GuardrailCheck | None,
        outcome: GuardrailOutcome,
        latency_seconds: float,
    ) -> None:
        """Model a failed host metrics sink without exposing its diagnostic."""
        del policy, check, outcome, latency_seconds
        recorded.set()
        raise RuntimeError("private-recorder-error")

    monkeypatch.setattr(engine, "_record_observation", unavailable)
    session = _session(engine)
    try:
        require_unguarded_surface(engine, _authorization(), "embeddings")
        assert recorded.wait(2)
        recorded.clear()
        assert session.inspect_input(_request()) == _request()
        classifier.release.set()
        assert recorded.wait(2)
        assert session.settlement_failure() is None
    finally:
        classifier.release.set()
        engine.close(timeout_seconds=2)


@pytest.mark.parametrize("surface", ["embeddings", "images", "Decisions"])
def test_observer_only_policies_do_not_fence_unsupported_surfaces(surface: str) -> None:
    """Unsupported observation coverage is metadata rather than a customer rejection."""
    classifier = _Classifier(GuardrailOutcome.FLAGGED)
    engine = _Engine(classifier)
    require_unguarded_surface(engine, _authorization(), surface)
    engine.close(timeout_seconds=2)
    assert engine.observed == [("platform-observer", None, GuardrailOutcome.UNSUPPORTED)]
    assert classifier.input_calls == 0
    engine.close(timeout_seconds=0)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("outcome", [GuardrailOutcome.FLAGGED, GuardrailOutcome.UNSUPPORTED])
def test_native_provider_finishes_and_is_charged_before_observation_completes(
    tmp_path: Path, surface: str, stream: bool, outcome: GuardrailOutcome
) -> None:
    """Actual native HTTP delivery and ledger settlement never wait for an observer."""
    classifier = _Classifier(outcome)
    engine = _Engine(classifier, _policy(timeout_ms=5000))
    marker = "provider response before observation"
    with _provider(_content_chunk(marker) + _terminal_frames()) as (url, received):
        _, key = _configured_gateway(
            tmp_path,
            base_url=url,
            billing_source=BillingSource.HOST_MANAGED,
            prices=GatewayTokenPrices(
                input_nano_usd_per_million_tokens=1_000_000_000,
                output_nano_usd_per_million_tokens=2_000_000_000,
            ),
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        control = NativeControlPlane(components, guardrails=engine)
        try:
            with _serving(control) as gateway:
                path, body = _body(surface, stream)
                response = httpx.post(
                    gateway + path,
                    headers={"authorization": f"Bearer {key}"},
                    json=body,
                    timeout=3,
                )
                assert response.status_code == 200, response.text
                assert marker in response.text
                assert classifier.started.wait(2)
                assert not classifier.release.is_set() and not classifier.cancelled.is_set()
                assert engine.observed == []
                with sqlite3.connect(components.ledger.database_path) as db:
                    rows = db.execute(
                        "select state, failure_class, budget_settled_nano_usd from gateway_attempts"
                    ).fetchall()
                assert len(rows) == 1
                assert rows[0][:2] == ("completed", None)
                assert rows[0][2] > 0
                assert len(received) == 1
                classifier.release.set()
                assert engine.recorded.wait(2)
                assert engine.observed == [("platform-observer", "input", outcome)]
                assert engine.enforced == []
        finally:
            classifier.release.set()
            engine.close(timeout_seconds=2)


@pytest.mark.parametrize("observing", [True, False])
def test_websocket_turn_uses_the_same_observation_or_coverage_authority(
    tmp_path: Path, observing: bool
) -> None:
    """A WebSocket turn delivers during observation but withholds an enforced coverage failure."""
    classifier = _Classifier(GuardrailOutcome.UNSUPPORTED)
    policy = _policy(timeout_ms=5000)
    if not observing:
        policy = policy.model_copy(update={"mode": "enforce", "input_execution": "parallel"})
    engine = _Engine(classifier, policy)
    marker = "websocket provider content"
    with _provider(_content_chunk(marker) + _terminal_frames()) as (url, received):
        _, key = _configured_gateway(
            tmp_path, base_url=url, billing_source=BillingSource.HOST_MANAGED
        )
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        try:
            with _serving(NativeControlPlane(components, guardrails=engine)) as gateway:
                with connect(
                    gateway.replace("http://", "ws://") + "/v1/responses",
                    additional_headers={"authorization": f"Bearer {key}"},
                    close_timeout=0.2,
                ) as connection:
                    connection.send(
                        json.dumps(
                            {"type": "response.create", "model": "coding", "input": "fixture"}
                        )
                    )
                    assert classifier.started.wait(2)
                    if not observing:
                        with pytest.raises(TimeoutError):
                            connection.recv(timeout=0.05)
                        classifier.release.set()
                    events: list[str] = []
                    for _ in range(20):
                        event = str(connection.recv(timeout=2))
                        events.append(event)
                        if json.loads(event)["type"] in {
                            "error",
                            "response.completed",
                            "response.failed",
                        }:
                            break
                    if observing:
                        assert marker in "".join(events)
                        assert json.loads(events[-1])["type"] == "response.completed"
                        assert not classifier.release.is_set()
                    else:
                        assert marker not in "".join(events)
                        assert "unsupported_capability" in "".join(events)
                    assert len(received) == 1
        finally:
            classifier.release.set()
            engine.close(timeout_seconds=2)


def test_native_tool_search_observes_expanded_context_without_output_enforcement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observation-only sessions neither invoke the output fence nor stop gateway tool rounds."""
    classifier = _Classifier(GuardrailOutcome.FLAGGED)
    engine = _Engine(classifier, _policy(timeout_ms=5000))
    provider = _Provider()

    def forbidden_output(*args: object) -> GuardrailCompletion:
        """Expose an accidental output enforcement path before tool execution."""
        del args
        raise AssertionError("observation invoked the output fence")

    monkeypatch.setattr("exp.runtime.gateway.native_tool_search._search_output", forbidden_output)
    try:
        key = _configure(tmp_path, f"http://127.0.0.1:{provider.server.server_port}/v1")
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "fixture"})
        with _serving(NativeControlPlane(components, guardrails=engine)) as gateway:
            response = httpx.post(
                gateway + "/v1/chat/completions",
                headers={"authorization": f"Bearer {key}"},
                json={
                    "model": "coding",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": _TOOLS_CHAT,
                },
                timeout=3,
            )
            assert response.status_code == 200, response.text
            assert "18C in Bern" in response.text
            assert len(provider.requests) == 2
            assert not classifier.release.is_set()
            classifier.release.set()
            engine.close(timeout_seconds=2)
            assert any(any(m.role == "tool" for m in r.messages) for r in classifier.subjects)
            assert engine.enforced == []
    finally:
        classifier.release.set()
        engine.close(timeout_seconds=2)
        provider.close()


@pytest.mark.parametrize("outcome", [GuardrailOutcome.ALLOW, GuardrailOutcome.FLAGGED])
def test_timely_observer_verdict_survives_delayed_control_loop_delivery(
    outcome: GuardrailOutcome,
) -> None:
    """Delivery latency after worker completion cannot relabel a real verdict as a timeout."""

    class DelayedDelivery(BoundedInspect):
        """Expose scheduler delay after the actual bounded executor obtained a verdict."""

        async def _await_isolated[T](self, task: Future[T], timeout: float) -> T:
            """Keep the actual timely worker result while the caller loop pauses."""
            result = await super()._await_isolated(task, timeout)
            time.sleep(timeout + 0.025)
            return result

    classifier = _Classifier(outcome)
    classifier.release.set()
    engine = _Engine(classifier, _policy(timeout_ms=500))
    engine._observation_inspects = DelayedDelivery(max_inflight=1)
    try:
        _session(engine).inspect_input(_request())
        assert engine.recorded.wait(5)
        assert engine.observed == [("platform-observer", "input", outcome)]
        assert engine.enforced == []
    finally:
        engine.close(timeout_seconds=2)


def test_blocked_observation_sink_cannot_stall_skips_or_unsupported_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saturated inspection and a blocked metadata sink leave subsequent admissions prompt."""
    classifier = _Classifier(GuardrailOutcome.ALLOW)
    engine = _Engine(
        classifier, _policy(timeout_ms=5000), max_observations=1, max_observation_records=2
    )
    started = threading.Event()
    release = threading.Event()
    threads: list[str] = []

    def blocked(
        policy: GuardrailPolicy,
        check: GuardrailCheck | None,
        outcome: GuardrailOutcome,
        latency_seconds: float,
    ) -> None:
        """Model an unavailable external logging sink without accessing request content."""
        del policy, check, outcome, latency_seconds
        threads.append(threading.current_thread().name)
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(engine, "_record_observation", blocked)
    try:
        session = _session(engine)
        session.inspect_input(_request("first"))
        assert classifier.started.wait(2)
        session.inspect_input(_request("second"))
        assert started.wait(2)
        before = time.monotonic()
        session.inspect_input(_request("third"))
        require_unguarded_surface(engine, _authorization(), "embeddings")
        assert time.monotonic() - before < 0.5
        assert engine.observation_recording_dropped == 1
        assert classifier.input_calls == 1
        assert threads == ["exp-guardrail-record"]
        before = time.monotonic()
        engine.close(timeout_seconds=0.2)
        assert time.monotonic() - before < 0.3
        assert not release.is_set()
        assert engine.observation_recording_dropped >= 2
        assert session.settlement_failure() is None
    finally:
        classifier.release.set()
        release.set()
        engine.close(timeout_seconds=2)


def test_shutdown_drains_a_timely_observer_verdict_without_cancelling_it() -> None:
    """A positive host drain budget preserves an observation that completes during shutdown."""
    classifier = _Classifier(GuardrailOutcome.FLAGGED)
    engine = _Engine(classifier, _policy(timeout_ms=5000))
    release = threading.Timer(0.02, classifier.release.set)
    try:
        _session(engine).inspect_input(_request())
        assert classifier.started.wait(2)
        release.start()
        engine.close(timeout_seconds=2)
        assert engine.observed == [("platform-observer", "input", GuardrailOutcome.FLAGGED)]
        assert not classifier.cancelled.is_set()
        assert engine.observation_recording_dropped == 0
        assert engine.observation_recording_failed == 0
    finally:
        release.cancel()
        classifier.release.set()
        engine.close(timeout_seconds=2)


def test_shutdown_late_cancellation_record_counts_as_loss_without_exceeding_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired shutdown budget cannot promise delivery of a later SKIPPED callback."""
    classifier = _Classifier(GuardrailOutcome.FLAGGED)
    engine = _Engine(classifier, _policy(timeout_ms=5000))
    callback_started = threading.Event()
    callback_release = threading.Event()
    emit = engine._emit_observation

    def delayed_emit(
        policy: GuardrailPolicy,
        check: GuardrailCheck | None,
        outcome: GuardrailOutcome,
        latency_seconds: float,
    ) -> None:
        """Place the shutdown cancellation callback deterministically after recorder closure."""
        if outcome is GuardrailOutcome.SKIPPED:
            callback_started.set()
            assert callback_release.wait(5)
        emit(policy, check, outcome, latency_seconds)

    monkeypatch.setattr(engine, "_emit_observation", delayed_emit)
    try:
        _session(engine).inspect_input(_request())
        assert classifier.started.wait(2)
        started = time.monotonic()
        engine.close(timeout_seconds=0.02)
        assert time.monotonic() - started < 0.5
        assert callback_started.wait(2)
        assert engine.observed == []
        callback_release.set()
        assert engine._observations._idle.wait(2)
        assert engine.observation_recording_dropped == 1
        assert engine.observed == []
    finally:
        callback_release.set()
        classifier.release.set()
        engine.close(timeout_seconds=2)

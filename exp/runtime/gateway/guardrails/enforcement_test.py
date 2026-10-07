"""Policy-chain, timeout, fail-closed, and tool-call enforcement tests."""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import time
import weakref
from collections.abc import Coroutine, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Never

import httpx
import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import (
    EncryptedReasoningBlock,
    GatewayApiSurface,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
    OpaqueReasoningContentBlock,
    SealedReasoningContentBlock,
)
from exp.runtime.gateway.guardrails import enforcement, observation
from exp.runtime.gateway.guardrails.bounded import BoundedInspect, start_on_native_loop
from exp.runtime.gateway.guardrails.classifiers import (
    ClassifierRegistry,
    ScriptedClassifier,
)
from exp.runtime.gateway.guardrails.client import DirectClassifierClient, InspectingClassifier
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailOutcome,
    GuardrailPolicy,
    GuardrailRejected,
    GuardrailToolCall,
    OutputGuardrailMode,
    request_content_bytes,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.http_json import HttpJsonClassifier
from exp.runtime.gateway.guardrails.observation import ObservationAdmission, ObservationLease
from exp.runtime.gateway.guardrails.regex import (
    BuiltinPattern,
    RegexAdapterDocument,
    RegexClassifier,
)
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.subjects import ObservedSubjects, observation_subject_bytes


@pytest.fixture(scope="module", autouse=True)
def _ready_observation_scheduler() -> None:
    """Qualify engine decisions on a running scheduler; cold admission has separate coverage."""
    start_on_native_loop(asyncio.sleep(0)).result(timeout=2)


class _Clock:
    """Monotonic clock that tests can advance between classifier calls."""

    def __init__(self) -> None:
        """Start at a fixed instant."""
        self.now = 100.0

    def __call__(self) -> float:
        """Return the current test instant."""
        return self.now


class _RaisingClassifier(ScriptedClassifier):
    """Adapter that fails every inspection."""

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Raise instead of returning a verdict."""
        del request, check
        self.input_calls += 1
        raise RuntimeError("classifier unavailable")

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Raise instead of returning a verdict."""
        del completion, check
        self.output_calls += 1
        raise RuntimeError("classifier unavailable")


class _HungAsyncClassifier:
    """Adapter that never returns unless the inspect is cancelled."""

    def __init__(self) -> None:
        """Start with empty call counts."""
        self.input_calls = 0
        self.output_calls = 0

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Wait until cancelled."""
        del request, check
        self.input_calls += 1
        await asyncio.Event().wait()
        return ClassifierVerdict(flagged=False)

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Wait until cancelled."""
        del completion, check
        self.output_calls += 1
        await asyncio.Event().wait()
        return ClassifierVerdict(flagged=False)


class _BlockingBeforeAwaitClassifier:
    """Adapter that blocks the inspect thread before its first await."""

    def __init__(self) -> None:
        """Start with empty call counts and a hold that never releases by default."""
        self.input_calls = 0
        self.output_calls = 0
        self.entered = threading.Event()
        self.hold = threading.Event()

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Block before yielding so a shared event loop cannot time out."""
        del request, check
        self.input_calls += 1
        self.entered.set()
        self.hold.wait(timeout=5.0)
        return ClassifierVerdict(flagged=False)

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Block before yielding on the output path."""
        del completion, check
        self.output_calls += 1
        self.entered.set()
        self.hold.wait(timeout=5.0)
        return ClassifierVerdict(flagged=False)


def _awaited[T](coro: Coroutine[object, object, T]) -> T:
    """Run one enforcement coroutine on a private loop."""
    return asyncio.run(coro)


async def _wait_hold(hold: threading.Event, *, timeout: float = 5.0) -> None:
    """Wait for a test to release an abandoned inspect, then give up."""
    deadline = time.monotonic() + timeout
    while not hold.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)


def _check(
    check_id: str,
    *,
    stage: GuardrailCheckStage = GuardrailCheckStage.INPUT,
    action: GuardrailAction = GuardrailAction.BLOCK,
    adapter_id: str = "scripted",
    timeout_ms: int = 200,
) -> GuardrailCheck:
    """Build one valid check."""
    return GuardrailCheck(
        check_id=check_id,
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=stage,
        action=action,
        timeout_ms=timeout_ms,
        adapter_id=adapter_id,
    )


def _engine[C: InspectingClassifier](
    *,
    classifier: C,
    checks: tuple[GuardrailCheck, ...],
    protected: bool = False,
    clock: _Clock | None = None,
    max_request_bytes: int = 1_048_576,
    max_response_bytes: int = 1_048_576,
) -> tuple[GuardrailEngine, C]:
    """Compose one engine over a single identity policy."""
    policy = GuardrailPolicy(
        policy_id="member-policy",
        organization_id="organization-one",
        identity_id="identity-one",
        protected=protected,
        checks=checks,
        max_request_bytes=max_request_bytes,
        max_response_bytes=max_response_bytes,
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"scripted": classifier})),
        monotonic=(clock or _Clock()),
    )
    return engine, classifier


def _request(*contents: str) -> GatewayRequest:
    """Build one chat request from user message contents."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=tuple(GatewayMessage(role="user", content=item) for item in contents),
    )


class _ObservationAdmissionEngine(GuardrailEngine):
    """Expose content-free admission records under small actual owner limits.

    Attributes:
        policy: One read-only observer with the configured complete subject limit.
        records: Recorded check identities and outcomes, populated off the request path.
        clock: Controllable admission deadline clock.
    """

    def __init__(self, *, max_jobs: int = 8, max_bytes: int = 65_536) -> None:
        """Build the real engine with a prompt-free scripted inspection result."""
        self.policy = GuardrailPolicy(
            policy_id="observer",
            protected=True,
            mode="observe",
            max_request_bytes=max_bytes,
            checks=(_check("input"),),
        )
        self.records: list[tuple[str | None, GuardrailOutcome]] = []
        self.clock = _Clock()
        super().__init__(
            store=MappingGuardrailStore((self.policy,)),
            client=DirectClassifierClient(ClassifierRegistry({"scripted": ScriptedClassifier()})),
            monotonic=self.clock,
            max_observations=max_jobs,
            max_observation_bytes=max_bytes,
        )

    def _record_observation(
        self,
        policy: GuardrailPolicy,
        check: GuardrailCheck | None,
        outcome: GuardrailOutcome,
        latency_seconds: float,
    ) -> None:
        """Keep only immutable admission or inspection metadata for assertions."""
        del policy, latency_seconds
        self.records.append((None if check is None else check.check_id, outcome))

    def observe(self, request: GatewayRequest, observed: ObservedSubjects) -> None:
        """Exercise the same request-session entry point with the original deadline."""
        self.observe_inputs(
            policies=(self.policy,), request=request, deadline_monotonic=200, observed=observed
        )


def _mutable_observation_subject(field: str) -> tuple[GatewayRequest, list[JsonValue]]:
    """Expose a caller-owned nested list within one ordinary or private subject field."""
    request = _request("visible")
    nested: dict[str, JsonValue] = {"items": ["original"]}
    match field:
        case "metadata":
            request = request.model_copy(update={"metadata": nested})
            items = request.metadata["items"]
        case "provider_native_item":
            message = request.messages[0].model_copy(update={field: nested})
            request = request.model_copy(update={"messages": (message,)})
            assert message.provider_native_item is not None
            items = message.provider_native_item["items"]
        case "tool_parameters":
            tool = GatewayToolDefinition(name="fixture", parameters=nested)
            request = request.model_copy(update={"tools": (tool,)})
            items = tool.parameters["items"]
        case "tool_input_examples":
            tool = GatewayToolDefinition(
                name="fixture", parameters={"type": "object"}, input_examples=(nested,)
            )
            request = request.model_copy(update={"tools": (tool,)})
            assert tool.input_examples is not None
            items = tool.input_examples[0]["items"]
        case _:
            raise AssertionError(f"unknown fixture field: {field}")
    assert isinstance(items, list)
    return request, items


@pytest.mark.parametrize(
    "field", ["metadata", "provider_native_item", "tool_parameters", "tool_input_examples"]
)
@pytest.mark.parametrize("replacement", ["modified", "x" * 100_000], ids=["same-size", "oversized"])
def test_observation_retains_the_subject_hashed_before_caller_mutation(
    field: str, replacement: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Queued classifiers inspect their admitted copy despite later nested caller mutations."""
    engine = _ObservationAdmissionEngine(max_jobs=1)
    request, caller_items = _mutable_observation_subject(field)
    original = request.model_copy(deep=True)
    expected = observation_subject_bytes(original)
    observed = ObservedSubjects()
    started = threading.Event()
    release = threading.Event()
    captured: list[bytes] = []

    async def inspect(
        *,
        policy: GuardrailPolicy,
        request: GatewayRequest,
        deadline_monotonic: float,
        retention: ObservationLease | None = None,
    ) -> GatewayRequest:
        """Pause the real queued engine job before reading its complete input."""
        del policy, deadline_monotonic
        assert retention is not None
        started.set()
        await _wait_hold(release)
        captured.append(observation_subject_bytes(request))
        return request

    monkeypatch.setattr(engine, "_inspect_input", inspect)
    try:
        engine.observe(request, observed)
        assert started.wait(2)
        caller_items[0] = replacement
        assert observation_subject_bytes(request) != expected
        assert engine._observations._bytes == len(expected)
        engine.observe(original, observed)
        assert len(observed.fingerprints) == 1
        assert len(engine._observations._jobs) == 1
    finally:
        release.set()
        engine.close(timeout_seconds=2)
    assert captured == [expected]
    assert not engine._observations._jobs
    assert engine._observations._bytes == 0


def test_observation_scheduler_failure_is_unavailable_and_releases_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable loop is distinct from a skip and cannot consume the next subject slot."""
    engine = _ObservationAdmissionEngine(max_jobs=1)
    original = observation.try_start_on_native_loop
    attempts: list[bool] = []

    def fail(_coroutine: Coroutine[object, object, None]) -> Never:
        """Fail before the native callback loop can own the observation coroutine."""
        attempts.append(True)
        raise RuntimeError("synthetic scheduler startup failure")

    monkeypatch.setattr(observation, "try_start_on_native_loop", fail)
    try:
        engine.observe(_request("first"), ObservedSubjects())
        assert attempts == [True]
        assert engine.input_invocations == 0
        assert not engine._observations._jobs
        assert engine._observations._bytes == 0
        assert engine._observations._idle.is_set()
        monkeypatch.setattr(observation, "try_start_on_native_loop", original)
        engine.observe(_request("second"), ObservedSubjects())
    finally:
        engine.close(timeout_seconds=2)
    assert engine.records == [
        (None, GuardrailOutcome.UNAVAILABLE),
        ("input", GuardrailOutcome.ALLOW),
    ]


@pytest.mark.parametrize("unavailable", ["jobs", "bytes", "closed", "deadline"])
@pytest.mark.parametrize("oversized", [False, True])
def test_new_observation_rejects_before_serialization_when_admission_is_unavailable(
    unavailable: str, oversized: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full, closed and expired sessions cannot allocate another near-cap encoded subject."""
    engine = _ObservationAdmissionEngine(max_jobs=1 if unavailable == "jobs" else 8)
    release = threading.Event()
    started = threading.Event()
    observed = ObservedSubjects()
    size = engine._observations.max_subject_bytes
    content_bytes = size - len(observation_subject_bytes(_request(""))) - 16
    request = _request("x" * (size * 2 if oversized else content_bytes))

    async def hold(_lease: ObservationLease) -> None:
        """Retain an existing subject while concurrent requests test unavailable capacity."""
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.001)

    def forbidden(_request: GatewayRequest) -> bytes:
        """Fail if rejection performs any complete subject encoding."""
        pytest.fail("unavailable admission serialized an input subject")

    def forbidden_bound(_request: GatewayRequest, _limit: int) -> bool:
        """Admission rejection has priority over an unknown oversized classification."""
        pytest.fail("unavailable admission projected an input subject")

    if unavailable in {"jobs", "bytes"}:
        assert (
            engine._observations.submit(
                hold,
                subject_bytes=size if unavailable == "bytes" else 1,
                on_interrupted=lambda _: None,
            )
            is ObservationAdmission.ACCEPTED
        )
        assert started.wait(2)
    elif unavailable == "closed":
        engine._observations.close(timeout_seconds=0)
    else:
        engine.clock.now = 201
    monkeypatch.setattr(enforcement, "observation_subject_bytes", forbidden)
    monkeypatch.setattr(enforcement, "observation_subject_exceeds", forbidden_bound)
    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            futures = [workers.submit(engine.observe, request, observed) for _ in range(4)]
            for future in futures:
                future.result(timeout=2)
        engine.observe(request.model_copy(deep=True), observed)
    finally:
        release.set()
        engine.close(timeout_seconds=2)
    assert observed.closed_reason == "preparation_unavailable"
    assert observed.fingerprints == set()
    assert engine.records == [(None, GuardrailOutcome.SKIPPED)]


@pytest.mark.parametrize("same_session", [False, True])
def test_concurrent_preparation_has_one_near_cap_encoder_and_no_waiting_requests(
    same_session: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contending sessions return without serializing and record coverage once each."""
    engine = _ObservationAdmissionEngine()
    original = enforcement.observation_subject_bytes
    release = threading.Event()
    started = threading.Event()
    prepared: list[int] = []
    size = engine._observations.max_subject_bytes
    request = _request("x" * (size - len(original(_request(""))) - 16))

    def blocked(payload: GatewayRequest) -> bytes:
        """Hold one real encoded allocation while other request threads enter the engine."""
        subject = original(payload)
        prepared.append(len(subject))
        if len(prepared) == 1:
            started.set()
            assert release.wait(3)
        return subject

    monkeypatch.setattr(enforcement, "observation_subject_bytes", blocked)
    first_state = ObservedSubjects()
    with ThreadPoolExecutor(max_workers=5) as workers:
        first = workers.submit(engine.observe, request, first_state)
        try:
            assert started.wait(2)
            rejected = [first_state if same_session else ObservedSubjects() for _ in range(4)]
            futures = [workers.submit(engine.observe, request, state) for state in rejected]
            for future in futures:
                future.result(timeout=1)
            assert prepared == [size - 16]
            assert engine._observations._preparing
            assert all(state.closed_reason == "preparation_unavailable" for state in rejected)
            for state in rejected:
                engine.observe(request.model_copy(deep=True), state)
        finally:
            release.set()
            first.result(timeout=2)
            engine.close(timeout_seconds=2)
    assert engine.records.count((None, GuardrailOutcome.SKIPPED)) == (1 if same_session else 4)
    assert engine.records.count(("input", GuardrailOutcome.ALLOW)) == 1


def test_known_subject_deduplicates_after_deadline_with_a_free_preparation_permit() -> None:
    """An exact private-context deep copy does not acquire a second expired outcome."""
    engine = _ObservationAdmissionEngine()
    request = _request("original").model_copy(update={"context_management": {"private": "value"}})
    observed = ObservedSubjects()
    try:
        engine.observe(request, observed)
        assert engine._observations._idle.wait(2)
        engine.clock.now = 201
        engine.observe(request.model_copy(deep=True), observed)
    finally:
        engine.close(timeout_seconds=2)
    assert engine.records == [("input", GuardrailOutcome.ALLOW)]
    assert not observed.closed


def test_contended_known_subject_reports_terminal_admission_not_a_distinct_verdict() -> None:
    """An unavailable preparation cannot identify an equal copy, so coverage ends once."""
    engine = _ObservationAdmissionEngine()
    request = _request("original")
    observed = ObservedSubjects()
    try:
        engine.observe(request, observed)
        assert engine._observations._idle.wait(2)
        with engine._observations.preparation() as admitted:
            assert admitted
            engine.observe(request.model_copy(deep=True), observed)
            engine.observe(request, observed)
        assert observed.closed_reason == "preparation_unavailable"
        assert len(observed.fingerprints) == 1
    finally:
        engine.close(timeout_seconds=2)
    assert engine.records == [("input", GuardrailOutcome.ALLOW), (None, GuardrailOutcome.SKIPPED)]


def test_engine_close_does_not_revoke_an_inflight_encoded_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded close leaves the actual encoder charged and rejects later preparation."""
    engine = _ObservationAdmissionEngine()
    original = enforcement.observation_subject_bytes
    started = threading.Event()
    release = threading.Event()
    prepared: list[int] = []

    def blocked(request: GatewayRequest) -> bytes:
        """Retain a real subject allocation beyond the engine's close deadline."""
        encoded = original(request)
        prepared.append(len(encoded))
        started.set()
        assert release.wait(3)
        return encoded

    monkeypatch.setattr(enforcement, "observation_subject_bytes", blocked)
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(engine.observe, _request("original"), ObservedSubjects())
        try:
            assert started.wait(2)
            before = time.monotonic()
            engine.close(timeout_seconds=0.02)
            assert time.monotonic() - before < 0.5
            assert engine._observations._preparing
            assert not engine._observations._idle.is_set()
            engine.observe(_request("new context"), ObservedSubjects())
            assert len(prepared) == 1
        finally:
            release.set()
            pending.result(timeout=2)
            engine.close(timeout_seconds=2)
    assert not engine._observations._preparing
    assert engine._observations._idle.is_set()
    assert not engine._observations._jobs
    assert engine.records == []
    assert engine.observation_recording_dropped == 2


@pytest.mark.parametrize("stage", ["bound", "snapshot", "encode", "hash"])
def test_failed_preparation_drops_temporary_traceback_before_releasing_the_permit(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Projection and hashing failures cannot leak a transient allocation past admission."""

    class Temporary:
        """Represent a private allocation retained only by a failing preparation frame."""

    engine = _ObservationAdmissionEngine()
    original = engine._observations.preparation
    retained: list[weakref.ReferenceType[Temporary]] = []

    def fail(_payload: GatewayRequest | bytes) -> Never:
        """Raise with an observable transient object in the traceback's local frame."""
        temporary = Temporary()
        retained.append(weakref.ref(temporary))
        raise ValueError("synthetic preparation failure")

    def fail_bound(payload: GatewayRequest, _limit: int) -> Never:
        """Fail the allocation-free projection before canonical encoding can begin."""
        fail(payload)

    def fail_snapshot(payload: GatewayRequest, *, deep: bool) -> Never:
        """Fail while allocating the detached nested subject."""
        assert deep
        fail(payload)

    @contextmanager
    def checked(*, deduplicating: bool = False) -> Iterator[bool]:
        """Verify the temporary is gone before granting any other preparation permission."""
        with original(deduplicating=deduplicating) as admitted:
            yield admitted
            gc.collect()
            assert engine._observations._preparing
            assert retained and retained[0]() is None

    monkeypatch.setattr(engine._observations, "preparation", checked)
    if stage == "bound":
        monkeypatch.setattr(enforcement, "observation_subject_exceeds", fail_bound)
    elif stage == "snapshot":
        monkeypatch.setattr(GatewayRequest, "model_copy", fail_snapshot)
    else:
        monkeypatch.setattr(
            enforcement, "observation_subject_bytes" if stage == "encode" else "sha256", fail
        )
    observed = ObservedSubjects()
    try:
        engine.observe(_request("original"), observed)
        assert not engine._observations._preparing
        assert engine._observations._idle.is_set()
    finally:
        engine.close(timeout_seconds=2)
    assert observed.closed_reason == "preparation_unavailable"
    assert engine.records == [(None, GuardrailOutcome.UNAVAILABLE)]


def test_input_chain_runs_once_and_can_transform_the_request() -> None:
    """A modify action replaces messages for every later consumer of the request."""
    replacement = (GatewayMessage(role="user", content="redacted"),)
    engine, classifier = _engine(
        classifier=ScriptedClassifier(
            input_verdict=ClassifierVerdict(flagged=True, replacement_messages=replacement)
        ),
        checks=(_check("input-one", action=GuardrailAction.MODIFY),),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    result = _awaited(
        engine.enforce_input(
            policy=policy,
            request=_request("original"),
            deadline_monotonic=200.0,
        )
    )

    assert result.messages == replacement
    assert classifier.input_calls == 1


def test_input_modifier_rejects_carrier_removal_with_bound_tool_history() -> None:
    """A guardrail cannot keep a Fireworks tool turn after removing its carrier."""
    call = ToolCall(
        call_id="call-one",
        name="lookup",
        arguments={},
        raw_arguments="{}",
    )
    original_messages = (
        GatewayMessage(role="user", content="Use a tool"),
        GatewayMessage(
            role="assistant",
            tool_calls=(call,),
            provider_reasoning=(
                OpaqueReasoningContentBlock(
                    route_sha256="a" * 64,
                    content="private reasoning",
                ),
            ),
        ),
        GatewayMessage(role="tool", content="done", tool_call_id="call-one"),
    )
    replacement = (
        original_messages[0],
        original_messages[1].model_copy(update={"provider_reasoning": ()}),
        original_messages[2],
    )
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(
            input_verdict=ClassifierVerdict(
                flagged=True,
                replacement_messages=replacement,
            )
        ),
        checks=(_check("input-modify", action=GuardrailAction.MODIFY),),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=original_messages,
    )

    with pytest.raises(GuardrailRejected):
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=request,
                deadline_monotonic=200.0,
            )
        )


def test_input_modifier_may_redact_after_an_exact_sealed_turn() -> None:
    """A modifier may redact a tool result after preserving the sealed turn exactly."""
    call = ToolCall(call_id="call-one", name="lookup", arguments={}, raw_arguments="{}")
    assistant = GatewayMessage(
        role="assistant",
        tool_calls=(call,),
        provider_reasoning=(
            SealedReasoningContentBlock(carrier="opaque-carrier", deployment_hint="rung-one"),
        ),
    )
    tool = GatewayMessage(role="tool", content="done", tool_call_id="call-one")
    original = (GatewayMessage(role="user", content="secret"), assistant, tool)
    replacement = (
        original[0],
        assistant,
        tool.model_copy(update={"content": "redacted"}),
    )
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(
            input_verdict=ClassifierVerdict(
                flagged=True,
                replacement_messages=replacement,
            )
        ),
        checks=(_check("input-modify", action=GuardrailAction.MODIFY),),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    result = _awaited(
        engine.enforce_input(
            policy=policy,
            request=GatewayRequest(surface=GatewayApiSurface.CHAT_COMPLETIONS, messages=original),
            deadline_monotonic=200.0,
        )
    )

    assert result.messages == replacement


def test_input_modifier_may_fully_redact_carrier_bound_history() -> None:
    """A user-only replacement safely clears the entire carrier-bound tool turn."""
    original_messages = (
        GatewayMessage(role="user", content="Use a tool"),
        GatewayMessage(
            role="assistant",
            content="working",
            provider_reasoning=(
                OpaqueReasoningContentBlock(
                    route_sha256="a" * 64,
                    content="private reasoning",
                ),
            ),
        ),
    )
    replacement = (GatewayMessage(role="user", content="redacted"),)
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(
            input_verdict=ClassifierVerdict(
                flagged=True,
                replacement_messages=replacement,
            )
        ),
        checks=(_check("input-modify", action=GuardrailAction.MODIFY),),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    result = _awaited(
        engine.enforce_input(
            policy=policy,
            request=GatewayRequest(
                surface=GatewayApiSurface.CHAT_COMPLETIONS,
                messages=original_messages,
            ),
            deadline_monotonic=200.0,
        )
    )

    assert result.messages == replacement
    assert engine.input_invocations == 1


def _reasoning_request() -> GatewayRequest:
    """Build one provider-reasoning tool continuation."""
    return GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="Use a tool"),
            GatewayMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        call_id="call-one",
                        name="lookup",
                        arguments={"query": "safe"},
                        raw_arguments='{"query":"safe"}',
                        provider_item_id="fc-one",
                        provider_output_index=1,
                    ),
                ),
                provider_reasoning=(
                    EncryptedReasoningBlock(
                        id="rs-one",
                        encrypted_content="authenticated hidden state",
                        output_index=0,
                    ),
                ),
            ),
            GatewayMessage(role="tool", tool_call_id="call-one", content="secret result"),
        ),
    )


def test_input_modify_can_redact_only_history_after_authenticated_reasoning() -> None:
    """A modifier may redact tool output without rebinding the authenticated prefix."""
    request = _reasoning_request()
    replacement = (
        *request.messages[:2],
        GatewayMessage(role="tool", tool_call_id="call-one", content="[REDACTED]"),
    )
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(
            input_verdict=ClassifierVerdict(flagged=True, replacement_messages=replacement)
        ),
        checks=(_check("input-one", action=GuardrailAction.MODIFY),),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    result = _awaited(
        engine.enforce_input(policy=policy, request=request, deadline_monotonic=200.0)
    )
    assert result.messages == replacement


def test_input_modify_rejects_removing_provider_reasoning_with_bound_history() -> None:
    """A modifier cannot truncate authenticated provider history to a new user turn."""
    request = _reasoning_request()
    replacement = (GatewayMessage(role="user", content="fully redacted"),)
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(
            input_verdict=ClassifierVerdict(flagged=True, replacement_messages=replacement)
        ),
        checks=(_check("input-one", action=GuardrailAction.MODIFY),),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    with pytest.raises(GuardrailRejected):
        _awaited(engine.enforce_input(policy=policy, request=request, deadline_monotonic=200.0))


def test_hosted_input_modify_restores_exact_hidden_continuation_authority() -> None:
    """Hosted visible edits are validated before exact hidden authority is reattached."""
    request = _reasoning_request()
    check = _check("hosted-input", action=GuardrailAction.MODIFY, adapter_id="hosted")
    policy = GuardrailPolicy(
        policy_id="member-policy",
        organization_id="organization-one",
        identity_id="identity-one",
        protected=True,
        checks=(check,),
    )

    def handler(outbound: httpx.Request) -> httpx.Response:
        """Redact classifier-visible tool output without receiving hidden authority."""
        payload = json.loads(outbound.content)
        visible = payload["request"]["messages"]
        assistant = visible[1]
        assert "provider_reasoning" not in assistant
        assert "raw_arguments" not in assistant["tool_calls"][0]
        assert "provider_item_id" not in assistant["tool_calls"][0]
        visible[2]["content"] = "[REDACTED]"
        return httpx.Response(
            200,
            json={"flagged": True, "replacement_messages": visible},
        )

    async def scenario() -> GatewayRequest:
        """Run one hosted adapter through the complete guardrail engine."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = HttpJsonClassifier(
                adapter_id="hosted",
                url="https://classifier.example.invalid/v1/inspect",
                client=client,
            )
            engine = GuardrailEngine(
                store=MappingGuardrailStore((policy,)),
                client=DirectClassifierClient(ClassifierRegistry({"hosted": adapter})),
                monotonic=_Clock(),
            )
            return await engine.enforce_input(
                policy=policy,
                request=request,
                deadline_monotonic=200.0,
            )

    result = asyncio.run(scenario())
    assert result.messages[:2] == request.messages[:2]
    assert result.messages[2].content == "[REDACTED]"
    original_call = request.messages[1].tool_calls[0]
    restored_call = result.messages[1].tool_calls[0]
    assert restored_call.raw_arguments == original_call.raw_arguments
    assert restored_call.provider_item_id == original_call.provider_item_id
    assert restored_call.provider_output_index == original_call.provider_output_index
    assert result.messages[1].provider_reasoning == request.messages[1].provider_reasoning


def test_input_modify_restores_carrier_omitted_from_classifier_projection() -> None:
    """A visible-only replacement receives the original authenticated carrier."""
    request = _reasoning_request()
    replacement = (
        request.messages[0],
        request.messages[1].model_copy(update={"provider_reasoning": ()}),
        request.messages[2],
    )
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(
            input_verdict=ClassifierVerdict(flagged=True, replacement_messages=replacement)
        ),
        checks=(_check("input-one", action=GuardrailAction.MODIFY),),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    result = _awaited(
        engine.enforce_input(policy=policy, request=request, deadline_monotonic=200.0)
    )
    assert result.messages == request.messages


def test_input_modify_restores_hidden_tool_authority_without_reasoning() -> None:
    """Raw call bytes and provider identity are restored after visible validation."""
    original = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="Use a tool"),
            GatewayMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        call_id="call-one",
                        name="lookup",
                        arguments={"query": "safe"},
                        raw_arguments='{"query":"safe"}',
                        provider_item_id="fc-one",
                        provider_output_index=0,
                        provider_status="completed",
                    ),
                ),
            ),
            GatewayMessage(role="tool", tool_call_id="call-one", content="result"),
        ),
    )
    stripped_call = (
        original.messages[1]
        .tool_calls[0]
        .model_copy(
            update={
                "raw_arguments": None,
                "provider_item_id": None,
                "provider_output_index": None,
                "provider_status": None,
            }
        )
    )
    replacement = (
        original.messages[0],
        original.messages[1].model_copy(update={"tool_calls": (stripped_call,)}),
        original.messages[2],
    )
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(
            input_verdict=ClassifierVerdict(flagged=True, replacement_messages=replacement)
        ),
        checks=(_check("input-one", action=GuardrailAction.MODIFY),),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    result = _awaited(
        engine.enforce_input(policy=policy, request=original, deadline_monotonic=200.0)
    )
    assert result.messages == original.messages


def test_input_block_is_terminal_and_content_free() -> None:
    """A block action raises a sanitized failure without request text."""
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(input_verdict=ClassifierVerdict(flagged=True)),
        checks=(_check("input-one"),),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    with pytest.raises(GuardrailRejected) as raised:
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=_request("secret-prompt"),
                deadline_monotonic=200.0,
            )
        )

    assert raised.value.failure.failure_class is GatewayFailureClass.GUARDRAIL
    assert raised.value.failure.failover_eligible is False
    assert "secret-prompt" not in raised.value.failure.safe_message


def test_protected_identity_fail_closes_on_adapter_error() -> None:
    """Protected identities treat classifier uncertainty as a terminal error."""
    engine, classifier = _engine(
        classifier=_RaisingClassifier(),
        checks=(_check("input-one"),),
        protected=True,
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    with pytest.raises(GuardrailRejected) as raised:
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=_request("hello"),
                deadline_monotonic=200.0,
            )
        )

    assert raised.value.failure.safe_details["action"] == "error"
    assert classifier.input_calls == 1


def test_unprotected_identity_skips_a_failed_check() -> None:
    """Non-protected identities continue after an uncertain check."""
    engine, _classifier = _engine(
        classifier=_RaisingClassifier(),
        checks=(_check("input-one"),),
        protected=False,
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    result = _awaited(
        engine.enforce_input(
            policy=policy,
            request=_request("hello"),
            deadline_monotonic=200.0,
        )
    )

    assert result.messages[0].content == "hello"


def test_expired_deadline_fail_closes_for_protected_identities() -> None:
    """A check that starts after the request deadline is an error."""
    clock = _Clock()
    clock.now = 200.0
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(),
        checks=(_check("input-one"),),
        protected=True,
        clock=clock,
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    with pytest.raises(GuardrailRejected):
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=_request("hello"),
                deadline_monotonic=200.0,
            )
        )


def test_output_modify_never_rewrites_tool_call_arguments() -> None:
    """Unsafe tool-call arguments are blocked instead of rewritten."""
    engine, _classifier = _engine(
        classifier=ScriptedClassifier(
            output_verdict=ClassifierVerdict(flagged=True, replacement_text="safe")
        ),
        checks=(
            _check(
                "output-one",
                stage=GuardrailCheckStage.OUTPUT,
                action=GuardrailAction.MODIFY,
            ),
        ),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None
    completion = GuardrailCompletion(
        text="call a tool",
        tool_calls=(GuardrailToolCall(call_id="call-1", name="lookup", arguments='{"q":"x"}'),),
    )

    with pytest.raises(GuardrailRejected) as raised:
        _awaited(
            engine.enforce_output(
                policy=policy,
                completion=completion,
                deadline_monotonic=200.0,
            )
        )

    assert raised.value.failure.safe_details["action"] == "block"


def test_oversized_payload_is_a_terminal_error() -> None:
    """Byte bounds fail closed without calling a classifier."""
    engine, classifier = _engine(
        classifier=ScriptedClassifier(),
        checks=(_check("input-one"),),
        max_request_bytes=4,
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    with pytest.raises(GuardrailRejected):
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=_request("too-large"),
                deadline_monotonic=200.0,
            )
        )

    assert classifier.input_calls == 0
    assert engine.policies_for("organization-one", "identity-two") == ()
    assert engine.policies_for("organization-two", "identity-one") == ()


def test_oversized_tool_schema_is_rejected_without_calling_the_adapter() -> None:
    """Tool descriptions and schemas count toward the request subject bound."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                description="x" * 200,
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string", "description": "y" * 200}},
                },
            ),
        ),
    )
    subject_bytes = request_content_bytes(request)
    assert subject_bytes > 200
    engine, classifier = _engine(
        classifier=ScriptedClassifier(),
        checks=(_check("input-one"),),
        max_request_bytes=subject_bytes - 1,
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    with pytest.raises(GuardrailRejected):
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=request,
                deadline_monotonic=200.0,
            )
        )

    assert classifier.input_calls == 0


def test_blocking_classifier_times_out_without_waiting_for_return() -> None:
    """A hung inspect fails closed at the check timeout, not after the wait."""
    hung = _HungAsyncClassifier()
    engine, _classifier = _engine(
        classifier=hung,
        checks=(_check("input-one", timeout_ms=50),),
        protected=True,
        clock=_Clock(),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    started = time.monotonic()
    with pytest.raises(GuardrailRejected) as raised:
        _awaited(
            engine.enforce_input(
                policy=policy,
                request=_request("hello"),
                deadline_monotonic=200.0,
            )
        )
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert raised.value.failure.safe_details["action"] == "error"
    assert hung.input_calls == 1


def test_blocking_classifier_leaves_the_event_loop_free() -> None:
    """A hung inspect is cancelled on the loop without stalling other tasks."""

    async def scenario() -> None:
        """Run one timed-out inspect alongside a short sleep."""
        engine, _classifier = _engine(
            classifier=_HungAsyncClassifier(),
            checks=(_check("input-one", timeout_ms=80),),
            protected=True,
            clock=_Clock(),
        )
        policy = engine.policies_for("organization-one", "identity-one")[0]
        assert policy is not None
        progressed = False

        async def marker() -> None:
            """Flip after a delay shorter than the inspect hang."""
            nonlocal progressed
            await asyncio.sleep(0.02)
            progressed = True

        task = asyncio.create_task(marker())
        with pytest.raises(GuardrailRejected):
            await engine.enforce_input(
                policy=policy,
                request=_request("hello"),
                deadline_monotonic=200.0,
            )
        await task
        assert progressed

    asyncio.run(scenario())


def test_blocking_before_await_classifier_times_out_without_freezing_enforcement() -> None:
    """A classifier that blocks before yielding still fail-closes at the check timeout."""
    blocked = _BlockingBeforeAwaitClassifier()
    engine, _classifier = _engine(
        classifier=blocked,
        checks=(_check("input-one", timeout_ms=50),),
        protected=True,
        clock=_Clock(),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None

    try:
        started = time.monotonic()
        with pytest.raises(GuardrailRejected) as raised:
            _awaited(
                engine.enforce_input(
                    policy=policy,
                    request=_request("hello"),
                    deadline_monotonic=200.0,
                )
            )
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert raised.value.failure.safe_details["action"] == "error"
        assert blocked.entered.wait(timeout=1.0)
        assert blocked.input_calls == 1
    finally:
        blocked.hold.set()


def test_repeated_blocked_adapter_does_not_starve_a_healthy_adapter() -> None:
    """Timeouts on one adapter leave another adapter free to succeed promptly."""

    async def scenario() -> None:
        """Fail a hung adapter past the inflight cap, then inspect a healthy one."""
        hung = _HungAsyncClassifier()
        healthy = ScriptedClassifier()
        hung_policy = GuardrailPolicy(
            policy_id="blocked-policy",
            organization_id="organization-one",
            identity_id="identity-blocked",
            protected=True,
            checks=(_check("blocked-input", adapter_id="blocked", timeout_ms=40),),
        )
        healthy_policy = GuardrailPolicy(
            policy_id="healthy-policy",
            organization_id="organization-one",
            identity_id="identity-healthy",
            checks=(_check("healthy-input", adapter_id="healthy", timeout_ms=200),),
        )
        engine = GuardrailEngine(
            store=MappingGuardrailStore((hung_policy, healthy_policy)),
            client=DirectClassifierClient(
                ClassifierRegistry({"blocked": hung, "healthy": healthy})
            ),
            monotonic=time.monotonic,
            inspects=BoundedInspect(max_inflight=4),
        )

        for _ in range(8):
            with pytest.raises(GuardrailRejected):
                await engine.enforce_input(
                    policy=hung_policy,
                    request=_request("hello"),
                    deadline_monotonic=time.monotonic() + 30,
                )

        started = time.monotonic()
        result = await engine.enforce_input(
            policy=healthy_policy,
            request=_request("hello"),
            deadline_monotonic=time.monotonic() + 30,
        )
        elapsed = time.monotonic() - started

        assert result.messages[0].content == "hello"
        assert healthy.input_calls == 1
        assert hung.input_calls >= 1
        assert elapsed < 0.2

    asyncio.run(scenario())


class _CancelSwallowingClassifier:
    """Adapter that swallows cancellation and waits until the test releases it."""

    def __init__(self) -> None:
        """Start with an empty call count and a closed hold."""
        self.input_calls = 0
        self.output_calls = 0
        self._hold = threading.Event()

    def release(self) -> None:
        """Allow every abandoned inspect to finish."""
        self._hold.set()

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Ignore cancellation and wait for teardown."""
        del request, check
        self.input_calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await _wait_hold(self._hold)
            return ClassifierVerdict(flagged=False)
        return ClassifierVerdict(flagged=False)

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Ignore cancellation and wait for teardown."""
        del completion, check
        self.output_calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await _wait_hold(self._hold)
            return ClassifierVerdict(flagged=False)
        return ClassifierVerdict(flagged=False)


def test_cancel_swallowing_adapter_fail_closes_without_starving_others() -> None:
    """Protected requests return at the timeout while another adapter stays healthy."""

    async def scenario() -> None:
        """Quarantine a cancel-swallowing adapter, then inspect a healthy one."""
        rogue = _CancelSwallowingClassifier()
        healthy = ScriptedClassifier()
        inspects = BoundedInspect(max_inflight=2)
        rogue_policy = GuardrailPolicy(
            policy_id="rogue-policy",
            organization_id="organization-one",
            identity_id="identity-rogue",
            protected=True,
            checks=(_check("rogue-input", adapter_id="rogue", timeout_ms=40),),
        )
        healthy_policy = GuardrailPolicy(
            policy_id="healthy-policy",
            organization_id="organization-one",
            identity_id="identity-healthy",
            checks=(_check("healthy-input", adapter_id="healthy", timeout_ms=200),),
        )
        engine = GuardrailEngine(
            store=MappingGuardrailStore((rogue_policy, healthy_policy)),
            client=DirectClassifierClient(ClassifierRegistry({"rogue": rogue, "healthy": healthy})),
            monotonic=time.monotonic,
            inspects=inspects,
        )
        try:
            started = time.monotonic()
            with pytest.raises(GuardrailRejected) as raised:
                await engine.enforce_input(
                    policy=rogue_policy,
                    request=_request("hello"),
                    deadline_monotonic=time.monotonic() + 30,
                )
            assert time.monotonic() - started < 0.5
            assert raised.value.failure.safe_details["action"] == "error"
            assert inspects.detached_inspect_count() == 1
            assert rogue.input_calls == 1

            started = time.monotonic()
            with pytest.raises(GuardrailRejected):
                await engine.enforce_input(
                    policy=rogue_policy,
                    request=_request("hello"),
                    deadline_monotonic=time.monotonic() + 30,
                )
            assert time.monotonic() - started < 0.2
            assert inspects.detached_inspect_count() == 1
            assert rogue.input_calls == 1

            started = time.monotonic()
            result = await engine.enforce_input(
                policy=healthy_policy,
                request=_request("hello"),
                deadline_monotonic=time.monotonic() + 30,
            )
            assert time.monotonic() - started < 0.2
            assert result.messages[0].content == "hello"
            assert healthy.input_calls == 1
        finally:
            rogue.release()
            deadline = asyncio.get_running_loop().time() + 1.0
            while inspects.detached_inspect_count() != 0:
                if asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.02)
            assert inspects.detached_inspect_count() == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("limit_offset", [-1, 0])
def test_signed_tool_ids_count_toward_output_subject_limit(limit_offset: int) -> None:
    """The complete serialized subject is bounded before classifier dispatch."""
    completion = GuardrailCompletion(
        tool_calls=(
            GuardrailToolCall(
                call_id="toolu~sig1:" + "é" * 1_000,
                name="terminal",
                arguments="{}",
            ),
        ),
    )
    subject_bytes = len(canonical_json_bytes(completion))
    engine, classifier = _engine(
        classifier=ScriptedClassifier(),
        checks=(_check("output-one", stage=GuardrailCheckStage.OUTPUT),),
        max_response_bytes=subject_bytes + limit_offset,
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None
    if limit_offset < 0:
        with pytest.raises(GuardrailRejected):
            _awaited(
                engine.enforce_output(
                    policy=policy,
                    completion=completion,
                    deadline_monotonic=200.0,
                )
            )
        assert classifier.output_calls == 0
    else:
        _awaited(
            engine.enforce_output(
                policy=policy,
                completion=completion,
                deadline_monotonic=200.0,
            )
        )
        assert classifier.output_calls == 1


def _regex_engine(
    *,
    action: GuardrailAction = GuardrailAction.MODIFY,
    extra: tuple[GuardrailCheck, ...] = (),
    max_response_bytes: int = 1_048_576,
) -> tuple[GuardrailEngine, GuardrailPolicy]:
    """Compose one engine whose output chain uses the deterministic detector.

    Args:
        action: Action of the deterministic output check.
        extra: Additional output checks appended to the chain.
        max_response_bytes: Completion bound for the identity.

    Returns:
        The engine and its resolved policy.
    """
    detector = RegexClassifier(
        RegexAdapterDocument(adapter_id="scripted", builtin_patterns=(BuiltinPattern.EMAIL,))
    )
    engine, _ = _engine(
        classifier=detector,
        checks=(
            _check("output-one", stage=GuardrailCheckStage.OUTPUT, action=action),
            *extra,
        ),
        max_response_bytes=max_response_bytes,
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None
    return engine, policy


def test_deterministic_modify_chain_streams() -> None:
    """One deterministic modify check on a plain stream releases incrementally."""
    engine, policy = _regex_engine()
    assert (
        engine.output_mode(
            policy,
            streaming=True,
            tools_offered=False,
            reasoning_text_requested=False,
        )
        is OutputGuardrailMode.STREAM
    )


def test_an_authored_regex_chain_keeps_the_buffered_path() -> None:
    """An expression of unknown span offers no redactor, so the stream buffers."""
    detector = RegexClassifier(
        RegexAdapterDocument(adapter_id="scripted", patterns=(r"SECRET-[0-9]+",))
    )
    engine, _ = _engine(
        classifier=detector,
        checks=(_check("output-one", stage=GuardrailCheckStage.OUTPUT),),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert (
        engine.output_mode(
            policy,
            streaming=True,
            tools_offered=False,
            reasoning_text_requested=False,
        )
        is OutputGuardrailMode.BUFFER
    )


@pytest.mark.parametrize(
    ("streaming", "tools_offered", "reasoning_text_requested"),
    [(False, False, False), (True, True, False), (True, False, True)],
)
def test_request_shape_keeps_the_buffered_path(
    streaming: bool,
    tools_offered: bool,
    reasoning_text_requested: bool,
) -> None:
    """Unary, tool-offering, and reasoning-text requests stay buffered."""
    engine, policy = _regex_engine()
    assert (
        engine.output_mode(
            policy,
            streaming=streaming,
            tools_offered=tools_offered,
            reasoning_text_requested=reasoning_text_requested,
        )
        is OutputGuardrailMode.BUFFER
    )


def test_blocking_check_keeps_the_buffered_path() -> None:
    """A check that can block cannot run after bytes are already out."""
    engine, policy = _regex_engine(action=GuardrailAction.BLOCK)
    assert (
        engine.output_mode(
            policy,
            streaming=True,
            tools_offered=False,
            reasoning_text_requested=False,
        )
        is OutputGuardrailMode.BUFFER
    )


def test_nondeterministic_adapter_keeps_the_buffered_path() -> None:
    """An adapter with no streaming redactor needs the whole completion."""
    engine, _ = _engine(
        classifier=ScriptedClassifier(),
        checks=(
            _check(
                "output-one",
                stage=GuardrailCheckStage.OUTPUT,
                action=GuardrailAction.MODIFY,
            ),
        ),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None
    assert (
        engine.output_mode(
            policy,
            streaming=True,
            tools_offered=False,
            reasoning_text_requested=False,
        )
        is OutputGuardrailMode.BUFFER
    )


def test_second_output_check_keeps_the_buffered_path() -> None:
    """A chain cannot compose redactions over partially released text."""
    engine, policy = _regex_engine(
        extra=(
            _check(
                "output-two",
                stage=GuardrailCheckStage.OUTPUT,
                action=GuardrailAction.MODIFY,
            ),
        )
    )
    assert (
        engine.output_mode(
            policy,
            streaming=True,
            tools_offered=False,
            reasoning_text_requested=False,
        )
        is OutputGuardrailMode.BUFFER
    )


def test_unguarded_identity_reports_no_output_work() -> None:
    """An identity with no output check runs neither path."""
    engine, _ = _engine(classifier=ScriptedClassifier(), checks=(_check("input-one"),))
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert (
        engine.output_mode(
            policy,
            streaming=True,
            tools_offered=False,
            reasoning_text_requested=False,
        )
        is OutputGuardrailMode.OFF
    )
    assert (
        engine.output_mode(
            None,
            streaming=True,
            tools_offered=False,
            reasoning_text_requested=False,
        )
        is OutputGuardrailMode.OFF
    )


def test_released_segment_is_redacted_and_bounded() -> None:
    """The released prefix is redacted and the unsettled tail stays buffered."""
    engine, policy = _regex_engine()
    segment = engine.release_output_segment(
        policy=policy,
        pending="write to ada@example.com " + "x" * 600 + " done",
        final=False,
        settled_bytes=0,
        deadline_monotonic=1e12,
    )
    assert "ada@example.com" not in segment.release
    assert segment.flagged
    assert segment.pending


def test_oversized_stream_fails_closed_and_releases_nothing() -> None:
    """A completion past the identity bound is terminal, as it is when buffered."""
    engine, policy = _regex_engine(max_response_bytes=16)
    with pytest.raises(GuardrailRejected) as failure:
        engine.release_output_segment(
            policy=policy,
            pending="x" * 64,
            final=True,
            settled_bytes=0,
            deadline_monotonic=1e12,
        )
    assert failure.value.failure.safe_details.get("action") == GuardrailAction.ERROR.value


class _FailingDetector(RegexClassifier):
    """Deterministic adapter whose redactor faults instead of deciding."""

    def release_boundary(self, text: str) -> int:
        """Fail instead of naming a boundary."""
        del text
        raise RuntimeError("detector unavailable")

    def redact(self, text: str) -> tuple[bool, str]:
        """Fail instead of redacting."""
        del text
        raise RuntimeError("detector unavailable")


def test_adapter_failure_mid_stream_fails_closed() -> None:
    """An adapter error releases nothing, for protected and open identities alike."""
    detector = _FailingDetector(
        RegexAdapterDocument(adapter_id="scripted", builtin_patterns=(BuiltinPattern.EMAIL,))
    )
    engine, _ = _engine(
        classifier=detector,
        checks=(_check("output-one", stage=GuardrailCheckStage.OUTPUT),),
    )
    policy = engine.policies_for("organization-one", "identity-one")[0]
    assert policy is not None
    with pytest.raises(GuardrailRejected):
        engine.release_output_segment(
            policy=policy,
            pending="anything",
            final=True,
            settled_bytes=0,
            deadline_monotonic=1e12,
        )


@pytest.mark.parametrize("deadline", [0.0, 1e12])
def test_stream_segment_never_releases_after_its_budget(deadline: float) -> None:
    """Expired admission and an over-budget scan both release no text."""
    engine, policy = _regex_engine()
    ticks = iter((0.0, 10.0))

    def clock() -> float:
        """Advance past the authored check timeout during inspection."""
        return next(ticks)

    engine._monotonic = clock
    with pytest.raises(GuardrailRejected):
        engine.release_output_segment(
            policy=policy,
            pending="ada@example.com ",
            final=True,
            settled_bytes=0,
            deadline_monotonic=deadline,
        )

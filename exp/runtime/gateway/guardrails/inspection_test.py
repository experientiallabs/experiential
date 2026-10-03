"""Mandatory classifiers share engine execution, deadlines, isolation, and failure semantics."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.bounded import BoundedInspect
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailPolicy,
    GuardrailRejected,
    MandatoryGuardrailPolicy,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.inspection import (
    GuardrailInspection,
    open_inspection,
    output_decision,
    validate_guardrail_engine,
)
from exp.runtime.gateway.guardrails.native_test import _authorization
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.streaming import GuardrailFragment, GuardrailOutput
from exp.runtime.gateway.tests.mandatory_guardrails_test import _Guard, _Session


class _ErrorSession(_Session):
    """An adapter whose private exception must never become a content verdict."""

    async def inspect_output(self, output: GuardrailOutput) -> ClassifierVerdict:
        """Raise content-bearing diagnostic text on the existing isolated executor."""
        raise ValueError("private prompt")


class _ErrorGuard(_Guard):
    """Compose an erroneous incremental adapter through the shared engine."""

    async def open_output_session(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> _ErrorSession:
        """Return a request-owned failing classifier."""
        return _ErrorSession()


def _session(engine: GuardrailEngine) -> GuardrailInspection:
    """Admit synthetic input through the real mandatory policy executor."""
    value = open_inspection(
        engine,
        authorization=_authorization(),
        request=GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="allowed input"),),
        ),
        deadline_monotonic=time.monotonic() + 10,
    )
    assert value is not None
    return value


def test_infrastructure_errors_do_not_become_violations_or_echo_content() -> None:
    """A failed model check is unavailable, and uses the engine's classifier accounting."""
    engine = _ErrorGuard()
    session = _session(engine)
    payload = output_decision(
        session,
        GuardrailOutput(request_id="req", fragments=(), final=True),
        deadline_monotonic=time.monotonic() + 10,
    )
    assert "private prompt" not in payload
    assert json.loads(payload)["failure"]["failure_class"] == "unavailable"
    assert engine.input_invocations == 1
    assert engine.output_invocations == 1
    assert engine.classifier_calls == 2


def test_flagged_verdict_uses_the_existing_policy_action() -> None:
    """The same classifier verdict and BLOCK action govern complete and incremental checks."""
    engine = _Guard()
    output = GuardrailOutput(
        request_id="req",
        fragments=(GuardrailFragment(kind="text", channel="text", text="withhold-marker"),),
        final=True,
    )
    payload = output_decision(_session(engine), output, deadline_monotonic=time.monotonic() + 10)
    assert json.loads(payload)["failure"]["failure_class"] == "guardrail"
    assert engine.classifier_calls == 2


def test_unknown_fragment_kind_is_not_silently_uninspected() -> None:
    """A new content channel requires an explicit contract update."""
    with pytest.raises(ValidationError):
        GuardrailOutput.model_validate(
            {
                "request_id": "req",
                "final": True,
                "fragments": [{"kind": "image", "channel": "0", "text": "x"}],
            }
        )


def test_expired_session_cannot_authorize_a_late_segment() -> None:
    """An expired request never releases pending content or calls the classifier."""
    engine = _Guard()
    session = _session(engine)
    result = json.loads(
        output_decision(
            session,
            GuardrailOutput(request_id="req", fragments=(), final=True),
            deadline_monotonic=0,
        )
    )
    assert result["failure"]["failure_class"] == "unavailable"
    assert engine.classifier_calls == 1


@pytest.mark.parametrize("marker", [None, True, "1", 0, 2])
def test_older_or_unknown_native_contract_cannot_bypass_output(marker: object) -> None:
    """Mandatory composition fails before a native package can ignore output inspection."""
    with (
        patch(
            "exp.runtime.gateway.guardrails.inspection.importlib.import_module",
            return_value=SimpleNamespace(GUARDRAIL_INSPECTION_CONTRACT_VERSION=marker),
        ),
        pytest.raises(ValueError, match="GUARDRAIL_INSPECTION_CONTRACT_VERSION=1"),
    ):
        validate_guardrail_engine(_Guard())


def test_existing_complete_only_adapter_uses_the_same_engine() -> None:
    """An ordinary registered classifier needs no incremental API to enforce output."""
    adapter = ScriptedClassifier(output_verdict=ClassifierVerdict(flagged=True))
    config = _Guard().mandatory_policy
    assert config is not None
    engine = GuardrailEngine(
        store=MappingGuardrailStore(()),
        client=DirectClassifierClient(ClassifierRegistry({"mandatory": adapter})),
        monotonic=time.monotonic,
        mandatory_policy=config,
    )
    session = _session(engine)
    assert session.buffers_output
    decision = json.loads(
        output_decision(
            session,
            GuardrailOutput(
                request_id="req",
                fragments=(GuardrailFragment(kind="text", channel="text", text="subject"),),
                final=True,
            ),
            deadline_monotonic=time.monotonic() + 10,
        )
    )
    assert decision["failure"]["failure_class"] == "guardrail"
    assert adapter.input_calls == adapter.output_calls == 1
    assert engine.classifier_calls == 2


@pytest.mark.parametrize("incremental", [False, True])
def test_request_wide_limit_survives_segment_release(incremental: bool) -> None:
    """Releasing small segments does not reset the request's coverage budget."""
    engine = _Guard(incremental=incremental, max_response_bytes=20)
    session = _session(engine)
    output = GuardrailOutput(
        request_id="req",
        fragments=(GuardrailFragment(kind="text", channel="t", text="a" * 10),),
        final=False,
    )
    assert (
        json.loads(output_decision(session, output, deadline_monotonic=time.monotonic() + 10))[
            "action"
        ]
        == "allow"
    )
    result = json.loads(output_decision(session, output, deadline_monotonic=time.monotonic() + 10))
    assert result["failure"]["failure_class"] == "unsupported_capability"


def test_output_context_is_request_owned_and_spans_channels() -> None:
    """A split marker is caught across channels without contaminating another request."""
    engine = _Guard()
    first, second = _session(engine), _session(engine)

    def emit(session: GuardrailInspection, kind: str, text: str) -> str:
        """Exercise the validated boundary with a single fragment."""
        return output_decision(
            session,
            GuardrailOutput.model_validate(
                {
                    "request_id": "req",
                    "fragments": [{"kind": kind, "channel": kind, "text": text}],
                    "final": False,
                }
            ),
            deadline_monotonic=time.monotonic() + 10,
        )

    assert json.loads(emit(first, "reasoning", "withhold-"))["action"] == "allow"
    assert json.loads(emit(second, "text", "marker"))["action"] == "allow"
    assert json.loads(emit(first, "text", "marker"))["failure"]["failure_class"] == "guardrail"
    assert engine.sessions[1].text == "marker"


def test_output_coverage_does_not_reset_after_tool_search() -> None:
    """A new context-bound adapter session preserves the same request-wide byte budget."""
    engine = _Guard(max_response_bytes=20)
    first, continued = _session(engine), _session(engine)
    output = GuardrailOutput(
        request_id="req",
        fragments=(GuardrailFragment(kind="text", channel="t", text="a" * 10),),
        final=False,
    )
    assert (
        json.loads(output_decision(first, output, deadline_monotonic=time.monotonic() + 10))[
            "action"
        ]
        == "allow"
    )
    continued.continue_request(first)
    result = json.loads(
        output_decision(continued, output, deadline_monotonic=time.monotonic() + 10)
    )
    assert result["failure"]["failure_class"] == "unsupported_capability"


class _BlockingSession(_Session):
    """Keep a real isolation slot occupied until the test releases it."""

    def __init__(self) -> None:
        """Create explicit start and release signals for the occupied worker."""
        super().__init__()
        self.started = threading.Event()
        self.finish = threading.Event()

    async def inspect_output(self, output: GuardrailOutput) -> ClassifierVerdict:
        """Block until released even if the caller's deadline has already expired."""
        self.started.set()
        assert self.finish.wait(5)
        return ClassifierVerdict(flagged=False)


def test_incremental_checks_share_customer_concurrency_and_timeouts() -> None:
    """A mandatory output check competes for the same slot as customer input checks."""
    asyncio.run(_exercise_shared_executor())


async def _exercise_shared_executor() -> None:
    """A blocked mandatory adapter cannot create extra workers or stall request deadlines."""
    executor = BoundedInspect(max_inflight=1)
    customer = ScriptedClassifier()
    engine = _Guard(inspects=executor, timeout_ms=200, adapters={"customer": customer})
    session = await asyncio.to_thread(_session, engine)
    blocker = _BlockingSession()
    session.sessions = ((session.sessions[0][0], blocker),)
    check = GuardrailCheck(
        check_id="customer-input",
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.BLOCK,
        timeout_ms=30,
        adapter_id="customer",
    )
    policy = GuardrailPolicy(
        policy_id="customer",
        organization_id="organization-one",
        identity_id="identity-one",
        protected=False,
        checks=(check,),
    )
    pending = asyncio.create_task(
        asyncio.to_thread(
            output_decision,
            session,
            GuardrailOutput(request_id="req", fragments=(), final=True),
            deadline_monotonic=time.monotonic() + 5,
        )
    )
    try:
        assert await asyncio.to_thread(blocker.started.wait, 2)
        request = engine.requests[0]
        result = await engine.enforce_input(
            policy=policy, request=request, deadline_monotonic=time.monotonic() + 5
        )
        assert result == request
        assert customer.input_calls == 0
        decision = json.loads(await asyncio.wait_for(pending, 2))
        assert decision["failure"]["failure_class"] == "unavailable"
        assert executor.isolation_worker_count() == 1
        assert executor.quarantined_adapter_ids() == frozenset({"mandatory"})
    finally:
        blocker.finish.set()
        await pending


class _SlowFactoryGuard(_Guard):
    """An incremental capability whose session factory exceeds its check budget."""

    async def open_output_session(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> _Session:
        """Delay session creation beyond the configured per-check budget."""
        await asyncio.sleep(1)
        return _Session()


def test_session_factory_is_bounded_and_fails_closed() -> None:
    """Session creation cannot bypass the existing timeout and fail-closed behavior."""
    engine = _SlowFactoryGuard(timeout_ms=30)
    started = time.monotonic()
    with pytest.raises(GuardrailRejected) as rejected:
        _session(engine)
    assert rejected.value.failure.failure_class.value == "unavailable"
    assert time.monotonic() - started < 0.5


@pytest.mark.parametrize("change", ["revision", "max_response_bytes", "checks"])
def test_replay_digest_covers_the_entire_mandatory_configuration(change: str) -> None:
    """Operator edits invalidate keyed replay even when the policy identity is unchanged."""
    original = _Guard()
    assert original.mandatory_policy is not None
    payload = original.mandatory_policy.model_dump(mode="json")
    if change == "checks":
        payload["checks"][0]["timeout_ms"] += 1
    elif change == "revision":
        payload[change] = "replacement-detector"
    else:
        payload[change] = 1000
    revised = GuardrailEngine(
        store=MappingGuardrailStore(()),
        client=DirectClassifierClient(ClassifierRegistry({"mandatory": original})),
        monotonic=time.monotonic,
        mandatory_policy=MandatoryGuardrailPolicy.model_validate(payload),
    )
    assert original.inspection_revision != revised.inspection_revision

"""One operation ceiling: reserve before intent, settle on response, keep unknown outcomes.

Every provider here is a scripted transport, so no credential or network call is made.
"""

import json
import multiprocessing
import os
import queue
import sqlite3
from datetime import UTC, datetime, timedelta
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.judging.lm import judge_response_schema
from exp.common.models import ModelCapabilities
from exp.common.project import ProjectConfig, ProjectStore
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.backpressure import (
    AdmissionLimits,
    HostProviderAdmission,
    ProviderRoute,
)
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.openai_compatible_test import _snapshot
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    ScriptedJsonTransport,
)
from exp.runtime.models.spend_ceiling import (
    OPERATION_DOLLAR_AUTHORIZATION,
    OPERATION_UNPRICED_AUTHORIZATION,
    AliasTariff,
    OperationSpendCeiling,
    SpendCeilingReached,
    UnpricedDispatchRefused,
    current_spend_ceiling,
    install_spend_ceiling,
    observed_token_usage,
    operation_spend_scope,
    request_token_bounds,
)
from exp.runtime.models.structured_completion import CompletionPlan, StructuredCompletion
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError, DispatchLedger

URL = "https://fixture.invalid/v1/chat/completions"
# One dollar per million tokens either way: a request's reservation in micro-dollars equals
# its byte-bounded input tokens plus its output allowance.
TARIFF = AliasTariff(input_usd_per_million=1.0, output_usd_per_million=1.0)
TARIFFS = {"actor": TARIFF, "judge": TARIFF}


def _payload(output_tokens: int = 1000) -> JsonObject:
    """Return a small chat body with an explicit output allowance."""
    return {
        "model": "fixture",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": output_tokens,
    }


def _worst_case(payload: JsonObject) -> float:
    """Return the fixture tariff's reservation for ``payload`` on the chat route."""
    bounds = request_token_bounds(URL, payload, TARIFF)
    assert bounds is not None
    cost = TARIFF.cost_usd(*bounds)
    assert cost is not None
    return cost


def _ok(input_tokens: int = 10, output_tokens: int = 5) -> JsonHttpResponse:
    """Return a successful chat response reporting its usage."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        },
    )


def _ceiling(tmp_path: Path, usd: float, *, unpriced: int = 0) -> OperationSpendCeiling:
    """Open one operation ledger with a single authorized stage."""
    ceiling = OperationSpendCeiling(
        tmp_path / "spend" / "ledger.sqlite", tariffs=TARIFFS, sleep=lambda _: None
    )
    ceiling.authorize("stage", ceiling_usd=usd, unpriced_attempts=unpriced)
    return ceiling


def _transport(
    tmp_path: Path,
    inner: ScriptedJsonTransport,
    ceiling: OperationSpendCeiling,
    *,
    alias: str | None = "actor",
    cap: int = 10,
) -> tuple[AdmittedJsonTransport, ProjectStore, DispatchLedger]:
    """Build an admitted transport bound to one alias route and the operation ceiling."""
    project = ProjectStore(tmp_path, "spend")
    if not project.paths.project_directory.exists():
        project.initialize(ProjectConfig(project_id="spend"))
    ledger = DispatchLedger(tmp_path / f"physical-{alias}.sqlite", cap, "a" * 64)
    transport = AdmittedJsonTransport(
        inner,
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision="fixture",
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        admission=HostProviderAdmission(tmp_path / "admission.sqlite", AdmissionLimits()),
        spend_ceiling=ceiling,
    )
    if alias is not None:
        transport = transport.for_route(ProviderRoute(provider="openai", alias=alias))
    return transport, project, ledger


def test_ceiling_refuses_before_any_ledger_claim_or_intent(tmp_path: Path) -> None:
    """The second reservation would exceed the ceiling, so nothing is claimed, written or sent."""
    cost = _worst_case(_payload())
    ceiling = _ceiling(tmp_path, cost * 1.5)
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={}), _ok()])
    transport, project, ledger = _transport(tmp_path, inner, ceiling)
    transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    artifacts = set(project.artifacts.list_ids())
    with pytest.raises(SpendCeilingReached) as stop:
        transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    assert stop.value.required_usd == pytest.approx(2 * cost)
    assert len(inner.requests) == 1 and ledger.used() == 1
    assert set(project.artifacts.list_ids()) == artifacts
    summary = ceiling.summary()
    # A response without usage keeps its full reservation as its settled charge.
    assert summary.settled_usd == pytest.approx(cost) and summary.in_flight_usd == 0


def test_settlement_frees_the_reserved_difference_for_the_next_request(tmp_path: Path) -> None:
    """Observed usage settles far below the worst case, so a second request fits."""
    cost = _worst_case(_payload())
    ceiling = _ceiling(tmp_path, cost * 1.5)
    inner = ScriptedJsonTransport([_ok(10, 5), _ok(10, 5)])
    transport, _, _ = _transport(tmp_path, inner, ceiling)
    transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    summary = ceiling.summary()
    assert len(inner.requests) == 2
    assert summary.settled_usd == pytest.approx(2 * 15 / 1_000_000)
    assert summary.roles[0].role == "actor" and summary.roles[0].attempts == 2


def test_unknown_outcome_keeps_its_full_reservation(tmp_path: Path) -> None:
    """A failed send is never refunded, so it still blocks a request it would have admitted."""
    cost = _worst_case(_payload())
    ceiling = _ceiling(tmp_path, cost * 1.5)
    inner = ScriptedJsonTransport([ProviderTransportError("connection reset"), _ok()])
    transport, _, _ = _transport(tmp_path, inner, ceiling)
    with pytest.raises(ProviderTransportError):
        transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    assert ceiling.summary().unknown_usd == pytest.approx(cost)
    with pytest.raises(SpendCeilingReached):
        transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    assert len(inner.requests) == 1


def test_unsent_requests_release_their_reservations(tmp_path: Path) -> None:
    """A refused ledger claim proves nothing left, so its reservation is freed at once."""
    cost = _worst_case(_payload())
    ceiling = _ceiling(tmp_path, cost * 3)
    inner = ScriptedJsonTransport([_ok()])
    transport, _, _ = _transport(tmp_path, inner, ceiling, cap=1)
    transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    with pytest.raises(DispatchCapExhaustedError):
        transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    summary = ceiling.summary()
    assert summary.released_attempts == 1 and summary.in_flight_usd == 0


def test_every_role_shares_the_one_operation_ceiling(tmp_path: Path) -> None:
    """Actor and judge transports draw from one ledger, not one ceiling each."""
    cost = _worst_case(_payload())
    ceiling = _ceiling(tmp_path, cost * 1.5)
    actor, _, _ = _transport(tmp_path, ScriptedJsonTransport([_ok(1000, 1000)]), ceiling)
    judge_inner = ScriptedJsonTransport([_ok()])
    judge, _, _ = _transport(tmp_path, judge_inner, ceiling, alias="judge")
    actor.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    with pytest.raises(SpendCeilingReached):
        judge.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    assert not judge_inner.requests


def test_unpriced_requests_are_refused_unless_explicitly_allowed(tmp_path: Path) -> None:
    """An unpriced alias or unbound route needs an explicit attempt ceiling and is reported."""
    refused = _ceiling(tmp_path / "refused", 1.0)
    inner = ScriptedJsonTransport([_ok(), _ok()])
    transport, project, ledger = _transport(tmp_path / "refused", inner, refused, alias="other")
    with pytest.raises(UnpricedDispatchRefused, match="no complete catalog prices"):
        transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    assert not inner.requests and ledger.used() == 0 and not project.artifacts.list_ids()
    unbound, _, _ = _transport(tmp_path / "refused", inner, refused, alias=None)
    with pytest.raises(UnpricedDispatchRefused, match="not bound to a catalog alias"):
        unbound.post(URL, headers={}, payload=_payload(), timeout_seconds=30)

    allowed = _ceiling(tmp_path / "allowed", 1.0, unpriced=1)
    transport, _, _ = _transport(tmp_path / "allowed", inner, allowed, alias="other")
    transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    with pytest.raises(UnpricedDispatchRefused):
        transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    summary = allowed.summary()
    assert len(inner.requests) == 1
    assert summary.unpriced_attempts == 1 and summary.unpriced_attempt_ceiling == 1
    assert summary.accounted_usd == 0 and summary.roles[0].unpriced_attempts == 1


def test_resumed_consent_replaces_its_stage_without_multiplying_the_ceiling(
    tmp_path: Path,
) -> None:
    """A resume re-authorizes its stage; another stage of the operation adds its own amount."""
    ceiling = _ceiling(tmp_path, 1.0)
    ceiling.authorize("stage", ceiling_usd=1.0)
    assert ceiling.summary().ceiling_usd == 1.0
    ceiling.authorize("second", ceiling_usd=0.5, unpriced_attempts=2)
    summary = ceiling.summary()
    assert summary.ceiling_usd == 1.5 and summary.unpriced_attempt_ceiling == 2
    assert [row.stage for row in summary.authorizations] == ["stage", "stage", "second"]


def test_reservation_requires_an_authorized_ceiling(tmp_path: Path) -> None:
    """An opened ledger authorizes nothing by itself."""
    ceiling = OperationSpendCeiling(tmp_path / "ledger.sqlite", tariffs=TARIFFS)
    with pytest.raises(ValueError, match="no spend ceiling is authorized"):
        ceiling.reserve(alias="actor", url=URL, payload=_payload())


def _reserve_and_exit(path: str, dispatch: bool) -> None:
    """Child process: reserve one request, optionally mark it sent, then die without cleanup."""
    ceiling = OperationSpendCeiling(Path(path), tariffs=TARIFFS)
    reservation = ceiling.reserve(alias="actor", url=URL, payload=_payload())
    if dispatch:
        ceiling.mark_dispatched(reservation, "attempt")
    os._exit(0)


def _run_child(path: Path, *, dispatch: bool) -> None:
    """Run ``_reserve_and_exit`` in a fresh interpreter and wait for it."""
    child = multiprocessing.get_context("spawn").Process(
        target=_reserve_and_exit, args=(str(path), dispatch)
    )
    child.start()
    child.join(60)
    assert child.exitcode == 0


def test_crash_between_reservation_and_dispatch_releases_on_resume(tmp_path: Path) -> None:
    """An unsent reservation of a dead process is released once, never lost or doubled."""
    cost = _worst_case(_payload())
    ceiling = _ceiling(tmp_path, cost * 1.5)
    _run_child(ceiling.path, dispatch=False)
    resumed = OperationSpendCeiling(ceiling.path, tariffs=TARIFFS)
    reservation = resumed.reserve(alias="actor", url=URL, payload=_payload())
    resumed.mark_dispatched(reservation, "attempt")
    resumed.settle(reservation, _ok(10, 5))
    summary = resumed.summary()
    assert summary.released_attempts == 1 and summary.in_flight_usd == 0
    assert summary.accounted_usd == pytest.approx(15 / 1_000_000)
    assert resumed.summary() == summary


def test_crash_after_dispatch_mark_keeps_the_full_reservation(tmp_path: Path) -> None:
    """A dead holder's sent request has an unknown outcome and stays charged on resume."""
    cost = _worst_case(_payload())
    ceiling = _ceiling(tmp_path, cost * 1.5)
    _run_child(ceiling.path, dispatch=True)
    resumed = OperationSpendCeiling(ceiling.path, tariffs=TARIFFS, sleep=lambda _: None)
    with pytest.raises(SpendCeilingReached):
        resumed.reserve(alias="actor", url=URL, payload=_payload())
    summary = resumed.summary()
    assert summary.unknown_usd == pytest.approx(cost) and summary.released_attempts == 0


def _spend_until_refused(path: str, results: "multiprocessing.Queue[int]") -> None:
    """Child process: reserve and settle at full cost until the shared ceiling refuses."""
    ceiling = OperationSpendCeiling(Path(path), tariffs=TARIFFS)
    sent = 0
    while True:
        try:
            reservation = ceiling.reserve(alias="actor", url=URL, payload=_payload())
        except SpendCeilingReached:
            break
        ceiling.mark_dispatched(reservation, f"attempt-{os.getpid()}-{sent}")
        ceiling.settle(reservation, JsonHttpResponse(status_code=200, body={}))
        sent += 1
    results.put(sent)


def test_two_processes_share_one_ceiling(tmp_path: Path) -> None:
    """Concurrent processes on one host admit exactly as many requests as one ceiling holds."""
    cost = _worst_case(_payload())
    ceiling = _ceiling(tmp_path, cost * 7.5)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    children = [
        context.Process(target=_spend_until_refused, args=(str(ceiling.path), results))
        for _ in range(2)
    ]
    for child in children:
        child.start()
    sent = [results.get(timeout=120) for _ in children]
    for child in children:
        child.join(60)
        assert child.exitcode == 0
    assert sum(sent) == 7
    summary = ceiling.summary()
    assert summary.settled_usd == pytest.approx(7 * cost)
    assert summary.accounted_usd <= summary.ceiling_usd


def test_request_bounds_cover_media_embeddings_choices_and_defaults() -> None:
    """Bounds never undercount: media takes the context window, choices multiply output."""
    windowed = AliasTariff(
        input_usd_per_million=1.0,
        output_usd_per_million=1.0,
        context_window_tokens=50,
        default_output_tokens=7,
    )
    text: JsonObject = {"messages": [{"role": "user", "content": "x" * 400}]}
    assert request_token_bounds(URL, text, windowed) == (50, 7)
    image: JsonObject = {"messages": [{"content": [{"type": "image_url", "image_url": {}}]}]}
    assert request_token_bounds(URL, image, windowed) == (50, 7)
    assert request_token_bounds(URL, image, TARIFF) is None
    assert request_token_bounds(URL, {"input": "x"}, TARIFF) is None
    embedding = request_token_bounds(
        "https://fixture.invalid/v1/embeddings", {"input": "x"}, TARIFF
    )
    assert embedding is not None and embedding[1] == 0
    embeddings = "https://fixture.invalid/v1/embeddings"
    batch: JsonObject = {"input": ["x" * 40] * 100}
    batch_size = len(json.dumps(batch).encode("utf-8"))
    assert request_token_bounds(embeddings, batch, windowed) == (batch_size, 0)
    assert request_token_bounds(embeddings, {"input": [{"type": "image"}]}, windowed) is None
    gemini: JsonObject = {
        "generationConfig": {
            "maxOutputTokens": 10,
            "candidateCount": 2,
            "thinkingConfig": {"thinkingBudget": 5},
        }
    }
    gemini_bounds = request_token_bounds(URL, gemini, TARIFF)
    assert gemini_bounds is not None and gemini_bounds[1] == 30
    decision = AliasTariff(input_usd_per_million=1.0, repeated_input=3, input_framing_tokens=4)
    body: JsonObject = {"input": "x", "max_tokens": 1}
    assert request_token_bounds(URL, body, decision) == (
        3 * (len(b'{"input": "x", "max_tokens": 1}') + 4),
        1,
    )


def test_observed_usage_counts_cache_and_reasoning_tokens() -> None:
    """Every provider family's usage includes cache and reasoning tokens in its totals."""
    assert observed_token_usage({"usage": {"prompt_tokens": 3, "completion_tokens": 4}}) == (3, 4)
    anthropic: JsonObject = {
        "usage": {
            "input_tokens": 1,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 3,
            "output_tokens": 4,
        }
    }
    assert observed_token_usage(anthropic) == (6, 4)
    gemini: JsonObject = {
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 6,
            "thoughtsTokenCount": 7,
        }
    }
    assert observed_token_usage(gemini) == (5, 13)
    assert observed_token_usage({"usage": {"prompt_tokens": -1}}) is None
    assert observed_token_usage({}) is None


def test_scope_installs_one_ambient_ceiling_for_transports(tmp_path: Path) -> None:
    """Transports built inside an operation scope after consent reserve against its ceiling."""
    ceiling = _ceiling(tmp_path, 1.0)
    with pytest.raises(ValueError, match="open operation scope"):
        install_spend_ceiling(ceiling)
    with operation_spend_scope():
        assert current_spend_ceiling() is None
        install_spend_ceiling(ceiling)
        project = ProjectStore(tmp_path, "spend")
        project.initialize(ProjectConfig(project_id="spend"))
        transport = AdmittedJsonTransport(
            ScriptedJsonTransport([]),
            ledger=DispatchLedger(tmp_path / "physical.sqlite", 1, "a" * 64),
            artifacts=project.artifacts,
            code_revision="fixture",
            deadline=datetime.now(UTC) + timedelta(minutes=1),
        )
        assert transport.spend_ceiling is ceiling
        other = OperationSpendCeiling(tmp_path / "other.sqlite", tariffs=TARIFFS)
        with pytest.raises(ValueError, match="exactly one spend ceiling"):
            install_spend_ceiling(other)
    assert current_spend_ceiling() is None


def test_priced_as_prices_an_unrouted_wire_without_changing_admission(tmp_path: Path) -> None:
    """The decision wire posts outside registry routes, so it names its alias explicitly."""
    ceiling = _ceiling(tmp_path, 1.0)
    inner = ScriptedJsonTransport([_ok(10, 5)])
    unbound, _, _ = _transport(tmp_path, inner, ceiling, alias=None)
    unbound.priced_as("judge").post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    summary = ceiling.summary()
    assert [row.role for row in summary.roles] == ["judge"]
    assert summary.settled_usd == pytest.approx(15 / 1_000_000)


@pytest.mark.parametrize(
    "body",
    [
        {"usage": {"prompt_tokens": 10}},
        {"usage": {"input_tokens": 10}},
        {"usageMetadata": {"promptTokenCount": 10}},
        {"usage": {"prompt_tokens": 10, "completion_tokens": None}},
        {"usage": {"prompt_tokens": None, "completion_tokens": 5}},
    ],
)
def test_incomplete_completion_usage_keeps_output_liability(
    tmp_path: Path, body: JsonObject
) -> None:
    """Missing billing counters cannot free reservations and admit another completion."""
    cost = _worst_case(_payload())
    ceiling = _ceiling(tmp_path, cost * 1.5)
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body=body), _ok()])
    transport, _, _ = _transport(tmp_path, inner, ceiling)
    transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    assert ceiling.summary().settled_usd == pytest.approx(cost)
    with pytest.raises(SpendCeilingReached):
        transport.post(URL, headers={}, payload=_payload(), timeout_seconds=30)
    assert len(inner.requests) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"usage": {"prompt_tokens": 10}},
        {"usage": {"input_tokens": 10}},
        {"usageMetadata": {"promptTokenCount": 10}},
    ],
)
def test_embedding_settles_input_usage_without_output_counters(
    tmp_path: Path, body: JsonObject
) -> None:
    """Only an embedding endpoint permits absent output counters to mean zero."""
    assert observed_token_usage(body) is None
    ceiling = _ceiling(tmp_path, 1.0)
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body=body)])
    transport, _, _ = _transport(tmp_path, inner, ceiling)
    transport.post(
        "https://fixture.invalid/v1/embeddings",
        headers={},
        payload={"input": "A long enough embedding input"},
        timeout_seconds=30,
    )
    assert ceiling.summary().settled_usd == pytest.approx(10 / 1_000_000)


def test_explicit_operation_limits_override_prior_stage_grants(tmp_path: Path) -> None:
    """An explicit policy replaces summed estimates and allowances without deleting history."""
    ceiling = _ceiling(tmp_path, 2.0, unpriced=5)
    ceiling.authorize("second", ceiling_usd=3.0, unpriced_attempts=7)
    assert ceiling.summary().ceiling_usd == 5.0
    assert ceiling.summary().unpriced_attempt_ceiling == 12
    history = ceiling.summary().authorizations
    ceiling.authorize(OPERATION_DOLLAR_AUTHORIZATION, ceiling_usd=1.0)
    ceiling.authorize(OPERATION_UNPRICED_AUTHORIZATION, ceiling_usd=0.0, unpriced_attempts=2)
    summary = ceiling.summary()
    assert summary.ceiling_usd == 1.0 and summary.unpriced_attempt_ceiling == 2
    assert summary.authorizations[: len(history)] == history
    with pytest.raises(SpendCeilingReached):
        ceiling.reserve(alias="actor", url=URL, payload=_payload(output_tokens=1_500_000))
    # Later stage estimates cannot increase either explicit operation limit.
    ceiling.authorize("third", ceiling_usd=10.0, unpriced_attempts=20)
    assert ceiling.summary().ceiling_usd == 1.0
    assert ceiling.summary().unpriced_attempt_ceiling == 2
    ceiling.authorize(OPERATION_DOLLAR_AUTHORIZATION, ceiling_usd=4.0)
    ceiling.authorize(OPERATION_DOLLAR_AUTHORIZATION, ceiling_usd=4.0)
    assert ceiling.summary().ceiling_usd == 4.0
    # Zero is an explicit refusal, not an absent policy or an additive allowance.
    ceiling.authorize(OPERATION_UNPRICED_AUTHORIZATION, ceiling_usd=0.0)
    with pytest.raises(UnpricedDispatchRefused):
        ceiling.reserve(alias="unpriced", url=URL, payload=_payload())


class _PausedPolicyCeiling(OperationSpendCeiling):
    """Pause a writer after the first policy row while its transaction remains open."""

    def __init__(self, path: Path, written: Event, release: Event) -> None:
        """Open the ledger and bind the interprocess transaction barrier."""
        super().__init__(path, tariffs={})
        self._written, self._release = written, release

    def _append_authorization(
        self, connection: sqlite3.Connection, stage: str, ceiling_usd: float, unpriced_attempts: int
    ) -> None:
        """Append the first row, then wait for the parent to release the transaction."""
        super()._append_authorization(connection, stage, ceiling_usd, unpriced_attempts)
        if stage == OPERATION_DOLLAR_AUTHORIZATION:
            self._written.set()
            if not self._release.wait(60):
                raise TimeoutError("policy transaction barrier timed out")


def _write_paused_policy(path: str, written: Event, release: Event) -> None:
    """Child process: replace both limits under one paused transaction."""
    _PausedPolicyCeiling(Path(path), written, release).authorize_policy(
        ceiling_usd=1.0, unpriced_attempts=0
    )


def _read_policy(
    path: str, reading: Event, results: "multiprocessing.Queue[tuple[float, int]]"
) -> None:
    """Child process: read the policy from a separate process and connection."""
    reading.set()
    summary = OperationSpendCeiling(Path(path), tariffs={}).summary()
    results.put((summary.ceiling_usd, summary.unpriced_attempt_ceiling))


def test_operation_policy_update_is_atomic_across_processes(tmp_path: Path) -> None:
    """A reader cannot observe the new dollar limit with the previous unpriced permission."""
    ceiling = _ceiling(tmp_path, 10.0)
    ceiling.authorize_policy(ceiling_usd=10.0, unpriced_attempts=5)
    context = multiprocessing.get_context("spawn")
    written, release, reading = context.Event(), context.Event(), context.Event()
    results = context.Queue()
    writer = context.Process(
        target=_write_paused_policy, args=(str(ceiling.path), written, release)
    )
    reader = context.Process(target=_read_policy, args=(str(ceiling.path), reading, results))
    writer.start()
    try:
        assert written.wait(60)
        reader.start()
        assert reading.wait(60)
        # The independent reader cannot finish while the two policy rows are uncommitted.
        with pytest.raises(queue.Empty):
            results.get(timeout=0.2)
    finally:
        release.set()
        writer.join(60)
        if reader.pid is not None:
            reader.join(60)
    assert writer.exitcode == reader.exitcode == 0
    assert results.get(timeout=60) == (1.0, 0)


def test_operation_policy_rolls_back_both_limits_on_second_row_failure(tmp_path: Path) -> None:
    """Failing the second append preserves the original policy and its exact evidence rows."""
    ceiling = _ceiling(tmp_path, 10.0)
    ceiling.authorize_policy(ceiling_usd=10.0, unpriced_attempts=5)
    before = ceiling.summary()
    # A real SQLite write failure occurs after the dollar insert, inside the same transaction.
    with sqlite3.connect(ceiling.path) as connection:
        connection.execute(
            "CREATE TRIGGER refuse_policy BEFORE INSERT ON authorizations "
            f"WHEN NEW.stage = '{OPERATION_UNPRICED_AUTHORIZATION}' "
            "BEGIN SELECT RAISE(ABORT, 'injected policy failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected policy failure"):
        ceiling.authorize_policy(ceiling_usd=1.0, unpriced_attempts=0)
    after = ceiling.summary()
    assert after.ceiling_usd == 10.0 and after.unpriced_attempt_ceiling == 5
    assert after.authorizations == before.authorizations


def test_context_generation_bound_covers_reasoning_beyond_request_and_blocks_next_send(
    tmp_path: Path,
) -> None:
    """Actual reasoning usage exceeding the requested limit settles honestly inside the bound."""
    tariff = AliasTariff(
        input_usd_per_million=1,
        output_usd_per_million=1,
        context_window_tokens=1000000,
        default_output_tokens=131072,
        output_uses_context_bound=True,
    )
    payload = _payload(16384)
    bounds = request_token_bounds(URL, payload, tariff)
    assert bounds is not None and bounds[1] == 1000000
    cost = tariff.cost_usd(*bounds)
    assert cost is not None
    ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs={"actor": tariff})
    ceiling.authorize("stage", ceiling_usd=cost, unpriced_attempts=0)
    response = _ok(12511, 17636)
    inner = ScriptedJsonTransport([response, _ok()])
    transport, _, _ = _transport(tmp_path, inner, ceiling)
    transport.post(URL, headers={}, payload=payload, timeout_seconds=30)
    assert ceiling.summary().settled_usd == pytest.approx((12511 + 17636) / 1000000)
    with pytest.raises(SpendCeilingReached):
        transport.post(URL, headers={}, payload=payload, timeout_seconds=30)
    assert len(inner.requests) == 1
    multiple = request_token_bounds(URL, {**payload, "n": 3}, tariff)
    assert multiple is not None and multiple[1] == 3000000
    assert multiple[0] > bounds[0]


def test_context_generation_bound_missing_window_is_unpriced_before_send(tmp_path: Path) -> None:
    """Neither a visible token limit nor a published output max substitutes for total capacity."""
    tariff = AliasTariff(
        input_usd_per_million=1,
        output_usd_per_million=1,
        default_output_tokens=131072,
        output_uses_context_bound=True,
    )
    assert request_token_bounds(URL, _payload(16384), tariff) is None
    ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs={"actor": tariff})
    ceiling.authorize("stage", ceiling_usd=100, unpriced_attempts=0)
    inner = ScriptedJsonTransport([_ok()])
    transport, _, _ = _transport(tmp_path, inner, ceiling)
    with pytest.raises(UnpricedDispatchRefused):
        transport.post(URL, headers={}, payload=_payload(16384), timeout_seconds=30)
    assert not inner.requests


def test_nullable_judge_schema_crosses_real_admission_and_settles_once(tmp_path: Path) -> None:
    """A standard judge's nullable schema is text, while real media remains separately bounded."""
    schema = judge_response_schema()
    verdict = {"dimensions": [{"dimension_id": "goal", "raw_score": 1, "rationale": None}]}
    inner = ScriptedJsonTransport(
        [
            JsonHttpResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "judge-result",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_proposal",
                                            "arguments": json.dumps(verdict),
                                        },
                                    }
                                ]
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                },
            )
        ]
    )
    ceiling = _ceiling(tmp_path, 1)
    admitted, project, physical = _transport(tmp_path, inner, ceiling)
    caps = ModelCapabilities(
        supports_tools=True, context_window_tokens=100000, maximum_output_tokens=1024
    )
    model = _snapshot().model_copy(update={"capabilities_sha256": caps.identity_sha256()})
    client = OpenAICompatibleClient(
        model=model, base_url="https://fixture.invalid/v1", api_key="fixture", transport=admitted
    )
    complete = StructuredCompletion(
        client=client,
        plan=CompletionPlan(
            model=model,
            capabilities=caps,
            deadline=datetime.now(UTC) + timedelta(minutes=1),
            output_tokens=1024,
            maximum_cost_per_call=1,
            code_revision="fixture",
            output_protocol="tool",
        ),
        budget=RequestBudget(project, identity="nullable-judge"),
        artifacts=project.artifacts,
    )
    body: JsonObject = {
        "messages": [{"role": "user", "content": "Assess the goal."}],
        "response_schema": schema,
    }
    assert complete(body, "one-judge-cell") == verdict
    assert complete(body, "one-judge-cell") == verdict
    assert len(inner.requests) == physical.used() == 1
    assert '["string", "null"]' in json.dumps(inner.requests[0].payload)
    assert ceiling.summary().settled_usd == pytest.approx(7 / 1000000)
    assert ceiling.summary().unknown_usd == 0

"""Pinned served identities and durable pairwise probe coordinates survive budget pauses."""

import asyncio
import json
from functools import partial
from pathlib import Path

import httpx
import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.judging import Rubric
from exp.common.judging.provenance import read_artifact_json
from exp.common.models import (
    AssistantAction,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    Usage,
    completion_cost_reservation,
)
from exp.common.models.pricing_test import _model
from exp.common.models.token_cost_test import prices
from exp.common.project import ProjectStore, artifact_input
from exp.common.project.request_budget import RequestBudgetStore
from exp.common.rollouts import RolloutArtifact
from exp.optimize.evaluation.spending import BudgetedCompletion
from exp.optimize.router.composition_test import _completion_reservation
from exp.optimize.router.judging.artifacts import write_production_rollout
from exp.optimize.router.judging.protocol import TemplateJudgeClient
from exp.optimize.router.judging.service import (
    commit_manual_judge_setup,
    prepare_manual_judge_calibration,
    prepare_manual_judge_setup,
)
from exp.optimize.router.judging.service_test import (
    _TIME,
    _built_store,
    _catalog,
    _template,
    _wide_axes,
)
from exp.runtime.models.budget import RequestBudget, SpendLimitReached
from exp.runtime.models.providers import async_transport, base
from exp.runtime.models.providers.async_transport import (
    HttpxAsyncJsonTransport,
    ProviderDeadlineExceeded,
)
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.transport import RetryPolicy, is_known_unbilled_failure
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.registry_test import _catalog as _runtime_catalog


class _Client:
    """Return a counted successful provider response with deterministic one-attempt usage."""

    def __init__(self, model: ModelSnapshot) -> None:
        """Bind the exact reported provider identity."""
        self.model = model
        self.calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return valid pairwise feedback, charging the full requested output for this fixture."""
        self.calls += 1
        return ModelResponse(
            model=self.model,
            output=AssistantAction(
                content='{"dimensions":[{"dimension_id":"task-success","winner":"tie"}]}'
            ),
            economics=OperationEconomics(
                usage=Usage(input_tokens=1, output_tokens=request.maximum_output_tokens or 1),
                provider_attempts=1,
            ),
        )


@pytest.mark.parametrize("drift", [None, "model_id", "provider", "connection_sha256"])
def test_budgeted_completion_accepts_only_the_configured_served_identity(
    tmp_path: Path, drift: str | None
) -> None:
    """An explicit served alias retains all other frozen identity pins and replays unchanged."""
    reservation = _completion_reservation("candidate")
    served = reservation.model.model_copy(update={"model_id": "served-name"})
    if drift is not None:
        served = served.model_copy(update={drift: "c" * 64 if "sha256" in drift else "other"})
    client = _Client(served)
    budget = RequestBudget(
        ProjectStore(tmp_path, "budget-test"), identity="served-pin", maximum_cost_usd=100
    )
    wrapper = BudgetedCompletion(
        client, budget, reservation, role="assistant", served_model_id="served-name"
    )
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="hello"),), maximum_output_tokens=1
    )
    if drift is not None:
        with budget.scope("cell"), pytest.raises(ValueError, match="identity"):
            wrapper.complete(request)
        return
    with budget.scope("cell"):
        first = wrapper.complete(request)
    store = RequestBudgetStore(ProjectStore(tmp_path, "budget-test"), "served-pin")
    receipt = store.read(sha256_json({"scope": "cell", "role": "assistant", "ordinal": 0}))
    assert receipt is not None
    assert "token_prices" not in reservation.model_dump(mode="json")
    assert receipt.fingerprint == sha256_json(
        {
            "request": request.model_dump(mode="json"),
            "reservation": reservation.model_dump(mode="json"),
            "served_model": served.model_dump(mode="json"),
            "response_contract": "priced-completion-v1",
        }
    )
    assert ModelResponse.model_validate(json.loads(store.response(receipt))["response"]) == first
    with budget.scope("cell"):
        assert wrapper.complete(request) == first
    assert first.model == served
    assert client.calls == 1


def test_old_flat_receipt_binding_is_rejected_without_redispatch(tmp_path: Path) -> None:
    """A stale raw-response contract stays untouched and cannot be silently re-executed."""
    project = ProjectStore(tmp_path, "stale-flat")
    reservation = _completion_reservation("candidate")
    client = _Client(reservation.model)
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="hello"),), maximum_output_tokens=1
    )
    budget = RequestBudget(project, identity="stale", maximum_cost_usd=100)
    fingerprint = sha256_json(
        {
            "request": request.model_dump(mode="json"),
            "reservation": reservation.model_dump(mode="json"),
            "served_model": reservation.model.model_dump(mode="json"),
        }
    )
    with budget.scope("cell"):
        budget.call(
            role="assistant",
            fingerprint=fingerprint,
            maximum_cost_usd=1,
            operation=lambda: client.complete(request),
            encode=lambda response: response.model_dump_json(),
            decode=ModelResponse.model_validate_json,
            charge=lambda _response: 0.001,
        )
    store = RequestBudgetStore(project, "stale")
    key = sha256_json({"scope": "cell", "role": "assistant", "ordinal": 0})
    before = store.read(key)
    assert before is not None
    raw = store.response(before)
    wrapper = BudgetedCompletion(client, budget, reservation, role="assistant")
    with budget.scope("cell"), pytest.raises(ValueError, match="changed"):
        wrapper.complete(request)
    assert client.calls == 1
    assert store.read(key) == before and store.response(before) == raw


def test_pairwise_budget_resume_preserves_reverse_probe_identity(tmp_path: Path) -> None:
    """A saved forward probe does not shift the reverse request's durable budget coordinate."""
    store = _built_store(tmp_path, paired=True)
    setup = commit_manual_judge_setup(
        store,
        prepare_manual_judge_setup(
            store,
            _catalog(),
            dimensions=_wide_axes(),
            prompt_template=_template("pairwise"),
            created_at=_TIME,
            code_revision="test-revision",
        ),
        confirmed=True,
    )
    plan = prepare_manual_judge_calibration(store, sample_size=1)
    reference = plan.reference_traces[0]
    assert reference is not None
    pointers = tuple(
        write_production_rollout(store, setup, plan.tasks[0], trace, _TIME, "test-revision")
        for trace in (plan.traces[0], reference)
    )
    rollouts = tuple(
        read_artifact_json(
            store,
            artifact_id=pointer.artifact_id,
            expected_artifact_type="rollout",
            relative_path="rollout.json",
            model_type=RolloutArtifact,
        )[0]
        for pointer in pointers
    )
    rubric, _ = read_artifact_json(
        store,
        artifact_id=setup.rubric.artifact_id,
        expected_artifact_type="rubric",
        relative_path="rubric.json",
        model_type=Rubric,
    )
    reservation = completion_cost_reservation(
        model=setup.judge_model,
        input_usd_per_million_tokens=0,
        output_usd_per_million_tokens=1,
        cached_input_usd_per_million_tokens=0,
        cache_write_usd_per_million_tokens=0,
        maximum_attempts=1,
        maximum_input_tokens=1_000_000,
        maximum_output_tokens=1_000,
    )
    client = _Client(setup.judge_model)
    request = ModelRequest(
        messages=(
            ModelMessage(
                role="system",
                content=setup.prompt_template.prompt.text,
            ),
        )
    )

    def execute(limit: float) -> ModelResponse:
        """Reconstruct both adapters as a fresh process would after increasing its allowance."""
        budget = RequestBudget(
            ProjectStore(tmp_path, "pairwise-budget"), identity="pairwise", maximum_cost_usd=limit
        )
        adapter = TemplateJudgeClient(
            BudgetedCompletion(client, budget, reservation, role="judge"),
            setup.prompt_template,
            rollouts[0],
            rubric,
            rollouts[1],
            store=store,
            setup_input=artifact_input(store.artifacts.read(setup.setup_id).manifest),
            rollout_input=pointers[0],
            reference_input=pointers[1],
            created_at=_TIME,
            code_revision="test-revision",
            maximum_output_tokens=1_000,
            request_scope=budget.scope,
        )
        return adapter.complete(request)

    with pytest.raises(SpendLimitReached):
        execute(0.001)
    assert client.calls == 1
    result = execute(0.002)
    assert client.calls == 2
    assert execute(0.002) == result
    assert client.calls == 2


@pytest.mark.parametrize("role", ["assistant", "world", "judge"])
def test_standard_catalog_admission_retries_charge_only_the_successful_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    """The ordinary catalog path handles six certified wire attempts and exact zero-call replay."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Refuse five requests at the authenticated origin before one successful completion."""
        requests.append(request)
        if len(requests) <= 5:
            return httpx.Response(
                429, json={}, headers={"Retry-After": "0", "x-gateway-admission-refused": "true"}
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    # Patch the reusable HTTP connection only. Catalog and client construction use the same
    # defaults as exp eval; no injected unbilled flag bypasses the public trust boundary.
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(async_transport, "_pooled_client", lambda: http_client)
    catalog = _runtime_catalog(
        provider="openai-compatible", base_url="https://api.experientiallabs.ai/v1"
    )
    resolved = RuntimeModelCatalog(catalog, environment={"FIXTURE_API_KEY": "fixture"}).resolve(
        "fixture-model"
    )
    assert isinstance(resolved.client, OpenAICompatibleClient)
    reservation = completion_cost_reservation(
        model=resolved.snapshot,
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=1,
        cache_write_usd_per_million_tokens=1,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )
    budget = RequestBudget(ProjectStore(tmp_path, "retry-test"), identity=role, maximum_cost_usd=1)
    wrapper = BudgetedCompletion(resolved.client, budget, reservation, role=role)
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="hello"),), maximum_output_tokens=500
    )
    with budget.scope("cell"):
        response = wrapper.complete(request)
    assert len(requests) == 6
    assert len({request.headers["Idempotency-Key"] for request in requests}) == 1
    assert response.economics.provider_attempts == 6
    assert response.economics.unbilled_attempts == 5
    assert budget.accounted_usd == pytest.approx(0.00014)
    with budget.scope("cell"):
        assert wrapper.complete(request) == response
    assert len(requests) == 6
    assert budget.accounted_usd == pytest.approx(0.00014)
    asyncio.run(http_client.aclose())


@pytest.mark.parametrize("mode", ["unpaid", "unknown_then_unpaid", "unpaid_then_inflight"])
def test_sync_completion_timeout_preserves_whole_request_billing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """The outer sync deadline cannot discard certified evidence when converting cancellation."""
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        """Exercise unpaid refusals, prior unknown outcomes, or cancelled active dispatch."""
        del request
        nonlocal attempts
        attempts += 1
        if mode == "unknown_then_unpaid" and attempts == 1:
            return httpx.Response(503, json={})
        if mode == "unpaid_then_inflight" and attempts == 2:
            await asyncio.Event().wait()
            raise AssertionError("active request returned")
        return httpx.Response(429, json={}, headers={"x-gateway-admission-refused": "true"})

    async def delayed_sleep(seconds: float) -> None:
        """Deterministically let the caller's earlier outer timeout interrupt unpaid backoff."""
        del seconds
        if mode != "unpaid" and attempts == 1:
            return
        await asyncio.Event().wait()

    monkeypatch.setattr(
        base,
        "run_with_retry_async",
        partial(async_transport.run_with_retry_async, sleep=delayed_sleep),
    )
    reservation = _completion_reservation("candidate")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(
        model=reservation.model,
        api_key="fixture",
        base_url="https://gateway.test/v1",
        transport=HttpxAsyncJsonTransport(
            http_client, trusted_admission_origin="https://gateway.test"
        ),
        timeout_seconds=0.05,
        retry_policy=RetryPolicy(initial_delay_seconds=0.001),
    )
    budget = RequestBudget(
        ProjectStore(tmp_path, "sync-timeout"), identity=mode, maximum_cost_usd=100
    )
    wrapper = BudgetedCompletion(client, budget, reservation, role="judge")
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="hello"),), maximum_output_tokens=1
    )
    with budget.scope("cell"), pytest.raises(ProviderDeadlineExceeded) as caught:
        wrapper.complete(request)
    assert attempts == (1 if mode == "unpaid" else 2)
    assert is_known_unbilled_failure(caught.value) is (mode == "unpaid")
    if mode == "unpaid":
        assert budget.accounted_usd == 0
    else:
        assert budget.accounted_usd > 0
    asyncio.run(http_client.aclose())


class _PaidClient:
    """Record a physical successful call whose reported reasoning has an unknown tariff."""

    def __init__(self) -> None:
        """Use a counter independent of persisted request state."""
        self.calls = 0
        self.response = ModelResponse(
            model=_model(),
            output=AssistantAction(content="paid response retained exactly"),
            economics=OperationEconomics(
                usage=Usage(input_tokens=10, output_tokens=10, reasoning_tokens=3),
                provider_attempts=1,
            ),
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return already generated output without claiming a price."""
        self.calls += 1
        return self.response


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("unknown_tariff", "not priceable"),
        ("missing_usage", "unknown usage"),
        ("input_overflow", "usage exceeds"),
        ("attempt_overflow", "attempts exceed"),
        ("retry_unknown_tariff", "earlier.*unbounded"),
    ],
)
def test_unknown_tariff_saves_paid_output_and_replays_error_under_lower_cap(
    tmp_path: Path, failure: str, message: str
) -> None:
    """Invalid or incomplete paid evidence is retained, never falsely settled or repeated."""
    project = ProjectStore(tmp_path, "scheduled")
    card = prices()
    if failure in ("unknown_tariff", "retry_unknown_tariff"):
        card = card.model_copy(update={"reasoning_nano_usd_per_million_tokens": None})
    reservation = completion_cost_reservation(
        model=_model(),
        token_prices=card,
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.1,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=100,
    )
    client = _PaidClient()
    if failure != "unknown_tariff":
        client.response = client.response.model_copy(
            update={
                "economics": OperationEconomics(
                    usage=(
                        None
                        if failure == "missing_usage"
                        else Usage(
                            input_tokens=1_001 if failure == "input_overflow" else 10,
                            output_tokens=10,
                            cached_input_tokens=0,
                            cache_write_input_tokens=0,
                            reasoning_tokens=0 if failure == "retry_unknown_tariff" else 3,
                        )
                    ),
                    provider_attempts=(
                        4
                        if failure == "attempt_overflow"
                        else 2
                        if failure == "retry_unknown_tariff"
                        else 1
                    ),
                )
            }
        )
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="hello"),), maximum_output_tokens=100
    )
    original = RequestBudget(project, identity="schedule", maximum_cost_usd=None)
    wrapper = BudgetedCompletion(client, original, reservation, role="assistant")
    with original.scope("cell"), pytest.raises(ValueError, match=message):
        wrapper.complete(request)
    store = RequestBudgetStore(project, "schedule")
    key = sha256_json({"scope": "cell", "role": "assistant", "ordinal": 0})
    receipt = store.read(key)
    assert receipt is not None and receipt.state == "unknown"
    assert not receipt.charge_is_upper_bound and store.has_unbounded_liability()
    raw = store.response(receipt)
    saved = json.loads(raw)
    assert ModelResponse.model_validate(saved["response"]) == client.response
    assert saved["charge_usd"] is None
    resumed = RequestBudget(project, identity="schedule", maximum_cost_usd=0.0000001)
    resumed_wrapper = BudgetedCompletion(client, resumed, reservation, role="assistant")
    with resumed.scope("cell"), pytest.raises(ValueError, match=message):
        resumed_wrapper.complete(request)
    with resumed.scope("new-cell"), pytest.raises(ValueError, match="complete applicable"):
        resumed_wrapper.complete(request)
    assert client.calls == 1
    assert store.read(key) == receipt and store.response(receipt) == raw

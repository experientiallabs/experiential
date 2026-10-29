"""Unusable judge output produces priced terminal evidence instead of paid replay loops."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from exp.common.judging import Judgment
from exp.common.models import (
    AssistantAction,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from exp.common.project import ProjectStore
from exp.optimize.evaluation.contracts import EvaluationBudget
from exp.optimize.evaluation.judge import DurableEvaluationJudge
from exp.optimize.evaluation.prepare import ModelEvaluationOptions, prepare_model_evaluation
from exp.optimize.evaluation.prepare_test import _prepare
from exp.optimize.evaluation.runtime import run_prepared_model_evaluation
from exp.optimize.router import judgment_budget
from exp.optimize.router.automatic.judge import AutomaticRouterJudge, ReservedJudgeClient
from exp.optimize.router.automatic.judge_test import _reserved_client, _UsageClient
from exp.optimize.router.automatic.service_test import _REVISION, _TIME, _RuntimeCatalog
from exp.optimize.router.errors import JudgeDispatchExhaustedError
from exp.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from exp.runtime.models.providers.async_transport import (
    ProviderDeadlineExceeded,
    RequestDeadline,
    run_with_retry_async,
)
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.transport import ProviderTransportError, RetryPolicy


def test_concurrent_judge_failures_charge_only_their_own_responses(tmp_path: Path) -> None:
    """A malformed judgment cannot charge a neighboring judgment's successful response."""
    client = _reserved_client(_UsageClient(), maximum_calls=2)
    barrier = threading.Barrier(2)

    class MalformedJudge(AutomaticRouterJudge):
        """Fail parsing only after both isolated provider responses have completed."""

        def __init__(self, reserved: ReservedJudgeClient) -> None:
            """Retain only the provider boundary exercised by this accounting fixture."""
            self._client = reserved

        def judge_persisted(
            self,
            store: ProjectStore,
            *,
            rollout_artifact_id: str,
            rubric_artifact_id: str,
            calibration_artifact_id: str,
        ) -> Judgment:
            """Produce distinct paid usage and then fail the output parser."""
            self._client.complete(
                ModelRequest(
                    messages=(ModelMessage(role="user", content=rollout_artifact_id),),
                    maximum_output_tokens=32,
                )
            )
            barrier.wait(timeout=5)
            raise ValueError("malformed judgment")

    judge = DurableEvaluationJudge(MalformedJudge(client), client, client._reservation)
    project = ProjectStore(tmp_path, "judge-accounting")

    def fail(tokens: int) -> float:
        """Collect this failed judgment's conservative charge from the durable wrapper."""
        with pytest.raises(JudgeDispatchExhaustedError) as raised:
            judge.judge_persisted(
                project,
                rollout_artifact_id=str(tokens),
                rubric_artifact_id="rubric-a",
                calibration_artifact_id="calibration-a",
            )
        return raised.value.conservative_cost_usd

    with ThreadPoolExecutor(max_workers=2) as pool:
        charged = tuple(pool.map(fail, (100, 200)))
    own_costs = {
        item.usage.input_tokens: item.cost_usd.value
        for item in client.economics
        if item.usage and item.cost_usd
    }
    assert charged == (own_costs[100], own_costs[200])


class _FailingClient:
    """Retain real provider-shaped economics while returning invalid structured output."""

    def __init__(self, delegate: ModelClient, failure: str) -> None:
        """Retain the instrumented deterministic provider client."""
        self._delegate = delegate
        self._failure = failure

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a paid response that cannot be parsed as a judgment."""
        if self._failure in {"unpaid", "mixed_unpaid"}:
            return _admission_deadline_failure(unknown_first=self._failure == "mixed_unpaid")
        if self._failure == "parameters":
            raise ProviderParameterError(
                message="temperature is incompatible with configured reasoning",
                param="temperature",
                code="invalid_parameter",
            )
        response = self._delegate.complete(request)
        if self._failure == "transport":
            raise ProviderTransportError("connection reset by provider")
        if self._failure == "deadline":
            raise ProviderDeadlineExceeded("provider deadline exceeded")
        if self._failure == "identity":
            return response.model_copy(
                update={"model": response.model.model_copy(update={"model_id": "wrong"})}
            )
        return response.model_copy(update={"output": AssistantAction(content="not JSON")})


def _admission_deadline_failure(*, unknown_first: bool) -> ModelResponse:
    """Produce authentic aggregate failure evidence through the production async retry loop."""
    attempts = 0
    now = 10.0

    async def attempt(timeout: float) -> ModelResponse:
        """Optionally lose a provider response, then receive only certified admission refusals."""
        del timeout
        nonlocal attempts
        attempts += 1
        if unknown_first and attempts == 1:
            raise ProviderTransportError("unknown response")
        raise ProviderTransportError(
            "admission busy", status_code=429, retry_after_seconds=1, known_unbilled=True
        )

    async def sleep(seconds: float) -> None:
        """Advance the test deadline without provider calls or wall-time waits."""
        nonlocal now
        now += seconds

    return asyncio.run(
        run_with_retry_async(
            attempt,
            policy=RetryPolicy(),
            deadline=RequestDeadline.after(0.75, now_monotonic=now),
            sleep=sleep,
            now_monotonic=lambda: now,
            random_sample=lambda: 0,
        )
    )


class _FailingCatalog(_RuntimeCatalog):
    """Change only judge provider output, leaving the complete runtime path intact."""

    failure: str = "malformed"

    def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
        """Wrap the selected judge provider with the malformed-output regression."""
        resolved = super().resolve(alias, role=role)
        return (
            replace(resolved, client=_FailingClient(resolved.client, self.failure))
            if alias == "judge"
            else resolved
        )


@pytest.mark.parametrize(
    "failure", ["malformed", "transport", "deadline", "identity", "unpaid", "mixed_unpaid"]
)
def test_judge_failure_has_durable_cost_and_never_redispatches(
    tmp_path: Path, failure: str
) -> None:
    """Each admitted unusable response is excluded once and retained in report-generation spend."""
    project, catalog, state, prepared = _prepare(tmp_path)
    failing = _FailingCatalog(catalog, state)
    failing.failure = failure
    runtime = cast(RuntimeModelCatalog, failing)
    budget = EvaluationBudget(
        maximum_cost_usd=prepared.cost.maximum_cost_usd,
        maximum_judgments=prepared.cost.judgment_count,
    )
    result = run_prepared_model_evaluation(
        project,
        prepared,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert result.report.compared_cells == 0
    assert result.report.excluded_cells == prepared.cost.scenario_count
    if failure == "unpaid":
        assert result.judge_cost_usd == 0
    else:
        assert result.judge_cost_usd > 0
    calls = len(state.completion_calls)
    replay = run_prepared_model_evaluation(
        project,
        prepared,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert replay == result
    assert len(state.completion_calls) == calls


def test_judge_parameter_error_stops_without_excluding_cells(tmp_path: Path) -> None:
    """A shared local request bug is actionable, rather than a failure for every candidate."""
    project, catalog, state, prepared = _prepare(tmp_path)
    failing = _FailingCatalog(catalog, state)
    failing.failure = "parameters"
    with pytest.raises(ValueError, match="judge request settings are invalid.*temperature"):
        run_prepared_model_evaluation(
            project,
            prepared,
            cast(RuntimeModelCatalog, failing),
            budget=EvaluationBudget(maximum_cost_usd=100, maximum_judgments=100),
            provider_spend_consented=True,
            created_at=_TIME,
            code_revision=_REVISION,
        )
    assert not any(alias == "judge" for alias, _ in state.completion_calls)
    assert not any(
        project.artifacts.read(key).manifest.artifact_type == "judgment-exclusion"
        for key in project.artifacts.list_ids()
    )


@pytest.mark.parametrize("failure", ["unpaid", "mixed_unpaid"])
def test_interrupted_unpaid_exclusion_resumes_without_dispatch_or_phantom_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A crash after the request receipt preserves unpaid proof and earlier paid judgments."""
    project, catalog, state, initial = _prepare(tmp_path)
    prepared = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        judge_setup=initial.judge_setup,
        calibration_id=initial.setup.simulation_protocol.judge_calibration_id,
        embedder_alias="embedder",
        options=ModelEvaluationOptions(maximum_steps=1, maximum_concurrency=1),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    failing = _FailingCatalog(catalog, state)
    failing.failure = failure
    runtime = cast(RuntimeModelCatalog, failing)
    budget = EvaluationBudget(
        maximum_cost_usd=prepared.cost.maximum_cost_usd,
        maximum_judgments=prepared.cost.judgment_count,
    )
    requests: list[str] = []
    original_complete = _FailingClient.complete

    def complete(client: _FailingClient, request: ModelRequest) -> ModelResponse:
        """Complete one paid judgment before interrupting the next request's exclusion."""
        requests.append(request.model_dump_json())
        if len(requests) == 1:
            return client._delegate.complete(request)
        return original_complete(client, request)

    monkeypatch.setattr(_FailingClient, "complete", complete)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(
            judgment_budget,
            "_record_judgment_exclusion",
            Mock(side_effect=KeyboardInterrupt("crash before exclusion persistence")),
        )
        with pytest.raises(KeyboardInterrupt, match="before exclusion persistence"):
            run_prepared_model_evaluation(
                project,
                prepared,
                runtime,
                budget=budget,
                provider_spend_consented=True,
                created_at=_TIME,
                code_revision=_REVISION,
            )
    assert len(requests) == 2
    failed_request = requests[-1]
    saved_judgments = {
        key: project.artifacts.read(key)
        for key in project.artifacts.list_ids()
        if project.artifacts.read(key).manifest.artifact_type == "judgment"
    }
    assert len(saved_judgments) == 1
    assert not any(
        project.artifacts.read(key).manifest.artifact_type == "judgment-exclusion"
        for key in project.artifacts.list_ids()
    )

    reopened = ProjectStore(project.paths.root, project.paths.project_id)
    result = run_prepared_model_evaluation(
        reopened,
        prepared,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert requests.count(failed_request) == 1
    assert all(reopened.artifacts.read(key) == value for key, value in saved_judgments.items())
    exclusions = [
        judgment_budget.JudgmentExclusionRecord.model_validate_json(
            reopened.artifacts.read_bytes(key, "exclusion.json")
        )
        for key in reopened.artifacts.list_ids()
        if reopened.artifacts.read(key).manifest.artifact_type == "judgment-exclusion"
    ]
    assert len(exclusions) == prepared.cost.judgment_count - 1
    costs = [item.conservative_cost_usd for item in exclusions]
    if failure == "unpaid":
        assert all(cost == 0 for cost in costs)
    else:
        assert all(cost > 0 for cost in costs)
    assert result.judge_cost_usd > 0  # The successful request still contributes its paid usage.
    calls = len(requests)
    assert (
        run_prepared_model_evaluation(
            reopened,
            prepared,
            runtime,
            budget=budget,
            provider_spend_consented=True,
            created_at=_TIME,
            code_revision=_REVISION,
        )
        == result
    )
    assert len(requests) == calls

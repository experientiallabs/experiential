"""Catalog-backed prepared evaluation through real simulator, LM judge and persisted reports."""

import json
import threading
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest

from exp.common.models import AssistantAction, ModelRequest, ModelResponse, Usage
from exp.common.models.catalog import GatewayDeploymentMetadata
from exp.common.models.catalog_prices import (
    GatewayLongContextTier,
    GatewayServiceTierPrices,
    GatewayTokenPrices,
)
from exp.common.progress import ProgressEvent
from exp.optimize.evaluation.contracts import EvaluationBudget
from exp.optimize.evaluation.prepare import ModelEvaluationOptions, prepare_model_evaluation
from exp.optimize.evaluation.prepare_test import _prepare
from exp.optimize.evaluation.runtime import run_prepared_model_evaluation
from exp.optimize.evaluation.spending import BudgetedCompletion
from exp.optimize.router.automatic.provisional import prepare_hosted_provisional_judge
from exp.optimize.router.automatic.service_test import (
    _REVISION,
    _TIME,
    _completed_project,
    _CompletionClient,
    _RuntimeCatalog,
)
from exp.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from exp.runtime.models.budget import SpendLimitReached
from exp.simulation.engines.text import simulator


class _ScheduledRuntimeCatalog(_RuntimeCatalog):
    """Keep real runtime schedule bindings around deterministic provider clients."""

    def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
        """Match the public resolver's immutable complete-price metadata handoff."""
        return replace(
            super().resolve(alias, role=role),
            token_prices=self._catalog_value.models[alias].token_prices,
        )


def test_finite_preflight_rejects_incomplete_judge_before_any_runtime_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A priceable simulation cannot run first when the later judge lacks a finite bound."""
    project, catalog, state = _completed_project(tmp_path)
    card = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=1_000_000_000,
        cached_input_nano_usd_per_million_tokens=500_000_000,
        cache_creation_input_nano_usd_per_million_tokens=1_500_000_000,
        output_nano_usd_per_million_tokens=2_000_000_000,
    )
    catalog = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "judge": catalog.models["judge"].model_copy(
                    update={"gateway": GatewayDeploymentMetadata(prices=card)}
                ),
            }
        }
    )
    judge = prepare_hosted_provisional_judge(
        project,
        catalog,
        maximum_input_tokens=32_768,
        maximum_output_tokens=8_192,
        maximum_attempts=3,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    prepared = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        judge_setup=judge.setup_input,
        calibration_id=judge.calibration_id,
        embedder_alias="embedder",
        options=ModelEvaluationOptions(maximum_steps=1),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert prepared.cost.workers.maximum_is_upper_bound
    assert not prepared.cost.judge.maximum_is_upper_bound
    before = (
        project.artifacts.list_ids(),
        len(state.completion_calls),
        len(state.embedding_calls),
        state.credential_resolutions,
    )
    with pytest.raises(ValueError, match="finite spending limit requires complete"):
        run_prepared_model_evaluation(
            project,
            prepared,
            cast(RuntimeModelCatalog, _ScheduledRuntimeCatalog(catalog, state)),
            budget=EvaluationBudget(maximum_cost_usd=100, maximum_judgments=100),
            provider_spend_consented=True,
            created_at=_TIME,
            code_revision=_REVISION,
        )
    assert before == (
        project.artifacts.list_ids(),
        len(state.completion_calls),
        len(state.embedding_calls),
        state.credential_resolutions,
    )
    original_complete = _CompletionClient.complete

    def complete(client: _CompletionClient, request: ModelRequest) -> ModelResponse:
        """Give uncapped execution actual zero subset meters, without filling unknown prices."""
        response = original_complete(client, request)
        assert response.economics.usage is not None
        return response.model_copy(
            update={
                "economics": response.economics.model_copy(
                    update={
                        "provider_attempts": 1,
                        "usage": response.economics.usage.model_copy(
                            update={
                                "cached_input_tokens": 0,
                                "cache_write_input_tokens": 0,
                                "reasoning_tokens": 0,
                            }
                        ),
                    }
                )
            }
        )

    monkeypatch.setattr(_CompletionClient, "complete", complete)
    runtime = cast(RuntimeModelCatalog, _ScheduledRuntimeCatalog(catalog, state))
    budget = EvaluationBudget(maximum_cost_usd=None, maximum_judgments=100)
    result = run_prepared_model_evaluation(
        project,
        prepared,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    calls = (len(state.completion_calls), len(state.embedding_calls))
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
    assert calls == (len(state.completion_calls), len(state.embedding_calls))


def test_full_schedule_preparation_execution_report_and_lower_cap_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual engine carries whole-request rates and disjoint usage through every role."""
    project, catalog, state = _completed_project(tmp_path)
    build = project.load_project().build
    assert build is not None
    original_world = project.artifacts.read_bytes(build.world_model.artifact_id, "world-model.json")
    card = GatewayTokenPrices(
        input_nano_usd_per_million_tokens=1_000_000_000,
        cached_input_nano_usd_per_million_tokens=500_000_000,
        cache_creation_input_nano_usd_per_million_tokens=1_500_000_000,
        cache_creation_1h_input_nano_usd_per_million_tokens=2_500_000_000,
        output_nano_usd_per_million_tokens=2_000_000_000,
        reasoning_nano_usd_per_million_tokens=3_000_000_000,
        flex=GatewayServiceTierPrices(input_nano_usd_per_million_tokens=90_000_000_000),
        priority=GatewayServiceTierPrices(output_nano_usd_per_million_tokens=90_000_000_000),
        long_context=GatewayLongContextTier(
            input_threshold_tokens=8,
            input_nano_usd_per_million_tokens=2_000_000_000,
            cached_input_nano_usd_per_million_tokens=1_000_000_000,
            cache_creation_input_nano_usd_per_million_tokens=3_000_000_000,
            cache_creation_1h_input_nano_usd_per_million_tokens=5_000_000_000,
            output_nano_usd_per_million_tokens=4_000_000_000,
            reasoning_nano_usd_per_million_tokens=6_000_000_000,
        ),
    )
    catalog = catalog.model_copy(
        update={
            "models": {
                alias: record.model_copy(update={"gateway": GatewayDeploymentMetadata(prices=card)})
                if alias != "embedder"
                else record
                for alias, record in catalog.models.items()
            }
        }
    )
    judge = prepare_hosted_provisional_judge(
        project,
        catalog,
        maximum_input_tokens=32_768,
        maximum_output_tokens=8_192,
        maximum_attempts=3,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    prepared = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        judge_setup=judge.setup_input,
        calibration_id=judge.calibration_id,
        embedder_alias="embedder",
        options=ModelEvaluationOptions(maximum_steps=1),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert prepared.judge_request.token_prices == card
    assert prepared.cost.maximum_is_upper_bound
    real_complete = _CompletionClient.complete

    def complete(client: _CompletionClient, request: ModelRequest) -> ModelResponse:
        """Supply one successful provider-shaped meter with all six token dimensions."""
        response = real_complete(client, request)
        return response.model_copy(
            update={
                "economics": response.economics.model_copy(
                    update={
                        "usage": Usage(
                            input_tokens=8,
                            output_tokens=4,
                            cached_input_tokens=2,
                            cache_write_input_tokens=2,
                            cache_write_1h_input_tokens=1,
                            reasoning_tokens=1,
                        ),
                        "provider_attempts": 1,
                    }
                )
            }
        )

    monkeypatch.setattr(_CompletionClient, "complete", complete)
    runtime = cast(RuntimeModelCatalog, _ScheduledRuntimeCatalog(catalog, state))
    result = run_prepared_model_evaluation(
        project,
        prepared,
        runtime,
        budget=EvaluationBudget(maximum_cost_usd=100, maximum_judgments=100),
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert result.report.compared_cells == 3
    assert all(row.operating_cost_usd == pytest.approx(0.000036) for row in result.report.models)
    assert {alias for alias, _ in state.completion_calls} == {
        "candidate-a",
        "candidate-b",
        "world",
        "judge",
    }
    calls = (len(state.completion_calls), len(state.embedding_calls))
    replay = run_prepared_model_evaluation(
        project,
        prepared,
        runtime,
        budget=EvaluationBudget(maximum_cost_usd=0.0000001, maximum_judgments=100),
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert replay == result
    assert calls == (len(state.completion_calls), len(state.embedding_calls))
    assert (
        project.artifacts.read_bytes(build.world_model.artifact_id, "world-model.json")
        == original_world
    )


@pytest.mark.parametrize("blank_worker", [False, True])
@pytest.mark.parametrize("uncapped", [False, True])
def test_prepared_evaluation_runs_real_lm_judge_and_replays_without_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blank_worker: bool,
    uncapped: bool,
) -> None:
    """Only provider transport is deterministic; all evaluation execution is production code."""
    project, catalog, state, prepared = _prepare(tmp_path)
    original_complete = _CompletionClient.complete
    progress: list[ProgressEvent] = []

    def complete(client: _CompletionClient, request: ModelRequest) -> ModelResponse:
        """Keep blank worker replies in the scored cohort with a scripted failing judgment."""
        assert progress[0].stage == "Verifying evaluation"
        assert any(event.stage == "Loading retrieval index" for event in progress)
        response = original_complete(client, request)
        if blank_worker and client._alias.startswith("candidate-"):
            return response.model_copy(update={"output": AssistantAction(content="")})
        if blank_worker and client._alias == "judge":
            judgment = {
                "dimensions": [
                    {
                        "dimension_id": "task-success",
                        "raw_score": 0,
                        "rationale": "The worker produced no answer.",
                    }
                ]
            }
            return response.model_copy(
                update={"output": AssistantAction(content=json.dumps(judgment))}
            )
        return response

    monkeypatch.setattr(_CompletionClient, "complete", complete)
    embeddings_before = len(state.embedding_calls)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    budget = EvaluationBudget(
        maximum_cost_usd=None if uncapped else prepared.cost.maximum_cost_usd,
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
        progress=progress.append,
    )
    stages = [event.stage for event in progress]
    assert stages.index("Checking evaluation estimate") < stages.index("Loading retrieval index")
    assert stages.index("Loading retrieval index") < stages.index("simulation")
    assert "judging" in stages
    cells = [event for event in progress if event.stage == "evaluation cells"]
    assert cells[0].completed == 0
    assert cells[-1].completed == cells[-1].total == 6
    assert result.report.compared_cells == prepared.cost.scenario_count
    assert all(row.quality == (0 if blank_worker else 1) for row in result.report.models)
    if blank_worker:
        assert len(state.embedding_calls) == embeddings_before
    assert {alias for alias, _ in state.completion_calls} == {
        "candidate-a",
        "candidate-b",
        "world",
        "judge",
    }
    before = (len(state.completion_calls), len(state.embedding_calls))
    replay = run_prepared_model_evaluation(
        project,
        prepared,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME + timedelta(hours=1),
        code_revision=_REVISION,
    )
    assert replay == result
    assert before == (len(state.completion_calls), len(state.embedding_calls))


def test_unapproved_execution_precedes_credentials_and_writes(tmp_path: Path) -> None:
    """No provider client or artifact is created before explicit spend consent."""
    project, catalog, state, prepared = _prepare(tmp_path)
    before = (project.artifacts.list_ids(), state.credential_resolutions)
    with pytest.raises(ValueError, match="consent"):
        run_prepared_model_evaluation(
            project,
            prepared,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
            budget=EvaluationBudget(maximum_cost_usd=1, maximum_judgments=100),
            provider_spend_consented=False,
            created_at=_TIME,
            code_revision=_REVISION,
        )
    assert before == (project.artifacts.list_ids(), state.credential_resolutions)


@pytest.mark.parametrize("pause_on_correction", [False, True])
def test_request_ledger_retries_without_scanning_rollouts_under_cell_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pause_on_correction: bool
) -> None:
    """Parallel admission uses its request ledger and only retries the invalid cell."""
    project, catalog, state, prepared = _prepare(tmp_path)
    original_complete = _CompletionClient.complete
    remaining_failure = [True]
    lock = threading.Lock()

    def complete(client: _CompletionClient, request: ModelRequest) -> ModelResponse:
        """Reject one world-model transition, then allow its independent retry to finish."""
        response = original_complete(client, request)
        with lock:
            if client._alias == "world" and remaining_failure[0]:
                remaining_failure[0] = False
                return response.model_copy(update={"output": AssistantAction(content="invalid")})
        return response

    def unexpected_rollout_scan(*args: object, **kwargs: object) -> float:
        """Fail if per-request budgeting rereads the entire corpus to schedule a cell."""
        pytest.fail("request-ledger admission must not reconcile every saved rollout")

    monkeypatch.setattr(_CompletionClient, "complete", complete)
    monkeypatch.setattr(simulator, "resolution_spend", unexpected_rollout_scan)
    original_budgeted = BudgetedCompletion.complete
    paused = [False]

    def budgeted_complete(client: BudgetedCompletion, request: ModelRequest) -> ModelResponse:
        """Pause before a corrected simulator dispatch after the first reply is durably saved."""
        correction = request.messages[-1].content or ""
        if (
            pause_on_correction
            and not paused[0]
            and correction.startswith("The previous simulator")
        ):
            paused[0] = True
            raise SpendLimitReached(100, 1, 101)
        return original_budgeted(client, request)

    monkeypatch.setattr(BudgetedCompletion, "complete", budgeted_complete)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    budget = EvaluationBudget(maximum_cost_usd=100, maximum_judgments=100)
    if pause_on_correction:
        with pytest.raises(SpendLimitReached):
            run_prepared_model_evaluation(
                project,
                prepared,
                runtime,
                budget=budget,
                provider_spend_consented=True,
                created_at=_TIME,
                code_revision=_REVISION,
            )
        assert paused[0]
    result = run_prepared_model_evaluation(
        project,
        prepared,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert not remaining_failure[0]
    assert result.report.compared_cells == prepared.cost.scenario_count
    assert all(row.quality == 1 for row in result.report.models)
    workers = [alias for alias, _ in state.completion_calls if alias.startswith("candidate-")]
    assert len(workers) == 6  # Simulator-only retries preserve every candidate response.
    assert sum(alias == "world" for alias, _ in state.completion_calls) == 7
    before = (len(state.completion_calls), len(state.embedding_calls))
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
    assert before == (len(state.completion_calls), len(state.embedding_calls))


@pytest.mark.parametrize("resumed_limit", [100, None])
def test_budget_pause_resumes_partial_turn_without_repeating_paid_calls(
    tmp_path: Path, resumed_limit: float | None
) -> None:
    """An allowance far below the theoretical bound pauses, then replays a paid prefix for free."""

    project, catalog, state, prepared = _prepare(tmp_path)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    with pytest.raises(SpendLimitReached):
        run_prepared_model_evaluation(
            project,
            prepared,
            runtime,
            budget=EvaluationBudget(maximum_cost_usd=0.2, maximum_judgments=100),
            provider_spend_consented=True,
            created_at=_TIME,
            code_revision=_REVISION,
        )
    workers = [alias for alias, _ in state.completion_calls if alias.startswith("candidate")]
    assert workers
    assert not any(
        project.artifacts.read(i).manifest.artifact_type == "model-evaluation-report"
        for i in project.artifacts.list_ids()
    )
    result = run_prepared_model_evaluation(
        project,
        prepared,
        runtime,
        budget=EvaluationBudget(maximum_cost_usd=resumed_limit, maximum_judgments=100),
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert result.report.compared_cells == prepared.cost.scenario_count
    assert sum(alias.startswith("candidate") for alias, _ in state.completion_calls) == 6


@pytest.mark.parametrize("drift", ["quote", "agent", "redaction", "worker"])
def test_runtime_refuses_changed_accepted_inputs_before_provider_dispatch(
    tmp_path: Path,
    drift: str,
) -> None:
    """A stored quote is not authority to execute changed selection or agent settings."""
    project, catalog, state, prepared = _prepare(tmp_path)
    if drift == "quote":
        prepared = prepared.model_copy(
            update={
                "cost": prepared.cost.model_copy(
                    update={"maximum_cost_usd": 0.0},
                )
            }
        )
    elif drift == "agent":
        prepared = prepared.model_copy(update={"agent_factory_sha256": "a" * 64})
    elif drift == "redaction":
        prepared = prepared.model_copy(update={"redacted_field_names": ("private-value",)})
    else:
        selected = catalog.models["candidate-b"]
        catalog = catalog.model_copy(
            update={
                "models": {
                    **catalog.models,
                    "candidate-b": selected.model_copy(update={"model": "other-model"}),
                }
            }
        )
    before = (project.artifacts.list_ids(), len(state.completion_calls), len(state.embedding_calls))
    with pytest.raises(ValueError, match="changed"):
        run_prepared_model_evaluation(
            project,
            prepared,
            cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
            budget=EvaluationBudget(maximum_cost_usd=1_000, maximum_judgments=100),
            provider_spend_consented=True,
            created_at=_TIME,
            code_revision=_REVISION,
        )
    assert before == (
        project.artifacts.list_ids(),
        len(state.completion_calls),
        len(state.embedding_calls),
    )


def test_prepared_workers_world_and_judge_accept_pinned_served_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider aliases remain usable through the request ledger and simulator recorders."""
    project, catalog, state, prepared = _prepare(tmp_path)
    original_complete = _CompletionClient.complete
    original_resolve = _RuntimeCatalog.resolve
    pinned = {"candidate-a", "world", "judge"}

    def complete(client: _CompletionClient, request: ModelRequest) -> ModelResponse:
        """Echo the configured served identity from worker, world and judge responses."""
        response = original_complete(client, request)
        if client._alias in pinned:
            return response.model_copy(
                update={
                    "model": response.model.model_copy(
                        update={"model_id": f"served-{client._alias}"}
                    )
                }
            )
        return response

    def resolve(
        runtime: _RuntimeCatalog, alias: str, *, role: CatalogRoleName | None = None
    ) -> ResolvedModel:
        """Expose the same explicit served identity that a configured runtime catalog retains."""
        result = original_resolve(runtime, alias, role=role)
        return replace(result, served_model_id=f"served-{alias}") if alias in pinned else result

    monkeypatch.setattr(_CompletionClient, "complete", complete)
    monkeypatch.setattr(_RuntimeCatalog, "resolve", resolve)
    result = run_prepared_model_evaluation(
        project,
        prepared,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
        budget=EvaluationBudget(maximum_cost_usd=100, maximum_judgments=100),
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert result.report.compared_cells == prepared.cost.scenario_count
    assert all(row.quality == 1 for row in result.report.models)

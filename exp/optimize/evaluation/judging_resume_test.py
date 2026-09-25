"""Judging recovery preserves completed rollouts, scores, and prior spend."""

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel, JsonValue

from exp.common.core.artifacts import ArtifactEnvelope
from exp.common.judging import Judgment, Rubric
from exp.common.models import AssistantAction, ModelCatalog, ModelRequest, ModelResponse, Usage
from exp.common.project import ArtifactManifest
from exp.common.rollouts import RolloutArtifact
from exp.optimize.evaluation.judging_resume import (
    prepare_judging_revision,
    read_judging_revision,
)
from exp.optimize.evaluation.prepare import ModelEvaluationOptions
from exp.optimize.evaluation.runs import (
    EvaluationDefaults,
    execute_run,
    load_run,
    prepare_run,
    save_run,
)
from exp.optimize.router.automatic import service_test as automatic_fixtures
from exp.optimize.router.automatic.service_test import (
    _REVISION,
    _completed_project,
    _CompletionClient,
    _RuntimeCatalog,
)
from exp.optimize.router.judging import protocol
from exp.optimize.router.judging.contracts import JudgePromptTemplate, ManualJudgeError
from exp.optimize.router.judgment_budget import JudgmentExclusionRecord
from exp.runtime.models import RuntimeModelCatalog


def test_judging_retry_preserves_rollouts_valid_scores_and_all_attempt_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recover two kinds of judge failure without another worker or world-model dispatch."""
    project, catalog, state = _completed_project(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(
            models=("candidate-a", "candidate-b"),
            minimum_scenarios=3,
            options=ModelEvaluationOptions(maximum_steps=1, maximum_judge_input_tokens=32_768),
        ),
        code_revision=_REVISION,
    ).model_copy(update={"spending_limit_usd": 100.0})
    save_run(project, run)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    original_complete = _CompletionClient.complete
    original_render = protocol._bounded_judge_request
    render_count = [0]
    judge_count = [0]

    def render(
        template: JudgePromptTemplate,
        rubric: Rubric,
        candidate_a: RolloutArtifact,
        candidate_b: RolloutArtifact | None,
        *,
        maximum_input_tokens: int | None,
        maximum_output_tokens: int,
    ) -> ModelRequest:
        """Reproduce the input-renderer crash after one exclusion and one valid judgment."""
        render_count[0] += 1
        if render_count[0] == 3:
            raise ManualJudgeError("judge request exceeds its reserved input ceiling")
        return original_render(
            template,
            rubric,
            candidate_a,
            candidate_b,
            maximum_input_tokens=maximum_input_tokens,
            maximum_output_tokens=maximum_output_tokens,
        )

    def complete(client: _CompletionClient, request: ModelRequest) -> ModelResponse:
        """Make the first judge response malformed; remaining responses are usable."""
        response = original_complete(client, request)
        if client._alias == "judge":
            judge_count[0] += 1
            if judge_count[0] == 1:
                return response.model_copy(update={"output": AssistantAction(content="invalid")})
        return response

    monkeypatch.setattr(protocol, "_bounded_judge_request", render)
    monkeypatch.setattr(_CompletionClient, "complete", complete)
    with pytest.raises(ManualJudgeError, match="input ceiling"):
        execute_run(project, run, runtime, provider_spend_consented=True)
    failed = load_run(project, run.run_id)
    assert failed.status == "failed" and failed.stage == "judgments"
    original_preparation = failed.prepared
    saved_rollouts = {
        key: project.artifacts.read_bytes(key, "rollout.json")
        for key in project.artifacts.list_ids()
        if project.artifacts.read(key).manifest.artifact_type == "rollout"
    }
    prior_judgments = {
        key
        for key in project.artifacts.list_ids()
        if project.artifacts.read(key).manifest.artifact_type == "judgment"
    }
    assert len(prior_judgments) == 1
    before = Counter(alias for alias, _ in state.completion_calls)
    embeddings_before = len(state.embedding_calls)
    pointer = prepare_judging_revision(
        project,
        failed.prepared,
        catalog,
        created_at=failed.created_at,
        code_revision=failed.code_revision,
    )
    assert before == Counter(alias for alias, _ in state.completion_calls)
    revision = read_judging_revision(project, failed.prepared, pointer)
    assert revision.request.maximum_input_tokens > 32_768
    resumed = failed.model_copy(update={"judging_revision": pointer})
    save_run(project, resumed)
    monkeypatch.setattr(protocol, "_bounded_judge_request", original_render)
    result = execute_run(project, resumed, runtime, provider_spend_consented=True)
    after = Counter(alias for alias, _ in state.completion_calls)
    assert after - before == {"judge": 5}
    assert len(state.embedding_calls) == embeddings_before
    assert result.report.compared_cells == 3
    assert prior_judgments.issubset(project.artifacts.list_ids())
    successful_cost = 0.0
    excluded_cost = 0.0
    for key in project.artifacts.list_ids():
        kind = project.artifacts.read(key).manifest.artifact_type
        if kind == "judgment":
            judgment = Judgment.model_validate_json(
                project.artifacts.read_bytes(key, "judgment.json")
            )
            assert judgment.judge_economics is not None
            assert judgment.judge_economics.cost_usd is not None
            successful_cost += judgment.judge_economics.cost_usd.value
        elif kind == "judgment-exclusion":
            exclusion = JudgmentExclusionRecord.model_validate_json(
                project.artifacts.read_bytes(key, "exclusion.json")
            )
            excluded_cost += exclusion.conservative_cost_usd
    assert excluded_cost > 0
    assert result.judge_cost_usd == pytest.approx(successful_cost + excluded_cost)
    for key, payload in saved_rollouts.items():
        assert project.artifacts.read_bytes(key, "rollout.json") == payload
    finished = load_run(project, run.run_id)
    assert finished.prepared == original_preparation
    assert finished.status == "completed"
    replay = execute_run(project, finished, runtime, provider_spend_consented=True)
    assert replay == result
    assert after == Counter(alias for alias, _ in state.completion_calls)


def test_judging_revision_rejects_another_preparation_without_provider_calls(
    tmp_path: Path,
) -> None:
    """A judging pass cannot be used to change the original worker/model matrix."""
    project, catalog, state = _completed_project(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(models=("candidate-a", "candidate-b"), minimum_scenarios=3),
        code_revision=_REVISION,
    )
    pointer = prepare_judging_revision(
        project,
        run.prepared,
        catalog,
        created_at=run.created_at,
        code_revision=run.code_revision,
    )
    before = len(state.completion_calls)
    changed = run.prepared.model_copy(update={"embedder_alias": "another-embedder"})
    with pytest.raises(ValueError, match="another evaluation"):
        read_judging_revision(project, changed, pointer)
    assert len(state.completion_calls) == before


def test_recovery_counts_paid_response_when_probe_write_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paid response survives in the request ledger even when no probe/exclusion was saved."""
    project, catalog, state = _completed_project(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(
            models=("candidate-a", "candidate-b"),
            minimum_scenarios=3,
            options=ModelEvaluationOptions(maximum_steps=1),
        ),
        code_revision=_REVISION,
    ).model_copy(update={"spending_limit_usd": 100.0})
    save_run(project, run)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    original_write = project.artifacts.write_json

    def reject_probe(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Simulate a disk error after the paid response reached durable request storage."""
        if artifact_type == "manual-judge-probe":
            raise OSError("disk error writing completed probe")
        return original_write(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            envelope=envelope,
            files=files,
        )

    monkeypatch.setattr(project.artifacts, "write_json", reject_probe)
    with pytest.raises(OSError, match="disk error"):
        execute_run(project, run, runtime, provider_spend_consented=True)
    monkeypatch.setattr(project.artifacts, "write_json", original_write)
    failed = load_run(project, run.run_id)
    assert not any(
        project.artifacts.read(key).manifest.artifact_type == "judgment-exclusion"
        for key in project.artifacts.list_ids()
    )
    revision = prepare_judging_revision(
        project,
        failed.prepared,
        catalog,
        created_at=failed.created_at,
        code_revision=failed.code_revision,
    )
    resumed = failed.model_copy(update={"judging_revision": revision})
    save_run(project, resumed)
    result = execute_run(project, resumed, runtime, provider_spend_consented=True)
    assert sum(alias == "judge" for alias, _ in state.completion_calls) == 7
    judgments = [
        Judgment.model_validate_json(project.artifacts.read_bytes(key, "judgment.json"))
        for key in project.artifacts.list_ids()
        if project.artifacts.read(key).manifest.artifact_type == "judgment"
    ]
    costs = [
        j.judge_economics.cost_usd.value
        for j in judgments
        if j.judge_economics is not None and j.judge_economics.cost_usd is not None
    ]
    assert len(costs) == 6
    assert result.judge_cost_usd == pytest.approx(sum(costs) + costs[0])


def test_judging_recovery_can_exceed_original_quote_under_approved_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free simulation plus long paid judging can exceed the old quote without changing rollouts."""
    original_catalog = automatic_fixtures._catalog
    original_complete = _CompletionClient.complete

    def catalog_with_free_simulation() -> ModelCatalog:
        """Use free worker/world/retrieval models and a priced large-context judge."""
        catalog = original_catalog()
        models = {}
        for alias, record in catalog.models.items():
            caps = record.capabilities
            assert caps is not None
            values = {"context_window_tokens": 1_000_000}
            if alias != "judge":
                values.update(
                    {
                        "input_cost_per_million_tokens_usd": 0.0,
                        "output_cost_per_million_tokens_usd": 0.0,
                        "cached_input_cost_per_million_tokens_usd": 0.0,
                        "cache_write_cost_per_million_tokens_usd": 0.0,
                    }
                )
            models[alias] = record.model_copy(
                update={"capabilities": caps.model_copy(update=values)}
            )
        return catalog.model_copy(update={"models": models})

    def complete(client: _CompletionClient, request: ModelRequest) -> ModelResponse:
        """Return long valid answers and observed judge usage below full request size."""
        response = original_complete(client, request)
        if client._alias.startswith("candidate-"):
            return response.model_copy(update={"output": AssistantAction(content="x" * 200_000)})
        if client._alias == "judge":
            assert len(request.messages[1].content or "") > 300_000
            return response.model_copy(
                update={
                    "economics": response.economics.model_copy(
                        update={
                            "usage": Usage(
                                input_tokens=300_000, output_tokens=4, cached_input_tokens=0
                            ),
                            "provider_attempts": 1,
                        }
                    )
                }
            )
        return response

    monkeypatch.setattr(automatic_fixtures, "_catalog", catalog_with_free_simulation)
    monkeypatch.setattr(_CompletionClient, "complete", complete)
    project, catalog, state = _completed_project(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(
            models=("candidate-a", "candidate-b"),
            minimum_scenarios=3,
            options=ModelEvaluationOptions(
                maximum_steps=1,
                maximum_judge_input_tokens=32_768,
                maximum_retrieval_query_tokens=1_000_000,
            ),
        ),
        code_revision=_REVISION,
    ).model_copy(update={"spending_limit_usd": 10.0})
    save_run(project, run)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    first = execute_run(project, run, runtime, provider_spend_consented=True)
    assert first.report.compared_cells == 0
    assert first.simulation_cost_usd == 0
    before = Counter(alias for alias, _ in state.completion_calls)
    revision = prepare_judging_revision(
        project,
        run.prepared,
        catalog,
        created_at=run.created_at,
        code_revision=run.code_revision,
    )
    resumed = run.model_copy(update={"judging_revision": revision, "spending_limit_usd": 20.0})
    save_run(project, resumed)
    result = execute_run(project, resumed, runtime, provider_spend_consented=True)
    assert result.simulation_spec == first.simulation_spec
    assert Counter(alias for alias, _ in state.completion_calls) - before == {"judge": 6}
    assert result.report.compared_cells == 3
    assert result.judge_cost_usd > run.prepared.cost.maximum_cost_usd
    assert result.judge_cost_usd < resumed.spending_limit_usd

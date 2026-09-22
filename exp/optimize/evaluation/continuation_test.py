"""Full native evaluation continuation with only the provider transport scripted."""

from pathlib import Path
from typing import cast

import pytest

from exp.common.models import AssistantAction, ModelRequest, ModelResponse
from exp.common.project import write_project_config
from exp.common.rollouts import RolloutArtifact, StopReason, unknown_dispatch_reserved_cost_usd
from exp.optimize.evaluation.contracts import EvaluationBudget
from exp.optimize.evaluation.prepare import ModelEvaluationOptions, prepare_model_evaluation
from exp.optimize.evaluation.prepare_test import _prepare
from exp.optimize.evaluation.runtime import run_prepared_model_evaluation
from exp.optimize.router.automatic.service_test import (
    _REVISION,
    _TIME,
    _CompletionClient,
    _RuntimeCatalog,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.simulation.engines.text.rollout_support import rollout_spend


@pytest.mark.parametrize("limit", ["steps", "tokens"])
@pytest.mark.parametrize("retry", [False, True])
def test_native_continuation_keeps_prefix_costs_and_excludes_incomplete_judgments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
    retry: bool,
) -> None:
    """A budget-limited matrix resumes with one new turn per cell and no prefix dispatches."""
    project, catalog, state, initial = _prepare(tmp_path)
    terminal = False
    failed = False
    original = _CompletionClient.complete

    def complete(client: _CompletionClient, request: ModelRequest) -> ModelResponse:
        """Keep worlds nonterminal until the next explicitly authorized execution."""
        nonlocal failed
        if retry and terminal and not failed and client._alias.startswith("candidate-"):
            failed = True
            raise TimeoutError("fixture unknown provider outcome")
        response = original(client, request)
        if client._alias == "world":
            return response.model_copy(
                update={
                    "output": AssistantAction(
                        content='{"message":"continue","terminal":' + str(terminal).lower() + "}",
                    )
                }
            )
        return response

    monkeypatch.setattr(_CompletionClient, "complete", complete)
    initial = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        judge_setup=initial.judge_setup,
        calibration_id=initial.setup.simulation_protocol.judge_calibration_id,
        embedder_alias="embedder",
        options=ModelEvaluationOptions(
            maximum_steps=1 if limit == "steps" else 100,
            maximum_rollout_output_tokens=4 if limit == "tokens" else 1_000_000,
        ),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    result = run_prepared_model_evaluation(
        project,
        initial,
        runtime,
        budget=EvaluationBudget(maximum_cost_usd=10_000, maximum_judgments=100),
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert all(m.incomplete_cells == 3 and m.failed_cells == 0 for m in result.report.models)
    assert all(m.quality is None and m.operating_cost_usd is None for m in result.report.models)
    assert not any(alias == "judge" for alias, _ in state.completion_calls)
    parent_bytes = {
        identity: project.artifacts.read_bytes(identity, "rollout.json")
        for identity in project.artifacts.list_ids()
        if project.artifacts.read(identity).manifest.artifact_type == "rollout"
    }
    terminal = True
    child = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        continuation_of=result.simulation_spec.simulation_id,
        judge_setup=initial.judge_setup,
        calibration_id=initial.setup.simulation_protocol.judge_calibration_id,
        embedder_alias="embedder",
        options=ModelEvaluationOptions(maximum_steps=101, maximum_rollout_output_tokens=2_000_000),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    before = len(state.completion_calls)
    resumed = run_prepared_model_evaluation(
        project,
        child,
        runtime,
        budget=EvaluationBudget(maximum_cost_usd=10_000, maximum_judgments=100),
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    calls = state.completion_calls[before:]
    candidates = [request for alias, request in calls if alias.startswith("candidate-")]
    assert len(candidates) == 6
    assert all(
        any(message.content == "continue" for message in request.messages) for request in candidates
    )
    assert resumed.report.compared_cells == 3
    assert all(m.incomplete_cells == 0 and m.failed_cells == 0 for m in resumed.report.models)
    for identity, payload in parent_bytes.items():
        assert project.artifacts.read_bytes(identity, "rollout.json") == payload
    children = [
        RolloutArtifact.model_validate_json(project.artifacts.read_bytes(identity, "rollout.json"))
        for identity in project.artifacts.list_ids()
        if identity not in parent_bytes
        and project.artifacts.read(identity).manifest.artifact_type == "rollout"
    ]
    failed_children = [item for item in children if item.stop_reason == StopReason.FAILURE]
    children = [item for item in children if item.stop_reason == StopReason.COMPLETED]
    assert len(children) == 6
    assert len(failed_children) == int(retry)
    expected = sum(rollout_spend(item) or 0 for item in children)
    if failed_children:
        reservation = unknown_dispatch_reserved_cost_usd(failed_children[0].failure)
        assert reservation is not None
        expected += reservation
    assert resumed.simulation_cost_usd == pytest.approx(expected)
    assert all(item.continuation_of is not None for item in children)
    for item in children:
        assert item.continuation_of is not None
        parent = RolloutArtifact.model_validate_json(parent_bytes[item.continuation_of.artifact_id])
        parent_cost = parent.candidate_economics.cost_usd
        child_cost = item.candidate_economics.cost_usd
        assert parent_cost is not None and child_cost is not None
        assert child_cost.value == pytest.approx(2 * parent_cost.value)
    assert resumed.simulation_cost_usd > result.simulation_cost_usd > 0
    assert all(
        item.candidate_economics.usage is not None
        and item.candidate_economics.usage.output_tokens == 8
        for item in children
    )
    before_replay = len(state.completion_calls)
    replay = run_prepared_model_evaluation(
        project,
        child,
        runtime,
        budget=EvaluationBudget(maximum_cost_usd=10_000, maximum_judgments=100),
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert replay == resumed
    assert len(state.completion_calls) == before_replay


@pytest.mark.parametrize("drift", [False, True])
def test_completed_prefix_is_reused_and_changed_redaction_is_rejected(
    tmp_path: Path,
    drift: bool,
) -> None:
    """Completed model calls are free to replay; changed redaction cannot reuse them."""
    project, catalog, state, initial = _prepare(tmp_path)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    budget = EvaluationBudget(maximum_cost_usd=10_000, maximum_judgments=100)
    previous = run_prepared_model_evaluation(
        project,
        initial,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    if drift:
        write_project_config(
            project.paths,
            project.load_project().model_copy(
                update={
                    "redacted_field_names": ("private-value",),
                }
            ),
        )
    child = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        continuation_of=previous.simulation_spec.simulation_id,
        judge_setup=initial.judge_setup,
        calibration_id=initial.setup.simulation_protocol.judge_calibration_id,
        embedder_alias="embedder",
        options=ModelEvaluationOptions(maximum_steps=2),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    before = len(state.completion_calls), state.credential_resolutions
    if drift:
        with pytest.raises(ValueError, match="settings changed"):
            run_prepared_model_evaluation(
                project,
                child,
                runtime,
                budget=budget,
                provider_spend_consented=True,
                created_at=_TIME,
                code_revision=_REVISION,
            )
        assert before == (len(state.completion_calls), state.credential_resolutions)
    else:
        result = run_prepared_model_evaluation(
            project,
            child,
            runtime,
            budget=budget,
            provider_spend_consented=True,
            created_at=_TIME,
            code_revision=_REVISION,
        )
        assert result.report.compared_cells == 3
        assert all(alias == "judge" for alias, _ in state.completion_calls[before[0] :])
        assert result.simulation_cost_usd == previous.simulation_cost_usd

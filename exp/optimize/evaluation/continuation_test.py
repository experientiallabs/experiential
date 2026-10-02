"""Full native evaluation continuation with only the provider transport scripted."""

import json
from pathlib import Path
from typing import cast

import pytest

from exp.common.models import AssistantAction, ModelRequest, ModelResponse, ModelSnapshot, ToolCall
from exp.common.project import write_project_config
from exp.common.rollouts import RolloutArtifact, StopReason, unknown_dispatch_reserved_cost_usd
from exp.common.tasks import ToolSchema
from exp.common.traces import Trace
from exp.optimize.evaluation.contracts import EvaluationBudget
from exp.optimize.evaluation.prepare import ModelEvaluationOptions, prepare_model_evaluation
from exp.optimize.evaluation.prepare_test import _prepare
from exp.optimize.evaluation.runtime import run_prepared_model_evaluation
from exp.optimize.router.automatic import service_test as build_fixtures
from exp.optimize.router.automatic.service_test import (
    _REVISION,
    _TIME,
    _CompletionClient,
    _RuntimeCatalog,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.simulation.engines.text.rollout_support import rollout_spend


@pytest.mark.parametrize("limit", ["steps", "tokens"])
def test_tool_result_checkpoint_continues_the_evaluation_without_replaying_paid_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    """Resume parallel tool results and private world state through the public evaluation flow."""
    tool = ToolSchema(name="lookup", description="Look up a support account.", input_schema={})
    original_trace = build_fixtures._trace

    def trace(index: int, model: ModelSnapshot) -> Trace:
        """Mine real project artifacts from synthetic traces declaring the same tool."""
        return original_trace(index, model).model_copy(update={"tools": (tool,)})

    monkeypatch.setattr(build_fixtures, "_trace", trace)
    project, catalog, state, initial = _prepare(tmp_path)
    original_complete = _CompletionClient.complete
    calls = (
        ToolCall(call_id="found", name="lookup", arguments={"account": "existing"}),
        ToolCall(call_id="missing", name="lookup", arguments={"account": "absent"}),
    )
    continuing = False

    def complete(client: _CompletionClient, request: ModelRequest) -> ModelResponse:
        """Script only model transport; production code owns artifacts, tools and budgets."""
        response = original_complete(client, request)
        if client._alias.startswith("candidate-"):
            assert request.tools == (tool,)
            if not continuing:
                return response.model_copy(update={"output": AssistantAction(tool_calls=calls)})
            results = [message for message in request.messages if message.role == "tool"]
            assert [(item.tool_call_id, item.content) for item in results] == [
                ("found", "Account located."),
                ("missing", "Account does not exist."),
            ]
            actions = [message for message in request.messages if message.role == "assistant"]
            assert len(actions) == 1 and actions[0].assistant_action is not None
            assert actions[0].assistant_action.tool_calls == calls
        elif client._alias == "world":
            evidence = json.loads(request.messages[-1].content or "")
            if continuing:
                assert evidence["environment_state"] == {"account": "located"}
            else:
                assert evidence["environment_state"] == {}
                return response.model_copy(
                    update={
                        "output": AssistantAction(
                            content=json.dumps(
                                {
                                    "tool_results": [
                                        {"call_id": "found", "content": "Account located."},
                                        {
                                            "call_id": "missing",
                                            "content": "Account does not exist.",
                                            "is_error": True,
                                        },
                                    ],
                                    "state": {"account": "located"},
                                }
                            )
                        )
                    }
                )
        return response

    monkeypatch.setattr(_CompletionClient, "complete", complete)
    initial = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        embedder_alias="embedder",
        options=ModelEvaluationOptions(
            maximum_steps=1 if limit == "steps" else 100,
            maximum_rollout_output_tokens=4 if limit == "tokens" else 1_000_000,
        ),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    budget = EvaluationBudget(
        maximum_cost_usd=initial.cost.maximum_cost_usd,
        maximum_judgments=initial.cost.judgment_count,
    )
    parent = run_prepared_model_evaluation(
        project,
        initial,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert all(model.incomplete_cells == 3 for model in parent.report.models)
    assert not any(alias == "judge" for alias, _ in state.completion_calls)
    parent_bytes = {
        identity: project.artifacts.read_bytes(identity, "rollout.json")
        for identity in project.artifacts.list_ids()
        if project.artifacts.read(identity).manifest.artifact_type == "rollout"
    }
    parent_rollouts = tuple(
        rollout
        for payload in parent_bytes.values()
        if (rollout := RolloutArtifact.model_validate_json(payload)).simulation_id
        == parent.simulation_spec.simulation_id
    )
    assert len(parent_rollouts) == 6
    for rollout in parent_rollouts:
        checkpoint = rollout.text_checkpoint
        assert checkpoint is not None
        assert [message.role for message in checkpoint.visible_transcript] == [
            "assistant",
            "tool",
            "tool",
        ]
    continuing = True
    child = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        embedder_alias="embedder",
        continuation_of=parent.simulation_spec.simulation_id,
        options=ModelEvaluationOptions(maximum_steps=101, maximum_rollout_output_tokens=2_000_000),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    budget = EvaluationBudget(
        maximum_cost_usd=child.cost.maximum_cost_usd,
        maximum_judgments=child.cost.judgment_count,
    )
    before = len(state.completion_calls)
    result = run_prepared_model_evaluation(
        project,
        child,
        runtime,
        budget=budget,
        provider_spend_consented=True,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    new_calls = state.completion_calls[before:]
    assert len(new_calls) == 18
    assert sum(alias.startswith("candidate-") for alias, _ in new_calls) == 6
    assert sum(alias == "world" for alias, _ in new_calls) == 6
    assert sum(alias == "judge" for alias, _ in new_calls) == 6
    assert result.report.compared_cells == 3
    assert all(model.incomplete_cells == 0 and model.quality == 1 for model in result.report.models)
    for identity, payload in parent_bytes.items():
        assert project.artifacts.read_bytes(identity, "rollout.json") == payload
    children = [
        RolloutArtifact.model_validate_json(project.artifacts.read_bytes(identity, "rollout.json"))
        for identity in project.artifacts.list_ids()
        if identity not in parent_bytes
        and project.artifacts.read(identity).manifest.artifact_type == "rollout"
    ]
    assert len(children) == 6
    assert result.simulation_cost_usd == pytest.approx(
        sum(rollout_spend(rollout) or 0 for rollout in children)
    )
    assert result.simulation_cost_usd > parent.simulation_cost_usd > 0
    before_replay = len(state.completion_calls), len(state.embedding_calls)
    assert (
        run_prepared_model_evaluation(
            project,
            child,
            runtime,
            budget=budget,
            provider_spend_consented=True,
            created_at=_TIME,
            code_revision=_REVISION,
        )
        == result
    )
    assert before_replay == (len(state.completion_calls), len(state.embedding_calls))


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

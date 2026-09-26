"""Provider-free catalog preparation and immutable identity regressions."""

from pathlib import Path

import pytest

from exp.common.models import ModelCatalog
from exp.common.progress import ProgressEvent
from exp.common.project import ProjectStore
from exp.optimize.evaluation.prepare import (
    ModelEvaluationOptions,
    PreparedModelEvaluation,
    prepare_model_evaluation,
)
from exp.optimize.router.automatic.provisional import prepare_hosted_provisional_judge
from exp.optimize.router.automatic.service_test import (
    _REVISION,
    _TIME,
    _completed_project,
    _ProviderState,
)
from exp.runtime.agents import ChatAgentRuntime, agent_factory_sha256
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.specs import load_simulation_completion_contract


def _prepare(
    root: Path,
) -> tuple[ProjectStore, ModelCatalog, _ProviderState, PreparedModelEvaluation]:
    """Create a real grounded project and default judge, then prepare its worker comparison."""
    project, catalog, state = _completed_project(root)
    judge = prepare_hosted_provisional_judge(
        project,
        catalog,
        maximum_input_tokens=32_768,
        maximum_output_tokens=8_192,
        maximum_attempts=RetryPolicy().maximum_attempts,
        created_at=_TIME,
        code_revision=_REVISION,
    )
    calls = (len(state.embedding_calls), len(state.completion_calls), state.credential_resolutions)
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
    assert calls == (
        len(state.embedding_calls),
        len(state.completion_calls),
        state.credential_resolutions,
    )
    return project, catalog, state, prepared


def test_prepare_freezes_replayable_catalog_selection_without_provider_access(
    tmp_path: Path,
) -> None:
    """Static preparation freezes only model, judge, pricing and scenario evidence."""
    project, catalog, _, prepared = _prepare(tmp_path)
    assert ModelEvaluationOptions().maximum_output_tokens is None
    assert ModelEvaluationOptions().maximum_judge_input_tokens is None
    judge_caps = catalog.models["judge"].capabilities
    assert judge_caps is not None and judge_caps.context_window_tokens is not None
    assert prepared.judge_request.maximum_input_tokens == (
        judge_caps.context_window_tokens - prepared.judge_request.maximum_output_tokens
    )
    assert prepared.setup.world_model_settings.maximum_output_tokens == 32_000
    assert prepared.setup.world_model_settings.json_object_output is True
    assert prepared.cost.worker_count == 2
    assert prepared.cost.scenario_count == 3
    assert prepared.cost.maximum_cost_usd > 0
    before = project.artifacts.list_ids()
    replay = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-b", "candidate-a"),
        judge_setup=prepared.judge_setup,
        calibration_id=prepared.setup.simulation_protocol.judge_calibration_id,
        embedder_alias="embedder",
        options=ModelEvaluationOptions(maximum_steps=1),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert replay == prepared
    assert project.artifacts.list_ids() == before
    assert PreparedModelEvaluation.model_validate_json(prepared.model_dump_json()) == prepared


def test_prepare_rejects_duplicate_workers_before_writes(tmp_path: Path) -> None:
    """Two aliases for the same selected entry must not manufacture extra comparison coverage."""
    project, catalog, _, prepared = _prepare(tmp_path)
    before = project.artifacts.list_ids()
    with pytest.raises(ValueError, match="distinct worker"):
        prepare_model_evaluation(
            project,
            catalog,
            ("candidate-a", "candidate-a"),
            judge_setup=prepared.judge_setup,
            calibration_id=prepared.setup.simulation_protocol.judge_calibration_id,
            embedder_alias="embedder",
            options=ModelEvaluationOptions(),
            created_at=_TIME,
            code_revision=_REVISION,
        )
    assert project.artifacts.list_ids() == before


def test_new_evaluation_after_upgrade_reuses_build_without_replaying_old_prices(
    tmp_path: Path,
) -> None:
    """A release update can prepare the same models while preserving prior evidence."""
    project, catalog, state, prepared = _prepare(tmp_path)
    before = (len(state.embedding_calls), len(state.completion_calls), state.credential_resolutions)
    build = project.load_project().build
    old_pricing = project.artifacts.read_bytes(prepared.setup.pricing_snapshot_id, "pricing.json")

    upgraded = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        embedder_alias="embedder",
        options=ModelEvaluationOptions(maximum_steps=1),
        created_at=_TIME,
        code_revision="upgraded-release",
    )

    assert upgraded.setup.pricing_snapshot_id != prepared.setup.pricing_snapshot_id
    assert upgraded.cost.estimated_cost_usd == prepared.cost.estimated_cost_usd
    assert upgraded.judge_setup == prepared.judge_setup
    assert project.load_project().build == build
    assert (
        project.artifacts.read_bytes(prepared.setup.pricing_snapshot_id, "pricing.json")
        == old_pricing
    )
    assert before == (
        len(state.embedding_calls),
        len(state.completion_calls),
        state.credential_resolutions,
    )


def test_prepare_defaults_to_binary_task_success_without_calibration_calls(tmp_path: Path) -> None:
    """An ordinary evaluation requires no manual calibration detour or private API."""
    import exp

    project, catalog, state = _completed_project(tmp_path)
    before = (len(state.completion_calls), state.credential_resolutions)
    progress: list[ProgressEvent] = []
    prepared = exp.prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        embedder_alias="embedder",
        options=exp.ModelEvaluationOptions(maximum_steps=1),
        created_at=_TIME,
        code_revision=_REVISION,
        progress=progress.append,
    )
    assert prepared.setup.judgment_status == "provisional"
    assert before == (len(state.completion_calls), state.credential_resolutions)
    assert progress[0].stage == "Verifying built project"
    assert progress[-1].stage == "Calculating evaluation quote"
    assert [event.completed for event in progress if event.stage == "Estimating model costs"] == [
        0,
        1,
        2,
        3,
    ]


def test_rollout_defaults_and_large_explicit_limits() -> None:
    """Long agent tasks are not rejected by an arbitrary small engine ceiling."""
    options = ModelEvaluationOptions()
    assert options.maximum_steps == 100
    assert options.maximum_rollout_output_tokens == 1_000_000
    assert ModelEvaluationOptions(maximum_steps=1000).maximum_steps == 1000
    ChatAgentRuntime(maximum_model_calls=1000)
    assert agent_factory_sha256(None, maximum_model_calls=1000)


@pytest.mark.parametrize("output_ceiling", [None, 1_000_000])
def test_heterogeneous_capacities_use_model_specific_input_estimates(
    tmp_path: Path,
    output_ceiling: int | None,
) -> None:
    """A large worker's output capacity cannot crowd a smaller worker out of preparation."""
    project, catalog, _, initial = _prepare(tmp_path)
    models = dict(catalog.models)
    for alias, context, output in (
        ("candidate-a", 65_536, 4_096),
        ("candidate-b", 2_000_000, 128_000),
    ):
        capabilities = models[alias].capabilities
        assert capabilities is not None
        models[alias] = models[alias].model_copy(
            update={
                "capabilities": capabilities.model_copy(
                    update={"context_window_tokens": context, "maximum_output_tokens": output},
                )
            }
        )
    catalog = catalog.model_copy(update={"models": models})
    prepared = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        judge_setup=initial.judge_setup,
        calibration_id=initial.setup.simulation_protocol.judge_calibration_id,
        embedder_alias="embedder",
        options=ModelEvaluationOptions(maximum_steps=1, maximum_output_tokens=output_ceiling),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    pointer = prepared.setup.simulation_completion_input
    assert pointer is not None
    contract, _ = load_simulation_completion_contract(project.artifacts, pointer.artifact_id)
    reservations = {item.candidate_alias: item.request for item in contract.candidate_requests}
    small, large = reservations["candidate-a"], reservations["candidate-b"]
    assert small.maximum_output_tokens == 4_096
    assert large.maximum_output_tokens == 128_000
    assert small.planning_input_tokens() < small.maximum_input_tokens
    assert large.planning_input_tokens() - small.planning_input_tokens() == 128_000 - 4_096


@pytest.mark.parametrize("context", [64_000, 1_310_720])
def test_unpublished_output_limit_uses_rollout_budget_without_changing_metadata(
    tmp_path: Path, context: int
) -> None:
    """Missing provider output metadata does not prevent a finite context-bounded evaluation."""
    project, catalog, state, initial = _prepare(tmp_path)
    record = catalog.models["candidate-b"]
    assert record.capabilities is not None
    catalog.models["candidate-b"] = record.model_copy(
        update={
            "capabilities": record.capabilities.model_copy(
                update={"maximum_output_tokens": None, "context_window_tokens": context}
            )
        }
    )
    before = len(state.completion_calls), state.credential_resolutions
    prepared = prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        judge_setup=initial.judge_setup,
        calibration_id=initial.setup.simulation_protocol.judge_calibration_id,
        embedder_alias="embedder",
        options=ModelEvaluationOptions(),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    pointer = prepared.setup.simulation_completion_input
    assert pointer is not None
    contract, _ = load_simulation_completion_contract(project.artifacts, pointer.artifact_id)
    request = next(
        item.request
        for item in contract.candidate_requests
        if item.candidate_alias == "candidate-b"
    )
    assert request.maximum_output_tokens == min(1_000_000, context)
    assert request.maximum_input_tokens == context
    assert prepared.setup.maximum_steps == 100
    assert prepared.setup.maximum_rollout_output_tokens == 1_000_000
    capabilities = catalog.models["candidate-b"].capabilities
    assert capabilities is not None and capabilities.maximum_output_tokens is None
    assert before == (len(state.completion_calls), state.credential_resolutions)

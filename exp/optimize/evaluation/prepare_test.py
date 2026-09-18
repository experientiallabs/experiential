"""Provider-free catalog preparation and immutable identity regressions."""

from pathlib import Path

import pytest

from exp.common.models import ModelCatalog
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
from exp.runtime.models.providers.transport import RetryPolicy


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


def test_prepare_defaults_to_binary_task_success_without_calibration_calls(tmp_path: Path) -> None:
    """An ordinary evaluation requires no manual calibration detour or private API."""
    import exp

    project, catalog, state = _completed_project(tmp_path)
    before = (len(state.completion_calls), state.credential_resolutions)
    prepared = exp.prepare_model_evaluation(
        project,
        catalog,
        ("candidate-a", "candidate-b"),
        embedder_alias="embedder",
        options=exp.ModelEvaluationOptions(maximum_steps=1),
        created_at=_TIME,
        code_revision=_REVISION,
    )
    assert prepared.setup.judgment_status == "provisional"
    assert before == (len(state.completion_calls), state.credential_resolutions)

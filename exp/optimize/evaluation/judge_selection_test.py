"""A run-local judge change preserves the syllabus and never reuses another model's calibration."""

from pathlib import Path

import pytest

from exp.common.judging import verify_persisted_calibration
from exp.optimize.evaluation.prepare import ModelEvaluationOptions, read_evaluation_judge
from exp.optimize.evaluation.runs import EvaluationDefaults, prepare_run
from exp.optimize.evaluation.runs_test import _twenty_scenarios
from exp.optimize.router.automatic.service_test import _REVISION


def test_changing_judge_preserves_syllabus_and_project_defaults(tmp_path: Path) -> None:
    """Frozen old runs and project judge settings survive a new provisional model selection."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    defaults = EvaluationDefaults(
        models=("candidate-a", "candidate-b"), options=ModelEvaluationOptions(maximum_steps=1)
    )
    original = prepare_run(project, catalog, defaults, code_revision=_REVISION)
    source = read_evaluation_judge(project, original.prepared.judge_setup)
    before = len(state.completion_calls), len(state.embedding_calls)
    project_config = project.load_project()
    changed = prepare_run(
        project,
        catalog,
        defaults.model_copy(update={"judge": "candidate-b"}),
        code_revision=_REVISION,
    )
    selected = read_evaluation_judge(project, changed.prepared.judge_setup)
    calibration, _ = verify_persisted_calibration(
        project, changed.prepared.setup.simulation_protocol.judge_calibration_id
    )
    assert selected.rubric == source.rubric and selected.prompt_template == source.prompt_template
    assert selected.judge_alias == "candidate-b"
    assert selected.judge_model != source.judge_model
    assert selected.setup_id != source.setup_id
    assert calibration.judge_model == selected.judge_model
    assert calibration.status == "provisional" and calibration.label_count == 0
    assert project.load_project() == project_config
    assert read_evaluation_judge(project, original.prepared.judge_setup) == source
    replay = prepare_run(
        project,
        catalog,
        defaults.model_copy(update={"judge": "candidate-b"}),
        code_revision=_REVISION,
    )
    assert replay.prepared.judge_setup == changed.prepared.judge_setup
    assert before == (len(state.completion_calls), len(state.embedding_calls))


def test_judge_override_requires_structured_output(tmp_path: Path) -> None:
    """Unsupported judge choices fail before a run or provider call can be created."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    model = catalog.models["candidate-b"]
    assert model.capabilities is not None
    incompatible = model.model_copy(
        update={
            "capabilities": model.capabilities.model_copy(
                update={"supports_structured_output": False}
            )
        }
    )
    catalog = catalog.model_copy(
        update={"models": {**catalog.models, "unsupported-judge": incompatible}}
    )
    before = len(state.completion_calls), len(state.embedding_calls)
    with pytest.raises(ValueError, match="structured output"):
        prepare_run(
            project,
            catalog,
            EvaluationDefaults(
                models=("candidate-a", "candidate-b"),
                judge="unsupported-judge",
            ),
            code_revision=_REVISION,
        )
    assert not list((project.paths.runtime_directory / "evaluations").glob("*/run.json"))
    assert before == (len(state.completion_calls), len(state.embedding_calls))

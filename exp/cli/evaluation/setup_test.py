"""Evaluation setup selects models, runs, and judge from an existing built project."""

from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from exp.cli.evaluation import setup
from exp.cli.shared.picker import PickerAction, PickerOption, PickerResult
from exp.optimize.evaluation.runs import EvaluationDefaults
from exp.optimize.evaluation.runs_test import _twenty_scenarios


def test_models_runs_and_judge_preserve_project_and_rollout_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh run uses the selected judge without a provider or import setup screen."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    models = dict(catalog.models)
    assert models["embedder"].capabilities is not None
    models["embedder"] = models["embedder"].model_copy(
        update={
            "capabilities": models["embedder"].capabilities.model_copy(
                update={"supports_completions": False}
            )
        }
    )
    assert models["candidate-a"].capabilities is not None
    models["candidate-a"] = models["candidate-a"].model_copy(
        update={
            "capabilities": models["candidate-a"].capabilities.model_copy(
                update={"supports_structured_output": False}
            )
        }
    )
    catalog = catalog.model_copy(update={"models": models})
    before = len(state.completion_calls), len(state.embedding_calls)
    before_project = project.load_project()
    screens: list[str] = []

    def choose_models(
        console: Console,
        *,
        title: str,
        minimum: int,
        preselected: Sequence[str],
        options: Sequence[PickerOption],
    ) -> PickerResult:
        """Select two models from a list with visible provider connection names."""
        del console, minimum, preselected
        screens.append(title)
        values = {option.value for option in options}
        assert "embedder" not in values
        assert all(option.detail == catalog.models[option.value].connection for option in options)
        return PickerResult(values=("candidate-a", "candidate-b"))

    def choose_one(
        console: Console, *, title: str, options: Sequence[PickerOption], default: str | None = None
    ) -> PickerResult:
        """Select a new judge, then accept the displayed execution defaults."""
        del console
        screens.append(title)
        if title == "Judge":
            assert "candidate-a" not in {option.value for option in options}
            assert "candidate-b" in {option.value for option in options}
            assert default == "judge"
            return PickerResult(values=("candidate-b",))
        return PickerResult(values=("review",))

    monkeypatch.setattr(setup, "choose_many", choose_models)
    monkeypatch.setattr(setup, "choose_one", choose_one)
    monkeypatch.setattr(setup.IntPrompt, "ask", lambda *args, **kwargs: 3)
    selected = setup.configure_evaluation(
        Console(file=StringIO(), width=80), project, catalog, EvaluationDefaults()
    )
    assert screens == ["Models", "Judge", "Settings"]
    assert selected is not None
    assert selected.models == ("candidate-a", "candidate-b") and selected.judge == "candidate-b"
    assert selected.options.repeats == 3
    assert selected.options.maximum_steps == 100
    assert selected.options.maximum_rollout_output_tokens == 1_000_000
    assert project.load_project() == before_project
    assert before == (len(state.completion_calls), len(state.embedding_calls))


def test_cancel_models_does_not_continue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel exits setup without another selection or provider work."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    before = len(state.completion_calls), len(state.embedding_calls)
    monkeypatch.setattr(
        setup, "choose_many", lambda *args, **kwargs: PickerResult(action=PickerAction.CANCEL)
    )
    assert (
        setup.configure_evaluation(Console(file=StringIO()), project, catalog, EvaluationDefaults())
        is None
    )
    assert before == (len(state.completion_calls), len(state.embedding_calls))

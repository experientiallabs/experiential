"""Project navigation keeps build and provider setup outside the evaluation command."""

from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from exp.cli.evaluation import flow
from exp.cli.shared.picker import PickerOption, PickerResult
from exp.optimize.evaluation.runs import EvaluationDefaults, prepare_run
from exp.optimize.evaluation.runs_test import _twenty_scenarios
from exp.optimize.router.automatic.service_test import _REVISION


def test_project_home_only_offers_evaluation_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A built project starts at New evaluation without import or provider setup controls."""
    project, _, _ = _twenty_scenarios(tmp_path)

    def choose(console: Console, *, title: str, options: Sequence[PickerOption]) -> PickerResult:
        """Assert the menu exposes only meaningful actions for this empty evaluation history."""
        del console, title
        assert [(option.value, option.label) for option in options] == [
            ("new", "New evaluation"),
            ("exit", "Back"),
        ]
        return PickerResult(values=("new",))

    monkeypatch.setattr(flow, "choose_one", choose)
    assert flow._project_screen(project) is None


def test_review_can_change_allowance_without_changing_rollout_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected usage and the operator's approved allowance are separate review controls."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(models=("candidate-a", "candidate-b")),
        code_revision=_REVISION,
    )
    choices = iter(("limit", "cost", "start"))
    monkeypatch.setattr(
        flow, "choose_one", lambda *args, **kwargs: PickerResult(values=(next(choices),))
    )
    monkeypatch.setattr(flow.FloatPrompt, "ask", lambda *args, **kwargs: 12.0)
    output = StringIO()
    monkeypatch.setattr(flow, "_console", Console(file=output, width=120))
    before = len(state.completion_calls), len(state.embedding_calls)
    reviewed = flow._review(project, run)
    assert reviewed is not None
    assert reviewed.spending_limit_usd == 12
    assert reviewed.prepared == run.prepared
    assert "Spending limit $12.00" in output.getvalue()
    assert "Maximum $" not in output.getvalue()
    assert "Captured turns with measured tokens:" in output.getvalue()
    assert before == (len(state.completion_calls), len(state.embedding_calls))

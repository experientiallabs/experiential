"""Terminal measurements preserve unavailable and positive operating costs."""

from io import StringIO
from pathlib import Path
from typing import cast

import pytest
from rich.console import Console

from exp.cli.evaluation import view
from exp.cli.evaluation.view import (
    _duration,
    _number,
    inspect_report,
    render_details,
    render_report,
)
from exp.cli.shared.picker import PickerResult
from exp.optimize.evaluation.prepare import ModelEvaluationOptions
from exp.optimize.evaluation.runs import (
    EvaluationDefaults,
    execute_run,
    load_run,
    prepare_run,
    run_directory,
    save_run,
)
from exp.optimize.evaluation.runs_test import _twenty_scenarios
from exp.optimize.router.automatic.service_test import _REVISION, _RuntimeCatalog
from exp.runtime.models import RuntimeModelCatalog


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "unavailable"), (0, "$0.0000"), (0.000016, "$0.000016"), (1e-9, "$1e-09")],
)
def test_cost_display_does_not_round_positive_usage_to_free(
    value: float | None, expected: str
) -> None:
    """A cheap measured rollout remains distinguishable from zero or unavailable usage."""
    assert _number(value, "$") == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(None, "unavailable"), (0, "0s"), (0.000035, "0.035ms"), (9.234, "9.2s"), (529, "8m 49s")],
)
def test_latency_uses_readable_units(seconds: float | None, expected: str) -> None:
    """Tiny fixture latency and actual multi-minute rollouts remain distinguishable."""
    assert _duration(seconds) == expected


def test_compact_results_and_explicit_report_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default 80-column view omits IDs and paths; details and browser exports retain them."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(
            models=("candidate-a", "candidate-b"),
            options=ModelEvaluationOptions(maximum_steps=1),
        ),
        code_revision=_REVISION,
    )
    run = run.model_copy(update={"spending_limit_usd": 100.0})
    save_run(project, run)
    execute_run(
        project,
        run,
        cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state)),
        provider_spend_consented=True,
    )
    run = load_run(project, run.run_id)
    before = len(state.completion_calls), len(state.embedding_calls)
    stream = StringIO()
    console = Console(file=stream, width=80, force_terminal=True, record=True)
    render_report(console, project, run)
    assert "\x1b[2J" not in stream.getvalue() and "\x1b[H" not in stream.getvalue()
    compact = console.export_text(clear=False)
    assert "Saved results" in compact and "20 scenarios" in compact
    assert "$0.000016" in compact
    assert run.run_id not in compact and str(tmp_path) not in compact
    assert "simulation $" not in compact
    assert len(compact.splitlines()) <= 16
    assert all(len(line) <= 80 for line in compact.splitlines())
    render_details(console, project, run)
    assert run.run_id in stream.getvalue()
    assert "Experiment spend" in stream.getvalue()
    assert "HTML:" in stream.getvalue() and "JSON:" in stream.getvalue()
    stream.seek(0)
    stream.truncate()
    actions = iter(
        (
            PickerResult(values=("details",)),
            PickerResult(values=("open",)),
            PickerResult(values=("back",)),
        )
    )
    monkeypatch.setattr(view, "choose_one", lambda *args, **kwargs: next(actions))
    opened: list[str] = []
    monkeypatch.setattr(view.typer, "launch", lambda url: opened.append(url) or 0)
    inspect_report(console, project, run)
    assert stream.getvalue().count("Saved results") == 1
    assert opened == [(run_directory(project, run.run_id) / "report.html").as_uri()]
    assert before == (len(state.completion_calls), len(state.embedding_calls))

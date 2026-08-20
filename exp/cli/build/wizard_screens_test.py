"""Interactive wizard screen tests: workflow step picker and explicit trace prompt."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from click import unstyle
from rich.console import Console

import exp.cli.build.wizard_screens as screens


def test_trace_selection_always_prompts_and_never_discovers_local_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even one likely-named local export is never chosen without an explicit path answer.

    Args:
        tmp_path: Isolated current directory with a candidate trace file.
        monkeypatch: Pytest patch fixture supplying deterministic prompt answers.
    """
    likely = tmp_path / "traces.otel.jsonl"
    likely.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    prompts: list[str] = []
    answers = iter(("", str(tmp_path / "missing.jsonl"), str(tmp_path), str(likely)))

    def scripted(prompt: str, **_kwargs: object) -> str:
        """Record every prompt and return the next scripted operator answer."""
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(screens.Prompt, "ask", scripted)
    output = StringIO()

    selected_source, selected_path = screens.select_trace(
        "otlp", console=Console(file=output, force_terminal=False)
    )

    assert (selected_source, selected_path) == ("otlp", likely)
    assert prompts == ["Trace path (otlp export)"] * 4
    printed = unstyle(output.getvalue())
    assert "a local trace path is required" in printed
    assert "trace file not found" in printed
    assert "must name a file" in printed
    assert "Discovered" not in printed


def test_workflow_selection_defaults_and_explicit_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every step is listed upfront; defaults keep rubric and calibration off.

    Args:
        monkeypatch: Pytest patch fixture supplying deterministic prompt answers.
    """
    answers = iter(("1,2,5", "0,9", "2,3,4"))
    monkeypatch.setattr(screens.Prompt, "ask", lambda *_args, **_kwargs: next(answers))
    output = StringIO()

    default_selection = screens.select_workflow(console=Console(file=output, force_terminal=False))

    assert default_selection == screens.WizardWorkflowSelection(
        providers=True,
        build=True,
        judge_rubric=False,
        judge_calibration=False,
        router=True,
    )
    printed = unstyle(output.getvalue())
    assert "providers" in printed
    assert "judge rubric" in printed
    assert "judge calibration" in printed
    assert "router optimization" in printed
    assert "edit the judge rubric" in printed

    output = StringIO()
    custom = screens.select_workflow(console=Console(file=output, force_terminal=False))
    assert "enter step numbers between 1 and 5" in unstyle(output.getvalue())
    assert custom == screens.WizardWorkflowSelection(
        providers=False,
        build=True,
        judge_rubric=True,
        judge_calibration=True,
        router=False,
    )

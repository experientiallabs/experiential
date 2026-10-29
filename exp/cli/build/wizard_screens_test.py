"""Interactive wizard screen tests: workflow step picker and explicit trace prompt."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from click import unstyle
from rich.console import Console

import exp.cli.build.wizard_screens as screens


def test_bare_build_detects_chat_json_without_a_format_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pasted conversation export selects its canonical loader without an OTel default."""
    path = tmp_path / "rollouts.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Research this company"}]}) + "\n"
    )
    prompts: list[str] = []

    def answer(prompt: str, **_kwargs: object) -> str:
        """Supply only the trace path, as in a bare interactive build."""
        prompts.append(prompt)
        return str(path)

    monkeypatch.setattr(screens.Prompt, "ask", answer)
    output = StringIO()

    selected = screens.select_trace(None, console=Console(file=output))

    assert selected == ("chat-json", path)
    assert prompts == ["Trace path"]
    assert "chat-json" in output.getvalue()


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


def test_ambiguous_format_is_selected_in_the_tui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown vendor export asks for its format without requiring another command."""
    path = tmp_path / "export.jsonl"
    path.write_text('{"vendor_specific":true}\n')
    answers = iter((str(path), "langfuse"))
    prompts: list[str] = []

    def answer(prompt: str, **_kwargs: object) -> str:
        """Supply the chosen path and explicit vendor format."""
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(screens.Prompt, "ask", answer)
    selected = screens.select_trace(None, console=Console(file=StringIO()))
    assert selected == ("langfuse", path)
    assert prompts == ["Trace path", "Trace format"]


def test_explicit_source_is_not_overridden_by_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit source goes unchanged to its canonical validator even when incompatible."""
    path = tmp_path / "export.jsonl"
    path.write_text('{"messages":[]}\n')
    monkeypatch.setattr(screens.Prompt, "ask", lambda *_args, **_kwargs: str(path))
    assert screens.select_trace("otlp", console=Console(file=StringIO())) == ("otlp", path)


def test_workflow_selection_defaults_and_explicit_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default prepares scenarios; judge work and router execution require selection.

    Args:
        monkeypatch: Pytest patch fixture supplying deterministic prompt answers.
    """
    answers = iter(("1", "0,9", "1,2,3"))

    def answer(_prompt: str, *, default: str, console: Console) -> str:
        """Verify the actual terminal default, then exercise explicit optional steps."""
        assert default == "1"
        return next(answers)

    monkeypatch.setattr(screens.Prompt, "ask", answer)
    output = StringIO()

    default_selection = screens.select_workflow(console=Console(file=output, force_terminal=False))

    assert default_selection == screens.WizardWorkflowSelection(
        build=True,
        judge_rubric=False,
        judge_calibration=False,
        router=False,
    )
    printed = unstyle(output.getvalue())
    assert "providers" not in printed
    assert "judge rubric" in printed
    assert "judge calibration" in printed
    assert "router optimization" in printed
    assert "edit the judge rubric" in printed

    output = StringIO()
    custom = screens.select_workflow(console=Console(file=output, force_terminal=False))
    assert "enter step numbers between 1 and 4" in unstyle(output.getvalue())
    assert custom == screens.WizardWorkflowSelection(
        build=True,
        judge_rubric=True,
        judge_calibration=True,
        router=False,
    )

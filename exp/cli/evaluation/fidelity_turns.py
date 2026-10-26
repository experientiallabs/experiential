"""Per-turn fidelity measurements: a one-line mark strip and a plain verbose transcript."""

import json

from rich.console import Console
from rich.text import Text

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.evaluations.reconstruction import (
    DualReport,
    EvaluationReferences,
    JudgeConfig,
    TurnOutcome,
    WorkflowEvidence,
    field_differences,
    saved_turn_outcomes,
)
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity import PreparedFidelity
from exp.optimize.evaluation.fidelity_memory import fidelity_decision_trace

REPLY_CHARS = 480
SHOWN_DIFFERENCES = 6
LABEL = 11


def saved_turns(
    store: ProjectStore, prepared: PreparedFidelity, result: DualReport
) -> tuple[WorkflowEvidence, EvaluationReferences, tuple[TurnOutcome, ...]] | None:
    """Load the evaluated turns and their outcomes from the saved evaluation inputs.

    Field agreement is recomputed from the exact saved inputs, so every report reads under
    one definition; judge scores come from the report as saved.

    Args:
        store: Project holding the saved evaluation intent.
        prepared: Exact saved registration.
        result: Validated saved dual report.

    Returns:
        Evidence, references and one outcome per turn, or None before evaluation inputs exist.
    """
    turns = saved_turn_outcomes(store.artifacts, prepared.id + "-report", result)
    if turns is None:
        return None
    intent = prepared.id + "-report-intent"
    evidence = WorkflowEvidence.model_validate_json(
        store.artifacts.read_bytes(intent, "evidence.json")
    )
    references = EvaluationReferences.model_validate_json(
        store.artifacts.read_bytes(intent, "references.json")
    )
    return evidence, references, turns


def turn_marks(turns: tuple[TurnOutcome, ...]) -> Text:
    """One mark per turn: exact, not reproduced, or no recorded reply to compare."""
    marks = Text()
    for index, turn in enumerate(turns):
        if index:
            marks.append(" ")
        divergence = turn.divergence
        if divergence is None:
            marks.append("·", style="dim")
        else:
            marks.append(
                "✓" if divergence == 0 else "✗", style="green" if divergence == 0 else "red"
            )
    return marks


def render_turns(
    console: Console,
    store: ProjectStore,
    prepared: PreparedFidelity,
    result: DualReport,
    judge: JudgeConfig,
) -> None:
    """Print every turn as plain labelled lines: call, reply, differences and measurements.

    The memory row appears only for runs that deliver memory or retrieved evidence.

    Args:
        console: Existing terminal rendering owner.
        store: Project holding the saved evaluation inputs and session.
        prepared: Exact saved registration.
        result: Validated saved dual report.
        judge: Registered rubric supplying criterion identities.
    """
    saved = saved_turns(store, prepared, result)
    if saved is None:
        return
    evidence, references, turns = saved
    decisions = {int(str(row["step"])): row for row in fidelity_decision_trace(store, prepared)}
    criteria = [dimension.dimension_id for dimension in judge.definition.dimensions]
    delivers_memory = prepared.spec.memory is not None or prepared.spec.retrieval is not None
    for step, turn in zip(evidence.steps, turns, strict=True):
        reference = references.steps[step.step] if references.steps else None
        console.print(Text(f"\nturn {step.step + 1}  {_call(step.action)}", style="bold"))
        if step.status == "valid":
            _row(console, "simulator", _reply(step.reply))
        else:
            reason = f": {step.error_type}" if step.error_type else ""
            _row(console, "simulator", f"no reply ({step.status}{reason})", style="red")
        if reference is not None and reference.reply_available and step.status == "valid":
            differences = field_differences(step.reply, reference.reply)
            for index, row in enumerate(differences[:SHOWN_DIFFERENCES]):
                _row(console, "differs" if index == 0 else "", _difference(row), style="red")
            if len(differences) > SHOWN_DIFFERENCES:
                more = len(differences) - SHOWN_DIFFERENCES
                _row(console, "", f"+{more} more differing fields", style="red")
        _row(console, "measured", _measured(turn))
        _row(
            console,
            "judge",
            " · ".join(
                f"{key} {'-' if turn.scores.get(key) is None else turn.scores[key]}"
                for key in criteria
            )
            if turn.scores
            else _no_turn_scores(judge, result),
        )
        decision = decisions.get(step.step)
        if decision is not None and delivers_memory:
            _row(console, "memory", _memory(decision))


def _row(console: Console, label: str, value: str, *, style: str | None = None) -> None:
    """Print one aligned label and value without markup interpretation."""
    line = Text("  " + label.ljust(LABEL), style="dim")
    line.append(value, style=style)
    console.print(line)


def _measured(turn: TurnOutcome) -> str:
    """Exactness, field agreement and divergence for one turn."""
    divergence = turn.divergence
    if divergence is None:
        return "no recorded reply to compare"
    shown = "<0.01" if 0 < divergence < 0.005 else f"{divergence:.2f}"
    return (
        f"exact {'yes' if divergence == 0 else 'no'} · fields "
        f"{turn.fields_matched}/{turn.fields_total} · divergence {shown}"
    )


def _no_turn_scores(judge: JudgeConfig, result: DualReport) -> str:
    """Why a turn has no judge scores: the judge did not finish, or scores whole runs only."""
    if result.judge_status != "completed":
        return f"none (judge {result.judge_status})"
    return "per-turn scores not returned" if judge.per_turn else "whole run only"


def _memory(decision: JsonObject) -> str:
    """Memory delivered to the turn's prediction and whether it produced a note."""
    selected = decision.get("memory_selected")
    ids = [str(key) for key in selected] if isinstance(selected, list) else []
    notes = sum(key.startswith("session-research") for key in ids)
    past = len(ids) - notes
    text = f"{notes} note{'' if notes == 1 else 's'} · {past} past step{'' if past == 1 else 's'}"
    return text + (" · note written" if decision.get("research_id") else "")


def _call(action: JsonObject) -> str:
    """Render a tool call as name(key=value, ...)."""
    arguments = action.get("arguments")
    pairs = (
        ", ".join(
            f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in arguments.items()
        )
        if isinstance(arguments, dict)
        else ""
    )
    return f"{action.get('name')}({pairs})"


def _reply(reply: JsonValue) -> str:
    """Compact reply text, cut at a readable length."""
    text = json.dumps(reply, ensure_ascii=False)
    return text if len(text) <= REPLY_CHARS else text[: REPLY_CHARS - 1] + "…"


def _difference(row: JsonObject) -> str:
    """One divergent field: what the simulator said against the recorded reply."""
    path = str(row["path"]) or "(reply)"
    simulated = "absent" if row["actual_missing"] else json.dumps(row["actual"], ensure_ascii=False)
    recorded = (
        "absent" if row["expected_missing"] else json.dumps(row["expected"], ensure_ascii=False)
    )
    return f"{path}: {simulated} · recorded {recorded}"

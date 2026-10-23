"""Simulation fidelity in the existing evaluation terminal conventions."""

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from exp.cli.evaluation.fidelity_turns import render_turns, saved_turns, turn_marks
from exp.cli.evaluation.view import heading
from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.picker import PickerOption, choose_one
from exp.cli.shared.progress import progress_display
from exp.cli.shared.theme import EXP_THEME
from exp.common.evaluations.reconstruction import DualReport, JudgeConfig
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.fidelity import (
    FidelitySpec,
    PreparedFidelity,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
    saved_fidelity_report,
)
from exp.optimize.evaluation.fidelity_memory import (
    fidelity_decision_trace,
    fidelity_memory_plan,
    saved_maintenance_receipt,
    saved_memory_receipt,
)


def run_fidelity(
    project: str,
    spec_path: Path,
    *,
    root: Path,
    yes: bool,
    non_interactive: bool,
    dry_run: bool,
    verbose: bool = False,
) -> None:
    """Review and execute a pinned fidelity workload or inspect its exact saved result.

    Args:
        project: Existing trace-built simulation project.
        spec_path: Typed workload exported by the experiment/evaluation owner.
        root: Existing project and artifact root.
        yes: Explicit authorization after the undefined-cost warning.
        non_interactive: Disable terminal questions and menus.
        dry_run: Freeze and show the plan without credentials or provider calls.
        verbose: Also print the full configuration, every turn and every criterion.
    """
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        spec = FidelitySpec.model_validate_json(spec_path.read_bytes())
        store = ProjectStore(root, project)
        prepared = prepare_fidelity_evaluation(
            store, spec, code_revision=installed_release_revision()
        )
        previous = saved_fidelity_report(store, prepared)
        heading(console, project, "Simulation fidelity")
        console.print(Text(_brief(spec), style="dim"))
        if verbose or dry_run:
            _configuration(console, store, spec, prepared)
        if dry_run:
            console.print(
                "Prepared without provider calls. Run the same command without --dry-run."
            )
            return
        interactive = not non_interactive and can_prompt(console)
        if previous is None:
            if not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=None,
                cost_is_upper_bound=False,
                command=f"exp eval {project} --fidelity {spec_path}",
                non_interactive=not interactive,
            ):
                return
            with progress_display(console) as progress:
                result = execute_fidelity_evaluation(store, prepared, progress=progress)
        else:
            console.print(Text("Saved result. No provider calls.", style="dim"))
            # The lifecycle owner replays the report and completes any interrupted publication.
            result = execute_fidelity_evaluation(store, prepared)
        render_summary(console, store, prepared, result, spec)
        if verbose:
            render_turns(console, store, prepared, result, spec.judge)
            console.print()
            render_fidelity(console, result, spec.judge)
            render_memory(console, store, prepared)
        if interactive:
            _results_menu(console, store, prepared, result)


def _configuration(
    console: Console, store: ProjectStore, spec: FidelitySpec, prepared: PreparedFidelity
) -> None:
    """Print the full pinned configuration for verbose and dry runs."""
    console.print(
        Text(f"{len(spec.workload.actions)} supplied actions · {spec.references.capability}")
    )
    console.print(Text(f"Simulation: {spec.world_model.artifact_id}"))
    console.print(Text(f"Judge: {spec.judge_alias} · {spec.judge.calibration}"))
    if spec.retrieval is not None:
        console.print(
            Text(
                f"Retrieval: JEV via {spec.retrieval.jev_alias} · "
                f"policy {spec.retrieval.jev.identity[:12]} · "
                f"memory top {spec.retrieval.memory_top_k}"
            )
        )
    memory = fidelity_memory_plan(store, spec, prepared)
    if memory is not None:
        console.print(Text(f"Memory: {memory}"))
    if spec.online_research is not None:
        console.print(
            Text(
                "Feedback: each native reply is disclosed only after its prediction "
                f"commits ({spec.online_research.feedback_source})"
            )
        )
    console.print(
        Text(
            f"Physical attempts: at most {spec.maximum_physical_attempts}, "
            "including retries and judging"
        )
    )
    console.print(
        Text(
            f"Output: {spec.maximum_output_tokens:,} tokens per call · "
            f"deadline {spec.deadline.isoformat()}"
        )
    )


def _results_menu(
    console: Console, store: ProjectStore, prepared: PreparedFidelity, result: DualReport
) -> None:
    """Offer the saved detail views without provider calls."""
    while True:
        selected = choose_one(
            console,
            title="Results",
            options=(
                PickerOption("inspect", "Inspect first divergence"),
                PickerOption("decisions", "Inspect model decisions and commits"),
                PickerOption("back", "Back"),
            ),
        )
        if not selected.values or selected.values[0] == "back":
            return
        if selected.values[0] == "decisions":
            render_decisions(console, store, prepared)
            continue
        console.print(
            Text(f"First semantic divergence: {result.mechanical.get('first_semantic_divergence')}")
        )
        console.print(Text(f"First hard stop: {result.mechanical.get('first_hard_stop')}"))
        for axis in result.axes:
            console.print(Text(f"{axis.dimension_id}: {axis.rationale}"))


def _brief(spec: FidelitySpec) -> str:
    """One line naming the task, its size, the memory it uses and the judge."""
    parts = [spec.task.task_id, f"{len(spec.workload.actions)} turns"]
    if spec.memory is not None:
        if spec.memory.mode == "learn":
            parts.append(f"learning into {spec.memory.lineage}")
        else:
            held = " · held out" if spec.memory.reserved else ""
            parts.append(f"memory v{spec.memory.checkpoint}{held}")
    parts.append(f"judge {spec.judge_alias}")
    return " · ".join(parts)


def render_summary(
    console: Console,
    store: ProjectStore,
    prepared: PreparedFidelity,
    result: DualReport,
    spec: FidelitySpec,
) -> None:
    """Print the four-line result: replies, judge, memory and where the detail lives.

    Args:
        console: Existing terminal rendering owner.
        store: Project holding the saved evaluation and memory receipts.
        prepared: Exact saved registration.
        result: Validated saved dual report.
        spec: Registered workload.
    """
    saved = saved_turns(store, prepared, result)
    rows = result.mechanical.get("steps")
    exact = sum(
        isinstance(row, dict) and row.get("trajectory_reply_exact") is True
        for row in (rows if isinstance(rows, list) else [])
    )
    replies = Text(f"{exact}/{result.planned_steps}   ")
    if saved is not None:
        replies.append_text(turn_marks(saved[2]))
    console.print()
    _summary_row(console, "Replies exact", replies)
    _summary_row(console, "Judge", Text(_judge_line(result)))
    memory = _memory_line(store, prepared, spec)
    if memory is not None:
        _summary_row(console, "Memory", Text(memory))
    _summary_row(console, "Report", Text(f"{prepared.id}-report · --verbose shows every turn"))


def _summary_row(console: Console, label: str, value: Text) -> None:
    """One aligned result row."""
    line = Text("  " + label.ljust(15), style="dim")
    line.append_text(value)
    console.print(line)


def _judge_line(result: DualReport) -> str:
    """Mean assessed score, or the explicit reason there is none."""
    if result.judge_status != "completed":
        reason = f" ({result.judge_failure})" if result.judge_failure else ""
        return f"{result.judge_status}{reason}"
    scores = [axis.score for axis in result.axes if axis.score is not None]
    if not scores:
        return "no criterion assessable"
    return f"{sum(scores) / len(scores):.1f}/5 mean of {len(scores)} criteria"


def _memory_line(store: ProjectStore, prepared: PreparedFidelity, spec: FidelitySpec) -> str | None:
    """What the run did to memory, in one line."""
    if spec.memory is None:
        return None
    if spec.memory.mode != "learn":
        return f"read v{spec.memory.checkpoint}, unchanged"
    receipt = saved_memory_receipt(store, prepared)
    if receipt is None:
        return "not published"
    if not receipt.get("published"):
        return f"not advanced ({receipt.get('reason')})"
    line = (
        f"v{receipt['parent_version']} → v{receipt['version']} · "
        f"{_plural(_count(receipt.get('research_ids')), 'new note')}"
    )
    maintenance = saved_maintenance_receipt(store, prepared)
    if maintenance is None:
        return line
    if maintenance.get("error_type"):
        return line + f" · compression stopped ({maintenance['error_type']})"
    line += f" · {_plural(_count(maintenance.get('built')), 'block')} compressed"
    refused = maintenance.get("refused")
    if isinstance(refused, list) and refused:
        line += " · refused " + ", ".join(str(block) for block in refused)
    ambiguous = maintenance.get("ambiguous")
    if isinstance(ambiguous, list) and ambiguous:
        line += " · unresolved purchase " + ", ".join(str(block) for block in ambiguous)
    return line


def render_fidelity(console: Console, result: DualReport, judge: JudgeConfig) -> None:
    """Render existing measurements without combining ratings with mechanical truth.

    Args:
        console: Existing terminal rendering owner.
        result: Validated saved dual report, including all unavailable outcomes.
        judge: Exact registered rubric supplying human-readable criterion names.
    """
    rows = result.mechanical.get("steps")
    if not isinstance(rows, list):
        raise ValueError("fidelity report is missing its planned mechanical inventory")
    exact = sum(isinstance(row, dict) and row.get("trajectory_reply_exact") is True for row in rows)
    unavailable = sum(
        isinstance(row, dict) and row.get("trajectory_reply_exact") is None for row in rows
    )
    console.print(
        Text(
            f"\nExact recorded replies: {exact}/{result.planned_steps} planned "
            f"({unavailable} lack a reply reference)"
        )
    )
    console.print(
        Text(
            "Whole trajectory exact: "
            + ("yes" if exact == result.planned_steps and result.planned_steps else "no")
        )
    )
    console.print(
        Text(
            f"Valid: {result.mechanical.get('valid_steps')} · "
            f"failed: {result.mechanical.get('failed_steps')} · "
            f"blocked: {result.mechanical.get('blocked_steps')}"
        )
    )
    state = result.mechanical.get("workflow_joint_exact")
    console.print(
        Text(f"Complete reply-and-state workflow: {'unmeasured' if state is None else state}")
    )
    console.print(Text(f"LLM judge: {result.judge_status}"))
    table = Table(box=None, padding=(0, 2))
    table.add_column("Criterion")
    table.add_column("Score /5", justify="right")
    table.add_column("Status")
    names = {dimension.dimension_id: dimension.name for dimension in judge.definition.dimensions}
    for axis in result.axes:
        table.add_row(
            Text(names[axis.dimension_id]),
            str(axis.score) if axis.score is not None else "-",
            Text(axis.status),
        )
    console.print(table)
    console.print(
        "[dim]LLM ratings are separate from exactness. "
        "Fixed actions; actor equivalence is not measured.[/dim]"
    )


def render_memory(console: Console, store: ProjectStore, prepared: PreparedFidelity) -> None:
    """Show the deterministic memory publication outcome for a learning trajectory.

    Args:
        console: Existing terminal rendering owner.
        store: Project holding the saved memory receipt.
        prepared: Exact saved registration.
    """
    receipt = saved_memory_receipt(store, prepared)
    if receipt is None:
        return
    if receipt.get("published"):
        console.print(
            Text(
                f"Memory: published {receipt['lineage']} v{receipt['version']} from "
                f"v{receipt['parent_version']} · consumed trajectories "
                f"{receipt['consumed_trajectories']} · "
                f"{_count(receipt.get('research_ids'))} unresolved research records"
            )
        )
    else:
        console.print(Text(f"Memory: not advanced ({receipt.get('reason')})"))
    maintenance = saved_maintenance_receipt(store, prepared)
    if maintenance is None:
        return
    if maintenance.get("error_type"):
        console.print(Text(f"Compression: stopped ({maintenance['error_type']})"))
        return
    refused = maintenance.get("refused")
    line = (
        f"Compression: built {_count(maintenance.get('built'))} · "
        f"pending {maintenance.get('pending')}"
    )
    if isinstance(refused, list) and refused:
        line += " · refused " + ", ".join(str(block) for block in refused)
    ambiguous = maintenance.get("ambiguous")
    if isinstance(ambiguous, list) and ambiguous:
        line += " · unresolved purchase " + ", ".join(str(block) for block in ambiguous)
    console.print(Text(line))


def render_decisions(console: Console, store: ProjectStore, prepared: PreparedFidelity) -> None:
    """List each model decision beside the deterministic commit that bounded it.

    Args:
        console: Existing terminal rendering owner.
        store: Project holding the durable session and saved receipts.
        prepared: Exact saved registration.
    """
    table = Table(box=None, padding=(0, 1))
    for column in ("Step", "JEV memory IDs", "GLM proposal", "Commit", "Reply", "Investigation"):
        table.add_column(column)
    for row in fidelity_decision_trace(store, prepared):
        proposal = row.get("proposal")
        matched = row.get("reply_matched")
        chosen = row.get("memory_selected")
        selected = chosen if isinstance(chosen, list) else []
        table.add_row(
            str(row["step"]),
            Text(", ".join(str(key) for key in selected) or "-"),
            Text(str(proposal.get("artifact_id")) if isinstance(proposal, dict) else "-"),
            f"v{row['committed_version']}",
            "-" if matched is None else ("exact" if matched else "wrong"),
            Text(str(row.get("research_id") or "-")),
        )
    console.print(table)
    console.print(
        "[dim]Model decisions: JEV selections, GLM proposals and investigations. "
        "Code decisions: guarded commits, feedback order and memory publication.[/dim]"
    )


def _plural(count: int, noun: str) -> str:
    """Count and noun, pluralized with a trailing s."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _count(value: object) -> int:
    """Count a saved list field, treating an absent field as empty."""
    return len(value) if isinstance(value, list) else 0

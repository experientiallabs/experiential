"""Terminal presentation of learning from traces, shared by ``exp build`` and the audit.

The build learns every trace; ``exp eval --audit`` learns fit traces only before its panel.
Both show the same preflight, ask the same spend consent and report the same outcome here, so
neither command depends on the other's module.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from exp.cli.evaluation.replay_view import stop_reason
from exp.cli.evaluation.spend_ceiling import ceiling_line, enforced_ceiling, inventory_terms
from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.progress import progress_display
from exp.common.core.artifacts import stable_id
from exp.common.progress import ProgressHook
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.model_costs import fidelity_costs
from exp.optimize.evaluation.fidelity.replay import DEADLINE
from exp.optimize.evaluation.learning.retirement import (
    RETIREMENT_FAILURES,
    SystemicRefusalError,
    retired_traces,
)
from exp.optimize.evaluation.learning.traces import (
    LearningOutcome,
    Scope,
    learn_from_traces,
    learning_cases,
    learning_curve,
    learning_plan,
    learning_progress,
    learning_spec,
)
from exp.simulation.evaluation.workflow import INFRASTRUCTURE_STOPS, deterministic_refusal


def run_learning(
    console: Console,
    store: ProjectStore,
    *,
    project: str,
    root: Path,
    recall_alias: str,
    maximum_attempts: int | None,
    yes: bool,
    non_interactive: bool,
    code_revision: str,
    retry_incomplete: bool = False,
    hold_out: bool = False,
    terminal: Console | None = None,
    progress: ProgressHook | None = None,
    pause: Callable[[], AbstractContextManager[None]] | None = None,
    command: str | None = None,
    evidence_reading: bool | None = None,
    dream_every: int | None = None,
) -> LearningOutcome | None:
    """Write the learning preflight, ask for spend consent, then learn from every trace.

    Args:
        console: Where the full preflight and outcome are written: the calling command's
            terminal, or its report file when ``terminal`` is given.
        store: Built project.
        project: Project name, for the replay command.
        root: Local ``.exp`` root holding the consent settings.
        recall_alias: Native decision alias recalling memory records.
        maximum_attempts: Physical-attempt budget; None allows every trace's full ceiling.
        yes: Explicit authorization of the undefined-cost warning.
        non_interactive: Disable terminal questions.
        code_revision: Exact producing implementation identity.
        retry_incomplete: Resume an interrupted trace in place: saved work replays, and a
            request lost in flight is sent again and reported as an unknown-outcome retry.
        hold_out: Learn fit traces only, keeping held-out traces for an independent panel.
        terminal: The user's terminal, for the spend question and errors; defaults to
            ``console``.
        progress: The command's single progress line; without it, learning owns a display.
        pause: Releases that progress line while the spend question is asked.
        command: Exact command a refusal names; defaults to the bare command.
        evidence_reading: Enable bounded readers, or follow the recorded learning plan.
        dream_every: Consolidation interval, or follow the recorded learning plan.

    Returns:
        The outcome, or None when the user declined the spend.

    Raises:
        typer.Exit: A provider request failed; completed traces stay saved.
    """
    screen = terminal or console
    scope: Scope = "fit" if hold_out else "all"
    plan = learning_plan(
        store, recall_alias, scope, evidence_reading=evidence_reading, dream_every=dream_every
    )
    cases = learning_cases(store, scope)
    now = datetime.now(UTC)
    specs = tuple(learning_spec(store, plan, case, deadline=now + DEADLINE) for case in cases)
    ceiling = sum(spec.maximum_physical_attempts for spec in specs)
    budget = ceiling if maximum_attempts is None else maximum_attempts
    console.print("Learning preflight")
    kept = "fit traces; held-out traces kept out" if hold_out else "traces, every trace"
    console.print(f"  [dim]traces[/dim]       {len(cases)} {kept}, in canonical dataset order")
    console.print(
        "  [dim]measured[/dim]     test-then-train: whole trace predicted on memory pinned "
        "before it"
    )
    console.print(f"  [dim]simulator[/dim]    hybrid {plan.world_model.artifact_id}")
    console.print(f"  [dim]memory[/dim]       {plan.lineage} · recall {plan.recall.jev_alias}")
    if plan.reading is not None:
        console.print(
            f"  [dim]evidence[/dim]     objective-driven investigations, at most "
            f"{plan.reading.additional_calls} extra calls per discrepancy"
        )
    if plan.dreaming is not None:
        console.print(
            f"  [dim]dreaming[/dim]     every {plan.dreaming.interval} traces and at completion, "
            f"at most {plan.dreaming.calls} extra logical calls per pass"
        )
    console.print(
        f"  [dim]attempts[/dim]     at most {budget:,} physical attempts, retries included "
        f"(all traces need at most {ceiling:,})"
    )
    # One ceiling over every trace of the plan: priced roles in dollars, unpriced ones within
    # their displayed attempts, which only an explicit confirmation authorizes.
    known, unpriced = inventory_terms(
        tuple(row for spec in specs for row in fidelity_costs(store, spec))
    )
    unpriced = min(unpriced, budget)  # the physical-attempt budget bounds every role
    operation = stable_id("learning", {"lineage": plan.lineage, "scope": scope})
    shown = command or (f"exp eval {project} --audit N" if hold_out else f"exp build {project}")
    # A pass over saved results asks nothing, so it authorizes no further spend.
    asked = False
    scored, _ = learning_progress(store, plan)
    retired = retired_traces(store, plan.lineage, tuple(case.trace_id for case in cases))
    if len(scored) + len(retired) == len(cases) and all(learned for *_, learned in scored):
        # Every trace is already learned: the pass replays saved runs without provider calls.
        console.print(Text("Saved results. No provider calls.", style="dim"))
    else:
        with pause() if pause is not None else nullcontext():
            if screen is not console:
                screen.print(
                    f"Learning {len(cases)} {'fit traces' if hold_out else 'traces'} with "
                    f"memory recall {plan.recall.jev_alias}: at most {budget:,} model attempts"
                )
            screen.print(Text(ceiling_line(known, unpriced), style="dim"))
            asked = True
            consented = require_spend_consent(
                screen,
                root=root,
                yes=yes,
                estimated_cost_usd=None if unpriced else known,
                cost_is_upper_bound=not unpriced,
                command=shown,
                non_interactive=non_interactive or not can_prompt(screen),
            )
        if not consented:
            return None
    try:
        with (
            enforced_ceiling(
                store,
                operation,
                stage=operation,
                ceiling_usd=known if asked else None,
                unpriced_attempts=unpriced,
                command=shown,
            ),
            nullcontext(progress) if progress is not None else progress_display(console) as hook,
        ):
            outcome = learn_from_traces(
                store,
                recall_alias=recall_alias,
                scope=scope,
                maximum_attempts=budget,
                code_revision=code_revision,
                clock=lambda: datetime.now(UTC),
                progress=hook,
                retry_incomplete=retry_incomplete,
                evidence_reading=evidence_reading,
                dream_every=dream_every,
            )
    except SystemicRefusalError as exc:
        screen.print(f"[red]error[/red] {exc}")
        screen.print("Learned and retired traces are saved; nothing more was retired.")
        raise typer.Exit(code=1) from exc
    except INFRASTRUCTURE_STOPS as exc:
        screen.print(f"[red]error[/red] {stop_reason(exc)}")
        refusal = deterministic_refusal(exc)
        if refusal is not None:
            screen.print(
                f"The gateway refused this exact request ({refusal.error_code}). Resending it "
                f"with --retry-incomplete retires the trace as not learned after "
                f"{RETIREMENT_FAILURES} identical refusals, then learning continues."
            )
        screen.print(
            "Learned traces are saved. The stopped trace is never bought again silently: "
            "run the same command with --retry-incomplete to resume it."
        )
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]✓[/green] learned {outcome.learned}/{outcome.traces} "
        f"{'fit traces' if hold_out else 'traces'} · "
        f"memory v{outcome.head_version} · {outcome.attempts_used:,} physical attempts"
    )
    if outcome.incomplete:
        console.print(
            f"[yellow]warning[/yellow] {outcome.incomplete} trace(s) have incomplete saved "
            "results and were not learned. Saved predictions are never re-purchased. "
            "--retry-incomplete resumes interrupted runs, not finalized reports."
        )
    for row in retired_traces(store, plan.lineage, tuple(case.trace_id for case in cases)):
        console.print(
            f"[yellow]retired[/yellow] trace {row.trace_id} ({row.cause}): {row.reason}; "
            "not learned and not scored"
        )
    if outcome.stopped is not None:
        console.print(f"[yellow]warning[/yellow] {outcome.stopped}")
    curve = learning_curve(store, scope)
    if curve is not None:
        console.print(Text(curve.headline()))
        console.print(Text(curve.line()))
        if not hold_out:
            console.print(
                Text(
                    "This measures learning progress over an evolving memory. For the final "
                    f"simulator on unseen traces, run exp eval {project} --audit N.",
                    style="dim",
                )
            )
    return outcome

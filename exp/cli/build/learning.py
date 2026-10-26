"""The learning stage of ``exp build``: memory grown trace by trace from the traces.

Learning is the build's default pipeline. It needs a native decision model to recall memory:
the catalog's ``memory_recall`` role, or else the catalog's best SystemOne decision model.
"""

import io
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from exp.cli.build.summary import save_details
from exp.cli.evaluation.miss_retest import RetestOutcome, run_miss_retest
from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.progress import ProgressDisplay, progress_display
from exp.cli.shared.theme import EXP_THEME
from exp.common.models import ModelCatalog, load_model_catalog
from exp.common.progress import ProgressHook
from exp.common.project import ProjectStore
from exp.optimize.evaluation.trace_learning import (
    LearningCurve,
    LearningOutcome,
    Scope,
    learn_from_traces,
    learned_runs,
    learning_cases,
    learning_curve,
    learning_plan,
    learning_progress,
    learning_spec,
)
from exp.optimize.evaluation.trace_replay import DEADLINE
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.models.structured_completion import CompletionNotDispatched
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

# Preferred native decision models for memory recall, best first.
_RECALL_PREFERENCE = ("clef-flash", "clef", "jev")
DECISION_PROVIDER = "typesafe"


def resolve_memory_recall(catalog: ModelCatalog) -> str | None:
    """Choose the decision alias that recalls simulator memory, without a flag.

    The ``memory_recall`` role wins when set. Otherwise the catalog's SystemOne decision models
    are ranked by ``_RECALL_PREFERENCE`` (Clef Flash first), then by alias.

    Args:
        catalog: The project's model catalog.

    Returns:
        The alias, or None when the catalog has no native decision model.

    Raises:
        ValueError: The ``memory_recall`` role names an alias that is not a decision model.
    """

    def decision(alias: str) -> bool:
        record = catalog.models.get(alias)
        connection = None if record is None else catalog.connections.get(record.connection)
        return connection is not None and connection.provider == DECISION_PROVIDER

    chosen = catalog.roles.memory_recall
    if chosen is not None:
        if not decision(chosen):
            raise ValueError(
                f"memory_recall role {chosen!r} is not a native decision model "
                f"({DECISION_PROVIDER} connection)"
            )
        return chosen

    def rank(alias: str) -> tuple[int, str]:
        model = catalog.models[alias].model.casefold()
        hits = [i for i, name in enumerate(_RECALL_PREFERENCE) if name in model]
        return (min(hits) if hits else len(_RECALL_PREFERENCE), alias)

    candidates = sorted((a for a in catalog.models if decision(a)), key=rank)
    return candidates[0] if candidates else None


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
) -> LearningOutcome | None:
    """Write the learning preflight, ask for spend consent, then learn from every trace.

    Args:
        console: Where the full preflight and outcome are written: the terminal, or the
            command's report file when ``terminal`` is given.
        store: Built project.
        project: Project name, for the replay command.
        root: Local ``.exp`` root holding the consent settings.
        recall_alias: Native decision alias recalling memory records.
        maximum_attempts: Physical-attempt budget; None allows every trace's full ceiling.
        yes: Explicit authorization of the undefined-cost warning.
        non_interactive: Disable terminal questions.
        code_revision: Exact producing implementation identity.
        retry_incomplete: Replace an interrupted trace, buying its calls again.
        hold_out: Learn fit traces only, keeping held-out traces for an independent panel.
        terminal: The user's terminal, for the spend question and errors; defaults to
            ``console``.
        progress: The command's single progress line; without it, learning owns a display.
        pause: Releases that progress line while the spend question is asked.
        command: Exact command a refusal names; defaults to the bare command.

    Returns:
        The outcome, or None when the user declined the spend.

    Raises:
        typer.Exit: A provider request failed; completed traces stay saved.
    """
    screen = terminal or console
    scope: Scope = "fit" if hold_out else "all"
    plan = learning_plan(store, recall_alias, scope)
    cases = learning_cases(store, scope)
    now = datetime.now(UTC)
    ceiling = sum(
        learning_spec(store, plan, case, deadline=now + DEADLINE).maximum_physical_attempts
        for case in cases
    )
    budget = ceiling if maximum_attempts is None else maximum_attempts
    console.print("Learning preflight")
    kept = "fit traces; held-out traces kept out" if hold_out else "traces, every trace"
    console.print(f"  [dim]traces[/dim]       {len(cases)} {kept}, in export order")
    console.print(
        "  [dim]measured[/dim]     test-then-train: whole trace predicted on memory pinned "
        "before it"
    )
    console.print(f"  [dim]simulator[/dim]    hybrid {plan.world_model.artifact_id}")
    console.print(f"  [dim]memory[/dim]       {plan.lineage} · recall {plan.recall.jev_alias}")
    console.print(
        f"  [dim]attempts[/dim]     at most {budget:,} physical attempts, retries included "
        f"(all traces need at most {ceiling:,})"
    )
    scored, _ = learning_progress(store, plan)
    if len(scored) == len(cases) and all(learned for *_, learned in scored):
        # Every trace is already learned: the pass replays saved runs without provider calls.
        console.print(Text("Saved results. No provider calls.", style="dim"))
    else:
        with pause() if pause is not None else nullcontext():
            if screen is not console:
                screen.print(
                    f"Learning {len(cases)} {'fit traces' if hold_out else 'traces'} with "
                    f"memory recall {plan.recall.jev_alias}: at most {budget:,} model attempts"
                )
            consented = require_spend_consent(
                screen,
                root=root,
                yes=yes,
                estimated_cost_usd=None,
                cost_is_upper_bound=False,
                command=command
                or (f"exp eval {project} --audit N" if hold_out else f"exp build {project}"),
                non_interactive=non_interactive or not can_prompt(screen),
            )
        if not consented:
            return None
    try:
        with nullcontext(progress) if progress is not None else progress_display(console) as hook:
            outcome = learn_from_traces(
                store,
                recall_alias=recall_alias,
                scope=scope,
                maximum_attempts=budget,
                code_revision=code_revision,
                clock=lambda: datetime.now(UTC),
                progress=hook,
                retry_incomplete=retry_incomplete,
            )
    except (ProviderTransportError, CompletionNotDispatched) as exc:
        screen.print(f"[red]error[/red] a provider request failed: {exc}")
        screen.print(
            "Learned traces are saved. The stopped trace is never bought again silently: "
            "run the same command with --retry-incomplete to replace it."
        )
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]✓[/green] learned {outcome.learned}/{outcome.traces} "
        f"{'fit traces' if hold_out else 'traces'} · "
        f"memory v{outcome.head_version} · {outcome.attempts_used:,} physical attempts"
    )
    if outcome.incomplete:
        console.print(
            f"[yellow]warning[/yellow] {outcome.incomplete} trace(s) stopped before completing "
            "on every allowed attempt and were not learned"
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


@dataclass(frozen=True)
class BuildLearning:
    """What the build's learning stage did, for the build's summary lines.

    Attributes:
        recall: Memory-recall alias, or None when the catalog has no decision model.
        outcome: The learning pass, or None when it did not run or the spend was declined.
        curve: Learning progress over the build's memory.
        retest: The miss retest, or None when no learned trace had a miss.
    """

    recall: str | None = None
    outcome: LearningOutcome | None = None
    curve: LearningCurve | None = None
    retest: RetestOutcome | None = None


def learn_and_retest(
    console: Console,
    store: ProjectStore,
    *,
    project: str,
    root: Path,
    yes: bool,
    non_interactive: bool,
    code_revision: str,
    retry_incomplete: bool = False,
    details: Console | None = None,
    pipeline: ProgressDisplay | None = None,
    command: str | None = None,
) -> BuildLearning:
    """The build's default pipeline: learn from every trace, then retest the misses.

    Args:
        console: Build command's terminal owner.
        store: Built project.
        project: Project name.
        root: Local ``.exp`` root holding the consent settings.
        yes: Explicit authorization of the undefined-cost warning.
        non_interactive: Disable terminal questions.
        code_revision: Exact producing implementation identity.
        retry_incomplete: Replace an interrupted trace, buying its calls again.
        details: Where the full learning and retest output is written; defaults to ``console``.
        pipeline: The build's single progress line, with ``learning`` and ``retest`` phases.
        command: The exact build command a refusal tells the operator to run again.

    Returns:
        What ran, for the build's summary.
    """
    details = details or console
    catalog = load_model_catalog(store.model_catalog_path)
    recall = resolve_memory_recall(catalog)
    if recall is None:
        console.print(
            "[yellow]note[/yellow] the simulator was built without learning: learning needs a "
            "native decision model to recall memory (for example cloudflare/clef-flash on the "
            "Experiential gateway). Add one to the model catalog and run the build again."
        )
        return BuildLearning()
    tool_less = _tool_less_world_model(store, catalog, recall)
    if tool_less is not None:
        console.print(
            f"[yellow]note[/yellow] the simulator was built without learning: learning needs a "
            f"tool-capable world model, and {tool_less!r} declares no tool support. Choose a "
            "world model with tool support and run the build again."
        )
        return BuildLearning(recall=recall)
    outcome = run_learning(
        details,
        store,
        project=project,
        root=root,
        recall_alias=recall,
        maximum_attempts=None,
        yes=yes,
        non_interactive=non_interactive,
        code_revision=code_revision,
        retry_incomplete=retry_incomplete,
        terminal=console,
        progress=None if pipeline is None else pipeline.phase("learning"),
        pause=None if pipeline is None else pipeline.paused,
        command=command,
    )
    if outcome is None:
        return BuildLearning(recall=recall)
    curve = learning_curve(store, "all")
    plan = learning_plan(store, scope="all")
    missed = [result for _, _, result in learned_runs(store, plan) if result.exact < result.steps]
    if not missed:
        if pipeline is not None:
            pipeline.complete("retest")
        return BuildLearning(recall=recall, outcome=outcome, curve=curve)
    details.print()
    retest = run_miss_retest(
        project,
        len(missed),
        root=root,
        yes=yes,
        non_interactive=non_interactive,
        dry_run=False,
        retry_incomplete=retry_incomplete,
        details=details,
        terminal=console,
        progress=None if pipeline is None else pipeline.phase("retest"),
        pause=None if pipeline is None else pipeline.paused,
        command=command,
    )
    return BuildLearning(recall=recall, outcome=outcome, curve=curve, retest=retest)


def learn_with_report(
    console: Console,
    store: ProjectStore,
    *,
    project: str,
    root: Path,
    yes: bool,
    non_interactive: bool,
    code_revision: str,
    retry_incomplete: bool = False,
    pipeline: ProgressDisplay | None = None,
    command: str | None = None,
) -> tuple[BuildLearning, Path | None]:
    """Run the build's learning stage and save its full output as the learning report.

    Args:
        console: Build command's terminal owner.
        store: Built project.
        project: Project name.
        root: Local ``.exp`` root holding the consent settings.
        yes: Explicit authorization of the undefined-cost warning.
        non_interactive: Disable terminal questions.
        code_revision: Exact producing implementation identity.
        retry_incomplete: Replace an interrupted trace, buying its calls again.
        pipeline: The build's single progress line, with ``learning`` and ``retest`` phases.
        command: The exact build command a refusal tells the operator to run again.

    Returns:
        What ran, and ``runtime/learning/<lineage>/report.txt`` when learning ran.
    """
    buffer = io.StringIO()
    details = Console(file=buffer, width=110, theme=EXP_THEME, no_color=True)
    learning = learn_and_retest(
        console,
        store,
        project=project,
        root=root,
        yes=yes,
        non_interactive=non_interactive,
        code_revision=code_revision,
        retry_incomplete=retry_incomplete,
        details=details,
        pipeline=pipeline,
        command=command,
    )
    if learning.outcome is None:
        return learning, None
    return learning, save_details(
        store, "learning", learning.outcome.plan.lineage, buffer.getvalue()
    )


def _tool_less_world_model(store: ProjectStore, catalog: ModelCatalog, recall: str) -> str | None:
    """The build's world-model alias when it declares no tool support, which trace replay needs."""
    plan = learning_plan(store, recall, "all")
    alias = load_grounded_world_model_artifact(store.artifacts, plan.world_model).model_alias
    _, capabilities = RuntimeModelCatalog(catalog).snapshot(alias)
    return None if capabilities.supports_tools else alias

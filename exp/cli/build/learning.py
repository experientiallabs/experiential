"""The learning stage of ``exp build``: memory grown trace by trace from the traces.

Learning is the build's default pipeline. It needs a native decision model to recall memory
(:func:`exp.optimize.evaluation.learning.traces.resolve_memory_recall`); the learning and retest
presentation it composes is shared with ``exp eval`` and lives there.
"""

import io
from pathlib import Path

from rich.console import Console

from exp.cli.evaluation.learning import run_learning
from exp.cli.evaluation.learning_summary import LearningSummary, save_details
from exp.cli.evaluation.miss_retest import run_miss_retest
from exp.cli.shared.progress import ProgressDisplay
from exp.cli.shared.theme import EXP_THEME
from exp.common.models import ModelCatalog, load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.learning.recall import resolve_memory_recall
from exp.optimize.evaluation.learning.traces import learned_runs, learning_curve, learning_plan
from exp.runtime.models import RuntimeModelCatalog
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


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
    evidence_reading: bool | None = None,
    dream_every: int | None = None,
) -> LearningSummary:
    """The build's default pipeline: learn from every trace, then retest the misses.

    Args:
        console: Build command's terminal owner.
        store: Built project.
        project: Project name.
        root: Local ``.exp`` root holding the consent settings.
        yes: Explicit authorization of the undefined-cost warning.
        non_interactive: Disable terminal questions.
        code_revision: Exact producing implementation identity.
        retry_incomplete: Resume an interrupted trace in place.
        details: Where the full learning and retest output is written; defaults to ``console``.
        pipeline: The build's single progress line, with ``learning`` and ``retest`` phases.
        command: The exact build command a refusal tells the operator to run again.
        evidence_reading: Enable bounded readers, or follow the recorded learning plan.
        dream_every: Consolidation interval, or follow the recorded learning plan.

    Returns:
        What ran, for the build's summary.
    """
    details = details or console
    catalog = load_model_catalog(store.model_catalog_path)
    recall = resolve_memory_recall(catalog)
    if recall is None:
        if evidence_reading or dream_every is not None:
            raise ValueError(
                "evidence reading and dreaming require a native decision model for memory recall; "
                "add one to the model catalog and run the build again"
            )
        console.print(
            "[yellow]note[/yellow] the simulator was built without learning: learning needs a "
            "native decision model to recall memory (for example cloudflare/clef-flash on the "
            "Experiential gateway). Add one to the model catalog and run the build again."
        )
        return LearningSummary()
    tool_less = _tool_less_world_model(store, catalog, recall)
    if tool_less is not None:
        console.print(
            f"[yellow]note[/yellow] the simulator was built without learning: learning needs a "
            f"tool-capable world model, and {tool_less!r} declares no tool support. Choose a "
            "world model with tool support and run the build again."
        )
        return LearningSummary(recall=recall)
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
        evidence_reading=evidence_reading,
        dream_every=dream_every,
    )
    if outcome is None:
        return LearningSummary(recall=recall)
    curve = learning_curve(store, "all")
    plan = learning_plan(store, scope="all")
    missed = [result for _, _, result in learned_runs(store, plan) if result.exact < result.steps]
    if not missed:
        if pipeline is not None:
            pipeline.complete("retest")
        return LearningSummary(recall=recall, outcome=outcome, curve=curve)
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
    return LearningSummary(recall=recall, outcome=outcome, curve=curve, retest=retest)


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
    evidence_reading: bool | None = None,
    dream_every: int | None = None,
) -> tuple[LearningSummary, Path | None]:
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
        evidence_reading: Enable bounded readers, or follow the recorded learning plan.
        dream_every: Consolidation interval, or follow the recorded learning plan.

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
        evidence_reading=evidence_reading,
        dream_every=dream_every,
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

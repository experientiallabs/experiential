"""The learning stage of ``exp build``: memory grown trace by trace from the traces.

Learning is the build's default pipeline. It needs a native decision model to recall memory
(:func:`exp.optimize.evaluation.learning.traces.resolve_memory_recall`); the learning and retest
presentation it composes is shared with ``exp eval`` and lives there.
"""

from pathlib import Path

from rich.console import Console

from exp.cli.evaluation.learning import run_learning
from exp.cli.evaluation.learning_summary import LearningSummary
from exp.cli.evaluation.miss_retest import run_miss_retest
from exp.cli.shared.progress import ProgressDisplay
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.learning.traces import (
    learned_runs,
    learning_curve,
    learning_plan,
    resolve_memory_recall,
)


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

    Returns:
        What ran, for the build's summary.
    """
    details = details or console
    recall = resolve_memory_recall(load_model_catalog(store.model_catalog_path))
    if recall is None:
        console.print(
            "[yellow]note[/yellow] the simulator was built without learning: learning needs a "
            "native decision model to recall memory (for example cloudflare/clef-flash on the "
            "Experiential gateway). Add one to the model catalog and run the build again."
        )
        return LearningSummary()
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
    )
    return LearningSummary(recall=recall, outcome=outcome, curve=curve, retest=retest)

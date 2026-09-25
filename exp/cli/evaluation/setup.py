"""Model, repeat, and judge choices over a previously built project."""

from rich.console import Console
from rich.prompt import IntPrompt

from exp.cli.evaluation.view import heading
from exp.cli.shared.picker import PickerOption, choose_many, choose_one
from exp.common.models import ModelCatalog, SetupRole, serves_role
from exp.common.project import ProjectStore
from exp.optimize.evaluation.prepare import ModelEvaluationOptions, read_evaluation_judge
from exp.optimize.evaluation.runs import EvaluationDefaults
from exp.optimize.router.judging.artifacts import read_review_state


def configure_evaluation(
    console: Console, project: ProjectStore, catalog: ModelCatalog, defaults: EvaluationDefaults
) -> EvaluationDefaults | None:
    """Choose models, independent runs, and a judge without building evidence.

    Args:
        console: Terminal used for selection screens and prompts.
        project: Previously built project supplying scenarios and judge provenance.
        catalog: Configured model aliases and verified role capabilities.
        defaults: Saved choices used to preselect models and execution settings.

    Returns:
        Confirmed choices for launch review, or ``None`` when the user cancels.
        This function neither saves defaults nor dispatches provider calls.

    Raises:
        ValueError: The project is unbuilt, fewer than two completion models are
            configured, saved judge evidence is invalid, or selected limits are invalid.
    """
    config = project.load_project()
    if config.build is None or config.models is None:
        raise ValueError(f"run exp build {project.paths.project_id} before evaluating")
    name = project.paths.project_id
    candidates = {
        alias: model
        for alias, model in catalog.models.items()
        if model.capabilities is not None and model.capabilities.supports_completions is not False
    }
    if len(candidates) < 2:
        raise ValueError("configure at least two completion models with exp config providers")
    chosen = choose_many(
        console,
        title="Models",
        minimum=2,
        preselected=defaults.models,
        options=tuple(
            PickerOption(alias, alias, model.connection)
            for alias, model in sorted(candidates.items())
        ),
    )
    if not chosen.values:
        return None
    repeats = IntPrompt.ask("Runs per scenario", default=defaults.options.repeats, console=console)
    selected = read_review_state(project)
    pointer = (
        selected.setup if selected else config.hosted_judge.setup if config.hosted_judge else None
    )
    project_judge = (
        read_evaluation_judge(project, pointer).judge_alias if pointer else config.models.judge
    )
    judge = choose_one(
        console,
        title="Judge",
        default=defaults.judge or project_judge,
        options=tuple(
            PickerOption(
                alias, alias, "project judge" if alias == project_judge else model.connection
            )
            for alias, model in sorted(candidates.items())
            if model.capabilities is not None and serves_role(model.capabilities, SetupRole.JUDGE)
        ),
    )
    if not judge.values:
        return None
    if judge.values[0] != project_judge:
        console.print("[dim]Same project rubric. New judge is provisional.[/dim]")
    options = ModelEvaluationOptions.model_validate(
        {**defaults.options.model_dump(), "repeats": repeats}
    )
    while True:
        console.print(
            f"Runs per scenario: {options.repeats} · "
            f"Parallel rollouts: {options.maximum_concurrency}"
        )
        console.print(
            f"[dim]{options.maximum_steps} steps · "
            f"{options.maximum_rollout_output_tokens:,} output tokens per rollout[/dim]\n"
        )
        action = choose_one(
            console,
            title="Settings",
            options=(
                PickerOption("review", "Review evaluation"),
                PickerOption("settings", "Advanced settings"),
                PickerOption("cancel", "Cancel"),
            ),
        )
        if not action.values or action.values[0] == "cancel":
            return None
        if action.values[0] == "review":
            return defaults.model_copy(
                update={"models": chosen.values, "judge": judge.values[0], "options": options}
            )
        options = _settings(console, name, options)


def _settings(
    console: Console, project: str, options: ModelEvaluationOptions
) -> ModelEvaluationOptions:
    """Edit advanced execution limits only when requested.

    Args:
        console: Terminal used for integer prompts.
        project: Project name displayed in the screen heading.
        options: Current settings supplying defaults for each editable limit.

    Returns:
        Validated settings with updated concurrency, step, and output-token limits.
        Other settings retain their existing values.

    Raises:
        ValueError: An entered limit violates the evaluation option constraints.
    """
    heading(console, project, "Execution settings")
    updates = {
        field: IntPrompt.ask(label, default=getattr(options, field), console=console)
        for field, label in (
            ("maximum_concurrency", "Parallel rollouts"),
            ("maximum_steps", "Steps per rollout"),
            ("maximum_rollout_output_tokens", "Output tokens per rollout"),
        )
    }
    return ModelEvaluationOptions.model_validate({**options.model_dump(), **updates})

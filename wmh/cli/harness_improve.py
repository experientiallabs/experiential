"""Local CLI composition for feedback-directed one-step harness improvement."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.prompt import Confirm

from wmh.agents.default import default_agent
from wmh.agents.optimizer import optimizer_agent
from wmh.agents.project import DEFAULT_PROJECT_TIMEOUT_S, AgentProject
from wmh.cli.harbor_inputs import write_json_atomic
from wmh.cli.model_roles import resolve_opt_in_model_provider, resolve_required_model_config
from wmh.config import ARTIFACT_DIR, ArtifactPaths, WorldModelStore
from wmh.config.store import validate_name
from wmh.core.types import JsonObject
from wmh.engine import load_world_model
from wmh.engine.knowledge import KnowledgeBase
from wmh.evals.gold import GoldJudge
from wmh.evals.tasks import TaskSpec, load_tasks
from wmh.evals.world_model_scorer import WorldModelScorer
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import E2B_TEMPLATE_ENV, resolve_e2b_template
from wmh.harness.improve import (
    DEFAULT_SUITE_MARGIN,
    ImproveGate,
    ImproveOutcome,
    improve_harness,
    verification_results,
)
from wmh.harness.population_archive import write_population_archive
from wmh.harness.project_proposer import CandidateProposer, ProjectCandidateProposer
from wmh.harness.source_tree import HarnessSourceTree
from wmh.harness.store import CHAMPION_ALIAS, HarnessStore
from wmh.providers.base import Provider, ProviderConfig, ToolCallingProvider
from wmh.providers.registry import get_provider
from wmh.scenarios.feedback import synthesize_verification_tasks

_console = Console()


@dataclass(frozen=True)
class ImproveAlreadySatisfied:
    """QA outcome: the seed already passes every synthesized verification task."""

    dropped: tuple[str, ...]
    run_dir: Path


@dataclass(frozen=True)
class HarnessImproveOutcome:
    """Completed local improve-run artifacts and the optional published version."""

    outcome: ImproveOutcome
    saved: HarnessDoc | None
    run_dir: Path
    dropped: tuple[str, ...]


def register(app: typer.Typer) -> None:
    """Register feedback-directed improvement under ``wmh harness``."""
    app.command("improve")(improve_harness_command)


def improve_harness_command(
    name: str = typer.Argument(..., help="Harness name to improve; accepted winners save here."),
    feedback: str | None = typer.Option(
        None,
        "--feedback",
        help="One piece of user feedback naming the capability the harness should gain.",
    ),
    feedback_file: str | None = typer.Option(
        None,
        "--feedback-file",
        help="Read the feedback text from this file instead of --feedback.",
    ),
    tasks_file: str = typer.Option(
        ...,
        "--tasks",
        help="JSONL standard suite the improved harness must not regress on.",
    ),
    model: str = typer.Option(
        ...,
        "--model",
        help="World model that simulates the environment for every evaluation.",
    ),
    verification_count: int = typer.Option(
        3,
        "--verification-count",
        min=1,
        help="Maximum number of must-pass verification tasks synthesized from the feedback.",
    ),
    margin: float = typer.Option(
        DEFAULT_SUITE_MARGIN,
        "--margin",
        help="Allowed relative suite regression, a fraction in [0, 1).",
    ),
    iterations: int = typer.Option(1, min=1, help="Fixed number of singular proposal slots."),
    attempts: int = typer.Option(3, min=1, help="Attempts per exact task and candidate."),
    harness_backend: str = typer.Option(
        "local",
        "--harness-backend",
        help="Where each evaluated harness process runs: local or e2b.",
    ),
    e2b_template: str | None = typer.Option(
        None,
        "--e2b-template",
        envvar=E2B_TEMPLATE_ENV,
        help="Optional template for the E2B proposer project and evaluated processes.",
    ),
    seed: str | None = typer.Option(
        None,
        "--seed",
        help="Stored harness name@ref to improve; default is NAME's champion when it exists, "
        "else the complete built-in Pi agent.",
    ),
    seed_knowledge: bool = typer.Option(
        True,
        "--seed-knowledge/--no-seed-knowledge",
        help="Write the synthesized environment knowledge into the world model's knowledge dir.",
    ),
    result_out: str | None = typer.Option(
        None,
        "--result-out",
        help="Local run directory (default: <root>/runs/<opaque-id>).",
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project artifact root and harness store."),
    yes: bool = typer.Option(False, "--yes", help="Acknowledge the displayed execution matrix."),
) -> None:
    """Improve a harness from one piece of feedback, gated on the suite plus verification.

    The feedback becomes the proposer directive and is synthesized (via settings
    ``models.meta``, which also drives the proposer) into must-pass verification tasks; tasks
    the seed already passes are dropped before any optimization spend. Candidates are scored
    closed-loop against the world model (the agent under test resolves from settings
    ``models.agent`` when set, else the world model's provider). A candidate is promoted only
    when its standard suite score stays within ``--margin`` of the seed AND every surviving
    verification task passes a strict majority of attempts.
    """
    try:
        validate_name(name)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if (feedback is None) == (feedback_file is None):
        raise typer.BadParameter("provide exactly one of --feedback or --feedback-file")
    if feedback_file is not None:
        try:
            feedback = Path(feedback_file).read_text(encoding="utf-8")
        except OSError as error:
            raise typer.BadParameter(
                f"cannot read --feedback-file {feedback_file!r}: {error}"
            ) from error
    assert feedback is not None
    if not feedback.strip():
        raise typer.BadParameter("the feedback text is empty")
    if harness_backend not in ("local", "e2b"):
        raise typer.BadParameter(
            f"unknown --harness-backend {harness_backend!r}; choose local or e2b"
        )
    try:
        gate = ImproveGate(suite_margin=margin)
    except ValidationError as error:
        raise typer.BadParameter(
            f"--margin must be a fraction in [0, 1): {error}", param_hint="--margin"
        ) from error
    try:
        suite_tasks = load_tasks(tasks_file)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"cannot load tasks from {tasks_file!r}: {error}") from error
    meta_config = resolve_required_model_config(root, "meta")
    effective_e2b_template = resolve_e2b_template(e2b_template)
    run_dir = Path(result_out) if result_out is not None else Path(root) / "runs" / uuid4().hex
    if run_dir.exists():
        raise typer.BadParameter(f"result directory already exists: {run_dir}")

    planned_cells = (iterations + 1) * (len(suite_tasks) + verification_count) * attempts
    qa_cells = verification_count * attempts
    _console.print(
        f"feedback improvement: 1 seed + {iterations} proposal slot(s), {len(suite_tasks)} "
        f"suite task(s) + up to {verification_count} verification task(s), {attempts} "
        f"attempt(s) -> up to {planned_cells} score cells (+ up to {qa_cells} seed QA cells); "
        f"gate: suite within {gate.suite_margin:.0%} of seed, every verification task passes; "
        f"evaluated harness backend={harness_backend}; proposer project=e2b"
    )
    if not yes:
        if not _console.is_terminal:
            raise typer.BadParameter("pass --yes to acknowledge the execution matrix")
        if not Confirm.ask("Proceed?", default=False):
            raise typer.Exit(0)

    result = _execute_improvement(
        name=name,
        root=root,
        run_dir=run_dir,
        feedback=feedback,
        suite_tasks=suite_tasks,
        model=model,
        verification_count=verification_count,
        gate=gate,
        iterations=iterations,
        attempts=attempts,
        harness_backend=cast("Literal['local', 'e2b']", harness_backend),
        e2b_template=effective_e2b_template,
        seed_reference=seed,
        seed_knowledge=seed_knowledge,
        meta_config=meta_config,
    )
    if isinstance(result, ImproveAlreadySatisfied):
        dropped = ", ".join(result.dropped)
        _console.print(
            f"[yellow]feedback appears already satisfied[/yellow]: the seed harness already "
            f"passes all {len(result.dropped)} synthesized verification task(s) ({dropped}); "
            "no optimization was run"
        )
        raise typer.Exit(1)
    outcome = result.outcome
    if outcome.accepted:
        assert result.saved is not None
        assert outcome.candidate_suite_score is not None
        passed = sum(1 for item in outcome.verification if item.passed)
        _console.print(
            f"[green]improved[/green] [bold]{name}[/bold] v{result.saved.version} (champion) "
            f"suite={outcome.candidate_suite_score:.6f} (seed {outcome.seed_suite_score:.6f}); "
            f"verification {passed}/{len(outcome.verification)} passed; "
            f"evidence -> {result.run_dir}"
        )
        return
    _console.print(
        f"[yellow]rejected[/yellow] [bold]{name}[/bold]: {outcome.reason}; "
        f"evidence -> {result.run_dir}"
    )


def _execute_improvement(
    *,
    name: str,
    root: str,
    run_dir: Path,
    feedback: str,
    suite_tasks: list[TaskSpec],
    model: str,
    verification_count: int,
    gate: ImproveGate,
    iterations: int,
    attempts: int,
    harness_backend: Literal["local", "e2b"],
    e2b_template: str | None,
    seed_reference: str | None,
    seed_knowledge: bool,
    meta_config: ProviderConfig,
) -> HarnessImproveOutcome | ImproveAlreadySatisfied:
    """Synthesize the gate, QA it against the seed, run one improvement, and publish evidence."""
    meta_provider = get_provider(meta_config)
    if not isinstance(meta_provider, ToolCallingProvider):
        raise typer.BadParameter("settings [models.meta] provider lacks structured tool calling")
    if not isinstance(meta_provider, Provider):
        raise typer.BadParameter("settings [models.meta] provider lacks plain completions")
    world_store = WorldModelStore(root)
    try:
        model_dir = world_store.resolve(model)
    except (FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    world_model, wm_provider = load_world_model(model_dir)
    agent_provider, _agent_model = resolve_opt_in_model_provider(root, "agent", wm_provider)
    judge = GoldJudge(wm_provider)
    seed_source = _resolve_seed_source(root, name=name, seed=seed_reference)
    seed_doc = seed_source.to_doc("seed")

    synthesis = synthesize_verification_tasks(
        feedback,
        provider=meta_provider,
        count=verification_count,
    )
    collisions = sorted(
        {task.task_id for task in synthesis.tasks} & {task.task_id for task in suite_tasks}
    )
    if collisions:
        raise ValueError(
            f"synthesized verification task id(s) collide with the suite: {collisions}"
        )

    def build_scorer(tasks: list[TaskSpec]) -> WorldModelScorer:
        return WorldModelScorer(
            world_model=world_model,
            tasks=tasks,
            agent_provider=agent_provider,
            judge=judge,
            model_identity={"world_model": model_dir.name},
            harness_backend=harness_backend,
            # "" freezes template absence for e2b (never an env re-read after this snapshot).
            e2b_template=(e2b_template or "") if harness_backend == "e2b" else None,
            eval_concurrency=1 if harness_backend == "local" else 0,
        )

    # QA: a verification task the seed already passes verifies nothing new; drop it before it
    # can make the gate pass vacuously.
    qa_scorer = build_scorer(list(synthesis.tasks))
    qa_score = qa_scorer.score(seed_doc, request=qa_scorer.request(attempts=attempts))
    qa_outcomes = verification_results(qa_score.report, [task.task_id for task in synthesis.tasks])
    dropped = tuple(item.task_id for item in qa_outcomes if item.passed)
    surviving = [task for task in synthesis.tasks if task.task_id not in set(dropped)]
    if not surviving:
        return ImproveAlreadySatisfied(dropped=dropped, run_dir=run_dir)

    knowledge_file: str | None = None
    if seed_knowledge and synthesis.knowledge_notes:
        knowledge_file = _write_feedback_knowledge(
            model_dir, feedback=feedback, notes=synthesis.knowledge_notes
        )
        _console.print(f"wrote environment knowledge -> {model_dir / 'knowledge'}/{knowledge_file}")

    scorer = build_scorer([*suite_tasks, *surviving])
    optimizer = optimizer_agent()
    with AgentProject.create(
        timeout=DEFAULT_PROJECT_TIMEOUT_S,
        template=e2b_template,
        metadata={"wmh_component": "harness_improve"},
    ) as project:

        def proposer_factory(directive: str) -> CandidateProposer:
            return ProjectCandidateProposer(
                project,
                optimizer,
                meta_provider,
                directive=directive,
            )

        outcome = improve_harness(
            seed=seed_source,
            feedback=feedback,
            suite_tasks=suite_tasks,
            verification_tasks=surviving,
            scorer=scorer,
            proposer_factory=proposer_factory,
            iterations=iterations,
            attempts=attempts,
            gate=gate,
        )

    write_population_archive(run_dir / "population", outcome.result)
    payload: JsonObject = {
        "schema_version": 1,
        "accepted": outcome.accepted,
        "reason": outcome.reason,
        "feedback": feedback,
        "seed_suite_score": outcome.seed_suite_score,
        "candidate_suite_score": outcome.candidate_suite_score,
        "selected_candidate_id": (
            outcome.selected.candidate_id if outcome.selected is not None else None
        ),
        "verification": [item.model_dump(mode="json") for item in outcome.verification],
        "suite_task_ids": [task.task_id for task in suite_tasks],
        "verification_task_ids": [task.task_id for task in surviving],
        "dropped_verification_task_ids": list(dropped),
        "suite_margin": gate.suite_margin,
        "knowledge_file": knowledge_file,
    }
    write_json_atomic(run_dir / "improve-outcome.json", payload)

    saved: HarnessDoc | None = None
    if outcome.accepted:
        assert outcome.selected is not None
        selected_doc = outcome.selected.candidate.model_copy(update={"name": name, "version": 0})
        saved = HarnessStore(root).save_version(selected_doc, alias=CHAMPION_ALIAS)
    return HarnessImproveOutcome(
        outcome=outcome,
        saved=saved,
        run_dir=run_dir,
        dropped=dropped,
    )


def _write_feedback_knowledge(model_dir: Path, *, feedback: str, notes: str) -> str:
    """Write the synthesized environment facts as one new feedback-keyed knowledge file."""
    digest = hashlib.sha256(feedback.encode("utf-8")).hexdigest()[:12]
    file_name = f"feedback-{digest}.md"
    KnowledgeBase(ArtifactPaths(model_dir).knowledge).write_file(file_name, notes.rstrip() + "\n")
    return file_name


def _resolve_seed_source(root: str, *, name: str, seed: str | None) -> HarnessSourceTree:
    """Resolve the seed: an explicit ref, NAME's champion when it exists, else the Pi baseline."""
    store = HarnessStore(root)
    if seed is not None:
        base, _, ref = seed.partition("@")
        try:
            return HarnessSourceTree.from_doc(store.load(base, ref or None))
        except (FileNotFoundError, ValueError) as error:
            raise typer.BadParameter(str(error)) from error
    if store.exists(name):
        return HarnessSourceTree.from_doc(store.load(name))
    return HarnessSourceTree.from_doc(default_agent())

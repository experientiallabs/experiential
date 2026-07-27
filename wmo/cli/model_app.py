"""`wmo optimize distill`: train the agent MODEL, leaving its harness pinned.

The third member of the optimizer family, beside `wmo optimize harness`
(prompt surfaces) and `wmo optimize route` (routing policy). Where those
produce a `prompt` or a `routing_policy` artifact, this one produces an
`adapter`: `run` drives one on-policy distillation of a Tinker LoRA student
from rollouts of harbor's own terminus-2 agent, and `report` reads a finished
run dir back.

`run` owns the run's CLI lifecycle: load and pin the inputs (config, task
splits, the harness document supplying the rollout params), project the run
cost into a confirmation table, drive `run_distillation` with progress
rendering, and print the gate verdict plus the serving handoff. The optional
`--promote` step writes `[models.agent]` through the settings save path after
an explicit confirmation.

Run-dir pinning mirrors `run-config.json` in the harbor search flow: a fresh
run records its CLI-level inputs in `distill-run.json` (task splits, backend,
the exact harness version and doc hash), and a resume reuses that record
instead of live flags, rejecting explicit flags that conflict with it. The run
config itself is snapshotted by the run store as `config.toml`, which is what
a bare `--resume` (no `--config`) loads.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import typer
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table

from wmo.agents.default import default_agent
from wmo.config import ARTIFACT_DIR
from wmo.config.settings import ModelRole, load_settings, save_settings, settings_path
from wmo.config.store import validate_name
from wmo.core.types import JsonObject
from wmo.distill.config import DistillConfig, load_distill_config
from wmo.distill.cost import CostEstimate, estimate_run_cost
from wmo.distill.gate import DistillGateRecord
from wmo.distill.loop import (
    DEFAULT_DISTILL_HARNESS,
    STUDENT_AFTER_EVAL,
    STUDENT_BEFORE_EVAL,
    TEACHER_BASELINE_EVAL,
    DistillBudgetError,
    DistillEvalReport,
    DistillProgress,
    DistillResult,
    run_distillation,
)
from wmo.distill.rollouts import E2B_SANDBOXES_PER_TRIAL
from wmo.distill.store import (
    DEFAULT_TINKER_OPENAI_ENDPOINT,
    STUDENT_CHAT_MAX_TOKENS_FIELD,
    AdapterStore,
    DistillRunStore,
    build_handoff_toml,
)
from wmo.harness.doc import HarnessDoc
from wmo.harness.e2b_reap import (
    DEFAULT_E2B_SANDBOX_CAP,
    E2B_API_KEY_ENV,
    E2B_SANDBOX_CAP_ENV,
    CapacityCheck,
    check_capacity,
    is_credential_error,
)
from wmo.harness.population import write_json_atomic
from wmo.harness.store import HarnessStore

DISTILL_RUN_RECORD = "distill-run.json"
"""The CLI-level pin file inside the run dir (see `DistillCliRunRecord`)."""

_PI_NODE_RUNTIME = "pi-node"

model_app = typer.Typer(
    help="Train the agent model itself: on-policy distillation of a Tinker LoRA student "
    "from harbor rollouts, gated on held-out solve rates.",
    no_args_is_help=True,
)

_console = Console()


@model_app.command("run")
def run(
    ctx: typer.Context,
    config: str = typer.Option(
        None,
        "--config",
        help="The run TOML (student, teacher, harbor, rollout, train, sampling, warmup, "
        "eval, gate, pricing, budget, tripwire, wandb sections). Required to start a run; "
        "a resume reuses the run dir's config.toml snapshot, and passing it on a resume is "
        "how you raise budget.max_usd.",
    ),
    run_dir: str = typer.Option(
        ...,
        "--run-dir",
        help="Directory holding ALL durable run state (config snapshot, metrics, "
        "checkpoints, evals, rollout artifacts). Always required.",
    ),
    task_ids: str = typer.Option(
        None,
        "--task-ids",
        help="JSON file with the exact train task-id list; rollouts and interim evals run "
        "here. Required to start a run.",
    ),
    holdout_task_ids: str = typer.Option(
        None,
        "--holdout-task-ids",
        help="JSON file with the exact holdout task-id list; the baselines and the promotion "
        "gate are measured here, disjoint from --task-ids. Required to start a run.",
    ),
    harness: str = typer.Option(
        DEFAULT_DISTILL_HARNESS,
        "--harness",
        help="Stored harness document supplying the rollout params (temperature, max turns, "
        "max output tokens) and the hash that keys every harbor job; the harbor agent is "
        f"always terminus-2, never this document's runtime. The bare literal "
        f"{DEFAULT_DISTILL_HARNESS!r} is the built-in default agent; 'name@ref' pins a stored "
        "version. Pinned for the whole run.",
    ),
    backend: str = typer.Option(
        None,
        "--backend",
        help="Override the run config's harbor.backend: local (docker tasks on this machine) "
        "or e2b (tasks in E2B sandboxes; needs E2B_API_KEY).",
    ),
    resume: bool = typer.Option(False, "--resume", help="Continue the run recorded in --run-dir."),
    promote: bool = typer.Option(
        False,
        "--promote",
        help="After an accepted gate, offer to point the models.agent role in settings.toml "
        "at the distilled adapter (always asks for confirmation).",
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the cost confirmation prompt."),
    root: str = typer.Option(ARTIFACT_DIR, "--root", help="Project dir."),
) -> None:
    """Train (or resume) an agent model by on-policy distillation on harbor tasks.

        wmo optimize distill run --config run.toml --run-dir runs/d1 \\
          --task-ids train.json --holdout-task-ids holdout.json --backend e2b --yes

    Harbor's own terminus-2 agent rolls out on real benchmark tasks while
    sampling from the student's current Tinker LoRA weights, a larger teacher
    scores the exact tokens the student sampled, and each step nudges the
    student toward the teacher with a per-token reverse-KL objective. A
    held-out gate compares teacher, student-before, and student-after solve
    rates, and only an adapter that closes enough of the gap is promoted.

    The harness is NOT the subject here and is never edited: it is pinned for
    the whole run. Search the scaffold with `wmo optimize harness` instead.
    """
    run_distill(
        _console,
        harness_name=harness,
        harness_explicit=_explicit(ctx, "harness"),
        config_path=config,
        task_ids_path=task_ids,
        holdout_task_ids_path=holdout_task_ids,
        run_dir=run_dir,
        backend=backend,
        resume=resume,
        yes=yes,
        promote=promote,
        root=root,
    )


@model_app.command("report")
def report(
    run_dir: str = typer.Option(
        ..., "--run-dir", help="A finished (or aborted) run directory to read back."
    ),
) -> None:
    """Print a run's gate verdict and its held-out before/after table.

    Reads only what the run dir already persisted (`gate.json`, `evals/*.json`,
    `metrics.jsonl`), so it is free to run and safe on a live run dir:

        wmo optimize distill report --run-dir runs/d1
    """
    store = DistillRunStore(run_dir)
    gate = _load_gate(store)
    color = "green" if gate.accepted else "yellow"
    _console.print(f"[{color}]gate[/{color}] {escape(gate.reason)}")
    _console.print(_solve_rate_table(store, gate))
    _print_trained_artifact(_console, store)
    _print_paired_delta(_console, store, gate)
    _print_training_summary(_console, store)


def _explicit(ctx: typer.Context, param: str) -> bool:
    """Whether `param` was explicitly passed on the command line.

    Compared by enum NAME: typer vendors click, so its ParameterSource enum is not
    click.core's class and an identity check would silently never match.
    """
    source = ctx.get_parameter_source(param)
    return source is not None and source.name == "COMMANDLINE"


class DistillCliRunRecord(BaseModel):
    """The CLI inputs pinned into `distill-run.json` when a distill run starts.

    A resume command carries only `--run-dir` (that is what a budget abort
    prints), so everything else the CLI resolved at start is recorded here and
    reloaded on resume; explicit flags that conflict with the record are
    rejected instead of silently changing what is being trained or gated.
    """

    model_config = ConfigDict(frozen=True)

    agent: str
    """The `--harness` value exactly as given (may carry an @ref).

    Named `agent` because that is the key already written into run dirs; it is
    the harness document supplying the rollout params, never the executing
    agent (harbor always runs terminus-2).
    """

    backend: Literal["local", "e2b"]
    seed_version: int | None
    """The stored harness version; None means the built-in default agent."""

    seed_doc_hash: str
    """The resolved harness document's hash; a resume must re-resolve to it."""

    train_task_ids: tuple[str, ...]
    holdout_task_ids: tuple[str, ...]


def run_distill(
    console: Console,
    *,
    harness_name: str,
    harness_explicit: bool,
    config_path: str | None,
    task_ids_path: str | None,
    holdout_task_ids_path: str | None,
    run_dir: str,
    backend: str | None,
    resume: bool,
    yes: bool,
    promote: bool,
    root: str,
) -> None:
    """Run (or resume) one on-policy distillation from the CLI.

    Args:
        console: The CLI's rich console (product output goes through it).
        harness_name: The `--harness` value; the bare default literal is the
            built-in default agent, 'name@ref' loads a stored version.
        harness_explicit: Whether `--harness` was typed rather than defaulted.
            A resume that did not type it adopts the recorded value instead of
            conflicting with it, which is why the printed resume command may
            omit the flag.
        config_path: The per-run TOML; required to start a fresh run. On
            resume None loads the run dir's config.toml snapshot, and an
            explicit path wins over it (the documented budget-abort recovery
            is editing budget.max_usd and resuming).
        task_ids_path: JSON array of train task ids; required to start.
        holdout_task_ids_path: JSON array of holdout task ids; required to
            start. Baselines and the gate are measured here.
        run_dir: The run's durable state directory; always required.
        backend: An explicit `--backend` override for the config's
            harbor.backend, or None when the flag was not given.
        resume: Continue the run recorded in `run_dir`.
        yes: Skip the cost confirmation (see `_confirm_cost` for the one
            case where confirmation is forced anyway).
        promote: After an accepted gate, offer to write `[models.agent]`
            pointing at the distilled adapter (explicit confirmation).
        root: The project dir (harness store, adapter store, settings).

    Raises:
        typer.BadParameter: On any invalid or conflicting input; the message
            names the flag and what to do.
        typer.Exit: When the user declines a confirmation (code 0) or the
            run fails/aborts (code 1).
    """
    # Deferred import: harness_app registers this module's typer app at module
    # scope, so importing its helpers back at module scope would be a circular
    # import.
    from wmo.cli.harness_app import _load_harbor_task_ids

    backend_override: Literal["local", "e2b"] | None
    if backend is None:
        backend_override = None
    elif backend == "e2b":
        backend_override = "e2b"
    elif backend == "local":
        backend_override = "local"
    else:
        raise typer.BadParameter(f"unknown --backend {backend!r}; choose local or e2b")

    run_path = Path(run_dir)
    record_path = run_path / DISTILL_RUN_RECORD
    store = DistillRunStore(run_path)
    seed_version: int | None
    if resume:
        record = _load_record(record_path)
        _reject_resume_conflicts(
            record,
            # A resume that did not type --harness adopts the record rather than
            # conflicting with the option's default, which is what lets the
            # printed resume command omit the flag at the default value.
            harness_name=harness_name if harness_explicit else None,
            backend=backend_override,
            task_ids_path=task_ids_path,
            holdout_task_ids_path=holdout_task_ids_path,
            load_task_ids=_load_harbor_task_ids,
        )
        harness_name = record.agent
        train_ids = record.train_task_ids
        holdout_ids = record.holdout_task_ids
        cfg = _load_config(Path(config_path) if config_path is not None else store.config_path)
        base, seed_doc = _pinned_seed_doc(root, record)
        seed_version = record.seed_version
        effective_backend = record.backend
    else:
        if store.config_path.exists():
            raise typer.BadParameter(
                f"{run_path} already holds a distillation run; pass --resume to "
                "continue it or choose a fresh --run-dir"
            )
        if record_path.exists():
            # The record is written before the loop starts, but the loop's very
            # first durable action is the config.toml snapshot: a record with no
            # snapshot means a previous start failed before doing (or spending)
            # anything, so treat the dir as fresh instead of bricking it.
            console.print(
                f"[yellow]note[/yellow] {run_path} holds a run record from a start "
                "that never began (no config.toml snapshot); starting fresh"
            )
        missing = [
            flag
            for flag, value in (
                ("--config", config_path),
                ("--task-ids", task_ids_path),
                ("--holdout-task-ids", holdout_task_ids_path),
            )
            if value is None
        ]
        if missing:
            raise typer.BadParameter(
                f"{', '.join(missing)} required to start a distillation run "
                "(a resume reuses the run dir's recorded inputs instead)"
            )
        assert config_path is not None  # narrowed by the missing check
        assert task_ids_path is not None and holdout_task_ids_path is not None
        cfg = _load_config(Path(config_path))
        train_ids = _load_harbor_task_ids(Path(task_ids_path))
        holdout_ids = _load_harbor_task_ids(Path(holdout_task_ids_path))
        base, seed_doc, seed_version = _resolve_seed_doc(root, harness_name)
        effective_backend = backend_override if backend_override is not None else cfg.harbor.backend

    overlap = sorted(set(train_ids) & set(holdout_ids))
    if overlap:
        raise typer.BadParameter(
            f"task id(s) {', '.join(overlap)} appear in BOTH --task-ids and "
            "--holdout-task-ids; the gate is only meaningful on tasks the student "
            "never trained on, so make the splits disjoint"
        )
    if effective_backend != cfg.harbor.backend:
        cfg = cfg.model_copy(
            update={"harbor": cfg.harbor.model_copy(update={"backend": effective_backend})}
        )
    runtime_kind = seed_doc.runtime_kind()
    if runtime_kind != _PI_NODE_RUNTIME:
        raise typer.BadParameter(
            f"distillation rollouts read their params from a pi-node harness document, "
            f"but --harness {harness_name!r} has runtime kind {runtime_kind!r}; pass a "
            f"pi-node harness (the built-in {DEFAULT_DISTILL_HARNESS!r} agent, or a "
            "version optimized from it)"
        )
    template_path = Path(cfg.harbor.job_template)
    if not template_path.is_file():
        raise typer.BadParameter(
            f"harbor.job_template {template_path} does not exist; point the distill "
            "config's [harbor] job_template at the harbor JobConfig YAML/JSON the "
            "rollouts should run"
        )
    if effective_backend == "e2b":
        _preflight_e2b_capacity(console, trial_concurrency=cfg.train.trial_concurrency)

    console.print(
        f"distilling [bold]{base}[/bold]: student {cfg.student.base_model} <- teacher "
        f"{cfg.teacher.checkpoint or cfg.teacher.model}, {cfg.train.steps} step(s) x "
        f"{cfg.train.tasks_per_batch} task(s) x {cfg.train.group_size} attempt(s), "
        f"{len(train_ids)} train / {len(holdout_ids)} holdout task(s), "
        f"backend {effective_backend} -> {run_path}"
    )
    estimate = estimate_run_cost(cfg, len(train_ids), len(holdout_ids))
    _print_cost_estimate(console, cfg, estimate)
    _confirm_cost(console, estimate, cfg.budget.max_usd, yes=yes)

    if not resume:
        # Recorded only now: inputs validated and the user confirmed, so a
        # declined or failed start never poisons the run dir.
        record = DistillCliRunRecord(
            agent=harness_name,
            backend=effective_backend,
            seed_version=seed_version,
            seed_doc_hash=seed_doc.doc_hash,
            train_task_ids=train_ids,
            holdout_task_ids=holdout_ids,
        )
        run_path.mkdir(parents=True, exist_ok=True)
        write_json_atomic(record_path, record.model_dump(mode="json"))

    def _on_progress(event: DistillProgress) -> None:
        spend = f" (${event.spent_usd:.2f} spent)" if event.spent_usd > 0 else ""
        # The literal bracket is escaped so rich does not eat the phase as a markup tag.
        console.print(f"  \\[{event.phase}] {escape(event.message)}{spend}")

    try:
        result = run_distillation(
            base,
            cfg,
            seed_doc,
            list(train_ids),
            list(holdout_ids),
            run_path,
            resume=resume,
            on_progress=_on_progress,
            adapter_store=AdapterStore(root),
            # Resume commands must print the --harness string as typed (it may
            # carry an @ref that `base` strips), or the printed command would
            # trip the CLI's resume conflict check.
            cli_agent=harness_name,
        )
    except DistillBudgetError as exc:
        console.print(f"[red]budget exhausted[/red] {escape(str(exc))}")
        console.print(f"resume with: [bold]{escape(exc.resume_command)}[/bold]", soft_wrap=True)
        raise typer.Exit(1) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except (RuntimeError, ImportError) as exc:
        console.print(f"[red]distillation failed[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    _print_result(
        console, result, store, adapters=AdapterStore(root), base_model=cfg.student.base_model
    )
    if promote:
        _maybe_promote(console, result, cfg, root)


# -- input resolution ------------------------------------------------------------------------


def _load_config(path: Path) -> DistillConfig:
    """Load the run TOML, turning load failures into usage errors."""
    try:
        return load_distill_config(path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"{exc} (a fresh run needs --config; a resume reads the run dir's config.toml snapshot)"
        ) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_record(path: Path) -> DistillCliRunRecord:
    """Load the pinned `distill-run.json` for a resume."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"--resume found no {DISTILL_RUN_RECORD} under {path.parent}; start the "
            "run once without --resume"
        ) from exc
    try:
        return DistillCliRunRecord.model_validate_json(text)
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot load {path}: {exc}") from exc


def _reject_resume_conflicts(
    record: DistillCliRunRecord,
    *,
    harness_name: str | None,
    backend: str | None,
    task_ids_path: str | None,
    holdout_task_ids_path: str | None,
    load_task_ids: Callable[[Path], tuple[str, ...]],
) -> None:
    """Reject explicit flags that conflict with the recorded run inputs.

    Every flag is compared only when it was actually typed (None means it was
    not), so a resume that carries just `--run-dir --resume` adopts the record
    wholesale instead of colliding with an option default.
    """
    conflicts: list[str] = []
    if harness_name is not None and harness_name != record.agent:
        conflicts.append(f"--harness {harness_name!r} != recorded {record.agent!r}")
    if backend is not None and backend != record.backend:
        conflicts.append(f"--backend {backend!r} != recorded {record.backend!r}")
    if task_ids_path is not None and load_task_ids(Path(task_ids_path)) != record.train_task_ids:
        conflicts.append("--task-ids differs from the recorded train split")
    if (
        holdout_task_ids_path is not None
        and load_task_ids(Path(holdout_task_ids_path)) != record.holdout_task_ids
    ):
        conflicts.append("--holdout-task-ids differs from the recorded holdout split")
    if conflicts:
        raise typer.BadParameter(
            f"--resume uses the recorded {DISTILL_RUN_RECORD}; conflicting flag(s): "
            + "; ".join(conflicts)
            + ". Drop them to continue this run, or start a fresh --run-dir"
        )


def _resolve_seed_doc(root: str, harness_ref: str) -> tuple[str, HarnessDoc, int | None]:
    """Resolve `--harness` to the document the trials read their params from.

    Mirrors the harbor search's seed protocol: the bare default literal is
    ALWAYS the built-in agent; 'name@ref' loads a stored version. Returns the
    base name, the document, and the resolved store version (None for the
    built-in seed) so a resume can pin exactly what the run started from.
    """
    base, _, ref = harness_ref.partition("@")
    try:
        validate_name(base)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if base == DEFAULT_DISTILL_HARNESS and not ref:
        return base, default_agent(base), None
    try:
        doc = HarnessStore(root).load(base, ref or None)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(
            f"{exc}; the built-in default agent is the literal {DEFAULT_DISTILL_HARNESS!r}"
        ) from exc
    return base, doc, doc.version


def _pinned_seed_doc(root: str, record: DistillCliRunRecord) -> tuple[str, HarnessDoc]:
    """Re-resolve the recorded seed for a resume, never a live movable ref.

    The record pins the exact version and doc hash, so champion movement (or
    any store edit) between sessions cannot silently change which harness the
    remaining trials run.
    """
    base = record.agent.partition("@")[0]
    if record.seed_version is None:
        doc = default_agent(base)
    else:
        try:
            doc = HarnessStore(root).load(base, str(record.seed_version))
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(
                f"cannot reload the recorded seed {base}@v{record.seed_version}: {exc}"
            ) from exc
    if doc.doc_hash != record.seed_doc_hash:
        raise typer.BadParameter(
            f"the recorded seed {base} resolved to doc hash {doc.doc_hash[:12]} but "
            f"this run pinned {record.seed_doc_hash[:12]}; restore the recorded "
            "harness version or start a fresh --run-dir"
        )
    return base, doc


# -- e2b capacity preflight ------------------------------------------------------------------


def _preflight_e2b_capacity(console: Console, *, trial_concurrency: int) -> None:
    """Refuse to start an e2b run that cannot claim the concurrency it asks for.

    E2B caps concurrent sandboxes per account, and a running trial holds
    `E2B_SANDBOXES_PER_TRIAL` of them (harbor's task environment, which lives for its own
    multi-hour timeout; terminus-2 itself runs in this process and needs no sandbox of its
    own). When orphans of an earlier crashed run fill
    the account, every trial fails at sandbox creation with a 429 and the run produces zero
    token spans, which reads exactly like a broken model. So: count what is running, reclaim
    this machine's provable orphans (exact ids whose owning process is gone), and fail with the
    numbers if that is still not enough. The account-wide sweep is never automatic; the message
    names it instead.

    Raises:
        typer.BadParameter: If capacity cannot be measured (missing extra or credential) or
            too few slots are free after reaping the safe class.
    """
    required = trial_concurrency * E2B_SANDBOXES_PER_TRIAL
    try:
        check = check_capacity(required=required)
    except ImportError as error:
        raise typer.BadParameter(
            f"{error}; the distill config selects harbor.backend = 'e2b'"
        ) from error
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    except Exception as error:  # noqa: BLE001 - a monitoring call must not break a resume
        if is_credential_error(error):
            raise typer.BadParameter(
                f"E2B rejected the sandbox capacity check ({error}); harbor.backend = 'e2b' "
                f"runs every trial in E2B, so set ${E2B_API_KEY_ENV} to an account key (or "
                "switch the distill config to backend = 'local')"
            ) from error
        console.print(
            f"[yellow]warning[/yellow] could not check E2B sandbox capacity "
            f"({type(error).__name__}: {escape(str(error))}); starting anyway"
        )
        return
    if check.reaped:
        console.print(
            f"reaped {check.reaped} orphaned E2B sandbox(es) from dead local runs "
            f"({check.alive_before} -> {check.alive} of {check.cap} in use)"
        )
    if not check.ok:
        raise typer.BadParameter(_capacity_failure_message(check, trial_concurrency))
    console.print(
        f"e2b capacity ok: {check.alive}/{check.cap} sandbox(es) in use, {check.free} free, "
        f"{required} needed ({E2B_SANDBOXES_PER_TRIAL} per trial x "
        f"train.trial_concurrency={trial_concurrency})"
    )


def _capacity_failure_message(check: CapacityCheck, trial_concurrency: int) -> str:
    """The actionable message for a run that cannot get enough sandbox slots."""
    reaped = (
        f" Reaping orphans of dead local runs freed {check.reaped} slot(s) and was not enough."
        if check.reaped
        else " No orphan of a dead local run was left to reclaim."
    )
    affordable = check.free // E2B_SANDBOXES_PER_TRIAL
    lower = f"lower train.trial_concurrency to at most {affordable}, " if affordable >= 1 else ""
    return (
        f"not enough free E2B sandbox slots: {check.alive} of {check.cap} concurrent "
        f"sandboxes are in use, leaving {check.free} free, but this run needs "
        f"{check.required} ({E2B_SANDBOXES_PER_TRIAL} per trial x "
        f"train.trial_concurrency={trial_concurrency}: harbor's task environment)"
        f".{reaped} Either run `wmo e2b reap --stale-minutes 60 --yes` to kill older "
        f"harbor trial sandboxes (account-wide: it can kill another machine's run), {lower}wait "
        f"for the other runs to finish, or raise the account cap (set ${E2B_SANDBOX_CAP_ENV} "
        f"when your cap is not {DEFAULT_E2B_SANDBOX_CAP})"
    )


# -- cost confirmation -----------------------------------------------------------------------


def _print_cost_estimate(console: Console, cfg: DistillConfig, estimate: CostEstimate) -> None:
    """Render the per-meter cost projection; unpriced meters print "unknown"."""
    table = Table(title="Distillation cost estimate")
    table.add_column("Meter", no_wrap=True)
    table.add_column("Tokens", justify="right")
    table.add_column("$/Mtok", justify="right")
    table.add_column("USD", justify="right")
    for line in estimate.lines:
        table.add_row(
            line.meter,
            f"{line.tokens:,}",
            "unknown" if line.price_per_mtok is None else f"{line.price_per_mtok:.3f}",
            "unknown" if line.usd is None else f"{line.usd:.2f}",
        )
    console.print(table)
    cap = (
        f"hard cap budget.max_usd=${cfg.budget.max_usd:.2f}"
        if cfg.budget.max_usd is not None
        else "no budget.max_usd cap"
    )
    warmup = f"{estimate.warmup_episodes} warmup + " if estimate.warmup_episodes > 0 else ""
    console.print(
        f"{estimate.train_episodes} train + {warmup}{estimate.eval_episodes} interim-eval + "
        f"{estimate.baseline_episodes} gate/baseline episode(s); priced total "
        f"${estimate.priced_usd:.2f}; {cap}"
    )


def _confirm_cost(
    console: Console, estimate: CostEstimate, max_usd: float | None, *, yes: bool
) -> None:
    """Confirm the projected spend before anything is run.

    The rule: `--yes` is honored whenever the spend is accountable, meaning
    the estimate is fully priced OR `budget.max_usd` caps the worst case.
    When unpriced meters exist AND `budget.max_usd` is unset, the run's spend
    is unbounded and unaccounted, so interactive confirmation is forced even
    with `--yes`; a non-interactive invocation in that state is rejected with
    instructions (price the meters or set the cap).

    Raises:
        typer.BadParameter: Unbounded spend in a non-interactive session.
        typer.Exit: The user declined (exit code 0).
    """
    if estimate.unpriced_meters and max_usd is None:
        meters = ", ".join(estimate.unpriced_meters)
        console.print(
            f"[yellow]warning[/yellow] meter(s) {meters} have no \\[pricing] entry and "
            "budget.max_usd is unset: the run's spend is unbounded and unaccounted, "
            "so --yes does not apply here"
        )
        if not console.is_terminal:
            raise typer.BadParameter(
                f"cannot start with unbounded spend non-interactively: meter(s) "
                f"{meters} are unpriced and budget.max_usd is unset; add [pricing] "
                "entries for them or set [budget] max_usd in the distill config, "
                "or run at a TTY to confirm explicitly"
            )
        if not Confirm.ask("Proceed with unbounded spend?", default=False):
            raise typer.Exit(0)
        return
    if console.is_terminal and not yes and not Confirm.ask("Proceed?", default=True):
        raise typer.Exit(0)


# -- completion output -----------------------------------------------------------------------


def _print_result(
    console: Console,
    result: DistillResult,
    store: DistillRunStore,
    *,
    adapters: AdapterStore,
    base_model: str,
) -> None:
    """Print the gate verdict, artifact paths, and the serving handoff snippet."""
    gate = result.gate
    color = "green" if gate.accepted else "yellow"
    console.print(f"[{color}]gate[/{color}] {escape(gate.reason)}")
    console.print(
        f"  holdout solve rates: teacher {gate.teacher_solve_rate:.3f}, "
        f"student before {gate.student_before_solve_rate:.3f}, "
        f"after {gate.student_after_solve_rate:.3f}"
    )
    if result.adapter_version is not None:
        console.print(
            f"[green]adapter[/green] [bold]{result.name}[/bold] v{result.adapter_version} "
            f"(champion) -> {adapters.dir_for(result.name) / f'v{result.adapter_version}'}",
            soft_wrap=True,
        )
    else:
        console.print("adapter not promoted; the run dir keeps every artifact for inspection")
    console.print(f"final sampler weights: {result.final_sampler_path}", soft_wrap=True)
    console.print(f"resumable training state: {result.final_state_path}", soft_wrap=True)
    console.print(
        f"spend: ${result.spend.total_usd:.2f} total "
        f"(this session ${result.spend.session_usd:.2f}) -> {result.run_dir}",
        soft_wrap=True,
    )
    try:
        handoff = build_handoff_toml(result.final_sampler_path, base_model=base_model)
    except ValueError as exc:
        console.print(f"[yellow]no handoff snippet[/yellow]: {escape(str(exc))}")
        return
    location = f" (written to {store.handoff_path})" if gate.accepted else ""
    console.print(f"serving handoff{location}:")
    console.print(escape(handoff))


def _maybe_promote(console: Console, result: DistillResult, cfg: DistillConfig, root: str) -> None:
    """Write `[models.agent]` for an accepted adapter, after an explicit confirm.

    The write changes what every subsequent local run and optimization uses as
    the agent model, so it always asks, even under `--yes`; a rejected gate
    skips the write with a warning.
    """
    if result.adapter_version is None:
        console.print(
            "[yellow]--promote skipped[/yellow]: the gate rejected this adapter, so "
            "\\[models.agent] was not changed (the handoff snippet above still works "
            "for manual experiments)"
        )
        return
    path = settings_path(root)
    try:
        confirmed = Confirm.ask(
            f"Write models.agent = {result.final_sampler_path} to {path}?",
            default=False,
        )
    except EOFError:
        confirmed = False
    if not confirmed:
        console.print(
            f"skipped writing \\[models.agent]; paste the handoff snippet into {path} when ready"
        )
        return
    try:
        settings = load_settings(root)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    settings.models.agent = ModelRole(
        provider="openai",
        model=result.final_sampler_path,
        model_type=cfg.student.base_model,
        endpoint=DEFAULT_TINKER_OPENAI_ENDPOINT,
        # A tinker:// path is outside the built-in catalog, so capability resolution would fall
        # back to `max_completion_tokens`, which Tinker's endpoint 400s on. Pin the name it takes.
        chat_max_tokens_field=STUDENT_CHAT_MAX_TOKENS_FIELD,
    )
    save_settings(settings, root)
    console.print(
        f"[green]wrote[/green] \\[models.agent] -> {path} (set WMO_ENDPOINT_API_KEY to "
        "your Tinker API key before running the agent)"
    )


# -- report ------------------------------------------------------------------------------------

_REPORT_ROWS: tuple[tuple[str, str], ...] = (
    ("teacher", TEACHER_BASELINE_EVAL),
    ("student before", STUDENT_BEFORE_EVAL),
    ("student after", STUDENT_AFTER_EVAL),
)
"""The three held-out measurements the gate compares, in table order, paired
with the `evals/<key>.json` each one was written to."""


def _load_gate(store: DistillRunStore) -> DistillGateRecord:
    """Read the run's `gate.json`, turning a missing or corrupt file into a usage error."""
    try:
        text = store.gate_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"no {store.gate_path}: this run has not reached its gate yet (or "
            f"{store.run_dir} is not a distillation run dir). Finish or resume it with "
            "`wmo optimize distill run --run-dir <dir> --resume`"
        ) from exc
    try:
        return DistillGateRecord.model_validate_json(text)
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot load {store.gate_path}: {exc}") from exc


def _load_eval_report(store: DistillRunStore, key: str) -> DistillEvalReport | None:
    """Read one `evals/<key>.json`, or None when the run never wrote it.

    A missing report is normal (an imported baseline is copied in, but an
    aborted run may have none), so the table degrades to the rates gate.json
    already carries rather than failing.
    """
    try:
        text = (store.evals_dir / f"{key}.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return DistillEvalReport.model_validate_json(text)
    except ValidationError as exc:
        raise typer.BadParameter(
            f"cannot load {store.evals_dir / f'{key}.json'}: {exc}; delete the file to "
            "report from gate.json alone"
        ) from exc


def _solve_rate_table(store: DistillRunStore, gate: DistillGateRecord) -> Table:
    """The teacher / student-before / student-after held-out comparison."""
    rates = (
        gate.teacher_solve_rate,
        gate.student_before_solve_rate,
        gate.student_after_solve_rate,
    )
    table = Table(title=f"Held-out solve rates ({store.run_dir})")
    table.add_column("Measurement", no_wrap=True)
    # Fold rather than ellipsize. A tinker:// sampler path is wider than the ~22 columns
    # this cell gets at an 80-column terminal, and rich's default would truncate it to
    # `tinker://weights/pi...`, silently dropping the identity of the artifact. Folding
    # keeps every character; the copyable form is printed on its own line below.
    table.add_column("Model", overflow="fold")
    table.add_column("Solve", justify="right")
    table.add_column("Graded", justify="right")
    table.add_column("Executed", justify="right")
    table.add_column("Scaffold", justify="right")
    for (label, key), rate in zip(_REPORT_ROWS, rates, strict=True):
        eval_report = _load_eval_report(store, key)
        if eval_report is None:
            table.add_row(label, "unknown", f"{rate:.3f}", "-", "-", "-")
            continue
        graded = (
            f"{eval_report.graded_solve_rate:.3f}" if eval_report.graded_trials else "unmeasured"
        )
        table.add_row(
            label,
            eval_report.provider_model,
            f"{rate:.3f}",
            graded,
            f"{eval_report.executed_trials}/{eval_report.trials}",
            f"{eval_report.scaffold_loss_rate:.0%}",
        )
    return table


def _print_trained_artifact(console: Console, store: DistillRunStore) -> None:
    """Print the sampler path the student-after numbers came from, on its own line.

    The table names it too, but a `tinker://` path is wider than the cell it gets at an
    80-column terminal, so there it folds across lines. This line is the copyable one: it
    is what you paste into a pool entry or a follow-on run's `init_from_state`.

    Args:
        console: Where to print.
        store: The run store to read the student-after eval report from.
    """
    after = _load_eval_report(store, STUDENT_AFTER_EVAL)
    if after is None or not after.provider_model:
        return
    console.print(f"trained artifact: {escape(after.provider_model)}")


def _print_paired_delta(console: Console, store: DistillRunStore, gate: DistillGateRecord) -> None:
    """Print what training moved, on the same holdout split the gate read."""
    binary = gate.student_after_solve_rate - gate.student_before_solve_rate
    console.print(f"paired delta (after - before): {binary:+.3f} solve rate")
    before = _load_eval_report(store, STUDENT_BEFORE_EVAL)
    after = _load_eval_report(store, STUDENT_AFTER_EVAL)
    if before is not None and after is not None and before.graded_trials and after.graded_trials:
        graded = after.graded_solve_rate - before.graded_solve_rate
        console.print(f"  graded (same trials at test resolution): {graded:+.3f}")
    fraction = (
        gate.student_after_solve_rate / gate.teacher_solve_rate
        if gate.teacher_solve_rate > 0
        else None
    )
    reached = "unmeasurable (teacher solved nothing)" if fraction is None else f"{fraction:.3f}"
    verdict = "passed" if gate.accepted else "FAILED"
    console.print(
        f"  after / teacher: {reached} against gate minimum "
        f"{gate.min_teacher_fraction:.2f}; gate {verdict}"
    )


def _print_training_summary(console: Console, store: DistillRunStore) -> None:
    """Print the last training row's health metrics, or say the run trained nothing.

    Turns per episode is deliberately absent: nothing in the run dir records
    it. `mean_generation_tokens` is the per-episode series the loop does
    measure (sampled tokens, pooled over the batch's span-bearing episodes).
    """
    rows = [row for row in store.read_metrics() if row.get("phase") is None]
    if not rows:
        console.print("no training step recorded in metrics.jsonl")
        return
    last = rows[-1]
    step = _row_int(last, "step")
    parts = [f"{len(rows)} training step(s) recorded"]
    for label, key, spec in (
        ("reverse KL/token", "reverse_kl_per_token", ".4f"),
        ("entropy ratio", "entropy_ratio", ".2f"),
        ("tokens/episode", "mean_generation_tokens", ".0f"),
        ("tokens/episode ratio", "generation_tokens_ratio", ".2f"),
    ):
        value = _row_float(last, key)
        if value is not None:
            parts.append(f"{label} {value:{spec}}")
    spent = _row_float(last, "cumulative_usd")
    if spent is not None:
        parts.append(f"${spent:.2f} spent")
    head = "training" if step is None else f"training (last row step {step})"
    console.print(f"{head}: {', '.join(parts)}")


def _row_float(row: JsonObject, key: str) -> float | None:
    """One metrics-row number, or None when absent or not numeric."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _row_int(row: JsonObject, key: str) -> int | None:
    """One metrics-row integer, or None when absent or not an integer."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value

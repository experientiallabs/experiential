"""`wmh eval --mode closed-loop` and `wmh eval agreement` — the closed-loop halves of eval.

Kept out of `app.py` so the (large) eval command stays readable; `app.py` routes here.
Closed-loop mode runs the fixed agent against a built world model (`--env sim`, the default) or
against real E2B sandboxes (`--env e2b`, one fresh sandbox per rollout) and scores task success;
`agreement` compares two saved closed-loop reports (e.g. one produced against the world model and
one against a real environment) — the outcome-agreement check docs/reference/closed_loop.md names.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from wmh.config import WorldModelStore, load_settings
from wmh.engine import load_world_model
from wmh.engine.world_model import WorldModel
from wmh.evals.agreement import compute_agreement
from wmh.evals.closed_loop import ClosedLoopEval, ClosedLoopReport, evaluate_with_env
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import load_tasks
from wmh.harness.doc import MAX_TURNS_ID, HarnessDoc, Surface, SurfaceKind
from wmh.harness.e2b_env import e2b_env_factory
from wmh.harness.runtime import DEFAULT_MAX_TURNS, AgentRuntime
from wmh.harness.store import HarnessStore
from wmh.providers import get_provider
from wmh.providers.base import Provider, ProviderConfig, ProviderKind

# The provider real-env runs fall back to when no world model anchors one (mirrors the scenario
# tools' default in app.py).
_DEFAULT_PROVIDER_KIND = ProviderKind.BEDROCK
_DEFAULT_PROVIDER_MODEL = "us.anthropic.claude-opus-4-8"


def default_worker_provider(root: str) -> tuple[Provider, str]:
    """(provider, model id) for runs that load no world model (`--env e2b` without `--model`).

    Sim runs reuse the provider their world model was built on; a real-sandbox run has no such
    anchor, so it resolves the `[models.worker]` role from the project settings and falls back to
    the built-in default the scenario tools use.
    """
    role = load_settings(root).models.resolve("worker")
    if role is None:
        config = ProviderConfig(kind=_DEFAULT_PROVIDER_KIND, model=_DEFAULT_PROVIDER_MODEL)
    else:
        try:
            kind = ProviderKind(role.provider)
        except ValueError as exc:
            kinds = ", ".join(k.value for k in ProviderKind)
            raise typer.BadParameter(
                f"settings [models.worker] names unknown provider {role.provider!r}; "
                f"choose one of: {kinds}"
            ) from exc
        config = ProviderConfig(
            kind=kind,
            model=role.model,
            region=role.region,
            endpoint=role.endpoint,
            deployment=role.deployment,
        )
    return get_provider(config), config.model


def run_closed_loop(
    console: Console,
    *,
    tasks_file: str,
    name: str | None,
    root: str,
    k: int,
    max_turns: int | None,
    out: str | None,
    harness: str | None = None,
    env: str = "sim",
    eval_concurrency: int | None = None,
    e2b_template: str | None = None,
) -> None:
    """Run an agent harness on each task against the chosen env; print and optionally save.

    `--harness <name>[@ref]` runs a stored harness version (ref = version or alias; default is
    the champion alias); without it the built-in baseline loop runs. `max_turns=None` means "the
    harness's own cap" (or the default for the baseline); an explicit value overrides either —
    never silently ignored. `--env e2b` swaps the world model for one fresh E2B sandbox per
    (task, attempt) cell — all cells at once unless `--eval-concurrency` caps them — and labels
    the report `<agent>@e2b` so `wmh eval agreement` reads naturally; the world model is then
    optional (`--name` only pins which provider runs the agent/judge).
    """
    if env not in ("sim", "e2b"):
        raise typer.BadParameter(f"unknown --env {env!r}; choose sim or e2b")
    try:
        tasks = load_tasks(tasks_file)
    except (OSError, ValueError) as exc:  # missing file, malformed JSONL, empty, duplicate ids
        raise typer.BadParameter(f"cannot load tasks from {tasks_file!r}: {exc}") from exc
    # The world model: required for sim; for e2b only loaded when --name pins one (its provider
    # then runs the agent + judge; the model itself plays no part in real rollouts).
    world_model: WorldModel | None = None
    if env == "sim" or name is not None:
        store = WorldModelStore(root)
        try:
            model_dir = store.resolve(name)
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        world_model, provider = load_world_model(model_dir)
        env_label = model_dir.name if env == "sim" else "e2b"
    else:
        provider, _model_id = default_worker_provider(root)
        env_label = "e2b"

    loaded_harness = _load_harness(harness, root)
    agent_label = (
        f"{loaded_harness.name}-v{loaded_harness.version}"
        if loaded_harness is not None
        else "baseline"
    )
    versus = (
        f"world model [bold]{env_label}[/bold]"
        if env == "sim"
        else f"[bold]real E2B sandboxes[/bold] ({k * len(tasks)} rollouts, one sandbox each)"
    )
    console.print(
        f"closed-loop: harness [bold]{agent_label}[/bold] vs {versus} "
        f"on {len(tasks)} task(s), k={k}…"
    )

    def _progress(task_id: str, attempt: int, verdict: GoldVerdict) -> None:
        mark = "[green]pass[/green]" if verdict.passed else "[red]fail[/red]"
        console.print(f"  {task_id} #{attempt}: {mark} ({verdict.rationale})")

    if loaded_harness is not None:
        if max_turns is not None and max_turns != loaded_harness.max_turns():
            console.print(
                f"  note: --max-turns {max_turns} overrides the harness's own "
                f"max_turns={loaded_harness.max_turns()}"
            )
            loaded_harness = _with_max_turns(loaded_harness, max_turns)
        runtime = loaded_harness.runtime(
            provider,
            backend="e2b" if env == "e2b" else "local",
            e2b_template=e2b_template,
        )
    else:
        runtime = AgentRuntime(provider, max_turns=max_turns or DEFAULT_MAX_TURNS)
    if env == "e2b":
        report = evaluate_with_env(
            tasks,
            e2b_env_factory(template=e2b_template),
            runtime,
            GoldJudge(provider),
            label=f"{agent_label}@e2b",
            k=k,
            concurrency=eval_concurrency if eval_concurrency is not None else 0,
            on_progress=_progress,
        )
    else:
        if world_model is None:  # unreachable: sim always resolved a world model above
            raise typer.BadParameter("--env sim needs a built world model")
        evaluation = ClosedLoopEval(
            tasks,
            world_model,
            provider,
            GoldJudge(provider),
            label=f"{agent_label}@{env_label}",
            k=k,
            concurrency=eval_concurrency if eval_concurrency is not None else 1,
            runtime=runtime,
            on_progress=_progress,
        )
        report = evaluation.run()
    for task_id, outcome in report.per_task.items():
        console.print(
            f"  {task_id:24} success={outcome.success_rate:.2f} "
            f"assertions={outcome.mean_fraction:.2f}"
        )
    console.print(f"[bold]OVERALL[/bold] {report.summary()}")
    if out:
        Path(out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"wrote closed-loop report -> {out}")


def run_agreement(console: Console, *, report_a: str, report_b: str, threshold: float) -> None:
    """Compare two saved closed-loop reports task-by-task and print the agreement verdict."""
    a = _load_report(report_a)
    b = _load_report(report_b)
    result = compute_agreement(a, b, pass_threshold=threshold)
    c = result.confusion
    la, lb = result.label_a or "A", result.label_b or "B"
    console.print(f"[bold]task verdict confusion[/bold] ({la} vs {lb}):")
    console.print(f"  {la}-pass & {lb}-pass: {c.a_pass_b_pass}")
    console.print(f"  {la}-pass & {lb}-FAIL: {c.a_pass_b_fail}  (A over-credits these)")
    console.print(f"  {la}-FAIL & {lb}-pass: {c.a_fail_b_pass}")
    console.print(f"  {la}-FAIL & {lb}-FAIL: {c.a_fail_b_fail}")
    console.print(f"[bold]VERDICT[/bold] {result.summary()}")


def _load_report(path: str) -> ClosedLoopReport:
    try:
        return ClosedLoopReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"cannot read closed-loop report {path!r}: {exc}") from exc


def _with_max_turns(doc: HarnessDoc, max_turns: int) -> HarnessDoc:
    """A copy of `doc` with its max-turns surface replaced (re-validated via the constructor)."""
    surfaces = [s for s in doc.surfaces if s.id != MAX_TURNS_ID]
    surfaces.append(Surface(id=MAX_TURNS_ID, kind=SurfaceKind.PARAM, content=str(max_turns)))
    return HarnessDoc(name=doc.name, version=doc.version, surfaces=surfaces)


def _load_harness(name: str | None, root: str) -> HarnessDoc | None:
    if name is None:
        return None
    base, _, ref = name.partition("@")
    try:
        return HarnessStore(root).load(base, ref or None)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

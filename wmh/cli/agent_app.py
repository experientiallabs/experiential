"""`wmh agent` — the managed agent harness + runtime (the build-agent path).

Three commands, mirroring the produce→build→evaluate→evolve loop:

    wmh agent collect  # run the agent for real in E2B sandboxes -> capture traces (feed build)
    wmh agent eval     # closed-loop: score a harness variant against a built world model (k=3)
    wmh agent evolve   # evolutionary search over harness variants, driven by closed-loop deltas

Collection needs the E2B extra + `$E2B_API_KEY`; eval/evolve run against a built world model and use
only that model's serve provider (as both the world model and, by default, the agent model).
"""

from __future__ import annotations

import typer
from rich.console import Console

from wmh.config import ARTIFACT_DIR

agent_app = typer.Typer(help="Managed agent harness: collect traces, closed-loop eval, evolve.")
_console = Console()

_TASKS_ARG = typer.Argument(..., help="JSONL task file (one TaskSpec per line).")


def _load_agent_provider(provider: str, model: str, region: str | None):  # noqa: ANN202
    """Build the agent/meta model provider from CLI flags (defaults match `wmh build`)."""
    from wmh.providers import ProviderConfig, ProviderKind, get_provider

    return get_provider(ProviderConfig(kind=ProviderKind(provider), model=model, region=region))


@agent_app.command("collect")
def collect(
    tasks_file: str = _TASKS_ARG,
    out: str = typer.Option(..., "--out", help="Output .otel.jsonl path for captured traces."),
    provider: str = typer.Option("bedrock", "--provider", help="Agent model provider."),
    model: str = typer.Option("us.anthropic.claude-opus-4-8", help="Agent model id."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    skills: str = typer.Option(
        None, "--skills", help="Skill library dir (grows as the agent saves)."
    ),
    template: str = typer.Option(None, "--template", help="E2B template id (default: base image)."),
    timeout: int = typer.Option(300, help="E2B sandbox timeout (seconds)."),
    max_turns: int = typer.Option(20, help="Agent turn cap per task."),
) -> None:
    """Run the baseline harness on each task in a real E2B sandbox and write captured traces.

    The traces land in the `otel-genai` shape, so `wmh build --file <out>` builds a world model from
    the agent's own behavior. Requires `uv sync --extra e2b` and `$E2B_API_KEY`.
    """
    from wmh.agent.capture import write_otel_traces
    from wmh.agent.collect import collect_traces
    from wmh.agent.runtime import RunResult
    from wmh.agent.skills import SkillLibrary
    from wmh.agent.spec import HarnessSpec
    from wmh.agent.tasks import load_tasks

    tasks = load_tasks(tasks_file)
    agent_provider = _load_agent_provider(provider, model, region)
    library = SkillLibrary(skills) if skills else None
    spec = HarnessSpec(max_turns=max_turns)

    def _progress(task_id: str, result: RunResult) -> None:
        _console.print(
            f"  {task_id}: {result.stop_reason.value} in {result.turns} turns "
            f"({len(result.saved_skills)} skills saved)"
        )

    _console.print(f"collecting {len(tasks)} task(s) in E2B sandboxes…")
    traces = collect_traces(
        spec,
        tasks,
        agent_provider,
        library=library,
        template=template,
        timeout=timeout,
        on_progress=_progress,
    )
    path = write_otel_traces(traces, out)
    steps = sum(len(t.steps) for t in traces)
    _console.print(f"[green]wrote[/green] {len(traces)} trace(s), {steps} step(s) -> {path}")
    _console.print(f"next: [bold]wmh build --file {path}[/bold] to build a world model from them")


@agent_app.command("eval")
def eval_harness(
    tasks_file: str = _TASKS_ARG,
    name: str = typer.Option(
        None, "--name", help="World model to eval against (default: only one)."
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding the world model."),
    spec_file: str = typer.Option(None, "--spec", help="HarnessSpec JSON (default: baseline)."),
    skills: str = typer.Option(None, "--skills", help="Skill library dir to seed from."),
    k: int = typer.Option(3, help="Passes per task (mean reported; never single-pass)."),
) -> None:
    """Closed-loop: run a harness variant against the world model, scoring gold-assertion pass."""
    from wmh.agent.closed_loop import evaluate_closed_loop
    from wmh.agent.gold import GoldJudge
    from wmh.agent.skills import SkillLibrary
    from wmh.agent.tasks import load_tasks
    from wmh.cli.agent_support import load_world_model_for_agent

    tasks = load_tasks(tasks_file)
    world_model, provider = load_world_model_for_agent(name, root)
    spec = _read_spec(spec_file)
    library = SkillLibrary(skills) if skills else None
    judge = GoldJudge(provider)

    _console.print(f"closed-loop eval of harness '{spec.name}' on {len(tasks)} task(s), k={k}…")
    report = evaluate_closed_loop(spec, tasks, world_model, provider, judge, library=library, k=k)
    for task_id, outcome in report.per_task.items():
        _console.print(
            f"  {task_id:24} success={outcome.success_rate:.2f} "
            f"assertions={outcome.mean_fraction:.2f}"
        )
    _console.print(
        f"[bold]{spec.name}[/bold]: success_rate={report.success_rate:.3f}"
        f"±{report.success_std:.3f}, mean_assertion_fraction={report.mean_fraction:.3f}"
    )


@agent_app.command("evolve")
def evolve_harness(
    tasks_file: str = _TASKS_ARG,
    name: str = typer.Option(
        None, "--name", help="World model to evolve against (default: only one)."
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding the world model."),
    generations: int = typer.Option(5, help="Mutation→eval rounds."),
    k: int = typer.Option(3, help="Passes per task per variant."),
    skills: str = typer.Option(None, "--skills", help="Skill library dir to seed from."),
    out: str = typer.Option(None, "--out", help="Path to write the winning HarnessSpec JSON."),
    archive_out: str = typer.Option(None, "--archive", help="Path to write the full archive JSON."),
) -> None:
    """Evolve harness variants: mutate from failure feedback, keep the best by closed-loop delta."""
    from wmh.agent.evolve import evolve, save_archive
    from wmh.agent.gold import GoldJudge
    from wmh.agent.skills import SkillLibrary
    from wmh.agent.spec import HarnessSpec
    from wmh.agent.tasks import load_tasks
    from wmh.cli.agent_support import load_world_model_for_agent

    tasks = load_tasks(tasks_file)
    world_model, provider = load_world_model_for_agent(name, root)
    library = SkillLibrary(skills) if skills else None
    judge = GoldJudge(provider)

    def _progress(gen: int, variant: str, score: float) -> None:
        tag = "seed" if gen == 0 else f"gen {gen}"
        _console.print(f"  [{tag}] {variant}: success_rate={score:.3f}")

    _console.print(f"evolving over {generations} generation(s), k={k}, on {len(tasks)} task(s)…")
    result = evolve(
        HarnessSpec(),
        tasks,
        world_model,
        provider,
        provider,
        judge,
        generations=generations,
        k=k,
        library=library,
        on_progress=_progress,
    )
    _console.print(
        f"[green]best[/green]: [bold]{result.best.name}[/bold] "
        f"(success_rate={result.best_score:.3f}) — {result.best.motivation}"
    )
    _console.print(f"  lineage: {_lineage(result)}")
    if out:
        from pathlib import Path

        Path(out).write_text(result.best.model_dump_json(indent=2), encoding="utf-8")
        _console.print(f"  wrote winning spec -> {out}")
    if archive_out:
        save_archive(result.archive, archive_out)
        _console.print(f"  wrote archive ({len(result.archive.entries)} variants) -> {archive_out}")


@agent_app.command("verify")
def verify(
    tasks_file: str = _TASKS_ARG,
    name: str = typer.Option(
        None, "--name", help="World model to check against (default: only one)."
    ),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding the world model."),
    archive: str = typer.Option(
        None, "--archive", help="Evolve archive JSON; verifies every variant it holds."
    ),
    spec_file: str = typer.Option(
        None, "--spec", help="Also verify this HarnessSpec JSON (e.g. an evolve winner)."
    ),
    limit: int = typer.Option(
        0, help="Cap the number of variants verified (0 = all). Cost control."
    ),
    k: int = typer.Option(3, help="Passes per task per variant, per world."),
    threshold: float = typer.Option(0.5, help="Pass threshold on a task's k-pass success rate."),
    provider: str = typer.Option("bedrock", "--provider", help="Agent model provider."),
    model: str = typer.Option("us.anthropic.claude-opus-4-8", help="Agent model id."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    skills: str = typer.Option(None, "--skills", help="Skill library dir to seed from."),
    template: str = typer.Option(None, "--template", help="E2B template id (default: base image)."),
    out: str = typer.Option(None, "--out", help="Write the full AgreementReport JSON here."),
) -> None:
    """Sim-real validity check: score variants in the world model AND real E2B, report agreement.

    The headline is whether the world model ranks harnesses like reality (rank correlation) and how
    often its per-task pass/fail matches (outcome agreement). Needs `uv sync --extra e2b` +
    `$E2B_API_KEY` for the real leg. This is the number that says whether sim-driven evolution
    transfers — run it on an evolve `--archive` after evolving.
    """
    from wmh.agent.agreement import sim_real_agreement
    from wmh.agent.gold import GoldJudge
    from wmh.agent.skills import SkillLibrary
    from wmh.agent.tasks import load_tasks
    from wmh.cli.agent_support import load_world_model_for_agent

    tasks = load_tasks(tasks_file)
    specs = _gather_specs(archive, spec_file)
    if limit > 0:
        specs = specs[:limit]
    if len(specs) < 2:
        _console.print(
            "[yellow]note[/yellow]: fewer than 2 variants — rank correlation will be n/a; "
            "pass an evolve --archive to compare a population"
        )
    world_model, _wm_provider = load_world_model_for_agent(name, root)
    agent_provider = _load_agent_provider(provider, model, region)
    library = SkillLibrary(skills) if skills else None
    judge = GoldJudge(agent_provider)

    _console.print(
        f"verifying {len(specs)} variant(s) on {len(tasks)} task(s), k={k}, "
        f"in the world model AND real E2B…"
    )
    report = sim_real_agreement(
        specs,
        tasks,
        world_model,
        agent_provider,
        judge,
        library=library,
        k=k,
        pass_threshold=threshold,
        e2b_template=template,
    )
    _print_agreement(report)
    if out:
        from pathlib import Path

        Path(out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        _console.print(f"  wrote agreement report -> {out}")


def _gather_specs(archive: str | None, spec_file: str | None):  # noqa: ANN202 - list[HarnessSpec]
    """Collect the variants to verify: every variant in the archive, plus an optional extra spec.

    Falls back to the baseline HarnessSpec when neither is given, so `verify` always has something
    to run (though a single variant yields no rank correlation).
    """
    from pathlib import Path

    from wmh.agent.evolve import load_archive
    from wmh.agent.spec import HarnessSpec

    by_name: dict[str, HarnessSpec] = {}
    if archive is not None:
        for entry in load_archive(archive).entries:
            by_name.setdefault(entry.spec.name, entry.spec)
    if spec_file is not None:
        spec = HarnessSpec.model_validate_json(Path(spec_file).read_text(encoding="utf-8"))
        # An explicitly-passed --spec WINS over an archive variant of the same name (the file may be
        # a hand-edited winner): override, and say so, rather than silently verifying the archived
        # copy or dropping the request.
        if spec.name in by_name:
            _console.print(
                f"[yellow]note[/yellow]: --spec '{spec.name}' overrides the archive entry by name"
            )
        by_name[spec.name] = spec
    specs = list(by_name.values())
    if not specs:
        specs.append(HarnessSpec())
    return specs


def _print_agreement(report) -> None:  # noqa: ANN001 - AgreementReport
    """Render the agreement scorecard: per-variant sim-vs-real, the 2x2 confusion, and headlines."""
    _console.print("\n[bold]per-variant success (sim vs real)[/bold]:")
    for v in sorted(report.per_variant, key=lambda x: x.real_success, reverse=True):
        flag = "  [red]<- sim over-credits[/red]" if v.gap > 0.15 else ""
        _console.print(
            f"  {v.harness:20} sim={v.sim_success:.3f}  real={v.real_success:.3f}  "
            f"gap={v.gap:+.3f}{flag}"
        )
    c = report.confusion
    _console.print("\n[bold]outcome confusion (variant×task cells)[/bold]:")
    _console.print(f"  sim-pass & real-pass: {c.sim_pass_real_pass}")
    _console.print(
        f"  sim-pass & real-FAIL: {c.sim_pass_real_fail}  (mirage: evolution chases this)"
    )
    _console.print(f"  sim-FAIL & real-pass: {c.sim_fail_real_pass}")
    _console.print(f"  sim-FAIL & real-FAIL: {c.sim_fail_real_fail}")
    if report.failed_variants:
        _console.print(
            f"\n[yellow]{len(report.failed_variants)} variant(s) skipped (raised)[/yellow]: "
            f"{', '.join(report.failed_variants)}"
        )
    rc = "n/a" if report.rank_correlation is None else f"{report.rank_correlation:.3f}"
    oa = "n/a (no cells)" if report.outcome_agreement is None else f"{report.outcome_agreement:.3f}"
    _console.print(
        f"\n[bold]VERDICT[/bold] outcome_agreement={oa}  rank_correlation={rc}  "
        f"mean_abs_gap={report.mean_abs_gap:.3f}  ({c.total} cells)"
    )


@agent_app.command("gate")
def gate(
    tasks_file: str = _TASKS_ARG,
    provider: str = typer.Option("bedrock", "--provider", help="Agent model provider."),
    model: str = typer.Option("us.anthropic.claude-opus-4-8", help="Agent model id."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    k: int = typer.Option(3, help="Passes per task."),
    threshold: float = typer.Option(
        0.5, help="Min real success rate for a task to be admitted (>= this passes the gate)."
    ),
    max_turns: int = typer.Option(20, help="Agent turn cap per task."),
    template: str = typer.Option(None, "--template", help="E2B template id (default: base image)."),
    out: str = typer.Option(None, "--out", help="Write the gate result (per-task JSON) here."),
) -> None:
    """Oracle-gate a task suite: run the baseline agent for real in E2B and flag unreliable tasks.

    A task nobody can solve — or whose gold assertions never fire on a genuine success — is a broken
    eval that corrupts every downstream score. Terminal-Bench admits a task only if a reference
    solution passes its verifier; here the strong baseline agent is that reference. Tasks below
    `--threshold` are reported as NOT admitted so you can fix or drop them before evolving.
    Needs `uv sync --extra e2b` + `$E2B_API_KEY`.
    """
    from wmh.agent.gold import GoldJudge
    from wmh.agent.real_loop import evaluate_real
    from wmh.agent.spec import HarnessSpec
    from wmh.agent.tasks import load_tasks

    tasks = load_tasks(tasks_file)
    agent_provider = _load_agent_provider(provider, model, region)
    judge = GoldJudge(agent_provider)
    spec = HarnessSpec(max_turns=max_turns)

    _console.print(f"oracle-gating {len(tasks)} task(s) with the baseline agent in E2B, k={k}…")
    report = evaluate_real(spec, tasks, agent_provider, judge, k=k, template=template)
    admitted, rejected = [], []
    for task_id, outcome in report.per_task.items():
        ok = outcome.success_rate >= threshold
        (admitted if ok else rejected).append((task_id, outcome.success_rate))
        mark = "[green]admit[/green]" if ok else "[red]REJECT[/red]"
        _console.print(f"  {mark} {task_id:20} real_success={outcome.success_rate:.2f}")
    _console.print(
        f"\n[bold]{len(admitted)}/{len(tasks)} admitted[/bold]"
        + (f"; fix or drop: {', '.join(t for t, _ in rejected)}" if rejected else "")
    )
    if out:
        import json
        from pathlib import Path

        payload = {
            "k": k,
            "threshold": threshold,
            "admitted": [t for t, _ in admitted],
            "rejected": [t for t, _ in rejected],
            "per_task": {tid: o.success_rate for tid, o in report.per_task.items()},
        }
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _console.print(f"  wrote gate result -> {out}")


def _read_spec(spec_file: str | None):  # noqa: ANN202 - HarnessSpec
    from pathlib import Path

    from wmh.agent.spec import HarnessSpec

    if spec_file is None:
        return HarnessSpec()
    return HarnessSpec.model_validate_json(Path(spec_file).read_text(encoding="utf-8"))


def _lineage(result) -> str:  # noqa: ANN001 - EvolveResult
    """Trace the winner's ancestry chain by name for a legible audit line."""
    by_name = {e.spec.name: e.spec for e in result.archive.entries}
    chain: list[str] = []
    node = result.best
    seen: set[str] = set()
    while node is not None and node.name not in seen:
        chain.append(node.name)
        seen.add(node.name)
        node = by_name.get(node.parent) if node.parent else None
    return " <- ".join(chain)

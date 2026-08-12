"""Open-loop evaluation command and report formatting helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from wmo.common.config import (
    ARTIFACT_DIR,
)

if TYPE_CHECKING:
    from wmo.common.providers import ProviderConfig
    from wmo.simulation.evaluation.open_loop import EvalReport

from wmo.cli.catalog_cmd import _prepare_out_path
from wmo.cli.command_common import _worker_role_provider_config

_console = Console()
_EVAL_TOKENS = typer.Argument(
    None,
    help="Trace files to score, or `agreement <a.json> <b.json>`.",
)


@dataclass(frozen=True)
class _EvalOptions:
    prompt_file: str | None
    train_split: float
    embed_dim: int
    use_rag: bool
    sample_turns: str
    seed: int
    top_k: int
    knowledge: bool
    reasoning: bool


_CLOSED_LOOP_ONLY_FLAGS = (
    ("name", "--name"),
    ("root", "--root"),
    ("k", "--k"),
    ("max_turns", "--max-turns"),
    ("harness", "--harness"),
    ("harness_backend", "--harness-backend"),
    ("eval_concurrency", "--eval-concurrency"),
    ("e2b_template", "--e2b-template"),
)


def eval_(  # noqa: A001 - `eval` is the user-facing command name; the builtin isn't used here
    ctx: typer.Context,
    tokens: list[str] | None = _EVAL_TOKENS,
    mode: str = typer.Option(
        "open-loop",
        "--mode",
        help="open-loop: replay traces, score per-step fidelity (default). "
        "closed-loop: a live agent runs tasks with the world model as its environment.",
    ),
    prompt_file: str | None = typer.Option(
        None, "--prompt", help="Prompt file; default=BASE_ENV_PROMPT."
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Provider running the model. Default: the worker role `wmo providers set` wrote to "
        "`.wmo/settings.toml`, else bedrock.",
    ),
    model: str | None = typer.Option(
        None,
        help="Canonical model type. Default: the configured worker role's model, else the "
        "flagship of whichever provider is in play (bedrock/claude-opus-4-8).",
    ),
    region: str | None = typer.Option(None, help="AWS region (Bedrock)."),
    chain: str | None = typer.Option(
        None, "--chain", help="Named failover chain from .wmo/fallback.toml (default: its default)."
    ),
    train_split: float | None = typer.Option(
        None,
        help="Train/holdout ratio per file (default: 0.8). "
        "Must match the ratio the model was built with, or the holdout contains train traces.",
    ),
    embed_dim: int | None = typer.Option(
        None, help="phi dimensionality for the offline embedder (default: 512)."
    ),
    rag: bool | None = typer.Option(
        None, "--rag/--no-rag", help="Enable retrieval, or disable it for zero-shot replay."
    ),
    sample_turns: str | None = typer.Option(
        None, help="Turns scored per trace: all | sampled (5). Default: all."
    ),
    seed: int | None = typer.Option(None, help="Seed for reproducible turn sampling."),
    top_k: int | None = typer.Option(
        None, help="Retrieved demos per step (default: 5, or suite config)."
    ),
    knowledge: bool | None = typer.Option(
        None,
        "--knowledge/--no-knowledge",
        help="Seed a knowledge base from the train split and render it into every prediction.",
    ),
    reasoning: bool | None = typer.Option(
        None,
        "--reasoning/--no-reasoning",
        help="Deliberate-then-answer output contract (explicit reasoning pass).",
    ),
    out: str | None = typer.Option(None, help="Optional path to write the full JSON report."),
    name: str | None = typer.Option(
        None, "--name", help="World model for --mode closed-loop (default: the only built one)."
    ),
    root: str = typer.Option(
        ARTIFACT_DIR, help="Project dir holding world models (--mode closed-loop)."
    ),
    k: int = typer.Option(
        3, min=1, help="Closed-loop passes per task (means reported, never 1-pass)."
    ),
    max_turns: int | None = typer.Option(
        None, min=1, help="Closed-loop agent turn cap (default: 20, or the harness's own)."
    ),
    threshold: float = typer.Option(
        0.5, help="Agreement pass threshold on a task's k-pass success rate."
    ),
    harness: str | None = typer.Option(
        None, "--harness", help="Stored harness to run for `closed-loop` (default: baseline)."
    ),
    harness_backend: str = typer.Option(
        "local",
        "--harness-backend",
        help="Where the closed-loop harness PROCESS runs: local (in/from this process) or e2b "
        "(the pi-node harness inside pooled E2B sandboxes). The environment is always the "
        "world model.",
    ),
    eval_concurrency: int | None = typer.Option(
        None,
        "--eval-concurrency",
        min=0,
        help="Closed-loop (task, attempt) cells run at once. Default: 1 for local; "
        "0 (= all cells at once) for e2b.",
    ),
    e2b_template: str | None = typer.Option(
        None,
        "--e2b-template",
        envvar="WMO_E2B_TEMPLATE",
        help="Prebaked E2B sandbox template for --harness-backend e2b (default: "
        "$WMO_E2B_TEMPLATE; without one, sandboxes bootstrap node + the pi runner deps on "
        "first use).",
    ),
) -> None:
    r"""Score reconstruction fidelity or compare two closed-loop reports.

    Flows:
    - `wmo eval <trace files...>`: ad hoc replay scoring (open-loop, teacher-forced, the
      default mode).
    - `wmo eval <tasks.jsonl> --mode closed-loop`: a live agent runs tasks WITH the world model
      as its environment. `\[models.agent]` selects a distinct agent provider when configured;
      otherwise the agent shares the world model's provider. `--harness-backend e2b` moves the
      pi-node harness process into pooled E2B sandboxes while the environment stays the world
      model. Score task success against gold assertions (see docs/reference/closed_loop.md).
    - `wmo eval agreement <a.json> <b.json>`: compare two closed-loop reports task-by-task
      (for example, world-model versus real environment), the outcome-agreement validity check.

    Open-loop scoring runs on the worker role `wmo providers set` writes to `.wmo/settings.toml`
    (bedrock/claude-opus-4-8 when no role is configured); `--provider`/`--model` override it.

    Args:
        ctx: Typer context used to distinguish explicitly supplied options.
        tokens: Trace files, or the `agreement` report pair.
        mode: Open-loop or closed-loop evaluation protocol.
        root: Project artifact directory used to resolve named models.
        out: Optional file for the evaluation result.

    Raises:
        typer.BadParameter: Evaluation inputs or selected protocol options are invalid.
    """
    from wmo.cli.eval_closed_loop import run_agreement, run_closed_loop
    from wmo.cli.ui import explicit_param as _explicit
    from wmo.common.observability.telemetry import capture_eval_completed

    args = tokens or []
    if mode not in ("open-loop", "closed-loop"):
        raise typer.BadParameter(f"unknown --mode {mode!r}; choose open-loop or closed-loop")
    # Reject flags this flow will never read, before anything runs: a silently dropped
    # --harness/--k means the user paid for an open-loop run they did not ask for.
    if mode != "closed-loop":
        inapplicable = [flag for param, flag in _CLOSED_LOOP_ONLY_FLAGS if _explicit(ctx, param)]
        if inapplicable:
            raise typer.BadParameter(
                f"{', '.join(inapplicable)} apply only to `--mode closed-loop`; add that flag "
                "or drop them"
            )
    if (not args or args[0] != "agreement") and _explicit(ctx, "threshold"):
        raise typer.BadParameter(
            "--threshold applies only to `wmo eval agreement <a.json> <b.json>`; drop it"
        )
    # --out is written last (after the paid work); make sure it is writable first.
    _prepare_out_path(out)
    if mode == "closed-loop":
        if len(args) != 1 or args[0] == "agreement":
            raise typer.BadParameter("usage: wmo eval <tasks.jsonl> --mode closed-loop")
        run_closed_loop(
            _console,
            tasks_file=args[0],
            name=name,
            root=root,
            k=k,
            max_turns=max_turns,
            out=out,
            harness=harness,
            harness_backend=harness_backend,
            eval_concurrency=eval_concurrency,
            e2b_template=e2b_template,
        )
        return
    if args and args[0] == "agreement":
        if len(args) != 3:
            raise typer.BadParameter("usage: wmo eval agreement <report_a.json> <report_b.json>")
        run_agreement(_console, report_a=args[1], report_b=args[2], threshold=threshold)
        return
    if not args:
        raise typer.BadParameter(
            "provide trace files, or use `wmo eval agreement <a.json> <b.json>`"
        )

    options = _eval_options(
        prompt_file=prompt_file,
        train_split=train_split,
        embed_dim=embed_dim,
        rag=rag,
        sample_turns=sample_turns,
        seed=seed,
        top_k=top_k,
        knowledge=knowledge,
        reasoning=reasoning,
    )
    report = _run_eval_files(
        [Path(f) for f in args],
        options,
        provider_config=_worker_role_provider_config(provider, model, region),
        chain=chain,
    )
    _require_scorable_steps(
        report,
        args,
        next_step=f"check the export with `wmo ingest --file {args[0]}`, or add "
        "`--mode closed-loop` for a tasks file",
    )
    _print_eval_report(report)
    if out:
        _write_ad_hoc_eval_report(Path(out), report)
    capture_eval_completed(
        mode="ad_hoc",
        file_count=len(args),
        scored_step_count=report.total_valid,
        rag_enabled=options.use_rag,
        sample_turns=options.sample_turns,
        train_split=options.train_split,
        top_k=options.top_k,
        root=ARTIFACT_DIR,
    )


# Default grid: one bar family per serving model. Qwen-AgentWorld is self-hosted (openai-compatible
# vLLM via OPENAI_BASE_URL); the rest are frontier models the registry builds directly.
def _eval_options(
    *,
    prompt_file: str | None,
    train_split: float | None,
    embed_dim: int | None,
    rag: bool | None,
    sample_turns: str | None,
    seed: int | None,
    top_k: int | None,
    knowledge: bool | None = None,
    reasoning: bool | None = None,
) -> _EvalOptions:
    from wmo.simulation.model.build import DEFAULT_TRAIN_SPLIT

    split = DEFAULT_TRAIN_SPLIT if train_split is None else train_split
    dim = 512 if embed_dim is None else embed_dim
    retrieval = True if rag is None else rag
    turns = "all" if sample_turns is None else sample_turns
    rng_seed = 0 if seed is None else seed
    demos = 5 if top_k is None else top_k
    if not 0.0 < split < 1.0:
        raise typer.BadParameter("--train-split must be between 0 and 1")
    if dim <= 0:
        raise typer.BadParameter("--embed-dim must be positive")
    if demos < 0:
        raise typer.BadParameter("--top-k must be >= 0")
    if turns not in {"all", "sampled"}:
        raise typer.BadParameter("--sample-turns must be one of: all, sampled")
    return _EvalOptions(
        prompt_file=prompt_file,
        train_split=split,
        embed_dim=dim,
        use_rag=retrieval,
        sample_turns=turns,
        seed=rng_seed,
        top_k=demos,
        knowledge=bool(knowledge),
        reasoning=bool(reasoning),
    )


def _run_eval_files(
    files: list[Path],
    options: _EvalOptions,
    *,
    provider_config: ProviderConfig,
    chain: str | None = None,
) -> EvalReport:
    import wmo.common.providers as providers
    from wmo.optimize.judge import RubricJudge
    from wmo.simulation.evaluation.open_loop import OpenLoopEval
    from wmo.simulation.model.prompts import BASE_ENV_PROMPT
    from wmo.simulation.retrieval import HashingEmbedder

    for path in files:
        if not path.exists():
            raise typer.BadParameter(f"trace file not found: {path}")
        if not path.is_file():
            raise typer.BadParameter(
                f"not a trace file: {path} is a directory; pass the export itself "
                f"(e.g. `wmo eval {path}/traces.otel.jsonl`)"
            )
    # Name the backend: the number below is only comparable against runs on the same model, and
    # with no flags the model comes from settings rather than the command line.
    _console.print(
        f"scoring with {provider_config.kind.value} "
        f"({provider_config.model_type or provider_config.model})"
    )
    # A missing/unknown chain, or a multi-chain fallback.toml with no `default`, is a usage
    # error: the message already names the file, so it must not arrive as a traceback.
    try:
        llm = providers.provider_or_chain(provider_config, chain=chain)
    except (ValueError, FileNotFoundError) as err:
        raise typer.BadParameter(str(err)) from None
    if isinstance(llm, providers.WaterfallProvider):
        _console.print("failover chain active (.wmo/fallback.toml) - world-model calls only")
    prompt = (
        Path(options.prompt_file).read_text(encoding="utf-8")
        if options.prompt_file
        else BASE_ENV_PROMPT
    )
    embedder = HashingEmbedder(dim=options.embed_dim) if options.use_rag else None
    # The judge is the metric: it stays PINNED to the single requested backend and never rides
    # the failover chain - a judge that silently switches models mid-run makes fidelity numbers
    # incomparable across steps. World-model prediction calls (above) may fail over freely.
    scorer = RubricJudge(providers.get_provider(provider_config))
    evaluation = OpenLoopEval(
        files,
        prompt,
        llm,
        scorer,
        embedder=embedder,
        train_split=options.train_split,
        top_k=options.top_k,
        sample_turns=options.sample_turns,
        seed=options.seed,
        knowledge=options.knowledge,
        reasoning=options.reasoning,
    )
    return evaluation.run()


def _require_scorable_steps(report: EvalReport, paths: list[str], *, next_step: str) -> None:
    """Refuse to print (or persist) a 0.000 scorecard for input that held nothing to score.

    `evaluate_files` skips a file that yields no OTel GenAI traces, so a wrong file type, a
    tasks.jsonl missing `--mode closed-loop`, or a train_split that leaves no holdout all used to
    render as a plausible `OVERALL fidelity=0.000 over 0 held-out steps` at exit 0. The command
    now stops before it can print or persist that misleading scorecard.

    `next_step` is the caller's flow-specific remedy, since the ad-hoc path can suggest
    `--mode closed-loop` for a tasks file and the suite path cannot.
    """
    if report.total_steps:
        return
    listed = ", ".join(paths)
    if not report.per_file:
        raise typer.BadParameter(f"no OTel GenAI traces in {listed}; {next_step}")
    raise typer.BadParameter(
        f"no held-out steps to score in {listed}; lower `--train-split` (currently reserving "
        "every trace for training) or pass a larger corpus"
    )


def _print_eval_report(report: EvalReport) -> None:
    for name, rep in report.per_file.items():
        _console.print(f"  {name:28} {rep.summary()}")
    invalid = f" ({report.total_invalid} judge-invalid excluded)" if report.total_invalid else ""
    _console.print(
        f"[bold]OVERALL[/bold] fidelity={report.overall_fidelity:.3f}±{report.overall_std:.3f} "
        f"over {report.total_steps} held-out steps{invalid}"
    )


def _write_ad_hoc_eval_report(path: Path, report: EvalReport) -> None:
    path.write_text(
        json.dumps({n: r.model_dump(mode="json") for n, r in report.per_file.items()}, indent=2),
        encoding="utf-8",
    )
    _console.print(f"wrote full report -> {path}")

#!/usr/bin/env python
"""Live runner for the GEPA scaling law: fidelity vs. GEPA iterations and vs. training traces.

The SIDECAR for `wmh.research.GepaScalingAblation` — it resolves a corpus, holds the same fixed
test/valid split as the trace scaling law, and sweeps an (n_train × budget) grid: for each point,
GEPA optimizes the base prompt for `budget` iterations on `n_train` traces (budget=0 = no GEPA, the
RAG-only anchor that reproduces the trace scaling law's point), then the winner is replay-scored on
the fixed test set. Mean ± std across seeds at each point.

    # budget axis at fixed n=64 (the primary figure; optimize on 4.7 to dodge 4.8 throttling):
    AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/run_gepa_scaling.py \
        tau-bench --counts 64 --budgets 0,1,2,4,8,16 --seeds 0,1 --sample-turns sampled \
        --test-cap 40 --concurrency 8 --opt-model us.anthropic.claude-opus-4-7 --out tau_budget.json
    # trace axis at fixed budget 8 (the secondary axis):
    AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/run_gepa_scaling.py \
        tau-bench --counts 1,4,16,648 --budgets 8 --seeds 0,1 --sample-turns sampled \
        --test-cap 40 --concurrency 8 --opt-model us.anthropic.claude-opus-4-7 --out tau_traces.json

The grid is the cartesian product of --counts and --budgets (points dedupe after pool-capping, so
overlapping runs can also be merged from separate JSONs). `--hard-threshold` pre-scores a probe of
the train sample + the GEPA valset with the base prompt and concentrates GEPA's reflection AND
candidate selection on the steps scoring below it — the fix for near-saturated benchmarks where a
random reflection minibatch is usually "all perfect -> skipping". Resolves the corpus from an
**eval suite** name (`tau-bench`) or a raw `--file`. The judge defaults to Opus 4.8.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from wmh.engine.eval_suites import resolve_eval_suite
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.ingest import get_adapter
from wmh.optimize.judge import JUDGE_VERSION, Judge, RubricJudge
from llm_waterfall import RetryPolicy

from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Embedder, Provider
from wmh.providers.waterfall import WaterfallProvider
from wmh.research import GepaScalingAblation, run_ablation
from wmh.research.ablation import AblationReport, Condition
from wmh.retrieval import HashingEmbedder
from wmh.tracking.metered import MeteredProvider, classify_build_call
from wmh.tracking.tracker import RunTracker

# The shared failover LADDER (engaged on capacity errors only, in this order) appended after each
# role's primary: endflow-account Opus 4.6 and Sonnet 4.6 (a separate AWS account = an independent
# Bedrock quota pool), then the default account's Opus 4.8 / 4.7 / 4.6, then GPT 5.5 (only when
# OPENAI_API_KEY is set — the `openai/` prefix routes to the OpenAI provider). The stackwise-agent
# IAM user has no bedrock:InvokeModel (checked 2026-07-02), so the "stackwise" links run on the
# `default` profile of the same AWS account. Duplicates of a role's primary are dropped at build.
FALLBACK_LADDER = (
    "us.anthropic.claude-opus-4-6-v1@endflow",
    "us.anthropic.claude-sonnet-4-6@endflow",
    "us.anthropic.claude-opus-4-8",
    "us.anthropic.claude-opus-4-7",
    # NB: the 4.6 inference-profile id carries "-v1" on BOTH accounts — the bare
    # "...claude-opus-4-6" is rejected with ValidationException (invalid id), which is a
    # non-capacity error and would crash the chain (it killed the first tau budget sweep at
    # its final point during a 4.7+4.8 brownout cascade). Every id here is invoke-verified.
    "us.anthropic.claude-opus-4-6-v1",
    "openai/gpt-5.5",
)


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_rung(spec: str, region: str | None) -> tuple[ProviderConfig, str | None] | None:
    """Parse one chain rung: `model[@aws_profile]` (Bedrock) or `openai/model` -> (config, profile).

    Returns None for an OpenAI rung when OPENAI_API_KEY is absent (skipped rather than adding a
    guaranteed-to-fail tail that would mask the real capacity error).
    """
    if spec.startswith("openai/"):
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        return ProviderConfig(kind=ProviderKind.OPENAI, model=spec.removeprefix("openai/")), None
    model, _, profile = spec.partition("@")
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model=model, region=region)
    return config, (profile or None)


def _chain(primary: str, region: str | None, ladder: bool) -> Provider:
    """The primary provider, optionally followed by the shared failover ladder (WaterfallProvider).

    Failover happens on capacity errors only. `retry` wraps the whole chain in bounded
    backoff rounds so a simultaneous brownout across every Bedrock rung (the cascade that killed
    the first tau budget sweep) rides out the window instead of raising WaterfallExhausted on the
    first pass. Ladder entries equal to
    the primary spec are dropped (no point retrying the same account+model immediately); OpenAI
    rungs without a key are skipped. A single rung with an @profile still goes through
    WaterfallProvider — wmh's bare BedrockProvider has no profile support and would silently run
    on the default account.
    """
    specs = [primary] + [s for s in FALLBACK_LADDER if ladder and s != primary]
    rungs = [r for r in (_parse_rung(s, region) for s in specs) if r is not None]
    if len(rungs) == 1 and rungs[0][1] is None:
        return get_provider(rungs[0][0])
    return WaterfallProvider(
        [c for c, _ in rungs],
        profiles=[p for _, p in rungs],
        retry=RetryPolicy(rounds=6, backoff_base_s=15.0),
    )


def _make_backends(
    judge_model: str,
    opt_model: str,
    region: str | None,
    embed_dim: int,
    no_rag: bool,
    ladder: bool,
    tracker: RunTracker,
) -> Callable[[], tuple[Provider, Judge, Embedder | None]]:
    """Factory the ablation calls per run for (provider, judge, embedder).

    `provider` is the model GEPA optimizes/serves with — `opt_model` as primary (4.7 dodges 4.8's
    Bedrock throttling on GEPA's many rollouts) followed by the shared `FALLBACK_LADDER` so a
    throttled call degrades down the chain instead of crashing a long run or scoring 0. The judge
    is `judge_model` (4.8, so fidelity stays comparable to the rest of the harness) over the same
    ladder. The embedder is the offline HashingEmbedder (no creds) unless --no-rag.
    """
    serve: Provider = MeteredProvider(
        _chain(opt_model, region, ladder), tracker, classify=classify_build_call
    )
    judge_provider: Provider = MeteredProvider(
        _chain(judge_model, region, ladder), tracker, classify=classify_build_call
    )
    scorer: Judge = RubricJudge(judge_provider)  # the single post-#83 judge (JUDGE_VERSION)
    embedder: Embedder | None = None if no_rag else HashingEmbedder(dim=embed_dim)

    def factory() -> tuple[Provider, Judge, Embedder | None]:
        return serve, scorer, embedder

    return factory


def _load_corpus(args: argparse.Namespace) -> tuple[list, str]:  # noqa: ANN201 - (traces, label)
    """Resolve the corpus from an eval-suite name (preferred) or a raw --file -> (traces, label)."""
    adapter = get_adapter("otel-genai")
    if args.suite:
        suite = resolve_eval_suite(args.suite, args.examples)
        files = suite.resolve_files()
        traces = [t for f in files for t in adapter.from_file(str(f))]
        return traces, args.suite
    if not args.file:
        raise SystemExit("pass an eval-suite name or --file <trace.jsonl>")
    return adapter.from_file(args.file), Path(args.file).name


def _run(args: argparse.Namespace, tracker: RunTracker) -> AblationReport:
    traces, label = _load_corpus(args)
    if not traces:
        raise SystemExit("no traces ingested")

    seeds = _parse_ints(args.seeds)
    grid = [(n, b) for n in _parse_ints(args.counts) for b in _parse_ints(args.budgets)]
    ablation = GepaScalingAblation(
        traces,
        BASE_ENV_PROMPT,
        make_backends=_make_backends(
            args.judge_model,
            args.opt_model,
            args.region,
            args.embed_dim,
            args.no_rag,
            not args.no_fallback,
            tracker,
        ),
        grid=grid,
        top_k=args.top_k,
        test_frac=args.test_frac,
        valid_frac=args.valid_frac,
        gepa_val_steps=args.gepa_val_steps,
        val_fill=args.val_fill,
        recheck_steps=args.recheck_steps,
        minibatch_size=args.minibatch,
        hard_threshold=args.hard_threshold,
        hard_probe_steps=args.hard_probe_steps,
        sample_turns=args.sample_turns,
        test_cap=args.test_cap,
        concurrency=args.concurrency,
    )
    split = ablation.split
    scored = len(ablation.scored_test)
    test_note = f"test {len(split.test)}"
    if scored != len(split.test):
        test_note += f" (scoring {scored})"
    gepa_val_steps = sum(len(t.steps) for t in ablation.gepa_valid)
    print(
        f"corpus {label}: {len(traces)} traces -> "
        f"train pool {len(split.train_pool)}, valid {len(split.valid)}, {test_note}"
    )
    print(
        f"grid={ablation.grid}, seeds={seeds}, "
        f"gepa valset {len(ablation.gepa_valid)} traces / {gepa_val_steps} steps, "
        f"hard-threshold={args.hard_threshold}\n"
        f"opt-model={args.opt_model}, judge={JUDGE_VERSION} ({args.judge_model})\n"
    )

    def _progress(condition: Condition, seed: int, score: float) -> None:
        print(f"  {condition.label:14} seed={seed}  fidelity={score:.3f}", flush=True)

    return run_ablation(ablation, seeds, on_run=_progress)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", help="Eval-suite name (e.g. tau-bench).")
    parser.add_argument("--file", default=None, help="Raw OTel trace file (instead of a suite).")
    parser.add_argument("--examples", default="examples", help="Examples root for suite lookup.")
    parser.add_argument("--counts", default="64", help="Comma-separated train-trace counts.")
    parser.add_argument(
        "--budgets", default="0,1,2,4,8,16", help="Comma-separated GEPA iteration budgets (0=off)."
    )
    parser.add_argument("--seeds", default="0,1", help="Comma-separated seeds (error bars).")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth.")
    parser.add_argument(
        "--sample-turns",
        default="sampled",
        help="Turns scored per test trace: all | sampled (Qwen 5-turn; cheaper, default).",
    )
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fixed test fraction.")
    parser.add_argument("--valid-frac", type=float, default=0.15, help="Fixed valid fraction.")
    parser.add_argument(
        "--gepa-val-steps",
        type=int,
        default=30,
        help="Step cap for GEPA's selection valset (its size multiplies per-iteration cost).",
    )
    parser.add_argument(
        "--minibatch",
        type=int,
        default=3,
        help="Reflection minibatch size (GEPA paper ~8; skip probability falls ~0.8^b).",
    )
    parser.add_argument(
        "--val-fill",
        default="greedy",
        choices=["greedy", "inclusive"],
        help="Selection valset fill: greedy (skip over-cap traces; short-trace-biased) | "
        "inclusive (take traces until the cap is reached; representative).",
    )
    parser.add_argument(
        "--recheck-steps",
        type=int,
        default=0,
        help="Guard v2: acceptance re-check on a valset-DISJOINT slice of the valid band capped "
        "at this many steps (0 = re-check on the valset itself).",
    )
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=None,
        help="Pre-score train probe + GEPA valset with the base prompt and concentrate GEPA on "
        "steps scoring below this (fix for 'all subsample perfect -> skipping').",
    )
    parser.add_argument(
        "--hard-probe-steps", type=int, default=40, help="Step cap for the hardness probe."
    )
    parser.add_argument(
        "--test-cap",
        type=int,
        default=None,
        help="Score a fixed seeded subsample of N test traces per point (bounds cost).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel test-step scoring (predict+judge are independent; 4.6/4.7 don't throttle).",
    )
    parser.add_argument(
        "--opt-model",
        default="us.anthropic.claude-opus-4-7",
        help="Primary Bedrock model GEPA optimizes/serves with (4.6/4.7 dodge 4.8 throttling); "
        "the shared failover ladder is appended unless --no-fallback.",
    )
    parser.add_argument(
        "--judge-model",
        default="us.anthropic.claude-opus-4-8",
        help="Primary Bedrock model for the judge; the shared failover ladder is appended "
        "unless --no-fallback.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Run the primaries bare (no failover ladder) — a capacity error then propagates.",
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region (Bedrock).")
    parser.add_argument("--embed-dim", type=int, default=512, help="phi dim (offline embedder).")
    parser.add_argument("--no-rag", action="store_true", help="Disable retrieval (zero-shot).")
    parser.add_argument("--out", default=None, help="Path to write the AblationReport JSON.")
    args = parser.parse_args()

    tracker = RunTracker(run_id="gepa-scaling", kind="research")
    with tracker.timed():
        report = _run(args, tracker)

    print(f"\n=== {report.name} (seeds={report.seeds}) ===")
    for cell in report.conditions:
        print(f"  {cell.summary()}")
    totals = tracker.totals()
    print(
        f"\nusage: {totals.calls} LLM calls, {totals.total_tokens} tokens, "
        f"${totals.cost_usd:.2f}, {tracker.duration_seconds():.0f}s"
    )
    for phase, t in tracker.by_phase().items():
        print(f"  {phase:12} calls={t.calls:6} tokens={t.total_tokens:9} ${t.cost_usd:.2f}")
    if args.out:
        Path(args.out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nwrote report -> {args.out}")


if __name__ == "__main__":
    main()

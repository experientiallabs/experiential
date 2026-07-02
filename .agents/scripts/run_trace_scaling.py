#!/usr/bin/env python
"""Live runner for the trace scaling law: fidelity vs. number of training traces.

The SIDECAR for `wmh.research.TraceScalingAblation` — it resolves a corpus, holds a fixed test/valid
split, and sweeps the TRAIN trace count (e.g. 10, 20, 50, … capped at the corpus) for one or both
modes (`base` = shipped prompt + RAG, `gepa` = GEPA-optimized per count), reporting test fidelity
mean ± std across seeds at each point. The curve says whether more traces keep buying fidelity or
saturate.

    # cheap base curve first, then the GEPA curve (optimize on Opus 4.7 to dodge 4.8 throttling):
    AWS_REGION=us-east-1 uv run python scripts/run_trace_scaling.py tau-bench \
        --counts 10,20,50,100,200,400 --modes base --seeds 0,1 --out scaling_base.json
    AWS_REGION=us-east-1 uv run python scripts/run_trace_scaling.py tau-bench \
        --counts 10,20,50,100,200,400 --modes gepa --budget 12 --seeds 0,1 \
        --opt-model us.anthropic.claude-opus-4-7 --out scaling_gepa.json

Resolves the corpus from an **eval suite** name (`tau-bench`) or a raw `--file`. The judge defaults
to Opus 4.8; `--opt-model` sets the model GEPA optimizes/serves with (4.6/4.7 are un-throttled).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from wmh.engine.eval_suites import resolve_eval_suite
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.ingest import get_adapter
from wmh.optimize.judge import Judge, LLMJudge, RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Embedder, Provider
from wmh.research import TraceScalingAblation, run_ablation
from wmh.research.ablation import AblationReport, Condition
from wmh.retrieval import HashingEmbedder

# Default scaling ladder: dense at the low end (where the curve bends), capped at the corpus by the
# ablation. Override with --counts.
DEFAULT_COUNTS = "10,20,50,100,200,400,800"


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_strs(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _make_backends(
    judge_model: str,
    opt_model: str,
    region: str | None,
    embed_dim: int,
    no_rag: bool,
    judge: str,
) -> Callable[[], tuple[Provider, Judge, Embedder | None]]:
    """Factory the ablation calls per run for (provider, judge, embedder).

    `provider` is the model GEPA optimizes/serves with — defaulting to `opt_model` (use 4.6/4.7 to
    avoid 4.8's Bedrock throttling). The judge is scored with `judge_model` so fidelity stays
    comparable to the rest of the harness. Both run on Bedrock; the embedder is the offline
    HashingEmbedder (no creds) unless --no-rag.
    """
    serve: Provider = get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=opt_model, region=region)
    )
    judge_provider: Provider = get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=judge_model, region=region)
    )
    scorer: Judge = RubricJudge(judge_provider) if judge == "rubric" else LLMJudge(judge_provider)
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


def _run(args: argparse.Namespace) -> AblationReport:
    traces, label = _load_corpus(args)
    if not traces:
        raise SystemExit("no traces ingested")

    seeds = _parse_ints(args.seeds)
    ablation = TraceScalingAblation(
        traces,
        BASE_ENV_PROMPT,
        make_backends=_make_backends(
            args.judge_model, args.opt_model, args.region, args.embed_dim, args.no_rag, args.judge
        ),
        counts=_parse_ints(args.counts),
        modes=_parse_strs(args.modes),
        budget=args.budget,
        top_k=args.top_k,
        test_frac=args.test_frac,
        valid_frac=args.valid_frac,
        sample_turns=args.sample_turns,
        test_cap=args.test_cap,
        concurrency=args.concurrency,
    )
    split = ablation.split
    scored = len(ablation.scored_test)
    test_note = f"test {len(split.test)}"
    if scored != len(split.test):
        test_note += f" (scoring {scored})"
    print(
        f"corpus {label}: {len(traces)} traces -> "
        f"train pool {len(split.train_pool)}, valid {len(split.valid)}, {test_note}"
    )
    print(
        f"counts={ablation.counts}, modes={args.modes}, seeds={seeds}, budget={args.budget}\n"
        f"opt-model={args.opt_model}, judge={args.judge} ({args.judge_model})\n"
    )

    def _progress(condition: Condition, seed: int, score: float) -> None:
        print(f"  {condition.label:14} seed={seed}  fidelity={score:.3f}", flush=True)

    return run_ablation(ablation, seeds, on_run=_progress)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", help="Eval-suite name (e.g. tau-bench).")
    parser.add_argument("--file", default=None, help="Raw OTel trace file (instead of a suite).")
    parser.add_argument("--examples", default="examples", help="Examples root for suite lookup.")
    parser.add_argument("--counts", default=DEFAULT_COUNTS, help="Comma-separated train counts.")
    parser.add_argument("--modes", default="base,gepa", help="Comma-separated: base, gepa.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds (error bars).")
    parser.add_argument("--budget", type=int, default=12, help="GEPA rollout budget (gepa mode).")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth.")
    parser.add_argument(
        "--sample-turns",
        default="sampled",
        help="Turns scored per test trace: all | sampled (Qwen 5-turn; cheaper, default).",
    )
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fixed test fraction.")
    parser.add_argument("--valid-frac", type=float, default=0.15, help="Fixed valid fraction.")
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
        help="Bedrock model GEPA optimizes/serves with (4.6/4.7 dodge 4.8 throttling).",
    )
    parser.add_argument(
        "--judge-model", default="us.anthropic.claude-opus-4-8", help="Bedrock model for the judge."
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region (Bedrock).")
    parser.add_argument("--embed-dim", type=int, default=512, help="phi dim (offline embedder).")
    parser.add_argument("--no-rag", action="store_true", help="Disable retrieval (zero-shot).")
    parser.add_argument("--judge", default="rubric", help="Scorer: rubric (5-dim) | match.")
    parser.add_argument("--out", default=None, help="Path to write the AblationReport JSON.")
    args = parser.parse_args()

    report = _run(args)

    print(f"\n=== {report.name} (seeds={report.seeds}) ===")
    for cell in report.conditions:
        print(f"  {cell.summary()}")
    if args.out:
        Path(args.out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nwrote report -> {args.out}")


if __name__ == "__main__":
    main()

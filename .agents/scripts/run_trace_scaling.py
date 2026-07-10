#!/usr/bin/env python
"""Live runner for the trace scaling law: fidelity vs. number of training traces.

The SIDECAR for `wmh.research.TraceScalingAblation` — it resolves a corpus, holds a fixed test/valid
split, and sweeps the TRAIN trace count (e.g. 10, 20, 50, … capped at the corpus) for one or both
modes (`base` = shipped prompt + RAG, `gepa` = GEPA-optimized per count), reporting test fidelity
mean ± std across seeds at each point. The curve says whether more traces keep buying fidelity or
saturate.

    # RAG-only base curve (the published run: log ladder, capped at the corpus, seeds 0/1):
    AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/run_trace_scaling.py \
        tau-bench --counts 1,4,16,64,256,648 --modes base --seeds 0,1 --sample-turns sampled \
        --test-cap 40 --concurrency 8 --opt-model us.anthropic.claude-opus-4-8 --out base.json
    # optional GEPA curve (optimize on Opus 4.7 to dodge 4.8 throttling):
    AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/run_trace_scaling.py \
        tau-bench --counts 1,4,16,64,256,648 --modes gepa --budget 12 --seeds 0,1 \
        --opt-model us.anthropic.claude-opus-4-7 --out scaling_gepa.json

Resolves the corpus from an **eval suite** name (`tau-bench`) or a raw `--file`. The judge defaults
to Opus 4.8; `--opt-model` sets the model GEPA optimizes/serves with (4.6/4.7 are un-throttled).
"""

from __future__ import annotations

import argparse
import uuid
import json
from collections.abc import Callable
from pathlib import Path

from wmh.engine.eval_suites import resolve_eval_suite
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.ingest import drop_degenerate_traces, get_adapter
from wmh.optimize.judge import Judge, RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider, provider_or_chain
from wmh.providers.base import Embedder, Provider
from wmh.providers.retry import RetryingProvider
from wmh.research import TraceScalingAblation, run_ablation
from wmh.research.ablation import AblationReport, Condition
from wmh.retrieval import HashingEmbedder
from wmh.tracking import MeteredProvider, Phase, RunRecord, RunTracker


# Default scaling ladder: log-spaced from a single trace up, capped at the corpus by the ablation
# (tau-bench tops out at its 648 pool; terminal-tasks/swe-bench at theirs). Override with --counts.
DEFAULT_COUNTS = "1,4,16,64,256,648"


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_strs(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


class _CachedEmbedder:
    """Memoize embeddings across runs: seeds re-index the SAME train steps."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self._inner.embed(missing), strict=True):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]


def _build_embedder(args: argparse.Namespace) -> Embedder | None:
    """The phi embedder: None (--no-rag), offline hashing, or Azure ada-002 (--embedder azure).

    Azure reads AZURE_OPENAI_API_KEY from the env; ada-002 has a fixed 1536-dim output (no
    `dimensions` param), so embed_dim is left None.
    """
    if args.no_rag:
        return None
    if args.embedder == "titan":
        # Bedrock Titan semantic phi — memoized: the search re-indexes the SAME train steps.
        return _CachedEmbedder(
            RetryingProvider(
                get_provider(
                    ProviderConfig(
                        kind=ProviderKind.BEDROCK,
                        model=args.opt_model,
                        region=args.region,
                        embed_model="amazon.titan-embed-text-v2:0",
                        embed_dim=512,
                    )
                )
            )
        )
    if args.embedder == "azure":
        return RetryingProvider(
            get_provider(
                ProviderConfig(
                    kind=ProviderKind.AZURE_OPENAI,
                    model=args.opt_model,
                    endpoint=args.embed_endpoint,
                    api_version=args.embed_api_version,
                    embed_model=args.embed_deployment,
                    embed_dim=None,
                )
            )
        )
    return HashingEmbedder(dim=args.embed_dim)


class _MeterBank:
    """One fresh serve-side RunTracker per ablation run (target cost per cell; judge separate)."""

    def __init__(self) -> None:
        self._pending: RunTracker | None = None
        self.by_run: list[tuple[str, int, RunRecord]] = []

    def fresh(self) -> RunTracker:
        tracker = RunTracker(run_id=uuid.uuid4().hex, kind="research")
        tracker.start()
        self._pending = tracker
        return tracker

    def take(self, label: str, seed: int) -> RunRecord | None:
        if self._pending is None:
            return None
        record = self._pending.record_summary()
        self.by_run.append((label, seed, record))
        self._pending = None
        return record


def _make_backends(
    args: argparse.Namespace,
    bank: _MeterBank,
) -> Callable[[], tuple[Provider, Judge, Embedder | None]]:
    """Factory the ablation calls per run for (provider, judge, embedder).

    `provider` is the model that serves the world model (defaulting to `--opt-model`); the judge runs
    on `--judge-model` so fidelity stays comparable to the rest of the harness. Both are wrapped in
    the shared `RetryingProvider` (llm-waterfall backoff) so a long fallback-less sweep rides through
    transient Bedrock capacity errors. The embedder is offline hashing (default), Azure ada-002
    (`--embedder azure`, semantic), or None (`--no-rag`).
    """
    # Serve rides the config-driven failover chain (.wmh/fallback.toml rungs incl. the
    # Anthropic-direct last resort) inside retry backoff; the JUDGE stays pinned to a single
    # backend — a judge that switches models mid-run scores steps on different scales.
    serve_raw: Provider = RetryingProvider(
        provider_or_chain(
            ProviderConfig(kind=ProviderKind.BEDROCK, model=args.opt_model, region=args.region)
        )
    )
    judge_provider: Provider = RetryingProvider(
        get_provider(
            ProviderConfig(kind=ProviderKind.BEDROCK, model=args.judge_model, region=args.region)
        )
    )
    scorer: Judge = RubricJudge(judge_provider)
    embedder = _build_embedder(args)

    def factory() -> tuple[Provider, Judge, Embedder | None]:
        metered = MeteredProvider(serve_raw, bank.fresh(), base_phase=Phase.SERVE)
        return metered, scorer, embedder

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


def _run(args: argparse.Namespace) -> tuple[AblationReport, _MeterBank]:
    traces, label = _load_corpus(args)
    if args.drop_degenerate:
        traces, dropped = drop_degenerate_traces(traces)
        print(f"corpus hygiene: dropped {dropped} degenerate (all-empty-observation) traces")
    if not traces:
        raise SystemExit("no traces ingested")

    bank = _MeterBank()
    seeds = _parse_ints(args.seeds)
    ablation = TraceScalingAblation(
        traces,
        BASE_ENV_PROMPT,
        make_backends=_make_backends(args, bank),
        counts=_parse_ints(args.counts),
        modes=_parse_strs(args.modes),
        budget=args.budget,
        top_k=args.top_k,
        test_frac=args.test_frac,
        valid_frac=args.valid_frac,
        sample_turns=args.sample_turns,
        test_cap=args.test_cap,
        concurrency=args.concurrency,
        max_retrieved_observation_chars=args.max_retrieved_observation_chars,
        retrieval_key=args.retrieval_key,
        score_dimension=args.score_dimension,
        source_pins=args.source_pins,
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
        f"opt-model={args.opt_model}, judge={args.judge_model}\n"
    )

    def _progress(condition: Condition, seed: int, score: float) -> None:
        record = bank.take(condition.label, seed)
        note = ""
        if record is not None:
            t = record.total
            note = (
                f"  serve: {t.calls} calls, {t.input_tokens}in/{t.output_tokens}out tok, "
                f"${t.cost_usd:.3f}, {record.duration_seconds:.0f}s"
            )
        print(f"  {condition.label:14} seed={seed}  fidelity={score:.3f}{note}", flush=True)

    return run_ablation(ablation, seeds, on_run=_progress), bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", help="Eval-suite name (e.g. tau-bench).")
    parser.add_argument("--file", default=None, help="Raw OTel trace file (instead of a suite).")
    parser.add_argument(
        "--examples",
        default="packages/environment-capture",
        help="Benchmark-suite root (all benchmark suites moved to packages/environment-capture/).",
    )
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
    parser.add_argument("--embed-dim", type=int, default=512, help="phi dim (hashing embedder).")
    parser.add_argument(
        "--embedder", default="hashing", choices=["hashing", "azure", "titan"],
        help="Retrieval phi: hashing (lexical, offline) | azure (semantic ada-002) | "
        "titan (Bedrock semantic; memoized per process).",
    )
    parser.add_argument(
        "--embed-endpoint", default="https://endflow-southcentralus.openai.azure.com",
        help="Azure OpenAI base endpoint (--embedder azure).",
    )
    parser.add_argument(
        "--embed-deployment", default="text-embedding-ada-002",
        help="Azure embedding deployment name (--embedder azure).",
    )
    parser.add_argument(
        "--embed-api-version", default="2024-12-01-preview", help="Azure API version.",
    )
    parser.add_argument(
        "--max-retrieved-observation-chars", type=int, default=None,
        help="Keep only the first N chars of each retrieved observation (bounds prompt growth).",
    )
    parser.add_argument(
        "--retrieval-key", default="state_action", choices=["state_action", "action"],
        help="What phi embeds: state_action (full summary) | action (command-only).",
    )
    parser.add_argument(
        "--score-dimension", default=None,
        help="Report one RubricJudge dimension (e.g. factuality) not the mean-of-5 headline.",
    )
    parser.add_argument("--no-rag", action="store_true", help="Disable retrieval (zero-shot).")
    parser.add_argument(
        "--source-pins", default=None,
        help="instance_id->repo/base_commit JSON for source/workspace modes (e.g. "
        "packages/environment-capture/swe-bench/instance_commits.json).",
    )
    parser.add_argument(
        "--drop-degenerate", action="store_true",
        help="Drop traces whose every observation is empty (failed captures) before splitting.",
    )
    parser.add_argument("--out", default=None, help="Path to write the AblationReport JSON.")
    args = parser.parse_args()

    report, bank = _run(args)

    print(f"\n=== {report.name} (seeds={report.seeds}) ===")
    for cell in report.conditions:
        print(f"  {cell.summary()}")
    if bank.by_run:
        print("\nserve-side usage per run (judge cost excluded):")
        for label, seed, record in bank.by_run:
            t = record.total
            print(
                f"  {label:14} seed={seed}  {t.calls} calls  "
                f"{t.input_tokens}in/{t.output_tokens}out tok  ${t.cost_usd:.3f}  "
                f"{record.duration_seconds:.0f}s"
            )
    if args.out:
        Path(args.out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        usage_out = Path(args.out).with_suffix(".usage.json")
        usage_out.write_text(
            json.dumps(
                [
                    {"label": label, "seed": seed, **record.model_dump(mode="json")}
                    for label, seed, record in bank.by_run
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote report -> {args.out} (+ {usage_out.name})")


if __name__ == "__main__":
    main()

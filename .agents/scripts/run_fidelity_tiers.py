#!/usr/bin/env python
"""Measure the fidelity-tier ladder (low -> medium -> high -> max) end to end on one corpus.

Each tier is realized exactly as `wmh build --fidelity <tier>` would realize it, driven by the
SAME `FIDELITY_TIERS` specs the CLI uses (no parallel tier definitions):

- prompt: base (low) or GEPA-optimized with the tier's budget, selecting on the fixed VALID band;
- retrieval phi: offline hashing (low/medium) or the provider's semantic embeddings (high/max);
- serving config: plain RAG (low/medium) or the auto-config search winner (high/max — the
  search also selects on VALID, pruned by corpus signature for high, full ladder for max).

The reported number is the winner configuration replay-scored on the fixed TEST slice — which no
tier's selection ever touched. Serve-side usage is metered per tier (judge excluded).

    AWS_REGION=us-east-1 uv run python scripts/run_fidelity_tiers.py terminal-tasks \\
        --test-cap 40 --out .agents/docs/research/fidelity_tiers/terminal-tasks.json
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from wmh.config import FIDELITY_TIERS, FidelityTier
from wmh.core.types import Trace
from wmh.engine.autoconfig import (
    CandidateConfig,
    CorpusSignature,
    search_max_fidelity,
    select_candidates,
)
from wmh.engine.eval_suites import resolve_eval_suite
from wmh.engine.grounding import FetchGrounder, Grounder
from wmh.engine.knowledge import seeded_knowledge_text
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.engine.replay import replay
from wmh.ingest import drop_degenerate_traces, get_adapter
from wmh.optimize.judge import RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider, provider_or_chain
from wmh.providers.base import Embedder, Provider
from wmh.research.pipeline import optimize_prompt
from wmh.research.scaling_split import partition_corpus, subsample_train
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder
from wmh.tracking import MeteredProvider, Phase, RunTracker


def _load_corpus(suite_name: str, examples_root: str, *, drop_degenerate: bool) -> list[Trace]:
    suite = resolve_eval_suite(suite_name, examples_root)
    adapter = get_adapter("otel-genai")
    traces = [t for f in suite.resolve_files() for t in adapter.from_file(str(f))]
    if drop_degenerate:
        traces, dropped = drop_degenerate_traces(traces)
        print(f"corpus hygiene: dropped {dropped} degenerate traces")
    return traces


class _CachedEmbedder:
    """Memoize embeddings: the search re-indexes the SAME train steps once per candidate."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._memo: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._memo]
        if missing:
            for text, vector in zip(missing, self._inner.embed(missing), strict=True):
                self._memo[text] = vector
        return [self._memo[t] for t in texts]


def _retried(model: str, region: str, *, embed_model: str | None = None) -> Provider:
    raw = provider_or_chain(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model=model,
            region=region,
            embed_model=embed_model,
            embed_dim=512 if embed_model else None,
        )
    )
    # Failover is config-driven now: `.wmh/fallback.toml` (llm-waterfall) supplies retry
    # rungs and the Anthropic-direct last resort that used to live here in code. Without the
    # file this is a plain single backend.
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", help="Eval-suite name (tau-bench, terminal-tasks, swe-bench).")
    parser.add_argument("--examples", default="examples")
    parser.add_argument("--tiers", default="low,medium,high,max", help="Comma-separated subset.")
    parser.add_argument("--test-cap", type=int, default=40)
    parser.add_argument(
        "--train-cap",
        type=int,
        default=200,
        help="RAG/demo train traces (retrieval saturates fast; caps embedding cost).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--opt-model", default="us.anthropic.claude-opus-4-7")
    # Default judge intentionally matches the serve model here: the table's purpose is the
    # LOW->MAX gradient (self-consistent within one judge), and 4.8 is contended by other
    # workloads on this account. Cross-session comparisons should note the judge difference.
    parser.add_argument("--judge-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument("--embed-model", default="amazon.titan-embed-text-v2:0")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--drop-degenerate", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    traces = _load_corpus(args.suite, args.examples, drop_degenerate=args.drop_degenerate)
    split = partition_corpus(traces, test_frac=0.2, valid_frac=0.15)
    test = subsample_train(split.test, args.test_cap, seed=0)
    train = subsample_train(split.train_pool, args.train_cap, seed=0)
    print(
        f"corpus {args.suite}: {len(traces)} traces -> train {len(train)}, "
        f"valid {len(split.valid)}, scoring {len(test)} test traces"
    )

    judge = RubricJudge(_retried(args.judge_model, args.region))
    results: dict[str, dict[str, object]] = {}
    for tier_name in [t.strip() for t in args.tiers.split(",") if t.strip()]:
        spec = FIDELITY_TIERS[FidelityTier(tier_name)]
        # The tier's serve provider doubles as the semantic embedder (Titan via the same config).
        tracker = RunTracker(run_id=uuid.uuid4().hex, kind="research")
        tracker.start()
        serve = MeteredProvider(
            _retried(args.opt_model, args.region, embed_model=args.embed_model),
            tracker,
            base_phase=Phase.SERVE,
        )
        embedder: Embedder = (
            _CachedEmbedder(serve) if spec.semantic_embeddings else HashingEmbedder(dim=512)
        )
        embed_label = "semantic" if spec.semantic_embeddings else "hashing"

        if spec.gepa_budget > 0:
            print(
                f"[{tier_name}] GEPA (iterations {spec.gepa_budget}, "
                f"val cap {spec.gepa_val_cap} steps) ...",
                flush=True,
            )
            gepa_val: list[Trace] = []
            gepa_steps = 0
            for trace in subsample_train(split.valid, len(split.valid), seed=0):
                if gepa_val and gepa_steps + len(trace.steps) > spec.gepa_val_cap:
                    break
                gepa_val.append(trace)
                gepa_steps += len(trace.steps)
            prompt = optimize_prompt(
                train,
                gepa_val,
                BASE_ENV_PROMPT,
                provider=serve,
                judge=judge,
                embedder=embedder,
                budget=spec.gepa_budget,
                seed=args.seed,
            ).prompt
        else:
            prompt = BASE_ENV_PROMPT

        winner = CandidateConfig(label="base")
        considered: list[str] = ["base"]
        knowledge: str | None = None
        if spec.config_search:
            print(f"[{tier_name}] config search (val_cap {spec.search_budget}) ...", flush=True)
            # Seed the KB ONCE and hand the same text to the search and the final replay: a
            # second extraction is a different nondeterministic KB, so the reported number
            # would not be the configuration the search selected.
            menu = select_candidates(
                CorpusSignature.from_traces(train),
                full_ladder=spec.full_ladder,
                cheap_only=spec.cheap_frontier_only,
            )
            if any(c.knowledge for c in menu):
                knowledge = seeded_knowledge_text(train, serve)
            auto = search_max_fidelity(
                prompt,
                train,
                split.valid,  # select on VALID; the TEST slice stays untouched
                serve,
                judge,
                embedder,
                val_cap=spec.search_budget,
                full_ladder=spec.full_ladder,
                cheap_only=spec.cheap_frontier_only,  # medium's cheap frontier, same as build
                knowledge_text=knowledge,
                concurrency=args.concurrency,
                on_candidate_done=lambda label, score: print(
                    f"    candidate {label}: {score:.3f}", flush=True
                ),
            )
            winner = auto.winner
            considered = auto.considered
        if not winner.knowledge:
            knowledge = None
        grounder: Grounder | None = FetchGrounder() if winner.grounder == "fetch" else None
        report = replay(
            prompt,
            test,
            serve,
            judge,
            retriever=EmbeddingRetriever(embedder),
            train=train,
            top_k=winner.top_k or 5,  # rag-deep winners are scored at their own depth
            sample_turns="sampled",
            seed=args.seed,
            concurrency=args.concurrency,
            knowledge=knowledge,
            reasoning=winner.reasoning,
            verify=winner.verify,
            grounder=grounder,
            max_retrieved_observation_chars=winner.demo_obs_cap,
        )
        usage = tracker.record_summary()
        results[tier_name] = {
            "fidelity": report.mean_score,
            "error_flag_accuracy": report.error_flag_accuracy,
            "n_steps": report.n_steps,
            "winner": winner.label,
            "considered": considered,
            "embeddings": embed_label,
            "gepa_budget": spec.gepa_budget,
            "serve_calls": usage.total.calls,
            "serve_cost_usd": round(usage.total.cost_usd, 3),
        }
        print(
            f"[{tier_name}] fidelity={report.mean_score:.3f} winner={winner.label} "
            f"embed={embed_label} cost=${usage.total.cost_usd:.2f}",
            flush=True,
        )
        if args.out:  # write incrementally: a crash mid-ladder must not lose finished tiers
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\ntier ladder:")
    for tier_name, row in results.items():
        print(
            f"  {tier_name:7} {row['fidelity']:.3f}  ({row['winner']}, {row['embeddings']}, "
            f"gepa={row['gepa_budget']}, ${row['serve_cost_usd']})"
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    # Referenced so a future candidate change keeps this runner honest about what "base" means.
    assert DEFAULT_CANDIDATES[0].label == "base"


if __name__ == "__main__":
    main()

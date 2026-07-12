#!/usr/bin/env python
"""Confidence-gated model escalation cells (WS-A6 item 8).

Drives `wmh.engine.replay` directly with TWO serve providers — a cheap draft model and a strong
escalation model — re-predicting a step on the strong model only when the cheap draft's stated
confidence < tau. Anchors: tau=0 (never escalate = all-cheap) and tau=1.01 (always escalate;
strong-only cost includes the wasted cheap draft — the honest deployment shape). D12 protocol
(fixed splits via partition_corpus, test caps, sampled turns, pinned 4.8 judge).

    AWS_REGION=us-east-1 uv run python .agents/scripts/run_escalation.py tau-bench \
        --n-train 200 --test-cap 40 --taus 0,0.6,1.01 --seeds 0,1 \
        --cheap us.anthropic.claude-haiku-4-5-20251001-v1:0 \
        --strong us.anthropic.claude-opus-4-7 \
        --results-dir <dir> --out <summary.json>
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from statistics import fmean

from wmh.engine.eval_suites import resolve_eval_suite
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.engine.replay import replay
from wmh.ingest import drop_degenerate_traces, get_adapter
from wmh.optimize.judge import RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Provider
from wmh.providers.fallback import FallbackProvider
from wmh.research.scaling_split import partition_corpus, subsample_train
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder
from wmh.tracking import MeteredProvider, Phase, RunTracker


def _retrying(model: str, region: str) -> Provider:
    raw = get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=model, region=region))
    return FallbackProvider([raw] * 4, backoff_seconds=15.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--test-cap", type=int, required=True)
    parser.add_argument("--taus", required=True, help="Comma list; 0 = never, >1 = always.")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--cheap", required=True)
    parser.add_argument("--strong", required=True)
    parser.add_argument("--judge-model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--drop-degenerate", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    suite = resolve_eval_suite(args.suite, "examples")
    adapter = get_adapter("otel-genai")
    traces = [t for f in suite.resolve_files() for t in adapter.from_file(str(f))]
    if args.drop_degenerate:
        traces, dropped = drop_degenerate_traces(traces)
        print(f"dropped {dropped} degenerate traces")
    split = partition_corpus(traces, test_frac=0.2, valid_frac=0.15)
    test = subsample_train(split.test, args.test_cap, seed=0)
    judge = RubricJudge(_retrying(args.judge_model, args.region))
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for seed in [int(s) for s in args.seeds.split(",")]:
        train = subsample_train(split.train_pool, args.n_train, seed=seed)
        for tau in [float(t) for t in args.taus.split(",")]:
            tracker = RunTracker(run_id=uuid.uuid4().hex, kind="research")
            tracker.start()
            cheap = MeteredProvider(
                _retrying(args.cheap, args.region), tracker, base_phase=Phase.SERVE
            )
            strong = MeteredProvider(
                _retrying(args.strong, args.region), tracker, base_phase=Phase.SERVE
            )
            retriever = EmbeddingRetriever(HashingEmbedder(dim=512))
            report = replay(
                BASE_ENV_PROMPT,
                test,
                cheap,
                judge,
                retriever=retriever,
                train=train,
                sample_turns="sampled",
                seed=seed,
                concurrency=args.concurrency,
                confidence=True,
                escalate_provider=strong if tau > 0 else None,
                escalate_below=tau if tau > 0 else None,
            )
            record = tracker.record_summary()
            if not report.results:
                raise SystemExit(f"no test steps scored for {args.suite} (empty split/cap?)")
            esc_rate = fmean(1.0 if r.escalated else 0.0 for r in report.results)
            cell = {
                "suite": args.suite,
                "tau": tau,
                "seed": seed,
                "fidelity": round(report.mean_score, 4),
                "escalation_rate": round(esc_rate, 4),
                "serve_cost_usd": round(record.total.cost_usd or 0.0, 3),
                "calls": record.total.calls,
                "n_steps": report.n_steps,
            }
            summary.append(cell)
            label = f"escalate@{tau}@{args.n_train}_seed{seed}"
            (out_dir / f"{label}.json").write_text(report.model_dump_json(indent=2))
            print(json.dumps(cell), flush=True)

    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

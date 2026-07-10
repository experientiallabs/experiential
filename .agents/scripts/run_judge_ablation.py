#!/usr/bin/env python
"""Judge-sensitivity ablation for the GEPA scaling law: same predictions, four judge models.

The scaling law's headline metric is RubricJudge fidelity scored by Opus 4.8. This ablation asks
whether the conclusions (the b=0 anchor level, and the ~flat GEPA lift) are artifacts of that one
scorer: it holds the *predictions* byte-identical and varies only the judge model.

Per benchmark:
1. Recreate the exact scaling-law setup (fixed split, train@64 seed 0, test-cap 40, sampled turns
   seed 0) and re-run GEPA at budget=8/seed=0 with the STANDARD optimizing judge (Rubric/Opus 4.8 —
   optimization is held fixed; this ablates measurement only). The evolved prompt is saved.
2. Replay both prompts (base b=0, evolved b=8) once with the serve model (Opus 4.7 chain), scoring
   with the Opus 4.8 judge chain — this pass yields the predictions AND the Opus 4.8 column.
3. Re-judge the saved predictions with each other judge (Haiku 4.5, GPT-5.4-mini, GPT-5.5): the
   same RubricJudge instrument, a different scorer model, on identical (predicted, actual, step)
   triples. Differences across judges are pure measurement effects.

Judge chains (per the project failover policy; capacity errors only):
- claude judges: Bedrock endflow account -> Bedrock default account -> Anthropic direct API.
- gpt judges: OpenAI direct, no fallback. OPENAI_API_KEY must be set for those two.

    AWS_PROFILE=default AWS_REGION=us-east-1 OPENAI_API_KEY=$(cat ~/.secrets/openai_api_key) \
        uv run python .agents/scripts/run_judge_ablation.py tau-bench \
        --out .agents/docs/research/gepa_scaling_results/judge_ablation/tau-bench.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wmh.core.types import Observation, Step, Trace
from wmh.engine.eval_suites import resolve_eval_suite
from wmh.engine.prompts import BASE_ENV_PROMPT
# _select_step_indices is private to wmh.engine.replay — imported so the re-judging pass
# reconstructs the EXACT (trace, step) work list replay scored (same seeded turn selection).
# Workspace-only fragility: if replay's selection changes, this script re-syncs or dies loudly
# on the strict zip, never silently misaligns.
from wmh.engine.replay import StepResult, _select_step_indices, replay
from wmh.ingest import get_adapter
from wmh.optimize.judge import Judge, RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Provider
from wmh.providers.waterfall import WaterfallProvider
from wmh.research.gepa_scaling import _cap_by_steps
from wmh.research.pipeline import optimize_prompt
from wmh.research.scaling_split import partition_corpus, subsample_train
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder
from wmh.tracking.metered import MeteredProvider, classify_build_call
from wmh.tracking.tracker import RunTracker

sys.path.insert(0, str(Path(__file__).parent))
from run_gepa_scaling import _chain  # noqa: E402

# Bedrock ids per account + the direct-API id, for the two Claude judge chains.
_HAIKU_BEDROCK = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_HAIKU_DIRECT = "claude-haiku-4-5-20251001"
_OPUS48_BEDROCK = "us.anthropic.claude-opus-4-8"
_OPUS48_DIRECT = "claude-opus-4-8"


def _claude_judge_chain(
    bedrock_model: str, direct_model: str, region: str, *, endflow: bool
) -> Provider:
    """[endflow Bedrock ->] default-account Bedrock -> Anthropic direct (capacity failover only).

    `endflow=False` drops the endflow rung: that account lists Opus 4.8 in its inference profiles
    but InvokeModel returns AccessDenied ("not available for this account" — verified live
    2026-07-02); only 4.6-generation models (incl. Haiku 4.5) are invocable there. AccessDenied is
    a non-capacity error, so a dead rung would crash the chain rather than fail over.
    """
    bedrock = ProviderConfig(kind=ProviderKind.BEDROCK, model=bedrock_model, region=region)
    direct = ProviderConfig(kind=ProviderKind.ANTHROPIC, model=direct_model)
    configs = ([bedrock] if endflow else []) + [bedrock, direct]
    profiles = (["endflow"] if endflow else []) + [None, None]
    return WaterfallProvider(configs, profiles=profiles)


class _MaxTokensFloor:
    """Provider wrapper raising max_tokens to a floor (reasoning models' headroom).

    RubricJudge asks for max_tokens=512 — enough for the rubric JSON, but GPT-5.x reasoning
    models spend hidden reasoning tokens from the same budget and 400 ("output limit reached")
    on hard steps. Judged output stays the same; only the ceiling moves.
    """

    def __init__(self, provider: Provider, floor: int) -> None:
        self._provider = provider
        self._floor = floor
        self.config = provider.config

    def complete(self, system, messages, *, temperature=0.7, max_tokens=512):  # noqa: ANN001, ANN202
        return self._provider.complete(
            system, messages, temperature=temperature, max_tokens=max(max_tokens, self._floor)
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed(texts)

    def verify(self):  # noqa: ANN202
        return self._provider.verify()


def _judges(region: str, tracker: RunTracker) -> dict[str, Judge]:
    """The four ablation judges: same RubricJudge instrument, different scorer models."""
    providers: dict[str, Provider] = {
        "haiku-4.5": _claude_judge_chain(_HAIKU_BEDROCK, _HAIKU_DIRECT, region, endflow=True),
        "opus-4.8": _claude_judge_chain(_OPUS48_BEDROCK, _OPUS48_DIRECT, region, endflow=False),
        "gpt-5.4-mini": _MaxTokensFloor(
            get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.4-mini")), 16384
        ),
        "gpt-5.5": _MaxTokensFloor(
            get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5")), 16384
        ),
    }
    return {
        name: RubricJudge(MeteredProvider(p, tracker, classify=classify_build_call))
        for name, p in providers.items()
    }


def _work_list(test: list[Trace], sample_turns: str, seed: int) -> list[tuple[Trace, int]]:
    """The exact (trace, step_index) list `replay` scores, in order (same seeded selection)."""
    rng = random.Random(seed)
    return [(trace, i) for trace in test for i in _select_step_indices(trace, sample_turns, rng)]


def _rejudge(
    judge: Judge, work: list[tuple[Trace, int]], results: list[StepResult], concurrency: int
) -> dict[str, float]:
    """Score saved predictions with `judge`; returns mean fidelity + mean factuality."""

    def _one(item: tuple[tuple[Trace, int], StepResult]) -> tuple[float, float] | None:
        (trace, idx), saved = item
        step: Step = trace.steps[idx]
        predicted = Observation(content=saved.predicted, is_error=saved.is_error_predicted)
        for attempt in (1, 2):  # one retry, then exclude the step (recorded, not silent)
            try:
                verdict = judge.score(predicted, step.observation, step)
                return verdict.score, verdict.dimensions.get("factuality", float("nan"))
            except Exception as exc:  # noqa: BLE001 - a hard step must not kill the whole column
                if attempt == 2:
                    print(f"    excluded {trace.trace_id}[{idx}]: {str(exc)[:120]}")
        return None

    pairs = list(zip(work, results, strict=True))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        scored = list(pool.map(_one, pairs))
    kept = [s for s in scored if s is not None]
    scores = [s for s, _ in kept]
    facts = [f for _, f in kept if f == f]  # drop NaNs (judge returned no factuality dim)
    return {
        "mean": sum(scores) / len(scores),
        "factuality": sum(facts) / len(facts) if facts else float("nan"),
        "n_steps": len(scores),
        "excluded": len(scored) - len(kept),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", help="Eval-suite name (tau-bench | terminal-tasks | swe-bench).")
    parser.add_argument("--examples", default="examples", help="Examples root for suite lookup.")
    parser.add_argument("--n-train", type=int, default=64, help="Train traces (scaling-law point).")
    parser.add_argument("--budget", type=int, default=8, help="GEPA iterations for the b>0 prompt.")
    parser.add_argument("--seed", type=int, default=0, help="Subsample/GEPA/turn-selection seed.")
    parser.add_argument("--test-cap", type=int, default=40, help="Fixed test subsample size.")
    parser.add_argument("--sample-turns", default="sampled", help="all | sampled (Qwen 5-turn).")
    parser.add_argument("--gepa-val-steps", type=int, default=30, help="GEPA valset step cap.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth.")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel step scoring.")
    parser.add_argument("--region", default="us-east-1", help="AWS region (Bedrock).")
    parser.add_argument("--embed-dim", type=int, default=512, help="phi dim (offline embedder).")
    parser.add_argument("--out", required=True, help="Path for the result JSON.")
    args = parser.parse_args()

    adapter = get_adapter("otel-genai")
    suite = resolve_eval_suite(args.suite, args.examples)
    traces = [t for f in suite.resolve_files() for t in adapter.from_file(str(f))]
    split = partition_corpus(traces)
    train = subsample_train(split.train_pool, args.n_train, seed=args.seed)
    test = subsample_train(split.test, args.test_cap, seed=0)
    gepa_valid = _cap_by_steps(
        subsample_train(split.valid, len(split.valid), seed=0), args.gepa_val_steps
    )
    embedder = HashingEmbedder(dim=args.embed_dim)
    retriever = EmbeddingRetriever(embedder)

    tracker = RunTracker(run_id=f"judge-ablation-{args.suite}", kind="research")
    tracker.start()
    serve = MeteredProvider(
        _chain("us.anthropic.claude-opus-4-7", args.region, ladder=True),
        tracker,
        classify=classify_build_call,
    )
    judges = _judges(args.region, tracker)
    # The optimizing judge stays the standard sweep judge (Opus 4.8 chain): we ablate MEASUREMENT,
    # not optimization, so the evolved prompt matches the scaling law's t{n}_b{budget} condition.
    print(f"{args.suite}: GEPA budget={args.budget} on {len(train)} traces (seed {args.seed})...")
    evolved = optimize_prompt(
        train,
        gepa_valid,
        BASE_ENV_PROMPT,
        provider=serve,
        judge=judges["opus-4.8"],
        embedder=embedder,
        budget=args.budget,
        seed=args.seed,
    ).prompt

    work = _work_list(test, args.sample_turns, args.seed)
    out: dict = {
        "suite": args.suite,
        "n_train": args.n_train,
        "budget": args.budget,
        "seed": args.seed,
        "prompts": {"base": BASE_ENV_PROMPT, "gepa": evolved},
        "judges": {},
    }
    predictions: dict[str, list[StepResult]] = {}
    for label, prompt in [("b0", BASE_ENV_PROMPT), (f"b{args.budget}", evolved)]:
        # One replay per prompt: predictions AND the Opus 4.8 reference scores in a single pass.
        report = replay(
            prompt,
            test,
            serve,
            judges["opus-4.8"],
            retriever=retriever,
            train=train,
            top_k=args.top_k,
            sample_turns=args.sample_turns,
            seed=args.seed,
            concurrency=args.concurrency,
        )
        predictions[label] = report.results
        facts = [r.dimensions.get("factuality") for r in report.results if r.dimensions]
        out["judges"].setdefault("opus-4.8", {})[label] = {
            "mean": report.mean_score,
            "factuality": sum(facts) / len(facts) if facts else float("nan"),
            "n_steps": report.n_steps,
        }
        print(f"  {label} predictions done: opus-4.8 fidelity={report.mean_score:.3f}")
        # Persist predictions + partial results after every stage: a crash in a later judge must
        # not lose the (expensive) GEPA run and replay passes again.
        out["predictions"] = {k: [r.model_dump() for r in v] for k, v in predictions.items()}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    for name in ["haiku-4.5", "gpt-5.4-mini", "gpt-5.5"]:
        for label, results in predictions.items():
            out["judges"].setdefault(name, {})[label] = _rejudge(
                judges[name], work, results, args.concurrency
            )
            print(f"  {name:12} {label}: {out['judges'][name][label]['mean']:.3f}")
            Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    tracker.stop()
    totals = tracker.totals()
    out["usage"] = {
        "calls": totals.calls,
        "tokens": totals.total_tokens,
        "cost_usd": totals.cost_usd,
        "seconds": tracker.duration_seconds(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"wrote {args.out} ({totals.calls} calls, ${totals.cost_usd:.2f}, "
        f"{tracker.duration_seconds():.0f}s)"
    )


if __name__ == "__main__":
    main()

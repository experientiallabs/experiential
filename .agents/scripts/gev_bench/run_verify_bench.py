"""Bench-VERIFY: score bird-sql trajectories with the production outcome judge (GoldJudge).

For every labeled trajectory we run the judge k times at temperature T and take the majority vote
as the verdict; vote agreement (3/3 vs 2/3) is the confidence proxy. The verdict is compared to the
deterministic recorded outcome (bird-sql execution match), never to another LLM.

DEVIATION FROM PRODUCTION: `GoldJudge.score` hardcodes temperature 0.0. To get a vote-agreement
confidence proxy we call the judge's EXACT system prompt (`GOLD_JUDGE_SYSTEM`), prompt builder
(`_build_prompt`), and parser (`_parse`) directly, with the temperature exposed. Everything else is
byte-identical to production. A k=1, temp=0.0 run reproduces `GoldJudge.score` exactly.

Usage:
    uv run python .agents/scripts/gev_bench/run_verify_bench.py --limit 4   # smoke
    uv run python .agents/scripts/gev_bench/run_verify_bench.py             # full 40, k=3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Workspace scripts live outside any package tree by design (gate-exempt, nothing in
# wmo/ may depend on them); this one-line bootstrap is what lets siblings share
# gev_bench.corpus without promoting workspace code into the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gev_bench.corpus import VerifyCase, load_cases, select_balanced  # noqa: E402

# Deliberate private-API reuse: the meta-eval must grade the EXACT production judge
# (same prompt builder, same parser); a public re-implementation would test a copy.
# The only deviation is temperature, which GoldJudge.score hardcodes to 0.0.
from wmo.evals.gold import (  # noqa: E402
    GOLD_JUDGE_SYSTEM,
    _build_prompt,
    _parse,
)
from wmo.providers import ProviderConfig, ProviderKind, get_provider  # noqa: E402
from wmo.providers.base import Message, Provider  # noqa: E402

_LOG = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parents[2] / "docs/research/gev_bench_results/verify"


def _judge_once(provider: Provider, case: VerifyCase, temperature: float) -> dict:
    """One judge pass, reusing GoldJudge's exact prompt + parser with temperature exposed."""
    user = _build_prompt(case.instruction, case.answer, case.transcript, case.gold)
    completion = provider.complete(
        GOLD_JUDGE_SYSTEM,
        [Message(role="user", content=user)],
        temperature=temperature,
        max_tokens=1024,
    )
    verdict = _parse(completion.text, case.gold)
    return {
        "passed": bool(verdict.passed),
        "fraction": verdict.fraction,
        "rationale": verdict.rationale,
    }


def _score_case(provider: Provider, case: VerifyCase, k: int, temperature: float) -> dict:
    votes = [_judge_once(provider, case, temperature) for _ in range(k)]
    n_pass = sum(v["passed"] for v in votes)
    predicted_pass = n_pass * 2 > k  # majority
    agreement = max(n_pass, k - n_pass)  # size of the winning bloc (k => unanimous)
    return {
        "trace_id": case.trace_id,
        "base_task_id": case.base_task_id,
        "task_id": case.task_id,
        "model": case.model,
        "n_steps": case.n_steps,
        "recorded_pass": case.recorded_pass,
        "predicted_pass": predicted_pass,
        "correct": predicted_pass == case.recorded_pass,
        "n_pass_votes": n_pass,
        "k": k,
        "agreement": agreement,
        "unanimous": agreement == k,
        "votes": votes,
        "instruction": case.instruction,
        "answer": case.answer,
        "gold_sql": case.gold_sql,
        "transcript": case.transcript,
    }


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-per-class", type=int, default=20)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="bird-sql corpus dir (default: this repo's committed copy)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None, help="Cap #cases (smoke test).")
    parser.add_argument("--concurrency", type=int, default=8, help="Cases graded in parallel.")
    parser.add_argument("--out", default=str(OUT_DIR / "results.json"))
    args = parser.parse_args()

    cases = select_balanced(
        load_cases(args.corpus_root) if args.corpus_root is not None else load_cases(),
        n_per_class=args.n_per_class,
        seed=args.seed,
    )
    if args.limit is not None:
        # Keep the smoke slice balanced: interleave pass/fail.
        passes = [c for c in cases if c.recorded_pass]
        fails = [c for c in cases if not c.recorded_pass]
        half = args.limit // 2
        cases = passes[:half] + fails[: args.limit - half]

    provider = get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=args.model, region=args.region)
    )

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        results = list(
            pool.map(lambda c: _score_case(provider, c, args.k, args.temperature), cases)
        )
    elapsed = time.time() - started

    n = len(results)
    correct = sum(r["correct"] for r in results)
    _LOG.info(f"graded {n} cases (k={args.k}, T={args.temperature}) in {elapsed:.0f}s")
    _LOG.info(f"accuracy: {correct}/{n} = {correct / n:.3f}")

    payload = {
        "config": {
            "judge_model": args.model,
            "region": args.region,
            "k": args.k,
            "temperature": args.temperature,
            "n_per_class": args.n_per_class,
            "seed": args.seed,
            "limit": args.limit,
            "corpus": "bird-sql",
            "judge": "GoldJudge (exact system prompt + parser; temperature exposed)",
        },
        "elapsed_s": elapsed,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOG.info(f"wrote {out}")


if __name__ == "__main__":
    main()

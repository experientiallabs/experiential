#!/usr/bin/env python
"""Headroom probe: predict, WITHOUT running GEPA, whether GEPA will pay for an executor x corpus.

Mechanism (plan: .agents/docs/proposals/gepa-value-per-dollar.md, Round 2 / E4): GEPA can only
fix failures whose root cause is learnable from the corpus. So the probe scores ~25 VALID-band
steps with base+RAG (retrieval from the train pool, leak-free, never touching test), collects the
failures (step score < 0.8), and asks the judge model to classify each failure's root cause with
the template-v2 taxonomy:

  derivable      - computable from the action + conventions a prompt could teach (GEPA-fixable)
  session        - establishable from earlier steps in the same trace (partially GEPA-fixable)
  unknowable     - fresh external values no prompt can supply (GEPA-immune)

headroom = fixable_failures / probed_steps  (fixable = derivable + session). Validation target:
rank-correlate against the measured GEPA lifts (round 1 T1/T2/T3 + round 2 E1).

    AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/probe_gepa_headroom.py \
        terminal-tasks --opt-model us.anthropic.claude-haiku-4-5-20251001-v1:0 \
        --out probe_haiku_terminal.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from wmh.engine.eval_suites import resolve_eval_suite
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.engine.replay import ReplayReport
from wmh.ingest import get_adapter
from wmh.optimize.judge import RubricJudge
from wmh.research.pipeline import score_prompt
from wmh.research.scaling_split import partition_corpus, subsample_train
from wmh.retrieval import HashingEmbedder
from wmh.tracking.metered import MeteredProvider, classify_build_call
from wmh.tracking.tracker import RunTracker

# Reuse the round-1 runner's provider chain (failover ladder, @profile rungs, openai/ prefix).
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("rgs", Path(__file__).parent / "run_gepa_scaling.py")
_rgs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rgs)

FAIL_BELOW = 0.8
_CLASSIFY_SYSTEM = (
    "You classify why a world-model's prediction of an environment observation missed. "
    "Given the agent's action, the real observation, the predicted observation, and a judge "
    "critique, answer with EXACTLY one word:\n"
    "derivable - the true observation is computable from the action itself plus environment "
    "conventions (output format, error phrasing, deterministic rules) that a better system "
    "prompt could teach.\n"
    "session - the true observation could be established from earlier steps of this session "
    "(values already seen, state already set).\n"
    "unknowable - the true observation contains fresh external values (live data, file "
    "contents never shown, record fields never revealed) no prompt could supply."
)


def _classify(judge_provider, action: str, actual: str, predicted: str, critique: str) -> str:  # noqa: ANN001
    from wmh.providers.base import Message

    user = (
        f"ACTION:\n{action[:2000]}\n\nREAL OBSERVATION:\n{actual[:2000]}\n\n"
        f"PREDICTED:\n{predicted[:2000]}\n\nJUDGE CRITIQUE:\n{critique[:800]}\n\n"
        "One word (derivable/session/unknowable):"
    )
    text = judge_provider.complete(_CLASSIFY_SYSTEM, [Message(role="user", content=user)],
                                   temperature=0.0, max_tokens=16).text.lower()
    match = re.search(r"derivable|session|unknowable", text)
    return match.group(0) if match else "unknowable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("--opt-model", required=True, help="Executor to probe.")
    parser.add_argument("--judge-model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--examples", default="packages/environment-capture")
    parser.add_argument("--steps", type=int, default=25, help="Valid-band steps to probe.")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    adapter = get_adapter("otel-genai")
    suite = resolve_eval_suite(args.suite, args.examples)
    traces = [t for f in suite.resolve_files() for t in adapter.from_file(str(f))]
    split = partition_corpus(traces, test_frac=0.2, valid_frac=0.15)  # same split as the runs

    # A step-capped valid-band slice (seed-0 shuffle, inclusive of long traces).
    probe: list = []
    total = 0
    for t in subsample_train(split.valid, len(split.valid), seed=0):
        probe.append(t)
        total += len(t.steps)
        if total >= args.steps:
            break

    tracker = RunTracker(run_id="gepa-headroom-probe", kind="research")
    serve = MeteredProvider(_rgs._chain(args.opt_model, args.region, True), tracker,
                            classify=classify_build_call)
    judge_provider = MeteredProvider(_rgs._chain(args.judge_model, args.region, True), tracker,
                                     classify=classify_build_call)

    reports: list[ReplayReport] = []
    score_prompt(
        BASE_ENV_PROMPT, probe,
        provider=serve, judge=RubricJudge(judge_provider), embedder=HashingEmbedder(dim=512),
        train=split.train_pool, top_k=args.top_k, sample_turns="all",
        concurrency=8, on_report=reports.append,
    )
    steps = [s for r in reports for s in r.results if s.valid]
    failures = [s for s in steps if s.score < FAIL_BELOW]

    counts = {"derivable": 0, "session": 0, "unknowable": 0}
    for s in failures:
        counts[_classify(judge_provider, s.action, s.actual, s.predicted, s.critique)] += 1

    fixable = counts["derivable"] + counts["session"]
    result = {
        "suite": args.suite,
        "executor": args.opt_model,
        "probed_steps": len(steps),
        "failures": len(failures),
        "failure_classes": counts,
        "headroom": fixable / len(steps) if steps else 0.0,
        "probe_fidelity": sum(s.score for s in steps) / len(steps) if steps else 0.0,
        "cost_usd": tracker.totals().cost_usd,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

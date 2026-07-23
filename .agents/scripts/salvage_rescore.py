#!/usr/bin/env python
"""N1: salvage round-1 winning prompts from run logs and rescore with per-step persistence.

Round 1 persisted only per-seed mean fidelity, so its headline lifts rest on seed-separation.
This script (a) extracts each seed's winning evolved prompt from the gepa library's log lines
(the last "Found a better program" iteration's "Proposed new text" block), (b) rescores that
prompt AND the base prompt on the identical fixed test set with full per-step ReplayReports,
enabling paired per-step bootstrap CIs and where-did-the-lift-come-from diffs.

Self-validation: the guard's revert-to-base is not visible in logs, so the rescored winner mean
is compared against the arm's recorded fidelity; a winner that rescores at base level while the
record shows a lift (or vice versa) is flagged rather than trusted.

    AWS_PROFILE=default AWS_REGION=us-east-1 uv run python .agents/scripts/salvage_rescore.py \
        .agents/docs/research/gepa_vpd_results/t3_terminal_mini_gepa_self.log terminal-tasks \
        --opt-model openai/gpt-5.4-mini --outdir .agents/docs/research/gepa_vpd_results/steps
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from wmh.engine.eval_suites import resolve_eval_suite
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.ingest import get_adapter
from wmh.optimize.judge import RubricJudge
from wmh.research.pipeline import score_prompt
from wmh.research.scaling_split import partition_corpus, subsample_train
from wmh.retrieval import HashingEmbedder
from wmh.tracking.metered import MeteredProvider, classify_build_call
from wmh.tracking.tracker import RunTracker

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("rgs", Path(__file__).parent / "run_gepa_scaling.py")
_rgs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rgs)

_SEED_LINE = re.compile(r"^\s+t\d+_b\d+\s+seed=(\d+)\s+fidelity=([\d.]+)")
_BETTER = re.compile(r"^Iteration (\d+): Found a better program on the valset")
_PROPOSED = "Proposed new text for env_prompt: "


def extract_winners(log_path: Path) -> dict[int, tuple[str | None, float]]:
    """Per seed: (winning prompt text or None for base, recorded arm fidelity)."""
    winners: dict[int, tuple[str | None, float]] = {}
    segment: list[str] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _SEED_LINE.match(line)
        if not m:
            segment.append(line)
            continue
        seed, recorded = int(m.group(1)), float(m.group(2))
        best_iter = None
        for seg_line in segment:
            b = _BETTER.match(seg_line)
            if b:
                best_iter = b.group(1)
        text: str | None = None
        if best_iter is not None:
            marker = f"Iteration {best_iter}: {_PROPOSED}"
            lines_iter = iter(segment)
            for seg_line in lines_iter:
                if seg_line.startswith(marker):
                    parts = [seg_line[len(marker):]]
                    for cont in lines_iter:
                        if re.match(r"^Iteration \d+: ", cont) or _SEED_LINE.match(cont):
                            break
                        parts.append(cont)
                    text = "\n".join(parts).strip()
                    break
        winners[seed] = (text, recorded)
        segment = []
    return winners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("suite")
    parser.add_argument("--opt-model", required=True)
    parser.add_argument("--judge-model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--examples", default="packages/environment-capture")
    parser.add_argument("--test-cap", type=int, default=40)
    parser.add_argument("--drop-degenerate", action="store_true")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    winners = extract_winners(args.log)
    print({s: (len(t) if t else None, rec) for s, (t, rec) in winners.items()})

    adapter = get_adapter("otel-genai")
    suite = resolve_eval_suite(args.suite, args.examples)
    traces = [t for f in suite.resolve_files() for t in adapter.from_file(str(f))]
    if args.drop_degenerate:
        from wmh.ingest import drop_degenerate_traces

        traces, _ = drop_degenerate_traces(traces)
    split = partition_corpus(traces, test_frac=0.2, valid_frac=0.15)
    test = subsample_train(split.test, args.test_cap, seed=0)

    tracker = RunTracker(run_id="salvage-rescore", kind="research")
    serve = MeteredProvider(_rgs._chain(args.opt_model, args.region, True), tracker,
                            classify=classify_build_call)
    judge = RubricJudge(MeteredProvider(_rgs._chain(args.judge_model, args.region, True),
                                        tracker, classify=classify_build_call))
    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = args.log.stem

    for seed, (text, recorded) in sorted(winners.items()):
        train = subsample_train(split.train_pool, len(split.train_pool), seed=seed)
        for name, prompt in (("winner", text), ("base", BASE_ENV_PROMPT)):
            if prompt is None:
                continue
            reports = []
            mean = score_prompt(
                prompt, test, provider=serve, judge=judge,
                embedder=HashingEmbedder(dim=512), train=train, top_k=5,
                sample_turns="sampled", seed=seed, concurrency=8,
                on_report=reports.append,
            )
            stem = args.outdir / f"{tag}_s{seed}_{name}"
            stem.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
            stem.with_suffix(".report.json").write_text(
                reports[-1].model_dump_json(indent=2), encoding="utf-8")
            flag = ""
            if name == "winner" and abs(mean - recorded) > 0.03:
                flag = f"  MISMATCH vs recorded {recorded:.3f} - possible guard revert"
            print(f"seed={seed} {name:6} rescored={mean:.4f}{flag}")

    totals = tracker.totals()
    print(f"usage: {totals.calls} calls ${totals.cost_usd:.2f}")


if __name__ == "__main__":
    main()

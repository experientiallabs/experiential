"""Run the judge-quality meta-eval (wmh.optimize.judge_quality) against a real Bedrock judge.

Usage:
    uv run python .agents/scripts/run_judge_quality.py --out .agents/docs/research/judge-overhaul/raw/jq.json
    uv run python .agents/scripts/run_judge_quality.py --defect outcome-flip -v
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmh.optimize.judge import RubricJudge
from wmh.optimize.judge_quality import JUDGE_QUALITY_CASES, run_judge_quality
from wmh.providers import ProviderConfig, ProviderKind, get_provider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--region", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--defect", default=None, help="Only run cases with this defect tag.")
    parser.add_argument("--case", default=None, help="Only run the case with this id.")
    parser.add_argument("--out", default=None, help="Write the full report JSON here.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print critiques too.")
    args = parser.parse_args()

    cases = list(JUDGE_QUALITY_CASES)
    if args.defect:
        cases = [c for c in cases if c.defect == args.defect]
    if args.case:
        cases = [c for c in cases if c.id == args.case]
    if not cases:
        raise SystemExit("no cases matched the filter")

    provider = get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=args.model, region=args.region)
    )
    judge = RubricJudge(provider)
    report = run_judge_quality(judge, cases, concurrency=args.concurrency)

    for verdict in report.verdicts:
        mark = "PASS" if verdict.passed else "FAIL"
        dims = " ".join(f"{k[:4]}={v:.2f}" for k, v in sorted(verdict.dimensions.items()))
        print(f"{mark}  {verdict.case_id:32} {verdict.defect:18} score={verdict.score:.3f}  {dims}")
        for failure in verdict.failures:
            print(f"      !! {failure}")
        if args.verbose and verdict.critique:
            print(f"      critique: {verdict.critique}")
    print(report.summary(), f"(judge model: {args.model})")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"judge_model": args.model, "report": report.model_dump(mode="json")}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

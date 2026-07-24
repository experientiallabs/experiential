"""Bench-GEN verification leg: back-agreement + solvability, persisted to JSON.

`wmh scenarios verify` prints a table but never writes the VerificationReport, so this runner
replicates the CLI's provider wiring and dumps the full report (per-scenario verdicts included)
for the metrics step. Roles resolve from .wmh/settings.toml (judge = Opus 4.8); the agent and the
world-model simulator fall back to the world model's own serve provider (haiku, cheap).

Usage:
    uv run python .agents/scripts/gev_bench/run_verify.py \
        --scenarios .agents/docs/research/gev_bench_results/gen/tau_scenarios.json \
        --file packages/environment-capture/tau-bench/traces.otel.jsonl \
        --name gev-tau --max-steps 10 \
        --out .agents/docs/research/gev_bench_results/gen/verification_report.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from wmh import providers
from wmh.cli.app import _load_model, _role_provider_config
from wmh.env.llm_agent import LLMAgent
from wmh.ingest import get_adapter
from wmh.scenarios.synthesis import ScenarioSet
from wmh.scenarios.verification import ChecklistJudge
from wmh.scenarios.verification.verify import verify_scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--name", default="gev-tau")
    parser.add_argument("--root", default=".wmh")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scenario_set = ScenarioSet.load(args.scenarios)
    traces = get_adapter("otel-genai").from_file(args.file)
    world_model, resolved_name, llm = _load_model(args.name, args.root)

    worker_config = _role_provider_config("worker", args.region)
    judge_config = _role_provider_config("judge", args.region)
    agent_llm = providers.get_provider(worker_config) if worker_config else llm
    judge_llm = providers.get_provider(judge_config) if judge_config else llm

    print(
        f"verifying {len(scenario_set.scenarios)} scenarios against WM '{resolved_name}' "
        f"(agent={'worker-role' if worker_config else 'wm-serve'}, "
        f"judge={'judge-role' if judge_config else 'wm-serve'}), max_steps={args.max_steps}"
    )
    start = time.time()
    report = verify_scenarios(
        scenario_set,
        traces,
        world_model,
        LLMAgent(agent_llm),
        ChecklistJudge(judge_llm),
        max_steps=args.max_steps,
    )
    elapsed = time.time() - start

    Path(args.out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"back-agreement {report.back_agreement_rate:.0%}, solvable {report.solvable_rate:.0%} "
        f"over {len(report.verdicts)} scenarios in {elapsed:.1f}s -> {args.out}"
    )


if __name__ == "__main__":
    main()

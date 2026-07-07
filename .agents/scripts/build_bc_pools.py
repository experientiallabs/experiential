"""Build the two filtered-BC arms: MINED selection vs RANDOM selection, everything else equal.

Both arms draw from the same train-trace split, use the same synthesizer (Kimi K2.5), and pass
the same inline back-agreement validity gate — the ONLY difference is which source traces become
scenarios: the mining pipeline (facets -> cluster -> hybrid select) vs a uniform random sample.
Both arms are trimmed to the same size (first N valid) so downstream budgets match exactly.

Usage (from the repo root):
    uv run python .agents/scripts/build_bc_pools.py [--target 60]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / ".agents" / "scripts"))

from collect_teacher import foundry, opus_judge  # noqa: E402
from run_scenario_e2e import TRACES, titan_embedder  # noqa: E402

from wmh.engine.build import split_traces  # noqa: E402
from wmh.ingest import get_adapter  # noqa: E402
from wmh.scenarios import ScenarioBuildConfig, ScenarioSet, build_scenario_set  # noqa: E402
from wmh.scenarios.builder import _checklist_agrees  # noqa: E402
from wmh.scenarios.facets import TraceFacet, trace_domain  # noqa: E402
from wmh.scenarios.synthesis import ScenarioSynthesizer  # noqa: E402
from wmh.scenarios.verification import ChecklistJudge  # noqa: E402

DISTILL = REPO / ".agents" / "docs" / "research" / "distill"
SYNTH_MODEL = "gpt-5.4"  # Foundry; judge = Opus 4.8 (AWS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=60, help="valid scenarios per arm")
    parser.add_argument("--arm", choices=["mined", "random", "both"], default="both")
    parser.add_argument("--mined-out", default="bc_pool_mined.json")
    parser.add_argument("--random-out", default="bc_pool_random.json")
    parser.add_argument("--random-seed", type=int, default=0, help="seed for the uniform draw")
    args = parser.parse_args()

    provider = foundry(SYNTH_MODEL)
    traces = get_adapter("otel-genai").from_file(str(TRACES))
    train_traces, _ = split_traces(traces, 0.8)
    facet_data = json.loads((DISTILL / "facets_full.json").read_text())["train"]
    facets = [TraceFacet.model_validate(f) for f in facet_data]
    assert len(facets) == len(train_traces), "cached facets misaligned with train split"
    # Cached facets predate the domain field; backfill it from trace metadata so the enriched
    # embed_text (domain + tool signature) has the domain without re-running facet extraction.
    facets = [
        facet.model_copy(update={"domain": trace_domain(trace)})
        for facet, trace in zip(facets, train_traces, strict=True)
    ]

    judge_llm = opus_judge()
    if args.arm == "random":
        _build_random_arm(args, provider, judge_llm, train_traces, facets)
        return
    # Arm A — MINED: the full pipeline with inline validation, over-budget then trim.
    mined = build_scenario_set(
        train_traces,
        facets,
        provider,
        titan_embedder(),
        ScenarioBuildConfig(budget=args.target + 20, seed=0),
        judge_provider=judge_llm,
    )
    mined.scenarios = mined.scenarios[: args.target]
    mined.save(DISTILL / args.mined_out)
    print(f"mined arm: {len(mined.scenarios)} valid scenarios -> {args.mined_out}", flush=True)
    if args.arm == "mined":
        return
    _build_random_arm(args, provider, judge_llm, train_traces, facets)


def _build_random_arm(args, provider, judge_llm, train_traces, facets) -> None:  # noqa: ANN001
    # Arm B — RANDOM: uniform random source traces, same synthesizer + same validity gate.
    facets_by_id = {f.trace_id: f for f in facets}
    rng = random.Random(args.random_seed)
    order = list(range(len(train_traces)))
    rng.shuffle(order)
    synthesizer = ScenarioSynthesizer(provider)
    judge = ChecklistJudge(judge_llm)
    random_scenarios = []
    attempts = 0
    for index in order:
        if len(random_scenarios) >= args.target:
            break
        trace = train_traces[index]
        attempts += 1
        scenario = synthesizer.synthesize(trace, facets_by_id[trace.trace_id])
        if not _checklist_agrees(judge, scenario, trace):
            scenario = synthesizer.synthesize(trace, facets_by_id[trace.trace_id])
            if not _checklist_agrees(judge, scenario, trace):
                continue
        scenario.cluster_name = "random-arm"
        scenario.weight = 1.0 / args.target
        random_scenarios.append(scenario)
        if len(random_scenarios) % 10 == 0:
            print(f"  random arm: {len(random_scenarios)}/{args.target}", flush=True)
    random_pool = ScenarioSet(scenarios=random_scenarios, corpus_traces=len(train_traces))
    random_pool.save(DISTILL / args.random_out)
    print(f"random arm: {len(random_scenarios)} valid scenarios ({attempts} attempts)", flush=True)


if __name__ == "__main__":
    main()

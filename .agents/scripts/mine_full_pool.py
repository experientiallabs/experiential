"""Mine the FULL tau-bench corpus into disjoint train/eval scenario pools (charter stage 1).

All 1033 traces: facets (Nova Lite, threaded) -> Titan embeddings -> cluster -> synthesize ->
verify (solvability + back-agreement vs the tau world model) -> drop unverified. The corpus is
split train/eval by the stable trace-id hash BEFORE pool construction, so the two pools can never
share a source trace.

Outputs (under .agents/docs/research/distill/):
    facets_full.json, train_pool.json, eval_pool.json, mine_report.json

Usage (from the repo root):
    uv run python .agents/scripts/mine_full_pool.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / ".agents" / "scripts"))

import numpy as np  # noqa: E402

from run_scenario_e2e import NOVA_LITE, TRACES, WM_DIR, bedrock, titan_embedder  # noqa: E402

from wmh.engine.build import split_traces  # noqa: E402
from wmh.engine.world_model import WorldModel  # noqa: E402
from wmh.env.llm_agent import LLMAgent  # noqa: E402
from wmh.ingest import get_adapter  # noqa: E402
from wmh.scenarios import (  # noqa: E402
    ChecklistJudge,
    FacetExtractor,
    ScenarioBuildConfig,
    build_scenario_set,
    verify_scenarios,
)

OUT = REPO / ".agents" / "docs" / "research" / "distill"
WORKERS = 8
TRAIN_BUDGET = 120
EVAL_BUDGET = 40
VERIFY_MAX_STEPS = 6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lite = bedrock(NOVA_LITE)
    embedder = titan_embedder()

    traces = get_adapter("otel-genai").from_file(str(TRACES))
    print(f"corpus: {len(traces)} traces", flush=True)
    train_traces, eval_traces = split_traces(traces, 0.8)
    print(f"split: {len(train_traces)} train / {len(eval_traces)} eval", flush=True)

    extractor = FacetExtractor(lite)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        train_facets = list(pool.map(extractor.extract, train_traces))
        eval_facets = list(pool.map(extractor.extract, eval_traces))
    outcome_counts = Counter(f.outcome.value for f in train_facets + eval_facets)
    print(f"facets done: {dict(outcome_counts)} ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "facets_full.json").write_text(
        json.dumps(
            {
                "train": [f.model_dump() for f in train_facets],
                "eval": [f.model_dump() for f in eval_facets],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    pools = {}
    for name, subset_traces, subset_facets, budget in (
        ("train", train_traces, train_facets, TRAIN_BUDGET),
        ("eval", eval_traces, eval_facets, EVAL_BUDGET),
    ):
        print(f"== building {name} pool (budget {budget}) ==", flush=True)
        scenario_set = build_scenario_set(
            subset_traces,
            subset_facets,
            lite,
            embedder,
            ScenarioBuildConfig(budget=budget, seed=0),
        )
        scenario_set.save(OUT / f"{name}_pool_unverified.json")
        print(
            f"{name}: {len(scenario_set.scenarios)} scenarios, "
            f"coverage {scenario_set.corpus_coverage:.0%} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        pools[name] = (scenario_set, subset_traces)

    # Verification: fresh WM per pool, frozen inside verify_scenarios. Threaded per scenario.
    world_model = WorldModel.load(str(WM_DIR), lite, telemetry_root=str(REPO / ".wmh"))
    report_payload: dict[str, object] = {}
    for name, (scenario_set, subset_traces) in pools.items():
        print(f"== verifying {name} pool ==", flush=True)
        # verify_scenarios is sequential; shard the set across threads for wall-clock.
        shards = [
            scenario_set.model_copy(update={"scenarios": scenario_set.scenarios[i::WORKERS]})
            for i in range(WORKERS)
        ]

        def verify_shard(shard):  # noqa: ANN001, ANN202
            return verify_scenarios(
                shard,
                subset_traces,  # noqa: B023 - bound per loop iteration below
                world_model,
                LLMAgent(lite),
                ChecklistJudge(lite),
                max_steps=VERIFY_MAX_STEPS,
            )

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            shard_reports = list(pool.map(verify_shard, shards))
        verdicts = [v for r in shard_reports for v in r.verdicts]
        ok_ids = {v.scenario_id for v in verdicts if v.ok}
        report_payload[name] = {
            "total": len(verdicts),
            "kept": len(ok_ids),
            "back_agreement_rate": sum(v.back_agreement is True for v in verdicts)
            / max(1, sum(v.back_agreement is not None for v in verdicts)),
            "solvable_rate": sum(v.solvable for v in verdicts) / max(1, len(verdicts)),
            "verdicts": [v.model_dump() for v in verdicts],
        }
        scenario_set.retain(ok_ids)
        scenario_set.save(OUT / f"{name}_pool.json")
        print(
            f"{name}: kept {len(ok_ids)}/{len(verdicts)} verified scenarios "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    report_payload["wall_clock_seconds"] = round(time.time() - t0, 1)
    (OUT / "mine_report.json").write_text(json.dumps(report_payload, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

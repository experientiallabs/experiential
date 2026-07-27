"""Eval-split power check: student-before per-task pass rates on the pinned holdout.

The TB2 lesson this exists to pre-pay: 14 of 17 holdout tasks sat at floor or
ceiling, leaving 3 informative tasks and an uninformative CI, discovered only
after the training spend. So BEFORE cycle 1 this measures the Qwen3.5-9B BASE
student on every pinned eval task (k attempts each, real tau2, pinned user sim)
and reports per-task pass rates. A task is INFORMATIVE if its pass rate is
strictly between 0 and 1 at this k, or sits at a boundary a warmup could
plausibly move it off (reported separately: floor tasks can only improve, so
they carry one-sided information; ceiling tasks are dead weight for a lift
measurement).

Usage (repo root, TINKER_API_KEY + AZURE_* set):

    uv run python .agents/distill/tau2_power_check.py \
        --tau2-root packages/environment-capture/tau-bench --k 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from wmo.distill.config import DistillConfig
from wmo.distill.tau2 import collect_tau2_rollouts
from wmo.harness.doc import HarnessDoc
from wmo.providers.base import ProviderConfig, ProviderKind

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("tau2_power_check")

_HOLDOUT = Path(__file__).resolve().parent / "tau2-holdout-task-ids.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--run-dir", default=".wmo/distill-runs/tau2-power-check")
    parser.add_argument("--out", default=".wmo/distill-runs/tau2-power-check/power.json")
    args = parser.parse_args()

    task_ids = json.loads(_HOLDOUT.read_text(encoding="utf-8"))
    tau2_root = Path(args.tau2_root).expanduser()
    cfg = DistillConfig.model_validate(
        {
            "student": {"base_model": args.model},
            "teacher": {"model": "Qwen/Qwen3.6-27B"},
            "tau2": {
                "tau2_bin": str(tau2_root / ".venv" / "bin" / "tau2"),
                "data_dir": str(tau2_root / "tau2-bench" / "data"),
            },
            "rollout": {"max_turns": 40, "episode_timeout_s": 900.0},
            # 8192: at 4096 the loop smoke truncated a turn in 1 of 4 episodes,
            # and a truncated turn reads as a broken action to the environment.
            "sampling": {"temperature": 1.0, "max_tokens": 8192},
            "train": {"group_size": args.k, "trial_concurrency": args.concurrency},
        }
    )
    provider = ProviderConfig(kind=ProviderKind.TINKER, model=args.model, model_type=args.model)

    records, stats = collect_tau2_rollouts(
        0, task_ids, cfg, HarnessDoc.baseline(), provider, Path(args.run_dir)
    )

    by_task: dict[str, list[bool]] = defaultdict(list)
    infra: dict[str, int] = defaultdict(int)
    for record in records:
        if record.infra_failed:
            infra[record.task_id] += 1
        else:
            by_task[record.task_id].append(record.passed)

    rows = []
    for task_id in task_ids:
        outcomes = by_task.get(task_id, [])
        rate = sum(outcomes) / len(outcomes) if outcomes else None
        rows.append(
            {
                "task_id": task_id,
                "attempts": len(outcomes),
                "infra_failed": infra.get(task_id, 0),
                "pass_rate": rate,
            }
        )
    measured = [row for row in rows if row["pass_rate"] is not None]
    interior = [row for row in measured if 0 < row["pass_rate"] < 1]
    floor = [row for row in measured if row["pass_rate"] == 0]
    ceiling = [row for row in measured if row["pass_rate"] == 1]

    summary = {
        "model": args.model,
        "k": args.k,
        "tasks": len(task_ids),
        "measured": len(measured),
        "interior": len(interior),
        "floor": len(floor),
        "ceiling": len(ceiling),
        "batch_stats": stats.model_dump(mode="json"),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    logger.info(
        "power: %d/%d tasks measured; %d interior, %d floor (one-sided headroom), %d ceiling "
        "(dead weight); solve rate %.3f over %d executed episodes",
        len(measured),
        len(task_ids),
        len(interior),
        len(floor),
        len(ceiling),
        stats.solve_rate,
        stats.executed_trials,
    )
    for row in rows:
        logger.info(
            "  %-14s attempts=%d infra=%d pass_rate=%s",
            row["task_id"],
            row["attempts"],
            row["infra_failed"],
            "n/a" if row["pass_rate"] is None else f"{row['pass_rate']:.2f}",
        )
    # The pre-registered bar from the transfer prompt: fewer than ~8 informative
    # tasks means the pin cannot resolve a warmup-sized effect and an expanded
    # eval pin should be proposed to the master BEFORE cycle 1. Floor tasks are
    # counted as informative for a LIFT measurement (they can only move up).
    informative = len(interior) + len(floor)
    logger.info(
        "VERDICT: %d informative task(s) (interior + floor) vs the ~8 bar -> %s",
        informative,
        "OK to proceed" if informative >= 8 else "PROPOSE EXPANDED EVAL PIN",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

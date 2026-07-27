"""Measure a hosted model's own tau2 solve rate on TRAIN tasks, under the canonical pins.

The question cycle 1 left: is there a teacher with real headroom over the
Qwen3.5-9B student (71.7% on the holdout)? Cycle 1's teacher was a peer (73.3%),
which is why warmup-only distillation had nothing to teach. This measures a
candidate teacher's capability cheaply, before anyone pays for a training leg.

TRAIN TASKS ONLY, deliberately. Measuring a candidate on the holdout and then
deciding whether to train would select on the same data the gate reads, which
breaks the one-gate-read-per-cycle pre-registration. The train split answers
the capability question just as well.

No span recording: capability is a reward question, so the model is reached
through tau2's own litellm route rather than the span-recording proxy. That
also means this probe works for ANY hosted model, including ones with no
logprob surface (which is the whole point of the text-bridge teacher path).

    uv run python .agents/distill/k3_headroom_probe.py \
        --model fireworks_ai/accounts/fireworks/models/kimi-k3 \
        --tasks /tmp/k3-probe-tasks.json --label kimi-k3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("k3_headroom_probe")

_TAU_ROOT = Path(__file__).resolve().parents[2] / "packages/environment-capture/tau-bench"
_TAU2_BIN = _TAU_ROOT / ".venv/bin/tau2"
_DATA_DIR = _TAU_ROOT / "tau2-bench/data"

# The canonical real-tau2 protocol (adopted 2026-07-27). Changing any of these
# makes a new capture cohort, so they are constants here, not flags.
MAX_STEPS = 100
EPISODE_TIMEOUT_S = 1800
MAX_TOKENS = 8192
TEMPERATURE = 1.0
USER_SIM = "azure/gpt-5.4-mini"
TASK_SPLIT_OVERRIDES = {"telecom": "full"}


def run_episode(model: str, task_id: str, label: str, out_root: Path) -> dict[str, object]:
    """Run one tau2 episode against a hosted model; return its graded row."""
    domain, _, tau2_task = task_id.partition("/")
    save_name = f"probe-{label}-{domain}-{tau2_task}".replace("/", "-")
    episode_dir = out_root / save_name
    episode_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(_TAU2_BIN),
        "run",
        "--domain",
        domain,
        *(
            ["--task-split-name", TASK_SPLIT_OVERRIDES[domain]]
            if domain in TASK_SPLIT_OVERRIDES
            else []
        ),
        "--task-ids",
        tau2_task,
        "--num-trials",
        "1",
        "--max-steps",
        str(MAX_STEPS),
        "--timeout",
        str(EPISODE_TIMEOUT_S),
        "--max-retries",
        "0",
        "--agent-llm",
        model,
        "--agent-llm-args",
        json.dumps(
            {
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                # Retries off for the same reason the collector pins them off:
                # a retried episode is a different episode, not a repair.
                "num_retries": 0,
                "timeout": EPISODE_TIMEOUT_S,
            }
        ),
        "--user-llm",
        USER_SIM,
        "--user-llm-args",
        "{}",
        "--save-to",
        save_name,
        "--auto-resume",
    ]
    env = os.environ | {"TAU2_DATA_DIR": str(_DATA_DIR)}
    log_path = episode_dir / "tau2.log"
    with log_path.open("wb") as log_file:
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, env=env, check=False)
    results = _DATA_DIR / "simulations" / save_name / "results.json"
    row: dict[str, object] = {"task_id": task_id, "reward": None, "passed": False, "note": ""}
    if not results.exists():
        row["note"] = f"no results.json (see {log_path})"
        return row
    payload = json.loads(results.read_text(encoding="utf-8"))
    simulations = payload.get("simulations") or []
    if not simulations:
        row["note"] = "no simulation recorded"
        return row
    simulation = simulations[0]
    reward = (simulation.get("reward_info") or {}).get("reward")
    row["termination_reason"] = simulation.get("termination_reason")
    row["messages"] = len(simulation.get("messages") or [])
    row["duration_s"] = simulation.get("duration")
    if isinstance(reward, int | float):
        row["reward"] = float(reward)
        row["passed"] = float(reward) >= 1.0 - 1e-9
    else:
        row["note"] = f"ungraded ({row['termination_reason']})"
    (episode_dir / "results.json").write_bytes(results.read_bytes())
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="litellm model spec for the candidate")
    parser.add_argument("--tasks", required=True, help="JSON array of composite train task ids")
    parser.add_argument("--label", required=True, help="short name for artifacts, e.g. kimi-k3")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--out-root", default=".wmo/probes")
    args = parser.parse_args()

    task_ids = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    out_root = Path(args.out_root) / args.label
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "probing %s on %d TRAIN task(s) (holdout untouched), canonical pins, concurrency %d",
        args.model,
        len(task_ids),
        args.concurrency,
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(
            pool.map(lambda t: run_episode(args.model, t, args.label, out_root), task_ids)
        )

    graded = [r for r in rows if r["reward"] is not None]
    by_domain: dict[str, list[bool]] = defaultdict(list)
    for row in graded:
        by_domain[str(row["task_id"]).split("/")[0]].append(bool(row["passed"]))
    solve = sum(bool(r["passed"]) for r in graded) / len(graded) if graded else 0.0
    summary = {
        "model": args.model,
        "label": args.label,
        "tasks": len(task_ids),
        "graded": len(graded),
        "solve_rate": solve,
        "by_domain": {d: sum(v) / len(v) for d, v in sorted(by_domain.items())},
        "rows": rows,
    }
    (out_root / "probe.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for row in rows:
        logger.info(
            "  %-14s reward=%s stop=%s msgs=%s %s",
            row["task_id"],
            row["reward"],
            row.get("termination_reason"),
            row.get("messages"),
            row["note"],
        )
    logger.info(
        "PROBE %s: solve %.3f over %d graded of %d task(s); by domain %s -> %s",
        args.label,
        solve,
        len(graded),
        len(task_ids),
        summary["by_domain"],
        out_root / "probe.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

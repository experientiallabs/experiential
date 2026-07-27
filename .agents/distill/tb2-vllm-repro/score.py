"""Score a two-arm TerminalBench-2 job pair, paired over tasks.

Reports binary solve rate, graded ``ctrf`` (fraction of individual tests passed), and the
paired per-task delta with a bootstrap CI resampled over *tasks* rather than trials — the
attempts within a task share a task, so treating them as independent understates the interval.

Also reports executed-vs-graded counts per arm. A trial that vanishes from the denominator
inflates the rate, and the trials that vanish are the long, hard ones, so a rate quoted without
its denominator can rise as the harness breaks more often.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Trial:
    task: str
    reward: float | None
    graded: float | None
    excepted: bool
    stop_reasons: list[str] = field(default_factory=list)


def _ctrf_score(trial_dir: Path) -> float | None:
    for path in sorted(trial_dir.glob("verifier/*.json")):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        summary = doc.get("results", {}).get("summary")
        if isinstance(summary, dict) and summary.get("tests"):
            return summary.get("passed", 0) / summary["tests"]
    return None


def load_arm(jobs_dir: Path) -> list[Trial]:
    trials: list[Trial] = []
    for result in sorted(jobs_dir.rglob("result.json")):
        try:
            doc = json.loads(result.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "task_name" not in doc:
            continue
        rewards = (doc.get("verifier_result") or {}).get("rewards") or {}
        details = (doc.get("agent_result") or {}).get("rollout_details") or [{}]
        raw_stops = (details[0].get("extra") or {}).get("stop_reason") or []
        trials.append(
            Trial(
                task=doc["task_name"].split("/")[-1],
                reward=rewards.get("reward"),
                graded=_ctrf_score(result.parent),
                excepted=doc.get("exception_info") is not None,
                stop_reasons=[s for s in raw_stops if s],
            )
        )
    return trials


def per_task(trials: list[Trial], key) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for t in trials:
        v = key(t)
        if v is not None:
            buckets[t.task].append(v)
    return {task: sum(vs) / len(vs) for task, vs in buckets.items()}


def bootstrap_ci(deltas: list[float], n: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def _rate(trials: list[Trial], key) -> tuple[float, int, int]:
    vals = [v for v in (key(t) for t in trials) if v is not None]
    return (sum(vals) / len(vals) if vals else 0.0), len(vals), len(trials)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-jobs", type=Path, required=True)
    ap.add_argument("--distill-jobs", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arms = {"base": load_arm(args.base_jobs), "distill": load_arm(args.distill_jobs)}

    binary = lambda t: None if t.reward is None else float(t.reward == 1.0)
    graded = lambda t: t.graded

    print(f"{'arm':10s} {'binary':>18s} {'graded ctrf':>13s} {'graded/exec':>13s} {'excepted':>9s}")
    print("-" * 70)
    for name, trials in arms.items():
        b, nb, total = _rate(trials, binary)
        g, _, _ = _rate(trials, graded)
        exc = sum(t.excepted for t in trials)
        print(f"{name:10s} {b:>10.1%} ({nb:3d}) {g:>12.1%} {nb:>6d}/{total:<6d} {exc:>9d}")

    tb, td = per_task(arms["base"], binary), per_task(arms["distill"], binary)
    shared = sorted(set(tb) & set(td))
    if not shared:
        print("\nno shared tasks scored yet")
        return

    deltas = [td[t] - tb[t] for t in shared]
    mean = sum(deltas) / len(deltas)
    lo, hi = bootstrap_ci(deltas, args.bootstrap, args.seed)

    print(f"\npaired per-task delta over {len(shared)} tasks: {mean:+.4f}")
    print(
        f"95% bootstrap CI (resampled over tasks): [{lo:+.4f}, {hi:+.4f}]"
        f"{'  <- includes zero' if lo <= 0 <= hi else '  <- excludes zero'}"
    )

    moved = [(t, tb[t], td[t]) for t in shared if tb[t] != td[t]]
    pinned = sum(1 for t in shared if tb[t] == td[t] and tb[t] in (0.0, 1.0))
    print(
        f"\ntasks that moved: {len(moved)} of {len(shared)}   (pinned at floor/ceiling: {pinned})"
    )
    for task, before, after in sorted(moved, key=lambda r: r[2] - r[1], reverse=True):
        print(f"  {task:36s} {before:.2f} -> {after:.2f}   ({after - before:+.2f})")


if __name__ == "__main__":
    main()

"""Export per-task, per-arm episode rows from a tau2 distill run dir.

The joint-tau master's ack condition (b): the ladder computes its own paired
stats, so aggregates are not enough. This walks a run dir's eval rollout roots
(the three gate arms) and emits one JSONL row per EPISODE: arm, task_id,
attempt, reward, passed, termination/stop reason, infra flag, reward_basis
(so the 7/20 NL-assertion holdout tasks are identifiable per row), duration,
and message count, read straight from each episode's copied tau2 results.json.

Usage (repo root):

    uv run python .agents/distill/export_tau2_rows.py \
        --run-dir .wmo/distill-runs/tau2-cycle1 \
        --out .wmo/distill-runs/tau2-cycle1/episode-rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARM_DIRS = {
    "teacher": "eval-rollouts/baseline-teacher",
    "student-before": "eval-rollouts/baseline-student-before",
    "student-after": "eval-rollouts/student-after",
}


def _episode_rows(arm: str, arm_dir: Path) -> list[dict[str, object]]:
    """One row per episode dir under `<arm_dir>/tau2/step-*/`."""
    rows: list[dict[str, object]] = []
    for step_dir in sorted(arm_dir.glob("tau2/step-*")):
        for episode_dir in sorted(step_dir.iterdir()):
            if episode_dir.name == "spans" or not episode_dir.is_dir():
                continue
            name = episode_dir.name
            task_id, _, attempt = name.rpartition("-a")
            results_path = episode_dir / "results.json"
            row: dict[str, object] = {
                "arm": arm,
                "episode": name,
                "task_id": task_id.replace("-", "/", 1),
                "attempt": int(attempt) if attempt.isdigit() else None,
                "infra_failed": True,
                "reward": None,
                "passed": None,
                "termination_reason": None,
                "reward_basis": None,
                "duration_s": None,
                "messages": None,
            }
            if results_path.exists():
                payload = json.loads(results_path.read_text(encoding="utf-8"))
                simulations = payload.get("simulations") or []
                if simulations:
                    sim = simulations[0]
                    reward_info = sim.get("reward_info") or {}
                    reward = reward_info.get("reward")
                    row.update(
                        {
                            "infra_failed": (
                                sim.get("termination_reason") == "infrastructure_error"
                                or not isinstance(reward, int | float)
                            ),
                            "reward": reward,
                            "passed": (
                                isinstance(reward, int | float) and reward >= 1.0 - 1e-9
                            ),
                            "termination_reason": sim.get("termination_reason"),
                            "reward_basis": reward_info.get("reward_basis"),
                            "duration_s": sim.get("duration"),
                            "messages": len(sim.get("messages") or []),
                        }
                    )
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows: list[dict[str, object]] = []
    for arm, rel in ARM_DIRS.items():
        arm_dir = run_dir / rel
        if not arm_dir.exists():
            print(f"note: no {arm} rollouts at {arm_dir}", file=sys.stderr)  # noqa: T201
            continue
        rows.extend(_episode_rows(arm, arm_dir))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    by_arm: dict[str, int] = {}
    for r in rows:
        by_arm[str(r["arm"])] = by_arm.get(str(r["arm"]), 0) + 1
    print(f"wrote {len(rows)} episode rows {by_arm} -> {out}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

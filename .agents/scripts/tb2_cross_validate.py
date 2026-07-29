"""Cross-validate the bench-defaults TB2 cohort against the tb2-cost corner's real TB2 leg.

The two runs share 17 task ids and up to 13 candidate models but nothing else: different
harness invocation, different scaffold pins (this cohort caps `max_turns` at 100 where that leg
left terminus-2's effectively unbounded default), different day, different sandboxes. Agreement
on the overlap is therefore evidence about the HARNESS rather than about either model, which is
the only reason to compute it.

What it reports, and deliberately what it does not:

* Per-cell agreement on the shared (task, model) pairs, as a paired sign count. This is the
  primary statistic per the corners charter, which bans Spearman headlines at these sample
  sizes.
* Per-task and per-model breakdowns of where the two runs disagree, so a disagreement can be
  read as "this task is stochastic" rather than "the harness is broken".
* NOTHING is pooled across the two runs. Their rows are never averaged together, never
  averaged into one rate, and no delta crosses them: both are `real_episode` but they are
  different cohorts, and the corners rule that no computed statistic may cross provenance
  applies with at least as much force to two cohorts of the same provenance.

Each side's own multiple episodes are collapsed to "solved at least once" before comparing,
because the prior leg ran 1 attempt per task and this cohort runs 2. Comparing a 2-episode
rate against a 1-episode outcome would report the episode count as a harness difference.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from wmo.optimize.outcomes import OutcomeMatrix

logger = logging.getLogger(__name__)

# The prior leg's per-model job roots; a `-repair`/`-r2` suffix is a later re-run of the same
# model, and a scored re-run supersedes an earlier infrastructure hole.
PRIOR_MODELS = (
    "fable-5",
    "gpt-5.5",
    "opus-4-8",
    "sonnet-5",
    "opus-5",
    "kimi-k3",
    "kimi-k2.6",
    "qwen3.6-27b",
    "deepseek-v4-pro",
    "glm-5.2",
    "haiku-4-5",
    "gpt-5.4-mini",
    "qwen3.5-9b",
)


def _prior_solved(jobs_root: Path) -> dict[tuple[str, str], bool]:
    """(task, model) -> solved, read from the prior leg's harbor trial results."""
    solved: dict[tuple[str, str], bool] = {}
    for result_path in jobs_root.rglob("result.json"):
        top = result_path.relative_to(jobs_root).parts[0]
        model = next(
            (
                m
                for m in sorted(PRIOR_MODELS, key=len, reverse=True)
                if top == m or top.startswith(f"{m}-")
            ),
            None,
        )
        if model is None:
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        task = payload.get("task_name")
        verifier = payload.get("verifier_result")
        reward = (verifier or {}).get("rewards", {}).get("reward") if verifier else None
        if not task or reward is None:
            continue
        key = (task, model)
        solved[key] = solved.get(key, False) or float(reward) >= 1.0
    return solved


def _cohort_solved(matrix_path: Path) -> dict[tuple[str, str], bool]:
    """(task, model) -> solved at least once, from this cohort's matrix. Unscored rows skipped."""
    matrix = OutcomeMatrix.model_validate_json(matrix_path.read_text(encoding="utf-8"))
    solved: dict[tuple[str, str], bool] = {}
    for row in matrix.outcomes:
        if row.reward is None:
            continue
        key = (row.scenario_id, row.model)
        solved[key] = solved.get(key, False) or row.success
    return solved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="this cohort's matrix.json")
    parser.add_argument(
        "--prior-jobs",
        type=Path,
        default=Path.home() / "Desktop/Projects/world-model-harness/.wmo/tb2-real/jobs",
        help="the tb2-cost real TB2 leg's harbor jobs root",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mine = _cohort_solved(args.matrix)
    theirs = _prior_solved(args.prior_jobs)
    shared = sorted(set(mine) & set(theirs))
    if not shared:
        raise SystemExit("no shared (task, model) cells: check the paths")

    agree = [key for key in shared if mine[key] == theirs[key]]
    mine_only = [key for key in shared if mine[key] and not theirs[key]]
    theirs_only = [key for key in shared if theirs[key] and not mine[key]]

    by_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_model: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for key in shared:
        hit = 1 if mine[key] == theirs[key] else 0
        by_task[key[0]][0] += hit
        by_task[key[0]][1] += 1
        by_model[key[1]][0] += hit
        by_model[key[1]][1] += 1

    summary = {
        "shared_cells": len(shared),
        "agree": len(agree),
        "agreement_rate": round(len(agree) / len(shared), 4),
        "solved_here_not_there": [f"{t}/{m}" for t, m in mine_only],
        "solved_there_not_here": [f"{t}/{m}" for t, m in theirs_only],
        "by_task": {t: f"{a}/{n}" for t, (a, n) in sorted(by_task.items())},
        "by_model": {m: f"{a}/{n}" for m, (a, n) in sorted(by_model.items())},
        "caveats": [
            "Episodes are collapsed to solved-at-least-once on BOTH sides: this cohort runs 2 "
            "episodes and the prior leg ran 1, and comparing a 2-episode rate against a "
            "1-episode outcome would report the episode count as a harness difference.",
            "Rows from the two cohorts are never pooled, averaged, or differenced; only "
            "per-cell agreement is computed.",
            "One deliberate protocol difference remains: this cohort caps max_turns at 100 "
            "where the prior leg left terminus-2's effectively unbounded default, so a "
            "disagreement on a long task may be that cap rather than sampling variance.",
        ],
    }
    logger.info(
        "shared %d cells across %d tasks x %d models: %d agree (%.1f%%)",
        len(shared),
        len(by_task),
        len(by_model),
        len(agree),
        100 * len(agree) / len(shared),
    )
    logger.info("solved here not there: %s", summary["solved_here_not_there"] or "none")
    logger.info("solved there not here: %s", summary["solved_there_not_here"] or "none")
    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()

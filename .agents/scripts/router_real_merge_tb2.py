"""Merge isolated Terminal-Bench 2 model shards into one dense matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmo.core.files import write_text_atomic
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.pool import load_pool


def _rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _selected(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        scenario_id = row.get("scenario_id")
        model = row.get("model")
        if isinstance(scenario_id, str) and isinstance(model, str):
            grouped.setdefault((scenario_id, model), []).append(row)
    selected: list[dict[str, object]] = []
    for attempts in grouped.values():
        ordered = sorted(attempts, key=lambda row: int(row.get("attempt_number", 0)))
        chosen = next(
            (
                row
                for row in ordered
                if isinstance(row.get("reward"), (int, float))
            ),
            ordered[-1],
        )
        # The frozen task and split manifests use Harbor's canonical bare task id. The
        # per-shard runner prefixes scenario_id for human readability; remove that prefix at
        # the one merge boundary so the measured matrix keys exactly match the preregistration.
        task_id = chosen.get("task_id")
        selected.append({**chosen, "scenario_id": task_id} if isinstance(task_id, str) else chosen)
    return sorted(
        selected,
        key=lambda row: (str(row.get("model")), str(row.get("scenario_id"))),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--central", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=89)
    args = parser.parse_args()

    sources = [args.central / "rows.jsonl", *sorted(args.shards.glob("*/rows.jsonl"))]
    all_rows = [row for path in sources for row in _rows(path)]
    chosen = _selected(all_rows)
    pool = load_pool(args.pool)
    outcomes = [ScenarioOutcome.model_validate(row) for row in chosen]
    matrix = OutcomeMatrix(pool=pool.models, outcomes=outcomes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        args.out_dir / "rows.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in chosen),
    )
    matrix.save(args.out_dir / "matrix.json")
    expected = args.expected_tasks * len(pool.models)
    summary = {
        "benchmark": "terminal-bench-2",
        "tasks": args.expected_tasks,
        "models": len(pool.models),
        "cells_expected": expected,
        "cells_present": len(outcomes),
        "gradeable": sum(row.reward is not None for row in outcomes),
        "missing": expected - len(outcomes),
        "model_cost_usd": sum(row.cost_usd for row in outcomes),
        "environment_cost_usd": None,
        "environment_cost_note": "E2B invoice rate is not exposed in Harbor artifacts",
        "sources": [str(path) for path in sources if path.is_file()],
    }
    write_text_atomic(
        args.out_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, sort_keys=True))  # noqa: T201
    return 0 if len(outcomes) == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())

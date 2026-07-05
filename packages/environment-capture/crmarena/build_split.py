"""Build the committed CRMArena train/test split from the upstream task file.

Reads ``crmarena_tasks.json`` (fetched by ``fetch_data.py --all``: the 1170 upstream CRMArena tasks,
each a query + gold answer + per-task domain instructions) and writes a small, seeded, task-type
stratified split: ``data/{train,test}.jsonl`` (the agent-visible prompt + task metadata) and
``gold/<task_id>.json`` (the gold answer + reward metric, never staged into the agent workspace).

The prompt the agent sees is the upstream ``query`` plus its ``metadata.required`` (task-specific
policy/instructions) and ``metadata.optional`` (shared domain definitions — quarters, seasons, time
periods) folded in, exactly the context the official harness injects. The split is stratified so
every one of the nine CRMArena task types appears in both train and test, and disjoint by
construction (each upstream task lands in exactly one split). Deterministic given ``--seed``.

Usage (from the repo root, after fetch_data.py --all):
    uv run python packages/environment-capture/crmarena/build_split.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from environment_capture.trajectory import JsonValue

_HERE = Path(__file__).parent


def _build_prompt(task: dict[str, JsonValue]) -> str:
    """Fold the upstream query together with its required + optional domain instructions."""
    query = str(task["query"]).strip()
    metadata = task.get("metadata", {})
    assert isinstance(metadata, dict)
    required = str(metadata.get("required", "")).strip()
    optional = str(metadata.get("optional", "")).strip()
    parts = [query]
    if required:
        parts.append(f"# Task Instructions\n{required}")
    if optional:
        parts.append(optional)
    parts.append(
        "Query the CRM database to determine the answer, then submit EXACTLY what the question "
        "asks for and nothing else (an Id, a value, or 'None' when no record applies)."
    )
    return "\n\n".join(parts)


def _write_split(
    tasks: list[dict[str, JsonValue]], split: str, data_dir: Path, gold_dir: Path
) -> int:
    lines: list[str] = []
    for index, task in enumerate(tasks):
        task_id = f"crm-{split}-{index}"
        task_type = str(task["task"])
        reward_metric = str(task["reward_metric"])
        record = {
            "task_id": task_id,
            "prompt": _build_prompt(task),
            "data": {"task_type": task_type, "reward_metric": reward_metric},
        }
        lines.append(json.dumps(record, ensure_ascii=False))
        answer = task["answer"]
        gold = {
            "answer": "None" if answer is None else str(answer),
            "reward_metric": reward_metric,
            "task_type": task_type,
        }
        (gold_dir / f"{task_id}.json").write_text(
            json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (data_dir / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-per-type", type=int, default=5, help="Train tasks per task type")
    parser.add_argument("--test-per-type", type=int, default=2, help="Test tasks per task type")
    parser.add_argument("--tasks-file", default=str(_HERE / "crmarena_tasks.json"))
    args = parser.parse_args()

    tasks_path = Path(args.tasks_file)
    if not tasks_path.exists():
        raise SystemExit(
            f"{tasks_path} not found; run "
            "`uv run python packages/environment-capture/crmarena/fetch_data.py --all` first"
        )

    raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    by_type: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
    for task in raw:
        by_type[str(task["task"])].append(task)

    rng = random.Random(args.seed)
    train: list[dict[str, JsonValue]] = []
    test: list[dict[str, JsonValue]] = []
    for task_type in sorted(by_type):
        pool = by_type[task_type]
        rng.shuffle(pool)
        needed = args.train_per_type + args.test_per_type
        if len(pool) < needed:
            raise SystemExit(f"task type {task_type} has only {len(pool)} tasks, need {needed}")
        test.extend(pool[: args.test_per_type])
        train.extend(pool[args.test_per_type : needed])
    rng.shuffle(train)
    rng.shuffle(test)

    data_dir = _HERE / "data"
    gold_dir = _HERE / "gold"
    data_dir.mkdir(exist_ok=True)
    gold_dir.mkdir(exist_ok=True)
    n_train = _write_split(train, "train", data_dir, gold_dir)
    n_test = _write_split(test, "test", data_dir, gold_dir)
    print(f"wrote {n_train} train + {n_test} test tasks over {len(by_type)} types -> {_HERE}")


if __name__ == "__main__":
    main()

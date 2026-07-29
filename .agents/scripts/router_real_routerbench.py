"""Run the refreshed nine-model RouterBench matrix with objective answer scoring."""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

from wmo.core.files import write_text_atomic
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import Message, Provider
from wmo.providers.pool import ModelPool, PoolEntry, load_pool, pool_provider

logger = logging.getLogger("router-real-routerbench")
BENCHMARK = "routerbench-ours9-refreshed"
LETTER = re.compile(r"\b([A-E])\b[).:]?", re.IGNORECASE)
MAX_ATTEMPTS = 3
RETRY_DELAYS_S = (15, 60)
GRADEABLE_PROVIDER_ERRORS = (
    "content_filter",
    "content management policy",
    "incomplete response: max_output_tokens",
)


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _tasks(path: Path) -> list[dict[str, str]]:
    raw = _json_object(path).get("tasks")
    if not isinstance(raw, list):
        raise ValueError(f"{path} has no task list")
    tasks: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{path} contains a non-object task")
        row = cast("dict[object, object]", item)
        task_id = row.get("task_id")
        prompt = row.get("prompt")
        answer = row.get("answer")
        if not all(isinstance(value, str) for value in (task_id, prompt, answer)):
            raise ValueError(f"{path} contains an incomplete task")
        tasks.append(
            {
                "task_id": cast("str", task_id),
                "prompt": cast("str", prompt),
                "answer": cast("str", answer).upper(),
            }
        )
    return tasks


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _normalize_gradeable_provider_failures(
    path: Path, rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Interpret policy refusal and exhausted output budget as model outcomes, never transport."""
    changed = False
    for row in rows:
        error = row.get("error")
        if row.get("reward") is not None or not isinstance(error, str):
            continue
        lowered = error.lower()
        if not any(marker in lowered for marker in GRADEABLE_PROVIDER_ERRORS):
            continue
        stop_reason = (
            "content_filter"
            if "content_filter" in lowered or "content management policy" in lowered
            else "max_output_tokens"
        )
        row.update(
            {
                "reward": 0.0,
                "success": False,
                "stop_reason": stop_reason,
                "completion_status": "scored_failure",
                "failure_class": stop_reason,
            }
        )
        changed = True
    if changed:
        write_text_atomic(
            path,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )
    return rows


def _attempts(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        scenario_id = row.get("scenario_id")
        model = row.get("model")
        if isinstance(scenario_id, str) and isinstance(model, str):
            grouped.setdefault((scenario_id, model), []).append(row)
    return grouped


def _is_done(rows: list[dict[str, object]]) -> bool:
    return any(isinstance(row.get("reward"), (int, float)) for row in rows) or len(
        rows
    ) >= MAX_ATTEMPTS


def _number(row: dict[str, object], key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _parse_letter(text: str) -> str | None:
    match = LETTER.search(text.strip().strip("'[]\"")[:160])
    return match.group(1).upper() if match else None


def _outcome(
    *,
    task: dict[str, str],
    entry: PoolEntry,
    attempt: int,
    provider: Provider,
) -> ScenarioOutcome:
    started = time.perf_counter()
    try:
        completion = provider.complete(
            "",
            [Message(role="user", content=task["prompt"])],
            max_tokens=1024,
        )
    except Exception as exc:  # noqa: BLE001
        seconds = time.perf_counter() - started
        error = f"{type(exc).__name__}: {exc}"[:1000]
        lowered = error.lower()
        gradeable = any(marker in lowered for marker in GRADEABLE_PROVIDER_ERRORS)
        stop_reason = (
            "content_filter"
            if gradeable
            and ("content_filter" in lowered or "content management policy" in lowered)
            else "max_output_tokens"
            if gradeable
            else "provider_error"
        )
        return ScenarioOutcome(
            scenario_id=task["task_id"],
            task=task["prompt"],
            model=entry.name,
            benchmark=BENCHMARK,
            attempt_number=attempt,
            reward=0.0 if gradeable else None,
            success=False,
            stop_reason=stop_reason,
            call_seconds=[seconds],
            wall_seconds=seconds,
            completion_status="scored_failure" if gradeable else "infrastructure_failure",
            failure_class=stop_reason if gradeable else "provider",
            error=error,
            remeasured=attempt > 1,
        )
    seconds = time.perf_counter() - started
    letter = _parse_letter(completion.text)
    reward = float(letter == task["answer"])
    return ScenarioOutcome(
        scenario_id=task["task_id"],
        task=task["prompt"],
        model=entry.name,
        benchmark=BENCHMARK,
        attempt_number=attempt,
        reward=reward,
        success=reward == 1.0,
        steps=1,
        stop_reason="exact_answer" if letter is not None else "unparsed_answer",
        usage=completion.usage,
        cost_usd=entry.cost_usd(completion.usage),
        call_seconds=[seconds],
        wall_seconds=seconds,
        completion_status="scored_pass" if reward == 1.0 else "scored_failure",
        failure_class="" if reward == 1.0 else "wrong_or_unparsed_answer",
        replies=[completion.text],
        remeasured=attempt > 1,
    )


def _append(path: Path, outcome: ScenarioOutcome, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = outcome.model_dump_json() + "\n"
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def _matrix(rows_path: Path, matrix_path: Path, pool: ModelPool) -> OutcomeMatrix:
    grouped = _attempts(_read_rows(rows_path))
    outcomes: list[ScenarioOutcome] = []
    for attempts in grouped.values():
        parsed = [ScenarioOutcome.model_validate(row) for row in attempts]
        selected = next((row for row in parsed if row.reward is not None), parsed[-1])
        outcomes.append(selected)
    matrix = OutcomeMatrix(pool=pool.models, outcomes=outcomes)
    matrix.save(matrix_path)
    return matrix


def _summary(path: Path, matrix: OutcomeMatrix, total_tasks: int) -> None:
    by_model: dict[str, dict[str, object]] = {}
    for model in matrix.model_names():
        rows = [row for row in matrix.outcomes if row.model == model]
        scored = [row for row in rows if row.reward is not None]
        by_model[model] = {
            "cells": len(rows),
            "scored": len(scored),
            "missing": total_tasks - len(scored),
            "accuracy": (
                sum(cast("float", row.reward) for row in scored) / len(scored)
                if scored
                else None
            ),
            "cost_usd": sum(row.cost_usd for row in rows),
        }
    _write_json(
        path,
        {
            "benchmark": BENCHMARK,
            "tasks": total_tasks,
            "models": len(matrix.pool),
            "cells_expected": total_tasks * len(matrix.pool),
            "cells_present": len(matrix.outcomes),
            "scored": sum(row.reward is not None for row in matrix.outcomes),
            "cost_usd": sum(row.cost_usd for row in matrix.outcomes),
            "by_model": by_model,
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    """Resume every missing RouterBench cell and rebuild the matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", action="append")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--budget-usd", type=float, default=500.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    tasks = _tasks(args.manifest)
    full_pool = load_pool(args.pool)
    selected = set(args.only or [entry.name for entry in full_pool.models])
    unknown = selected - {entry.name for entry in full_pool.models}
    if unknown:
        raise ValueError(f"unknown pool models: {sorted(unknown)}")
    pool = ModelPool(models=[entry for entry in full_pool.models if entry.name in selected])
    rows_path = args.out_dir / "rows.jsonl"
    matrix_path = args.out_dir / "matrix.json"
    lock = threading.Lock()

    for entry in pool.models:
        rows = _normalize_gradeable_provider_failures(rows_path, _read_rows(rows_path))
        grouped = _attempts(rows)
        pending = [
            task
            for task in tasks
            if not _is_done(grouped.get((task["task_id"], entry.name), []))
        ]
        logger.info("%s: %d/%d cells pending", entry.name, len(pending), len(tasks))
        if args.dry_run:
            continue
        provider = pool_provider(entry)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            rows = _normalize_gradeable_provider_failures(rows_path, _read_rows(rows_path))
            grouped = _attempts(rows)
            batch = [
                task
                for task in pending
                if len(grouped.get((task["task_id"], entry.name), [])) == attempt - 1
                and not _is_done(grouped.get((task["task_id"], entry.name), []))
            ]
            if not batch:
                continue
            if attempt > 1:
                time.sleep(RETRY_DELAYS_S[attempt - 2])
            spent = sum(
                _number(row, "cost_usd")
                for row in rows
            )
            if spent >= args.budget_usd:
                raise RuntimeError(
                    f"RouterBench spend ${spent:.2f} reached cap ${args.budget_usd:.2f}"
                )
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(
                        _outcome,
                        task=task,
                        entry=entry,
                        attempt=attempt,
                        provider=provider,
                    ): task
                    for task in batch
                }
                for index, future in enumerate(as_completed(futures), start=1):
                    outcome = future.result()
                    _append(rows_path, outcome, lock)
                    if index % 100 == 0 or index == len(futures):
                        logger.info(
                            "%s attempt %d: %d/%d persisted",
                            entry.name,
                            attempt,
                            index,
                            len(futures),
                        )
            refreshed = _attempts(_read_rows(rows_path))
            pending = [
                task
                for task in pending
                if not _is_done(
                    refreshed.get((task["task_id"], entry.name), [])
                )
            ]

    matrix = _matrix(rows_path, matrix_path, full_pool)
    _summary(args.out_dir / "summary.json", matrix, len(tasks))
    logger.info(
        "matrix: %d/%d cells, $%.4f",
        len(matrix.outcomes),
        len(tasks) * len(full_pool.models),
        sum(row.cost_usd for row in matrix.outcomes),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raise SystemExit(main())

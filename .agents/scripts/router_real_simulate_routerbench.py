"""Predict RouterBench cell correctness with a leak-free Azure GPT-5.5 WMO artifact."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from wmo.core.files import write_text_atomic
from wmo.core.parsing import extract_json_object
from wmo.core.types import Action, ActionKind
from wmo.engine import load_world_model
from wmo.engine.world_model import WorldModel
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import TokenUsage
from wmo.providers.pool import ModelPool, load_pool, pool_api_key
from wmo.tracking import RunRecord

logger = logging.getLogger("router-real-simulate-routerbench")
MAX_ATTEMPTS = 3
RETRY_DELAYS_S = (15, 60)


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _append(path: Path, row: dict[str, object], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _activate_world_model(pool: ModelPool) -> None:
    entry = pool.entry("gpt-5.5")
    config = entry.provider_config()
    key = pool_api_key(entry)
    if not config.endpoint or not config.deployment or not key:
        raise ValueError("gpt-5.5 pool route did not resolve endpoint, deployment, and key")
    os.environ["AZURE_OPENAI_ENDPOINT"] = config.endpoint
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = config.deployment
    os.environ["AZURE_OPENAI_API_KEY"] = key


def _answer(row: ScenarioOutcome) -> str:
    return next((reply for reply in row.replies if reply.strip()), "<no answer>")


def _prediction(content: str) -> tuple[float, str]:
    raw = extract_json_object(content)
    if raw is not None:
        try:
            value = json.loads(raw)
        except ValueError:
            value = None
        if isinstance(value, dict):
            reward = value.get("reward")
            if isinstance(reward, (int, float)):
                return max(0.0, min(1.0, float(reward))), "json-reward"
            correct = value.get("correct")
            if isinstance(correct, bool):
                return float(correct), "json-correct"
    normalized = content.strip().lower()
    if "incorrect" in normalized:
        return 0.0, "text-incorrect"
    if "correct" in normalized:
        return 1.0, "text-correct"
    # A malformed prediction is a world-model outcome, not provider infrastructure. Keeping it
    # as a scored zero prevents format failures from disappearing from the simulated matrix.
    return 0.0, "format-failure"


def _priced_usage(pool: ModelPool, record: RunRecord) -> tuple[dict[str, object], float]:
    dumped = record.model_dump(mode="json")
    total = record.total
    usage = TokenUsage(
        input_tokens=total.input_tokens,
        output_tokens=total.output_tokens,
        cached_input_tokens=total.cached_input_tokens,
        cache_write_input_tokens=total.cache_write_input_tokens,
    )
    return dumped, pool.entry("gpt-5.5").cost_usd(usage)


def _cell(
    world_model: WorldModel,
    pool: ModelPool,
    real: ScenarioOutcome,
    attempt: int,
) -> dict[str, object]:
    session = None
    started = time.monotonic()
    try:
        session = world_model.new_session(task=real.task, enrich=False)
        observation = world_model.step(
            session.id,
            Action(kind=ActionKind.MESSAGE, content=_answer(real)),
        )
        usage_record = world_model.end_session(session.id)
        usage, cost = _priced_usage(pool, usage_record)
        prediction, method = _prediction(observation.content)
        return {
            "scenario_id": real.scenario_id,
            "task": real.task,
            "model": real.model,
            "attempt_number": attempt,
            "reward": prediction,
            "success": prediction >= 0.5,
            "critique": "",
            "steps": 1,
            "tool_calls": 0,
            "stop_reason": f"world-model-{method}",
            "usage": real.usage.model_dump(mode="json"),
            "cost_usd": real.cost_usd,
            "call_seconds": real.call_seconds,
            "wall_seconds": time.monotonic() - started,
            "completion_status": (
                "world_model_format_failure" if method == "format-failure" else "predicted"
            ),
            "failure_class": "world_model_format" if method == "format-failure" else "",
            "replies": real.replies,
            "error": None,
            "remeasured": attempt > 1,
            "world_model_observation": observation.model_dump(mode="json"),
            "world_model_usage": usage,
            "world_model_cost_usd": cost,
            "ground_truth_reward": real.reward,
        }
    except Exception as exc:  # noqa: BLE001
        if session is not None:
            try:
                world_model.end_session(session.id)
            except Exception:  # noqa: BLE001
                pass
        return {
            "scenario_id": real.scenario_id,
            "task": real.task,
            "model": real.model,
            "attempt_number": attempt,
            "reward": None,
            "success": False,
            "stop_reason": "world_model_provider_error",
            "usage": real.usage.model_dump(mode="json"),
            "cost_usd": real.cost_usd,
            "call_seconds": real.call_seconds,
            "wall_seconds": time.monotonic() - started,
            "completion_status": "infrastructure_failure",
            "failure_class": "world_model_provider",
            "replies": real.replies,
            "error": f"{type(exc).__name__}: {exc}"[:1000],
            "remeasured": attempt > 1,
            "world_model_cost_usd": 0.0,
            "ground_truth_reward": real.reward,
        }


def _selected(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        scenario_id, model = row.get("scenario_id"), row.get("model")
        if isinstance(scenario_id, str) and isinstance(model, str):
            grouped.setdefault((scenario_id, model), []).append(row)
    return [
        next(
            (
                row
                for row in sorted(
                    attempts, key=lambda item: int(item.get("attempt_number", 0))
                )
                if isinstance(row.get("reward"), (int, float))
            ),
            attempts[-1],
        )
        for attempts in grouped.values()
    ]


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--real-matrix", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spend-ledger", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--budget-usd", type=float, default=5000.0)
    args = parser.parse_args()
    matrix_path = args.out_dir / "matrix.json"
    if matrix_path.is_file():
        logger.info("simulated RouterBench matrix already complete: %s", matrix_path)
        return 0
    pool = load_pool(args.pool)
    real_matrix = OutcomeMatrix.load(args.real_matrix)
    expected = len(real_matrix.scenario_ids()) * len(real_matrix.model_names())
    _activate_world_model(pool)
    world_model, _provider = load_world_model(
        args.model_dir, telemetry_root=args.out_dir / "telemetry"
    )
    rows_path = args.out_dir / "rows.jsonl"
    lock = threading.Lock()
    real_by_key = {
        (row.scenario_id, row.model): row
        for row in real_matrix.outcomes
        if row.reward is not None
    }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        selected = _selected(_read_rows(rows_path))
        done = {
            (str(row["scenario_id"]), str(row["model"]))
            for row in selected
            if isinstance(row.get("reward"), (int, float))
        }
        pending = [row for key, row in real_by_key.items() if key not in done]
        if not pending:
            break
        if attempt > 1:
            time.sleep(RETRY_DELAYS_S[attempt - 2])
        spent = sum(
            _number(row.get("world_model_cost_usd", 0.0))
            for row in _read_rows(rows_path)
            if isinstance(row.get("world_model_cost_usd"), (int, float))
        )
        if spent >= args.budget_usd:
            raise RuntimeError(
                f"RouterBench WMO spend ${spent:.2f} reached cap ${args.budget_usd:.2f}"
            )
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(_cell, world_model, pool, row, attempt): row
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                _append(rows_path, future.result(), lock)
                if index % 100 == 0 or index == len(futures):
                    logger.info(
                        "attempt %d: %d/%d predictions persisted",
                        attempt,
                        index,
                        len(futures),
                    )
    selected = _selected(_read_rows(rows_path))
    outcomes = [ScenarioOutcome.model_validate(row) for row in selected]
    matrix = OutcomeMatrix(pool=real_matrix.pool, outcomes=outcomes)
    matrix.save(matrix_path)
    gradeable = sum(row.reward is not None for row in outcomes)
    wm_cost = sum(
        _number(row.get("world_model_cost_usd", 0.0))
        for row in selected
        if isinstance(row.get("world_model_cost_usd"), (int, float))
    )
    summary = {
        "benchmark": "routerbench",
        "phase": "world-model",
        "world_model": "azure-gpt-5.5",
        "cells_expected": expected,
        "cells_present": len(outcomes),
        "gradeable": gradeable,
        "format_failures": sum(
            row.get("failure_class") == "world_model_format" for row in selected
        ),
        "world_model_cost_usd": wm_cost,
        "candidate_cost_usd": 0.0,
        "candidate_cost_note": "phase-1 candidate replies and costs were reused, not called again",
    }
    write_text_atomic(
        args.out_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    with args.spend_ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "phase": "world-model",
                    "benchmark": "routerbench",
                    "provider_role": "world-model-serve",
                    "model": "azure-gpt-5.5",
                    "run": str(args.out_dir),
                    "realized_usd": wm_cost,
                    "cells": len(outcomes),
                    "source": str(rows_path),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
    return 0 if gradeable == expected else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raise SystemExit(main())

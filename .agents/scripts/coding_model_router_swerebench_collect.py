"""Collect the complete external SWE-rebench matrix into fit-ready outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger("coding-router-swerebench-collect")

PROTOCOL = "coding-router-swerebench-development-collection-v1"
CORPUS_SHA256 = "7d846b5576d15e68fd18ac21bfe0610cc1614b3b35ec0ae0cb8cfae0b82962c1"
SMOKE_REPORT_SHA256 = "ee76a57040cbe7aaef692d2fc3f3df66d7a556cbf6dda74119e0802cb4230e13"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
USAGE_FIELDS = (
    "prompt_tokens",
    "cached_input_tokens",
    "completion_tokens",
    "reasoning_tokens",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path))


def _usage(value: object, label: str) -> dict[str, int]:
    raw = _object(value, label)
    result: dict[str, int] = {}
    for field in USAGE_FIELDS:
        amount = raw.get(field)
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError(f"{label} has invalid {field}")
        result[field] = amount
    if result["reasoning_tokens"] > result["completion_tokens"]:
        raise ValueError(f"{label} reasoning exceeds completion tokens")
    return result


def _cost(usage: dict[str, int]) -> float:
    return (
        usage["prompt_tokens"] / 1_000_000
        + usage["cached_input_tokens"] * 0.1 / 1_000_000
        + usage["completion_tokens"] * 6.0 / 1_000_000
    )


def _smoke_cells(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if _sha256(path) != SMOKE_REPORT_SHA256:
        raise ValueError("smoke report hash mismatch")
    report = _read_object(path)
    if report.get("valid") is not True:
        raise ValueError("smoke report is not valid")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    archives = report.get("archives")
    if not isinstance(archives, list):
        raise ValueError("smoke report has no archives")
    for raw_archive in archives:
        archive = _object(raw_archive, "smoke archive")
        effort = archive.get("effort")
        cells = archive.get("cells")
        if effort not in {"xhigh", "max"} or not isinstance(cells, list):
            raise ValueError("smoke archive has invalid effort or cells")
        for raw_cell in cells:
            cell = _object(raw_cell, "smoke cell")
            task_id = cell.get("task_id")
            if not isinstance(task_id, str):
                raise ValueError("smoke cell has no task id")
            key = (task_id, str(effort))
            if key in result:
                raise ValueError(f"duplicate smoke cell {key}")
            result[key] = cell
    if len(result) != 4:
        raise ValueError("smoke report does not contain exactly four reused cells")
    return result


def _outcome(
    task: dict[str, Any],
    effort: str,
    attempt: int,
    cell: dict[str, Any],
    *,
    provenance: str,
) -> dict[str, object]:
    reward = cell.get("reward")
    if (
        not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or float(reward) not in {0.0, 1.0}
    ):
        raise ValueError("cell has invalid gradeable reward")
    usage = _usage(cell.get("usage"), "cell usage")
    return {
        "task_id": str(task["task_id"]),
        "repository": str(task["repository"]),
        "language": str(task["language"]),
        "prompt": str(task["prompt"]),
        "prompt_sha256": str(task["prompt_sha256"]),
        "arm": f"luna-{effort}",
        "model": "gpt-5.6-luna",
        "reasoning_effort": effort,
        "attempt_number": attempt,
        "reward": float(reward),
        "reward_provenance": cell.get("reward_provenance", "official verifier"),
        "official_verifier_reached": cell.get("official_verifier_reached", True),
        "cost_usd": _cost(usage),
        "cost_provenance": "trace-derived frozen list-price estimate",
        "usage": usage,
        "provider_calls": int(cell.get("provider_calls", 0)),
        "stop_condition": cell.get("stop_condition"),
        "patch_sha256": cell.get("patch_sha256"),
        "provenance": provenance,
        "target_outcomes_used": False,
    }


def collect(root: Path, corpus_path: Path, smoke_report_path: Path, output: Path) -> None:
    """Validate every completed effort report and write exactly 2,000 outcomes."""
    if _sha256(corpus_path) != CORPUS_SHA256:
        raise ValueError("development corpus hash mismatch")
    progress = _read_object(root / "progress.json")
    if (
        progress.get("complete_tasks") != 200
        or progress.get("completed_scientific_cells") != 2_000
        or progress.get("failed_tasks") != 0
    ):
        raise ValueError("development matrix is not complete and failure-free")
    corpus = _read_object(corpus_path)
    raw_tasks = corpus.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 200:
        raise ValueError("development corpus does not contain 200 tasks")
    tasks = [_object(task, f"corpus task {index}") for index, task in enumerate(raw_tasks)]
    reused = _smoke_cells(smoke_report_path)
    outcomes: list[dict[str, object]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    for index, task in enumerate(tasks):
        task_id = str(task["task_id"])
        task_dir = root / "tasks" / f"{index:04d}"
        state_path = task_dir / "state.json"
        state = _read_object(state_path)
        if state.get("stage") != "complete" or state.get("task_id") != task_id:
            raise ValueError(f"task {index} is not complete with frozen identity")
        efforts = _object(state.get("efforts"), f"task {index} efforts")
        input_hashes[task_id] = {"state": _sha256(state_path)}
        for effort in EFFORTS:
            payload = _object(efforts.get(effort), f"task {index} {effort} state")
            report_path = task_dir / f"{effort}.report.json"
            archive_path = task_dir / f"{effort}.tar.gz"
            report_sha = _sha256(report_path)
            archive_sha = _sha256(archive_path)
            if report_sha != payload.get("report_sha256"):
                raise ValueError(f"task {index} {effort} report hash mismatch")
            if archive_sha != payload.get("archive_sha256"):
                raise ValueError(f"task {index} {effort} archive hash mismatch")
            report = _read_object(report_path)
            if (
                report.get("valid") is not True
                or report.get("task_id") != task_id
                or report.get("effort") != effort
            ):
                raise ValueError(f"task {index} {effort} report identity mismatch")
            cells = report.get("cells")
            if not isinstance(cells, list):
                raise ValueError(f"task {index} {effort} report has no cells")
            attempt_cells: dict[int, dict[str, Any]] = {}
            for raw_cell in cells:
                cell = _object(raw_cell, f"task {index} {effort} cell")
                attempt = cell.get("attempt_number")
                if not isinstance(attempt, int) or isinstance(attempt, bool):
                    raise ValueError(f"task {index} {effort} has invalid attempt")
                if attempt in attempt_cells:
                    raise ValueError(f"task {index} {effort} duplicates attempt {attempt}")
                attempt_cells[attempt] = cell
            smoke_cell = reused.get((task_id, effort))
            if smoke_cell is not None:
                if 0 in attempt_cells or set(attempt_cells) != {1}:
                    raise ValueError(f"task {index} {effort} did not preserve smoke attempt zero")
                attempt_cells[0] = smoke_cell
            if set(attempt_cells) != {0, 1}:
                raise ValueError(f"task {index} {effort} attempts are incomplete")
            for attempt in (0, 1):
                provenance = (
                    "reused-valid-smoke"
                    if smoke_cell is not None and attempt == 0
                    else "development-matrix"
                )
                outcomes.append(
                    _outcome(
                        task,
                        effort,
                        attempt,
                        attempt_cells[attempt],
                        provenance=provenance,
                    )
                )
            input_hashes[task_id][f"{effort}_report"] = report_sha
            input_hashes[task_id][f"{effort}_archive"] = archive_sha
    if len(outcomes) != 2_000:
        raise ValueError(f"expected 2,000 outcomes, found {len(outcomes)}")
    identities = {
        (row["task_id"], row["reasoning_effort"], row["attempt_number"])
        for row in outcomes
    }
    if len(identities) != 2_000:
        raise ValueError("collected outcome identities are not unique")
    output.mkdir(parents=True, exist_ok=False)
    outcomes_path = output / "outcomes.jsonl"
    outcomes_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in outcomes),
        encoding="utf-8",
    )
    total_cost = math.fsum(float(row["cost_usd"]) for row in outcomes)
    reused_cost = math.fsum(
        float(row["cost_usd"])
        for row in outcomes
        if row["provenance"] == "reused-valid-smoke"
    )
    audit = {
        "protocol": PROTOCOL,
        "valid": True,
        "tasks": 200,
        "efforts": list(EFFORTS),
        "attempts_per_effort": 2,
        "cells": 2_000,
        "unique_cell_identities": len(identities),
        "reused_smoke_cells": 4,
        "new_matrix_cells": 1_996,
        "outcome_cost_usd": total_cost,
        "reused_smoke_cost_usd": reused_cost,
        "new_matrix_cost_usd": total_cost - reused_cost,
        "rough_cumulative_experiment_spend_usd": 405.7678502 + total_cost - reused_cost,
        "target_outcomes_used": False,
        "deep_swe_outcomes_accessed": False,
        "outcomes_sha256": _sha256(outcomes_path),
        "corpus_sha256": CORPUS_SHA256,
        "smoke_report_sha256": SMOKE_REPORT_SHA256,
        "input_hashes": input_hashes,
    }
    (output / "completion-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "collected cells=%d new_cost_usd=%.6f output_sha256=%s",
        len(outcomes),
        total_cost - reused_cost,
        audit["outcomes_sha256"],
    )


def main() -> None:
    """Parse paths and collect a complete matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.corpus, args.smoke_report, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()

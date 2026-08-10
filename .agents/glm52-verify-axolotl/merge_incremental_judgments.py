#!/usr/bin/env python3
"""Merge completed trajectory judgments against an expanded candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file without silently accepting blank records."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line {line_number} in {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} in {path} is not an object")
            rows.append(value)
    return rows


def merge_judgments(
    candidates: list[dict[str, Any]],
    judgment_sets: list[list[dict[str, Any]]],
    *,
    expected_model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return one validated judgment per candidate in manifest order."""
    candidate_by_row: dict[int, dict[str, Any]] = {}
    task_ids: set[str] = set()
    orders: set[int] = set()
    for candidate in candidates:
        row_index = int(candidate["row_index"])
        task_id = str(candidate["task_id"])
        manifest_order = int(candidate["manifest_order"])
        if row_index in candidate_by_row:
            raise ValueError(f"duplicate candidate row_index {row_index}")
        if task_id in task_ids:
            raise ValueError(f"duplicate candidate task_id {task_id!r}")
        if manifest_order in orders:
            raise ValueError(f"duplicate candidate manifest_order {manifest_order}")
        candidate_by_row[row_index] = candidate
        task_ids.add(task_id)
        orders.add(manifest_order)

    judgments: dict[int, dict[str, Any]] = {}
    for source_number, rows in enumerate(judgment_sets, start=1):
        for judgment in rows:
            row_index = int(judgment["row_index"])
            if row_index in judgments:
                raise ValueError(
                    f"duplicate judgment for row_index {row_index} across source files"
                )
            candidate = candidate_by_row.get(row_index)
            if candidate is None:
                raise ValueError(f"judgment row_index {row_index} is not a candidate")
            for key in ("task_id", "rollout_id", "manifest_order"):
                if judgment.get(key) != candidate.get(key):
                    raise ValueError(
                        f"source {source_number} row {row_index} mismatch at {key}: "
                        f"{judgment.get(key)!r} != {candidate.get(key)!r}"
                    )
            if judgment.get("schema") != "glm52-terminal-trajectory-judgment-v1":
                raise ValueError(f"row {row_index} has unexpected judgment schema")
            if judgment.get("decision") is None:
                raise ValueError(f"row {row_index} has no normalized decision")
            if (
                expected_model_id is not None
                and judgment.get("judge_model_id") != expected_model_id
            ):
                raise ValueError(
                    f"row {row_index} judge model {judgment.get('judge_model_id')!r} "
                    f"does not match {expected_model_id!r}"
                )
            judgments[row_index] = judgment

    missing = sorted(set(candidate_by_row) - set(judgments))
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"missing judgments for {len(missing)} candidates: {preview}")
    return [
        judgments[int(candidate["row_index"])]
        for candidate in sorted(candidates, key=lambda row: int(row["manifest_order"]))
    ]


def main() -> int:
    """Merge existing and incremental judge outputs and write an audit manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--judgments", required=True, type=Path, nargs="+")
    parser.add_argument("--expected-model-id")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    for output in (args.output, args.manifest):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    candidates = load_jsonl(args.candidates)
    source_rows = [load_jsonl(path) for path in args.judgments]
    merged = merge_judgments(
        candidates,
        source_rows,
        expected_model_id=args.expected_model_id,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema": "glm52-incremental-judgment-merge-v1",
        "candidates": str(args.candidates),
        "candidates_sha256": sha256_file(args.candidates),
        "candidate_count": len(candidates),
        "judgment_sources": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
            for path, rows in zip(args.judgments, source_rows, strict=True)
        ],
        "expected_model_id": args.expected_model_id,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "output_rows": len(merged),
        "unique_tasks": len({str(row["task_id"]) for row in merged}),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(manifest, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

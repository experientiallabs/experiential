#!/usr/bin/env python3
"""Build a Qwen chat view for replay candidates without granting admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from build_sft_subset import normalize_messages


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest for bytes."""
    return hashlib.sha256(value).hexdigest()


def build_rows(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ordered Qwen chat rows linked exactly to replay audit rows."""
    output: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    source_indices: set[int] = set()
    for position, audit in enumerate(audit_rows):
        if audit.get("schema") != "glm52-real-verifier-replay-candidate-v1":
            raise ValueError(f"audit row {position}: unexpected schema")
        admission = audit.get("admission")
        if not isinstance(admission, dict):
            raise ValueError(f"audit row {position}: missing admission")
        if admission.get("selected_for_replay") is not True:
            raise ValueError(f"audit row {position}: not selected for replay")
        if admission.get("selected_for_sft") is not False:
            raise ValueError(f"audit row {position}: premature SFT admission")
        source = audit.get("source")
        source_raw = audit.get("source_raw_json")
        if not isinstance(source, dict) or not isinstance(source_raw, str):
            raise ValueError(f"audit row {position}: missing exact source")
        if json.loads(source_raw) != source:
            raise ValueError(f"audit row {position}: source JSON mismatch")
        source_hash = sha256_bytes(source_raw.encode("utf-8"))
        if source_hash != audit.get("source_row_sha256"):
            raise ValueError(f"audit row {position}: source hash mismatch")

        task_id = source.get("task_id")
        source_index = audit.get("source_row_index")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"audit row {position}: missing task_id")
        if not isinstance(source_index, int):
            raise ValueError(f"audit row {position}: invalid source row index")
        if task_id in task_ids:
            raise ValueError(f"duplicate task_id: {task_id}")
        if source_index in source_indices:
            raise ValueError(f"duplicate source row index: {source_index}")
        task_ids.add(task_id)
        source_indices.add(source_index)

        raw_messages = json.loads(source["message_log_json"])
        tools = json.loads(source["tools_json"])
        if not isinstance(raw_messages, list) or not isinstance(tools, list):
            raise ValueError(f"audit row {position}: invalid messages or tools")
        candidate = audit.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError(f"audit row {position}: missing candidate evidence")
        output.append(
            {
                "messages": normalize_messages(raw_messages),
                "tools": tools,
                "provenance": {
                    "source_row_index": source_index,
                    "source_row_sha256": source_hash,
                    "rollout_id": source["rollout_id"],
                    "task_id": task_id,
                    "manifest_order": source["manifest_order"],
                    "replica": source["replica"],
                    "source_student_tokens": source["n_student_tokens"],
                    "source_supervised_tokens": source["n_supervised_tokens"],
                    "candidate_prompt_sha256": candidate["prompt_sha256"],
                    "nearest_evaluation_similarity": candidate[
                        "nearest_evaluation_similarity"
                    ],
                    "real_verifier_admission_pending": True,
                    "selected_for_sft": False,
                },
            }
        )
    if not output:
        raise ValueError("audit dataset is empty")
    return output


def main() -> int:
    """Write and hash the pre-admission Qwen training view."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output, args.summary):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite: {output}")
    audit_rows = [
        json.loads(line)
        for line in args.audit_dataset.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = build_rows(audit_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    summary = {
        "schema": "glm52-replay-candidate-qwen-training-view-v1",
        "rows": len(rows),
        "unique_tasks": len({row["provenance"]["task_id"] for row in rows}),
        "audit_dataset": str(args.audit_dataset),
        "audit_dataset_sha256": sha256_bytes(args.audit_dataset.read_bytes()),
        "output": str(args.output),
        "output_sha256": sha256_bytes(args.output.read_bytes()),
        "real_verifier_admission_pending": True,
        "training_eligible": False,
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Select teacher-perfect SFT rows where base has a scored non-perfect result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Hash one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL objects."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid or empty JSONL: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact deterministic JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def unique_index(
    rows: list[dict[str, Any]], task_id_getter: Any, label: str
) -> dict[str, dict[str, Any]]:
    """Index rows by a required unique task ID."""
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        task_id = task_id_getter(row)
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{label} row {index}: missing task ID")
        if task_id in result:
            raise ValueError(f"{label}: duplicate task ID {task_id}")
        result[task_id] = row
    return result


def load_base_results(root: Path) -> dict[str, dict[str, Any]]:
    """Load unique episode results from a base evaluation root."""
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in root.rglob("episode_result.json")]
    return unique_index(rows, lambda row: row.get("task_id"), "base")


def select_headroom(
    *,
    audit_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    base_rows: dict[str, dict[str, Any]],
    max_base_reward: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select teacher-perfect rows with scored base reward below the threshold."""
    audit = unique_index(
        audit_rows, lambda row: row.get("source", {}).get("task_id"), "audit"
    )
    qwen = unique_index(
        qwen_rows, lambda row: row.get("provenance", {}).get("task_id"), "qwen"
    )
    if set(audit) != set(qwen):
        raise ValueError("audit and Qwen task sets differ")
    if not set(audit) <= set(base_rows):
        raise ValueError(f"base results missing tasks: {sorted(set(audit) - set(base_rows))}")

    selected_audit: list[dict[str, Any]] = []
    selected_qwen: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for task_id in audit:
        audit_row = audit[task_id]
        qwen_row = qwen[task_id]
        base = base_rows[task_id]
        verifier = audit_row.get("environment_verifier_replay", {})
        teacher_perfect = (
            verifier.get("status") == "scored" and verifier.get("reward") == 1.0
        )
        base_scored = (
            base.get("status") == "scored"
            and isinstance(base.get("reward"), (int, float))
        )
        selected = (
            teacher_perfect
            and base_scored
            and float(base["reward"]) < max_base_reward
        )
        reasons: list[str] = []
        if not teacher_perfect:
            reasons.append("teacher_replay_not_perfect")
        if not base_scored:
            reasons.append(f"base_not_scored:{base.get('status')}")
        elif float(base["reward"]) >= max_base_reward:
            reasons.append(f"base_reward_at_or_above_{max_base_reward:g}")
        ledger.append(
            {
                "task_id": task_id,
                "source_row_sha256": audit_row.get("source_row_sha256"),
                "teacher_real_verifier_status": verifier.get("status"),
                "teacher_real_verifier_reward": verifier.get("reward"),
                "base_status": base.get("status"),
                "base_reward": base.get("reward"),
                "selected_for_base_headroom_sft": selected,
                "exclusion_reasons": reasons,
                "evidence_scope": "training_overlap_headroom_selection_not_held_out",
            }
        )
        if selected:
            selected_audit.append(audit_row)
            selected_qwen.append(qwen_row)

    if not selected_qwen:
        raise ValueError("base-headroom filter selected no rows")
    return selected_audit, selected_qwen, ledger


def main() -> int:
    """Build a hashed base-headroom SFT bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dataset", type=Path, required=True)
    parser.add_argument("--qwen-dataset", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--max-base-reward", type=float, default=1.0)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--output-qwen", type=Path, required=True)
    parser.add_argument("--output-ledger", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.max_base_reward <= 1:
        raise ValueError("max base reward must be in (0, 1]")

    audit_rows = read_jsonl(args.audit_dataset)
    qwen_rows = read_jsonl(args.qwen_dataset)
    selected_audit, selected_qwen, ledger = select_headroom(
        audit_rows=audit_rows,
        qwen_rows=qwen_rows,
        base_rows=load_base_results(args.base_root),
        max_base_reward=args.max_base_reward,
    )
    write_jsonl(args.output_audit, selected_audit)
    write_jsonl(args.output_qwen, selected_qwen)
    write_jsonl(args.output_ledger, ledger)
    manifest = {
        "schema": "xtoken-base-headroom-sft-bundle-v1",
        "evidence_scope": "training_overlap_headroom_selection_not_held_out",
        "selection_rule": "teacher_real_verifier_reward_eq_1_and_base_scored_reward_lt_threshold",
        "max_base_reward_exclusive": args.max_base_reward,
        "input_rows": len(audit_rows),
        "selected_rows": len(selected_qwen),
        "selected_task_ids": [row["provenance"]["task_id"] for row in selected_qwen],
        "unscored_base_tasks_are_excluded_not_zeroed": True,
        "model_judgment_is_official_task_verification": False,
        "final_environment_verifier_used": True,
        "audit_dataset_sha256": sha256(args.audit_dataset),
        "qwen_dataset_sha256": sha256(args.qwen_dataset),
        "output_audit_sha256": sha256(args.output_audit),
        "output_qwen_sha256": sha256(args.output_qwen),
        "output_ledger_sha256": sha256(args.output_ledger),
        "audit_companion_preserves_all_model_decisions_rationales_and_provenance": True,
        "training_messages_are_byte_equivalent_to_the_selected_materialized_rows": True,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a real-verifier-filtered SFT bundle without losing judge audit data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest for bytes."""
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    """Hash a JSON object with stable separators and key ordering."""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a non-empty JSONL file."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows are not objects: {path}")
    return rows


def index_unique(
    rows: list[dict[str, Any]], key: str, *, context: str
) -> dict[str, dict[str, Any]]:
    """Index rows on a required unique string key."""
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context} row {index}: missing {key}")
        if value in result:
            raise ValueError(f"{context}: duplicate {key} {value}")
        result[value] = row
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact JSONL deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def build_bundle(
    *,
    audit_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    minimum_reward: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Return selected audit rows, train rows, and a complete admission ledger."""
    audit_by_task: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(audit_rows):
        source = row.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"audit row {index}: missing source")
        task_id = source.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"audit row {index}: missing source task_id")
        if task_id in audit_by_task:
            raise ValueError(f"audit: duplicate task_id {task_id}")
        audit_by_task[task_id] = row

    qwen_by_task: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(qwen_rows):
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"qwen row {index}: missing provenance")
        task_id = provenance.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"qwen row {index}: missing provenance task_id")
        if task_id in qwen_by_task:
            raise ValueError(f"qwen: duplicate task_id {task_id}")
        qwen_by_task[task_id] = row

    replay_by_task = index_unique(replay_rows, "task_id", context="replay")
    expected_tasks = set(audit_by_task)
    for label, actual in (
        ("qwen", set(qwen_by_task)),
        ("replay", set(replay_by_task)),
    ):
        if actual != expected_tasks:
            raise ValueError(
                f"{label} task set differs: missing={sorted(expected_tasks - actual)} "
                f"extra={sorted(actual - expected_tasks)}"
            )

    selected_audit: list[dict[str, Any]] = []
    selected_qwen: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for task_id, audit in audit_by_task.items():
        qwen = qwen_by_task[task_id]
        replay = replay_by_task[task_id]
        source_row_sha256 = str(audit["source_row_sha256"])
        provenance = qwen["provenance"]
        if provenance.get("source_row_sha256") != source_row_sha256:
            raise ValueError(f"{task_id}: qwen source row hash mismatch")
        if replay.get("source_row_sha256") != source_row_sha256:
            raise ValueError(f"{task_id}: replay source row hash mismatch")
        status = replay.get("status")
        reward = replay.get("reward")
        selected = (
            status == "scored"
            and isinstance(reward, (int, float))
            and float(reward) >= minimum_reward
        )
        exclusion_reasons: list[str] = []
        if status != "scored":
            exclusion_reasons.append(f"replay_status:{status}")
        if not isinstance(reward, (int, float)):
            exclusion_reasons.append("replay_reward_missing")
        elif float(reward) < minimum_reward:
            exclusion_reasons.append(
                f"replay_reward_below_{minimum_reward:g}:{float(reward):g}"
            )
        replay_sha256 = canonical_sha256(replay)
        ledger.append(
            {
                "task_id": task_id,
                "source_row_index": audit["source_row_index"],
                "source_row_sha256": source_row_sha256,
                "rollout_id": audit["source"]["rollout_id"],
                "model_judge_selected_for_sft": audit["admission"][
                    "selected_for_sft"
                ],
                "replay_status": status,
                "real_verifier_reward": reward,
                "minimum_reward": minimum_reward,
                "selected_for_real_verifier_sft": selected,
                "exclusion_reasons": exclusion_reasons,
                "replay_result_sha256": replay_sha256,
            }
        )
        if not selected:
            continue

        augmented_audit = dict(audit)
        augmented_audit["environment_verifier_replay"] = replay
        augmented_audit["real_verifier_admission"] = {
            "rule": f"replay_status_scored_and_reward_gte_{minimum_reward:g}",
            "selected_for_sft": True,
            "replay_result_sha256": replay_sha256,
            "model_judgment_is_official_task_verification": False,
            "final_environment_verifier_used": True,
        }
        selected_audit.append(augmented_audit)

        augmented_qwen = dict(qwen)
        augmented_provenance = dict(provenance)
        augmented_provenance.update(
            {
                "real_verifier_reward": float(reward),
                "real_verifier_replay_status": status,
                "real_verifier_replay_result_sha256": replay_sha256,
                "real_verifier_filter_minimum_reward": minimum_reward,
                "model_judgment_is_official_task_verification": False,
                "final_environment_verifier_used": True,
            }
        )
        augmented_qwen["provenance"] = augmented_provenance
        selected_qwen.append(augmented_qwen)

    return selected_audit, selected_qwen, ledger


def main() -> int:
    """Build and hash the filtered training bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dataset", type=Path, required=True)
    parser.add_argument("--qwen-dataset", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path, required=True)
    parser.add_argument("--minimum-reward", type=float, default=1.0)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--output-qwen", type=Path, required=True)
    parser.add_argument("--output-ledger", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not 0 <= args.minimum_reward <= 1:
        raise ValueError("minimum reward must be in [0, 1]")
    replay_summary = json.loads(args.replay_summary.read_text(encoding="utf-8"))
    if not replay_summary.get("final_environment_verifier_used"):
        raise ValueError("replay summary does not attest final environment verifiers")
    if replay_summary.get("model_judgment_is_official_task_verification") is not False:
        raise ValueError("replay summary has an invalid model-judgment boundary")

    audit_rows = read_jsonl(args.audit_dataset)
    qwen_rows = read_jsonl(args.qwen_dataset)
    replay_paths = sorted(args.replay_root.glob("*/replay_result.json"))
    if len(replay_paths) != replay_summary.get("attempted"):
        raise ValueError("replay episode count differs from summary attempted count")
    replay_rows = [json.loads(path.read_text(encoding="utf-8")) for path in replay_paths]
    selected_audit, selected_qwen, ledger = build_bundle(
        audit_rows=audit_rows,
        qwen_rows=qwen_rows,
        replay_rows=replay_rows,
        minimum_reward=args.minimum_reward,
    )
    if not selected_qwen:
        raise ValueError("real-verifier filter selected no SFT rows")

    write_jsonl(args.output_audit, selected_audit)
    write_jsonl(args.output_qwen, selected_qwen)
    write_jsonl(args.output_ledger, ledger)
    manifest = {
        "schema": "xtoken-real-verifier-filtered-sft-bundle-v1",
        "minimum_reward": args.minimum_reward,
        "model_judgment_is_official_task_verification": False,
        "final_environment_verifier_used": True,
        "input_rows": len(audit_rows),
        "selected_rows": len(selected_qwen),
        "excluded_rows": len(audit_rows) - len(selected_qwen),
        "selected_task_ids": [row["provenance"]["task_id"] for row in selected_qwen],
        "audit_dataset": str(args.audit_dataset),
        "audit_dataset_sha256": sha256_bytes(args.audit_dataset.read_bytes()),
        "qwen_dataset": str(args.qwen_dataset),
        "qwen_dataset_sha256": sha256_bytes(args.qwen_dataset.read_bytes()),
        "replay_summary": str(args.replay_summary),
        "replay_summary_sha256": sha256_bytes(args.replay_summary.read_bytes()),
        "output_audit": str(args.output_audit),
        "output_audit_sha256": sha256_bytes(args.output_audit.read_bytes()),
        "output_qwen": str(args.output_qwen),
        "output_qwen_sha256": sha256_bytes(args.output_qwen.read_bytes()),
        "output_ledger": str(args.output_ledger),
        "output_ledger_sha256": sha256_bytes(args.output_ledger.read_bytes()),
        "audit_companion_preserves_all_model_decisions_rationales_and_provenance": True,
        "training_messages_are_byte_equivalent_to_the_selected_materialized_rows": True,
    }
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("filtered SFT bundle %s", json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

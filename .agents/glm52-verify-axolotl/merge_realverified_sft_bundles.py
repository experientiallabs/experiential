#!/usr/bin/env python3
"""Merge disjoint real-verifier-admitted SFT bundles without reordering rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 without loading the full file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a non-empty object JSONL file."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid or empty JSONL: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact JSONL deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def merge_bundles(
    bundles: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ordered audit and Qwen rows after strict admission checks."""
    if not bundles:
        raise ValueError("no input bundles")
    output_audit: list[dict[str, Any]] = []
    output_qwen: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    source_indices: set[int] = set()
    source_hashes: set[str] = set()
    for label, audit_rows, qwen_rows in bundles:
        if len(audit_rows) != len(qwen_rows):
            raise ValueError(f"{label}: audit and Qwen row counts differ")
        for position, (audit, qwen) in enumerate(zip(audit_rows, qwen_rows, strict=True)):
            source = audit.get("source")
            provenance = qwen.get("provenance")
            admission = audit.get("real_verifier_admission")
            if not isinstance(source, dict) or not isinstance(provenance, dict):
                raise ValueError(f"{label} row {position}: missing provenance")
            if not isinstance(admission, dict) or admission.get("selected_for_sft") is not True:
                raise ValueError(f"{label} row {position}: missing verifier admission")
            if provenance.get("real_verifier_reward") != 1.0:
                raise ValueError(f"{label} row {position}: reward is not perfect")
            explicitly_admitted = (
                provenance.get("selected_for_sft") is True
                and provenance.get("real_verifier_admission_pending") is False
            )
            legacy_admitted = (
                "selected_for_sft" not in provenance
                and "real_verifier_admission_pending" not in provenance
                and provenance.get("final_environment_verifier_used") is True
                and provenance.get("model_judgment_is_official_task_verification")
                is False
            )
            if not explicitly_admitted and not legacy_admitted:
                raise ValueError(
                    f"{label} row {position}: Qwen row lacks a valid admission boundary"
                )
            task_id = source.get("task_id")
            source_index = audit.get("source_row_index")
            source_hash = audit.get("source_row_sha256")
            if task_id != provenance.get("task_id"):
                raise ValueError(f"{label} row {position}: task alignment mismatch")
            if source_hash != provenance.get("source_row_sha256"):
                raise ValueError(f"{label} row {position}: source hash mismatch")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"{label} row {position}: invalid task ID")
            if not isinstance(source_index, int) or not isinstance(source_hash, str):
                raise ValueError(f"{label} row {position}: invalid source identity")
            if task_id in task_ids:
                raise ValueError(f"duplicate task across bundles: {task_id}")
            if source_index in source_indices:
                raise ValueError(f"duplicate source row across bundles: {source_index}")
            if source_hash in source_hashes:
                raise ValueError(f"duplicate source hash across bundles: {source_hash}")
            task_ids.add(task_id)
            source_indices.add(source_index)
            source_hashes.add(source_hash)
            output_audit.append(audit)
            normalized_qwen = dict(qwen)
            normalized_provenance = dict(provenance)
            normalized_provenance["selected_for_sft"] = True
            normalized_provenance["real_verifier_admission_pending"] = False
            normalized_provenance["legacy_real_verifier_admission_upgraded"] = (
                legacy_admitted
            )
            normalized_qwen["provenance"] = normalized_provenance
            output_qwen.append(normalized_qwen)
    return output_audit, output_qwen


def main() -> int:
    """Validate, merge, and hash real-verifier-admitted bundles."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-bundle",
        nargs=4,
        action="append",
        metavar=("LABEL", "AUDIT", "QWEN", "MANIFEST"),
        required=True,
    )
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--output-qwen", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_audit, args.output_qwen, args.output_manifest):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite: {output}")

    bundles: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    input_records: list[dict[str, Any]] = []
    for label, audit_value, qwen_value, manifest_value in args.input_bundle:
        audit_path = Path(audit_value)
        qwen_path = Path(qwen_value)
        manifest_path = Path(manifest_value)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "xtoken-real-verifier-filtered-sft-bundle-v1":
            raise ValueError(f"{label}: unexpected manifest schema")
        if manifest.get("final_environment_verifier_used") is not True:
            raise ValueError(f"{label}: final verifier boundary is missing")
        if manifest.get("model_judgment_is_official_task_verification") is not False:
            raise ValueError(f"{label}: model-judgment boundary is invalid")
        audit_rows = read_jsonl(audit_path)
        qwen_rows = read_jsonl(qwen_path)
        selected_rows = manifest.get("selected_rows")
        if selected_rows != len(audit_rows) or selected_rows != len(qwen_rows):
            raise ValueError(f"{label}: selected row count mismatch")
        if manifest.get("output_audit_sha256") != sha256_file(audit_path):
            raise ValueError(f"{label}: audit hash mismatch")
        if manifest.get("output_qwen_sha256") != sha256_file(qwen_path):
            raise ValueError(f"{label}: Qwen hash mismatch")
        bundles.append((label, audit_rows, qwen_rows))
        input_records.append(
            {
                "label": label,
                "audit": str(audit_path),
                "audit_sha256": sha256_file(audit_path),
                "qwen": str(qwen_path),
                "qwen_sha256": sha256_file(qwen_path),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "rows": selected_rows,
            }
        )

    output_audit, output_qwen = merge_bundles(bundles)
    write_jsonl(args.output_audit, output_audit)
    write_jsonl(args.output_qwen, output_qwen)
    manifest = {
        "schema": "xtoken-merged-real-verifier-sft-bundle-v1",
        "inputs": input_records,
        "rows": len(output_qwen),
        "unique_tasks": len({row["provenance"]["task_id"] for row in output_qwen}),
        "task_ids": [row["provenance"]["task_id"] for row in output_qwen],
        "output_audit": str(args.output_audit),
        "output_audit_sha256": sha256_file(args.output_audit),
        "output_qwen": str(args.output_qwen),
        "output_qwen_sha256": sha256_file(args.output_qwen),
        "final_environment_verifier_used_for_every_row": True,
        "model_judgment_is_official_task_verification": False,
        "input_order_and_row_order_preserved": True,
    }
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(json.dumps(manifest, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

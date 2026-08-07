"""Build exact audit and admission artifacts for a double-judged SFT subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ADMISSION_RULE = (
    "both_judges_pass_confidence_gte_90_and_use_for_sft_true"
)


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a potentially large artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl_exact(path: Path) -> dict[int, tuple[str, dict[str, Any]]]:
    """Load judgment JSONL keyed by row index while retaining exact JSON text."""
    output: dict[int, tuple[str, dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.rstrip("\n")
            if not raw:
                continue
            value = json.loads(raw)
            row_index = int(value["row_index"])
            if row_index in output:
                raise ValueError(f"duplicate row_index {row_index} in {path}")
            output[row_index] = (raw, value)
    return output


def admission_failures(value: dict[str, Any] | None, label: str) -> list[str]:
    """Return conservative reasons one judge does not admit a source row."""
    if value is None:
        return [f"{label}:missing_judgment"]
    decision = value.get("decision")
    if not isinstance(decision, dict):
        return [f"{label}:no_valid_decision"]
    failures: list[str] = []
    if decision.get("verdict") != "PASS":
        failures.append(f"{label}:verdict_{decision.get('verdict', 'MISSING')}")
    confidence = decision.get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < 90:
        failures.append(f"{label}:confidence_below_90")
    if decision.get("use_for_sft") is not True:
        failures.append(f"{label}:use_for_sft_not_true")
    return failures


def load_selected_sources(
    corpus_path: Path, row_indices: set[int]
) -> dict[int, tuple[str, dict[str, Any]]]:
    """Stream the source corpus and retain only calibrated rows exactly."""
    sources: dict[int, tuple[str, dict[str, Any]]] = {}
    with corpus_path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index not in row_indices:
                continue
            raw = line.rstrip("\n")
            sources[row_index] = (raw, json.loads(raw))
            if len(sources) == len(row_indices):
                break
    missing = row_indices - sources.keys()
    if missing:
        raise ValueError(f"calibrated source rows missing: {sorted(missing)}")
    return sources


def training_view_indices(path: Path) -> list[int]:
    """Return ordered source indices linked by an existing training JSONL."""
    output: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            output.append(int(row["provenance"]["source_row_index"]))
    return output


def build_records(
    *,
    sources: dict[int, tuple[str, dict[str, Any]]],
    primary: dict[int, tuple[str, dict[str, Any]]],
    adjudicator: dict[int, tuple[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return complete admission ledger and admitted audit dataset rows."""
    ledger: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    for row_index in sorted(sources):
        source_raw, source = sources[row_index]
        primary_item = primary.get(row_index)
        adjudicator_item = adjudicator.get(row_index)
        failures = admission_failures(
            primary_item[1] if primary_item else None, "primary"
        ) + admission_failures(
            adjudicator_item[1] if adjudicator_item else None, "adjudicator"
        )
        selected = not failures
        record = {
            "schema": "glm52-double-judge-admission-record-v1",
            "source_row_index": row_index,
            "source_row_sha256": sha256_bytes(source_raw.encode("utf-8")),
            "source_raw_json": source_raw,
            "source": source,
            "primary_judgment_raw_json": primary_item[0] if primary_item else None,
            "primary_judgment": primary_item[1] if primary_item else None,
            "adjudicator_judgment_raw_json": (
                adjudicator_item[0] if adjudicator_item else None
            ),
            "adjudicator_judgment": (
                adjudicator_item[1] if adjudicator_item else None
            ),
            "admission": {
                "selected_for_sft": selected,
                "rule": ADMISSION_RULE,
                "exclusion_reasons": failures,
            },
        }
        ledger.append(record)
        if selected:
            admitted.append(record)
    return ledger, admitted


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def main() -> None:
    """Materialize complete judge provenance without changing the training view."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--adjudicator", required=True, type=Path)
    parser.add_argument("--training-view", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--admitted", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    primary = load_jsonl_exact(args.primary)
    adjudicator = load_jsonl_exact(args.adjudicator)
    calibrated_indices = set(primary) | set(adjudicator)
    sources = load_selected_sources(args.corpus, calibrated_indices)
    ledger, admitted = build_records(
        sources=sources,
        primary=primary,
        adjudicator=adjudicator,
    )
    admitted_indices = [row["source_row_index"] for row in admitted]
    linked_training_indices = training_view_indices(args.training_view)
    if admitted_indices != linked_training_indices:
        raise ValueError(
            "admitted audit rows do not exactly match ordered training view: "
            f"admitted={admitted_indices}, training={linked_training_indices}"
        )

    write_jsonl(args.ledger, ledger)
    write_jsonl(args.admitted, admitted)
    verdict_counts: Counter[str] = Counter()
    for label, judgments in (("primary", primary), ("adjudicator", adjudicator)):
        for _, value in judgments.values():
            decision = value.get("decision")
            verdict = decision.get("verdict") if isinstance(decision, dict) else "ERROR"
            verdict_counts[f"{label}:{verdict}"] += 1
    manifest = {
        "schema": "glm52-double-judge-audit-manifest-v1",
        "admission_rule": ADMISSION_RULE,
        "calibrated_rows": len(ledger),
        "admitted_rows": len(admitted),
        "excluded_rows": len(ledger) - len(admitted),
        "admitted_source_row_indices": admitted_indices,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "artifacts": {
            "source_corpus": {
                "path": str(args.corpus),
                "sha256": sha256_file(args.corpus),
            },
            "primary_judgments": {
                "path": str(args.primary),
                "sha256": sha256_file(args.primary),
            },
            "adjudicator_judgments": {
                "path": str(args.adjudicator),
                "sha256": sha256_file(args.adjudicator),
            },
            "training_view": {
                "path": str(args.training_view),
                "sha256": sha256_file(args.training_view),
            },
            "admission_ledger": {
                "path": str(args.ledger),
                "sha256": sha256_file(args.ledger),
            },
            "admitted_audit_dataset": {
                "path": str(args.admitted),
                "sha256": sha256_file(args.admitted),
            },
        },
        "model_judgment_is_official_task_verification": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize held-out-safe trajectory candidates for real-verifier replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPLAY_RULE = "deterministic_benchmark_disjoint_candidate_pending_real_verifier"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file while failing on blanks and non-object records."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line {line_number} in {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} in {path} is not an object")
            rows.append(row)
    return rows


def index_candidates(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index unique candidate rows while preserving the input as the order source."""
    indexed: dict[int, dict[str, Any]] = {}
    task_ids: set[str] = set()
    for row in rows:
        row_index = int(row["row_index"])
        task_id = str(row["task_id"])
        if row_index in indexed:
            raise ValueError(f"duplicate candidate row_index {row_index}")
        if task_id in task_ids:
            raise ValueError(f"duplicate candidate task_id {task_id!r}")
        indexed[row_index] = row
        task_ids.add(task_id)
    if not indexed:
        raise ValueError("candidate set is empty")
    return indexed


def load_sources(
    corpus: Path, row_indices: set[int]
) -> dict[int, tuple[str, dict[str, Any]]]:
    """Stream the source corpus and retain only requested rows exactly."""
    sources: dict[int, tuple[str, dict[str, Any]]] = {}
    with corpus.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index not in row_indices:
                continue
            raw = line.rstrip("\n")
            sources[row_index] = (raw, json.loads(raw))
            if len(sources) == len(row_indices):
                break
    missing = sorted(row_indices - set(sources))
    if missing:
        raise ValueError(f"candidate source rows missing: {missing[:10]}")
    return sources


def build_records(
    candidates: list[dict[str, Any]],
    sources: dict[int, tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build replay-only audit records with exact source identity checks."""
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        row_index = int(candidate["row_index"])
        raw, source = sources[row_index]
        for key in ("task_id", "rollout_id", "manifest_order"):
            if candidate.get(key) != source.get(key):
                raise ValueError(
                    f"row {row_index} {key} mismatch: "
                    f"{candidate.get(key)!r} != {source.get(key)!r}"
                )
        records.append(
            {
                "schema": "glm52-real-verifier-replay-candidate-v1",
                "source_row_index": row_index,
                "source_row_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "source_raw_json": raw,
                "source": source,
                "candidate": candidate,
                "admission": {
                    "selected_for_replay": True,
                    "selected_for_sft": False,
                    "rule": REPLAY_RULE,
                    "exclusion_reasons": [],
                },
            }
        )
    return records


def main() -> int:
    """Write a provenance-rich replay audit and manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    for output in (args.output, args.manifest):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    actual_source_sha256 = sha256_file(args.corpus)
    if actual_source_sha256 != args.source_sha256:
        raise ValueError("source corpus SHA-256 differs from the pinned digest")
    candidate_manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    candidate_sha256 = sha256_file(args.candidates)
    if candidate_manifest.get("output_sha256") != candidate_sha256:
        raise ValueError("candidate JSONL SHA-256 differs from its manifest")
    if candidate_manifest.get("real_verifier_replay_performed") is not False:
        raise ValueError("candidate manifest has an invalid real-verifier boundary")
    if candidate_manifest.get("training_eligible") is not False:
        raise ValueError("candidate manifest unexpectedly marks rows training-eligible")

    candidates = load_jsonl(args.candidates)
    indexed = index_candidates(candidates)
    sources = load_sources(args.corpus, set(indexed))
    records = build_records(candidates, sources)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema": "glm52-real-verifier-replay-candidate-manifest-v1",
        "replay_rule": REPLAY_RULE,
        "source_corpus": str(args.corpus),
        "source_corpus_sha256": actual_source_sha256,
        "candidates": str(args.candidates),
        "candidates_sha256": candidate_sha256,
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "rows": len(records),
        "unique_tasks": len({record["source"]["task_id"] for record in records}),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "model_review_performed": False,
        "real_verifier_replay_performed": False,
        "selected_for_replay": True,
        "training_eligible": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(manifest, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

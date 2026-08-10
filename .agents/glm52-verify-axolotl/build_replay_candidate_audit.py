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


def validate_candidate_delta(
    *,
    candidates: list[dict[str, Any]],
    prior_candidates: list[dict[str, Any]],
    expanded_candidates: list[dict[str, Any]],
    delta_manifest: dict[str, Any],
    expanded_manifest: dict[str, Any],
) -> None:
    """Recompute the expanded-minus-prior delta and its evaluation boundary."""
    if delta_manifest.get("schema") != "glm52-candidate-delta-v1":
        raise ValueError("candidate delta manifest has an unexpected schema")
    if (
        expanded_manifest.get("schema")
        != "glm52-diverse-concise-candidate-selection-v1"
    ):
        raise ValueError("expanded candidate manifest has an unexpected schema")
    if expanded_manifest.get("evaluation_prompts_are_excluded_not_used_for_training") is not True:
        raise ValueError("expanded manifest does not attest evaluation-prompt exclusion")
    selection = expanded_manifest.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("expanded manifest selection metadata is missing")
    threshold = selection.get("near_duplicate_jaccard_threshold")
    maximum_similarity = expanded_manifest.get("maximum_selected_evaluation_similarity")
    if not isinstance(threshold, (int, float)) or not isinstance(
        maximum_similarity, (int, float)
    ):
        raise ValueError("expanded manifest evaluation similarity metadata is invalid")
    if float(maximum_similarity) >= float(threshold):
        raise ValueError("expanded candidates cross the evaluation similarity threshold")

    index_candidates(prior_candidates)
    index_candidates(expanded_candidates)
    prior_tasks = {str(row["task_id"]) for row in prior_candidates}
    expected = [
        row for row in expanded_candidates if str(row["task_id"]) not in prior_tasks
    ]
    if candidates != expected:
        raise ValueError("candidate delta is not exactly expanded candidates minus prior tasks")


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
    parser.add_argument("--prior-candidates", type=Path, required=True)
    parser.add_argument("--expanded-candidates", type=Path, required=True)
    parser.add_argument("--expanded-manifest", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if len(args.code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.code_commit
    ):
        raise ValueError("code commit must be a full lowercase Git SHA")

    for output in (args.output, args.manifest):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    actual_source_sha256 = sha256_file(args.corpus)
    if actual_source_sha256 != args.source_sha256:
        raise ValueError("source corpus SHA-256 differs from the pinned digest")
    candidate_manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    expanded_manifest = json.loads(args.expanded_manifest.read_text(encoding="utf-8"))
    candidate_sha256 = sha256_file(args.candidates)
    if candidate_manifest.get("output_sha256") != candidate_sha256:
        raise ValueError("candidate JSONL SHA-256 differs from its manifest")
    if candidate_manifest.get("real_verifier_replay_performed") is not False:
        raise ValueError("candidate manifest has an invalid real-verifier boundary")
    if candidate_manifest.get("training_eligible") is not False:
        raise ValueError("candidate manifest unexpectedly marks rows training-eligible")
    prior_sha256 = sha256_file(args.prior_candidates)
    expanded_sha256 = sha256_file(args.expanded_candidates)
    if candidate_manifest.get("prior_candidate_set_sha256") != prior_sha256:
        raise ValueError("prior candidate SHA-256 differs from the delta manifest")
    if candidate_manifest.get("expanded_candidate_set_sha256") != expanded_sha256:
        raise ValueError("expanded candidate SHA-256 differs from the delta manifest")
    if expanded_manifest.get("output_sha256") != expanded_sha256:
        raise ValueError("expanded candidate SHA-256 differs from its manifest")

    candidates = load_jsonl(args.candidates)
    prior_candidates = load_jsonl(args.prior_candidates)
    expanded_candidates = load_jsonl(args.expanded_candidates)
    validate_candidate_delta(
        candidates=candidates,
        prior_candidates=prior_candidates,
        expanded_candidates=expanded_candidates,
        delta_manifest=candidate_manifest,
        expanded_manifest=expanded_manifest,
    )
    indexed = index_candidates(candidates)
    sources = load_sources(args.corpus, set(indexed))
    records = build_records(candidates, sources)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema": "glm52-real-verifier-replay-candidate-manifest-v1",
        "code_commit": args.code_commit,
        "replay_rule": REPLAY_RULE,
        "source_corpus": str(args.corpus),
        "source_corpus_sha256": actual_source_sha256,
        "candidates": str(args.candidates),
        "candidates_sha256": candidate_sha256,
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "prior_candidates": str(args.prior_candidates),
        "prior_candidates_sha256": prior_sha256,
        "expanded_candidates": str(args.expanded_candidates),
        "expanded_candidates_sha256": expanded_sha256,
        "expanded_manifest": str(args.expanded_manifest),
        "expanded_manifest_sha256": sha256_file(args.expanded_manifest),
        "evaluation_instruction_set_sha256": expanded_manifest[
            "evaluation_instruction_set_sha256"
        ],
        "maximum_selected_evaluation_similarity": expanded_manifest[
            "maximum_selected_evaluation_similarity"
        ],
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

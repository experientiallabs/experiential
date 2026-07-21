"""Durable, neutral evidence for one completed multi-harness score batch."""

from __future__ import annotations

from pathlib import Path

from wmh.core.types import JsonObject
from wmh.harness.archive_io import (
    copy_score_artifacts,
    publish_json_manifest,
    relative_path,
    write_source_tree,
    write_text,
)
from wmh.harness.score_batch import HarnessScoreBatch

_SCHEMA_VERSION = 1


def write_score_archive(
    destination: str | Path,
    batch: HarnessScoreBatch,
) -> Path:
    """Write sources, reports, and verified artifacts without selecting a winner.

    The destination must not exist. ``manifest.json`` is written last and is the completion
    marker. An interrupted copy remains inspectable but cannot be mistaken for a complete batch.
    """
    root = Path(destination)
    if root.exists():
        raise FileExistsError(f"score archive destination already exists: {root}")
    root.mkdir(parents=True)

    entries: list[JsonObject] = []
    for index, evaluated in enumerate(batch.entries):
        entry_dir = root / "entries" / f"{index:04d}"
        source_dir = entry_dir / "source"
        write_source_tree(source_dir, evaluated.target.source)
        document_path = entry_dir / "document.json"
        write_text(document_path, evaluated.target.harness.model_dump_json(indent=2))
        report_path = entry_dir / "score.json"
        write_text(report_path, evaluated.score.report.model_dump_json(indent=2))
        artifacts_dir = entry_dir / "artifacts"
        copy_score_artifacts(artifacts_dir, evaluated.score)
        entries.append(
            {
                "index": index,
                "label": evaluated.target.label,
                "name": evaluated.target.harness.name,
                "version": evaluated.target.harness.version,
                "document_hash": evaluated.target.harness.doc_hash,
                "source_tree_hash": evaluated.target.source.tree_hash,
                "source_path": relative_path(root, source_dir),
                "document_path": relative_path(root, document_path),
                "score_path": relative_path(root, report_path),
                "artifacts_path": relative_path(root, artifacts_dir),
                "score": evaluated.score.report.score,
                "pass_rate": evaluated.score.report.pass_rate,
                "report_hash": evaluated.score.report.report_hash,
            }
        )

    manifest_path = root / "manifest.json"
    manifest: JsonObject = {
        "schema_version": _SCHEMA_VERSION,
        "score_request": batch.request.model_dump(mode="json"),
        "entries": entries,
    }
    return publish_json_manifest(manifest_path, manifest)

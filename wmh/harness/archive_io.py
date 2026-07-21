"""Shared integrity-preserving I/O for local harness evidence archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import JsonValue

from wmh.harness.scoring import HarnessScore
from wmh.harness.source_tree import HarnessSourceTree


def write_source_tree(destination: Path, source: HarnessSourceTree) -> None:
    """Materialize one portable source tree, including an empty root."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.files:
        target = destination / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.content, encoding="utf-8")


def copy_score_artifacts(destination: Path, score: HarnessScore) -> None:
    """Copy every score artifact after verifying its declared size and digest."""
    destination.mkdir(parents=True, exist_ok=True)
    for artifact in score.report.artifacts:
        content = score.artifacts.read_bytes(artifact.path)
        if len(content) != artifact.size_bytes:
            raise ValueError(f"artifact {artifact.path!r} size differs from its score manifest")
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        if digest != artifact.content_hash:
            raise ValueError(f"artifact {artifact.path!r} content differs from its score manifest")
        target = destination / artifact.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def write_text(path: Path, content: str) -> None:
    """Write UTF-8 text after creating its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def relative_path(root: Path, path: Path) -> str:
    """Return one canonical archive-relative POSIX path."""
    return path.relative_to(root).as_posix()


def publish_json_manifest(path: Path, value: JsonValue) -> Path:
    """Publish deterministic JSON atomically as an archive completion marker."""
    temporary = path.with_name(f"{path.name}.tmp")
    write_text(temporary, json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path

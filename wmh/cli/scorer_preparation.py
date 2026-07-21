"""Durable local lifecycle helpers for scorer preparation artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

SCORER_PREPARATION_FILENAME = "scorer-preparation.bin"


def preparation_staging_path(run_dir: Path) -> Path:
    """Return the deterministic audit-only sibling used before a run is claimed."""
    if not run_dir.name or run_dir.name in {".", ".."}:
        raise ValueError("run directory must have a canonical name")
    return run_dir.parent / f".{run_dir.name}.{SCORER_PREPARATION_FILENAME}"


def publish_preparation_artifact(run_dir: Path, content: bytes) -> Path:
    """Durably publish exact scorer preparation bytes into a newly claimed run."""
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError("scorer preparation run directory is not a regular directory")
    destination = run_dir / SCORER_PREPARATION_FILENAME
    if os.path.lexists(destination):
        raise RuntimeError("scorer preparation artifact already exists")
    temporary = run_dir / f".{SCORER_PREPARATION_FILENAME}.tmp-{uuid4().hex}"
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    _sync_directory(run_dir)
    return destination


def remove_preparation_staging(path: Path) -> None:
    """Remove a successfully checkpointed staging artifact and sync its parent."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("scorer preparation staging is not a regular file")
    path.unlink()
    _sync_directory(path.parent)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

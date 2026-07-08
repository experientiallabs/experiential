"""Deterministic model-bundle packing and safe unpacking for push/pull.

A pushed bundle is byte-compatible with the bundles the platform's own build
pipeline produces: a gzipped tarball of the model directory with
archive-relative member paths. Packing is an include-list — the model's
`config.toml`, `metrics.json`, `card.json`, `prompts/`, and `index/` — so
local `runs/` cost records and raw `traces/` (customer data) never leave the
machine.

Bundles can reach the platform's 1GB cap, so packing and unpacking are
file-based: bytes stream between disk and the network without ever being held
in memory whole.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tomllib
import uuid
from pathlib import Path

from pydantic import BaseModel

from wmh.config.config import HarnessConfig
from wmh.core.types import JsonValue

_INCLUDED_FILES = ("config.toml", "metrics.json", "card.json")
_INCLUDED_DIRS = ("prompts", "index")

_HASH_CHUNK_BYTES = 1024 * 1024


class PackedModelBundle(BaseModel):
    """A packed model bundle on disk, ready for upload."""

    path: Path
    sha256: str
    byte_size: int


class BundleFormatError(ValueError):
    """The directory or bytes are not a valid world-model bundle."""


def sha256_file(path: Path) -> str:
    """Digest a file's contents without loading it whole."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def pack_model_dir(directory: Path, dest: Path) -> PackedModelBundle:
    """Pack a model directory into the platform's bundle format at ``dest``.

    Args:
        directory: A built model directory (must contain `config.toml`).
        dest: Where to write the gzipped tarball (parent must exist).

    Returns:
        The bundle file plus integrity metadata; member order is sorted so
        identical inputs produce identical archives.

    Raises:
        BundleFormatError: If the directory is missing or has no config.toml.
    """
    if not directory.is_dir():
        msg = f"model directory does not exist: {directory}"
        raise BundleFormatError(msg)
    if not (directory / "config.toml").is_file():
        msg = f"{directory} has no config.toml; only built world models can be pushed"
        raise BundleFormatError(msg)

    members: list[Path] = []
    for name in _INCLUDED_FILES:
        path = directory / name
        if path.is_file():
            members.append(path)
    for name in _INCLUDED_DIRS:
        root = directory / name
        if root.is_dir():
            members.extend(sorted(path for path in root.rglob("*")))
            members.append(root)

    with tarfile.open(dest, mode="w:gz") as tar:
        for path in sorted(set(members)):
            tar.add(path, arcname=str(path.relative_to(directory)), recursive=False)
    return PackedModelBundle(
        path=dest,
        sha256=sha256_file(dest),
        byte_size=dest.stat().st_size,
    )


def unpack_model_bundle(source: Path, dest_dir: Path, *, force: bool = False) -> None:
    """Unpack a pulled bundle file into a local model directory.

    Extraction happens in a temporary sibling renamed into place, so a crashed
    unpack never leaves a half-written model that later loads as real.

    Args:
        source: Downloaded gzipped tarball.
        dest_dir: Target model directory (`.wmh/models/<name>`).
        force: Replace an existing directory instead of refusing.

    Raises:
        BundleFormatError: If the file is not a readable bundle or a member
            would escape the destination.
        FileExistsError: If ``dest_dir`` exists and ``force`` is not set.
    """
    if dest_dir.exists() and not force:
        msg = f"{dest_dir} already exists; pass --force to replace it"
        raise FileExistsError(msg)
    staging_dir = dest_dir.with_name(f"{dest_dir.name}.pull-{uuid.uuid4().hex}")
    staging_dir.mkdir(parents=True)
    try:
        with tarfile.open(source, mode="r:gz") as tar:
            # The "data" filter rejects absolute paths, traversal, and special
            # members instead of writing them.
            tar.extractall(staging_dir, filter="data")
    except (tarfile.TarError, OSError) as error:
        shutil.rmtree(staging_dir, ignore_errors=True)
        msg = f"bundle could not be unpacked: {error}"
        raise BundleFormatError(msg) from error
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    staging_dir.rename(dest_dir)


def extract_push_meta(directory: Path) -> dict[str, JsonValue]:
    """Derive the push metadata the platform stores alongside a bundle.

    Parses the model's own `config.toml` (and `metrics.json` when present)
    through wmh's typed config so the platform never reads files out of the
    tarball.
    """
    config = HarnessConfig.model_validate(
        tomllib.loads((directory / "config.toml").read_text(encoding="utf-8"))
    )
    meta: dict[str, JsonValue] = {
        "serve_provider": config.serve_provider.value,
        "embed_provider": config.embed_provider.value,
        "embed_dim": config.embed_dim,
        "gepa_budget": config.gepa_budget,
    }
    try:
        meta["serve_model"] = config.serve_provider_config().model
    except ValueError:
        # No provider block for the serve provider; the platform column stays unset.
        pass
    metrics_path = directory / "metrics.json"
    if metrics_path.is_file():
        meta["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return meta

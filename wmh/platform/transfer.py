"""Deterministic model-bundle packing and safe unpacking for push/pull.

A pushed bundle is byte-compatible with the bundles the platform's own build
pipeline produces: a gzipped tarball of the model directory with
archive-relative member paths. Packing is an include-list — the model's
`config.toml`, `metrics.json`, `card.json`, `prompts/`, and `index/` — so
local `runs/` cost records and raw `traces/` (customer data) never leave the
machine.
"""

from __future__ import annotations

import hashlib
import io
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


class PackedModelBundle(BaseModel):
    """A packed model directory ready for upload."""

    content: bytes
    sha256: str
    byte_size: int


class BundleFormatError(ValueError):
    """The directory or bytes are not a valid world-model bundle."""


def pack_model_dir(directory: Path) -> PackedModelBundle:
    """Pack a model directory into the platform's bundle format.

    Args:
        directory: A built model directory (must contain `config.toml`).

    Returns:
        Bundle bytes plus integrity metadata; member order is sorted so
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

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(set(members)):
            tar.add(path, arcname=str(path.relative_to(directory)), recursive=False)
    content = buffer.getvalue()
    return PackedModelBundle(
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        byte_size=len(content),
    )


def unpack_model_bundle(content: bytes, dest_dir: Path, *, force: bool = False) -> None:
    """Unpack pulled bundle bytes into a local model directory.

    Extraction happens in a temporary sibling renamed into place, so a crashed
    unpack never leaves a half-written model that later loads as real.

    Args:
        content: gzipped tarball bytes from the platform.
        dest_dir: Target model directory (`.wmh/models/<name>`).
        force: Replace an existing directory instead of refusing.

    Raises:
        BundleFormatError: If the bytes are not a readable bundle or a member
            would escape the destination.
        FileExistsError: If ``dest_dir`` exists and ``force`` is not set.
    """
    if dest_dir.exists():
        if not force:
            msg = f"{dest_dir} already exists; pass --force to replace it"
            raise FileExistsError(msg)
    staging_dir = dest_dir.with_name(f"{dest_dir.name}.pull-{uuid.uuid4().hex}")
    staging_dir.mkdir(parents=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            # The "data" filter rejects absolute paths, traversal, and special
            # members instead of writing them.
            tar.extractall(staging_dir, filter="data")
    except (tarfile.TarError, OSError) as error:
        shutil.rmtree(staging_dir, ignore_errors=True)
        msg = f"bundle bytes could not be unpacked: {error}"
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

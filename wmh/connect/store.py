"""Context bundle persistence and rendering under `<project>/.wmh/context/`.

A bundle is one pull's replayable artifact: `manifest.json` (what was pulled, when, from where)
plus `items.jsonl` (one normalized `ContextItem` per line). "Filesystem as DB", like the model
store: loading a bundle is just reading its folder. `ContextStore.save`/`load` persist and read
bundles, and `render_markdown` turns a bundle into a deterministic markdown document callers can
write into a model's knowledge dir.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel

from wmh.config.config import ARTIFACT_DIR
from wmh.config.store import validate_name
from wmh.connect.types import ContextItem, PullQuery

_MANIFEST_FILENAME = "manifest.json"
_ITEMS_FILENAME = "items.jsonl"


class BundleManifest(BaseModel):
    """Provenance for one saved bundle: what was pulled, when, by which connector.

    Attributes:
        name: Bundle name (the directory name under `.wmh/context/`).
        connector: The connector that produced the bundle.
        query: The exact `PullQuery` used, kept for replayable re-pulls.
        pulled_at: ISO-8601 timestamp of the pull.
        item_count: Number of items in `items.jsonl`.
        account: Human-readable identity the pull ran as, when known.
    """

    name: str
    connector: str
    query: PullQuery
    pulled_at: str
    item_count: int
    account: str | None = None


class ContextStore:
    """Named context bundles on disk under `<root>/.wmh/context/<name>/`.

    `root` is the PROJECT directory (the parent of `.wmh/`), defaulting to the current working
    directory; tests pass a tmp path. Note this differs from `WorldModelStore`, whose root is
    the `.wmh` artifact dir itself.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path.cwd()
        self.context_dir = self.root / ARTIFACT_DIR / "context"

    def bundle_dir(self, name: str) -> Path:
        """The directory a bundle named `name` lives in (may not exist)."""
        return self.context_dir / _validated_bundle_name(name)

    def save(
        self, manifest: BundleManifest, items: list[ContextItem], *, overwrite: bool = False
    ) -> Path:
        """Write one bundle (`manifest.json` + `items.jsonl`); returns its directory.

        Raises:
            FileExistsError: When the bundle already exists and `overwrite` is False.
            ValueError: When the bundle name is not a safe single path segment.
        """
        directory = self.bundle_dir(manifest.name)
        if directory.exists():
            if not overwrite:
                raise FileExistsError(
                    f"context bundle {manifest.name!r} already exists at {directory}; "
                    "pass overwrite=True to replace it"
                )
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        manifest_text = manifest.model_dump_json(indent=2) + "\n"
        (directory / _MANIFEST_FILENAME).write_text(manifest_text, encoding="utf-8")
        lines = "".join(item.model_dump_json() + "\n" for item in items)
        (directory / _ITEMS_FILENAME).write_text(lines, encoding="utf-8")
        return directory

    def load(self, name: str) -> tuple[BundleManifest, list[ContextItem]]:
        """Read one bundle back as (manifest, items).

        Raises:
            FileNotFoundError: When no bundle named `name` exists.
        """
        directory = self.bundle_dir(name)
        manifest_path = directory / _MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no context bundle named {name!r} under {self.context_dir}; "
                "pull a bundle and persist it with ContextStore.save first"
            )
        manifest = BundleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        items_text = (directory / _ITEMS_FILENAME).read_text(encoding="utf-8")
        items = [
            ContextItem.model_validate_json(line)
            for line in items_text.splitlines()
            if line.strip()
        ]
        return manifest, items

    def list_bundles(self) -> list[BundleManifest]:
        """Manifests of every saved bundle, sorted by directory name."""
        if not self.context_dir.exists():
            return []
        manifests: list[BundleManifest] = []
        for child in sorted(self.context_dir.iterdir()):
            manifest_path = child / _MANIFEST_FILENAME
            if child.is_dir() and manifest_path.exists():
                manifests.append(
                    BundleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
                )
        return manifests

    def delete(self, name: str) -> bool:
        """Remove one bundle directory; returns whether it existed."""
        directory = self.bundle_dir(name)
        if not directory.exists():
            return False
        shutil.rmtree(directory)
        return True


def render_markdown(
    manifest: BundleManifest, items: list[ContextItem], *, max_chars: int | None = None
) -> str:
    """Render a bundle as one deterministic markdown document.

    A provenance header (connector, pulled_at, query) is followed by one `## title` section per
    item (a kind/date/url fact line, then the body). When the result would exceed `max_chars`,
    whole items are dropped from the tail and a final "... n items omitted" line makes the
    truncation visible, never silent.
    """
    header = _render_header(manifest)
    sections = [_render_item(item) for item in items]
    full = "\n\n".join([header, *sections]) + "\n"
    if max_chars is None or len(full) <= max_chars:
        return full
    candidate = full
    for kept in range(len(items) - 1, -1, -1):
        omitted = len(items) - kept
        tail = f"... {omitted} items omitted"
        candidate = "\n\n".join([header, *sections[:kept], tail]) + "\n"
        if len(candidate) <= max_chars:
            return candidate
    # Even the header alone is over budget; return the loud minimal form anyway.
    return candidate


def _validated_bundle_name(name: str) -> str:
    """Reject unsafe bundle names with bundle-specific wording (same rules as model names)."""
    try:
        return validate_name(name)
    except ValueError as exc:
        raise ValueError(
            f"invalid context bundle name {name!r}: use letters, digits, '.', '_', '-' "
            "(must start with a letter or digit, no path separators)"
        ) from exc


def _render_header(manifest: BundleManifest) -> str:
    """The provenance block: bundle name, connector, pull time, identity, query, item count."""
    query_parts = [
        f"{field}={value}"
        for field, value in manifest.query.model_dump(mode="json").items()
        if value is not None
    ]
    lines = [
        f"# Context bundle: {manifest.name}",
        "",
        f"- connector: {manifest.connector}",
        f"- pulled_at: {manifest.pulled_at}",
    ]
    if manifest.account:
        lines.append(f"- account: {manifest.account}")
    lines.append(f"- query: {', '.join(query_parts)}")
    lines.append(f"- items: {manifest.item_count}")
    return "\n".join(lines)


def _render_item(item: ContextItem) -> str:
    """One `## title` section: a kind/date/url fact line, then the body."""
    facts = [item.kind.value]
    if item.created_at:
        facts.append(f"created {item.created_at}")
    if item.updated_at:
        facts.append(f"updated {item.updated_at}")
    if item.url:
        facts.append(item.url)
    section = f"## {item.title}\n\n{' | '.join(facts)}"
    body = item.body.strip()
    if body:
        section += f"\n\n{body}"
    return section

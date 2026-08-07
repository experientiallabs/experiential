"""Harnesses on disk: immutable numbered versions plus movable aliases, per name.

Like world models under `.wmo/models/<name>/`, a harness is a named artifact — but harnesses
accumulate *versions*, because they are the thing `wmo optimize harness` iterates on. A
version, once
written, never changes; deployment state lives in movable aliases; and every eval result keys to an
immutable version rather than to "whatever the harness currently is".

    .wmo/harnesses/<name>/
      aliases.toml        # [aliases]  champion = 3   (movable pointers; rollback = re-point)
      v1/
        doc.json          # the authoritative HarnessDoc serialization
        SYSTEM.md         # rendered export of the same document, for running the harness
        config.toml       #   outside wmo — regenerated on every save, never read back
        skills/<slug>.md  #   when doc.json is present
      v3/ ...

`doc.json` is authoritative; the rendered files are an export. A directory with rendered files but
no `doc.json` (a hand-authored harness) still loads: the files parse into a single-prompt document.
Rendered-only loads are strict: unknown config.toml tables or fields, non-.md files under skills/,
and a skill filename that does not match its frontmatter name are errors with actionable messages,
not silently ignored. Dotfiles (.DS_Store and friends) are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from wmo.common.config.store import validate_name
from wmo.common.config.versioned_store import (
    ALIASES_FILE,
    CHAMPION_ALIAS,
    VersionedArtifactStore,
)
from wmo.runtime.harness.doc import HarnessDoc
from wmo.runtime.harness.source_tree import SYSTEM_FILE, HarnessSourceFile, HarnessSourceTree

HARNESSES_DIR = "harnesses"

__all__ = ["CHAMPION_ALIAS", "HARNESSES_DIR", "HarnessStore"]

_DOC_FILE = "doc.json"


class HarnessStore(VersionedArtifactStore):
    """Named, versioned harnesses under `<root>/harnesses/<name>/`."""

    subdirectory = HARNESSES_DIR
    kind = "harness"
    promotion_command = "`wmo optimize harness`"
    alias_repair_extra = " (`wmo harness list` shows the versions this name has)"

    @property
    def harnesses_dir(self) -> Path:
        """The directory holding every named harness."""
        return self.artifacts_dir

    # -- load / save ---------------------------------------------------------------------------

    def load(self, name: str, ref: str | None = None) -> HarnessDoc:
        version = self.resolve_version(name, ref)
        directory = self.dir_for(name) / f"v{version}"
        doc_path = directory / _DOC_FILE
        if doc_path.exists():
            doc = HarnessDoc.model_validate_json(doc_path.read_text(encoding="utf-8"))
        else:
            doc = _parse_rendered(name, directory)
        return doc.model_copy(update={"name": name, "version": version})

    def save_version(self, doc: HarnessDoc, *, alias: str | None = None) -> HarnessDoc:
        """Write `doc` as the next version of its name; optionally point `alias` at it.

        Versions are append-only: this never touches an existing version directory.
        """
        validate_name(doc.name, kind="harness")
        version = (self.versions(doc.name)[-1] + 1) if self.exists(doc.name) else 1
        stamped = doc.model_copy(update={"version": version})
        directory = self.dir_for(doc.name) / f"v{version}"
        directory.mkdir(parents=True, exist_ok=False)  # append-only: collision is a bug
        (directory / _DOC_FILE).write_text(stamped.model_dump_json(indent=2), encoding="utf-8")
        for rel_path, content in _render(stamped).items():
            target = directory / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        if alias is not None:
            self.set_alias(doc.name, alias, version)
        return stamped


def _render(doc: HarnessDoc) -> dict[str, str]:
    """Render the document to its file export (relative path -> content)."""
    return HarnessSourceTree.from_doc(doc).file_map()


def _parse_rendered(name: str, directory: Path) -> HarnessDoc:
    """Parse a rendered/hand-authored directory (no doc.json) into a document.

    The whole `SYSTEM.md` becomes one `prompt:core` surface — section boundaries are not
    recoverable from a rendered prompt, which is exactly why `doc.json` is the authoritative form.
    """
    files: list[HarnessSourceFile] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(directory)
        if any(part.startswith(".") for part in rel_path.parts):
            # Finder and editors drop metadata like .DS_Store (often with NUL bytes) into
            # hand-authored dirs. A dotfile can never be a harness surface (its code_surface_id
            # is not a valid slug), so skip it instead of failing the load on its content.
            continue
        rel = rel_path.as_posix()
        if rel in {_DOC_FILE, ALIASES_FILE}:
            continue
        files.append(HarnessSourceFile(path=rel, content=path.read_text(encoding="utf-8")))
    if not files:
        raise ValueError(f"harness dir {directory} has neither {_DOC_FILE} nor {SYSTEM_FILE}")
    return HarnessSourceTree(files=tuple(files)).to_doc(name)


def write_json_atomic(path: Path, payload: object) -> None:
    """Write `payload` as JSON via a same-directory temp file and an atomic replace.

    A torn artifact must never be loadable; the temp name is unique per call so two
    concurrent writers cannot publish each other's half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        staging.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        staging.replace(path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise

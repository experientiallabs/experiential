"""Harnesses on disk: immutable numbered versions plus movable aliases, per name.

Like world models under `.wmh/models/<name>/`, a harness is a named artifact — but harnesses
accumulate *versions*, because they are the thing `wmh optimize` iterates on. A version, once
written, never changes; deployment state lives in movable aliases; and every eval result keys to an
immutable version rather than to "whatever the harness currently is".

    .wmh/harnesses/<name>/
      aliases.toml        # [aliases]  champion = 3   (movable pointers; rollback = re-point)
      v1/
        doc.json          # the authoritative HarnessDoc serialization
        SYSTEM.md         # rendered export of the same document, for running the harness
        config.toml       #   outside wmh — regenerated on every save, never read back
        skills/<slug>.md  #   when doc.json is present
      v3/ ...

`doc.json` is authoritative; the rendered files are an export. A directory with rendered files but
no `doc.json` (a hand-authored harness) still loads: the files parse into a single-prompt document.
Rendered-only loads are strict: unknown config.toml tables or fields, non-.md files under skills/,
and a skill filename that does not match its frontmatter name are errors with actionable messages,
not silently ignored. Dotfiles (.DS_Store and friends) are skipped.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w

from wmh.config.store import validate_name
from wmh.harness.doc import HarnessDoc
from wmh.harness.source_tree import SYSTEM_FILE, HarnessSourceFile, HarnessSourceTree

HARNESSES_DIR = "harnesses"
CHAMPION_ALIAS = "champion"

_DOC_FILE = "doc.json"
_ALIASES_FILE = "aliases.toml"


class HarnessStore:
    """Named, versioned harnesses under `<root>/harnesses/<name>/`."""

    def __init__(self, root: str | Path = ".wmh") -> None:
        self.root = Path(root)

    @property
    def harnesses_dir(self) -> Path:
        return self.root / HARNESSES_DIR

    def dir_for(self, name: str) -> Path:
        return self.harnesses_dir / validate_name(name)

    # -- enumeration -------------------------------------------------------------------------

    def list_names(self) -> list[str]:
        if not self.harnesses_dir.exists():
            return []
        return sorted(
            d.name for d in self.harnesses_dir.iterdir() if d.is_dir() and self.versions(d.name)
        )

    def versions(self, name: str) -> list[int]:
        directory = self.dir_for(name)
        if not directory.exists():
            return []
        found: list[int] = []
        for child in directory.iterdir():
            if child.is_dir() and child.name.startswith("v") and child.name[1:].isdigit():
                found.append(int(child.name[1:]))
        return sorted(found)

    def exists(self, name: str) -> bool:
        return bool(self.versions(name))

    # -- aliases -----------------------------------------------------------------------------

    def aliases(self, name: str) -> dict[str, int]:
        path = self.dir_for(name) / _ALIASES_FILE
        if not path.exists():
            return {}
        data = tomllib.loads(path.read_text(encoding="utf-8")).get("aliases", {})
        return {k: v for k, v in data.items() if isinstance(v, int)}

    def set_alias(self, name: str, alias: str, version: int) -> None:
        """Point `alias` at `version` (moving it if it exists). Rollback is re-pointing."""
        if version not in self.versions(name):
            raise ValueError(f"harness {name!r} has no version v{version}")
        current = self.aliases(name)
        current[alias] = version
        path = self.dir_for(name) / _ALIASES_FILE
        path.write_text(tomli_w.dumps({"aliases": current}), encoding="utf-8")

    # -- load / save ---------------------------------------------------------------------------

    def resolve_version(self, name: str, ref: str | None = None) -> int:
        """Resolve a version ref: `None` -> champion alias, else latest; `"vN"`/`"N"`; an alias."""
        available = self.versions(name)
        if not available:
            raise FileNotFoundError(
                f"no harness named {name!r} under {self.harnesses_dir} "
                f"(have: {', '.join(self.list_names()) or 'none'})"
            )
        aliases = self.aliases(name)
        if ref is None:
            return aliases.get(CHAMPION_ALIAS, available[-1])
        normalized = ref.removeprefix("v")
        if normalized.isdigit():
            version = int(normalized)
            if version not in available:
                raise ValueError(f"harness {name!r} has no version v{version}")
            return version
        if ref in aliases:
            return aliases[ref]
        raise ValueError(f"harness {name!r} has no version or alias {ref!r}")

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
        validate_name(doc.name)
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
        if rel in {_DOC_FILE, _ALIASES_FILE}:
            continue
        files.append(HarnessSourceFile(path=rel, content=path.read_text(encoding="utf-8")))
    if not files:
        raise ValueError(f"harness dir {directory} has neither {_DOC_FILE} nor {SYSTEM_FILE}")
    return HarnessSourceTree(files=tuple(files)).to_doc(name)

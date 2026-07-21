"""Harnesses on disk: immutable numbered versions plus movable aliases, per name.

Like world models under `.wmh/models/<name>/`, a harness is a named artifact — but harnesses
accumulate *versions*, because they are the thing `wmh harness create` iterates on. A version, once
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
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import tomli_w
from filelock import FileLock

from wmh.config.store import validate_name
from wmh.harness.doc import HarnessDoc
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree

HARNESSES_DIR = "harnesses"
CHAMPION_ALIAS = "champion"

_DOC_FILE = "doc.json"
_ALIASES_FILE = "aliases.toml"
_LOCK_FILE = ".store.lock"
_RESERVATIONS_DIR = ".reservations"
_PUBLICATION_ID_PATTERN = r"^sha256:[0-9a-f]{64}$"


@dataclass(frozen=True)
class HarnessVersionReservation:
    """Exact version slot and alias snapshot owned by one retryable publication."""

    version: int
    publication_id: str
    aliases: tuple[tuple[str, int], ...]

    def alias_version(self, alias: str) -> int | None:
        """Return the alias target captured when the version was reserved."""
        return dict(self.aliases).get(alias)


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
        with self._locked(name):
            self._set_alias_unlocked(name, alias, version)

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
        with self._locked(doc.name):
            version = self._next_available_version_unlocked(doc.name)
            stamped = self._write_version_unlocked(doc, version=version)
            if alias is not None:
                self._set_alias_unlocked(doc.name, alias, version)
            return stamped

    def reserve_version(self, name: str, *, publication_id: str) -> int:
        """Reserve one exact future version for a retryable local publication."""
        _validate_publication_id(publication_id)
        with self._locked(name):
            reservations = self._reservations_unlocked(name)
            matches = [
                item.version
                for item in reservations.values()
                if item.publication_id == publication_id
            ]
            if len(matches) > 1:
                raise ValueError("publication owns multiple harness version reservations")
            if matches:
                return matches[0]
            version = self._next_available_version_unlocked(name)
            path = self._reservation_path(name, version)
            _write_text_atomic(
                path,
                json.dumps(
                    {
                        "aliases": self.aliases(name),
                        "publication_id": publication_id,
                        "version": version,
                    },
                    sort_keys=True,
                )
                + "\n",
            )
            return version

    def assert_version_reservation(
        self,
        name: str,
        *,
        version: int,
        publication_id: str,
    ) -> HarnessVersionReservation:
        """Return an exact durable reservation or fail on missing or changed ownership."""
        _validate_publication_id(publication_id)
        with self._locked(name):
            return self._assert_reservation_unlocked(name, version, publication_id)

    def save_reserved_version(
        self,
        doc: HarnessDoc,
        *,
        version: int,
        publication_id: str,
    ) -> HarnessDoc:
        """Idempotently write the exact immutable version owned by a reservation."""
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("reserved harness version must be a positive integer")
        _validate_publication_id(publication_id)
        validate_name(doc.name)
        with self._locked(doc.name):
            self._assert_reservation_unlocked(doc.name, version, publication_id)
            expected = doc.model_copy(update={"version": version})
            directory = self.dir_for(doc.name) / f"v{version}"
            if directory.exists():
                if not _version_directory_matches(directory, expected):
                    raise ValueError("reserved harness version differs from publication document")
                return expected
            return self._write_version_unlocked(doc, version=version)

    def commit_alias_from_reservation(
        self,
        name: str,
        alias: str,
        *,
        version: int,
        publication_id: str,
        commit: Callable[[], None],
    ) -> None:
        """Compare and move an alias, then commit terminal state under the same lock."""
        _validate_publication_id(publication_id)
        with self._locked(name):
            reservation = self._assert_reservation_unlocked(name, version, publication_id)
            prior = reservation.alias_version(alias)
            current = self.aliases(name).get(alias)
            if current not in (prior, version):
                raise ValueError(f"harness alias {alias!r} changed after version reservation")
            if current != version:
                self._set_alias_unlocked(name, alias, version)
            commit()

    @contextmanager
    def _locked(self, name: str) -> Iterator[Path]:
        directory = self.dir_for(name)
        directory.mkdir(parents=True, exist_ok=True)
        _sync_directory(directory.parent)
        with FileLock(directory / _LOCK_FILE):
            yield directory

    def _next_available_version_unlocked(self, name: str) -> int:
        occupied = {*self.versions(name), *self._reservations_unlocked(name)}
        return max(occupied, default=0) + 1

    def _reservations_unlocked(self, name: str) -> dict[int, HarnessVersionReservation]:
        directory = self.dir_for(name) / _RESERVATIONS_DIR
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError(f"invalid harness reservation directory: {directory}")
        if not directory.is_dir():
            return {}
        reservations: dict[int, HarnessVersionReservation] = {}
        for path in sorted(directory.glob("v*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"invalid harness version reservation: {path}")
            suffix = path.stem.removeprefix("v")
            if not suffix.isdigit():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                publication_id = raw["publication_id"]
                version = raw["version"]
                aliases = raw["aliases"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid harness version reservation: {path}") from error
            if (
                not isinstance(publication_id, str)
                or isinstance(version, bool)
                or not isinstance(version, int)
                or version < 1
                or version != int(suffix)
                or not isinstance(aliases, dict)
            ):
                raise ValueError(f"invalid harness version reservation: {path}")
            alias_entries: list[tuple[str, int]] = []
            for alias, target in aliases.items():
                if (
                    not isinstance(alias, str)
                    or isinstance(target, bool)
                    or not isinstance(target, int)
                    or target < 1
                ):
                    raise ValueError(f"invalid harness version reservation: {path}")
                alias_entries.append((alias, target))
            _validate_publication_id(publication_id)
            reservations[version] = HarnessVersionReservation(
                version=version,
                publication_id=publication_id,
                aliases=tuple(sorted(alias_entries)),
            )
        return reservations

    def _assert_reservation_unlocked(
        self,
        name: str,
        version: int,
        publication_id: str,
    ) -> HarnessVersionReservation:
        reservation = self._reservations_unlocked(name).get(version)
        if reservation is None or reservation.publication_id != publication_id:
            raise ValueError(f"harness {name!r} version v{version} is not reserved by publication")
        return reservation

    def _reservation_path(self, name: str, version: int) -> Path:
        return self.dir_for(name) / _RESERVATIONS_DIR / f"v{version}.json"

    def _write_version_unlocked(self, doc: HarnessDoc, *, version: int) -> HarnessDoc:
        stamped = doc.model_copy(update={"version": version})
        harness_dir = self.dir_for(doc.name)
        directory = harness_dir / f"v{version}"
        if directory.exists():
            raise FileExistsError(f"harness {doc.name!r} version v{version} already exists")
        temporary = harness_dir / f".v{version}.tmp-{uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        for rel_path, content in _version_files(stamped).items():
            target = temporary / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_text_durable(target, content)
        _sync_tree(temporary)
        temporary.replace(directory)
        _sync_directory(harness_dir)
        return stamped

    def _set_alias_unlocked(self, name: str, alias: str, version: int) -> None:
        if version not in self.versions(name):
            raise ValueError(f"harness {name!r} has no version v{version}")
        current = self.aliases(name)
        current[alias] = version
        path = self.dir_for(name) / _ALIASES_FILE
        _write_text_atomic(path, tomli_w.dumps({"aliases": current}))


def _validate_publication_id(publication_id: str) -> None:
    if not re.fullmatch(_PUBLICATION_ID_PATTERN, publication_id):
        raise ValueError("publication_id must be a sha256 content identity")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid4().hex}")
    _write_text_durable(temporary, content)
    temporary.replace(path)
    _sync_directory(path.parent)
    if path.parent.parent != path.parent:
        _sync_directory(path.parent.parent)


def _write_text_durable(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _sync_tree(directory: Path) -> None:
    directories = [path for path in directory.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _sync_directory(path)
    _sync_directory(directory)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _version_files(doc: HarnessDoc) -> dict[str, str]:
    return {_DOC_FILE: doc.model_dump_json(indent=2), **_render(doc)}


def _version_directory_matches(directory: Path, expected: HarnessDoc) -> bool:
    if directory.is_symlink() or not directory.is_dir():
        return False
    expected_files = _version_files(expected)
    actual_files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            return False
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            try:
                actual_files[relative] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return False
    return actual_files == expected_files


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
        rel = path.relative_to(directory).as_posix()
        if rel in {_DOC_FILE, _ALIASES_FILE}:
            continue
        files.append(HarnessSourceFile(path=rel, content=path.read_text(encoding="utf-8")))
    if not files:
        raise ValueError(f"harness dir {directory} has neither {_DOC_FILE} nor SYSTEM.md")
    return HarnessSourceTree(files=tuple(files)).to_doc(name)

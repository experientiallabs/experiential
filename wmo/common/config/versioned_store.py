"""Named artifacts on disk as append-only numbered versions plus a movable alias table.

Two product surfaces need the same durability contract for different artifacts: harnesses
(`wmo/runtime/harness/store.py`) and distilled adapters (`wmo/optimize/model/store.py`). Both lay
out `<root>/<subdirectory>/<name>/vN/` version directories that are never rewritten once written,
plus an `aliases.toml` naming the version each deployment pointer resolves to, so promotion and
rollback are a pointer move rather than an edit to an artifact an eval result already keys to.

    <root>/<subdirectory>/<name>/
      aliases.toml        # [aliases]  champion = 3   (movable pointers; rollback = re-point)
      v1/ ... v3/ ...     # immutable version directories

This module owns only the naming, enumeration, and alias mechanics. What a version directory
*contains*, and how it is written, stays with the subclass: this base never reads or writes a
version directory's files.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import ClassVar

import tomli_w

from wmo.common.config.store import validate_name
from wmo.common.core.files import write_text_atomic
from wmo.common.core.locks import file_write_lock

CHAMPION_ALIAS = "champion"
ALIASES_FILE = "aliases.toml"


class VersionedArtifactStore:
    """Named, versioned artifacts under `<root>/<subdirectory>/<name>/`.

    Subclasses set the three class variables that name the artifact, and add the load/save methods
    that know its file layout.
    """

    # The directory under the project root that holds every artifact of this kind.
    subdirectory: ClassVar[str]
    # The noun this store's messages call one artifact ("harness", "adapter").
    kind: ClassVar[str]
    # The command that re-points this artifact's champion, so the "your aliases.toml is corrupt"
    # message advises a repair the operator can actually run, plus any extra guidance for it.
    promotion_command: ClassVar[str]
    alias_repair_extra: ClassVar[str] = ""

    def __init__(self, root: str | Path = ".wmo") -> None:
        self.root = Path(root)

    @property
    def artifacts_dir(self) -> Path:
        """The directory holding every named artifact of this kind."""
        return self.root / self.subdirectory

    def dir_for(self, name: str) -> Path:
        """The directory holding every version of `name` (the name is validated, not created)."""
        return self.artifacts_dir / validate_name(name, kind=self.kind)

    # -- enumeration -------------------------------------------------------------------------

    def list_names(self) -> list[str]:
        """Every name that has at least one version, sorted. Empty when nothing is stored."""
        if not self.artifacts_dir.exists():
            return []
        return sorted(
            d.name for d in self.artifacts_dir.iterdir() if d.is_dir() and self.versions(d.name)
        )

    def versions(self, name: str) -> list[int]:
        """The version numbers stored for `name`, ascending. Empty when the name is unknown."""
        directory = self.dir_for(name)
        if not directory.exists():
            return []
        found: list[int] = []
        for child in directory.iterdir():
            if child.is_dir() and child.name.startswith("v") and child.name[1:].isdigit():
                found.append(int(child.name[1:]))
        return sorted(found)

    def exists(self, name: str) -> bool:
        """True when `name` has at least one stored version."""
        return bool(self.versions(name))

    # -- aliases -----------------------------------------------------------------------------

    def aliases(self, name: str) -> dict[str, int]:
        """The alias table for `name`, empty when it has none.

        Names the file on a decode error: `resolve_version(None)` reads this to find the champion,
        so a bare `tomllib` message would reach an operator as a parse error with no path, for a
        file they never edited.
        """
        path = self.dir_for(name) / ALIASES_FILE
        if not path.exists():
            return {}
        try:
            parsed = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"{self.kind} alias file {path} is not valid TOML ({exc}); it maps alias names to "
                f"version numbers under [aliases]. Repair it, or delete it to fall back to the "
                f"latest version until the next {self.promotion_command} promotion re-points "
                f"{CHAMPION_ALIAS}{self.alias_repair_extra}"
            ) from exc
        data = parsed.get("aliases", {})
        return {k: v for k, v in data.items() if isinstance(v, int)}

    def set_alias(self, name: str, alias: str, version: int) -> None:
        """Point `alias` at `version` (moving it if it exists). Rollback is re-pointing.

        Locked and atomic, because this is a read-modify-write of the file that holds the champion
        pointer. Written in place, a crash or a full disk mid-write leaves a truncated
        `aliases.toml` and the champion is gone; done unlocked, two promotions of DIFFERENT
        aliases each read the same table and the later write drops the earlier one, with both
        reporting success.
        """
        if version not in self.versions(name):
            raise ValueError(f"{self.kind} {name!r} has no version v{version}")
        path = self.dir_for(name) / ALIASES_FILE
        with file_write_lock(path, what=f"the {self.kind} alias table"):
            current = self.aliases(name)
            current[alias] = version
            write_text_atomic(path, tomli_w.dumps({"aliases": current}))

    # -- version refs ------------------------------------------------------------------------

    def resolve_version(self, name: str, ref: str | None = None) -> int:
        """Resolve a version ref: `None` -> champion alias, else latest; `"vN"`/`"N"`; an alias.

        Raises:
            FileNotFoundError: `name` has no stored version.
            ValueError: `ref` names a version or alias this name does not have.
        """
        available = self.versions(name)
        if not available:
            raise FileNotFoundError(
                f"no {self.kind} named {name!r} under {self.artifacts_dir} "
                f"(have: {', '.join(self.list_names()) or 'none'})"
            )
        aliases = self.aliases(name)
        if ref is None:
            return aliases.get(CHAMPION_ALIAS, available[-1])
        normalized = ref.removeprefix("v")
        if normalized.isdigit():
            version = int(normalized)
            if version not in available:
                raise ValueError(f"{self.kind} {name!r} has no version v{version}")
            return version
        if ref in aliases:
            return aliases[ref]
        raise ValueError(f"{self.kind} {name!r} has no version or alias {ref!r}")

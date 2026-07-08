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

import tomllib
from pathlib import Path

import tomli_w

from wmh.config.store import validate_name
from wmh.harness.doc import (
    MAX_TURNS_ID,
    TEMPERATURE_ID,
    TOOL_POLICY_ID,
    HarnessDoc,
    Surface,
    SurfaceKind,
)
from wmh.harness.skills import Skill

HARNESSES_DIR = "harnesses"
CHAMPION_ALIAS = "champion"

_DOC_FILE = "doc.json"
_SYSTEM_FILE = "SYSTEM.md"
_CONFIG_FILE = "config.toml"
_SKILLS_DIR = "skills"
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
    files = {
        _SYSTEM_FILE: doc.system_prompt(),
        _CONFIG_FILE: tomli_w.dumps(
            {
                "harness": {
                    "tools": doc.tools(),
                    "max_turns": doc.max_turns(),
                    "temperature": doc.temperature(),
                }
            }
        ),
    }
    for skill in doc.skills():
        files[f"{_SKILLS_DIR}/{skill.name}.md"] = skill.to_markdown()
    return files


def _parse_rendered(name: str, directory: Path) -> HarnessDoc:
    """Parse a rendered/hand-authored directory (no doc.json) into a document.

    The whole `SYSTEM.md` becomes one `prompt:core` surface — section boundaries are not
    recoverable from a rendered prompt, which is exactly why `doc.json` is the authoritative form.
    """
    system_path = directory / _SYSTEM_FILE
    if not system_path.exists():
        raise ValueError(f"harness dir {directory} has neither {_DOC_FILE} nor {_SYSTEM_FILE}")
    surfaces = [
        Surface(
            id="prompt:core",
            kind=SurfaceKind.PROMPT,
            content=system_path.read_text(encoding="utf-8"),
        )
    ]
    config_path = directory / _CONFIG_FILE
    if config_path.exists():
        config = tomllib.loads(config_path.read_text(encoding="utf-8")).get("harness", {})
        if "tools" in config:
            surfaces.append(
                Surface(
                    id=TOOL_POLICY_ID,
                    kind=SurfaceKind.TOOL_POLICY,
                    content="\n".join(str(t) for t in config["tools"]),
                )
            )
        if "max_turns" in config:
            surfaces.append(
                Surface(id=MAX_TURNS_ID, kind=SurfaceKind.PARAM, content=str(config["max_turns"]))
            )
        if "temperature" in config:
            surfaces.append(
                Surface(
                    id=TEMPERATURE_ID, kind=SurfaceKind.PARAM, content=str(config["temperature"])
                )
            )
    skills_dir = directory / _SKILLS_DIR
    if skills_dir.exists():
        for path in sorted(skills_dir.glob("*.md")):
            skill = Skill.from_markdown(path.read_text(encoding="utf-8"))
            surfaces.append(
                Surface(
                    id=f"skill:{skill.name}", kind=SurfaceKind.SKILL, content=skill.to_markdown()
                )
            )
    return HarnessDoc(name=name, surfaces=surfaces)

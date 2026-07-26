"""Skills: reusable techniques a harness ships, one `SKILL.md`-style unit each.

A skill is one markdown file with tiny frontmatter (`name`, `description`) and a body. Harnesses
surface skills by **progressive disclosure**: only the name+description index is preloaded into the
system prompt (`render_index`); the agent pulls a full body on demand with the `read_skill` tool,
so always-loaded context stays small while the library grows.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, field_validator

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


class Skill(BaseModel):
    """One reusable skill: a name, a one-line trigger description, and a body to reuse.

    Validation lives on the MODEL, not just the markdown parser: skill names become file paths
    (`skills/<name>.md`), so a programmatically-constructed skill with a hostile name (`../../x`)
    must be impossible, not merely improbable. The description is coerced to one line (it lives in
    frontmatter, where a newline silently truncates on round-trip); the body is stripped so
    round-trips compare equal.
    """

    name: str
    description: str
    body: str

    @field_validator("name")
    @classmethod
    def _kebab_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError(f"skill name {v!r} must be kebab-case ([a-z0-9-])")
        return v

    @field_validator("description")
    @classmethod
    def _one_line(cls, v: str) -> str:
        return " ".join(v.split())

    @field_validator("body")
    @classmethod
    def _stripped(cls, v: str) -> str:
        return v.strip()

    def to_markdown(self) -> str:
        """Serialize to a SKILL.md-style file (frontmatter + body)."""
        return f"---\nname: {self.name}\ndescription: {self.description}\n---\n{self.body}"

    @classmethod
    def from_markdown(cls, text: str) -> Skill:
        match = _FRONTMATTER_RE.match(text)
        if match is None:
            raise ValueError("skill file has no frontmatter")
        meta = parse_frontmatter(match.group(1))
        name = meta.get("name", "")
        if not _NAME_RE.match(name):
            raise ValueError(f"skill name {name!r} must be kebab-case")
        return cls(name=name, description=meta.get("description", ""), body=match.group(2))


def parse_frontmatter(block: str) -> dict[str, str]:
    """Parse the tiny `key: value` frontmatter (one field per line)."""
    out: dict[str, str] = {}
    for line in block.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


class SkillLibrary:
    """An in-memory set of skills, loadable from / writable to a `skills/` directory."""

    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills: dict[str, Skill] = {s.name: s for s in (skills or [])}

    @classmethod
    def from_dir(cls, root: str | Path) -> SkillLibrary:
        """Load every `*.md` under `root` (a malformed skill file is an error, not skipped —
        a harness that ships a broken skill should fail loudly at load time, not at rollout)."""
        library = cls()
        root = Path(root)
        if not root.exists():
            return library
        for path in sorted(root.glob("*.md")):
            skill = Skill.from_markdown(path.read_text(encoding="utf-8"))
            library._skills[skill.name] = skill
        return library

    def write_dir(self, root: str | Path) -> None:
        """Write one `<name>.md` per skill under `root` (created if missing)."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for skill in self._skills.values():
            (root / f"{skill.name}.md").write_text(skill.to_markdown(), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._skills)

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def render_index(self) -> str:
        """Progressive-disclosure index: name + description only (bodies via read_skill)."""
        if not self._skills:
            return ""
        return "\n".join(f"- {s.name}: {s.description}" for s in self._skills.values())

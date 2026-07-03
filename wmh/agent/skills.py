"""The agent's self-written skill library (Voyager + Anthropic Agent-Skills).

The agent accumulates skills across runs: when it works out a repeatable way to do something it
calls `save_skill`, and the skill persists to disk as a `SKILL.md`-style file. On later runs the
library is injected into the system prompt via **progressive disclosure** — only each skill's
name+description is preloaded (`render_index`); the agent pulls a full body on demand with the
`read_skill` tool. This keeps always-loaded context small (pi's token budget) as the library grows.

Storage is one markdown file per skill under `<root>/skills/<name>.md` with YAML-ish frontmatter, so
the library is human-auditable and git-diffable (the Agent-Skills format), not an opaque blob.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


class Skill(BaseModel):
    """One reusable skill: a name, a one-line trigger description, and a body to reuse."""

    name: str
    description: str
    body: str

    def to_markdown(self) -> str:
        """Serialize to a SKILL.md-style file (frontmatter + body)."""
        return f"---\nname: {self.name}\ndescription: {self.description}\n---\n{self.body}"

    @classmethod
    def from_markdown(cls, text: str) -> Skill:
        match = _FRONTMATTER_RE.match(text)
        if match is None:
            raise ValueError("skill file has no frontmatter")
        meta = _parse_frontmatter(match.group(1))
        return cls(
            name=meta.get("name", ""),
            description=meta.get("description", ""),
            body=match.group(2).strip(),
        )


def _parse_frontmatter(block: str) -> dict[str, str]:
    """Parse the tiny `key: value` frontmatter (one field per line)."""
    out: dict[str, str] = {}
    for line in block.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def normalize_skill_name(name: str) -> str:
    """Coerce a proposed name into a safe kebab-case slug (the agent's names are untrusted)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "skill"


class SkillLibrary:
    """A directory of skill files. In-memory when `root` is None (evolution scratch libraries).

    The library is the unit of transfer between runs and between harness variants: a variant carries
    the *names* of the seed skills it starts from, and successful runs grow the shared on-disk
    library that later variants can seed from.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else None
        self._skills: dict[str, Skill] = {}
        if self._root is not None and self._root.exists():
            self._load()

    def _load(self) -> None:
        assert self._root is not None
        for path in sorted(self._root.glob("*.md")):
            try:
                skill = Skill.from_markdown(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if skill.name:
                self._skills[skill.name] = skill

    def __len__(self) -> int:
        return len(self._skills)

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name) or self._skills.get(normalize_skill_name(name))

    def save(self, name: str, description: str, body: str) -> Skill:
        """Add or overwrite a skill (persisting to disk when the library is on-disk)."""
        skill = Skill(name=normalize_skill_name(name), description=description.strip(), body=body)
        self._skills[skill.name] = skill
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
            (self._root / f"{skill.name}.md").write_text(skill.to_markdown(), encoding="utf-8")
        return skill

    def subset(self, names: list[str]) -> SkillLibrary:
        """An in-memory library holding just `names` present here (a variant's seed skills)."""
        lib = SkillLibrary()
        for name in names:
            skill = self.get(name)
            if skill is not None:
                lib._skills[skill.name] = skill
        return lib

    def render_index(self) -> str:
        """Progressive-disclosure index: name + description only (bodies loaded via read_skill)."""
        if not self._skills:
            return "(none yet — save skills as you discover reusable techniques)"
        return "\n".join(f"- {s.name}: {s.description}" for s in self._skills.values())

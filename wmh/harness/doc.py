"""`HarnessDoc`: a harness as a typed document of identity-keyed surfaces.

A harness is not a directory of files — it is a set of named **surfaces**, each an independently
addressable unit of behavior: prompt sections, the tool policy, scalar loop parameters, and skills.
Files (`SYSTEM.md`, `config.toml`, `skills/*.md`) are a *render target* the store exports for
running the harness elsewhere; the document is the interface everything else programs against.

Why surfaces instead of files:
- **Identity.** Every surface has a stable id (`prompt:core`, `skill:count-words`). An update names
  its target; nothing is ever addressed by position or filename, so "which thing changed" is never
  inferred.
- **Content addressing.** Each surface has a content hash, and the document has a hash over its
  surfaces. "The score of harness X" is well-defined because X is a hash; an update can assert
  exactly what it believes it is editing.
- **Typed validation.** A document validates as a whole (tools resolve, `submit` present, params in
  range, budgets respected) the moment it is constructed — an invalid harness cannot exist as a
  value, so nothing downstream re-checks.

Surface *content* stays a free-form string on purpose: structure lives in the envelope (ids, kinds,
hashes, budgets), not in the payload, so richer surface kinds can be added without changing how
updates work.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from wmh.harness.runtime import DEFAULT_MAX_TURNS, DEFAULT_SYSTEM_PROMPT, AgentRuntime
from wmh.harness.skills import Skill, SkillLibrary
from wmh.harness.tools import DEFAULT_TOOLS, resolve_tools
from wmh.providers.base import Provider

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Well-known surface ids. Only TOOL_POLICY and the two params are singletons; prompt and skill
# surfaces may be added freely (an update can split `prompt:core` into finer sections).
TOOL_POLICY_ID = "tool_policy:main"
MAX_TURNS_ID = "param:max-turns"
TEMPERATURE_ID = "param:temperature"

DEFAULT_TEMPERATURE = 0.7


class SurfaceKind(StrEnum):
    PROMPT = "prompt"  # a section of the system prompt (joined in id order)
    SKILL = "skill"  # one skill: frontmatter (name, description) + body
    TOOL_POLICY = "tool_policy"  # the tool list, one tool name per line
    PARAM = "param"  # a scalar loop knob, serialized as its string form


class Surface(BaseModel):
    """One named, independently addressable unit of harness behavior."""

    id: str  # "<kind>:<slug>"
    kind: SurfaceKind
    content: str
    # Optional size budget (characters). Enforced at construction: a surface that exceeds its
    # budget is invalid, so context cost is a schema property rather than a runtime surprise.
    budget: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate(self) -> Surface:
        prefix, sep, slug = self.id.partition(":")
        if not sep or prefix != self.kind.value or not _SLUG_RE.match(slug):
            raise ValueError(
                f"surface id {self.id!r} must be '{self.kind.value}:<kebab-slug>' matching its kind"
            )
        if self.budget is not None and len(self.content) > self.budget:
            raise ValueError(
                f"surface {self.id!r} content is {len(self.content)} chars, "
                f"over its budget of {self.budget}"
            )
        return self

    @property
    def slug(self) -> str:
        return self.id.partition(":")[2]

    @property
    def content_hash(self) -> str:
        return _digest(self.content)


class HarnessDoc(BaseModel):
    """A complete, validated harness: the value the runtime runs and updates are applied to."""

    name: str
    version: int = Field(default=0, ge=0)  # assigned by the store on save; 0 = unsaved
    surfaces: list[Surface]

    @field_validator("surfaces")
    @classmethod
    def _canonical_order(cls, v: list[Surface]) -> list[Surface]:
        return sorted(v, key=lambda s: s.id)

    @model_validator(mode="after")
    def _validate_document(self) -> HarnessDoc:
        ids = [s.id for s in self.surfaces]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate surface id(s): {duplicates}")
        if not any(s.kind is SurfaceKind.PROMPT for s in self.surfaces):
            raise ValueError("a harness needs at least one prompt surface")
        # These validations construct the derived values; failures surface here, at the boundary.
        self.tools()
        self.max_turns()
        self.temperature()
        for surface in self.surfaces:
            if surface.kind is SurfaceKind.SKILL:
                skill = Skill.from_markdown(surface.content)
                if skill.name != surface.slug:
                    raise ValueError(
                        f"skill surface {surface.id!r} declares frontmatter name "
                        f"{skill.name!r}; the slug and frontmatter name must match"
                    )
        return self

    # -- surface access ---------------------------------------------------------------------

    def surface(self, surface_id: str) -> Surface | None:
        for s in self.surfaces:
            if s.id == surface_id:
                return s
        return None

    def surface_hashes(self) -> dict[str, str]:
        return {s.id: s.content_hash for s in self.surfaces}

    @property
    def doc_hash(self) -> str:
        """Content identity of the whole document (order-independent via canonical sort)."""
        joined = "\n".join(f"{s.id}\x00{s.content_hash}" for s in self.surfaces)
        return _digest(joined)

    # -- derived, validated views ------------------------------------------------------------

    def system_prompt(self) -> str:
        """All prompt surfaces joined in id order (a single `prompt:core` is the common case)."""
        parts = [s.content for s in self.surfaces if s.kind is SurfaceKind.PROMPT]
        return "\n\n".join(parts)

    def tools(self) -> list[str]:
        policy = self.surface(TOOL_POLICY_ID)
        if policy is None:
            return list(DEFAULT_TOOLS)
        names = [line.strip() for line in policy.content.splitlines() if line.strip()]
        resolve_tools(names)  # raises on unknown tools / missing submit
        return names

    def max_turns(self) -> int:
        raw = self.surface(MAX_TURNS_ID)
        if raw is None:
            return DEFAULT_MAX_TURNS
        try:
            value = int(raw.content.strip())
        except ValueError as exc:
            raise ValueError(f"{MAX_TURNS_ID} must be an integer, got {raw.content!r}") from exc
        if value < 1:
            raise ValueError(f"{MAX_TURNS_ID} must be >= 1, got {value}")
        return value

    def temperature(self) -> float:
        raw = self.surface(TEMPERATURE_ID)
        if raw is None:
            return DEFAULT_TEMPERATURE
        try:
            value = float(raw.content.strip())
        except ValueError as exc:
            raise ValueError(f"{TEMPERATURE_ID} must be a float, got {raw.content!r}") from exc
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"{TEMPERATURE_ID} must be in [0, 2], got {value}")
        return value

    def skills(self) -> list[Skill]:
        return [
            Skill.from_markdown(s.content) for s in self.surfaces if s.kind is SurfaceKind.SKILL
        ]

    def runtime(self, provider: Provider) -> AgentRuntime:
        """The configured agent runtime this document describes."""
        return AgentRuntime(
            provider,
            system_prompt=self.system_prompt(),
            tools=self.tools(),
            max_turns=self.max_turns(),
            temperature=self.temperature(),
            skills=SkillLibrary(self.skills()),
        )

    @classmethod
    def baseline(cls, name: str = "baseline") -> HarnessDoc:
        """The default harness: one core prompt, the default tools, default loop params."""
        return cls(
            name=name,
            surfaces=[
                Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content=DEFAULT_SYSTEM_PROMPT),
                Surface(
                    id=TOOL_POLICY_ID,
                    kind=SurfaceKind.TOOL_POLICY,
                    content="\n".join(DEFAULT_TOOLS),
                ),
                Surface(id=MAX_TURNS_ID, kind=SurfaceKind.PARAM, content=str(DEFAULT_MAX_TURNS)),
                Surface(
                    id=TEMPERATURE_ID, kind=SurfaceKind.PARAM, content=str(DEFAULT_TEMPERATURE)
                ),
            ],
        )


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

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
import os
import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from wmh.harness.code_runtime import (
    DEFAULT_RUNTIME_CODE,
    CodeRuntime,
    compile_harness_code,
)
from wmh.harness.runtime import DEFAULT_MAX_TURNS, DEFAULT_SYSTEM_PROMPT, AgentRuntime, Runtime
from wmh.harness.skills import Skill, SkillLibrary
from wmh.harness.tools import DEFAULT_TOOLS, render_tools, resolve_tools
from wmh.providers.base import Provider

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Well-known surface ids. Only TOOL_POLICY and the two params are singletons; prompt and skill
# surfaces may be added freely (an update can split `prompt:core` into finer sections).
TOOL_POLICY_ID = "tool_policy:main"
MAX_TURNS_ID = "param:max-turns"
TEMPERATURE_ID = "param:temperature"
RUNTIME_KIND_ID = "param:runtime-kind"  # absent/"kit-python" -> in-process; "pi-node" -> PiRuntime
CODE_RUNTIME_ID = "code:runtime"

_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")

DEFAULT_TEMPERATURE = 0.7


class SurfaceKind(StrEnum):
    PROMPT = "prompt"  # a section of the system prompt (joined in id order)
    SKILL = "skill"  # one skill: frontmatter (name, description) + body
    TOOL_POLICY = "tool_policy"  # the tool list, one tool name per line
    PARAM = "param"  # a scalar loop knob, serialized as its string form
    CODE = "code"  # the agent loop itself: a module defining `run(kit)` (see code_runtime)


class Surface(BaseModel):
    """One named, independently addressable unit of harness behavior."""

    id: str  # "<kind>:<slug>"
    kind: SurfaceKind
    content: str
    # For CODE surfaces of a vendored multi-file harness: the file path the content materializes
    # to (relative, no traversal). A path-less CODE surface is the legacy in-process
    # `code:runtime` module.
    path: str | None = None
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
        if self.path is not None:
            if self.kind is not SurfaceKind.CODE:
                raise ValueError(f"surface {self.id!r}: only code surfaces may carry a path")
            if not _SAFE_PATH_RE.match(self.path) or ".." in self.path.split("/"):
                raise ValueError(f"surface {self.id!r}: unsafe path {self.path!r}")
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
            elif surface.kind is SurfaceKind.CODE:
                if surface.path is None:
                    # The legacy in-process runtime module: a singleton, compile-checked here.
                    if surface.id != CODE_RUNTIME_ID:
                        raise ValueError(
                            f"path-less code surface must be {CODE_RUNTIME_ID!r} "
                            f"(got {surface.id!r}); vendored files carry a `path`"
                        )
                    compile_harness_code(surface.content)
        paths = [s.path for s in self.surfaces if s.path is not None]
        dup_paths = sorted({p for p in paths if paths.count(p) > 1})
        if dup_paths:
            raise ValueError(f"duplicate code surface path(s): {dup_paths}")
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

    def runtime_kind(self) -> str:
        raw = self.surface(RUNTIME_KIND_ID)
        return raw.content.strip() if raw is not None else "kit-python"

    def code_files(self) -> list[Surface]:
        """The vendored code surfaces (those carrying a file path), in id order."""
        return [s for s in self.surfaces if s.kind is SurfaceKind.CODE and s.path is not None]

    def runtime(self, provider: Provider, *, backend: str = "local") -> Runtime:
        """The configured agent runtime this document describes.

        `backend` chooses WHERE the harness executes; only `local` (this process) is built in here.
        `param:runtime-kind` = "pi-node" dispatches the vendored-pi runner (over the SSH shim, or
        the RunnerLink frame transport when PI_TRANSPORT=link). Otherwise a `code:runtime` surface
        drives episodes with the harness's own in-process program; with neither, the fixed baseline
        loop runs. All expose the same `run(task_id, instruction, environment) -> RunResult` shape
        closed-loop eval drives.
        """
        if backend != "local":
            raise ValueError(f"unknown backend {backend!r}; choose local")
        if self.runtime_kind() == "pi-node":
            skills = SkillLibrary(self.skills())
            code_files = {s.path: s.content for s in self.code_files() if s.path is not None}
            # PI_TRANSPORT=link routes pi to the RunnerLink frame transport (a persistent runner the
            # host set via runner_link.set_active_channel) instead of the per-episode SSH shim; the
            # default (unset / "ssh") keeps PiRuntime. The worker LLM reads the same PI_AGENT_* env.
            if os.environ.get("PI_TRANSPORT") == "link":
                from wmh.harness.runner_link import (
                    RunnerLink,
                    active_channel,
                    worker_config_from_env,
                )

                channel = active_channel()
                if channel is None:
                    raise RuntimeError(
                        "PI_TRANSPORT=link but no active runner channel; call "
                        "runner_link.set_active_channel(channel) before running episodes"
                    )
                return RunnerLink(
                    channel,
                    tools=resolve_tools(self.tools()),
                    worker=worker_config_from_env(),
                    system_prompt=self._assembled_prompt(skills),
                    files=code_files,
                )
            from wmh.harness.pi_runtime import PiRuntime  # circular: pi_runtime imports doc

            return PiRuntime(
                provider,
                files=code_files,
                tools=resolve_tools(self.tools()),
                temperature=self.temperature(),
                skills=skills,
                system_prompt=self._assembled_prompt(skills),
            )
        code = self.surface(CODE_RUNTIME_ID)
        skills = SkillLibrary(self.skills())
        if code is not None:
            return CodeRuntime(
                provider,
                code=code.content,
                tools=resolve_tools(self.tools()),
                temperature=self.temperature(),
                skills=skills,
                system_prompt=self._assembled_prompt(skills),
            )
        return AgentRuntime(
            provider,
            system_prompt=self.system_prompt(),
            tools=self.tools(),
            max_turns=self.max_turns(),
            temperature=self.temperature(),
            skills=skills,
        )

    def _assembled_prompt(self, skills: SkillLibrary) -> str:
        """The full system prompt handed to harness code: sections + tools + skills index."""
        prompt = f"{self.system_prompt()}\n\n## Tools\n{render_tools(resolve_tools(self.tools()))}"
        index = skills.render_index()
        if index:
            prompt += f"\n\n## Your skills (read a body with read_skill)\n{index}"
        return prompt

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


def code_baseline(name: str = "baseline") -> HarnessDoc:
    """The baseline harness with its loop as an editable `code:runtime` surface.

    Behaviorally equivalent to `HarnessDoc.baseline()` — same prompt, tools, and one-call-per-turn
    loop — but the loop is data, so `wmh harness create` can propose structural changes to it.
    """
    base = HarnessDoc.baseline(name)
    code = Surface(id=CODE_RUNTIME_ID, kind=SurfaceKind.CODE, content=DEFAULT_RUNTIME_CODE)
    return HarnessDoc(name=name, surfaces=[*base.surfaces, code])


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

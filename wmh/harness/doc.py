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
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator, model_validator

from wmh.core.text import validate_durable_text
from wmh.harness.code_runtime import (
    DEFAULT_RUNTIME_CODE,
    CodeRuntime,
    compile_harness_code,
)
from wmh.harness.runtime import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_SYSTEM_PROMPT,
    AgentRuntime,
    Runtime,
)
from wmh.harness.skills import Skill, SkillLibrary
from wmh.harness.tools import DEFAULT_TOOLS, READ_SKILL, render_tools, resolve_tools
from wmh.providers.base import Provider, ToolCallingProvider

if TYPE_CHECKING:
    # Import-time neutral: pi_e2b (the optional e2b extra's consumer) is imported lazily inside
    # runtime(); this name exists only for the e2b_pool annotation.
    from wmh.harness.pi_e2b import E2BSandboxPool

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Well-known surface ids. The tool policy and scalar parameters are singletons; prompt and skill
# surfaces may be added freely (an update can split `prompt:core` into finer sections).
TOOL_POLICY_ID = "tool_policy:main"
MAX_TURNS_ID = "param:max-turns"
MAX_OUTPUT_TOKENS_ID = "param:max-output-tokens"
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
        validate_durable_text(self.content, field=f"surface {self.id!r} content")
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
        self.max_output_tokens()
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

    def max_output_tokens(self) -> int:
        """Return the per-model-call output cap carried by every pi execution mode."""
        raw = self.surface(MAX_OUTPUT_TOKENS_ID)
        if raw is None:
            return DEFAULT_MAX_OUTPUT_TOKENS
        try:
            value = int(raw.content.strip())
        except ValueError as exc:
            raise ValueError(
                f"{MAX_OUTPUT_TOKENS_ID} must be an integer, got {raw.content!r}"
            ) from exc
        if value < 1:
            raise ValueError(f"{MAX_OUTPUT_TOKENS_ID} must be >= 1, got {value}")
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

    def runtime(
        self,
        provider: Provider,
        *,
        backend: str = "local",
        e2b_template: str | None = None,
        e2b_pool: E2BSandboxPool | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Runtime:
        """The configured agent runtime this document describes.

        `backend` chooses WHERE the harness process executes; the ENVIRONMENT its tool calls hit
        is whatever `AgentEnvironment` the eval binds (normally the world-model simulation),
        regardless of backend. `local` runs in/from this process. `e2b` runs the harness process
        in E2B sandboxes the runtime owns — only meaningful for `param:runtime-kind` = "pi-node"
        (the vendored pi agent, whose real context management is the point of running it); any
        other kind raises, because its loop already runs in-process and "e2b" would silently mean
        nothing. `e2b_template` names a prebaked sandbox template whose bootstrap (node 22 + pi's
        npm deps) is already done; default is $WMH_E2B_TEMPLATE. Under `local`, "pi-node" uses
        the SSH shim (or the RunnerLink frame transport when PI_TRANSPORT=link); otherwise a
        `code:runtime` surface drives episodes with the harness's own in-process program; with
        neither, the fixed baseline loop runs. All expose the same
        `run(task_id, instruction, environment) -> RunResult` shape closed-loop eval drives.
        """
        if backend not in ("local", "e2b"):
            raise ValueError(f"unknown backend {backend!r}; choose local or e2b")
        if self.runtime_kind() == "pi-node":
            skills = SkillLibrary(self.skills())
            code_files = {s.path: s.content for s in self.code_files() if s.path is not None}
            tool_names = self.tools()
            # Progressive disclosure is runtime plumbing, not a burden on every persisted tool
            # policy. Keep pi-node behavior aligned with AgentRuntime: a skill-bearing document
            # always exposes read_skill, even when the authored policy lists only env tools.
            if len(skills) and READ_SKILL.name not in tool_names:
                tool_names.append(READ_SKILL.name)
            tools = resolve_tools(tool_names)
            structured_provider = provider if isinstance(provider, ToolCallingProvider) else None
            if (
                backend == "e2b" or os.environ.get("PI_TRANSPORT") == "link"
            ) and structured_provider is None:
                raise TypeError(
                    "pi-node link/e2b execution needs a ToolCallingProvider; "
                    "use a structured provider or WaterfallProvider"
                )
            if backend == "e2b":
                # Lazy: the e2b backend is an optional extra; `local` must import none of it.
                from wmh.harness.pi_e2b import E2BPiRuntime

                assert structured_provider is not None
                return E2BPiRuntime(
                    provider=structured_provider,
                    files=code_files,
                    tools=tools,
                    system_prompt=self.assembled_prompt(skills),
                    temperature=self.temperature(),
                    skills=skills,
                    template=e2b_template,
                    pool=e2b_pool,
                    max_turns=self.max_turns(),
                    max_output_tokens=self.max_output_tokens(),
                    should_cancel=should_cancel,
                )
            # PI_TRANSPORT=link routes pi to the RunnerLink frame transport (a persistent runner the
            # host set via runner_link.set_active_channel) instead of the per-episode SSH shim; the
            # default (unset / "ssh") keeps PiRuntime. The worker LLM reads the same PI_AGENT_* env.
            if os.environ.get("PI_TRANSPORT") == "link":
                from wmh.harness.runner_link import (
                    RunnerLink,
                    active_channel,
                )

                channel = active_channel()
                if channel is None:
                    raise RuntimeError(
                        "PI_TRANSPORT=link but no active runner channel; call "
                        "runner_link.set_active_channel(channel) before running episodes"
                    )
                assert structured_provider is not None
                return RunnerLink(
                    channel,
                    tools=tools,
                    provider=structured_provider,
                    system_prompt=self.assembled_prompt(skills),
                    files=code_files,
                    temperature=self.temperature(),
                    skills=skills,
                    max_turns=self.max_turns(),
                    max_output_tokens=self.max_output_tokens(),
                    should_cancel=should_cancel,
                )
            from wmh.harness.pi_runtime import PiRuntime  # circular: pi_runtime imports doc

            return PiRuntime(
                provider,
                files=code_files,
                tools=tools,
                temperature=self.temperature(),
                skills=skills,
                system_prompt=self.assembled_prompt(skills),
                max_turns=self.max_turns(),
                max_output_tokens=self.max_output_tokens(),
            )
        if backend == "e2b":
            raise ValueError(
                f"backend='e2b' runs the pi-node harness process in a sandbox; this harness's "
                f"runtime kind is {self.runtime_kind()!r}, which already runs in-process — "
                "use backend='local'"
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
                system_prompt=self.assembled_prompt(skills),
            )
        return AgentRuntime(
            provider,
            system_prompt=self.system_prompt(),
            tools=self.tools(),
            max_turns=self.max_turns(),
            temperature=self.temperature(),
            skills=skills,
        )

    def assembled_prompt(self, skills: SkillLibrary | None = None) -> str:
        """Return the system prompt shared by episode and project session runtimes."""
        resolved_skills = skills if skills is not None else SkillLibrary(self.skills())
        tool_names = self.tools()
        if len(resolved_skills) and READ_SKILL.name not in tool_names:
            tool_names.append(READ_SKILL.name)
        prompt = f"{self.system_prompt()}\n\n## Tools\n{render_tools(resolve_tools(tool_names))}"
        index = resolved_skills.render_index()
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
    loop, but the loop is data, so `wmh optimize` can propose structural changes to it.
    """
    base = HarnessDoc.baseline(name)
    code = Surface(id=CODE_RUNTIME_ID, kind=SurfaceKind.CODE, content=DEFAULT_RUNTIME_CODE)
    return HarnessDoc(name=name, surfaces=[*base.surfaces, code])


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

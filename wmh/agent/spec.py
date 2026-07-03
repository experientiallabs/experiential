"""`HarnessSpec`: the evolvable description of one agent harness variant.

This is the artifact the evolutionary manager mutates — the AlphaEvolve "EVOLVE-BLOCK" of this
system. It is deliberately small and declarative (prompt + tool set + seed skills + loop knobs) so a
mutation is a legible diff, the archive is human-auditable (DGM lineage), and every variant is
reconstructable from its persisted JSON. The runtime is *fixed*; only the spec varies, so any score
delta is attributable to the spec alone.

Following pi, the default `system_prompt` is short; following ADAS, each variant carries a `name` +
`motivation` so the archive doubles as design memory the meta-agent reads when proposing mutations.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from wmh.agent.tools import DEFAULT_TOOLS, SUBMIT, resolve_tools

DEFAULT_SYSTEM_PROMPT = """You are a capable command-line agent working inside a Linux environment.
You are given a task. Accomplish it by taking ONE action at a time.

Every reply MUST be a single JSON object and nothing else:
{"tool": "<tool name>", "arguments": {<the tool's arguments>}}

Work in small, verifiable steps: inspect state, act, check the result, then continue. When the task
is done, call `submit` with your answer. If you discover a reusable technique, `save_skill` it so
future runs start ahead. Prefer `bash` and composing small commands over guessing."""


class HarnessSpec(BaseModel):
    """One point in harness design space: the knobs that define how the agent behaves."""

    name: str = "base"
    motivation: str = "the minimal pi-style baseline harness"  # why this variant exists (ADAS)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tools: list[str] = Field(default_factory=lambda: list(DEFAULT_TOOLS))
    seed_skills: list[str] = Field(default_factory=list)  # skill names to preload from the library
    max_turns: int = 20  # turn cap (Ralph/closed-loop stop condition)
    temperature: float = 0.7
    parent: str | None = None  # name of the variant this was mutated from (lineage)

    @field_validator("max_turns")
    @classmethod
    def _positive_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_turns must be >= 1")
        return v

    @field_validator("temperature")
    @classmethod
    def _valid_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        return v

    @model_validator(mode="after")
    def _validate_tools(self) -> HarnessSpec:
        # Deduplicate while preserving order, and confirm every tool is real and `submit` is present
        # (without it the agent can never end a run).
        seen: set[str] = set()
        deduped = [t for t in self.tools if not (t in seen or seen.add(t))]
        object.__setattr__(self, "tools", deduped)
        resolve_tools(deduped)  # raises on unknown tool
        if SUBMIT.name not in deduped:
            raise ValueError(f"a harness must include the {SUBMIT.name!r} tool")
        return self

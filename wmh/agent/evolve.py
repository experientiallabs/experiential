"""Evolutionary harness manager: search harness space using closed-loop score deltas.

This is the meta-loop. It maintains an **archive** of scored `HarnessSpec` variants (DGM) and, each
generation, (1) selects a parent from the archive with probability rising in its score and falling
in how many children it already has — so mediocre-but-unexpanded variants stay viable stepping
stones, not pure hill-climbing; (2) has an LLM **reflect** on that parent's failing transcripts and
propose a targeted mutation (GEPA reflection over the OpenEvolve "artifacts" channel), returning a
full child spec with a name + motivation (ADAS); (3) closed-loop **evaluates** the child (k=3)
against the world model and appends it to the archive. Selection keeps any variant best on *some*
(instance-level Pareto, GEPA), so complementary specialists survive an aggregate metric.

Determinism: no wall-clock or RNG entropy (repo rule + reproducible evolution, OpenEvolve). Parent
selection is seeded; the "randomness" across a generation comes from varying the seed by index.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from pydantic import BaseModel, Field, ValidationError

from wmh.agent.closed_loop import (
    ClosedLoopReport,
    evaluate_closed_loop,
    failing_transcripts,
)
from wmh.agent.gold import GoldJudge
from wmh.agent.skills import SkillLibrary
from wmh.agent.spec import DEFAULT_SYSTEM_PROMPT, HarnessSpec
from wmh.agent.tasks import TaskSpec
from wmh.agent.tools import DEFAULT_TOOLS, TOOL_REGISTRY
from wmh.core.parsing import extract_json_object
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Message, Provider

MUTATE_SYSTEM = """You are a meta-agent that improves the *harness* of a command-line AI agent — its
system prompt, its tool set, which saved skills it starts with, and its loop settings. You do NOT
solve the tasks yourself; you redesign the scaffold so the agent solves more of them.

You are given the current harness, how it scored, and WHY it failed (its worst tasks and the unmet
success assertions). Propose ONE improved harness. Change what the failures implicate: if the agent
flailed or never verified its work, tighten the system prompt; if it lacked a capability, adjust the
tools or seed skills; if it ran out of turns, raise the cap. Make a focused change, not a rewrite.

Available tools (you may only reference these names): {tools}

Respond with ONLY a JSON object, no prose:
{{"name": "<short-kebab-name>", "motivation": "<one sentence: what you changed and why>",
  "system_prompt": "<the full system prompt for the new harness>",
  "tools": [<subset/reordering of available tool names; must include \\"submit\\">],
  "seed_skills": [<names of skills to preload; may be empty>],
  "max_turns": <int>, "temperature": <float 0..2>}}"""


class ArchiveEntry(BaseModel):
    """One evaluated variant in the archive: its spec, score, and lineage bookkeeping."""

    spec: HarnessSpec
    report: ClosedLoopReport
    generation: int = 0
    children: int = 0  # how many mutations have used this as a parent (DGM expansion discount)

    @property
    def fitness(self) -> float:
        return self.report.success_rate


class HarnessArchive(BaseModel):
    """The population of scored harness variants, with DGM-style parent selection.

    Kept append-only so lineage is fully auditable (DGM) — any suspicious score jump can be traced
    through `spec.parent`. `best()` is the current champion; `pareto_names()` are the variants worth
    keeping because each wins on at least one task.
    """

    entries: list[ArchiveEntry] = Field(default_factory=list)

    def add(self, entry: ArchiveEntry) -> None:
        self.entries.append(entry)

    def best(self) -> ArchiveEntry:
        return max(self.entries, key=lambda e: e.fitness)

    def select_parent(self, seed: int) -> ArchiveEntry:
        """Pick a parent ∝ fitness, discounted by children count (DGM stepping-stone selection).

        Weight = (fitness + eps) / (1 + children). Deterministic given `seed` (a hash of the seed
        maps into the cumulative weight), so a run is reproducible.
        """
        weights = [(e.fitness + 0.05) / (1.0 + e.children) for e in self.entries]
        total = sum(weights)
        if total <= 0:
            return self.entries[0]
        target = (_seed_fraction(seed)) * total
        cumulative = 0.0
        for entry, weight in zip(self.entries, weights, strict=True):
            cumulative += weight
            if cumulative >= target:
                return entry
        return self.entries[-1]

    def pareto_names(self) -> list[str]:
        """Names of variants that are strictly best on at least one task (instance-level Pareto)."""
        task_ids = {tid for e in self.entries for tid in e.report.per_task}
        keep: set[str] = set()
        for tid in task_ids:
            best_entry = max(
                self.entries,
                key=lambda e: (
                    e.report.per_task[tid].success_rate if tid in e.report.per_task else -1.0
                ),
            )
            keep.add(best_entry.spec.name)
        return sorted(keep)


def _seed_fraction(seed: int) -> float:
    """Map an int seed deterministically into [0, 1) (no RNG, per repo rules)."""
    digest = hashlib.blake2b(str(seed).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


class _RawMutation(BaseModel):
    """Lenient view of the meta-agent's proposed harness before validation."""

    name: str = "variant"
    motivation: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tools: list[str] = Field(default_factory=lambda: list(DEFAULT_TOOLS))
    seed_skills: list[str] = Field(default_factory=list)
    max_turns: int = 20
    temperature: float = 0.7


def mutate(
    parent: HarnessSpec,
    report: ClosedLoopReport,
    tasks: list[TaskSpec],
    provider: Provider,
    *,
    library_names: list[str] | None = None,
) -> HarnessSpec:
    """Ask the meta-agent for one improved child of `parent`, grounded in why the parent failed.

    Falls back to the parent (unchanged but re-parented) if the reply can't be parsed into a valid
    spec, so a single bad mutation never aborts a generation.
    """
    prompt = _mutation_prompt(parent, report, tasks, library_names or [])
    completion = provider.complete(
        MUTATE_SYSTEM.format(tools=", ".join(sorted(TOOL_REGISTRY))),
        [Message(role="user", content=prompt)],
        temperature=0.9,
        max_tokens=2048,
    )
    child = _parse_mutation(completion.text, parent)
    return child


def _mutation_prompt(
    parent: HarnessSpec, report: ClosedLoopReport, tasks: list[TaskSpec], library_names: list[str]
) -> str:
    available = ", ".join(library_names) if library_names else "(none saved yet)"
    return (
        f"CURRENT HARNESS ({parent.name}):\n{parent.model_dump_json(indent=2)}\n\n"
        f"SCORE: success_rate={report.success_rate:.3f} (±{report.success_std:.3f}), "
        f"mean_assertion_fraction={report.mean_fraction:.3f} over {len(report.per_task)} tasks, "
        f"k={report.k} passes each.\n\n"
        f"WHY IT FAILED (worst tasks and unmet success assertions):\n"
        f"{failing_transcripts(report, tasks) or '(no failures — try a bolder generalization)'}\n\n"
        f"SKILLS AVAILABLE TO SEED: {available}\n\n"
        f"Propose one improved harness as JSON."
    )


def _parse_mutation(text: str, parent: HarnessSpec) -> HarnessSpec:
    raw = extract_json_object(text)
    if raw is not None:
        try:
            proposed = _RawMutation.model_validate_json(raw)
            # Drop tool names the meta-agent invented; the spec validator would reject and abort.
            tools = [t for t in proposed.tools if t in TOOL_REGISTRY] or list(DEFAULT_TOOLS)
            return HarnessSpec(
                name=proposed.name,
                motivation=proposed.motivation,
                system_prompt=proposed.system_prompt,
                tools=tools,
                seed_skills=proposed.seed_skills,
                max_turns=proposed.max_turns,
                temperature=proposed.temperature,
                parent=parent.name,
            )
        except (ValidationError, ValueError):
            pass
    # Unparseable/invalid: re-parent the unchanged parent so the generation still advances.
    return parent.model_copy(update={"parent": parent.name})


class EvolveResult(BaseModel):
    """The outcome of an evolution run: the best variant and the full audited archive."""

    best: HarnessSpec
    best_score: float
    archive: HarnessArchive
    generations: int


ProgressFn = Callable[[int, str, float], None]  # (generation, variant_name, score)


def evolve(
    seed_spec: HarnessSpec,
    tasks: list[TaskSpec],
    world_model: WorldModel,
    agent_provider: Provider,
    meta_provider: Provider,
    judge: GoldJudge,
    *,
    generations: int = 5,
    k: int = 3,
    library: SkillLibrary | None = None,
    on_progress: ProgressFn | None = None,
) -> EvolveResult:
    """Run the meta-loop: score the seed, then mutate→evaluate→archive for `generations` rounds.

    `agent_provider` runs the inner agent; `meta_provider` proposes mutations (they can be the same
    model). Returns the best variant found and the archive (persist it to keep lineage). Every
    evaluation is k passes (default 3), honoring the eval-reporting convention.
    """
    archive = HarnessArchive()
    library_names = library.names() if library is not None else []

    seed_report = evaluate_closed_loop(
        seed_spec, tasks, world_model, agent_provider, judge, library=library, k=k
    )
    archive.add(ArchiveEntry(spec=seed_spec, report=seed_report, generation=0))
    if on_progress is not None:
        on_progress(0, seed_spec.name, seed_report.success_rate)

    for generation in range(1, generations + 1):
        parent_entry = archive.select_parent(seed=generation)
        parent_entry.children += 1
        child = mutate(
            parent_entry.spec,
            parent_entry.report,
            tasks,
            meta_provider,
            library_names=library_names,
        )
        child = _dedupe_name(child, archive)
        child_report = evaluate_closed_loop(
            child, tasks, world_model, agent_provider, judge, library=library, k=k
        )
        archive.add(ArchiveEntry(spec=child, report=child_report, generation=generation))
        if on_progress is not None:
            on_progress(generation, child.name, child_report.success_rate)

    best = archive.best()
    return EvolveResult(
        best=best.spec, best_score=best.fitness, archive=archive, generations=generations
    )


def _dedupe_name(spec: HarnessSpec, archive: HarnessArchive) -> HarnessSpec:
    """Ensure the child's name is unique in the archive (names key the lineage)."""
    existing = {e.spec.name for e in archive.entries}
    if spec.name not in existing:
        return spec
    i = 2
    while f"{spec.name}-{i}" in existing:
        i += 1
    return spec.model_copy(update={"name": f"{spec.name}-{i}"})


def save_archive(archive: HarnessArchive, path: str) -> None:
    """Persist the archive JSON (lineage + scores) for audit and resumption."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(archive.model_dump_json(indent=2), encoding="utf-8")


def load_archive(path: str) -> HarnessArchive:
    from pathlib import Path

    return HarnessArchive.model_validate_json(Path(path).read_text(encoding="utf-8"))

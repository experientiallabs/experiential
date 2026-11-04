"""Ongoing governance of the rule memory set: inspectors in cycles, merged like small PRs.

Each cycle runs every configured single-responsibility inspector
(:mod:`~exp.optimize.evaluation.rules.inspectors`) concurrently over the same snapshot of the set,
each under its own prompt variants. Their proposals are merged in registry order by
:func:`~exp.optimize.evaluation.rules.memory_edits.merge_proposals`: each applies atomically
against the digests of the files it patches, and a proposal whose file another proposal changed
first is rejected as a conflict and recorded. The coverage auditor's objectives go to System Two
(:mod:`~exp.optimize.evaluation.rules.system_two`) through the caller's research hook, and every
researched rule comes back as its own ``add`` proposal through the same merge. The fidelity
auditor reads the misses of a held-in replay of the current set.

Every cycle saves the merged set as a new version of the ``rule-memory-set`` artifact. Governance
stops early after a cycle in which no inspector proposed an edit or asked for research, and never
reads a held-out trace: the corpus and the replay are world-learning only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.agent_loop import AgentProfile, run_bounded
from exp.optimize.evaluation.rules.inspectors import (
    INSPECTORS,
    PERMISSIONS,
    RESEARCH_AUTHOR,
    FidelityReplay,
    InspectionDraw,
    InspectionMaterial,
    run_inspector,
)
from exp.optimize.evaluation.rules.memory_edits import (
    FilePatch,
    Proposal,
    ProposalRecord,
    merge_proposals,
)
from exp.optimize.evaluation.rules.system_one import Objective, RegionCorpus
from exp.optimize.evaluation.rules.system_two import TopicResearch
from exp.simulation.rules.memory_set import MemorySet, Passage, memory_set_digest, topic_slug
from exp.simulation.rules.model import ToolContract

ResearchHook = Callable[[tuple[Objective, ...], int], tuple[TopicResearch, ...]]
"""Answer objectives with System Two in a given research round."""

SaveVersion = Callable[[MemorySet, str, Sequence[ArtifactInput]], ArtifactInput]
"""Save one version of the set under an identity suffix, with its inputs."""


class GovernanceSettings(ContractModel):
    """Frozen governance bounds.

    Attributes:
        cycles: Most cycles; governance stops earlier when a cycle proposes nothing.
        inspectors: Inspectors run each cycle, in merge order.
        variants_per_inspector: Prompt variants per inspector per cycle.
        profile: Budgets of each inspector draw.
        research_objectives_per_cycle: Most coverage objectives researched per cycle.
    """

    cycles: int = Field(default=3, ge=1, le=8, strict=True)
    inspectors: tuple[str, ...] = tuple(INSPECTORS)
    variants_per_inspector: int = Field(default=1, ge=1, le=2, strict=True)
    profile: AgentProfile = AgentProfile()
    research_objectives_per_cycle: int = Field(default=4, ge=0, le=16, strict=True)

    @model_validator(mode="after")
    def _known(self) -> GovernanceSettings:
        """Require distinct, registered inspectors."""
        unknown = sorted(set(self.inspectors) - set(INSPECTORS))
        if unknown or len(set(self.inspectors)) != len(self.inspectors) or not self.inspectors:
            raise ValueError(f"inspectors must be distinct registered names; unknown: {unknown}")
        return self


class CycleRecord(ContractModel):
    """One governance cycle.

    Attributes:
        cycle: Cycle index, from zero.
        start_sha256: Set digest the inspectors read.
        end_sha256: Set digest after the merge.
        draws: Every inspector draw, failures included.
        objectives: Coverage objectives sent to System Two.
        research: System Two draws answering them.
        misses: Held-in fidelity misses the fidelity auditor read.
        proposals: Every proposal, merged or rejected, in merge order.
        version: The saved set after this cycle.
    """

    cycle: int = Field(ge=0)
    start_sha256: Sha256
    end_sha256: Sha256
    draws: tuple[InspectionDraw, ...]
    objectives: tuple[Objective, ...] = ()
    research: tuple[TopicResearch, ...] = ()
    misses: int = Field(default=0, ge=0)
    proposals: tuple[ProposalRecord, ...] = ()
    version: ArtifactInput


class GovernanceReport(ContractModel):
    """Every cycle of one governance run.

    Attributes:
        cycles: Cycles run, in order.
        stopped_early: Whether a cycle proposed nothing before the cycle limit.
        final: The last saved version.
    """

    cycles: tuple[CycleRecord, ...]
    stopped_early: bool
    final: ArtifactInput

    def merged(self) -> int:
        """Proposals merged across every cycle."""
        return sum(1 for c in self.cycles for p in c.proposals if p.status == "merged")


def research_proposals(research: Sequence[TopicResearch]) -> list[tuple[str, str, Proposal]]:
    """One ``add`` proposal per researched rule, authored by System Two."""
    proposals: list[tuple[str, str, Proposal]] = []
    for item in research:
        if item.answer is None:
            continue
        for draft in item.answer.rules:
            proposals.append(
                (
                    RESEARCH_AUTHOR,
                    item.variant,
                    Proposal(
                        title=f"Add {draft.kind} rule for {draft.tool or 'the environment'}",
                        rationale=draft.status_rationale or draft.reasoning[:4000],
                        citations=draft.evidence,
                        patches=(FilePatch(op="add", rule=draft),),
                    ),
                )
            )
    return proposals


def govern(
    project: ProjectStore,
    settings: AuthorSettings,
    governance: GovernanceSettings,
    memory: MemorySet,
    *,
    corpus: RegionCorpus,
    passages: Mapping[str, Passage],
    contracts: Mapping[str, ToolContract],
    identity: str,
    research: ResearchHook | None,
    fidelity: FidelityReplay | None,
    save: SaveVersion,
    concurrency: int,
    code_revision: str,
    sources: tuple[ArtifactInput, ...] = (),
) -> tuple[MemorySet, GovernanceReport]:
    """Run, or replay, every governance cycle over the set.

    Args:
        project: Project holding every draw and version.
        settings: Frozen inspector model and per-call allowance.
        governance: Cycle bounds and inspectors.
        memory: The seed set.
        corpus: World-learning regions.
        passages: Original passages by locator.
        contracts: Parsed tool contracts.
        identity: Run identity every draw and version derives from.
        research: System Two hook for coverage objectives, or None to skip research.
        fidelity: Held-in replay for the fidelity auditor, or None for no misses.
        save: Saves one version of the set.
        concurrency: Most inspector draws in flight at once.
        code_revision: Producing source identity.
        sources: Saved artifacts the traces came from.

    Returns:
        The governed set and the report.

    Raises:
        FileLockTimeout: Another process holds a cycle's lock.
    """
    cycles: list[CycleRecord] = []
    stopped_early = False
    version = save(memory, "v0", ())
    for cycle in range(governance.cycles):
        start = memory_set_digest(memory)
        misses = () if fidelity is None else fidelity.misses(memory)
        material = InspectionMaterial(corpus, passages, contracts, misses)
        calls = [
            partial(
                run_inspector,
                project,
                settings,
                governance.profile,
                INSPECTORS[name],
                variant,
                memory,
                material,
                identity=f"{identity}-c{cycle}-{start[:16]}-{name}-{variant}",
                cycle=cycle,
                code_revision=code_revision,
                sources=sources,
            )
            for name in governance.inspectors
            for variant in tuple(INSPECTORS[name].variants)[: governance.variants_per_inspector]
        ]
        with CoordinationLock(
            project.artifacts.project_directory / f"{identity}-c{cycle}.lock", timeout=0
        ):
            draws: tuple[InspectionDraw, ...] = run_bounded(calls, concurrency)
        proposals = [
            (draw.inspector, draw.variant, proposal)
            for draw in draws
            if draw.answer is not None
            for proposal in draw.answer.proposals
        ]
        objectives = tuple(
            dict.fromkeys(
                Objective(
                    topic=topic_slug(request.tool),
                    tool=request.tool,
                    question=request.question,
                    origin="coverage",
                )
                for draw in draws
                if draw.answer is not None
                for request in draw.answer.objectives
            )
        )[: governance.research_objectives_per_cycle]
        researched = () if not objectives or research is None else research(objectives, cycle + 1)
        memory, records = merge_proposals(
            memory,
            [*proposals, *research_proposals(researched)],
            passages,
            cycle=cycle,
            permissions=PERMISSIONS,
        )
        inputs = [d.transcript for d in draws if d.transcript is not None]
        inputs += [r.transcript for r in researched if r.transcript is not None]
        version = save(memory, f"v{cycle + 1}", inputs)
        cycles.append(
            CycleRecord(
                cycle=cycle,
                start_sha256=start,
                end_sha256=memory_set_digest(memory),
                draws=draws,
                objectives=objectives,
                research=researched,
                misses=len(misses),
                proposals=records,
                version=version,
            )
        )
        if not proposals and not objectives:
            stopped_early = cycle + 1 < governance.cycles
            break
    return memory, GovernanceReport(
        cycles=tuple(cycles), stopped_early=stopped_early, final=version
    )

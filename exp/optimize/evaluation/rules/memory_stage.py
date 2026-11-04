"""The ``jev_memory`` rules profile: System Two plus System One research into a governed memory set.

One run, over world-learning traces only:

1. **Corpus.** Every recorded step becomes a region with its locator, trace and step index.
2. **Research.** For every topic (each tool with a contract, plus the environment) System Two
   draws plan objectives and System One answers them with parallel Jev judgments, returning the
   original passages (:mod:`~exp.optimize.evaluation.rules.system_two`,
   :mod:`~exp.optimize.evaluation.rules.system_one`). Leads System One discovered are followed in
   a second research round. Every researched rule becomes its own ``add`` proposal, merged into
   the seed set like a small PR.
3. **Weighing.** Each rule is weighed by the existing agentic weigher
   (:mod:`~exp.optimize.evaluation.rules.weighing`); its status and rationale are merged into the
   rule file as one ``patch`` proposal each.
4. **Governance.** Single-responsibility inspectors run in cycles
   (:mod:`~exp.optimize.evaluation.rules.governance`), each cycle saved as a new version.

The governed set is saved as the final ``rule-memory-set`` version and also turned into a
:class:`~exp.simulation.rules.model.RuleBook`, so the rule book checker and the boundary
challenges keep working unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, sha256_json, stable_id
from exp.common.project import ProjectStore
from exp.common.traces import Trace
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent, decision_cost
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.agent_loop import AGENT_LOOP_PROTOCOL, AgentProfile
from exp.optimize.evaluation.rules.governance import (
    GovernanceReport,
    GovernanceSettings,
    govern,
    research_proposals,
)
from exp.optimize.evaluation.rules.inspectors import (
    PERMISSIONS,
    FidelityReplay,
    PredicateReplay,
    inspector_protocol_sha256,
)
from exp.optimize.evaluation.rules.memory_edits import (
    FilePatch,
    PatchField,
    PatchOp,
    Proposal,
    ProposalRecord,
    RuleFields,
    merge_proposals,
)
from exp.optimize.evaluation.rules.research import ResearchMaterial
from exp.optimize.evaluation.rules.system_one import (
    DecisionBoundary,
    Objective,
    ReadingPolicy,
    RegionCorpus,
    SystemOne,
)
from exp.optimize.evaluation.rules.system_two import (
    MAXIMUM_TOPIC_RULES,
    PLANNER_VARIANTS,
    TopicResearch,
    research_topics,
    system_two_protocol_sha256,
)
from exp.optimize.evaluation.rules.weighing import (
    WeighingDraw,
    weigh_rules_in_parallel,
    weighing_protocol_sha256,
)
from exp.simulation.rules.memory_set import (
    CoverageGap,
    MemorySet,
    memory_set_digest,
    rule_sha256,
    save_memory_set,
    topic_slug,
)
from exp.simulation.rules.model import ResearchQuestion, Rule, RuleBook, ToolContract

MEMORY_STAGE_PROTOCOL = "jev-memory-stage-v2"
"""Version of the stage sequence and seed, weighing and gap bookkeeping."""

WEIGHER = "weigher"
"""Author name of status proposals from the weighing."""

_WEIGHER: tuple[frozenset[PatchOp], frozenset[PatchField]] = (
    frozenset(("patch",)),
    frozenset(("status", "status_rationale")),
)


class MemorySetSettings(ContractModel):
    """Registration of the ``jev_memory`` profile.

    Attributes:
        decision_alias: Catalog alias of the System One decision model (priced in the plan).
        reading: System One's adaptive reading bounds.
        planner: Budgets of each System Two draw.
        planner_variants: Planning stances.
        draws_per_topic: Stances run per topic.
        follow_leads: Whether discovered leads get a second research round.
        maximum_topics: Most topics (tools plus the environment); more are refused.
        governance: Inspector cycles.
    """

    decision_alias: str = Field(min_length=1)
    reading: ReadingPolicy = ReadingPolicy()
    planner: AgentProfile = AgentProfile()
    planner_variants: tuple[str, ...] = tuple(PLANNER_VARIANTS)
    draws_per_topic: int = Field(default=1, ge=1, le=3, strict=True)
    follow_leads: bool = True
    maximum_topics: int = Field(default=32, ge=1, le=256, strict=True)
    governance: GovernanceSettings = GovernanceSettings()

    def protocol(self) -> str:
        """Bind every protocol the profile runs under."""
        return sha256_json(
            {
                "stage": MEMORY_STAGE_PROTOCOL,
                "system_one": self.reading.identity,
                "system_two": system_two_protocol_sha256(),
                "inspectors": inspector_protocol_sha256(),
                "weighing": weighing_protocol_sha256(),
                "loop": AGENT_LOOP_PROTOCOL,
            }
        )


class ResearchCoverage(ContractModel):
    """Coverage manifest of the research: every planned topic and objective counted once.

    Attributes:
        planned_topics: Topics planned.
        researched_topics: Topics with at least one usable draw.
        failed_draws: Draw identity mapped to why it failed.
        objectives: Objectives System One answered.
        regions_judged: Regions judged across objectives.
        passages_returned: Original passages returned.
        leads: Leads discovered.
        decisions_bought: Decisions this run paid for.
        decisions_replayed: Decisions replayed from the cache.
        decision_usd: Reported cost of the bought decisions.
    """

    planned_topics: int = Field(ge=0)
    researched_topics: int = Field(ge=0)
    failed_draws: dict[str, str] = Field(default_factory=dict)
    objectives: int = Field(default=0, ge=0)
    regions_judged: int = Field(default=0, ge=0)
    passages_returned: int = Field(default=0, ge=0)
    leads: int = Field(default=0, ge=0)
    decisions_bought: int = Field(default=0, ge=0)
    decisions_replayed: int = Field(default=0, ge=0)
    decision_usd: float = Field(default=0.0, ge=0.0)


class MemoryStageResult(ContractModel):
    """Everything the profile produced.

    Attributes:
        memory_set: The final saved memory set.
        research: Every System Two draw, both rounds.
        seed: Proposals that built and weighed the seed set.
        weighings: Every rule's weighing.
        governance: Every governance cycle.
        coverage: The research coverage manifest.
    """

    memory_set: ArtifactInput
    research: tuple[TopicResearch, ...]
    seed: tuple[ProposalRecord, ...]
    weighings: tuple[WeighingDraw, ...]
    governance: GovernanceReport
    coverage: ResearchCoverage

    def transcripts(self) -> tuple[ArtifactInput, ...]:
        """Every saved transcript the set was learned, weighed and governed by."""
        found = [r.transcript for r in self.research if r.transcript is not None]
        found += [w.transcript for w in self.weighings if w.transcript is not None]
        for cycle in self.governance.cycles:
            found += [d.transcript for d in cycle.draws if d.transcript is not None]
            found += [r.transcript for r in cycle.research if r.transcript is not None]
        return tuple(dict.fromkeys(found))


def topics_of(
    contracts: Mapping[str, ToolContract], limit: int
) -> tuple[tuple[str, str | None], ...]:
    """Every tool topic plus the environment topic, refusing more than ``limit``.

    Raises:
        ValueError: There are more topics than the registered limit.
    """
    topics = (*((topic_slug(tool), tool) for tool in sorted(contracts)), (topic_slug(None), None))
    if len(topics) > limit:
        raise ValueError(
            f"{len(topics)} research topics exceed memory.maximum_topics={limit}; raise the cap"
        )
    return topics


def coverage_gaps(
    memory: MemorySet, contracts: Mapping[str, ToolContract]
) -> tuple[CoverageGap, ...]:
    """Host-derived gaps: tools without rules, failure forms without preconditions."""
    ruled = {rule.tool for rule in memory.rules}
    gated = {rule.tool for rule in memory.rules if rule.kind == "precondition"}
    gaps = [
        CoverageGap(kind="tool", subject=tool, note="no rule describes this tool")
        for tool in memory.tools
        if tool not in ruled
    ]
    gaps += [
        CoverageGap(
            kind="transition", subject=f"{name}: {form}", note="no precondition explains it"
        )
        for name, contract in sorted(contracts.items())
        if name not in gated
        for form in contract.error_forms
    ]
    return tuple(gaps)


def memory_to_rulebook(memory: MemorySet, contracts: Mapping[str, ToolContract]) -> RuleBook:
    """The set as a rule book, so the checker and the challenges read it unchanged."""
    rules = tuple(
        Rule(
            rule_id=rule.rule_id,
            tool=rule.tool,
            kind=rule.kind,
            statement=rule.truism,
            predicate=rule.predicate,
            status=rule.status,
            support=len({p.trace_id for p in rule.evidence}),
            contradiction=len({p.trace_id for p in rule.counter_evidence}),
            evidence=tuple(p.locator for p in rule.evidence),
            counter_evidence=tuple(p.locator for p in rule.counter_evidence),
        )
        for rule in memory.rules
    )
    questions = tuple(
        ResearchQuestion(
            tool=rule.tool,
            question=question[:600],
            why_it_matters=f"Open question of rule {rule.rule_id}: {rule.truism}"[:600],
            evidence=tuple(p.locator for p in rule.evidence),
        )
        for rule in memory.rules
        for question in rule.open_questions
    )
    return RuleBook(
        contracts=dict(contracts), rules=rules, questions=questions, sources=memory.sources
    )


def build_memory_set(
    project: ProjectStore,
    author: AuthorSettings,
    weighing: AgentProfile,
    settings: MemorySetSettings,
    traces: Sequence[Trace],
    contracts: Mapping[str, ToolContract],
    *,
    boundary: DecisionBoundary,
    identity: str,
    excluded: frozenset[str],
    concurrency: int,
    code_revision: str,
    sources: tuple[ArtifactInput, ...] = (),
    fidelity: FidelityReplay | None = None,
) -> tuple[MemoryStageResult, MemorySet, RuleBook]:
    """Run, or replay, research, weighing and governance into the final memory set.

    Args:
        project: Project holding every saved step.
        author: Frozen model shared by System Two, the weigher and the inspectors.
        weighing: Budgets of each weighing draw.
        settings: The profile's registration.
        traces: World-learning traces only.
        contracts: Parsed tool contracts.
        boundary: The System One decisions client.
        identity: Run identity every step derives from.
        excluded: Traces no citation may reach (every trace not reserved for world learning).
        concurrency: Most agentic draws in flight at once.
        code_revision: Producing source identity.
        sources: Saved artifacts the traces came from.
        fidelity: Held-in replay; the call-free predicate replay when None.

    Returns:
        The result record, the final set and its rule book.

    Raises:
        ValueError: A trace is excluded, or there are more topics than the cap.
    """
    reached = sorted({trace.trace_id for trace in traces} & excluded)
    if reached:
        raise ValueError(f"the memory set learns from world-learning traces only: {reached}")
    corpus = RegionCorpus(traces)
    passages = {locator: region.passage() for locator, region in corpus.by_locator.items()}
    topics = topics_of(contracts, settings.maximum_topics)
    system_one = SystemOne(
        project,
        boundary,
        corpus,
        settings.reading,
        run_identity=identity,
        code_revision=code_revision,
    )

    def research(
        units: Sequence[tuple[str, str | None, tuple[Objective, ...], tuple[str, ...]]], round_: int
    ) -> tuple[TopicResearch, ...]:
        """Run one research round over the given topics."""
        return research_topics(
            project,
            author,
            settings.planner,
            system_one,
            contracts,
            units,
            identity=identity,
            variants=settings.planner_variants,
            draws_per_topic=settings.draws_per_topic,
            round_=round_,
            concurrency=concurrency,
            code_revision=code_revision,
            sources=sources,
        )

    found = research([(topic, tool, (), ()) for topic, tool in topics], 0)
    if settings.follow_leads:
        leads: dict[str, list[Objective]] = {}
        for item in found:
            for lead in item.leads:
                leads.setdefault(lead.topic, []).append(lead)
        tool_of = dict(topics)
        found += research(
            [
                (topic, tool_of.get(topic), tuple(dict.fromkeys(rows))[:4], ())
                for topic, rows in sorted(leads.items())
                if topic in tool_of
            ],
            1,
        )
    seed = MemorySet(tools=tuple(sorted(contracts)), sources=corpus.trace_ids)
    seed, built = merge_proposals(
        seed, research_proposals(found), passages, cycle=-1, permissions=PERMISSIONS
    )
    book = memory_to_rulebook(seed, contracts)
    weighed = weigh_rules_in_parallel(
        project,
        author,
        weighing,
        book,
        ResearchMaterial(traces, contracts, maximum_bytes=weighing.maximum_result_bytes),
        {},
        identity=stable_id("memory-weighing", {"run": identity, "seed": memory_set_digest(seed)}),
        excluded=excluded,
        code_revision=code_revision,
        concurrency=concurrency,
        sources=sources,
    )
    statuses: list[tuple[str, str, Proposal]] = []
    for draw in weighed:
        rule = None if draw.weighing is None else seed.rule(draw.weighing.rule_id)
        if rule is None or draw.weighing is None:
            continue
        statuses.append(
            (
                WEIGHER,
                "",
                Proposal(
                    title=f"Weigh {rule.rule_id}: {draw.weighing.status}",
                    rationale=draw.weighing.rationale,
                    citations=(*draw.weighing.supporting, *draw.weighing.contradicting)
                    or (rule.rule_id,),
                    patches=(
                        FilePatch(
                            op="patch",
                            path=rule.path,
                            base_sha256=rule_sha256(rule),
                            fields=RuleFields(
                                status=draw.weighing.status,
                                status_rationale=draw.weighing.rationale,
                            ),
                        ),
                    ),
                ),
            )
        )
    seed, weighed_records = merge_proposals(
        seed, statuses, passages, cycle=-1, permissions={**PERMISSIONS, WEIGHER: _WEIGHER}
    )

    def save(memory: MemorySet, suffix: str, inputs: Sequence[ArtifactInput]) -> ArtifactInput:
        """Save one version with host-derived gaps."""
        memory = memory.model_copy(update={"gaps": coverage_gaps(memory, contracts)})
        return save_memory_set(
            project.artifacts,
            memory,
            artifact_id=f"{identity}-memory-{suffix}-{memory_set_digest(memory)[:16]}",
            inputs=(*inputs, *sources),
            code_revision=code_revision,
        )

    def follow(objectives: tuple[Objective, ...], round_: int) -> tuple[TopicResearch, ...]:
        """Answer coverage objectives with System Two."""
        grouped: dict[tuple[str, str | None], list[Objective]] = {}
        for objective in objectives:
            grouped.setdefault((objective.topic, objective.tool), []).append(objective)
        return research(
            [(t, tool, tuple(rows), ()) for (t, tool), rows in grouped.items()], round_ + 1
        )

    governed, report = govern(
        project,
        author,
        settings.governance,
        seed,
        corpus=corpus,
        passages=passages,
        contracts=contracts,
        identity=identity,
        research=follow,
        fidelity=fidelity if fidelity is not None else PredicateReplay(corpus),
        save=save,
        concurrency=concurrency,
        code_revision=code_revision,
        sources=sources,
    )
    final = governed.model_copy(update={"gaps": coverage_gaps(governed, contracts)})
    result = MemoryStageResult(
        memory_set=report.final,
        research=found,
        seed=(*built, *weighed_records),
        weighings=weighed,
        governance=report,
        coverage=ResearchCoverage(
            planned_topics=len(topics),
            researched_topics=len({r.topic for r in found if r.answer is not None}),
            failed_draws={r.identity: r.failure for r in found if r.failure is not None},
            objectives=sum(len(r.objectives) for r in found),
            regions_judged=sum(r.regions_judged for r in found),
            passages_returned=sum(r.passages_returned for r in found),
            leads=sum(len(r.leads) for r in found),
            decisions_bought=sum(r.decisions_bought for r in found),
            decisions_replayed=sum(r.decisions_replayed for r in found),
            decision_usd=sum(r.decision_usd for r in found),
        ),
    )
    return result, final, memory_to_rulebook(final, contracts)


def memory_stage_costs(
    project: ProjectStore,
    author: AuthorSettings,
    weighing: AgentProfile,
    settings: MemorySetSettings,
    topics: int,
) -> tuple[tuple[CostComponent, ...], str]:
    """Price the profile's upper bound before any call.

    System Two draws: topics times stances, twice when leads are followed, plus one round per
    governance cycle for coverage objectives. Each draw asks System One at most one objective
    per tool call, and each objective judges at most ``reading.max_regions`` regions with one
    decision each. Inspectors: cycles times inspectors times variants. Each seed research draw
    admits at most MAXIMUM_TOPIC_RULES drafts, each weighed once before governance. This finite
    bound prices every possible weighing, including the author transport attempt allowance.

    Args:
        project: Project whose catalog prices the aliases.
        author: Frozen author settings.
        weighing: Weighing budgets.
        settings: The profile's registration.
        topics: Topics researched, at most (tools plus the environment).

    Returns:
        Cost components and one line describing them.
    """
    per_call = evaluation_costs(project, author)
    rounds = 2 if settings.follow_leads else 1
    planner_draws = topics * settings.draws_per_topic * rounds
    governance = settings.governance
    coverage_draws = (
        governance.cycles * governance.research_objectives_per_cycle * settings.draws_per_topic
    )
    draws = planner_draws + coverage_draws
    planner_calls = draws * settings.planner.budget.model_calls
    inspector_calls = (
        governance.cycles
        * len(governance.inspectors)
        * governance.variants_per_inspector
        * governance.profile.budget.model_calls
    )
    decisions = draws * settings.planner.budget.tool_calls * settings.reading.max_regions
    components = [row.scaled(planner_calls + inspector_calls) for row in per_call]
    components.append(decision_cost(project, settings.decision_alias, decisions))
    maximum_rules = planner_draws * MAXIMUM_TOPIC_RULES
    weighing_calls = maximum_rules * weighing.budget.model_calls
    components.extend(
        CostComponent(
            f"{row.name} weighing (up to {maximum_rules} rules)",
            row.attempts * weighing_calls,
            None if row.usd is None else row.usd * weighing_calls,
        )
        for row in per_call
    )
    line = (
        f"{draws} System Two draws (up to {planner_calls} calls) over {topics} topics, up to "
        f"{decisions} System One decisions with {settings.decision_alias}, "
        f"{governance.cycles} governance cycles of {len(governance.inspectors)} inspectors (up "
        f"to {inspector_calls} calls) and up to {maximum_rules} rule weighings "
        f"({weighing_calls} calls) with {author.model_alias}"
    )
    return tuple(components), line

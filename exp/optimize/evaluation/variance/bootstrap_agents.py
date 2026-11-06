"""The agentic roles of the start bootstrap: author, reviewers, researchers and reveal checkers.

Every role is a bounded draw of the shared agentic loop
(:func:`~exp.optimize.evaluation.rules.agent_loop.run_agent`): it plans, reads the frozen
:class:`StartMaterial` with read-only tools, drafts, self-checks and answers through its typed
answer tool. Prompts and stances come from
:mod:`~exp.optimize.evaluation.variance.bootstrap_prompts`.

Code only checks structure: a draft's manifest, tools, state, pointers and verifier agree; every
cited reference is in the material inventory; an observed fact cites recorded evidence and an
inferred fact cites support; the claimed provenance is one the evidence allows. Whether a start is
coherent, sufficiently specified and supports its task, what a missing fact is, and whether a
reveal contradicts a fact are the roles' judgments. A draw whose answer fails its checks is kept
with its failure and never redrawn; a saved draw replays without a call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property

from pydantic import JsonValue

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, sha256_json
from exp.common.core.artifacts import stable_id as _stable_id
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.common.tasks import ToolSchema
from exp.common.traces import Trace
from exp.optimize.evaluation.executable.grounded import RuleMemoryBinding
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.agent_loop import (
    AgentProfile,
    ToolStatus,
    bounded,
    reason,
    run_agent,
    run_bounded,
)
from exp.optimize.evaluation.rules.locator_inventory import encode_inventory
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    BootstrapSpec,
    BootstrapTask,
    Finding,
    Provenance,
    ResearchItem,
    ResearchRequest,
    Reveal,
    RevealAnswer,
    ReviewAnswer,
    RoleDraw,
    StartDraft,
    StartTool,
    Verdict,
    domain_state_roots,
)
from exp.optimize.evaluation.variance.bootstrap_prompts import (
    AUTHOR_INSTRUCTION,
    AUTHOR_VARIANTS,
    READ_TOOLS,
    RESEARCH_ANSWER,
    RESEARCH_INSTRUCTION,
    RESEARCH_VARIANTS,
    REVEAL_INSTRUCTION,
    REVEAL_VARIANTS,
    REVIEW_INSTRUCTION,
    REVIEW_VARIANTS,
    author_answer_tool,
    bootstrap_prompts_sha256,
    instruction,
    reveal_answer_tool,
    review_answer_tool,
)
from exp.optimize.evaluation.variance.bootstrap_reads import read_trace_steps, read_whole_trace
from exp.simulation.evaluation.tasks import RuleVerifierDraft, grade
from exp.simulation.reconstruction.models import resolve
from exp.simulation.rules.contracts import trajectory_view
from exp.simulation.rules.model import RuleBook, ToolContract

RECORDED_STATE_REF = "recorded-state"
"""Inventory reference of the recorded environment state snapshot."""


@dataclass(frozen=True)
class FamilyKnowledge:
    """A family's reconciled knowledge: tool contracts, shared domains, rules and research.

    Attributes:
        family_id: Environment family.
        report: Saved rules-stage report, when one is known.
        rulebook: Saved rule book.
        book: The rule book: contracts, weighed rules and open questions.
        rule_memory: Rule memory set and recall policy every start's runtime recalls from.
        traces: The family's world-learning traces by identity.
        contracts: Reconciled tool contracts: the family's own over the rule book's.
        state_domains: Names of the family's shared state domains; empty when it declares none.
        research_memory: Native rules/research document, separate from governed memory sets.
    """

    family_id: str
    report: ArtifactInput | None
    rulebook: ArtifactInput
    book: RuleBook
    rule_memory: RuleMemoryBinding | None
    traces: Mapping[str, Trace]
    contracts: Mapping[str, ToolContract]
    state_domains: tuple[str, ...] = ()
    research_memory: ArtifactInput | None = None


@dataclass(frozen=True)
class StartMaterial:
    """The frozen read-only material every role of one task reads.

    Attributes:
        knowledge: The family's learned knowledge.
        task: The selected task.
        sources: The task's recorded source traces by identity.
        recorded_state: The recorded environment state, when one was saved.
        findings: Research findings made so far.
    """

    knowledge: FamilyKnowledge
    task: BootstrapTask
    sources: Mapping[str, Trace]
    recorded_state: JsonObject | None = None
    findings: tuple[Finding, ...] = field(default=())

    @cached_property
    def views(self) -> dict[str, tuple[JsonObject, frozenset[str]]]:
        """Each readable trace's step view and locators."""
        traces = {**self.knowledge.traces, **self.sources}
        return {trace_id: trajectory_view(trace) for trace_id, trace in traces.items()}

    @cached_property
    def inventory(self) -> frozenset[str]:
        """Every reference a role may cite."""
        book = self.knowledge.book
        refs = {locator for _, locators in self.views.values() for locator in locators}
        refs |= {f"rule:{rule.rule_id}" for rule in book.rules}
        refs |= {f"contract:{name}" for name in self.knowledge.contracts}
        refs |= {f"finding:{row.finding_id}" for row in self.findings}
        if self.recorded_state is not None:
            refs.add(RECORDED_STATE_REF)
        return frozenset(refs)

    def with_findings(self, findings: tuple[Finding, ...]) -> StartMaterial:
        """Return the material with more research findings."""
        return StartMaterial(self.knowledge, self.task, self.sources, self.recorded_state, findings)

    def allowed(self) -> tuple[Provenance, ...]:
        """Return the provenances the evidence allows, strongest first."""
        rows: list[Provenance] = []
        if self.recorded_state is not None:
            rows.append("observed")
        if any(self.recorded_prefix(trace_id) for trace_id in self.sources):
            rows.append("reconstructed")
        return (*rows, "authored")

    def recorded_prefix(self, trace_id: str) -> tuple[JsonObject, ...]:
        """Return a source trace's leading steps that recorded an observation or a failure."""
        view, _ = self.views[trace_id]
        steps = view["steps"]
        assert isinstance(steps, list)
        rows: list[JsonObject] = []
        for step in steps:
            assert isinstance(step, dict)
            if step.get("observation_locator") is None and step.get("failure") is None:
                break
            rows.append(step)
        return tuple(rows)

    def context(self) -> JsonObject:
        """List source evidence for explicit reads without duplicating complete trace bodies."""
        source_traces: list[JsonObject] = []
        for trace_id in sorted(self.sources):
            steps = self.views[trace_id][0]["steps"]
            assert isinstance(steps, list)
            source_traces.append(
                {"trace_id": trace_id, "total_steps": len(steps), "content_delivered": False}
            )
        task = self.task.task
        return {
            "task": {
                "task_id": task.task_id,
                "instruction": task.instruction,
                "tools": [tool.model_dump(mode="json") for tool in task.tools],
            },
            "family": {
                "family_id": self.knowledge.family_id,
                "member_traces": sorted(self.knowledge.traces),
                "tools": sorted(self.knowledge.contracts),
                "state_domains": list(self.knowledge.state_domains),
                "domain_state_roots": domain_state_roots(self.knowledge.state_domains),
                "rules": len(self.knowledge.book.rules),
                "open_questions": [
                    row.model_dump(mode="json") for row in self.knowledge.book.questions
                ],
            },
            "source_traces": source_traces,
            "recorded_state_available": self.recorded_state is not None,
            "allowed_provenance": list(self.allowed()),
            "findings": [_finding_view(row) for row in self.findings],
            "inventory": encode_inventory(self.inventory),
        }

    def run(
        self, call: ToolCall, maximum_bytes: int
    ) -> tuple[ToolStatus, JsonValue, tuple[str, ...]]:
        """Answer one read-only call: status, result and the references it served."""
        arguments = call.arguments
        value: JsonValue
        served: tuple[str, ...]
        if call.name == "read_trace":
            return read_whole_trace(self.views, arguments, maximum_bytes)
        elif call.name == "read_trace_steps":
            return read_trace_steps(self.views, arguments, maximum_bytes)
        elif call.name == "read_contract":
            tool = str(arguments.get("tool", ""))
            contract = self.knowledge.contracts.get(tool)
            if contract is None:
                return "refused", f"no contract for tool {tool!r}", ()
            value, served = contract.model_dump(mode="json"), (f"contract:{tool}",)
        elif call.name == "read_rules":
            tool = arguments.get("tool")
            rules = [
                rule for rule in self.knowledge.book.rules if tool is None or rule.tool == tool
            ]
            value = [
                {
                    "reference": f"rule:{rule.rule_id}",
                    "tool": rule.tool,
                    "kind": rule.kind,
                    "statement": rule.statement,
                    "status": rule.status,
                    "evidence": list(rule.evidence),
                }
                for rule in rules
            ]
            served = tuple(f"rule:{rule.rule_id}" for rule in rules)
        elif call.name == "read_findings":
            value = [_finding_view(row) for row in self.findings]
            served = tuple(f"finding:{row.finding_id}" for row in self.findings)
        elif call.name == "read_recorded_state":
            if self.recorded_state is None:
                return "refused", "no recorded environment state was saved for this task", ()
            value, served = self.recorded_state, (RECORDED_STATE_REF,)
        else:
            return "refused", f"unknown tool {call.name!r}", ()
        status, result = bounded(value, maximum_bytes)
        return status, result, served if status == "ok" else ()


def assemble_start_proposal(answer: JsonObject) -> StartDraft:
    """Expand one authored declaration into the canonical executable start contract."""
    if "manifest" in answer:
        raise ValueError("manifest is host-derived; declare schema, reads and writes per tool")
    proposal = dict(answer)
    declarations = proposal.pop("tools", None)
    if not isinstance(declarations, list):
        raise ValueError("tools must contain schema-and-effects declarations")
    tools = tuple(StartTool.model_validate(tool) for tool in declarations)
    state = proposal.get("initial_state")
    if not isinstance(state, dict):
        raise ValueError("initial_state must map each domain name to its state")
    proposal["tools"] = [
        tool.model_dump(mode="json", exclude={"reads", "writes"}) for tool in tools
    ]
    proposal["manifest"] = {
        "domains": [{"name": name, "owner": "simulated"} for name in sorted(state)],
        "tools": [
            {"name": tool.name, "reads": list(tool.reads), "writes": list(tool.writes)}
            for tool in tools
        ],
    }
    proposal["verifier"] = (
        RuleVerifierDraft.model_validate(proposal.get("verifier")).compile().model_dump(mode="json")
    )
    return StartDraft.model_validate(proposal)


def validate_author_answer(
    answer: JsonObject, material: StartMaterial, read: frozenset[str]
) -> StartDraft:
    """Expand the proposal and verify its state, capabilities and delivered fact evidence."""
    draft = assemble_start_proposal(answer)
    return validate_draft(draft.model_dump(mode="json"), material, read)


def validate_draft(answer: JsonObject, material: StartMaterial, read: frozenset[str]) -> StartDraft:
    """Check a draft's structure, delivered citations and claimed provenance.

    Args:
        answer: The author's answer arguments.
        material: What the author could cite.

    Returns:
        The draft.

    Raises:
        ValueError: Any structural check fails.
    """
    draft = StartDraft.model_validate(answer)
    names = [tool.name for tool in draft.tools]
    declared = {tool.name: tool for tool in draft.manifest.tools}
    if len(set(names)) != len(names) or set(names) != set(declared):
        raise ValueError("the tool schemas and the manifest must name exactly the same tools")
    domains = {domain.name for domain in draft.manifest.domains}
    if any(domain.owner != "simulated" for domain in draft.manifest.domains):
        raise ValueError("every domain of a bootstrapped start is simulated")
    if set(draft.initial_state) != domains:
        raise ValueError("the initial state must name exactly the manifest's domains")
    shared = set(material.knowledge.state_domains)
    if shared and domains != shared:
        raise ValueError(f"a start declares exactly the family's shared domains {sorted(shared)}")
    pointers = [
        *((f"probe_paths[{key!r}]", value) for key, value in draft.probe_paths.items()),
        *(
            (f"fact[{fact.fact_id!r}].state_paths", path)
            for fact in draft.facts
            for path in fact.state_paths
        ),
    ]
    unresolved: list[str] = []
    for location, pointer in pointers:
        try:
            resolve(draft.initial_state, pointer)
        except ValueError:
            unresolved.append(f"{location}={pointer!r}")
    issues: list[str] = []
    if unresolved:
        roots = domain_state_roots(tuple(draft.initial_state))
        issues.append(
            f"Unresolved initial_state pointers: {'; '.join(unresolved)}. "
            f"Valid domain root pointers: {roots}"
        )
    allowed = draft.verifier.allowed_tools
    if allowed is not None and not set(allowed) <= set(names):
        raise ValueError("the verifier allows tools the start does not offer")
    opening_refs = {f"finding:{row.finding_id}" for row in material.findings}
    try:
        _check_facts(draft, material.inventory & (read | opening_refs))
    except ValueError as error:
        issues.append(str(error))
    if issues:
        raise ValueError("\n".join(issues))
    _check_verifier_projection(draft)
    if draft.provenance not in material.allowed():
        raise ValueError(
            f"provenance {draft.provenance!r} is not supported by the evidence; allowed: "
            f"{list(material.allowed())}"
        )
    if draft.provenance == "observed" and draft.initial_state != material.recorded_state:
        raise ValueError("an observed start holds the recorded state unchanged")
    if draft.provenance == "reconstructed":
        if draft.replay_trace not in material.sources:
            raise ValueError("a reconstructed start names one of the task's source traces")
        recorded = len(material.recorded_prefix(draft.replay_trace))
        if not 1 <= draft.prefix_steps <= recorded:
            raise ValueError(
                f"a reconstructed start replays 1 to {recorded} recorded steps of "
                f"{draft.replay_trace}"
            )
        required: set[str] = set()
        for step in material.recorded_prefix(draft.replay_trace)[: draft.prefix_steps]:
            action = step["action"]
            assert isinstance(action, dict)
            required.add(str(action["name"]))
        if missing := sorted(required - set(names)):
            raise ValueError(
                f"a reconstructed start must declare its recorded prefix tools: {missing}; "
                "their schemas are available in task.tools"
            )
    if draft.provenance == "authored" and not draft.reveal_probes:
        raise ValueError("an authored start names reveal probes for its first-reveal check")
    for probe in draft.reveal_probes:
        policy = declared.get(probe.name)
        if policy is None or policy.writes:
            raise ValueError(f"reveal probe {probe.name!r} is not a declared read-only tool")
    return draft


def _check_verifier_projection(draft: StartDraft) -> None:
    """Require predicates to address produced outcome fields and compatible state probes."""
    roots = {"initial_state", "final_state", "final_answer", "steps", "stop_reason"}
    verifier = draft.verifier
    required = verifier.outcome_schema.get("required", [])
    if isinstance(required, list):
        unsupported = {name for name in required if isinstance(name, str)} - roots
        if unsupported:
            raise ValueError(
                f"the workflow outcome does not produce {sorted(unsupported)}; "
                "put task-specific evidence under final_state through probe_paths"
            )
    paths = list(verifier.required_paths)
    paths.extend(rule.path for rule in verifier.rules)
    for relation in verifier.relations:
        paths.extend((relation.left, relation.right))
    for relation in verifier.table_relations:
        paths.extend((relation.source, relation.target))
    for pointer in paths:
        if not pointer:
            continue
        parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]
        if not parts or parts[0] not in roots:
            raise ValueError(
                f"{pointer!r} is not a produced workflow outcome path; "
                "put task-specific evidence under /final_state/<probe-key>"
            )
        if parts[0] not in {"initial_state", "final_state"} or len(parts) < 2:
            continue
        alias = parts[1]
        if alias not in draft.probe_paths:
            raise ValueError(f"verifier path {pointer!r} names unknown probe key {alias!r}")
        value = resolve(draft.initial_state, draft.probe_paths[alias])
        if len(parts) > 2 and not isinstance(value, (dict, list)):
            raise ValueError(
                f"verifier path {pointer!r} descends into scalar probe {alias!r}; "
                "project its full mutable state domain instead of a note or marker"
            )

    if draft.provenance == "authored":
        projection = {
            name: resolve(draft.initial_state, pointer)
            for name, pointer in draft.probe_paths.items()
        }
        control: JsonObject = {
            "initial_state": projection,
            "final_state": projection,
            "steps": [],
            "final_answer": "",
            "stop_reason": "completed",
        }
        if grade(verifier, control).status == "pass":
            raise ValueError(
                "the authored start passes a no-action control; leave the task unfinished "
                "and make the verifier require the actor's actual result"
            )


def validate_review(
    answer: JsonObject, draft: StartDraft, inventory: frozenset[str]
) -> ReviewAnswer:
    """Check a review's citations and fact references.

    Raises:
        ValueError: A citation is outside the inventory or a fact is not the draft's.
    """
    review = ReviewAnswer.model_validate(answer)
    facts = {fact.fact_id for fact in draft.facts}
    cited = {ref for row in review.findings for ref in row.references}
    if outside := sorted(cited - inventory):
        raise ValueError(f"the review cites references outside the inventory: {outside}")
    named = {fid for row in review.findings for fid in row.fact_ids}
    named |= {row.fact_id for row in review.research_items if row.fact_id is not None}
    if unknown := sorted(named - facts):
        raise ValueError(f"the review names facts the draft lacks: {unknown}")
    return review


class _FindingAnswer(ContractModel):
    """A researcher's answer."""

    status: str
    answer: str
    fact_status: str
    evidence: tuple[str, ...] = ()


def validate_finding(
    answer: JsonObject, inventory: frozenset[str], read: frozenset[str]
) -> _FindingAnswer:
    """Check that a resolved finding cites evidence the researcher read.

    Raises:
        ValueError: A resolved finding cites nothing, or cites what was not read.
    """
    finding = _FindingAnswer.model_validate(answer)
    if finding.status not in ("resolved", "unresolved"):
        raise ValueError("a finding is resolved or unresolved")
    if finding.fact_status not in ("observed", "inferred", "unknown"):
        raise ValueError("a finding's fact status is observed, inferred or unknown")
    if outside := sorted(set(finding.evidence) - inventory):
        raise ValueError(f"the finding cites references outside the inventory: {outside}")
    if finding.status == "resolved":
        if not finding.evidence:
            raise ValueError("a resolved finding cites the evidence it rests on")
        if unread := sorted(set(finding.evidence) - read):
            raise ValueError(f"a resolved finding cites evidence it did not read: {unread}")
    return finding


def validate_reveal(answer: JsonObject, draft: StartDraft, probes: int) -> RevealAnswer:
    """Check a first-reveal answer's probe and fact references.

    Raises:
        ValueError: A conflict names an unknown probe or fact, or the verdict and conflicts
            disagree.
    """
    check = RevealAnswer.model_validate(answer)
    facts = {fact.fact_id for fact in draft.facts}
    for conflict in check.conflicts:
        if not 0 <= conflict.probe < probes or not set(conflict.fact_ids) <= facts:
            raise ValueError("a conflict names an unknown probe or fact")
    if check.consistent == bool(check.conflicts):
        raise ValueError("an inconsistent reveal lists its conflicts; a consistent one none")
    return check


def author_start(
    project: ProjectStore,
    spec: BootstrapSpec,
    material: StartMaterial,
    *,
    identity: str,
    round_: int,
    feedback: JsonObject,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[RoleDraw, StartDraft | None]:
    """Run, or replay, one author draw for a round.

    Args:
        project: Project holding the draw.
        spec: The registration.
        material: What the author reads.
        identity: Task bootstrap identity.
        round_: Round number; the stance rotates by task and round.
        feedback: The previous round's reviews and research, or empty.
        sources: Saved artifacts the material came from.
        code_revision: Producing source identity.

    Returns:
        The draw and its draft, or None when the draw gave no usable draft.
    """
    task_index = sum(ord(ch) for ch in material.task.task.task_id)
    variant = spec.author_variants[(task_index + round_) % len(spec.author_variants)]
    opening = {**material.context(), "round": round_, "feedback": feedback}
    draw, draft = _draw(
        project,
        spec.author,
        spec.author_profile,
        material,
        role="start_author",
        variant=variant,
        round_=round_,
        identity=_draw_id(identity, "start_author", variant, round_, 0, opening),
        system=instruction(AUTHOR_INSTRUCTION, AUTHOR_VARIANTS, variant),
        opening=opening,
        answer=author_answer_tool(material.knowledge.state_domains),
        validate=lambda answer, read: validate_author_answer(answer, material, read),
        sources=sources,
        code_revision=code_revision,
    )

    if draft is not None:
        draw = draw.model_copy(update={"answer": draft.model_dump(mode="json")})
    return draw, draft


def review_start(
    project: ProjectStore,
    spec: BootstrapSpec,
    material: StartMaterial,
    draft: StartDraft,
    *,
    identity: str,
    round_: int,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[tuple[RoleDraw, ReviewAnswer | None], ...]:
    """Run, or replay, every reviewer stance over one draft, concurrently.

    Returns:
        Each stance's draw and review, in stance order.
    """
    opening = {**material.context(), "round": round_, "draft": draft.model_dump(mode="json")}

    def one(variant: str) -> Callable[[], tuple[RoleDraw, ReviewAnswer | None]]:
        return lambda: _draw(
            project,
            spec.role("reviewer"),
            spec.review_profile,
            material,
            role="start_reviewer",
            variant=variant,
            round_=round_,
            identity=_draw_id(identity, "start_reviewer", variant, round_, 0, opening),
            system=instruction(REVIEW_INSTRUCTION, REVIEW_VARIANTS, variant),
            opening=opening,
            answer=review_answer_tool(),
            validate=lambda answer, read: validate_review(answer, draft, material.inventory),
            sources=sources,
            code_revision=code_revision,
        )

    return run_bounded(
        [one(variant) for variant in spec.review_variants], len(spec.review_variants)
    )


def reconcile_reviews(
    round_: int,
    reviews: Sequence[tuple[RoleDraw, ReviewAnswer | None]],
    draft: StartDraft,
    limit: int,
) -> tuple[Verdict | None, tuple[ResearchItem, ...], str]:
    """Reconcile the stances' reviews into one verdict and the round's research items.

    Any unusable review leaves the round without a verdict; any refusal refuses; any research
    request, or a deciding fact the draft leaves unknown, asks for research; otherwise accept.

    Returns:
        The verdict (None when a review is unusable), research items (at most ``limit``) and
        the reconciled diagnosis.
    """
    answers = [answer for _, answer in reviews]
    if any(answer is None for _, answer in reviews):
        failures = "; ".join(
            f"{draw.variant}: {draw.failure}" for draw, answer in reviews if answer is None
        )
        return None, (), f"a review gave no usable answer ({failures})"
    usable = [answer for answer in answers if answer is not None]
    refusals = [answer.refusal or answer.summary for answer in usable if answer.verdict == "refuse"]
    if refusals:
        return "refuse", (), " | ".join(refusals)
    requests = [item for answer in usable for item in answer.research_items]
    asked = {row.fact_id for row in requests}
    requests.extend(
        ResearchRequest(
            fact_id=fact.fact_id,
            question=f"What does the recorded evidence show about: {fact.statement}",
            why_deciding="The draft marks this deciding fact unknown.",
        )
        for fact in draft.facts
        if fact.deciding and fact.status == "unknown" and fact.fact_id not in asked
    )
    items: list[ResearchItem] = []
    seen: set[tuple[str | None, str]] = set()
    for row in requests:
        key = (row.fact_id, row.question)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            ResearchItem(
                item_id=f"{round_}-{len(items)}",
                round=round_,
                fact_id=row.fact_id,
                question=row.question,
                why_deciding=row.why_deciding,
            )
        )
    if items:
        return "research", tuple(items[:limit]), "deciding facts are missing"
    return "accept", (), " | ".join(answer.summary for answer in usable)


def research(
    project: ProjectStore,
    spec: BootstrapSpec,
    material: StartMaterial,
    items: Sequence[ResearchItem],
    *,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[Finding, ...]:
    """Answer research items with the family's research step, one draw each, concurrently.

    Returns:
        One finding per item, in item order; a draw without a usable answer stays unresolved
        with its failure.
    """
    context = material.context()

    def one(index: int, item: ResearchItem) -> Callable[[], Finding]:
        variant = spec.research_variants[index % len(spec.research_variants)]
        opening = {**context, "research_item": item.model_dump(mode="json")}

        def run() -> Finding:
            draw, answer = _draw(
                project,
                spec.role("researcher"),
                spec.research_profile,
                material,
                role="start_researcher",
                variant=variant,
                round_=item.round,
                identity=_draw_id(
                    identity, "start_researcher", variant, item.round, index, opening
                ),
                system=instruction(RESEARCH_INSTRUCTION, RESEARCH_VARIANTS, variant),
                opening=opening,
                answer=RESEARCH_ANSWER,
                validate=lambda raw, read: validate_finding(raw, material.inventory, read),
                sources=sources,
                code_revision=code_revision,
            )
            finding = Finding(
                finding_id=f"r{item.item_id}",
                item=item,
                variant=variant,
                transcript=draw.transcript,
                failure=draw.failure,
            )
            if answer is None:
                return finding
            return finding.model_copy(
                update={
                    "status": answer.status,
                    "answer": answer.answer,
                    "fact_status": answer.fact_status,
                    "evidence": answer.evidence,
                }
            )

        return run

    calls = [one(index, item) for index, item in enumerate(items)]
    return run_bounded(calls, max(1, len(calls)))


def check_reveals(
    project: ProjectStore,
    spec: BootstrapSpec,
    material: StartMaterial,
    draft: StartDraft,
    reveals: Sequence[Reveal],
    *,
    identity: str,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[tuple[RoleDraw, RevealAnswer | None], ...]:
    """Run, or replay, every first-reveal stance over an authored start's reveals.

    Returns:
        Each stance's draw and check, in stance order.
    """
    opening: JsonObject = {
        "instruction": draft.instruction,
        "facts": [fact.model_dump(mode="json") for fact in draft.facts],
        "reveals": [row.model_dump(mode="json") for row in reveals],
    }

    def one(variant: str) -> Callable[[], tuple[RoleDraw, RevealAnswer | None]]:
        return lambda: _draw(
            project,
            spec.role("researcher"),
            spec.research_profile,
            material,
            role="reveal_checker",
            variant=variant,
            round_=0,
            identity=_draw_id(identity, "reveal_checker", variant, 0, 0, opening),
            system=instruction(REVEAL_INSTRUCTION, REVEAL_VARIANTS, variant),
            opening=opening,
            answer=reveal_answer_tool(),
            validate=lambda answer, read: validate_reveal(answer, draft, len(reveals)),
            sources=sources,
            code_revision=code_revision,
        )

    variants = spec.reveal_variants
    return run_bounded([one(variant) for variant in variants], len(variants))


def _draw[Answer: ContractModel](
    project: ProjectStore,
    settings: AuthorSettings,
    profile: AgentProfile,
    material: StartMaterial,
    *,
    role: str,
    variant: str,
    round_: int,
    identity: str,
    system: str,
    opening: JsonObject,
    answer: ToolSchema,
    validate: Callable[[JsonObject, frozenset[str]], Answer],
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[RoleDraw, Answer | None]:
    """Run, or replay, one draw and validate its answer; provider errors propagate."""

    def check(draft: JsonObject, read: frozenset[str]) -> str | None:
        try:
            validate(draft, read)
        except ValueError as error:
            return reason(error)
        return None

    reference, transcript = run_agent(
        project,
        settings,
        profile,
        identity=identity,
        role=role,
        variant=variant,
        instruction=system,
        opening=opening,
        tools=READ_TOOLS,
        runner=lambda call: material.run(call, profile.maximum_result_bytes),
        answer=answer,
        check=check,
        sources=sources,
        code_revision=code_revision,
    )
    draw = RoleDraw(role=role, variant=variant, round=round_, transcript=reference)
    if transcript.answer is None or transcript.stop != "answered":
        return draw.model_copy(update={"failure": transcript.failure or transcript.stop}), None
    try:
        parsed = validate(transcript.answer, transcript.served())
    except ValueError as error:
        return draw.model_copy(update={"failure": reason(error)}), None
    return draw.model_copy(update={"answer": transcript.answer}), parsed


def _check_facts(draft: StartDraft, inventory: frozenset[str]) -> None:
    """Refuse repeated facts, citations outside the inventory and unsupported known facts."""
    ids = [fact.fact_id for fact in draft.facts]
    issues: list[str] = []
    if len(set(ids)) != len(ids):
        issues.append("fact identities repeat")
    for fact in draft.facts:
        if outside := sorted(set(fact.references) - inventory):
            issues.append(f"fact {fact.fact_id} cites unavailable or unread references: {outside}")
        recorded = [
            ref for ref in fact.references if ref.startswith("trace:") or ref == RECORDED_STATE_REF
        ]
        if fact.status == "observed" and not recorded:
            issues.append(f"observed fact {fact.fact_id} cites no recorded evidence")
        if fact.status == "inferred" and not fact.references:
            issues.append(f"inferred fact {fact.fact_id} cites nothing it is inferred from")
    if issues:
        raise ValueError("\n".join(issues))


def _finding_view(row: Finding) -> JsonObject:
    """Return the part of a finding a role reads."""
    return {
        "reference": f"finding:{row.finding_id}",
        "question": row.item.question,
        "fact_id": row.item.fact_id,
        "status": row.status,
        "answer": row.answer,
        "fact_status": row.fact_status,
        "evidence": list(row.evidence),
    }


def _draw_id(
    identity: str, role: str, variant: str, round_: int, index: int, opening: JsonObject
) -> str:
    """Return a draw identity bound to the task, role, stance, round, opening and prompts."""
    return _stable_id(
        "start-bootstrap-draw",
        {
            "task": identity,
            "role": role,
            "variant": variant,
            "round": round_,
            "index": index,
            "opening": sha256_json(opening),
            "prompts": bootstrap_prompts_sha256(),
        },
    )

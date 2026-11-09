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
with its failure and never redrawn, except that a failed author draft gets up to
``author_repairs`` validation-repair draws, each shown the draft and its errors; a saved draw
replays without a call.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property

from pydantic import JsonValue, ValidationError

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
    tool_schema,
)
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    BootstrapSpec,
    BootstrapTask,
    Finding,
    Provenance,
    ResearchItem,
    ResearchRequest,
    Reveal,
    ReviewAnswer,
    RoleDraw,
    StartDraft,
    Verdict,
)
from exp.optimize.evaluation.variance.bootstrap_prompts import (
    AUTHOR_ANSWER,
    AUTHOR_INSTRUCTION,
    AUTHOR_REPAIR,
    AUTHOR_VARIANTS,
    READ_TOOLS,
    RESEARCH_ANSWER,
    RESEARCH_INSTRUCTION,
    RESEARCH_VARIANTS,
    REVEAL_ANSWER,
    REVEAL_INSTRUCTION,
    REVEAL_VARIANTS,
    REVIEW_ANSWER,
    REVIEW_INSTRUCTION,
    REVIEW_VARIANTS,
    bootstrap_prompts_sha256,
    instruction,
)
from exp.simulation.reconstruction.models import resolve
from exp.simulation.rules.contracts import trace_locator, trajectory_view
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
    """

    family_id: str
    report: ArtifactInput | None
    rulebook: ArtifactInput
    book: RuleBook
    rule_memory: RuleMemoryBinding | None
    traces: Mapping[str, Trace]
    contracts: Mapping[str, ToolContract]
    state_domains: tuple[str, ...] = ()


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
        """Return the opening context every role shares."""
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
                "rules": len(self.knowledge.book.rules),
                "open_questions": [
                    row.model_dump(mode="json") for row in self.knowledge.book.questions
                ],
            },
            "source_traces": [
                _opening_view(self.views[trace_id][0]) for trace_id in sorted(self.sources)
            ],
            "recorded_state_available": self.recorded_state is not None,
            "allowed_provenance": list(self.allowed()),
            "findings": [_finding_view(row) for row in self.findings],
            "inventory": _inventory_summary(self.inventory),
            "requirements": self.requirements(),
        }

    def requirements(self) -> JsonObject:
        """The fixed shape a start must take; the checks enforce exactly these values."""
        domains = list(self.knowledge.state_domains)
        tools = [tool.name for tool in self.task.task.tools]
        return {
            "manifest.domains": [{"name": name, "owner": "simulated"} for name in domains]
            or "one or more simulated domains, every owner 'simulated'",
            "initial_state": (
                f"an object keyed by exactly these domain names: {domains}"
                if domains
                else "an object keyed by exactly the manifest's domain names"
            ),
            "tools": f"one schema for each of exactly these tools, no others: {tools}",
            "manifest.tools": f"a contract for each of exactly these tools, no others: {tools}",
            "probe_paths": (
                "an object mapping a probe name to a JSON pointer into initial_state; every "
                "probe_paths and facts[].state_paths pointer must resolve in initial_state"
            ),
            "json_pointers": (
                "RFC 6901: escape '~' as '~0' and '/' inside a key as '~1'. "
                + (
                    "The domain roots are exactly "
                    + json.dumps([pointer_token(name) for name in domains])
                    if domains
                    else "A domain named a/b is rooted at /a~1b"
                )
            ),
            "provenance": self._provenance_requirements(),
            "verifier.argument_schemas": (
                f"optional; maps a tool name from {tools} to a JSON Schema object over that "
                'tool\'s arguments, e.g. {"type": "object", "properties": {"command": '
                '{"type": "string"}}}; no \'pattern\'. Put the verifier\'s own fields '
                "(description, outcome_schema) beside it, never inside it"
            ),
        }

    def _provenance_requirements(self) -> JsonObject:
        """What each provenance the evidence allows obliges the draft to name."""
        rows: JsonObject = {}
        for provenance in self.allowed():
            if provenance == "observed":
                rows[provenance] = "initial_state is the recorded state, unchanged"
            elif provenance == "reconstructed":
                prefixes = {
                    trace_id: len(self.recorded_prefix(trace_id))
                    for trace_id in sorted(self.sources)
                    if self.recorded_prefix(trace_id)
                }
                rows[provenance] = (
                    "replay_trace is one of these source trace identities (bare, no 'trace:') "
                    "and prefix_steps is 1 to its recorded prefix length, so the start replays "
                    f"those steps: {json.dumps(prefixes)}"
                )
            else:
                rows[provenance] = (
                    "reveal_probes names at least one call of a tool whose manifest contract "
                    "writes nothing"
                )
        return rows

    def run(
        self, call: ToolCall, maximum_bytes: int
    ) -> tuple[ToolStatus, JsonValue, tuple[str, ...]]:
        """Answer one read-only call: status, result and the references it served."""
        arguments = call.arguments
        value: JsonValue
        served: tuple[str, ...]
        if call.name == "read_trace":
            trace_id = str(arguments.get("trace_id", ""))
            if trace_id not in self.views:
                return "refused", f"no readable trace {trace_id!r}", ()
            view, locators = self.views[trace_id]
            value, served = view, tuple(sorted(locators))
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


def validate_draft(answer: JsonObject, material: StartMaterial) -> StartDraft:
    """Check a draft's structure, citations and claimed provenance.

    Args:
        answer: The author's answer arguments.
        material: What the author could cite.

    Returns:
        The draft.

    Raises:
        ValueError: Any structural check fails.
    """
    answer = canonical_pointers(answer)
    replay = answer.get("replay_trace")
    if isinstance(replay, str) and replay not in material.sources:
        # The inventory names a whole trace by its locator; replay_trace takes the bare identity.
        bare = next((t for t in material.sources if trace_locator(t) == replay), None)
        if bare is not None:
            answer = {**answer, "replay_trace": bare}
    draft = StartDraft.model_validate(answer)
    names = [tool.name for tool in draft.tools]
    declared = {tool.name: tool for tool in draft.manifest.tools}
    if len(set(names)) != len(names):
        raise ValueError(f"the tool schemas repeat a tool: {names}")
    if set(names) != set(declared):
        raise ValueError(
            "the tool schemas and the manifest must name exactly the same tools; only in tools: "
            f"{sorted(set(names) - set(declared))}; only in manifest.tools: "
            f"{sorted(set(declared) - set(names))}"
        )
    if unknown := sorted(set(draft.verifier.argument_schemas) - set(names)):
        raise ValueError(
            f"verifier.argument_schemas keys must be tool names from {sorted(names)}; got {unknown}"
        )
    domains = {domain.name for domain in draft.manifest.domains}
    if any(domain.owner != "simulated" for domain in draft.manifest.domains):
        raise ValueError("every domain of a bootstrapped start is simulated")
    if set(draft.initial_state) != domains:
        raise ValueError("the initial state must name exactly the manifest's domains")
    shared = set(material.knowledge.state_domains)
    if shared and domains != shared:
        raise ValueError(f"a start declares exactly the family's shared domains {sorted(shared)}")
    for pointer in (*draft.probe_paths.values(), *(p for f in draft.facts for p in f.state_paths)):
        try:
            resolve(draft.initial_state, pointer)
        except ValueError:
            raise ValueError(
                f"pointer {pointer!r} does not resolve in initial_state; its domain roots are "
                f"{[pointer_token(key) for key in sorted(draft.initial_state)[:12]]}"
            ) from None
    allowed = draft.verifier.allowed_tools
    if allowed is not None and not set(allowed) <= set(names):
        raise ValueError("the verifier allows tools the start does not offer")
    _check_facts(draft, material.inventory)
    if draft.provenance not in material.allowed():
        raise ValueError(
            f"provenance {draft.provenance!r} is not supported by the evidence; allowed: "
            f"{list(material.allowed())}"
        )
    if draft.provenance == "observed" and draft.initial_state != material.recorded_state:
        raise ValueError("an observed start holds the recorded state unchanged")
    if draft.provenance == "reconstructed":
        if draft.replay_trace not in material.sources:
            choices = {
                trace_id: len(material.recorded_prefix(trace_id)) for trace_id in material.sources
            }
            raise ValueError(
                f"a reconstructed start names one of the task's source traces as replay_trace, "
                f"got {draft.replay_trace!r}; recorded prefix steps by source trace: {choices}"
            )
        recorded = len(material.recorded_prefix(draft.replay_trace))
        if not 1 <= draft.prefix_steps <= recorded:
            raise ValueError(
                f"a reconstructed start replays 1 to {recorded} recorded steps of "
                f"{draft.replay_trace}"
            )
    if draft.provenance == "authored" and not draft.reveal_probes:
        raise ValueError("an authored start names reveal probes for its first-reveal check")
    for probe in draft.reveal_probes:
        policy = declared.get(probe.name)
        if policy is None or policy.writes:
            raise ValueError(f"reveal probe {probe.name!r} is not a declared read-only tool")
    return draft


def pointer_token(key: str) -> str:
    """Return the RFC 6901 pointer naming one top-level key, such as ``/a~1b`` for ``a/b``."""
    return "/" + key.replace("~", "~0").replace("/", "~1")


def canonical_pointer(state: JsonValue, pointer: str) -> str:
    """Return the escaped pointer a slash-split pointer unambiguously means, or it unchanged.

    State keys may hold ``/`` (a domain named ``Python/Node runtime environment``, a file key
    ``/opt/venv/bin/python``), which RFC 6901 escapes as ``~1``. A pointer that leaves such a slash
    unescaped does not resolve; when exactly one way of rejoining its tokens into existing keys
    resolves, that is the location meant, so the pointer is rewritten to it. A pointer that
    resolves as written, resolves no way, or resolves more than one way is returned unchanged
    and judged by the resolution check.
    """
    if not pointer.startswith("/") or "~" in pointer:
        return pointer
    try:
        resolve(state, pointer)
    except ValueError:
        pass
    else:
        return pointer
    tokens = pointer[1:].split("/")
    found: list[tuple[str, ...]] = []

    def walk(value: JsonValue, index: int, keys: tuple[str, ...]) -> None:
        if len(found) > 1:
            return
        if index == len(tokens):
            found.append(keys)
            return
        if isinstance(value, dict):
            for end in range(index + 1, len(tokens) + 1):
                key = "/".join(tokens[index:end])
                if key in value:
                    walk(value[key], end, (*keys, key))
        elif isinstance(value, list):
            token = tokens[index]
            if token.isascii() and token.isdigit() and str(int(token)) == token:
                if int(token) < len(value):
                    walk(value[int(token)], index + 1, (*keys, token))

    walk(state, 0, ())
    if len(found) != 1:
        return pointer
    return "".join(pointer_token(key) for key in found[0])


def canonical_pointers(answer: JsonObject) -> JsonObject:
    """Return the answer with its probe and fact pointers in canonical escaped form."""
    state = answer.get("initial_state")
    if not isinstance(state, dict):
        return answer
    result = dict(answer)
    probes = answer.get("probe_paths")
    if isinstance(probes, dict):
        result["probe_paths"] = {
            name: canonical_pointer(state, value) if isinstance(value, str) else value
            for name, value in probes.items()
        }
    facts = answer.get("facts")
    if isinstance(facts, list):
        rows: list[JsonValue] = []
        for fact in facts:
            paths = fact.get("state_paths") if isinstance(fact, dict) else None
            if isinstance(fact, dict) and isinstance(paths, list):
                fact = {
                    **fact,
                    "state_paths": [
                        canonical_pointer(state, row) if isinstance(row, str) else row
                        for row in paths
                    ],
                }
            rows.append(fact)
        result["facts"] = rows
    return result


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


class _RevealConflict(ContractModel):
    """One conflict between a reveal and the start."""

    probe: int
    fact_ids: tuple[str, ...] = ()
    explanation: str


class RevealAnswer(ContractModel):
    """A first-reveal checker's answer.

    Attributes:
        consistent: Whether no reveal contradicts the start.
        summary: The check in brief.
        conflicts: Each contradiction.
    """

    consistent: bool
    summary: str
    conflicts: tuple[_RevealConflict, ...] = ()


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
    system = instruction(AUTHOR_INSTRUCTION, AUTHOR_VARIANTS, variant)

    def one(
        index: int, body: JsonObject, prompt: str, repair: JsonObject | None
    ) -> tuple[RoleDraw, StartDraft | None, JsonObject | None]:
        return _draw_full(
            project,
            spec.author,
            spec.author_profile,
            material,
            role="start_author",
            variant=variant,
            round_=round_,
            identity=_draw_id(identity, "start_author", variant, round_, index, body, repair),
            system=prompt,
            opening=body,
            answer=task_answer_schema(material),
            validate=lambda answer, read: validate_draft(answer, material),
            sources=sources,
            code_revision=code_revision,
        )

    draw, draft, invalid = one(0, opening, system, None)
    if invalid is None or spec.author_repairs == 0:
        return draw, draft
    repairs: list[RoleDraw] = []
    answer = invalid
    for index in range(1, spec.author_repairs + 1):
        errors = validation_errors(answer, material)
        body = {**opening, "previous_draft": answer, "validation_errors": list(errors)}
        repair: JsonObject = {"repair": index, "errors": sha256_json(list(errors))}
        fixed, draft, again = one(index, body, system + "\n\n" + AUTHOR_REPAIR, repair)
        repairs.append(fixed)
        if draft is not None:
            return draw.model_copy(
                update={
                    "answer": fixed.answer,
                    "failure": None,
                    "invalid": draw.failure,
                    "repairs": tuple(repairs),
                }
            ), draft
        if again is None:
            break
        answer = again
    last = repairs[-1].failure
    return draw.model_copy(
        update={
            "failure": f"{draw.failure}; {len(repairs)} repair draw(s) failed, last: {last}"[:2000],
            "invalid": draw.failure,
            "repairs": tuple(repairs),
        }
    ), None


VALIDATION_ERRORS = 40
VALIDATION_ERROR_CHARACTERS = 299
"""Bounds of the validation errors a repair draw is shown."""


def validation_errors(answer: JsonObject, material: StartMaterial) -> tuple[str, ...]:
    """Return a failed draft's validation errors as ``path: message`` lines, bounded.

    Args:
        answer: The author's answer arguments.
        material: What the author could cite.

    Returns:
        At most :data:`VALIDATION_ERRORS` lines, each under 300 characters; empty when the draft
        validates.
    """
    try:
        validate_draft(answer, material)
    except ValidationError as error:
        rows = [
            f"{'.'.join(str(part) for part in row['loc']) or '(draft)'}: {row['msg']}"
            for row in error.errors(include_url=False)
        ]
    except ValueError as error:
        rows = [f"(draft): {reason(error)}"]
    else:
        return ()
    if len(rows) > VALIDATION_ERRORS:
        rows = [
            *rows[: VALIDATION_ERRORS - 1],
            f"(draft): {len(rows) - VALIDATION_ERRORS + 1} more",
        ]
    return tuple(
        row
        if len(row) <= VALIDATION_ERROR_CHARACTERS
        else row[: VALIDATION_ERROR_CHARACTERS - 1] + "…"
        for row in rows
    )


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
            answer=REVIEW_ANSWER,
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
            answer=REVEAL_ANSWER,
            validate=lambda answer, read: validate_reveal(answer, draft, len(reveals)),
            sources=sources,
            code_revision=code_revision,
        )

    variants = spec.reveal_variants
    return run_bounded([one(variant) for variant in variants], len(variants))


def _draw_full[Answer: ContractModel](
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
) -> tuple[RoleDraw, Answer | None, JsonObject | None]:
    """Run, or replay, one draw and validate its answer; provider errors propagate.

    Returns:
        The draw, its validated answer or None, and the answer when it failed validation.
    """

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
        failure = transcript.failure or transcript.stop
        return draw.model_copy(update={"failure": failure}), None, None
    try:
        parsed = validate(transcript.answer, transcript.served())
    except ValueError as error:
        return draw.model_copy(update={"failure": reason(error)}), None, transcript.answer
    return draw.model_copy(update={"answer": transcript.answer}), parsed, None


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
    draw, parsed, _ = _draw_full(
        project, settings, profile, material, role=role, variant=variant, round_=round_,
        identity=identity, system=system, opening=opening, answer=answer, validate=validate,
        sources=sources, code_revision=code_revision,
    )  # fmt: skip
    return draw, parsed


def _check_facts(draft: StartDraft, inventory: frozenset[str]) -> None:
    """Refuse repeated facts, citations outside the inventory and unsupported known facts."""
    ids = [fact.fact_id for fact in draft.facts]
    if len(set(ids)) != len(ids):
        raise ValueError("fact identities repeat")
    for fact in draft.facts:
        if outside := sorted(set(fact.references) - inventory):
            raise ValueError(
                f"fact {fact.fact_id} cites references outside the inventory: {outside}; a trace "
                "span is citable only as a step's locator or observation_locator in its trace view"
            )
        recorded = [
            ref for ref in fact.references if ref.startswith("trace:") or ref == RECORDED_STATE_REF
        ]
        if fact.status == "observed" and not recorded:
            raise ValueError(f"observed fact {fact.fact_id} cites no recorded evidence")
        if fact.status == "inferred" and not fact.references:
            raise ValueError(f"inferred fact {fact.fact_id} cites nothing it is inferred from")


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
    identity: str,
    role: str,
    variant: str,
    round_: int,
    index: int,
    opening: JsonObject,
    repair: JsonObject | None = None,
) -> str:
    """Return a draw identity bound to the task, role, stance, round, opening and prompts.

    A validation-repair draw adds its repair index and the digest of the errors it repairs.
    """
    return _stable_id(
        "start-bootstrap-draw",
        {
            **({} if repair is None else {"repair": repair}),
            "task": identity,
            "role": role,
            "variant": variant,
            "round": round_,
            "index": index,
            "opening": sha256_json(opening),
            "prompts": bootstrap_prompts_sha256(),
        },
    )


OPENING_STEPS_HEAD = 40
OPENING_STEPS_TAIL = 20
OPENING_TEXT_CHARACTERS = 1_500
"""Bounds of the opening context; the full trace stays readable through ``read_trace``."""


def _clip(value: JsonValue) -> JsonValue:
    """Clip long strings anywhere in a JSON value, keeping head and tail."""
    if isinstance(value, str) and len(value) > OPENING_TEXT_CHARACTERS:
        half = OPENING_TEXT_CHARACTERS // 2
        return f"{value[:half]}…[{len(value) - OPENING_TEXT_CHARACTERS} characters]…{value[-half:]}"
    if isinstance(value, dict):
        return {key: _clip(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clip(item) for item in value]
    return value


def _opening_view(view: JsonObject) -> JsonValue:
    """A source trace for the opening context: first and last steps, long texts clipped.

    Long agent sessions run to hundreds of steps with large tool output, which overflowed the
    author's input allowance; every step stays citable and readable in full via ``read_trace``.
    """
    steps = view.get("steps")
    if isinstance(steps, list) and len(steps) > OPENING_STEPS_HEAD + OPENING_STEPS_TAIL:
        omitted = len(steps) - OPENING_STEPS_HEAD - OPENING_STEPS_TAIL
        view = {
            **view,
            "steps": [
                *steps[:OPENING_STEPS_HEAD],
                {"omitted_steps": omitted, "read_with": "read_trace"},
                *steps[-OPENING_STEPS_TAIL:],
            ],
        }
    return _clip(view)


def _inventory_summary(inventory: frozenset[str] | set[str]) -> JsonObject:
    """Count the citable references by kind instead of listing every step locator."""
    kinds: dict[str, int] = {}
    for reference in inventory:
        kind = reference.split(":", 1)[0]
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "counts": dict(sorted(kinds.items())),
        "note": "cite references returned by read_trace and read_rules; every one is valid",
    }


@cache
def author_answer_schema() -> ToolSchema:
    """The start author's submit tool, typed by the real ``StartDraft`` contract.

    The hand-written tool only said ``manifest``/``initial_state``/``verifier`` were objects, so
    every model invented its own shape and drafts failed validation (``owner`` values, tuples given
    as maps, extra fields). The submit schema is now the contract's own JSON schema, references
    inlined so any provider can read it.
    """
    schema = StartDraft.model_json_schema()
    definitions = schema.pop("$defs", {})

    def inline(node: JsonValue, depth: int = 0) -> JsonValue:
        if isinstance(node, dict):
            reference = node.get("$ref")
            if isinstance(reference, str) and depth < 12:
                name = reference.rsplit("/", 1)[-1]
                rest = {k: v for k, v in node.items() if k != "$ref"}
                return inline({**definitions.get(name, {}), **rest}, depth + 1)
            return {k: inline(v, depth) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(item, depth) for item in node]
        return node

    body = inline(schema)
    assert isinstance(body, dict)
    return tool_schema(
        AUTHOR_ANSWER.name,
        AUTHOR_ANSWER.description,
        body.get("properties", {}),
        tuple(body.get("required", ())),
    )


def task_answer_schema(material: StartMaterial) -> ToolSchema:
    """The submit tool narrowed to this task's fixed shape: its tools, domains and policies.

    The generic contract leaves tool and domain names free and ``verifier.argument_schemas`` an
    open map, so drafts repeated or swapped tool names and kept writing verifier fields
    (``description``, ``outcome_schema``) inside the argument-policy map. The requirements were
    already exact; the schema now says the same: names come from the task's tools and the
    family's domains, each listed once, and argument policies are keyed only by tool name.
    """
    base = author_answer_schema()
    body = copy.deepcopy(base.input_schema)
    tools = sorted({tool.name for tool in material.task.task.tools})
    domains = sorted(material.knowledge.state_domains)
    properties = body["properties"]

    def exact(array: JsonValue, names: list[str]) -> None:
        if not isinstance(array, dict) or not names:
            return
        array["minItems"] = array["maxItems"] = len(names)
        item = array.get("items")
        if isinstance(item, dict) and isinstance(item.get("properties"), dict):
            item["properties"]["name"] = {"enum": names, "type": "string"}

    exact(properties["tools"], tools)
    manifest = properties["manifest"]["properties"]
    exact(manifest["tools"], tools)
    exact(manifest["domains"], domains)
    verifier = properties["verifier"]["properties"]
    verifier["argument_schemas"] = {
        "type": "object",
        "description": "Optional JSON Schema over each named tool's call arguments.",
        "properties": {name: {"type": "object"} for name in tools},
        "additionalProperties": False,
    }
    return base.model_copy(update={"input_schema": body})

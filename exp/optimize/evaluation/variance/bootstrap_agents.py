"""The agentic roles of the start bootstrap: author, reviewers, researchers and reveal checkers.

Every role is a bounded draw of the shared agentic loop
(:func:`~exp.optimize.evaluation.rules.agent_loop.run_agent`): it plans, reads the frozen
:class:`StartMaterial` with read-only tools, drafts, self-checks and answers through its typed
answer tool. Prompts and stances come from
:mod:`~exp.optimize.evaluation.variance.bootstrap_prompts`.

Code only checks structure: a draft's manifest, tools, state and pointers agree; every
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
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import Literal

from pydantic import Field, JsonValue, ValidationError

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
    REPLAY_JUDGE_ANSWER,
    REPLAY_JUDGE_INSTRUCTION,
    REPLAY_JUDGE_VARIANTS,
    RESEARCH_ANSWER,
    RESEARCH_INSTRUCTION,
    RESEARCH_VARIANTS,
    REVEAL_ANSWER,
    REVEAL_INSTRUCTION,
    REVEAL_VARIANTS,
    REVIEW_ANSWER,
    REVIEW_INSTRUCTION,
    REVIEW_VARIANTS,
    ROLE_REPAIR,
    bootstrap_prompts_sha256,
    instruction,
)
from exp.optimize.evaluation.variance.state_pointers import (
    keyed_by_paths,
    split,
)
from exp.runtime.models.providers.errors import (
    ProviderResponseError,
    ProviderRetryableResponseError,
)
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleResponseError
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.simulation.reconstruction.models import resolve
from exp.simulation.rules.contracts import trace_locator, trajectory_view
from exp.simulation.rules.model import RuleBook, ToolContract

LEFT_FRACTION = 0.25
LEFT_STEPS = 3
"""A reconstructed start leaves at least this share, and this many, of a trace's steps."""

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
        if any(self.maximum_prefix(trace_id) for trace_id in self.sources):
            rows.append("reconstructed")
        return (*rows, "authored")

    def trace_steps(self, trace_id: str) -> int:
        """Return how many tool steps a trace recorded."""
        steps = self.views[trace_id][0]["steps"]
        assert isinstance(steps, list)
        return len(steps)

    def maximum_prefix(self, trace_id: str) -> int:
        """Most steps a reconstructed start may replay from a source trace.

        A prefix replaying the whole trace, or its decisive end, leaves the agent nothing to
        decide; at least the last :data:`LEFT_FRACTION` of the steps, and at least
        :data:`LEFT_STEPS`, stay unreplayed. Zero when the trace is too short to reconstruct.
        """
        total = self.trace_steps(trace_id)
        left = max(LEFT_STEPS, math.ceil(total * LEFT_FRACTION))
        return max(0, min(len(self.recorded_prefix(trace_id)), total - left))

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
            "tools": "filled by the host from the task's own tool definitions; do not write it",
            "manifest.tools": (
                f"one effect policy (reads, writes) for each of exactly these tools: {tools}"
            ),
            "probe_paths": (
                "an object mapping a probe name to a JSON pointer into initial_state; every "
                "probe_paths and facts[].state_paths pointer must resolve in initial_state"
            ),
            "json_pointers": (
                "RFC 6901: escape '~' as '~0' and '/' inside a key as '~1' at every depth, "
                "so the key '/opt/venv/bin/python' is the token '~1opt~1venv~1bin~1python'. "
                "Escaping applies only inside pointers: initial_state keys stay plain text "
                "('/workspace', never '~1workspace'). "
                + (
                    "The domain roots are exactly "
                    + json.dumps([pointer_token(name) for name in domains])
                    if domains
                    else "A domain named a/b is rooted at /a~1b"
                )
            ),
            "provenance": self._provenance_requirements(),
        }

    def _provenance_requirements(self) -> JsonObject:
        """What each provenance the evidence allows obliges the draft to name."""
        rows: JsonObject = {}
        for provenance in self.allowed():
            if provenance == "observed":
                rows[provenance] = "initial_state is the recorded state, unchanged"
            elif provenance == "reconstructed":
                prefixes = {
                    trace_id: self.maximum_prefix(trace_id)
                    for trace_id in sorted(self.sources)
                    if self.maximum_prefix(trace_id)
                }
                rows[provenance] = (
                    "replay_trace is one of these source trace identities (bare, no 'trace:') "
                    "and prefix_steps is 1 to the most steps given here. The prefix must end "
                    "before the trace's decisive work: it sets the scene, and the remaining "
                    "steps (at least the last quarter, and at least 3) are left for the agent "
                    f"to decide. Most prefix steps by trace: {json.dumps(prefixes)}"
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


def normalise_draft(
    answer: JsonObject, material: StartMaterial
) -> tuple[JsonObject, tuple[str, ...]]:
    """Fill what the task fixes and repair known shapes, recording every change.

    The tool schemas are the task's own definitions and the manifest's tool policies are one per
    task tool, so the author's versions never decide either: ``tools`` is filled from the task,
    ``manifest.tools`` keeps the author's first policy for each task tool, drops repeats and
    extras, and declares a missing tool as reading and writing every domain (an effect the
    family's contracts do not settle is not assumed absent). Slash-split pointers and a replay
    trace named by its locator are canonicalised. The checks still run on the result.
    """
    notes: list[str] = []
    answer = dict(answer)
    task_tools = [tool.model_dump(mode="json") for tool in material.task.task.tools]
    if answer.get("tools") != task_tools:
        if "tools" in answer:
            notes.append("tools: replaced by the task's own tool definitions")
        answer["tools"] = task_tools
    manifest = answer.get("manifest")
    if isinstance(manifest, dict) and isinstance(manifest.get("tools"), list):
        names = [tool["name"] for tool in task_tools]
        written: dict[str, JsonValue] = {}
        for row in manifest["tools"]:
            name = row.get("name") if isinstance(row, dict) else None
            if name in names and name not in written:
                written[name] = row
            else:
                notes.append(f"manifest.tools: dropped a repeated or unknown policy for {name!r}")
        domains = [
            row["name"]
            for row in manifest.get("domains", [])
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        ]
        for name in names:
            if name not in written:
                notes.append(
                    f"manifest.tools: {name!r} had no policy; it reads and writes every domain"
                )
                written[name] = {"name": name, "reads": domains, "writes": domains}
        answer["manifest"] = {**manifest, "tools": [written[name] for name in names]}
    canonical = canonical_pointers(answer)
    if canonical != answer:
        notes.append("pointers: rewrote slash-split pointers to their escaped form")
    answer = canonical
    state, probes = answer.get("initial_state"), answer.get("probe_paths")
    manifest = answer.get("manifest")
    if isinstance(manifest, dict) and isinstance(state, dict):
        rows: list[JsonValue] = []
        for row in manifest.get("domains", []):
            if (
                isinstance(row, dict)
                and row.get("state_schema") is None
                and isinstance(name := row.get("name"), str)
                and name in state
            ):
                probed = [
                    keys[1:]
                    for pointer in (probes.values() if isinstance(probes, dict) else ())
                    if isinstance(pointer, str)
                    and (keys := _pointer_keys(pointer))
                    and keys[0] == name
                ]
                row = {**row, "state_schema": shape_schema(state[name], probed)}
                notes.append(f"manifest.domains: {name!r} keeps its initial shape on every move")
            rows.append(row)
        answer = {**answer, "manifest": {**manifest, "domains": rows}}
    replay = answer.get("replay_trace")
    if isinstance(replay, str) and replay not in material.sources:
        # The inventory names a whole trace by its locator; replay_trace takes the bare identity.
        bare = next((t for t in material.sources if trace_locator(t) == replay), None)
        if bare is not None:
            answer = {**answer, "replay_trace": bare}
            notes.append("replay_trace: named by its bare trace identity")
    return answer, tuple(notes)


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
    answer, _ = normalise_draft(answer, material)
    draft = StartDraft.model_validate(answer)
    names = [tool.name for tool in draft.tools]
    declared = {tool.name: tool for tool in draft.manifest.tools}
    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        expected = {tool.name for tool in material.task.task.tools}
        raise ValueError(
            f"the tool schemas repeat {repeated} and lack {sorted(expected - set(names))}; give "
            f"exactly one schema per tool of {sorted(expected)}"
        )
    if set(names) != set(declared):
        raise ValueError(
            "the tool schemas and the manifest must name exactly the same tools; only in tools: "
            f"{sorted(set(names) - set(declared))}; only in manifest.tools: "
            f"{sorted(set(declared) - set(names))}"
        )
    domains = {domain.name for domain in draft.manifest.domains}
    if any(domain.owner != "simulated" for domain in draft.manifest.domains):
        raise ValueError("every domain of a bootstrapped start is simulated")
    if set(draft.initial_state) != domains:
        raise ValueError(
            "the initial state must name exactly the manifest's domains; only in initial_state: "
            f"{sorted(set(draft.initial_state) - domains)}; only in manifest.domains: "
            f"{sorted(domains - set(draft.initial_state))}"
        )
    shared = set(material.knowledge.state_domains)
    if shared and domains != shared:
        raise ValueError(f"a start declares exactly the family's shared domains {sorted(shared)}")
    for pointer in (*draft.probe_paths.values(), *(p for f in draft.facts for p in f.state_paths)):
        try:
            resolve(draft.initial_state, pointer)
        except ValueError:
            raise ValueError(
                f"pointer {pointer!r} does not resolve in initial_state; "
                f"{pointer_hint(draft.initial_state, pointer)}"
            ) from None
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
        most = material.maximum_prefix(draft.replay_trace)
        if not 1 <= draft.prefix_steps <= most:
            total = material.trace_steps(draft.replay_trace)
            raise ValueError(
                f"prefix_steps {draft.prefix_steps} is outside 1 to {most} for "
                f"{draft.replay_trace}, which recorded {total} steps: the prefix must end before "
                f"the decisive work and leave at least the last {total - most} steps for the "
                "agent; shorten the prefix, or choose provenance authored"
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

    State keys may hold ``/`` at any depth (a domain named ``Python/Node runtime environment``, a
    file key ``/opt/venv/bin/python``), which RFC 6901 escapes as ``~1``. A pointer that leaves
    such a slash unescaped, in full or in part, does not resolve; when exactly one way of
    rejoining its tokens into existing keys resolves, that is the location meant, so the pointer
    is rewritten to it with every key escaped. A pointer that resolves as written, resolves no
    way, or resolves more than one way is returned unchanged and judged by the resolution check.
    """
    if not pointer.startswith("/"):
        return pointer
    try:
        resolve(state, pointer)
    except ValueError:
        pass
    else:
        return pointer
    raw = pointer[1:].split("/")
    if any("~" in token.replace("~0", "").replace("~1", "") for token in raw):
        return pointer
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in raw]
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


def pointer_hint(state: JsonValue, pointer: str) -> str:
    """Say how far a pointer resolves and the escaped tokens available there."""
    raw = pointer[1:].split("/") if pointer.startswith("/") else []
    prefix, value = "", state
    for count in range(len(raw), 0, -1):
        candidate = "/" + "/".join(raw[:count])
        try:
            value = resolve(state, candidate)
        except ValueError:
            continue
        prefix = candidate
        break
    where = prefix or "(root)"
    if isinstance(value, dict):
        keys = [pointer_token(key)[1:] for key in sorted(value)[:12]]
        more = "" if len(value) <= 12 else f" and {len(value) - 12} more"
        return f"{where} resolves; its keys as pointer tokens are {keys}{more}"
    if isinstance(value, list):
        return f"{where} resolves to a list of {len(value)} items"
    return f"{where} resolves to a {type(value).__name__}, which has no children"


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


class FindingAnswer(ContractModel):
    """A researcher's answer.

    Attributes:
        status: ``resolved`` only with evidence read; otherwise ``unresolved``.
        answer: The answer, or what is missing.
        fact_status: Whether the answer is observed, inferred or still unknown.
        evidence: Inventory references the answer rests on.
    """

    status: Literal["resolved", "unresolved"]
    answer: str
    fact_status: Literal["observed", "inferred", "unknown"]
    evidence: tuple[str, ...] = ()


def validate_finding(
    answer: JsonObject, inventory: frozenset[str], read: frozenset[str]
) -> FindingAnswer:
    """Check that a resolved finding cites evidence the researcher read.

    Raises:
        ValueError: A resolved finding cites nothing, or cites what was not read.
    """
    finding = FindingAnswer.model_validate(answer)
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
    """Run, or replay, one author draw for a round, with its validation repairs.

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
    return _repaired(
        project,
        spec.author,
        spec.author_profile,
        material,
        spec.author_repairs,
        role="start_author",
        variant=variant,
        round_=round_,
        identity=identity,
        index=0,
        system=instruction(AUTHOR_INSTRUCTION, AUTHOR_VARIANTS, variant),
        repair_prompt=AUTHOR_REPAIR,
        opening={**material.context(), "round": round_, "feedback": feedback},
        answer=task_answer_schema(material),
        validate=lambda answer, read: validate_draft(answer, material),
        notes=lambda answer: normalise_draft(answer, material)[1],
        sources=sources,
        code_revision=code_revision,
    )


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
    return answer_errors(lambda: validate_draft(answer, material))


def answer_errors(check: Callable[[], object]) -> tuple[str, ...]:
    """Run one answer check and return its errors as bounded ``path: message`` lines."""
    try:
        check()
    except ValidationError as error:
        rows = [
            f"{'.'.join(str(part) for part in row['loc']) or '(answer)'}: {row['msg']}"
            for row in error.errors(include_url=False)
        ]
    except ValueError as error:
        rows = [f"(answer): {reason(error)}"]
    else:
        return ()
    if len(rows) > VALIDATION_ERRORS:
        rows = [
            *rows[: VALIDATION_ERRORS - 1],
            f"(answer): {len(rows) - VALIDATION_ERRORS + 1} more",
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
        return lambda: _repaired(
            project,
            spec.role("reviewer"),
            spec.review_profile,
            material,
            spec.author_repairs,
            role="start_reviewer",
            variant=variant,
            round_=round_,
            identity=identity,
            index=0,
            system=instruction(REVIEW_INSTRUCTION, REVIEW_VARIANTS, variant),
            repair_prompt=ROLE_REPAIR,
            opening=opening,
            answer=review_answer_schema(),
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
            draw, answer = _repaired(
                project,
                spec.role("researcher"),
                spec.research_profile,
                material,
                spec.author_repairs,
                role="start_researcher",
                variant=variant,
                round_=item.round,
                identity=identity,
                index=index,
                system=instruction(RESEARCH_INSTRUCTION, RESEARCH_VARIANTS, variant),
                repair_prompt=ROLE_REPAIR,
                opening=opening,
                answer=research_answer_schema(),
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


class ReplayJudgement(ContractModel):
    """The replay judge's answer.

    Attributes:
        same_outcome: Whether the replayed observation conveys the recorded outcome.
        reason: The judgement in one sentence.
    """

    same_outcome: bool
    reason: str = Field(min_length=1)


def judge_replay_step(
    project: ProjectStore,
    spec: BootstrapSpec,
    material: StartMaterial,
    step: JsonObject,
    *,
    identity: str,
    index: int,
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
) -> tuple[RoleDraw, ReplayJudgement | None]:
    """Run, or replay, the judge of one replayed step normalisation left undecided.

    Args:
        project: Project holding the draw.
        spec: The registration; ``replay_judge`` judges, else the reviewer.
        material: What the start was authored from.
        step: The call, both observations and error flags, and the host's scores.
        identity: Task bootstrap identity.
        index: The step's position in the replayed prefix.
        sources: Saved artifacts the material came from.
        code_revision: Producing source identity.

    Returns:
        The draw and its judgement, or None when the judge gave no usable answer.
    """
    variant = next(iter(REPLAY_JUDGE_VARIANTS))
    return _repaired(
        project,
        spec.replay_judge or spec.role("reviewer"),
        spec.review_profile,
        material,
        spec.author_repairs,
        role="replay_judge",
        variant=variant,
        round_=0,
        identity=identity,
        index=index,
        system=instruction(REPLAY_JUDGE_INSTRUCTION, REPLAY_JUDGE_VARIANTS, variant),
        repair_prompt=ROLE_REPAIR,
        opening={**step, "question": "Does the replayed observation convey the same outcome?"},
        answer=replay_judge_answer_schema(),
        validate=lambda answer, read: ReplayJudgement.model_validate(answer),
        sources=sources,
        code_revision=code_revision,
    )


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
        return lambda: _repaired(
            project,
            spec.role("researcher"),
            spec.research_profile,
            material,
            spec.author_repairs,
            role="reveal_checker",
            variant=variant,
            round_=0,
            identity=identity,
            index=0,
            system=instruction(REVEAL_INSTRUCTION, REVEAL_VARIANTS, variant),
            repair_prompt=ROLE_REPAIR,
            opening=opening,
            answer=reveal_answer_schema(),
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
) -> tuple[RoleDraw, Answer | None, tuple[JsonObject, tuple[str, ...]] | None]:
    """Run, or replay, one draw and validate its answer; provider errors propagate.

    Returns:
        The draw, its validated answer or None, and the answer with its validation errors when
        it failed validation.
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
    submitted, served = transcript.answer, transcript.served()
    try:
        parsed = validate(submitted, served)
    except ValueError as error:
        errors = answer_errors(lambda: validate(submitted, served))
        return draw.model_copy(update={"failure": reason(error)}), None, (submitted, errors)
    return draw.model_copy(update={"answer": submitted}), parsed, None


def _repaired[Answer: ContractModel](
    project: ProjectStore,
    settings: AuthorSettings,
    profile: AgentProfile,
    material: StartMaterial,
    repairs: int,
    *,
    role: str,
    variant: str,
    round_: int,
    identity: str,
    index: int,
    system: str,
    repair_prompt: str,
    opening: JsonObject,
    answer: ToolSchema,
    validate: Callable[[JsonObject, frozenset[str]], Answer],
    sources: tuple[ArtifactInput, ...],
    code_revision: str,
    notes: Callable[[JsonObject], tuple[str, ...]] | None = None,
) -> tuple[RoleDraw, Answer | None]:
    """Run, or replay, one role draw and up to ``repairs`` validation-repair draws.

    Every role answers through a typed submit tool, yet a draw can still fail its checks. A
    failed answer gets repair draws, each shown the previous answer and its validation errors;
    the first that validates is the draw's answer. A draw lost to a transient provider failure
    or an exhausted output allowance is redrawn up to :data:`REDRAWS` times under a new
    content-addressed identity (the cause joins it; an allowance redraw is also asked to be
    concise); only then does the provider error propagate. ``notes`` names the deterministic
    normalisations applied to the final answer, recorded on the draw.

    Returns:
        The draw, carrying its repairs, and its validated answer or None.
    """

    def one(
        body: JsonObject, prompt: str, repair: JsonObject | None
    ) -> tuple[RoleDraw, Answer | None, tuple[JsonObject, tuple[str, ...]] | None]:
        causes: list[str] = []
        while True:
            key = repair if not causes else {**(repair or {}), "redraws": list(causes)}
            try:
                row, parsed, invalid = _draw_full(
                    project, settings, profile, material, role=role, variant=variant,
                    round_=round_, identity=_draw_id(identity, role, variant, round_, index,
                    body, key), system=prompt, opening=body, answer=answer, validate=validate,
                    sources=sources, code_revision=code_revision,
                )  # fmt: skip
            except (ProviderTransportError, ProviderResponseError) as error:
                cause = redraw_cause(error)
                if cause is None or len(causes) == REDRAWS:
                    raise
                causes.append(cause)
                if cause == "output_allowance":
                    body = {**body, "conciseness": CONCISE}
                continue
            return (
                (row.model_copy(update={"redraws": tuple(causes)}) if causes else row),
                (parsed),
                invalid,
            )

    def noted(row: RoleDraw, final: JsonObject | None) -> RoleDraw:
        if notes is None or final is None:
            return row
        return row.model_copy(update={"normalisations": notes(final)})

    draw, parsed, invalid = one(opening, system, None)
    if invalid is None or repairs == 0:
        return noted(draw, draw.answer if invalid is None else invalid[0]), parsed
    rows: list[RoleDraw] = []
    prompt = system + "\n\n" + repair_prompt.replace("{answer}", answer.name)
    for attempt in range(1, repairs + 1):
        previous, errors = invalid
        body = {**opening, "previous_answer": previous, "validation_errors": list(errors)}
        repair: JsonObject = {"repair": attempt, "errors": sha256_json(list(errors))}
        fixed, parsed, again = one(body, prompt, repair)
        rows.append(fixed)
        if parsed is not None:
            return noted(
                draw.model_copy(
                    update={
                        "answer": fixed.answer,
                        "failure": None,
                        "invalid": draw.failure,
                        "repairs": tuple(rows),
                    }
                ),
                fixed.answer,
            ), parsed
        if again is None:
            break
        invalid = again
    last = rows[-1].failure
    return draw.model_copy(
        update={
            "failure": f"{draw.failure}; {len(rows)} repair draw(s) failed, last: {last}"[:2000],
            "invalid": draw.failure,
            "repairs": tuple(rows),
        }
    ), None


REDRAWS = 2
"""Fresh redraws of a role draw lost to a transient provider failure or exhausted allowance."""

CONCISE = (
    "Your previous draw ran out of output tokens before it submitted. Be concise: short "
    "descriptions, only the facts the task depends on, no repeated content, and submit early."
)
"""Added to a redraw's opening after an exhausted output allowance."""


def redraw_cause(error: Exception) -> str | None:
    """Why a lost draw may be redrawn, or None when it may not.

    A transport failure with no status, a 429 or a 5xx (520 included) is transient; an
    exhausted output allowance is a size failure; a reply outside the typed contract (duplicate
    JSON keys, unparsable tool arguments) is a lost draw. Any other provider error is not
    redrawn.
    """
    if isinstance(error, ProviderRetryableResponseError):
        return "output_allowance" if error.hit_length_limit else "unusable_response"
    if isinstance(error, OpenAICompatibleResponseError):
        return "malformed_response"
    if not isinstance(error, ProviderTransportError):
        return None
    status = error.status_code if isinstance(error, ProviderTransportError) else None
    if status is None or status == 429 or status >= 500:
        return f"transport_{status or 'unknown'}"
    return None


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
            raise ValueError(
                f"observed fact {fact.fact_id} cites no recorded evidence; cite the trace step or "
                "recorded state that shows it, or mark it inferred and cite the rule, contract or "
                "finding it follows from"
            )
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


def contract_schema(tool: ToolSchema, contract: type[ContractModel]) -> ToolSchema:
    """A role's submit tool typed by its real answer contract, references inlined.

    Hand-written submit tools only said fields were objects, so every model invented its own
    shape and answers failed validation (``owner`` values, tuples given as maps, an author's fact
    fields in a review finding). The submit schema is the contract's own JSON schema, references
    inlined so any provider can read it.
    """
    schema = contract.model_json_schema()
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
        tool.name, tool.description, body.get("properties", {}), tuple(body.get("required", ()))
    )


@cache
def author_answer_schema() -> ToolSchema:
    """The start author's submit tool, typed by ``StartDraft``."""
    return contract_schema(AUTHOR_ANSWER, StartDraft)


@cache
def review_answer_schema() -> ToolSchema:
    """The start reviewer's submit tool, typed by ``ReviewAnswer``."""
    return contract_schema(REVIEW_ANSWER, ReviewAnswer)


@cache
def research_answer_schema() -> ToolSchema:
    """The researcher's submit tool, typed by its finding answer."""
    return contract_schema(RESEARCH_ANSWER, FindingAnswer)


@cache
def replay_judge_answer_schema() -> ToolSchema:
    """The replay judge's submit tool, typed by ``ReplayJudgement``."""
    return contract_schema(REPLAY_JUDGE_ANSWER, ReplayJudgement)


@cache
def reveal_answer_schema() -> ToolSchema:
    """The first-reveal checker's submit tool, typed by ``RevealAnswer``."""
    return contract_schema(REVEAL_ANSWER, RevealAnswer)


def task_answer_schema(material: StartMaterial) -> ToolSchema:
    """The submit tool narrowed to this task's fixed shape: its tools, domains and policies.

    The generic contract leaves tool and domain names free, so drafts repeated or swapped tool
    names. The requirements were already exact; the schema now says the same: names come from
    the task's tools and the family's domains, each listed once, and initial_state is keyed by
    exactly those domains.
    """
    base = author_answer_schema()
    body = copy.deepcopy(base.input_schema)
    tools = sorted({tool.name for tool in material.task.task.tools})
    domains = sorted(material.knowledge.state_domains)
    properties = body["properties"]
    assert isinstance(properties, dict)

    def exact(array: JsonValue, names: list[str]) -> None:
        if not isinstance(array, dict) or not names:
            return
        array["minItems"] = array["maxItems"] = len(names)
        item = array.get("items")
        if isinstance(item, dict) and isinstance(item.get("properties"), dict):
            item["properties"]["name"] = {"enum": names, "type": "string"}

    # The host fills the tool schemas from the task; the author never writes them.
    properties.pop("tools", None)
    required = body.get("required", [])
    body["required"] = (
        [name for name in required if name != "tools"] if isinstance(required, list) else []
    )
    manifest = properties["manifest"]["properties"]
    exact(manifest["tools"], tools)
    exact(manifest["domains"], domains)
    if domains:
        properties["initial_state"] = {
            "type": "object",
            "description": "Each domain's initial state, keyed by exactly the family's domains.",
            "properties": {name: {} for name in domains},
            "required": domains,
            "additionalProperties": False,
        }
    return base.model_copy(update={"input_schema": body})


def shape_schema(value: JsonValue, probed: Sequence[Sequence[str]] = ()) -> JsonObject:
    """The shape a simulated domain must keep: its type and, for an object, its top-level keys.

    Run 28's simulator replaced a whole filesystem domain with a string on the first write, and
    every protected file under it read as removed. A domain's declared ``state_schema`` is
    checked on every transition, so a prediction changing a type or dropping a named field (to
    :data:`SHAPE_DEPTH` levels) is refused and re-asked instead of corrupting the world. Values
    may change; only container and text types are pinned, values starting null or numeric
    stay loose, and maps keyed by paths or file names may gain and lose entries, except the
    entries a probe reads: run 30's simulator dropped a probed file from the files map while
    editing another in 10 of 10 such branches, so the files the evaluator reads are pinned too.

    Args:
        value: The domain's initial value.
        probed: Key paths, relative to the domain, that probes read.
    """
    return _shape(value, SHAPE_DEPTH, [list(keys) for keys in probed if keys])


SHAPE_DEPTH = 3
"""Object levels whose keys a domain's state_schema pins."""


def _shape(value: JsonValue, depth: int, probed: list[list[str]] | None = None) -> JsonObject:
    """The pinned shape of one value: types, the keys of named-field objects, probed entries."""
    probed = probed or []
    kind = _json_kind(value)
    if not isinstance(value, dict):
        return {} if kind is None else {"type": kind}
    schema: JsonObject = {"type": "object"}
    # A map keyed by paths or file names (a files map) may gain and lose entries; an object of
    # named fields keeps every field.
    named = not keyed_by_paths(value)
    kept = {keys[0] for keys in probed if keys[0] in value}
    if named and depth > 0:
        kept |= set(value)
    if kept:
        schema["required"] = sorted(kept)
        schema["properties"] = {
            key: _shape(
                value[key],
                depth - 1 if named else 0,
                [keys[1:] for keys in probed if keys[0] == key and keys[1:]],
            )
            for key in sorted(kept)
        }
    return schema


def _json_kind(value: JsonValue) -> str | None:
    """A JSON type to pin, or None for a value whose type may legitimately change."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    return None


def _pointer_keys(pointer: str) -> list[str]:
    """A pointer's keys, or none for one that is not a valid absolute pointer."""
    try:
        return split(pointer)
    except ValueError:
        return []

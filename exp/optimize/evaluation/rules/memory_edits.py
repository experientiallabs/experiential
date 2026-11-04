"""Small, versioned edits to the rule memory set, merged and recorded like small pull requests.

Models never write rule files directly. They write a :class:`RuleDraft` (locators, not quotes),
and the host resolves every locator to the original passage of a world-learning trace, so every
quote in the memory is exact by construction. An edit is a :class:`Proposal`: a title, a
rationale, citations and one to eight :class:`FilePatch` es, each touching one rule file:

- ``patch`` replaces named fields of one existing file, against that file's digest;
- ``add`` creates one new file from a draft;
- ``remove`` deletes one file, against its digest (a duplicate after a merge, say).

:func:`merge_proposals` applies proposals in order, each atomically: every patch of a proposal
applies, or none does. A patch written against a file another proposal changed first is a
conflict, and the proposal is rejected and recorded with the reason, never rebased silently.
Every proposal, merged or rejected, becomes a :class:`ProposalRecord` with the set digest before
and after and the digests of the files it changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, cast

from pydantic import Field

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json, stable_id
from exp.simulation.rules.memory_set import (
    MemorySet,
    Passage,
    RuleFile,
    add_rule,
    memory_set_digest,
    remove_rule,
    replace_rule,
    rule_sha256,
    topic_slug,
)
from exp.simulation.rules.model import Predicate, RuleKind, RuleStatus

PatchOp = Literal["patch", "add", "remove"]
PatchField = Literal[
    "truism",
    "reasoning",
    "evidence",
    "counter_evidence",
    "status",
    "status_rationale",
    "applicability",
    "open_questions",
    "predicate",
]


class RuleDraft(ContractModel):
    """A rule as a model writes it: locators instead of quotes.

    Attributes:
        tool: Tool, or None for an environment-wide rule.
        kind: Precondition, effect, observation or invariant.
        truism: The rule in one plain sentence.
        reasoning: Why it holds: the mechanism or motivation the passages reveal; never a
            restatement of the truism.
        evidence: Locators of supporting passages.
        counter_evidence: Locators of contradicting passages.
        status: Supported, contradicted or unknown.
        status_rationale: Why the cited passages decide, or fail to decide, the status.
        applicability: When the rule applies.
        open_questions: What stays open.
        predicate: Checkable form, only when it can be written exactly.
    """

    tool: str | None = None
    kind: RuleKind
    truism: str = Field(min_length=1, max_length=600)
    reasoning: str = Field(min_length=1, max_length=6000)
    evidence: tuple[str, ...] = Field(min_length=1, max_length=16)
    counter_evidence: tuple[str, ...] = Field(default=(), max_length=16)
    status: RuleStatus = "unknown"
    status_rationale: str = Field(default="", max_length=4000)
    applicability: str = Field(default="", max_length=2000)
    open_questions: tuple[str, ...] = Field(default=(), max_length=8)
    predicate: Predicate | None = None


class RuleFields(ContractModel):
    """Replacement values for named fields of one rule file; unset fields are kept.

    Attributes:
        truism: New truism.
        reasoning: New reasoning.
        evidence: New supporting locators.
        counter_evidence: New contradicting locators.
        status: New status.
        status_rationale: New rationale.
        applicability: New applicability.
        open_questions: New open questions.
        predicate: New predicate.
    """

    truism: str | None = Field(default=None, min_length=1, max_length=600)
    reasoning: str | None = Field(default=None, min_length=1, max_length=6000)
    evidence: tuple[str, ...] | None = Field(default=None, max_length=16)
    counter_evidence: tuple[str, ...] | None = Field(default=None, max_length=16)
    status: RuleStatus | None = None
    status_rationale: str | None = Field(default=None, max_length=4000)
    applicability: str | None = Field(default=None, max_length=2000)
    open_questions: tuple[str, ...] | None = Field(default=None, max_length=8)
    predicate: Predicate | None = None

    def named(self) -> tuple[PatchField, ...]:
        """The fields this patch sets."""
        return tuple(
            cast("PatchField", name)
            for name in sorted(self.model_fields_set)
            if getattr(self, name) is not None
        )


class FilePatch(ContractModel):
    """One edit to one rule file.

    Attributes:
        op: ``patch``, ``add`` or ``remove``.
        path: The rule file (``rules/<topic>/<rule-id>.md``); ignored for ``add``.
        base_sha256: Digest of the file the patch was written against (``patch``/``remove``).
        fields: Replacement fields (``patch``).
        rule: The new rule (``add``).
    """

    op: PatchOp
    path: str = Field(default="", max_length=512)
    base_sha256: Sha256 | None = None
    fields: RuleFields | None = None
    rule: RuleDraft | None = None


class Proposal(ContractModel):
    """One small pull request against the memory set.

    Attributes:
        title: One line naming the change.
        rationale: Why the change is right.
        citations: Locators or rule ids the rationale rests on.
        patches: The file edits, applied atomically.
    """

    title: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=4000)
    citations: tuple[str, ...] = Field(min_length=1, max_length=32)
    patches: tuple[FilePatch, ...] = Field(min_length=1, max_length=8)


class ProposalRecord(ContractModel):
    """A proposal as merged or rejected.

    Attributes:
        proposal_id: Stable identity.
        cycle: Governance cycle, or -1 for the research seed.
        author: Inspector, or research role, that proposed it.
        variant: Prompt variant of the proposing draw.
        proposal: The proposal.
        status: ``merged`` or ``rejected``.
        reason: Why it was rejected.
        before_sha256: Set digest before.
        after_sha256: Set digest after (equal to before when rejected).
        files: Changed file path mapped to its new digest, or None when removed.
    """

    proposal_id: str
    cycle: int = Field(ge=-1)
    author: str
    variant: str = ""
    proposal: Proposal
    status: Literal["merged", "rejected"]
    reason: str | None = None
    before_sha256: Sha256
    after_sha256: Sha256
    files: dict[str, Sha256 | None] = Field(default_factory=dict)


def rule_id_for(draft: RuleDraft) -> str:
    """A stable, readable identity for a drafted rule."""
    digest = sha256_json([draft.tool, draft.kind, " ".join(draft.truism.lower().split())])[:12]
    return f"{draft.kind}.{topic_slug(draft.tool)}.{digest}"


def resolve_draft(
    draft: RuleDraft,
    passages: Mapping[str, Passage],
    tools: frozenset[str],
) -> RuleFile:
    """Turn a draft into a rule file, quoting every cited locator from its source.

    Args:
        draft: The model's draft.
        passages: Original passages by locator (the world-learning corpus).
        tools: Known tool names.

    Returns:
        The rule file, version 1.

    Raises:
        ValueError: The tool is unknown or a locator names no world-learning passage.
    """
    if draft.tool is not None and draft.tool not in tools:
        raise ValueError(f"rule names a tool no contract knows: {draft.tool}")
    return RuleFile(
        rule_id=rule_id_for(draft),
        topic=topic_slug(draft.tool),
        tool=draft.tool,
        kind=draft.kind,
        truism=draft.truism,
        reasoning=draft.reasoning,
        evidence=quote(draft.evidence, passages),
        counter_evidence=quote(draft.counter_evidence, passages),
        status=draft.status,
        status_rationale=draft.status_rationale,
        applicability=draft.applicability,
        open_questions=draft.open_questions,
        predicate=draft.predicate,
    )


def quote(locators: Sequence[str], passages: Mapping[str, Passage]) -> tuple[Passage, ...]:
    """Original passages for locators, deduplicated in order.

    Raises:
        ValueError: A locator names no world-learning passage.
    """
    missing = sorted({loc for loc in locators if loc not in passages})
    if missing:
        raise ValueError(f"locators name no world-learning passage: {missing}")
    seen: dict[str, Passage] = {}
    for locator in locators:
        passage = passages[locator]
        seen.setdefault(passage.locator, passage)
    return tuple(seen.values())


def apply_proposal(
    memory: MemorySet,
    proposal: Proposal,
    passages: Mapping[str, Passage],
    *,
    allowed: frozenset[PatchOp],
    fields: frozenset[PatchField],
) -> tuple[MemorySet, dict[str, Sha256 | None]]:
    """Apply every patch of one proposal, or none.

    Args:
        memory: The set.
        proposal: The proposal.
        passages: Original passages by locator.
        allowed: Operations the proposing role may use.
        fields: Fields the proposing role may patch.

    Returns:
        The edited set and the changed files with their new digests (None when removed).

    Raises:
        ValueError: A patch uses an operation or field the role may not, names no file, is
            stale, or cites a locator outside the corpus.
    """
    tools = frozenset(memory.tools)
    changed: dict[str, Sha256 | None] = {}
    for patch in proposal.patches:
        if patch.op not in allowed:
            raise ValueError(f"this role may not {patch.op} rule files")
        if patch.op == "add":
            if patch.rule is None:
                raise ValueError("an add patch needs a rule")
            rule = resolve_draft(patch.rule, passages, tools)
            memory = add_rule(memory, rule)
            changed[rule.path] = rule_sha256(memory.rule(rule.rule_id) or rule)
            continue
        current = memory.by_path(patch.path)
        if current is None or patch.base_sha256 is None:
            raise ValueError(f"patch names no rule file with a base digest: {patch.path!r}")
        if patch.op == "remove":
            memory = remove_rule(memory, patch.base_sha256, current.rule_id)
            changed[current.path] = None
            continue
        if patch.fields is None or not patch.fields.named():
            raise ValueError(f"patch to {patch.path} names no field")
        named = patch.fields.named()
        refused = sorted(set(named) - fields)
        if refused:
            raise ValueError(f"this role may not patch {refused}")
        update: dict[str, object] = {name: getattr(patch.fields, name) for name in named}
        for side in ("evidence", "counter_evidence"):
            if side in update:
                update[side] = quote(cast("tuple[str, ...]", update[side]), passages)
        memory = replace_rule(memory, patch.base_sha256, current.model_copy(update=update))
        edited = memory.rule(current.rule_id)
        assert edited is not None
        changed[edited.path] = rule_sha256(edited)
    return memory, changed


def merge_proposals(
    memory: MemorySet,
    proposals: Sequence[tuple[str, str, Proposal]],
    passages: Mapping[str, Passage],
    *,
    cycle: int,
    permissions: Mapping[str, tuple[frozenset[PatchOp], frozenset[PatchField]]],
) -> tuple[MemorySet, tuple[ProposalRecord, ...]]:
    """Merge proposals in order, each atomically, recording every one.

    Args:
        memory: The set at the start of the merge.
        proposals: ``(author, variant, proposal)`` in merge order.
        passages: Original passages by locator.
        cycle: Governance cycle.
        permissions: Operations and fields each author may use.

    Returns:
        The merged set and one record per proposal, in order.
    """
    records: list[ProposalRecord] = []
    for author, variant, proposal in proposals:
        before = memory_set_digest(memory)
        proposal_id = stable_id(
            "proposal",
            {
                "cycle": cycle,
                "author": author,
                "variant": variant,
                "before": before,
                "p": proposal.model_dump(mode="json"),
            },
        )
        allowed, fields = permissions[author]
        try:
            edited, changed = apply_proposal(
                memory, proposal, passages, allowed=allowed, fields=fields
            )
        except ValueError as error:
            records.append(
                ProposalRecord(
                    proposal_id=proposal_id,
                    cycle=cycle,
                    author=author,
                    variant=variant,
                    proposal=proposal,
                    status="rejected",
                    reason=str(error)[:2000],
                    before_sha256=before,
                    after_sha256=before,
                )
            )
            continue
        memory = edited.model_copy(
            update={"revision": memory.revision + 1, "edits": (*memory.edits, proposal_id)}
        )
        records.append(
            ProposalRecord(
                proposal_id=proposal_id,
                cycle=cycle,
                author=author,
                variant=variant,
                proposal=proposal,
                status="merged",
                before_sha256=before,
                after_sha256=memory_set_digest(memory),
                files=changed,
            )
        )
    return memory, tuple(records)

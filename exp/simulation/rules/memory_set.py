"""The task-family rule memory as a set of small Markdown files, versioned as one artifact.

The environment simulator's learned memory is not one long document. Each rule is its own file,
``rules/<topic>/<rule-id>.md``, holding the truism, the long-form reasoning about why it holds
(its motivation), the original passages it rests on with their source coordinates, its status
with the weighing rationale, where it applies and what is still open. Around the rule files sit
an ``index.md`` (one row per rule, the only thing runtime recall scores), one ``tools/<tool>.md``
per tool, ``contradictions.md`` and ``unknowns.md``.

The whole set is one immutable ``rule-memory-set`` artifact. ``memory-set.json`` holds the typed
:class:`MemorySet`; ``files.json`` digests every other file and binds them into one set digest,
so a definition that pins the artifact pins every file. Edits are small: one rule file is
replaced, added or removed against the digest of the file it patches, and each rule file carries
its own version.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
)
from exp.common.project import ArtifactAlreadyExistsError, ArtifactStore
from exp.common.project.manifests import artifact_input
from exp.simulation.rules.model import Predicate, RuleKind, RuleStatus

MEMORY_SET_TYPE = "rule-memory-set"
"""Artifact type of a saved rule memory set."""

MEMORY_SET_FORMAT = "rule-memory-set-v1"
"""Version of the file layout and rendering."""

SET_JSON = "memory-set.json"
DIGESTS_JSON = "files.json"
INDEX_MD = "index.md"
CONTRADICTIONS_MD = "contradictions.md"
UNKNOWNS_MD = "unknowns.md"
ENVIRONMENT_TOPIC = "environment"
"""Topic of rules about no single tool."""

_ID = r"^[a-z0-9][a-z0-9_.-]{0,127}$"


class Passage(ContractModel):
    """One original passage of a world-learning trace, quoted exactly with its coordinates.

    Attributes:
        locator: Step locator, ``trace:<trace_id>:span:<span_id>``.
        trace_id: Trace the passage belongs to.
        step: Step index within the trace.
        quote: The original step (action, observation, failure) as recorded, never paraphrased.
        source_sha256: Digest of the original text, so a drifted quote is detectable.
    """

    locator: str = Field(min_length=1, max_length=512)
    trace_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    quote: str = Field(min_length=1, max_length=16_000)
    source_sha256: Sha256


class RuleFile(ContractModel):
    """One rule file: a truism with its reasoning, evidence, status and scope.

    Attributes:
        rule_id: Stable identity.
        topic: Topic directory, a tool slug or ``environment``.
        tool: Tool the rule is about, or None for an environment-wide rule.
        kind: Precondition, effect, observation or invariant.
        truism: The rule in one plain sentence.
        reasoning: Why the rule holds: the mechanism or motivation the evidence reveals.
        evidence: Original supporting passages.
        counter_evidence: Original contradicting passages.
        status: Supported, contradicted or unknown.
        status_rationale: The weighing rationale behind the status.
        applicability: When the rule applies (states, arguments, orderings).
        open_questions: What the evidence leaves open about this rule.
        predicate: Checkable form over one transition, when it can be written exactly.
        version: Incremented by every merged edit of this file.
    """

    rule_id: str = Field(pattern=_ID)
    topic: str = Field(pattern=_ID)
    tool: str | None = None
    kind: RuleKind
    truism: str = Field(min_length=1, max_length=600)
    reasoning: str = Field(min_length=1, max_length=6000)
    evidence: tuple[Passage, ...] = ()
    counter_evidence: tuple[Passage, ...] = ()
    status: RuleStatus = "unknown"
    status_rationale: str = Field(default="", max_length=4000)
    applicability: str = Field(default="", max_length=2000)
    open_questions: tuple[str, ...] = ()
    predicate: Predicate | None = None
    version: int = Field(default=1, ge=1)

    @property
    def path(self) -> str:
        """The file's path inside the set."""
        return f"rules/{self.topic}/{self.rule_id}.md"

    def locators(self) -> frozenset[str]:
        """Every locator the file quotes, on either side."""
        return frozenset(p.locator for p in (*self.evidence, *self.counter_evidence))


class CoverageGap(ContractModel):
    """A tool, state or transition the memory says nothing about.

    Attributes:
        kind: ``tool``, ``state`` or ``transition``.
        subject: What is uncovered (a tool name, a failure form, a result shape).
        note: Why it matters.
    """

    kind: Literal["tool", "state", "transition"]
    subject: str = Field(min_length=1, max_length=600)
    note: str = Field(default="", max_length=1200)


class IndexRow(ContractModel):
    """One index line: what runtime recall scores before a rule file is opened.

    Attributes:
        rule_id: The rule.
        path: Its file.
        tool: Its tool, or None.
        kind: Its kind.
        status: Its status.
        truism: Its truism.
        applicability: When it applies.
    """

    rule_id: str
    path: str
    tool: str | None
    kind: RuleKind
    status: RuleStatus
    truism: str
    applicability: str


class MemorySet(ContractModel):
    """The typed content of a memory set; every Markdown file is rendered from it.

    Attributes:
        format: Layout version.
        revision: Incremented by every merged proposal.
        tools: Every tool the traces declare or call.
        rules: Rule files, in a stable order.
        gaps: Known coverage gaps.
        edits: Merged proposal identities, in merge order.
        sources: World-learning trace identities the set was learned from.
    """

    format: Literal["rule-memory-set-v1"] = MEMORY_SET_FORMAT
    revision: int = Field(default=0, ge=0)
    tools: tuple[str, ...] = ()
    rules: tuple[RuleFile, ...] = ()
    gaps: tuple[CoverageGap, ...] = ()
    edits: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    def rule(self, rule_id: str) -> RuleFile | None:
        """Return one rule file by identity."""
        return next((rule for rule in self.rules if rule.rule_id == rule_id), None)

    def by_path(self, path: str) -> RuleFile | None:
        """Return one rule file by path."""
        return next((rule for rule in self.rules if rule.path == path), None)

    def index(self) -> tuple[IndexRow, ...]:
        """Return one index row per rule, in set order."""
        return tuple(
            IndexRow(
                rule_id=rule.rule_id,
                path=rule.path,
                tool=rule.tool,
                kind=rule.kind,
                status=rule.status,
                truism=rule.truism,
                applicability=rule.applicability,
            )
            for rule in self.rules
        )


def topic_slug(tool: str | None) -> str:
    """Return the topic directory of a tool, or ``environment`` for no tool."""
    if tool is None:
        return ENVIRONMENT_TOPIC
    slug = "".join(c if c.isalnum() or c in "_.-" else "-" for c in tool.lower()).strip("-._")
    return slug[:64].strip("-._") or "tool"


def rule_sha256(rule: RuleFile) -> Sha256:
    """Digest of one rule file's content, the base a patch to it names."""
    return sha256_json(rule)


def replace_rule(memory: MemorySet, base: Sha256, rule: RuleFile) -> MemorySet:
    """Replace one rule file, bumping its version, if it still has the digest the patch names.

    Args:
        memory: The set.
        base: Digest of the file the patch was written against.
        rule: The new content; its identity must already exist.

    Returns:
        The edited set.

    Raises:
        ValueError: The rule does not exist, or the file changed since the patch was written.
    """
    current = memory.rule(rule.rule_id)
    if current is None:
        raise ValueError(f"no rule file {rule.rule_id} to patch")
    if rule_sha256(current) != base:
        raise ValueError(f"rule file {current.path} changed since the patch was written")
    updated = rule.model_copy(update={"version": current.version + 1})
    return memory.model_copy(
        update={"rules": tuple(updated if r.rule_id == rule.rule_id else r for r in memory.rules)}
    )


def add_rule(memory: MemorySet, rule: RuleFile) -> MemorySet:
    """Add one new rule file.

    Raises:
        ValueError: A rule with this identity already exists.
    """
    if memory.rule(rule.rule_id) is not None:
        raise ValueError(f"rule file {rule.path} already exists")
    return memory.model_copy(
        update={"rules": (*memory.rules, rule.model_copy(update={"version": 1}))}
    )


def remove_rule(memory: MemorySet, base: Sha256, rule_id: str) -> MemorySet:
    """Remove one rule file if it still has the digest the patch names.

    Raises:
        ValueError: The rule does not exist, or it changed since the patch was written.
    """
    current = memory.rule(rule_id)
    if current is None:
        raise ValueError(f"no rule file {rule_id} to remove")
    if rule_sha256(current) != base:
        raise ValueError(f"rule file {current.path} changed since the removal was written")
    return memory.model_copy(
        update={"rules": tuple(r for r in memory.rules if r.rule_id != rule_id)}
    )


def render_memory_set(memory: MemorySet) -> dict[str, str]:
    """Render every Markdown file of the set.

    Args:
        memory: The set.

    Returns:
        Markdown text keyed by POSIX path: the index, every rule file, one file per tool,
        contradictions and unknowns.
    """
    files = {rule.path: render_rule(rule) for rule in memory.rules}
    files[INDEX_MD] = _index(memory)
    for tool in memory.tools:
        files[f"tools/{topic_slug(tool)}.md"] = _tool(memory, tool)
    files[CONTRADICTIONS_MD] = _contradictions(memory)
    files[UNKNOWNS_MD] = _unknowns(memory)
    return files


def memory_set_files(memory: MemorySet) -> dict[str, bytes]:
    """Return the exact files a memory set artifact holds, with the per-file digests.

    Args:
        memory: The set.

    Returns:
        UTF-8 payloads keyed by path: the Markdown files, ``memory-set.json`` and ``files.json``.
    """
    files = {path: text.encode() for path, text in render_memory_set(memory).items()}
    files[SET_JSON] = memory.model_dump_json(indent=2).encode() + b"\n"
    digests = {path: sha256_json(payload.decode()) for path, payload in sorted(files.items())}
    manifest = {"format": memory.format, "revision": memory.revision, "files": digests}
    manifest["digest"] = sha256_json(digests)
    files[DIGESTS_JSON] = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    return dict(sorted(files.items()))


def memory_set_digest(memory: MemorySet) -> Sha256:
    """The digest binding every file of the set."""
    return json.loads(memory_set_files(memory)[DIGESTS_JSON])["digest"]


def save_memory_set(
    store: ArtifactStore,
    memory: MemorySet,
    *,
    artifact_id: str,
    inputs: Sequence[ArtifactInput],
    code_revision: str,
) -> ArtifactInput:
    """Save a memory set as one artifact, or adopt the identical one already saved.

    Raises:
        ValueError: An artifact with this identity holds a different set.
    """
    files = memory_set_files(memory)
    if not store.exists(artifact_id):
        try:
            manifest = store.write(
                artifact_id=artifact_id,
                artifact_type=MEMORY_SET_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
                ),
                files=files,
            )
        except ArtifactAlreadyExistsError:
            pass
        else:
            return artifact_input(manifest)
    stored = store.read(artifact_id)
    if stored.manifest.artifact_type != MEMORY_SET_TYPE or dict(stored.payloads) != files:
        raise ValueError(f"saved memory set {artifact_id} differs from this set")
    return artifact_input(stored.manifest)


def load_memory_set(store: ArtifactStore, reference: ArtifactInput) -> MemorySet:
    """Read and verify one pinned memory set.

    Every file must match its digest in ``files.json`` and the set must re-render to exactly
    the saved files, so a hand-edited file is refused rather than served.

    Raises:
        ValueError: The manifest digest changed, the type is wrong, or a file differs.
    """
    stored = store.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError(f"memory set {reference.artifact_id} manifest digest changed")
    if stored.manifest.artifact_type != MEMORY_SET_TYPE:
        raise ValueError(f"artifact {reference.artifact_id} is not a {MEMORY_SET_TYPE}")
    payload = stored.payloads.get(SET_JSON)
    if payload is None:
        raise ValueError(f"memory set {reference.artifact_id} holds no {SET_JSON}")
    memory = MemorySet.model_validate_json(payload)
    if memory_set_files(memory) != dict(stored.payloads):
        raise ValueError(f"memory set {reference.artifact_id} files differ from their digests")
    return memory


def render_rule(rule: RuleFile) -> str:
    """Render one rule file."""
    lines = [
        f"# {_one(rule.truism)}",
        "",
        f"- Rule: `{rule.rule_id}` (version {rule.version})",
        f"- Tool: {'`' + rule.tool + '`' if rule.tool else 'environment'}; kind: {rule.kind}",
        f"- Status: **{rule.status}**",
        "",
        "## Status rationale",
        "",
        rule.status_rationale.strip() or "Not weighed.",
        "",
        "## Why it holds",
        "",
        rule.reasoning.strip(),
        "",
        "## Applicability",
        "",
        rule.applicability.strip() or "Not stated.",
        "",
    ]
    if rule.predicate is not None:
        lines += ["## Predicate", "", "~~~json", rule.predicate.model_dump_json(), "~~~", ""]
    lines += ["## Evidence", ""] + _passages(rule.evidence)
    lines += ["## Counter-evidence", ""] + _passages(rule.counter_evidence)
    lines += ["## Open questions", ""]
    lines += [f"- {_one(q)}" for q in rule.open_questions] or ["None recorded."]
    return "\n".join(lines) + "\n"


def _passages(passages: Sequence[Passage]) -> list[str]:
    """Quoted passages with coordinates, or a note that there are none."""
    if not passages:
        return ["None.", ""]
    lines: list[str] = []
    for passage in passages:
        lines += [
            f"### `{passage.locator}` (trace `{passage.trace_id}`, step {passage.step})",
            "",
            "~~~json",
            passage.quote,
            "~~~",
            "",
        ]
    return lines


def _index(memory: MemorySet) -> str:
    """The index: one row per rule."""
    lines = [
        "# Rule memory index",
        "",
        f"Revision {memory.revision}; {len(memory.rules)} rules over {len(memory.tools)} tools, "
        f"learned from {len(memory.sources)} world-learning traces.",
        "",
        "| Rule | Tool | Kind | Status | Truism |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in memory.index():
        lines.append(
            f"| [{row.rule_id}]({row.path}) | {row.tool or 'environment'} | {row.kind} | "
            f"{row.status} | {_one(row.truism).replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def _tool(memory: MemorySet, tool: str) -> str:
    """One tool's rule list."""
    rules = [rule for rule in memory.rules if rule.tool == tool]
    lines = [f"# Tool `{tool}`", ""]
    for kind in ("precondition", "effect", "observation", "invariant"):
        mine = [rule for rule in rules if rule.kind == kind]
        if mine:
            lines += [f"## {kind.capitalize()}s", ""]
            lines += [f"- [{r.status}] [{_one(r.truism)}](../{r.path})" for r in mine]
            lines.append("")
    if not rules:
        lines += ["No rules yet; this tool is a coverage gap.", ""]
    return "\n".join(lines)


def _contradictions(memory: MemorySet) -> str:
    """Every contradicted rule or rule with counter-evidence, both sides kept."""
    lines = ["# Contradictions", ""]
    found = [r for r in memory.rules if r.status == "contradicted" or r.counter_evidence]
    for rule in found:
        lines += [
            f"- [{rule.status}] [{_one(rule.truism)}]({rule.path})",
            "  - For: " + (", ".join(f"`{p.locator}`" for p in rule.evidence) or "none"),
            "  - Against: "
            + (", ".join(f"`{p.locator}`" for p in rule.counter_evidence) or "none"),
        ]
        if rule.status_rationale:
            lines.append(f"  - Rationale: {_one(rule.status_rationale)}")
    if not found:
        lines.append("None recorded.")
    return "\n".join(lines) + "\n"


def _unknowns(memory: MemorySet) -> str:
    """Unknown rules, open questions and coverage gaps."""
    lines = ["# Unknowns", "", "## Unknown rules", ""]
    unknown = [r for r in memory.rules if r.status == "unknown"]
    lines += [f"- [{_one(r.truism)}]({r.path})" for r in unknown] or ["None."]
    lines += ["", "## Open questions", ""]
    questions = [(r, q) for r in memory.rules for q in r.open_questions]
    lines += [f"- {_one(q)} ([{r.rule_id}]({r.path}))" for r, q in questions] or ["None."]
    lines += ["", "## Coverage gaps", ""]
    lines += [f"- {g.kind} `{g.subject}`: {_one(g.note)}" for g in memory.gaps] or ["None."]
    return "\n".join(lines) + "\n"


def _one(text: str) -> str:
    """Collapse whitespace onto one line."""
    return " ".join(text.split())

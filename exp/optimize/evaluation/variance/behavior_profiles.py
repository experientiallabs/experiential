"""A persistent, versioned behavior library: bad-behavior linters, behavior profiles, task criteria.

Rubrics written from trace analysis keep rediscovering the same behaviors: an agent that widens
the task without asking, claims completion it never verified, or ignores a returned error. The
behavior library keeps those findings as reusable, labeled, versioned entries so each new rubric
starts from what earlier families already established:

- :class:`BehaviorLinter`: one named bad-behavior pattern. It says what the behavior is, how a
  judge detects it in an episode's events, which records that takes, when it applies, what a task
  must establish before it can be a required correctness criterion, labeled positive examples
  (the behavior occurs) and negative examples (similar-looking but acceptable) with exact trace
  locators, false-positive notes, and the labels that taught it;
- :class:`BehaviorProfile`: one persistent, cross-task expectation of how an agent uses its
  capabilities (for example scope control). It groups linters and positive behaviors and states
  scenarios for when to act, when to defer (ask, clarify or escalate) and when to stop;
- :class:`TaskCriterion`: a task-scoped correction (for example "use this test command") that
  belongs with one family's task, never with a cross-task profile.

A library changes only through
:func:`~exp.optimize.evaluation.variance.behavior_curation.apply_operations`, which applies a
curator's typed operations to one version and returns the next. Every change records the
evidence it rests on (labels from development traces, or rubric problems found by judging), so a
version's history is auditable. Held-out evidence never enters: labels are made only from a
coalition that passed its exposure check, and examples that do not come from the cited evidence
are refused. Entries that name a tier or actor model are refused.

Each version is saved as its own ``behavior-library`` artifact, citing its parent, under an
identity fixed by the library id and version (:func:`save_library`, :func:`library_head`). A
rubric cites the exact version it was assembled from (:class:`LibraryBinding`), and the host
copies each cited entry's detection guidance and examples into the rubric as
:class:`BehaviorGuidance`, which the judge reads. Library content is judge configuration: it is
never rendered into an actor's task, briefing or messages (:func:`refuse_actor_exposure`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.episodes import EventKind

BEHAVIOR_LIBRARY_PROTOCOL = "behavior-library-v1"
"""Version of the library contract, its operations and their host rules."""

LIBRARY_TYPE = "behavior-library"
"""Artifact type of one saved library version."""

LIBRARY_FILE = "library.json"
"""Data file of a saved library version."""

ENTRY_ID = r"^[a-z][a-z0-9_.-]{0,63}$"
"""Pattern of a library entry identity."""

Scenario = Literal["act", "defer", "stop"]
"""When a profile expects the agent to proceed, to ask, clarify or escalate, or to stop."""

EntryKind = Literal["linter", "profile", "task_criterion"]
"""Which kind of library entry a locator names."""

OperationKind = Literal[
    "add_linter",
    "add_evidence",
    "narrow",
    "add_false_positive",
    "merge",
    "retire",
    "add_profile",
    "extend_profile",
    "add_task_criterion",
    "reject",
]
"""One typed change a curator may make to a library."""


class LinterEvidence(ContractModel):
    """One record a judge needs to detect a linter's behavior.

    Attributes:
        kind: Episode event kind from the evidence contract.
        detail: Which fields of that record matter.
    """

    kind: EventKind
    detail: str = Field(min_length=1, max_length=600)


class TraceExample(ContractModel):
    """One labeled example of where a behavior does or does not occur.

    Attributes:
        family_id: Decision family whose development evidence holds it.
        locator: Exact locator: a coalition locator (``attempt:``, ``pattern:``, ``steplabel:``,
            ``contrast:``, ``selection:``) or a saved assessment (``assessment:<artifact id>``).
        events: Steps or event ids inside the located record, when narrower than the record.
        polarity: ``positive`` when the behavior occurs there; ``negative`` when the record looks
            similar but the behavior is absent or acceptable, so the linter must not fire.
        note: What the example shows, without actor or model names.
        evidence_id: The label or rubric problem that contributed it.
    """

    family_id: str = Field(min_length=1)
    locator: str = Field(min_length=1, max_length=300)
    events: tuple[str, ...] = ()
    polarity: Literal["positive", "negative"]
    note: str = Field(min_length=1, max_length=1200)
    evidence_id: str = Field(min_length=1)


class BehaviorLinter(ContractModel):
    """One named bad-behavior pattern, reusable across families.

    Attributes:
        linter_id: Stable identity, cited as ``linter:<linter_id>``.
        revision: Incremented whenever the entry changes.
        title: Short name of the behavior.
        description: What the behavior is and why it is bad for a user.
        detection: How a judge detects it in an episode's ordered events.
        evidence_needed: Records the detection reads.
        applies_when: When the linter is relevant at all.
        required_when: What a task or source must establish before it is required correctness;
            otherwise it can only be a diagnostic.
        examples: Labeled positive and negative examples with exact locators.
        false_positive_notes: Situations where the behavior looks present but is acceptable.
        source_evidence: Labels and rubric problems that created or changed the entry.
        status: ``active``, or ``retired`` when merged away or withdrawn.
    """

    linter_id: str = Field(pattern=ENTRY_ID)
    revision: int = Field(default=1, ge=1)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1500)
    detection: str = Field(min_length=1, max_length=2000)
    evidence_needed: tuple[LinterEvidence, ...] = Field(min_length=1)
    applies_when: str = Field(min_length=1, max_length=1000)
    required_when: str = Field(min_length=1, max_length=1000)
    examples: tuple[TraceExample, ...] = ()
    false_positive_notes: tuple[str, ...] = ()
    source_evidence: tuple[str, ...] = Field(min_length=1)
    status: Literal["active", "retired"] = "active"

    def texts(self) -> tuple[str, ...]:
        """Every authored text of the linter."""
        return (
            self.title,
            self.description,
            self.detection,
            self.applies_when,
            self.required_when,
            *(row.detail for row in self.evidence_needed),
            *(row.note for row in self.examples),
            *self.false_positive_notes,
        )


class ScenarioExpectation(ContractModel):
    """What a profile expects in one kind of situation.

    Attributes:
        scenario: Act, defer or stop.
        situation: The situation, stated generally.
        expected: The expected behavior there.
    """

    scenario: Scenario
    situation: str = Field(min_length=1, max_length=1000)
    expected: str = Field(min_length=1, max_length=1000)


class BehaviorProfile(ContractModel):
    """A persistent, cross-task expectation of how an agent uses its capabilities.

    Attributes:
        profile_id: Stable identity, cited as ``profile:<profile_id>``.
        revision: Incremented whenever the entry changes.
        title: Short name, such as scope control.
        expectation: The expectation, independent of any one task.
        linters: Linters that detect violations of it.
        positive_behaviors: Behaviors that satisfy it.
        scenarios: When to act, defer and stop; every one of the three is stated.
        source_evidence: Labels that created or changed the entry.
        status: ``active`` or ``retired``.
    """

    profile_id: str = Field(pattern=ENTRY_ID)
    revision: int = Field(default=1, ge=1)
    title: str = Field(min_length=1, max_length=200)
    expectation: str = Field(min_length=1, max_length=1500)
    linters: tuple[str, ...] = ()
    positive_behaviors: tuple[str, ...] = ()
    scenarios: tuple[ScenarioExpectation, ...] = Field(min_length=3)
    source_evidence: tuple[str, ...] = Field(min_length=1)
    status: Literal["active", "retired"] = "active"

    @model_validator(mode="after")
    def _three_scenarios(self) -> Self:
        """Require a stated expectation for acting, deferring and stopping."""
        missing = {"act", "defer", "stop"} - {row.scenario for row in self.scenarios}
        if missing:
            raise ValueError(f"profile {self.profile_id} states no {sorted(missing)} scenario")
        return self

    def texts(self) -> tuple[str, ...]:
        """Every authored text of the profile."""
        return (
            self.title,
            self.expectation,
            *self.positive_behaviors,
            *(text for row in self.scenarios for text in (row.situation, row.expected)),
        )


class TaskCriterion(ContractModel):
    """A task-scoped correction that belongs with one family's task.

    Attributes:
        criterion_id: Stable identity, cited as ``task_criterion:<criterion_id>``.
        revision: Incremented whenever the entry changes.
        family_id: The only family it is offered to.
        statement: The correction.
        rationale: Why the evidence makes it task-specific.
        examples: Labeled examples from that family.
        source_evidence: Labels that created it.
    """

    criterion_id: str = Field(pattern=ENTRY_ID)
    revision: int = Field(default=1, ge=1)
    family_id: str = Field(min_length=1)
    statement: str = Field(min_length=1, max_length=1500)
    rationale: str = Field(min_length=1, max_length=1500)
    examples: tuple[TraceExample, ...] = ()
    source_evidence: tuple[str, ...] = Field(min_length=1)

    def texts(self) -> tuple[str, ...]:
        """Every authored text of the criterion."""
        return (self.statement, self.rationale, *(row.note for row in self.examples))


class CurationEvidence(ContractModel):
    """One piece of evidence a curator may cite: a behavior label or a rubric problem.

    Attributes:
        evidence_id: Stable identity cited by operations.
        kind: A label from development traces, or a rubric problem found by judging.
        family_id: The family it came from.
        summary: What it shows.
        examples: Exact examples it contributes; the only examples an operation may add.
        hint: The labeler's or reporter's proposal, kept as a hypothesis.
    """

    evidence_id: str = Field(min_length=1, max_length=200)
    kind: Literal["label", "rubric_problem"]
    family_id: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=2000)
    examples: tuple[TraceExample, ...] = ()
    hint: JsonObject = Field(default_factory=dict)


class LinterDraft(ContractModel):
    """A curator's or labeler's proposed new linter, before examples are attached.

    Attributes:
        linter_id: Proposed identity.
        title: Short name.
        description: What the behavior is.
        detection: How to detect it in events.
        evidence_needed: Records detection reads.
        applies_when: When it is relevant.
        required_when: What a task must establish before it is required.
        false_positive_notes: Known acceptable look-alikes.
    """

    linter_id: str = Field(pattern=ENTRY_ID)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1500)
    detection: str = Field(min_length=1, max_length=2000)
    evidence_needed: tuple[LinterEvidence, ...] = Field(min_length=1)
    applies_when: str = Field(min_length=1, max_length=1000)
    required_when: str = Field(min_length=1, max_length=1000)
    false_positive_notes: tuple[str, ...] = ()


class LibraryChange(ContractModel):
    """One applied change, recorded in the version it produced.

    Attributes:
        operation: The applied operation.
        subject: Locator of the changed entry, or ``evidence:<id>`` for a rejection.
        evidence: Evidence ids it rests on.
        families: Families of that evidence.
        rationale: The curator's reason.
    """

    operation: OperationKind
    subject: str = Field(min_length=1)
    evidence: tuple[str, ...] = Field(min_length=1)
    families: tuple[str, ...] = ()
    rationale: str = Field(min_length=1, max_length=1500)


class BehaviorLibrary(ContractModel):
    """One immutable version of a behavior library.

    Attributes:
        protocol: Library protocol.
        library_id: Stable identity of the library across versions.
        version: Version number; 0 is the empty library.
        parent_sha256: Digest of the version it was derived from.
        linters: Every linter, retired ones kept.
        profiles: Every profile.
        task_criteria: Every task-scoped criterion.
        changes: The changes that produced this version from its parent.
        families: Families whose evidence has shaped the library.
    """

    protocol: str = BEHAVIOR_LIBRARY_PROTOCOL
    library_id: str = Field(pattern=ENTRY_ID)
    version: int = Field(default=0, ge=0)
    parent_sha256: Sha256 | None = None
    linters: tuple[BehaviorLinter, ...] = ()
    profiles: tuple[BehaviorProfile, ...] = ()
    task_criteria: tuple[TaskCriterion, ...] = ()
    changes: tuple[LibraryChange, ...] = ()
    families: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Require distinct identities and profiles that group known linters."""
        for name, identities in (
            ("linter", [row.linter_id for row in self.linters]),
            ("profile", [row.profile_id for row in self.profiles]),
            ("task criterion", [row.criterion_id for row in self.task_criteria]),
        ):
            if len(set(identities)) != len(identities):
                raise ValueError(f"library {name} identities must be distinct")
        known = {row.linter_id for row in self.linters}
        unknown = sorted({lid for row in self.profiles for lid in row.linters} - known)
        if unknown:
            raise ValueError(f"profiles group unknown linters {unknown}")
        if (self.version == 0) != (self.parent_sha256 is None):
            raise ValueError("exactly the empty version 0 has no parent")
        return self

    @property
    def identity(self) -> Sha256:
        """Digest of this version."""
        return sha256_json(self)

    def linter(self, linter_id: str) -> BehaviorLinter:
        """Return one linter by identity.

        Raises:
            KeyError: No linter has that identity.
        """
        for row in self.linters:
            if row.linter_id == linter_id:
                return row
        raise KeyError(f"library has no linter {linter_id!r}")

    def offered(self, family_id: str) -> tuple[str, ...]:
        """Return every locator offered to one family's rubric author, sorted.

        Active linters and profiles are offered to every family; a task criterion only to its
        own family.
        """
        return tuple(
            sorted(
                {
                    *(f"linter:{r.linter_id}" for r in self.linters if r.status == "active"),
                    *(f"profile:{r.profile_id}" for r in self.profiles if r.status == "active"),
                    *(
                        f"task_criterion:{r.criterion_id}"
                        for r in self.task_criteria
                        if r.family_id == family_id
                    ),
                }
            )
        )

    def view(self, family_id: str) -> JsonObject:
        """Render what one family's authors and critics may read of this version.

        Args:
            family_id: The family being authored.

        Returns:
            The version identity and every offered entry with its guidance.
        """
        return {
            "library_id": self.library_id,
            "version": self.version,
            "library_sha256": self.identity,
            "entries": [
                self.guidance(locator).model_dump(mode="json")
                for locator in self.offered(family_id)
            ],
        }

    def guidance(self, locator: str) -> BehaviorGuidance:
        """Return the exact judge guidance of one entry.

        Args:
            locator: ``linter:``, ``profile:`` or ``task_criterion:`` locator.

        Returns:
            The guidance copied from this version.

        Raises:
            KeyError: The locator names no entry.
        """
        kind, _, entry = locator.partition(":")
        for linter in self.linters if kind == "linter" else ():
            if linter.linter_id == entry:
                return BehaviorGuidance(
                    locator=locator, kind="linter", revision=linter.revision, title=linter.title,
                    statement=linter.description, detection=linter.detection,
                    evidence_needed=linter.evidence_needed, applies_when=linter.applies_when,
                    required_when=linter.required_when, examples=linter.examples,
                    false_positive_notes=linter.false_positive_notes,
                )  # fmt: skip
        for profile in self.profiles if kind == "profile" else ():
            if profile.profile_id == entry:
                return BehaviorGuidance(
                    locator=locator, kind="profile", revision=profile.revision,
                    title=profile.title, statement=profile.expectation,
                    linters=tuple(f"linter:{lid}" for lid in profile.linters),
                    positive_behaviors=profile.positive_behaviors, scenarios=profile.scenarios,
                )  # fmt: skip
        for row in self.task_criteria if kind == "task_criterion" else ():
            if row.criterion_id == entry:
                return BehaviorGuidance(
                    locator=locator, kind="task_criterion", revision=row.revision,
                    title=row.criterion_id, statement=row.statement, examples=row.examples,
                )  # fmt: skip
        raise KeyError(f"library {self.library_id} v{self.version} has no entry {locator!r}")

    def texts(self) -> tuple[str, ...]:
        """Every authored text in the library."""
        rows: list[str] = []
        for entry in (*self.linters, *self.profiles, *self.task_criteria):
            rows += entry.texts()
        return tuple(rows)


class BehaviorGuidance(ContractModel):
    """One library entry exactly as the judge reads it inside a rubric.

    Attributes:
        locator: The entry's locator.
        kind: Linter, profile or task criterion.
        revision: The entry's revision in the cited version.
        title: Short name.
        statement: Description, expectation or correction.
        detection: How to detect the behavior in events (linters).
        evidence_needed: Records detection reads (linters).
        applies_when: When it is relevant (linters).
        required_when: What makes it required correctness (linters).
        examples: Labeled positive and negative examples.
        false_positive_notes: Acceptable look-alikes (linters).
        linters: Linters a profile groups.
        positive_behaviors: Behaviors that satisfy a profile.
        scenarios: A profile's act, defer and stop expectations.
    """

    locator: str = Field(min_length=1)
    kind: EntryKind
    revision: int = Field(ge=1)
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    detection: str | None = None
    evidence_needed: tuple[LinterEvidence, ...] = ()
    applies_when: str | None = None
    required_when: str | None = None
    examples: tuple[TraceExample, ...] = ()
    false_positive_notes: tuple[str, ...] = ()
    linters: tuple[str, ...] = ()
    positive_behaviors: tuple[str, ...] = ()
    scenarios: tuple[ScenarioExpectation, ...] = ()


class LibraryCitation(ContractModel):
    """The exact library version a rubric was assembled from.

    Attributes:
        reference: The saved ``behavior-library`` artifact.
        library_id: Library identity.
        version: Version number.
        library_sha256: Digest of that version.
    """

    reference: ArtifactInput
    library_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    library_sha256: Sha256


class LibraryBinding(ContractModel):
    """A saved library version bound for one authoring, revision or review.

    Attributes:
        reference: The saved ``behavior-library`` artifact.
        library: Its content.
    """

    reference: ArtifactInput
    library: BehaviorLibrary

    def citation(self) -> LibraryCitation:
        """Return the citation a rubric records."""
        return LibraryCitation(
            reference=self.reference,
            library_id=self.library.library_id,
            version=self.library.version,
            library_sha256=self.library.identity,
        )


class LibraryDecision(ContractModel):
    """How one offered library entry applies to the family a rubric is authored for.

    Attributes:
        entry: The entry's locator.
        decision: Required correctness, diagnostic only, or inapplicable.
        reason: Why.
        cited: Packet locators that support the decision; a required one cites the task or a
            source, never only the library or a hypothesis.
    """

    entry: str = Field(min_length=1)
    decision: Literal["required", "diagnostic", "inapplicable"]
    reason: str = Field(min_length=1, max_length=1200)
    cited: tuple[str, ...] = ()


def check_library_decisions(
    decisions: Sequence[LibraryDecision],
    roles: Mapping[str, set[str]],
    offered: set[str],
    unsupported: set[str],
) -> None:
    """Require one decision per offered entry, matched by the roles of the criteria citing it.

    A ``required`` entry needs a required correctness criterion citing it and task support; a
    ``diagnostic`` one only diagnostic criteria; an ``inapplicable`` one no criterion.

    Args:
        decisions: The author's decisions.
        roles: Roles of the criteria citing each offered entry.
        offered: Every offered entry locator.
        unsupported: Locators that cannot establish an obligation (hypotheses and the library).

    Raises:
        ValueError: An entry is undecided or decided twice, a decision disagrees with the
            criteria citing it, or a required decision rests only on unsupported locators.
    """
    decided = [row.entry for row in decisions]
    if len(set(decided)) != len(decided) or set(decided) != offered:
        raise ValueError(
            f"library decisions must decide each offered entry {sorted(offered)} once, "
            f"not {decided}"
        )
    for row in decisions:
        found = roles.get(row.entry, set())
        matches = {
            "required": "required_correctness" in found,
            "diagnostic": found == {"efficiency_diagnostic"},
            "inapplicable": not found,
        }[row.decision]
        if not matches:
            raise ValueError(
                f"library entry {row.entry} is decided {row.decision} but criteria citing it "
                f"have roles {sorted(found)}"
            )
        if row.decision == "required" and set(row.cited) <= unsupported:
            raise ValueError(
                f"library entry {row.entry} is required without task support; cite the task or "
                "a source that establishes it, or make it a diagnostic"
            )


def empty_library(library_id: str) -> BehaviorLibrary:
    """Return version 0 of a library: no entries, no parent."""
    return BehaviorLibrary(library_id=library_id)


def actor_mentions(texts: Sequence[str], names: Iterable[str]) -> tuple[str, ...]:
    """Return the actor names any text mentions, by exact identifier match, sorted.

    A name matches only as a whole identifier, never as part of a longer one.

    Args:
        texts: Authored texts.
        names: Tier labels and model names.

    Returns:
        Every mentioned name.
    """
    joined = "\n".join(texts).lower()
    found: set[str] = set()
    for name in names:
        lowered = name.lower()
        start = joined.find(lowered)
        while start >= 0:
            before = joined[start - 1] if start else " "
            end = start + len(lowered)
            after = joined[end] if end < len(joined) else " "
            if not _identifier(before) and not (after.isalnum() or after in "_-"):
                found.add(name)
                break
            start = joined.find(lowered, start + 1)
    return tuple(sorted(found))


def _identifier(character: str) -> bool:
    """Whether a character continues an identifier on its left."""
    return character.isalnum() or character in "_.-"


def refuse_actor_exposure(library: BehaviorLibrary, actor_texts: Sequence[str]) -> None:
    """Refuse actor-visible text that reproduces judge-only library guidance.

    Library detection guidance, false-positive notes and examples are judge configuration. An
    actor's task, briefing or messages must not contain any of them verbatim.

    Args:
        library: The library version in use.
        actor_texts: Every text an actor will be shown.

    Raises:
        ValueError: An actor-visible text contains library guidance.
    """
    private = [
        text
        for row in library.linters
        for text in (row.detection, *row.false_positive_notes, *(e.note for e in row.examples))
    ]
    private += [e.note for row in library.task_criteria for e in row.examples]
    shown = "\n".join(actor_texts)
    leaked = sorted({text for text in private if len(text) >= 24 and text in shown})
    if leaked:
        raise ValueError(
            f"{len(leaked)} judge-only library texts reach the actor, for example "
            f"{leaked[0][:120]!r}; keep library guidance in the rubric only"
        )


def library_artifact_id(library_id: str, version: int) -> str:
    """Return the fixed artifact identity of one library version."""
    return stable_id(LIBRARY_TYPE, {"library": library_id, "version": version})


def save_library(
    project: ProjectStore,
    library: BehaviorLibrary,
    *,
    parent: ArtifactInput | None,
    sources: Sequence[ArtifactInput] = (),
    code_revision: str,
) -> ArtifactInput:
    """Save one library version, citing its parent and the saved evidence behind it.

    Args:
        project: Project receiving the artifact.
        library: The version, at least 1.
        parent: The saved parent version; None only for version 1 derived from version 0.
        sources: Saved label runs, curations or revisions the changes came from.
        code_revision: Producing source identity.

    Returns:
        The saved reference; an identical saved version is adopted.

    Raises:
        ValueError: The version is 0, its parent does not match, or another version with the
            same number exists.
    """
    if library.version < 1:
        raise ValueError("version 0 is the empty library and is never saved")
    if parent is not None and load_library(project, parent).identity != library.parent_sha256:
        raise ValueError("the parent reference is not the version this one was derived from")
    if parent is None and library.parent_sha256 != empty_library(library.library_id).identity:
        raise ValueError(f"version {library.version} needs its saved parent version")
    store = project.artifacts
    artifact_id = library_artifact_id(library.library_id, library.version)
    if store.exists(artifact_id):
        reference = artifact_input(store.read(artifact_id).manifest)
        if load_library(project, reference) != library:
            raise ValueError(
                f"library {library.library_id} already has a different version "
                f"{library.version}; curate from the current head instead"
            )
        return reference
    inputs = {*sources, *((parent,) if parent is not None else ())}
    manifest = store.write_json(
        artifact_id=artifact_id,
        artifact_type=LIBRARY_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
        ),
        files={LIBRARY_FILE: library},
    )
    return artifact_input(manifest)


def load_library(project: ProjectStore, reference: ArtifactInput) -> BehaviorLibrary:
    """Read one exact saved library version.

    Raises:
        ValueError: The reference is not the saved version it pins.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        LIBRARY_TYPE
    ):
        raise ValueError(f"artifact {reference.artifact_id} is not a pinned {LIBRARY_TYPE}")
    return BehaviorLibrary.model_validate_json(stored.payloads[LIBRARY_FILE])


def library_head(project: ProjectStore, library_id: str) -> LibraryBinding | None:
    """Return the newest saved version of a library, or None when none is saved.

    Args:
        project: Project holding the versions.
        library_id: Library identity.

    Returns:
        The newest version and its reference.
    """
    store = project.artifacts
    head: LibraryBinding | None = None
    version = 1
    while store.exists(library_artifact_id(library_id, version)):
        reference = artifact_input(store.read(library_artifact_id(library_id, version)).manifest)
        head = LibraryBinding(reference=reference, library=load_library(project, reference))
        version += 1
    return head

"""Contracts of the start bootstrap: its registration, drafts, reviews, outcomes and batch manifest.

The bootstrap (:mod:`~exp.optimize.evaluation.variance.bootstrap`) turns selected environment
families into registered mining jobs. A :class:`BootstrapSpec` names the families (each read
through the small :class:`FamilyInput` adapter, with its selected tasks), the rules registration
a family without frozen rules learns with, the agentic roles and their prompt variants, the
runtime the starts are captured in and the mining job each registration freezes.

Every start records its provenance (:data:`Provenance`) and the validation that provenance needs:

========================  ===============================================================
Provenance                Validation
========================  ===============================================================
``observed``              Exact restore of the recorded state snapshot
``reconstructed``         Replay of the recorded prefix agrees with what was recorded
``authored``              Coherence review plus a first-reveal consistency check
========================  ===============================================================

Only an observed start claims to be the original environment; an authored start never does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
)
from exp.common.tasks import TaskCase, ToolSchema
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    RuleMemoryBinding,
)
from exp.optimize.evaluation.executable.state_recall import StateRecallBinding
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.agent_loop import AgentProfile
from exp.optimize.evaluation.rules.stage import RulesStageSpec
from exp.optimize.evaluation.variance.bootstrap_prompts import (
    AUTHOR_VARIANTS,
    REVEAL_VARIANTS,
    REVIEW_VARIANTS,
)
from exp.optimize.evaluation.variance.bootstrap_prompts import (
    RESEARCH_VARIANTS as RESEARCHER_VARIANTS,
)
from exp.runtime.environments.world_contracts import WorldManifest
from exp.simulation.evaluation.tasks import TaskVerifier
from exp.simulation.mining.decisions import DEFAULT_DECISION_ALIAS
from exp.simulation.retrieval.scope import CorpusScope
from exp.simulation.rules.model import ToolContract
from exp.simulation.world_model.state_memory import StateMemoryPolicy

Provenance = Literal["observed", "reconstructed", "authored"]
"""How a start relates to the original environment."""

FactStatus = Literal["observed", "inferred", "unknown"]
"""How a fact of the start is known."""

Verdict = Literal["accept", "research", "refuse"]
"""A reviewer's verdict on one draft."""

EntryStatus = Literal["registered", "refused", "failed"]
"""A task's state in the batch manifest: two saved outcomes and one retryable failure."""

RefusalKind = Literal[
    "author_unusable",
    "review_unusable",
    "review_refused",
    "research_exhausted",
    "validation_failed",
]
"""Why a start was refused; every refusal is saved and replays."""

BATCH_TYPE = "start-bootstrap-batch"
"""Artifact type of a frozen copy of the batch manifest."""

REGISTRATION_TYPE = "start-bootstrap-registration"
"""Artifact type of one registered start and its mining job."""

REFUSAL_TYPE = "start-bootstrap-refusal"
"""Artifact type of one diagnostic refusal."""

RECORDED_STATE = "state.json"
"""File of a recorded environment state snapshot artifact."""

MAX_REVEAL_PROBES = 16
"""Finite number of fresh simulator lifetimes an authored start may reveal."""

PROVENANCE_STATEMENTS: dict[Provenance, str] = {
    "observed": "Observed start: the recorded environment state, restored exactly.",
    "reconstructed": (
        "Reconstructed start: rebuilt from recorded evidence; the recorded prefix replays to "
        "the recorded observations. It is not the original environment."
    ),
    "authored": (
        "Authored start: generated from the family's evidence and reviewed for coherence. It "
        "is not, and never claims to be, the original environment."
    ),
}
"""The statement each registration carries about its start."""

_ID = r"^[a-z0-9][a-z0-9_.-]{0,63}$"


class BootstrapTask(ContractModel):
    """One selected task of an environment family.

    Attributes:
        task: The selected development task; its identity, lineage, partition and sources are
            kept, its instruction and tools are authored.
        source_traces: Mining-probe traces that recorded this task; the author reads them and a
            reconstructed start replays one of them.
        recorded_state: Saved snapshot of the original environment (``state.json``), when one
            was recorded; only then may a start be observed.
    """

    task: TaskCase
    source_traces: tuple[str, ...] = ()
    recorded_state: ArtifactInput | None = None


class RuntimeTemplate(ContractModel):
    """The grounded runtime every start is captured in, without the per-task world.

    Attributes:
        runtime_file: Trusted file exporting ``continuation_runtime`` over a grounded definition.
        runtime_sha256: Its exact source digest.
        world_model: Build-produced grounded world model.
        catalog_sha256: Frozen catalog digest.
        tag_retrieval: Optional tag retrieval binding.
        bind_rule_book: Bind the family's rule book (and its rule memory set, when learned).
        jev_alias: Native decision alias rule memory recall uses.
        seed: World scheduler seed.
        maximum_output_tokens: Per-prediction output ceiling.
        maximum_reasks: Extra proposals after known contract refusals.
        maximum_physical_attempts: Provider attempt ceiling of each served world.
        deadline: Deadline of every simulator call.
        state_recall: Each family's pinned state-tag index and policy, by family; a start of a
            listed family retrieves recorded transitions by pre-turn state agreement.
        state_memory: Per-task state description policy every start's simulator uses.
    """

    runtime_file: Path
    runtime_sha256: Sha256
    world_model: ArtifactInput
    catalog_sha256: Sha256
    tag_retrieval: ArtifactInput | None = None
    bind_rule_book: bool = True
    jev_alias: str = Field(default=DEFAULT_DECISION_ALIAS, min_length=1)
    seed: int = 0
    maximum_output_tokens: int = Field(default=4096, ge=1, le=65536)
    maximum_reasks: int = Field(default=0, ge=0, le=3, strict=True)
    maximum_physical_attempts: int = Field(ge=1, le=100_000)
    deadline: AwareDatetime
    state_recall: dict[str, StateRecallBinding] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    state_memory: StateMemoryPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class FamilyInput(ContractModel):
    """The bootstrap's input for one environment family, with its frozen knowledge exact.

    Build it from the families owner's shared simulator with :func:`family_from_simulator`;
    simulation code never imports this optimize type.

    Attributes:
        family_id: Family identity.
        member_traces: Member world-learning traces the roles may read.
        rules_traces: World-learning traces the family's rules are learned from; defaults to
            ``member_traces``. Read only when the family has no frozen rules.
        tool_contracts: Reconciled tool contracts by tool name; they take precedence over the
            contracts the rule book parsed.
        state_domains: Names of the family's shared state domains as its reviewers named them.
            They are descriptive: the bootstrap decides executable ownership, and every
            bootstrapped domain is simulated. When given, every start declares exactly these.
        corpus: The family's retrieval corpus, which may select a subset of family members.
        rule_book: The family's frozen ``rule-book``; with it no rules are learned here.
        rule_memory: The family's frozen rule memory set and recall policy, kept exactly.
        rules_report: The ``rules-stage-report`` that produced ``rule_book``, when saved; the
            pipeline's families stage needs it, so a bootstrapped pipeline run requires it.
        runtime: The family's frozen runtime definition; each start replaces only its task,
            manifest, initial state and evaluator projection. None builds the definition from
            the spec's runtime template.
    """

    family_id: str = Field(pattern=_ID)
    member_traces: tuple[str, ...] = Field(min_length=1)
    rules_traces: tuple[str, ...] = ()
    tool_contracts: dict[str, ToolContract] = Field(default_factory=dict)
    state_domains: tuple[str, ...] = ()
    corpus: CorpusScope | None = None
    rule_book: ArtifactInput | None = None
    rule_memory: RuleMemoryBinding | None = None
    rules_report: ArtifactInput | None = None
    runtime: GroundedRuntimeDefinition | None = None

    @model_validator(mode="after")
    def _scoped(self) -> Self:
        """Require scoped members, memory only beside its book, and a runtime bound to both."""
        if self.corpus is not None and self.corpus.kind == "trace":
            outside = sorted(set(self.corpus.ids) - set(self.member_traces))
            if outside:
                raise ValueError(f"retrieval traces outside the family members: {outside}")
        outside_rules = sorted(set(self.rules_traces) - set(self.member_traces))
        if outside_rules:
            raise ValueError(f"rules traces outside the family members: {outside_rules}")
        if self.rule_memory is not None and self.rule_book is None:
            raise ValueError("a frozen rule memory comes with its frozen rule book")
        if self.rules_report is not None and self.rule_book is None:
            raise ValueError("a frozen rules report names its rule book; pass rule_book too")
        if self.runtime is not None and (
            self.runtime.rule_book != self.rule_book or self.runtime.rule_memory != self.rule_memory
        ):
            raise ValueError("the family runtime binds other rules than the family's frozen rules")
        return self

    @property
    def learning_traces(self) -> tuple[str, ...]:
        """Return the traces rules are learned from when the family has none frozen."""
        return self.rules_traces or self.member_traces

    @property
    def frozen(self) -> bool:
        """Whether the family brings its own rules, so the bootstrap learns none."""
        return self.rule_book is not None


class NamedDomain(Protocol):
    """A reviewed state domain; only its name crosses into the bootstrap."""

    @property
    def name(self) -> str:
        """The domain name."""
        ...


class FamilySimulatorFields(Protocol):
    """The fields of the families owner's ``FamilySimulator`` the bootstrap reads."""

    @property
    def family_id(self) -> str:
        """Family identity."""
        ...

    @property
    def members(self) -> tuple[str, ...]:
        """Member world-learning traces."""
        ...

    @property
    def rules_traces(self) -> tuple[str, ...]:
        """Traces its rule memory is learned from."""
        ...

    @property
    def tool_contracts(self) -> Mapping[str, ToolContract]:
        """Reconciled tool contracts."""
        ...

    @property
    def state_domains(self) -> Sequence[NamedDomain]:
        """Reviewed shared state domains."""
        ...

    @property
    def corpus(self) -> CorpusScope | None:
        """Scoped corpus."""
        ...

    @property
    def rule_book(self) -> ArtifactInput | None:
        """Frozen rule book."""
        ...

    @property
    def rule_memory(self) -> RuleMemoryBinding | None:
        """Frozen rule memory binding."""
        ...

    @property
    def rules_report(self) -> ArtifactInput | None:
        """The rules-stage report its rule book came from."""
        ...

    @property
    def runtime(self) -> GroundedRuntimeDefinition | None:
        """Frozen runtime definition."""
        ...


def family_from_simulator(simulator: FamilySimulatorFields) -> FamilyInput:
    """Adapt the families owner's shared simulator to the bootstrap input, field by field.

    The simulator (``FamilySimulator``) is read by attribute, so this module does not depend on
    its owner. Read fields: ``family_id``, ``members``, ``rules_traces``, ``tool_contracts``,
    ``state_domains`` (each item's ``name``), ``corpus``, ``rule_book``, ``rule_memory``,
    ``rules_report`` and ``runtime``. Nothing is renamed silently and nothing is invented; the
    simulator's ``index``, ``world_model`` and ``tag_retrieval`` reach the starts through its
    frozen ``runtime``, which the bootstrap keeps exactly.

    Args:
        simulator: The family's shared simulator definition.

    Returns:
        The validated input.
    """
    domains = tuple(domain.name for domain in simulator.state_domains)
    return FamilyInput.model_validate(
        {
            "family_id": simulator.family_id,
            "member_traces": simulator.members,
            "rules_traces": simulator.rules_traces,
            "tool_contracts": dict(simulator.tool_contracts),
            "state_domains": domains,
            "corpus": simulator.corpus,
            "rule_book": simulator.rule_book,
            "rule_memory": simulator.rule_memory,
            "rules_report": simulator.rules_report,
            "runtime": simulator.runtime,
        }
    )


class BootstrapFamily(ContractModel):
    """One selected environment family and its selected tasks.

    Attributes:
        family: The family.
        tasks: Selected tasks bootstrapped in this family.
    """

    family: FamilyInput
    tasks: tuple[BootstrapTask, ...] = Field(min_length=1)


class JobTemplate(ContractModel):
    """The mining job every registered start freezes.

    Attributes:
        actor_alias: Catalog alias of the actor.
        maximum_model_calls: Actor horizon (policy call ceiling); None uses the full length of
            the task's longest source trajectory (its model calls), so decision points can be
            found anywhere in a long trajectory.
        horizon_ceiling: Most calls a source-length horizon may grant; an authored start with no
            source trajectory gets this ceiling.
        system_prompt: Actor policy instruction.
        maximum_physical_attempts: Provider attempt ceiling of the job.
        deadline: Deadline of every actor call.
        repeats: Continuations per study.
        temperature: Actor sampling temperature.
        maximum_output_tokens: Actor per-call output allowance.
    """

    actor_alias: str = Field(min_length=1)
    maximum_model_calls: int | None = Field(default=None, ge=1, le=1000)
    horizon_ceiling: int = Field(default=1000, ge=1, le=1000, strict=True)
    system_prompt: str | None = None
    maximum_physical_attempts: int = Field(ge=1, le=100_000)
    deadline: AwareDatetime
    repeats: int = Field(default=4, ge=1, le=100, strict=True)
    temperature: float = Field(default=1.0, ge=0, le=2)
    maximum_output_tokens: int = Field(default=1024, gt=0)

    def with_horizon(self, source_calls: int) -> JobTemplate:
        """Return the template with its horizon resolved for one task.

        Args:
            source_calls: Model calls in the task's longest source trajectory; zero when it has
                none.

        Returns:
            The template with ``maximum_model_calls`` set: the explicit value, else the source
            length within the ceiling, else the ceiling.
        """
        if self.maximum_model_calls is not None:
            return self
        calls = min(self.horizon_ceiling, source_calls) if source_calls else self.horizon_ceiling
        return self.model_copy(update={"maximum_model_calls": calls})

    @property
    def horizon(self) -> int:
        """The resolved horizon; only a template resolved for a task has one."""
        if self.maximum_model_calls is None:
            raise ValueError("the job horizon is resolved per task from its source trajectory")
        return self.maximum_model_calls


class BootstrapSettings(ContractModel):
    """Bootstrap role and execution settings before family tasks are selected.

    Attributes:
        rules: Rules registration a family without frozen rules learns its rules and memory
            with; required only then.
        author: Start author settings.
        reviewer: Reviewer settings; None uses ``author``.
        researcher: Researcher and first-reveal checker settings; None uses ``author``.
        author_profile: Budgets of one author draw.
        review_profile: Budgets of one review draw.
        research_profile: Budgets of one research or first-reveal draw.
        author_variants: Author stances, one assigned per task and round.
        review_variants: Reviewer stances; every stance reviews every draft.
        research_variants: Researcher stances, assigned per research item.
        reveal_variants: First-reveal checker stances; every stance checks an authored start.
        research_rounds: Most research rounds per task before a still-missing deciding fact
            becomes a diagnostic refusal.
        research_items: Most research items answered per round.
        author_repairs: Most validation-repair draws after an author draft fails its checks; each
            sees the failed draft and its validation errors and resubmits the complete start.
        runtime: The trusted runtime file starts are captured in, and the definition fields
            used when a family brings no frozen runtime.
        job: The mining job each registration freezes.
        concurrency: Tasks bootstrapped at once; not part of any identity.
    """

    rules: RulesStageSpec | None = None
    author: AuthorSettings
    reviewer: AuthorSettings | None = None
    researcher: AuthorSettings | None = None
    author_profile: AgentProfile = AgentProfile()
    review_profile: AgentProfile = AgentProfile()
    research_profile: AgentProfile = AgentProfile()
    author_variants: tuple[str, ...] = Field(default=tuple(AUTHOR_VARIANTS), min_length=1)
    review_variants: tuple[str, ...] = Field(default=tuple(REVIEW_VARIANTS), min_length=2)
    research_variants: tuple[str, ...] = Field(default=tuple(RESEARCHER_VARIANTS), min_length=1)
    reveal_variants: tuple[str, ...] = Field(default=tuple(REVEAL_VARIANTS), min_length=1)
    research_rounds: int = Field(default=2, ge=0, le=4, strict=True)
    research_items: int = Field(default=4, ge=1, le=16, strict=True)
    author_repairs: int = Field(default=2, ge=0, le=4, strict=True)
    runtime: RuntimeTemplate
    job: JobTemplate
    concurrency: int = Field(default=1, ge=1, le=32, strict=True)

    @model_validator(mode="after")
    def _closed(self) -> Self:
        """Require distinct known prompt stances for every bootstrap role."""
        for chosen, known in (
            (self.author_variants, AUTHOR_VARIANTS),
            (self.review_variants, REVIEW_VARIANTS),
            (self.research_variants, RESEARCHER_VARIANTS),
            (self.reveal_variants, REVEAL_VARIANTS),
        ):
            unknown = sorted(set(chosen) - set(known))
            if unknown or len(set(chosen)) != len(chosen):
                raise ValueError(f"prompt variants must be distinct known stances: {unknown}")
        return self

    def role(self, name: Literal["reviewer", "researcher"]) -> AuthorSettings:
        """Return a role's settings, falling back to the author's."""
        settings = self.reviewer if name == "reviewer" else self.researcher
        return settings or self.author


class BootstrapSpec(BootstrapSettings):
    """One registered batch with exact selected families and their tasks.

    Attributes:
        families: Selected environment families, each with its selected tasks.
    """

    families: tuple[BootstrapFamily, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _families_closed(self) -> Self:
        """Require distinct family and task identities and available family knowledge."""
        families = [row.family.family_id for row in self.families]
        if len(set(families)) != len(families):
            raise ValueError("family identities repeat")
        tasks = [task.task.task_id for row in self.families for task in row.tasks]
        if len(set(tasks)) != len(tasks):
            raise ValueError("a task is selected twice; each task belongs to one family")
        if self.rules is None and not all(row.family.frozen for row in self.families):
            raise ValueError("a family without frozen rules needs the spec's rules registration")
        return self


class FactClaim(ContractModel):
    """One fact of a start, with how it is known and what supports it.

    Attributes:
        fact_id: Identity within the draft.
        statement: The fact in one sentence.
        status: Observed, inferred or unknown.
        deciding: Whether the right course of action depends on it.
        state_paths: JSON pointers into the initial state the fact is about.
        references: Inventory references supporting it; required unless unknown.
    """

    fact_id: str = Field(pattern=_ID)
    statement: str = Field(min_length=1)
    status: FactStatus
    deciding: bool
    state_paths: tuple[str, ...] = ()
    references: tuple[str, ...] = ()


class RevealProbe(ContractModel):
    """One read-only call executed once from a fresh copy of an authored start.

    Attributes:
        name: Tool name; the manifest must declare it without writes.
        arguments: Call arguments.
    """

    name: str = Field(min_length=1)
    arguments: JsonObject = Field(default_factory=dict)


class StartDraft(ContractModel):
    """The author's starting state for one task.

    Attributes:
        instruction: Public task instruction.
        tools: Tool schemas the actor sees.
        manifest: World manifest; every domain simulated.
        initial_state: Exact initial state of every domain.
        probe_paths: Evaluator field names to JSON pointers into the state.
        verifier: Declared assessment predicate over the final state.
        provenance: Observed, reconstructed or authored.
        replay_trace: Source trace whose prefix a reconstructed start replays.
        prefix_steps: Recorded steps of that trace the replay must reproduce.
        facts: The fact ledger.
        reveal_probes: Read-only first-reveal probes of an authored start.
        limitations: What the author could not settle.
    """

    instruction: str = Field(min_length=1)
    tools: tuple[ToolSchema, ...] = Field(min_length=1)
    manifest: WorldManifest
    initial_state: JsonObject
    probe_paths: dict[str, str] = Field(min_length=1, max_length=128)
    verifier: TaskVerifier
    provenance: Provenance
    replay_trace: str | None = None
    prefix_steps: int = Field(default=0, ge=0, le=64)
    facts: tuple[FactClaim, ...] = Field(min_length=1)
    reveal_probes: tuple[RevealProbe, ...] = Field(default=(), max_length=MAX_REVEAL_PROBES)
    limitations: tuple[str, ...] = ()


class ResearchRequest(ContractModel):
    """A missing deciding fact a reviewer names.

    Attributes:
        fact_id: The draft fact it concerns, or None for a fact the draft lacks.
        question: What must be found out.
        why_deciding: Why the right course of action depends on it.
    """

    fact_id: str | None = None
    question: str = Field(min_length=1)
    why_deciding: str = Field(min_length=1)


class ReviewFinding(ContractModel):
    """One reviewer finding.

    Attributes:
        issue: What is wrong or doubtful.
        fact_ids: Draft facts it concerns.
        references: Inventory references supporting it.
    """

    issue: str = Field(min_length=1)
    fact_ids: tuple[str, ...] = ()
    references: tuple[str, ...] = ()


class ReviewAnswer(ContractModel):
    """One reviewer's answer.

    Attributes:
        verdict: Accept, research or refuse.
        summary: The review in brief.
        findings: Cited findings.
        research_items: Missing deciding facts; required for ``research``.
        refusal: The diagnosis; required for ``refuse``.
    """

    verdict: Verdict
    summary: str = Field(min_length=1)
    findings: tuple[ReviewFinding, ...] = ()
    research_items: tuple[ResearchRequest, ...] = ()
    refusal: str | None = None

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Require research items for research and a diagnosis for a refusal."""
        if self.verdict == "research" and not self.research_items:
            raise ValueError("a research verdict names at least one research item")
        if self.verdict == "refuse" and not self.refusal:
            raise ValueError("a refusal gives its diagnosis")
        return self


class ResearchItem(ResearchRequest):
    """A research work item fed back to the family's research step.

    Attributes:
        item_id: Identity within the task, ``<round>-<n>``.
        round: Review round that raised it.
    """

    item_id: str = Field(pattern=_ID)
    round: int = Field(ge=0)


class Finding(ContractModel):
    """The research step's answer to one item.

    Attributes:
        finding_id: Identity, cited as ``finding:<finding_id>``.
        item: The research item.
        variant: Researcher stance.
        transcript: Saved draw, when the draw ran.
        status: Resolved or unresolved.
        answer: The answer, or what is missing.
        fact_status: How the answer is known.
        evidence: References the researcher read and cites.
        failure: Why the draw gave no usable answer.
    """

    finding_id: str = Field(pattern=_ID)
    item: ResearchItem
    variant: str
    transcript: ArtifactInput | None = None
    status: Literal["resolved", "unresolved"] = "unresolved"
    answer: str = ""
    fact_status: FactStatus = "unknown"
    evidence: tuple[str, ...] = ()
    failure: str | None = None


class RoleDraw(ContractModel):
    """One agentic draw of a role.

    Attributes:
        role: Role name.
        variant: Stance.
        round: Bootstrap round.
        transcript: Saved draw.
        answer: Its validated answer, or None; after a successful repair, the repaired answer.
        failure: Why there is no usable answer, after any repairs.
        invalid: Why this draw's own answer failed its checks, when repair draws followed.
        repairs: The validation-repair draws that followed, in order; the first one carrying an
            answer supplied the draw's answer.
    """

    role: str
    variant: str
    round: int = Field(ge=0)
    transcript: ArtifactInput | None = None
    answer: JsonObject | None = None
    failure: str | None = None
    invalid: str | None = None
    repairs: tuple[RoleDraw, ...] = ()

    def transcripts(self) -> tuple[ArtifactInput, ...]:
        """Return this draw's saved transcript and every repair draw's, in order."""
        own = () if self.transcript is None else (self.transcript,)
        return (*own, *(ref for row in self.repairs for ref in row.transcripts()))


class ReviewRound(ContractModel):
    """One round's draft and its reviews.

    Attributes:
        round: Round number.
        author: The author draw.
        reviews: Every reviewer draw.
        verdict: The reconciled verdict: refuse when any usable review refuses, research when
            any asks for research or a deciding fact is unknown, accept otherwise.
        items: Research items raised this round.
    """

    round: int = Field(ge=0)
    author: RoleDraw
    reviews: tuple[RoleDraw, ...] = ()
    verdict: Verdict | None = None
    items: tuple[ResearchItem, ...] = ()


class ReplayStep(ContractModel):
    """One recorded step replayed in a reconstructed start.

    Attributes:
        locator: The recorded call's locator.
        action: Tool name and arguments.
        recorded: The recorded observation.
        recorded_error: Whether the recorded call failed.
        replayed: The replayed observation, decoded when it is JSON.
        replayed_error: Whether the replayed call failed.
        agrees: Whether error status and observation are equal.
    """

    locator: str
    action: JsonObject
    recorded: JsonValue
    recorded_error: bool
    replayed: JsonValue
    replayed_error: bool
    agrees: bool


class Reveal(ContractModel):
    """What one reveal probe showed from a fresh copy of an authored start.

    Attributes:
        probe: Probe index.
        name: Tool name.
        arguments: Call arguments.
        content: Observation, decoded when it is JSON.
        is_error: Whether the call failed.
    """

    probe: int = Field(ge=0)
    name: str
    arguments: JsonObject
    content: JsonValue
    is_error: bool


class StartValidation(ContractModel):
    """The validation a start's provenance requires, and its result.

    Attributes:
        provenance: The start's provenance.
        method: ``exact_restore``, ``prefix_replay`` or ``coherence_and_first_reveal``.
        passed: Whether it held.
        detail: Why it did not hold, or what held.
        replay: Replayed steps of a reconstructed start.
        reveals: First reveals of an authored start.
        checks: First-reveal checker draws.
    """

    provenance: Provenance
    method: Literal["exact_restore", "prefix_replay", "coherence_and_first_reveal"]
    passed: bool
    detail: str
    replay: tuple[ReplayStep, ...] = ()
    reveals: tuple[Reveal, ...] = ()
    checks: tuple[RoleDraw, ...] = ()


class BootstrapRegistration(ContractModel):
    """One registered start and its mining job.

    Attributes:
        family_id: Environment family.
        task_id: Task.
        identity: Task bootstrap identity.
        provenance: How the start relates to the original environment.
        claims_original: True only for an observed start.
        statement: The provenance statement.
        rounds: Every authoring and review round.
        findings: Every research finding.
        rules_report: The family's saved rules-stage report, when one is known.
        rulebook: The family's rule book.
        rule_memory: The family's rule memory binding the start's runtime recalls from.
        draft: The accepted draft.
        checkpoint: The captured start.
        validation: The provenance validation.
        job: The registered mining job, runnable by the actor-critic stage.
    """

    family_id: str
    task_id: str
    identity: str
    provenance: Provenance
    claims_original: bool
    statement: str
    rounds: tuple[ReviewRound, ...]
    findings: tuple[Finding, ...] = ()
    rules_report: ArtifactInput | None = None
    rulebook: ArtifactInput
    rule_memory: RuleMemoryBinding | None = None
    draft: StartDraft
    checkpoint: ArtifactInput
    validation: StartValidation
    job: ContinuationJob

    @model_validator(mode="after")
    def _honest(self) -> Self:
        """Only an observed start may claim to be the original environment."""
        if self.claims_original != (self.provenance == "observed"):
            raise ValueError("only an observed start claims to be the original environment")
        if self.statement != PROVENANCE_STATEMENTS[self.provenance]:
            raise ValueError("the provenance statement differs from the start's provenance")
        return self


class BootstrapRefusal(ContractModel):
    """A diagnostic refusal: the task gets no start, and why.

    Attributes:
        family_id: Environment family.
        task_id: Task.
        identity: Task bootstrap identity.
        kind: Why.
        diagnosis: The diagnosis.
        rounds: Every authoring and review round.
        findings: Every research finding.
        unresolved: Research items still open.
        validation: The failed validation, for ``validation_failed``.
    """

    family_id: str
    task_id: str
    identity: str
    kind: RefusalKind
    diagnosis: str = Field(min_length=1)
    rounds: tuple[ReviewRound, ...] = ()
    findings: tuple[Finding, ...] = ()
    unresolved: tuple[ResearchItem, ...] = ()
    validation: StartValidation | None = None


class BootstrapBinding(ContractModel):
    """One task's registered start in a frozen batch, as a pipeline registration pins it.

    Attributes:
        batch: Frozen ``start-bootstrap-batch`` artifact.
        task_id: The task whose registered mining job the pipeline runs.
    """

    batch: ArtifactInput
    task_id: str = Field(min_length=1)


class ManifestEntry(ContractModel):
    """One task's row in the batch manifest.

    Attributes:
        family_id: Environment family.
        task_id: Task.
        identity: Task bootstrap identity; the outcome artifact is ``<identity>-registration``
            or ``<identity>-refusal``.
        status: Registered, refused (both saved and replayed) or failed (retried on resume).
        outcome: The saved outcome artifact.
        provenance: The registered start's provenance.
        job_sha256: Digest of the registered mining job.
        failure: Why the last attempt failed.
        attempts: Attempts that ran in this batch (replays not counted).
        updated_at: When the row last changed.
    """

    family_id: str
    task_id: str
    identity: str
    status: EntryStatus
    outcome: ArtifactInput | None = None
    provenance: Provenance | None = None
    job_sha256: Sha256 | None = None
    failure: str | None = None
    attempts: int = Field(default=0, ge=0)
    updated_at: datetime


class BatchManifest(ContractModel):
    """The batch manifest: one row per task that has run, updated as each task finishes.

    Attributes:
        protocol: Bootstrap protocol.
        batch: Batch identity.
        spec_sha256: Digest of the registration.
        code_revision: Producing source identity.
        entries: Rows in spec order.
    """

    protocol: str
    batch: str
    spec_sha256: Sha256
    code_revision: str
    entries: tuple[ManifestEntry, ...] = ()

    def entry(self, task_id: str) -> ManifestEntry | None:
        """Return a task's row, or None when it has not run."""
        return next((row for row in self.entries if row.task_id == task_id), None)

    def by_status(self, status: EntryStatus) -> tuple[str, ...]:
        """Return the tasks in one status, in manifest order."""
        return tuple(row.task_id for row in self.entries if row.status == status)

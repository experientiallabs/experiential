"""The learned-rules stage: world-learning traces in, a saved rule book and its challenges out.

One run follows a fixed sequence over saved, replayable owners:

1. **Exposure.** Every trace must be reserved for world learning; any other trace is refused
   before anything is read, bought or replayed.
2. **Contracts.** Tool contracts are parsed from every trace, with no model call.
3. **Research.** Under the default ``agentic_research`` profile every trajectory is researched
   by bounded agentic draws, one per assigned prompt variant, concurrently
   (:mod:`~exp.optimize.evaluation.rules.research`), over one or more rounds (a later round reads
   the earlier rounds' notes). Each trace's research history is saved. Under the explicit
   ``single_draw`` profile the earlier path runs instead: one forced extraction draw per
   trajectory (:mod:`~exp.optimize.evaluation.rules.extraction`). Either way a refused or failed
   draw stays a failure record and is never redrawn.
4. **Merge.** The usable per-trajectory analyses merge into one rule book, excluding every trace
   the exposure plan does not reserve for world learning. Merging decides no support.
5. **Weighing.** Each merged rule's status is decided by an agentic evidence weighing
   (:mod:`~exp.optimize.evaluation.rules.weighing`); code checks only its structure.
6. **Witnesses.** Optional independent witnesses attach by rule identity.
7. **Rule book.** The rendered Markdown files and ``rules.json`` are saved as one
   ``rule-book`` artifact whose inputs are the responses or transcripts it was learned from and
   weighed by.
8. **Memory.** Under ``agentic_research`` an agentic condenser turns the rule book and every
   research history into the ``world-model-memory`` artifact (Markdown plus JSON) the
   environment simulator reads at runtime (:mod:`~exp.optimize.evaluation.rules.condense`),
   keeping every contradiction and unknown.
9. **Challenges.** Properties derived from the rules are checked on supplied simulated
   sessions and boundary workloads are generated from a supplied base state. Observed workload
   transitions, when supplied, are triaged. With an explicit ``challenge`` executor the workloads
   are instead executed in that bound world by
   :func:`~exp.optimize.evaluation.rules.challenges.run_challenges`, which saves every executed
   case, updates each challenged rule's status with its executed witnesses and replays without
   executing or calling anything.

Research and extraction identities depend only on what those draws read (the author settings,
the profile, the traces, their sources, the protocols and the code revision), and weighing
identities only on the merged book, so a later run that adds witnesses, sessions or observed
workload transitions replays every research and weighing draw. Witnesses change the saved book,
so the memory is condensed again. The report is
saved as a ``rules-stage-report`` artifact under an identity that also binds those additions,
and a repeated run returns it exactly.

A rule that a challenge contradicts and that no independent witness supports means the rule or
the simulator is wrong, and nothing here decides which. Such rules are listed in
``families_must_stay_diagnostic``: any environment family built on them stays diagnostic until a
native or specification witness exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.reconstruction import WorkflowEvidence
from exp.common.project import ArtifactAlreadyExistsError, ProjectStore
from exp.common.project.manifests import artifact_input
from exp.common.traces import Trace
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.agent_loop import AGENT_LOOP_PROTOCOL, AgentProfile
from exp.optimize.evaluation.rules.challenges import (
    ChallengeCase,
    ChallengeExecutor,
    ChallengeReport,
    run_challenges,
)
from exp.optimize.evaluation.rules.condense import condensation_protocol_sha256, condense_memory
from exp.optimize.evaluation.rules.extraction import (
    ExtractionDraw,
    extract_rules_in_parallel,
    extraction_protocol_sha256,
)
from exp.optimize.evaluation.rules.research import (
    RESEARCH_VARIANTS,
    ResearchDraw,
    ResearchMaterial,
    TraceResearch,
    research_in_parallel,
    research_protocol_sha256,
)
from exp.optimize.evaluation.rules.weighing import (
    WeighingDraw,
    weigh_rules_in_parallel,
    weighing_protocol_sha256,
)
from exp.optimize.evaluation.variance.evidence import TraceSearch
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.simulation.rules.contracts import parse_tool_contracts
from exp.simulation.rules.merge import (
    apply_weighings,
    attach_witnesses,
    diagnostic_rules,
    merge_rules,
)
from exp.simulation.rules.model import (
    RuleBook,
    RuleStatus,
    ToolContract,
    TrajectoryRules,
    Transition,
    Witness,
)
from exp.simulation.rules.properties import (
    DEFAULT_REFUSAL,
    PropertyReport,
    RefusalCriteria,
    check_transitions,
    properties_from,
    transitions_from_workflow,
)
from exp.simulation.rules.rulebook import RULES_JSON, render_rulebook, rulebook_json, write_rulebook
from exp.simulation.rules.workloads import (
    Triage,
    Workload,
    boundary_workloads,
    requires_witness,
    triage,
)

RULE_BOOK_TYPE = "rule-book"
"""Artifact type of a saved rule book: rendered Markdown files plus ``rules.json``."""

REPORT_TYPE = "rules-stage-report"
"""Artifact type of a saved stage report."""

SESSION_EVIDENCE = "evidence.json"
"""Data file holding the workflow evidence of a saved simulated session."""

_STATUSES: tuple[RuleStatus, ...] = ("supported", "contradicted", "unknown")

RulesProfile = Literal["agentic_research", "single_draw"]
"""How rules are learned: agentic per-trace research (default) or one forced draw per trace."""


class RulesStageSpec(ContractModel):
    """One registered learned-rules stage.

    Attributes:
        author: Frozen model and per-call allowance, shared by every research, extraction,
            weighing and condensation call.
        concurrency: Most research, extraction or weighing draws in flight at once. It does not
            affect what is bought, so it is left out of every identity.
        profile: ``agentic_research`` (default) or the explicit earlier ``single_draw`` path.
        research: Budgets of each research draw.
        variants: Research prompt variants assigned to traces.
        draws_per_trace: Research variants run per trace.
        rounds: Research rounds; a later round reads the earlier rounds' notes.
        weighing: Budgets of each rule's evidence weighing.
        condensation: Budgets of the memory condenser.
        sources: Saved artifacts the traces came from, pinned on every extraction draw.
        witnesses: Independent witnesses keyed by ``rule_id``; merged with any passed to
            :func:`run_rules_stage`.
        base_state: State that boundary workloads edit, or None to generate no workloads.
        examples: Example arguments per tool name, copied into each workload's action.
        observed_values: Extra observed values per read, keyed ``before:<pointer>`` or
            ``action:<pointer>``, used for ``in`` non-members.
        epsilon: Relative step for float thresholds.
        refusal: How a refused action is recognized in sessions and observed workloads.
    """

    author: AuthorSettings
    concurrency: int = Field(default=4, ge=1, le=64, strict=True)
    profile: RulesProfile = "agentic_research"
    research: AgentProfile = AgentProfile()
    variants: tuple[str, ...] = tuple(RESEARCH_VARIANTS)
    draws_per_trace: int = Field(default=1, ge=1, le=8, strict=True)
    rounds: int = Field(default=1, ge=1, le=4, strict=True)
    weighing: AgentProfile = AgentProfile()
    condensation: AgentProfile = AgentProfile()
    sources: tuple[ArtifactInput, ...] = ()
    witnesses: dict[str, tuple[Witness, ...]] = Field(default_factory=dict)
    base_state: JsonObject | None = None
    examples: dict[str, JsonObject] = Field(default_factory=dict)
    observed_values: dict[str, tuple[JsonValue, ...]] = Field(default_factory=dict)
    epsilon: float = Field(default=1e-6, gt=0.0, lt=1.0)
    refusal: RefusalCriteria = DEFAULT_REFUSAL

    def identity(self, code_revision: str) -> str:
        """Bind the registration, the extraction protocol and the producing code.

        Args:
            code_revision: Producing source identity.

        Returns:
            Stable identity of the registration.
        """
        return stable_id(
            "rules-stage-spec",
            {
                "spec": sha256_json(self.model_dump(mode="json", exclude={"concurrency"})),
                "extraction": extraction_protocol_sha256(),
                "research": research_protocol_sha256(),
                "weighing": weighing_protocol_sha256(),
                "condensation": condensation_protocol_sha256(),
                "loop": AGENT_LOOP_PROTOCOL,
                "code": code_revision,
            },
        )

    def extraction_identity(self, traces: Sequence[Trace], code_revision: str) -> str:
        """Bind exactly what research or extraction reads, so challenges never re-key a draw.

        Args:
            traces: The trajectories the draws read.
            code_revision: Producing source identity.

        Returns:
            Run identity from which each trajectory's draw identities are derived.
        """
        fields: JsonObject = {
            "author": sha256_json(self.author),
            "sources": [source.model_dump(mode="json") for source in self.sources],
            "traces": sorted(sha256_json(trace) for trace in traces),
            "extraction": extraction_protocol_sha256(),
            "code": code_revision,
        }
        if self.profile == "single_draw":
            return stable_id("rules-extraction", fields)
        return stable_id(
            "rules-research",
            {
                **fields,
                "research": research_protocol_sha256(),
                "loop": AGENT_LOOP_PROTOCOL,
                "profile": sha256_json(self.research),
                "variants": list(self.variants),
                "draws_per_trace": self.draws_per_trace,
                "rounds": self.rounds,
            },
        )


class RulesStageReport(ContractModel):
    """The complete record of one stage run, including failed draws.

    Attributes:
        spec: The registration.
        identity: Identity of this run, which names its saved report.
        extraction_identity: Identity the research or extraction draws derive from.
        contracts: Tool contracts parsed from every trace.
        draws: Every single-draw extraction draw, in trace order; empty under research.
        failures: The extraction draws that produced no usable rules; never redrawn.
        rulebook: The saved rule book artifact.
        rules: Number of merged rules.
        rule_counts: Merged rules per status.
        open_questions: Number of open questions in the book.
        diagnostic: Rule ids without a supporting witness, in book order.
        requires_witness: Rule ids whose boundaries are challengeable but that have no supporting
            witness, in book order.
        sessions: Saved simulated sessions read, by reference.
        session_ids: Workflow ids of every simulated session checked, in input order.
        properties: Property results over the sessions, or None when none were supplied.
        workloads: Boundary workloads, empty when no base state was registered.
        triage: Classification of every observed workload, in workload order.
        families_must_stay_diagnostic: Rule ids without a supporting witness that a challenge
            contradicted (a property violation on a session or a contradicting workload), in
            book order. Any environment family built on them stays diagnostic.
        challenges: The executed challenges, when a challenge executor was supplied.
        research: Every trace's research history, in trace order, under ``agentic_research``.
        research_failures: Research draws that produced no usable answer; never redrawn.
        weighings: Every rule's evidence weighing, in merged-book order.
        memory: The saved ``world-model-memory``, under ``agentic_research``.
    """

    spec: RulesStageSpec
    identity: str
    extraction_identity: str
    contracts: dict[str, ToolContract]
    draws: tuple[ExtractionDraw, ...]
    failures: tuple[ExtractionDraw, ...]
    rulebook: ArtifactInput
    rules: int = Field(ge=0)
    rule_counts: dict[RuleStatus, int]
    open_questions: int = Field(ge=0)
    diagnostic: tuple[str, ...]
    requires_witness: tuple[str, ...]
    sessions: tuple[ArtifactInput, ...]
    session_ids: tuple[str, ...]
    properties: PropertyReport | None
    workloads: tuple[Workload, ...]
    triage: tuple[Triage, ...]
    families_must_stay_diagnostic: tuple[str, ...]
    challenges: ChallengeReport | None = Field(default=None, exclude_if=lambda value: value is None)
    research: tuple[TraceResearch, ...] = Field(default=(), exclude_if=lambda value: not value)
    research_failures: tuple[ResearchDraw, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    weighings: tuple[WeighingDraw, ...] = Field(default=(), exclude_if=lambda value: not value)
    memory: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)


def run_rules_stage(
    project: ProjectStore,
    spec: RulesStageSpec,
    *,
    exposure: ExposurePlan,
    traces: Sequence[Trace],
    code_revision: str,
    witnesses: Mapping[str, Sequence[Witness]] | None = None,
    sessions: Sequence[WorkflowEvidence | ArtifactInput] = (),
    observed: Mapping[str, Transition] | None = None,
    challenge: ChallengeExecutor | None = None,
    search: TraceSearch | None = None,
) -> RulesStageReport:
    """Run, or replay, the learned-rules stage over world-learning traces.

    Args:
        project: Project holding every saved step.
        spec: The registration.
        exposure: The frozen data roles; every trace must be reserved for world learning.
        traces: The trajectories to learn from, with distinct trace identities.
        code_revision: Producing source identity.
        witnesses: Extra independent witnesses keyed by ``rule_id``, merged with the spec's.
        sessions: Simulated sessions to check properties on, as workflow evidence or as saved
            artifacts holding ``evidence.json``.
        observed: Observed transition per executed workload id, for triage. Workload ids are
            ``<rule_id>/<n>`` as listed in an earlier report's ``workloads``.
        challenge: Explicit bound world to execute every boundary workload in, instead of
            supplying observed transitions; its identity joins the run identity.
        search: Registered world-learning retrieval research draws may query; its index joins
            the run identity.

    Returns:
        The report, also saved as a ``rules-stage-report`` artifact.

    Raises:
        ValueError: No trace is supplied, a trace is not reserved for world learning, trace
            identities repeat, a witness names an unknown rule, a session reference is not the
            saved artifact it pins, observed transitions or a challenge executor are supplied
            without a registered base state, both are supplied, or observed transitions name
            unknown workloads.
        FileLockTimeout: Another process is running the same stage or extraction.
    """
    if not traces:
        raise ValueError("the rules stage needs at least one world-learning trace")
    exposure.require((trace.trace_id for trace in traces), "world_learning")
    ids = [trace.trace_id for trace in traces]
    if len(set(ids)) != len(ids):
        raise ValueError("the rules stage needs distinct traces; a trace ID repeats")
    spec = RulesStageSpec.model_validate_json(spec.model_dump_json())
    if (observed or challenge is not None) and spec.base_state is None:
        raise ValueError(
            "observed or executed workload transitions need a registered base_state; register "
            "the base state the workloads are generated from"
        )
    if observed and challenge is not None:
        raise ValueError(
            "supply either observed workload transitions or a challenge executor, not both"
        )
    store = project.artifacts
    combined = _combine_witnesses(spec.witnesses, witnesses or {})
    loaded = tuple(_load_session(project, session) for session in sessions)
    extraction_identity = spec.extraction_identity(traces, code_revision)
    fields: JsonObject = {} if challenge is None else {"challenge": challenge.identity}
    if search is not None:
        fields["search"] = search.inventory()
    identity = stable_id(
        "rules-stage",
        {
            **fields,
            "spec": spec.identity(code_revision),
            "extraction": extraction_identity,
            "witnesses": {
                rule_id: [witness.model_dump(mode="json") for witness in found]
                for rule_id, found in sorted(combined.items())
            },
            "sessions": [
                reference.model_dump(mode="json")
                if reference is not None
                else sha256_json(evidence)
                for reference, evidence in loaded
            ],
            "observed": {
                workload: transition.model_dump(mode="json")
                for workload, transition in sorted((observed or {}).items())
            },
        },
    )
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return RulesStageReport.model_validate_json(store.read_bytes(result_id, "report.json"))
        contracts = parse_tool_contracts(traces)
        excluded = frozenset(
            trace for trace in exposure.traces if exposure.role_of(trace) != "world_learning"
        )
        draws: tuple[ExtractionDraw, ...] = ()
        research: tuple[TraceResearch, ...] = ()
        if spec.profile == "single_draw":
            draws = extract_rules_in_parallel(
                project,
                spec.author,
                traces,
                contracts,
                identity=extraction_identity,
                code_revision=code_revision,
                concurrency=spec.concurrency,
                sources=spec.sources,
            )
            analyses: list[TrajectoryRules] = [d.rules for d in draws if d.rules is not None]
            learned = tuple(d.response for d in draws if d.rules is not None and d.response)
        else:
            research = _research(
                project, spec, traces, contracts, extraction_identity, code_revision, search
            )
            analyses = [analysis for item in research for analysis in item.analyses()]
            learned = tuple(
                draw.transcript
                for item in research
                for draw in item.draws
                if draw.answer is not None and draw.transcript is not None
            )
        merged = merge_rules(contracts, analyses, excluded=excluded)
        weighings = weigh_rules_in_parallel(
            project,
            spec.author,
            spec.weighing,
            merged,
            ResearchMaterial(traces, contracts, maximum_bytes=spec.weighing.maximum_result_bytes),
            {item.trace_id: item for item in research},
            identity=stable_id(
                "rules-weighing",
                {
                    "extraction": extraction_identity,
                    "book": sha256_json(merged),
                    "author": sha256_json(spec.author),
                    "profile": sha256_json(spec.weighing),
                    "weighing": weighing_protocol_sha256(),
                    "loop": AGENT_LOOP_PROTOCOL,
                    "excluded": sorted(excluded),
                },
            ),
            excluded=excluded,
            code_revision=code_revision,
            concurrency=spec.concurrency,
            sources=spec.sources,
        )
        book = apply_weighings(
            merged, [w.weighing for w in weighings if w.weighing is not None], excluded=excluded
        )
        if combined:
            book = attach_witnesses(book, combined)
        responses = learned + tuple(w.transcript for w in weighings if w.transcript is not None)
        rulebook = _save_rulebook(project, book, responses, extraction_identity, code_revision)
        memory: ArtifactInput | None = None
        if spec.profile == "agentic_research":
            memory, _ = condense_memory(
                project,
                spec.author,
                spec.condensation,
                book,
                rulebook,
                research,
                identity=stable_id(
                    "rules-condensation",
                    {
                        "rulebook": rulebook.model_dump(mode="json"),
                        "research": [
                            item.history.model_dump(mode="json")
                            for item in research
                            if item.history is not None
                        ],
                        "author": sha256_json(spec.author),
                        "profile": sha256_json(spec.condensation),
                        "condensation": condensation_protocol_sha256(),
                        "loop": AGENT_LOOP_PROTOCOL,
                    },
                ),
                code_revision=code_revision,
                sources=spec.sources,
            )
        workloads: tuple[Workload, ...] = ()
        if spec.base_state is not None:
            workloads = boundary_workloads(
                book,
                spec.base_state,
                examples=spec.examples,
                observed_values=spec.observed_values,
                epsilon=spec.epsilon,
            )
        executed = None
        if challenge is not None:
            executed = run_challenges(
                project,
                book=book,
                rulebook=rulebook,
                cases=tuple(ChallengeCase(workload=workload) for workload in workloads),
                executor=challenge,
                code_revision=code_revision,
                refusal=spec.refusal,
            )
            observed = executed.observed()
        report = _report(
            spec,
            identity,
            extraction_identity,
            contracts,
            draws,
            book,
            rulebook,
            loaded,
            workloads,
            observed or {},
        ).model_copy(
            update={
                "challenges": executed,
                "research": research,
                "research_failures": tuple(
                    draw for item in research for draw in item.draws if draw.answer is None
                ),
                "weighings": weighings,
                "memory": memory,
            }
        )
        dependencies = {rulebook, *responses, *spec.sources}
        dependencies |= {item.history for item in research if item.history is not None}
        if memory is not None:
            dependencies.add(memory)
        if executed is not None:
            dependencies.add(artifact_input(store.read(executed.identity + "-result").manifest))
        dependencies |= {reference for reference, _ in loaded if reference is not None}
        store.write_json(
            artifact_id=result_id,
            artifact_type=REPORT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(dependencies, key=lambda ref: ref.artifact_id)),
            ),
            files={"report.json": report},
        )
        return report


def _research(
    project: ProjectStore,
    spec: RulesStageSpec,
    traces: Sequence[Trace],
    contracts: Mapping[str, ToolContract],
    identity: str,
    code_revision: str,
    search: TraceSearch | None,
) -> tuple[TraceResearch, ...]:
    """Run, or replay, every research round; each round reads the previous rounds' notes.

    Args:
        project: Project holding every draw.
        spec: The registration.
        traces: World-learning trajectories.
        contracts: Parsed tool contracts.
        identity: Research run identity.
        code_revision: Producing source identity.
        search: Registered world-learning retrieval, when any.

    Returns:
        Each trace's research history across every round, in trace order.
    """
    prior: dict[str, TraceResearch] = {}
    for round_ in range(spec.rounds):
        found = research_in_parallel(
            project,
            spec.author,
            spec.research,
            traces,
            contracts,
            identity=identity,
            code_revision=code_revision,
            concurrency=spec.concurrency,
            variants=spec.variants,
            draws_per_trace=spec.draws_per_trace,
            round_=round_,
            prior=prior or None,
            search=search,
            sources=spec.sources,
        )
        prior = {item.trace_id: item for item in found}
    return tuple(prior[trace.trace_id] for trace in traces)


def load_rulebook(project: ProjectStore, reference: ArtifactInput) -> RuleBook:
    """Read a saved rule book back from its artifact.

    Args:
        project: Project holding the rule book.
        reference: The rule book artifact, as a stage report names it.

    Returns:
        The rule book its ``rules.json`` holds.

    Raises:
        ValueError: The reference is not the saved rule book artifact it pins.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        RULE_BOOK_TYPE
    ):
        raise ValueError(
            f"artifact {reference.artifact_id} is not the pinned rule book; pass the rulebook "
            "reference from a rules stage report"
        )
    return RuleBook.model_validate_json(stored.payloads[RULES_JSON])


def export_rulebook(
    project: ProjectStore, reference: ArtifactInput, directory: Path
) -> tuple[Path, ...]:
    """Write a saved rule book into a new directory for reading and diffing.

    The directory receives exactly the files the artifact holds.

    Args:
        project: Project holding the rule book.
        reference: The rule book artifact.
        directory: Destination directory; it must not exist yet.

    Returns:
        Paths of every written file, sorted.

    Raises:
        ValueError: The reference is not the saved rule book artifact it pins.
        FileExistsError: ``directory`` already exists.
    """
    return write_rulebook(load_rulebook(project, reference), directory)


def rulebook_files(book: RuleBook) -> dict[str, bytes]:
    """Return the exact files a rule book artifact holds.

    These are the files :func:`~exp.simulation.rules.rulebook.write_rulebook` writes: the
    rendered Markdown store plus ``rules.json``.

    Args:
        book: The rule book.

    Returns:
        UTF-8 file payloads keyed by POSIX path, in path order.
    """
    files = render_rulebook(book)
    files[RULES_JSON] = rulebook_json(book)
    return {path: text.encode("utf-8") for path, text in sorted(files.items())}


def _combine_witnesses(
    registered: Mapping[str, Sequence[Witness]], supplied: Mapping[str, Sequence[Witness]]
) -> dict[str, tuple[Witness, ...]]:
    """Merge registered and supplied witnesses per rule, deduplicated in a stable order.

    Args:
        registered: Witnesses from the spec.
        supplied: Witnesses passed to the run.

    Returns:
        Witnesses keyed by rule id, omitting rules with none.
    """
    merged: dict[str, tuple[Witness, ...]] = {}
    for rule_id in sorted({*registered, *supplied}):
        found = {*registered.get(rule_id, ()), *supplied.get(rule_id, ())}
        if found:
            merged[rule_id] = tuple(
                sorted(
                    found, key=lambda witness: (witness.kind, witness.reference, witness.supports)
                )
            )
    return merged


def _load_session(
    project: ProjectStore, session: WorkflowEvidence | ArtifactInput
) -> tuple[ArtifactInput | None, WorkflowEvidence]:
    """Resolve one simulated session to its evidence, verifying a saved reference.

    Args:
        project: Project holding saved sessions.
        session: Workflow evidence, or a saved artifact holding ``evidence.json``.

    Returns:
        The reference, or None for inline evidence, and the evidence.

    Raises:
        ValueError: The reference is not the saved artifact it pins, or holds no evidence.
    """
    if isinstance(session, WorkflowEvidence):
        return None, session
    stored = project.artifacts.read(session.artifact_id)
    if artifact_input(stored.manifest) != session:
        raise ValueError(
            f"session {session.artifact_id} changed since it was pinned; pin the saved reference"
        )
    payload = stored.payloads.get(SESSION_EVIDENCE)
    if payload is None:
        raise ValueError(
            f"session {session.artifact_id} holds no {SESSION_EVIDENCE}; pass a saved workflow "
            "evaluation intent or the evidence itself"
        )
    return session, WorkflowEvidence.model_validate_json(payload)


def _save_rulebook(
    project: ProjectStore,
    book: RuleBook,
    responses: tuple[ArtifactInput, ...],
    extraction_identity: str,
    code_revision: str,
) -> ArtifactInput:
    """Save a rule book as one artifact, or adopt the identical one already saved.

    Args:
        project: Project receiving the artifact.
        book: The merged rule book, with witnesses attached.
        responses: Responses or transcripts the book was learned from and weighed by.
        extraction_identity: Identity the draws derive from.
        code_revision: Producing source identity.

    Returns:
        The rule book artifact reference.

    Raises:
        ValueError: An artifact with the same identity holds a different book.
    """
    store = project.artifacts
    files = rulebook_files(book)
    artifact_id = stable_id(
        "rule-book", {"extraction": extraction_identity, "book": sha256_json(book)}
    )
    if not store.exists(artifact_id):
        try:
            manifest = store.write(
                artifact_id=artifact_id,
                artifact_type=RULE_BOOK_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=tuple(sorted(set(responses), key=lambda ref: ref.artifact_id)),
                ),
                files=files,
            )
        except ArtifactAlreadyExistsError:
            pass
        else:
            return artifact_input(manifest)
    stored = store.read(artifact_id)
    if stored.manifest.artifact_type != RULE_BOOK_TYPE or dict(stored.payloads) != files:
        raise ValueError(f"saved rule book {artifact_id} differs from the book this run merged")
    return artifact_input(stored.manifest)


def _report(
    spec: RulesStageSpec,
    identity: str,
    extraction_identity: str,
    contracts: dict[str, ToolContract],
    draws: tuple[ExtractionDraw, ...],
    book: RuleBook,
    rulebook: ArtifactInput,
    loaded: tuple[tuple[ArtifactInput | None, WorkflowEvidence], ...],
    workloads: tuple[Workload, ...],
    observed: Mapping[str, Transition],
) -> RulesStageReport:
    """Challenge the rule book and assemble the report.

    Args:
        spec: The registration.
        identity: Run identity.
        extraction_identity: Identity the draws derive from.
        contracts: Parsed tool contracts.
        draws: Every extraction draw.
        book: The merged rule book.
        rulebook: Its saved artifact.
        loaded: Resolved simulated sessions.
        workloads: Boundary workloads generated from the registered base state.
        observed: Observed or executed transitions per workload id.

    Returns:
        The report.

    Raises:
        ValueError: ``observed`` names a workload that was not generated.
    """
    properties: PropertyReport | None = None
    if loaded:
        transitions = tuple(
            located for _, evidence in loaded for located in transitions_from_workflow(evidence)
        )
        properties = check_transitions(properties_from(book), transitions, spec.refusal)
    triaged = triage(workloads, observed, spec.refusal) if observed else ()
    witnessed = {
        rule.rule_id for rule in book.rules if any(witness.supports for witness in rule.witnesses)
    }
    contradicted = {item.rule_id for item in triaged if item.needs_witness}
    if properties is not None:
        contradicted |= {result.rule_id for result in properties.results if result.violated}
    return RulesStageReport(
        spec=spec,
        identity=identity,
        extraction_identity=extraction_identity,
        contracts=contracts,
        draws=draws,
        failures=tuple(draw for draw in draws if draw.failure is not None),
        rulebook=rulebook,
        rules=len(book.rules),
        rule_counts={
            status: sum(1 for rule in book.rules if rule.status == status) for status in _STATUSES
        },
        open_questions=len(book.questions),
        diagnostic=tuple(rule.rule_id for rule in diagnostic_rules(book)),
        requires_witness=requires_witness(book),
        sessions=tuple(reference for reference, _ in loaded if reference is not None),
        session_ids=tuple(evidence.workflow_id for _, evidence in loaded),
        properties=properties,
        workloads=workloads,
        triage=triaged,
        families_must_stay_diagnostic=tuple(
            rule.rule_id
            for rule in book.rules
            if rule.rule_id in contradicted and rule.rule_id not in witnessed
        ),
    )

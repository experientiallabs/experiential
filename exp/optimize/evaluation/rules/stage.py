"""The learned-rules stage: world-learning traces in, a saved rule book and its challenges out.

One run follows a fixed sequence over saved, replayable owners:

1. **Exposure.** Every trace must be reserved for world learning; any other trace is refused
   before anything is read, bought or replayed.
2. **Contracts.** Tool contracts are parsed from every trace, with no model call.
3. **Extraction.** One rule-extraction draw per trajectory runs concurrently. Draws are bought
   once and replayed exactly; a refused or failed draw stays a failure record and is never
   redrawn.
4. **Merge.** The successful per-trajectory analyses merge into one rule book.
5. **Witnesses.** Optional independent witnesses attach by rule identity.
6. **Rule book.** The rendered Markdown files and ``rules.json`` are saved as one
   ``rule-book`` artifact whose inputs are the extraction responses it was learned from.
7. **Challenges.** Properties derived from the rules are checked on supplied simulated
   sessions and boundary workloads are generated from a supplied base state. Observed workload
   transitions, when supplied, are triaged. With an explicit ``challenge`` executor the workloads
   are instead executed in that bound world by
   :func:`~exp.optimize.evaluation.rules.challenges.run_challenges`, which saves every executed
   case, updates each challenged rule's status with its executed witnesses and replays without
   executing or calling anything.

Draw identities depend only on what extraction reads (the author settings, the traces, their
sources, the extraction protocol and the code revision), so a later run that adds witnesses,
sessions or observed workload transitions replays every draw and buys nothing. The report is
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
from exp.optimize.evaluation.rules.challenges import (
    ChallengeCase,
    ChallengeExecutor,
    ChallengeReport,
    run_challenges,
)
from exp.optimize.evaluation.rules.extraction import (
    ExtractionDraw,
    extract_rules_in_parallel,
    extraction_protocol_sha256,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.simulation.rules.contracts import parse_tool_contracts
from exp.simulation.rules.merge import attach_witnesses, diagnostic_rules, merge_rules
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


class RulesStageSpec(ContractModel):
    """One registered learned-rules stage.

    Attributes:
        author: Frozen extraction model and allowance, shared by every draw.
        concurrency: Most extraction draws in flight at once. It does not affect what is
            bought, so it is left out of every identity.
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
                "code": code_revision,
            },
        )

    def extraction_identity(self, traces: Sequence[Trace], code_revision: str) -> str:
        """Bind exactly what extraction reads, so later challenges never re-key a draw.

        Args:
            traces: The trajectories the draws read.
            code_revision: Producing source identity.

        Returns:
            Run identity from which each trajectory's draw identity is derived.
        """
        return stable_id(
            "rules-extraction",
            {
                "author": sha256_json(self.author),
                "sources": [source.model_dump(mode="json") for source in self.sources],
                "traces": sorted(sha256_json(trace) for trace in traces),
                "extraction": extraction_protocol_sha256(),
                "code": code_revision,
            },
        )


class RulesStageReport(ContractModel):
    """The complete record of one stage run, including failed draws.

    Attributes:
        spec: The registration.
        identity: Identity of this run, which names its saved report.
        extraction_identity: Identity the extraction draws derive from.
        contracts: Tool contracts parsed from every trace.
        draws: Every extraction draw, in trace order.
        failures: The draws that produced no usable rules, in trace order; never redrawn.
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
        analyses: list[TrajectoryRules] = [draw.rules for draw in draws if draw.rules is not None]
        book = merge_rules(contracts, analyses)
        if combined:
            book = attach_witnesses(book, combined)
        responses = tuple(
            draw.response for draw in draws if draw.rules is not None and draw.response is not None
        )
        rulebook = _save_rulebook(project, book, responses, extraction_identity, code_revision)
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
        ).model_copy(update={"challenges": executed})
        dependencies = {rulebook, *responses, *spec.sources}
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
        responses: Extraction responses the book was learned from.
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

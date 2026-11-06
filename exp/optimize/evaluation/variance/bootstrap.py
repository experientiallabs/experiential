"""Automatic startup from traces: environment families in, registered mining jobs out.

:func:`run_bootstrap` runs one :class:`~exp.optimize.evaluation.variance.bootstrap_contracts.
BootstrapSpec` over selected environment families:

1. **Family knowledge.** A family's frozen ``rules-stage-report`` is read as it is; a family
   without one learns its rules with
   :func:`~exp.optimize.evaluation.rules.stage.run_rules_stage` (and, under ``jev_memory``, its
   governed rule memory set) over its member traces. The family's reconciled tool contracts take
   precedence over those the rules stage parses; its shared state domains bind every start.
2. **Author.** For each selected task an agentic author drafts the public instruction, tool
   schemas, world manifest, initial state, evaluator projection, verifier and a fact ledger that
   separates observed, inferred and unknown facts, each with its trace references.
3. **Review.** Every reviewer stance checks whether the start is coherent, sufficiently specified
   and supports the task. Missing deciding facts (named by a reviewer, or deciding facts the
   draft leaves unknown) become research work items answered by the family's research step and
   fed back to the author, for at most ``research_rounds`` rounds; then a diagnostic refusal.
4. **Capture and validate.** An accepted start is captured through the existing capture owner
   and validated by its provenance (exact restore, prefix replay, or coherence plus first
   reveal). A failed validation is a diagnostic refusal. Only an observed start claims to be the
   original environment.
5. **Register.** The mining job is frozen and preflighted, and the registration saved.

**Batch manifest.** Each task's outcome (registration or refusal) is one immutable artifact,
keyed by the task's own identity, so it replays with no call. The batch manifest
(``<project>/start-bootstrap/<batch>.json``, rewritten atomically after every task) lists every
task that ran with its status. An exception (provider outage, interrupted draw, lock held) marks
only that task ``failed``; the others proceed, and a rerun retries it while replaying the rest.
:func:`freeze_batch` saves the manifest as an artifact a pipeline registration can pin, and
:func:`registered_job` returns one task's job from it.

Replay rechecks current source presence and exposure. An immutable per-task evidence binding
pins every family and task source trace's content before learning or authoring begins; changed
evidence fails that task instead of returning a validation of different traces.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore, artifact_input
from exp.common.state_machine import TransitionJournal
from exp.common.traces import Trace
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.executable.grounded import RuleMemoryBinding
from exp.optimize.evaluation.rules.agent_loop import run_bounded
from exp.optimize.evaluation.rules.stage import (
    REPORT_TYPE,
    RulesStageReport,
    load_rulebook,
    run_rules_stage,
)
from exp.optimize.evaluation.rules.system_one import DecisionBoundary
from exp.optimize.evaluation.variance.bootstrap_agents import (
    FamilyKnowledge,
    StartMaterial,
    author_start,
    reconcile_reviews,
    research,
    review_start,
)
from exp.optimize.evaluation.variance.bootstrap_capture import (
    capture_start,
    mining_job,
    start_definition,
    validate_start,
)
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    BATCH_TYPE,
    PROVENANCE_STATEMENTS,
    RECORDED_STATE,
    REFUSAL_TYPE,
    REGISTRATION_TYPE,
    BatchManifest,
    BootstrapBinding,
    BootstrapFamily,
    BootstrapRefusal,
    BootstrapRegistration,
    BootstrapSpec,
    BootstrapTask,
    Finding,
    ManifestEntry,
    RefusalKind,
    ResearchItem,
    ReviewRound,
    StartValidation,
)
from exp.optimize.evaluation.variance.bootstrap_prompts import (
    BOOTSTRAP_PROTOCOL,
    bootstrap_prompts_sha256,
)
from exp.optimize.evaluation.variance.bootstrap_states import TASK_MACHINE, TaskState
from exp.optimize.evaluation.variance.evidence import TraceSearch
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.rules.recall import RecallPolicy

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.pipeline_spec import DecisionVarianceSpec

Outcome = BootstrapRegistration | BootstrapRefusal
"""A task's saved outcome."""

MANIFEST_DIRECTORY = "start-bootstrap"
"""Directory, under the project's artifact directory, holding batch manifests."""


def batch_identity(spec: BootstrapSpec, code_revision: str) -> str:
    """Return the identity of a batch: the registration without its concurrency, and the code."""
    return stable_id(
        "start-bootstrap-batch",
        {
            "protocol": BOOTSTRAP_PROTOCOL,
            "spec": sha256_json(spec.model_dump(mode="json", exclude={"concurrency"})),
            "code": code_revision,
        },
    )


def task_identity(
    spec: BootstrapSpec, family: BootstrapFamily, task: BootstrapTask, code_revision: str
) -> str:
    """Return a task's identity: only what its outcome depends on.

    Adding a family or a task, or changing another family, never re-keys this task.
    """
    roles = spec.model_dump(
        mode="json",
        include={
            "author",
            "reviewer",
            "researcher",
            "author_profile",
            "review_profile",
            "research_profile",
            "author_variants",
            "review_variants",
            "research_variants",
            "reveal_variants",
            "research_rounds",
            "research_items",
            "job",
        },
    )
    rules = None if family.family.frozen or spec.rules is None else spec.rules
    return stable_id(
        "start-bootstrap-task",
        {
            "protocol": BOOTSTRAP_PROTOCOL,
            "prompts": bootstrap_prompts_sha256(),
            "family": sha256_json(family.family),
            "runtime": sha256_json(spec.runtime),
            "rules": None if rules is None else sha256_json(rules.model_dump(mode="json")),
            "task": sha256_json(task),
            "roles": sha256_json(roles),
            "code": code_revision,
        },
    )


def family_knowledge(
    project: ProjectStore,
    spec: BootstrapSpec,
    family: BootstrapFamily,
    *,
    exposure: ExposurePlan,
    traces: Mapping[str, Trace],
    code_revision: str,
    search: TraceSearch | None = None,
    decisions: DecisionsTraceTagger | DecisionBoundary | None = None,
) -> FamilyKnowledge:
    """Read a family's frozen rules, or learn them over its member traces, and reconcile.

    Args:
        project: Project holding the rules.
        spec: The registration.
        family: The family.
        exposure: Frozen data roles; member traces must be world-learning traces.
        traces: Every trace the batch reads, by identity.
        code_revision: Producing source identity.
        search: World-learning retrieval research draws may query.
        decisions: System One decisions client under ``jev_memory``.

    Returns:
        The family's knowledge.

    Raises:
        ValueError: A member trace is missing or not reserved for world learning, or the frozen
            rules are not a saved rules-stage report.
    """
    adapter = family.family
    needed = {*adapter.member_traces, *(() if adapter.frozen else adapter.learning_traces)}
    missing = sorted(needed - set(traces))
    if missing:
        raise ValueError(f"family {adapter.family_id} traces were not supplied: {missing}")
    exposure.require(sorted(needed), "world_learning")
    members = {trace_id: traces[trace_id] for trace_id in adapter.member_traces}
    if adapter.rule_book is not None:
        report = adapter.rules_report
        if report is not None and frozen_report(project, report).rulebook != adapter.rule_book:
            raise ValueError(f"family {adapter.family_id} rules report names another rule book")
        rulebook, memory = adapter.rule_book, adapter.rule_memory
    else:
        assert spec.rules is not None
        learned = run_rules_stage(
            project,
            spec.rules,
            exposure=exposure,
            traces=tuple(traces[trace_id] for trace_id in adapter.learning_traces),
            code_revision=code_revision,
            search=search,
            decisions=decisions,
            family=adapter.family_id,
        )
        report = artifact_input(project.artifacts.read(learned.identity + "-result").manifest)
        rulebook, memory = learned.rulebook, None
        if learned.memory_set is not None:
            recall = RecallPolicy(jev_alias=spec.runtime.jev_alias)
            staged = learned.memory_stage
            memory = RuleMemoryBinding(
                memory_set=learned.memory_set,
                recall=recall,
                guards=None if staged is None else staged.guards_artifact,
            )
    book = load_rulebook(project, rulebook)
    return FamilyKnowledge(
        family_id=adapter.family_id,
        report=report,
        rulebook=rulebook,
        book=book,
        rule_memory=memory,
        traces=members,
        contracts={**book.contracts, **adapter.tool_contracts},
        state_domains=tuple(sorted(adapter.state_domains)),
    )


def frozen_report(project: ProjectStore, reference: ArtifactInput) -> RulesStageReport:
    """Read a pinned rules-stage report.

    Raises:
        ValueError: The reference is not a saved rules-stage report.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        REPORT_TYPE
    ):
        raise ValueError(f"{reference.artifact_id} is not a pinned rules-stage report")
    return RulesStageReport.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "report.json")
    )


def run_bootstrap(
    project: ProjectStore,
    spec: BootstrapSpec,
    *,
    exposure: ExposurePlan,
    traces: Mapping[str, Trace],
    code_revision: str,
    tasks: Collection[str] | None = None,
    search: TraceSearch | None = None,
    decisions: DecisionsTraceTagger | DecisionBoundary | None = None,
) -> BatchManifest:
    """Bootstrap, resume or replay every selected task and keep the batch manifest current.

    Args:
        project: Project holding every saved step.
        spec: The registration.
        exposure: Frozen data roles.
        traces: Every member and source trace, by identity.
        code_revision: Producing source identity.
        tasks: Process only these task identities; None processes every task.
        search: World-learning retrieval for families learning their rules here.
        decisions: System One decisions client for families learning ``jev_memory`` here.

    Returns:
        The batch manifest after this run; a failed task never stops the others.
    """
    spec = BootstrapSpec.model_validate_json(spec.model_dump_json())
    batch = batch_identity(spec, code_revision)
    manifest = load_batch_manifest(project, batch) or BatchManifest(
        protocol=BOOTSTRAP_PROTOCOL,
        batch=batch,
        spec_sha256=sha256_json(spec.model_dump(mode="json", exclude={"concurrency"})),
        code_revision=code_revision,
    )
    guard = threading.Lock()
    journal = TransitionJournal(_manifest_path(project, batch).with_suffix(".transitions.jsonl"))
    order = [row.task.task_id for family in spec.families for row in family.tasks]

    def record(entry: ManifestEntry) -> None:
        nonlocal manifest
        with guard:
            previous = manifest.entry(entry.task_id)
            if previous is None or previous.status != entry.status or entry.status == "failed":
                TASK_MACHINE.move(
                    entry.task_id,
                    None if previous is None else TaskState(previous.status),
                    entry.status,
                    journal,
                )
            rows = {row.task_id: row for row in manifest.entries} | {entry.task_id: entry}
            manifest = manifest.model_copy(
                update={"entries": tuple(rows[task] for task in order if task in rows)}
            )
            _write_manifest(project, manifest)

    for family in spec.families:
        selected = [row for row in family.tasks if tasks is None or row.task.task_id in tasks]
        ids = {
            row.task.task_id: task_identity(spec, family, row, code_revision) for row in selected
        }
        admitted: list[BootstrapTask] = []
        for row in selected:
            try:
                _bind_evidence(
                    project, family, row, ids[row.task.task_id], exposure, traces, code_revision
                )
            except Exception as error:  # noqa: BLE001 - invalid evidence fails only its task
                previous = manifest.entry(row.task.task_id)
                record(_failed(family, ids[row.task.task_id], row, previous, error, "evidence"))
            else:
                admitted.append(row)
        selected = admitted
        pending = [row for row in selected if load_outcome(project, ids[row.task.task_id]) is None]
        knowledge: FamilyKnowledge | None = None
        if pending:
            try:
                knowledge = family_knowledge(
                    project,
                    spec,
                    family,
                    exposure=exposure,
                    traces=traces,
                    code_revision=code_revision,
                    search=search,
                    decisions=decisions,
                )
            except Exception as error:  # noqa: BLE001 - every family failure is reported per task
                for row in pending:
                    previous = manifest.entry(row.task.task_id)
                    record(_failed(family, ids[row.task.task_id], row, previous, error, "family"))
                selected = [row for row in selected if row not in pending]
        calls = [
            _task_call(
                project,
                spec,
                family,
                row,
                knowledge,
                ids[row.task.task_id],
                exposure,
                traces,
                code_revision,
                manifest.entry(row.task.task_id),
                record,
            )  # fmt: skip
            for row in selected
        ]
        run_bounded(calls, spec.concurrency)
    return manifest


def _bind_evidence(
    project: ProjectStore,
    family: BootstrapFamily,
    task: BootstrapTask,
    identity: str,
    exposure: ExposurePlan,
    traces: Mapping[str, Trace],
    code_revision: str,
) -> None:
    """Validate current access and freeze exact trace content before replay or paid work.

    Raises:
        ValueError: Required evidence is absent, excluded, changed or lacks a saved binding.
    """
    members = family.family.member_traces
    needed = {*members, *task.source_traces}
    missing = sorted(needed - set(traces))
    if missing:
        raise ValueError(f"traces of {task.task.task_id} were not supplied: {missing}; supply them")
    exposure.require(members, "world_learning")
    exposure.require_lineage(task.task.lineage_group_id, "mining_probes")
    exposure.require(task.source_traces, "mining_probes")
    evidence: JsonObject = {trace_id: sha256_json(traces[trace_id]) for trace_id in sorted(needed)}
    store = project.artifacts
    artifact_id = identity + "-evidence"
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        if store.exists(artifact_id):
            saved = json.loads(store.read_bytes(artifact_id, "evidence.json"))
            if saved != evidence:
                raise ValueError(
                    f"trace evidence of {task.task.task_id} changed; restore the exact sources "
                    "or prepare a fresh bootstrap"
                )
            return
        if load_outcome(project, identity) is not None:
            raise ValueError(
                f"saved bootstrap of {task.task.task_id} has no trace evidence binding; "
                "prepare a fresh bootstrap"
            )
        store.write_json(
            artifact_id=artifact_id,
            artifact_type="start-bootstrap-evidence",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
            ),
            files={"evidence.json": evidence},
        )


def load_outcome(project: ProjectStore, identity: str) -> Outcome | None:
    """Read a task's saved registration or refusal, or None when it has none.

    Args:
        project: Project holding the outcome.
        identity: Task bootstrap identity.

    Returns:
        The outcome, or None.
    """
    store = project.artifacts
    if store.exists(identity + "-registration"):
        raw = store.read_bytes(identity + "-registration", "registration.json")
        return BootstrapRegistration.model_validate_json(raw)
    if store.exists(identity + "-refusal"):
        return BootstrapRefusal.model_validate_json(
            store.read_bytes(identity + "-refusal", "refusal.json")
        )
    return None


def load_batch_manifest(project: ProjectStore, batch: str) -> BatchManifest | None:
    """Read a batch's current manifest, or None when no task has run.

    Args:
        project: Project holding the manifest.
        batch: Batch identity.

    Returns:
        The manifest, or None.
    """
    path = _manifest_path(project, batch)
    if not path.exists():
        return None
    return BatchManifest.model_validate_json(path.read_bytes())


def freeze_batch(project: ProjectStore, manifest: BatchManifest) -> ArtifactInput:
    """Save the manifest as an immutable artifact whose inputs are its saved outcomes.

    Args:
        project: Project holding the outcomes.
        manifest: The manifest to freeze.

    Returns:
        The frozen batch reference; the same manifest freezes to the same artifact.
    """
    store = project.artifacts
    entries = [row.model_dump(mode="json", exclude={"updated_at"}) for row in manifest.entries]
    artifact_id = stable_id("start-bootstrap-frozen", {"batch": manifest.batch, "rows": entries})
    if not store.exists(artifact_id):
        outcomes = {row.outcome for row in manifest.entries if row.outcome is not None}
        store.write_json(
            artifact_id=artifact_id,
            artifact_type=BATCH_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=manifest.code_revision,
                inputs=tuple(sorted(outcomes, key=lambda ref: ref.artifact_id)),
            ),
            files={"batch.json": manifest},
        )
    return artifact_input(store.read(artifact_id).manifest)


def registration_entry(project: ProjectStore, binding: BootstrapBinding) -> ManifestEntry:
    """Return the registered row a binding names in its frozen batch.

    Raises:
        ValueError: The batch is not the pinned frozen batch, or the task has no registration.
    """
    store = project.artifacts
    stored = store.read(binding.batch.artifact_id)
    if artifact_input(stored.manifest) != binding.batch or (
        stored.manifest.artifact_type != BATCH_TYPE
    ):
        raise ValueError("the bootstrap binding does not name a pinned frozen batch")
    manifest = BatchManifest.model_validate_json(
        store.read_bytes(binding.batch.artifact_id, "batch.json")
    )
    entry = manifest.entry(binding.task_id)
    if entry is None or entry.status != "registered" or entry.outcome is None:
        raise ValueError(f"task {binding.task_id} has no registered start in the batch")
    return entry


def registered_job(project: ProjectStore, binding: BootstrapBinding) -> ContinuationJob:
    """Return the mining job one task registered in a frozen batch.

    Args:
        project: Project holding the batch.
        binding: The frozen batch and the task.

    Returns:
        The registered job.

    Raises:
        ValueError: The binding does not resolve, or the registration differs from the batch.
    """
    entry = registration_entry(project, binding)
    registration = load_outcome(project, entry.identity)
    if not isinstance(registration, BootstrapRegistration):
        raise ValueError(f"task {binding.task_id} has no saved registration")
    if sha256_json(registration.job) != entry.job_sha256:
        raise ValueError(f"task {binding.task_id} registration differs from the frozen batch")
    return registration.job


def resolve_bootstrap(project: ProjectStore, spec: DecisionVarianceSpec) -> DecisionVarianceSpec:
    """Resolve a pipeline registration's bootstrap binding into its mining job.

    The pipeline entry calls this before planning, so every stage (the rules stage's planning
    and keys included) reads the job the bootstrap registered, never a hand-registered one.

    Args:
        project: Project holding the frozen batch.
        spec: The pipeline registration.

    Returns:
        The registration with ``mining.job`` set to the registered job; unchanged without a
        binding.

    Raises:
        ValueError: Neither a job nor a binding is given, the binding does not resolve, or a job
            given beside it differs.
    """
    binding = spec.mining.bootstrap
    if binding is None:
        if spec.mining.job is None:
            raise ValueError(
                "register a mining job or bind a task's start-bootstrap batch; the front door "
                "binds every start its preparation registers"
            )
        return spec
    job = registered_job(project, binding)
    if spec.mining.job is not None and spec.mining.job != job:
        raise ValueError("mining.job differs from the job the bootstrap registered for the task")
    return spec.model_copy(update={"mining": spec.mining.model_copy(update={"job": job})})


def _task_call(
    project: ProjectStore,
    spec: BootstrapSpec,
    family: BootstrapFamily,
    task: BootstrapTask,
    knowledge: FamilyKnowledge | None,
    identity: str,
    exposure: ExposurePlan,
    traces: Mapping[str, Trace],
    code_revision: str,
    previous: ManifestEntry | None,
    record: Callable[[ManifestEntry], None],
) -> Callable[[], None]:
    """Return one task's isolated run: replay, bootstrap, or record its failure."""

    def run() -> None:
        saved = load_outcome(project, identity)
        attempts = 0 if previous is None else previous.attempts
        if saved is None:
            attempts += 1
            try:
                assert knowledge is not None
                saved = _bootstrap_task(
                    project, spec, family, task, knowledge, identity, exposure, traces,
                    code_revision,
                )  # fmt: skip
            except Exception as error:  # noqa: BLE001 - one failing task never blocks the others
                record(_failed(family, identity, task, previous, error, "task"))
                return
        record(_entry(project, family, identity, saved, attempts))

    return run


def _bootstrap_task(
    project: ProjectStore,
    spec: BootstrapSpec,
    family: BootstrapFamily,
    task: BootstrapTask,
    knowledge: FamilyKnowledge,
    identity: str,
    exposure: ExposurePlan,
    traces: Mapping[str, Trace],
    code_revision: str,
) -> Outcome:
    """Author, review, research, capture, validate and register one task, under its lock."""
    store = project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = load_outcome(project, identity)
        if saved is not None:
            return saved
        exposure.require_lineage(task.task.lineage_group_id, "mining_probes")
        missing = sorted(set(task.source_traces) - set(traces))
        if missing:
            raise ValueError(f"source traces of {task.task.task_id} were not supplied: {missing}")
        exposure.require(task.source_traces, "mining_probes")
        turns = max(
            (
                sum(span.model is not None for span in traces[trace].spans)
                for trace in task.source_traces
            ),
            default=0,
        )
        spec = spec.model_copy(update={"job": spec.job.with_horizon(turns)})
        recorded = _recorded_state(project, task.recorded_state)
        material = StartMaterial(
            knowledge, task, {trace: traces[trace] for trace in task.source_traces}, recorded
        )
        refs = {
            artifact_input(store.read(identity + "-evidence").manifest),
            knowledge.rulebook,
            *(() if knowledge.report is None else (knowledge.report,)),
        }
        if knowledge.rule_memory is not None:
            refs.add(knowledge.rule_memory.memory_set)
        if task.recorded_state is not None:
            refs.add(task.recorded_state)
        sources = tuple(sorted(refs, key=lambda ref: ref.artifact_id))
        rounds: list[ReviewRound] = []
        findings: tuple[Finding, ...] = ()
        feedback: JsonObject = {}
        refuse = _refuser(project, family, task, identity, code_revision, rounds)
        draft = None
        for round_ in range(spec.research_rounds + 1):
            author, draft = author_start(
                project, spec, material, identity=identity, round_=round_, feedback=feedback,
                sources=sources, code_revision=code_revision,
            )  # fmt: skip
            if draft is None:
                rounds.append(ReviewRound(round=round_, author=author))
                return refuse("author_unusable", f"no usable draft: {author.failure}", findings)
            reviews = review_start(
                project, spec, material, draft, identity=identity, round_=round_,
                sources=sources, code_revision=code_revision,
            )  # fmt: skip
            verdict, items, diagnosis = reconcile_reviews(
                round_, reviews, draft, spec.research_items
            )
            rounds.append(
                ReviewRound(
                    round=round_,
                    author=author,
                    reviews=tuple(draw for draw, _ in reviews),
                    verdict=verdict,
                    items=items,
                )
            )
            if verdict is None:
                return refuse("review_unusable", diagnosis, findings)
            if verdict == "refuse":
                return refuse("review_refused", diagnosis, findings)
            if verdict == "accept":
                break
            if round_ == spec.research_rounds:
                return refuse(
                    "research_exhausted",
                    f"deciding facts are still missing after {spec.research_rounds} research "
                    f"rounds: {[item.question for item in items]}",
                    findings,
                    unresolved=items,
                )
            new = research(
                project, spec, material, items, identity=identity, sources=sources,
                code_revision=code_revision,
            )  # fmt: skip
            findings = (*findings, *new)
            material = material.with_findings(findings)
            feedback = {
                "reviews": [
                    answer.model_dump(mode="json") for _, answer in reviews if answer is not None
                ],
                "research_items": [item.model_dump(mode="json") for item in items],
                "new_findings": [f"finding:{row.finding_id}" for row in new],
            }
        assert draft is not None
        template = spec.runtime
        definition = start_definition(template, family.family, knowledge, task, draft)
        evidence = {
            *sources,
            *(draw.transcript for row in rounds for draw in (row.author, *row.reviews)),
            *(row.transcript for row in findings),
        }
        checkpoint = capture_start(
            project, spec, template, definition,
            [ref for ref in evidence if ref is not None], code_revision=code_revision,
        )  # fmt: skip
        validation = validate_start(
            project, spec, template, material, draft, definition, checkpoint,
            identity=identity, sources=sources, code_revision=code_revision,
        )  # fmt: skip
        if not validation.passed:
            return refuse("validation_failed", validation.detail, findings, validation=validation)
        registration = BootstrapRegistration(
            family_id=family.family.family_id,
            task_id=task.task.task_id,
            identity=identity,
            provenance=draft.provenance,
            claims_original=draft.provenance == "observed",
            statement=PROVENANCE_STATEMENTS[draft.provenance],
            rounds=tuple(rounds),
            findings=findings,
            rules_report=knowledge.report,
            rulebook=knowledge.rulebook,
            rule_memory=definition.rule_memory,
            draft=draft,
            checkpoint=checkpoint,
            validation=validation,
            job=mining_job(project, spec, template, definition, checkpoint, draft),
        )
        _save(project, identity + "-registration", REGISTRATION_TYPE, "registration.json",
              registration, _inputs(registration.rounds, findings, validation, checkpoint, sources),
              code_revision)  # fmt: skip
        return registration


def _refuser(
    project: ProjectStore,
    family: BootstrapFamily,
    task: BootstrapTask,
    identity: str,
    code_revision: str,
    rounds: Sequence[ReviewRound],
) -> Callable[..., BootstrapRefusal]:
    """Return a function saving one diagnostic refusal for this task."""

    def refuse(
        kind: RefusalKind,
        diagnosis: str,
        findings: tuple[Finding, ...],
        *,
        unresolved: tuple[ResearchItem, ...] = (),
        validation: StartValidation | None = None,
    ) -> BootstrapRefusal:
        refusal = BootstrapRefusal(
            family_id=family.family.family_id,
            task_id=task.task.task_id,
            identity=identity,
            kind=kind,
            diagnosis=diagnosis,
            rounds=tuple(rounds),
            findings=findings,
            unresolved=unresolved,
            validation=validation,
        )
        evidence = artifact_input(project.artifacts.read(identity + "-evidence").manifest)
        inputs = _inputs(refusal.rounds, findings, validation, None, (evidence,))
        _save(project, identity + "-refusal", REFUSAL_TYPE, "refusal.json", refusal, inputs,
              code_revision)  # fmt: skip
        return refusal

    return refuse


def _inputs(
    rounds: Sequence[ReviewRound],
    findings: Sequence[Finding],
    validation: StartValidation | None,
    checkpoint: ArtifactInput | None,
    sources: Sequence[ArtifactInput],
) -> tuple[ArtifactInput, ...]:
    """Every saved artifact an outcome rests on."""
    refs = {*sources, *(row.transcript for row in findings)}
    refs |= {draw.transcript for row in rounds for draw in (row.author, *row.reviews)}
    if validation is not None:
        refs |= {draw.transcript for draw in validation.checks}
    if checkpoint is not None:
        refs.add(checkpoint)
    return tuple(sorted((ref for ref in refs if ref is not None), key=lambda r: r.artifact_id))


def _save(
    project: ProjectStore,
    artifact_id: str,
    artifact_type: str,
    name: str,
    value: Outcome,
    inputs: tuple[ArtifactInput, ...],
    code_revision: str,
) -> None:
    """Save one outcome artifact."""
    project.artifacts.write_json(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=inputs,
        ),
        files={name: value},
    )


def _entry(
    project: ProjectStore,
    family: BootstrapFamily,
    identity: str,
    outcome: Outcome,
    attempts: int,
) -> ManifestEntry:
    """Return the manifest row of a saved outcome."""
    registered = isinstance(outcome, BootstrapRegistration)
    artifact_id = identity + ("-registration" if registered else "-refusal")
    return ManifestEntry(
        family_id=family.family.family_id,
        task_id=outcome.task_id,
        identity=identity,
        status="registered" if registered else "refused",
        outcome=artifact_input(project.artifacts.read(artifact_id).manifest),
        provenance=outcome.provenance if registered else None,
        job_sha256=sha256_json(outcome.job) if registered else None,
        failure=None if registered else f"{outcome.kind}: {outcome.diagnosis}",
        attempts=attempts,
        updated_at=datetime.now(UTC),
    )


def _failed(
    family: BootstrapFamily,
    identity: str,
    task: BootstrapTask,
    previous: ManifestEntry | None,
    error: Exception,
    where: str,
) -> ManifestEntry:
    """Return the manifest row of a failed attempt; it is retried on the next run."""
    text = (str(error) or type(error).__name__)[:2000]
    return ManifestEntry(
        family_id=family.family.family_id,
        task_id=task.task.task_id,
        identity=identity,
        status="failed",
        failure=f"{where}: {type(error).__name__}: {text}",
        attempts=(0 if previous is None else previous.attempts) + 1,
        updated_at=datetime.now(UTC),
    )


def _recorded_state(project: ProjectStore, reference: ArtifactInput | None) -> JsonObject | None:
    """Read a pinned recorded state snapshot."""
    if reference is None:
        return None
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError("the recorded state differs from its pinned reference")
    value = json.loads(project.artifacts.read_bytes(reference.artifact_id, RECORDED_STATE))
    if not isinstance(value, dict):
        raise ValueError("a recorded state snapshot is a JSON object of domains")
    return value


def _manifest_path(project: ProjectStore, batch: str) -> Path:
    """Where a batch's manifest lives."""
    return project.artifacts.project_directory / MANIFEST_DIRECTORY / f"{batch}.json"


def _write_manifest(project: ProjectStore, manifest: BatchManifest) -> None:
    """Replace the batch manifest atomically."""
    path = _manifest_path(project, manifest.batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staging = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".json")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(manifest.model_dump_json(indent=2).encode())
        os.replace(staging, path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise

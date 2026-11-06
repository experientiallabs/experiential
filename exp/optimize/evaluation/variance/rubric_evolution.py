"""The evolving rubric: revise the two-part rubric at each epoch boundary, ahead of reward hacking.

A policy trained with GRPO against a frozen rubric learns whatever the rubric rewards, including
ways to satisfy it without doing the task. This loop keeps the rubric ahead. The rubric is frozen
within a training epoch and revised only at the boundary, by :func:`run_boundary`, which walks one
subject through :data:`~.rubric_evo_states.BOUNDARY_MACHINE` (every move checked against its
table and persisted to the evolution's journal):

1. **Ingest** the epoch's ``rollout-ingest-v1`` export (:mod:`~.rollout_ingest`). It must name the
   open epoch and the version frozen for it; anything else is refused, so epochs never mix.
   Rollouts of a held-out lineage are excluded from every later step.
2. **Tag** every actor step with the decision model over the version's step bank (the base
   behavior bank, the error bank and the active items' questions).
3. **Audit** a bounded sample of high-reward rollouts with the five-actor council
   (:mod:`~.rubric_audit`); a strict-majority fail flags a rollout. The epoch's mean in-loop
   reward and audit pass rate are recorded, and the **divergence trigger** compares them with the
   previous epoch: reward rising by ``reward_rise`` or more while the pass rate moves by
   ``audit_bound`` or less forces the next boundary (larger sample, no early exit). When a tag
   monitor is bound (:class:`EpochMonitor`), its drift evidence is read first and recorded on the
   trigger: reward rising by ``reward_rise`` while the monitor reports a hack tag drifting up
   also forces the next boundary, ahead of the audit.
4. **Diff** flagged rollouts against genuine passes by tag signature, authoring probe questions
   when the vocabulary cannot see the hack (:mod:`~.rubric_signatures`).
5. **Admit** each adversarial signature with the rule admission (two or more independent
   lineages, no contradiction), then **compile** it into a dense question and a judge criterion
   and **consolidate** against the active items (:mod:`~.rubric_items`).
6. **Retire** active items that no longer discriminate.
7. **Recall**: grade the held-out known-hack set under the candidate with both parts
   (:mod:`~.rubric_grading`); a candidate that lowers either recall is retried without its
   retirements, else the current version is kept.
8. **Publish** the candidate as the next version, frozen for the next epoch.

Every spend is priced (:func:`boundary_costs`) and, when a ledger is given, reserved against the
single cumulative front-door ledger before any call and settled after. A finished boundary is
saved as one artifact; running it again replays it with no call.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.files import write_text_atomic
from exp.common.project import ProjectStore
from exp.common.state_machine import Refusal, Transition, TransitionJournal
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent, decision_cost
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.dense_hybrid import RoutingSpec
from exp.optimize.evaluation.variance.dense_prompt import dense_model_calls
from exp.optimize.evaluation.variance.evidence_tools import EvidenceToolProfile
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger, measure
from exp.optimize.evaluation.variance.judge_prompt import judge_model_calls
from exp.optimize.evaluation.variance.pipeline_dense import MAX_STEP_QUESTIONS
from exp.optimize.evaluation.variance.pipeline_spec import COUNCIL_ACTORS
from exp.optimize.evaluation.variance.rollout_ingest import RolloutExport
from exp.optimize.evaluation.variance.rubric_audit import (
    AUDIT_STANCES,
    AuditSpec,
    RolloutAudit,
)
from exp.optimize.evaluation.variance.rubric_evo_states import BOUNDARY_MACHINE, BoundaryState
from exp.optimize.evaluation.variance.rubric_grading import HeldOutSet, TaskRubric, measure_recall
from exp.optimize.evaluation.variance.rubric_items import (
    ITEM_STANCES,
    Consolidation,
    ItemCompilation,
    RetirementCheck,
    RetirementSpec,
)
from exp.optimize.evaluation.variance.rubric_signatures import (
    PROBE_STANCES,
    AdmissionSpec,
    ProbeRecord,
    ProbeSpec,
    RolloutReading,
    SignatureAdmission,
    SignatureGroup,
)
from exp.optimize.evaluation.variance.rubric_versions import (
    RecallRecord,
    RubricVersion,
    first_version,
    load_version,
    save_version,
)
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.mining.decisions import DEFAULT_DECISION_ALIAS
from exp.simulation.mining.questions import QuestionBank, resolve_question_bank
from exp.simulation.mining.tagging import TraceTagger

RUBRIC_EVOLUTION_PROTOCOL = "rubric-evolution-v1"
"""Version of the boundary flow, its records and its identity; bump when any changes."""

BOUNDARY_TYPE = "rubric-evolution-boundary"
"""Artifact type of a saved boundary record (``boundary.json``)."""

AUTHOR_ALIASES = re.compile(r"^(glm-[0-9.]+-flash|deepseek-[a-z0-9.-]+)$")
"""Open-weight authoring aliases the loop accepts for probes, compilation and consolidation."""

B = BoundaryState


class TriggerSpec(ContractModel):
    """The divergence trigger.

    Attributes:
        reward_rise: Rise in mean in-loop reward, epoch over epoch, at or above which reward is
            rising.
        audit_bound: Change in audit pass rate at or below which the audit stayed flat or fell.
    """

    reward_rise: float = Field(default=0.05, ge=0.0)
    audit_bound: float = 0.0


class EvolutionSpec(ContractModel):
    """Frozen settings of the evolving rubric.

    Attributes:
        decision_alias: The open-weight decision model that tags steps and answers questions.
        auditors: The five-actor council, one settings per actor.
        author: The open-weight author (probes, compilation, consolidation).
        profile: Evidence-tool profile of every role draw.
        audit: Audit bounds.
        probes: Probe authoring and selection.
        admission: Signature compilation and admission.
        item_stances: Item drafting stances.
        retirement: Retirement rules.
        trigger: The divergence trigger.
        routing: Hybrid routing and reading thresholds.
        maximum_rollouts: Most rollouts one export may hold (the priced bound).
        maximum_turns: Most actor turns one rollout may hold (the priced bound).
    """

    decision_alias: str = Field(default=DEFAULT_DECISION_ALIAS, min_length=1)
    auditors: tuple[AuthorSettings, ...]
    author: AuthorSettings
    profile: EvidenceToolProfile = EvidenceToolProfile()
    audit: AuditSpec = AuditSpec()
    probes: ProbeSpec = ProbeSpec()
    admission: AdmissionSpec = AdmissionSpec()
    item_stances: tuple[str, ...] = tuple(ITEM_STANCES)
    retirement: RetirementSpec = RetirementSpec()
    trigger: TriggerSpec = TriggerSpec()
    routing: RoutingSpec = RoutingSpec()
    maximum_rollouts: int = Field(default=512, ge=1, le=100_000)
    maximum_turns: int = Field(default=64, ge=1, le=4096)

    @model_validator(mode="after")
    def _open_weight(self) -> EvolutionSpec:
        """Require the whole open-weight council and an open-weight author."""
        aliases = [row.model_alias for row in self.auditors]
        if sorted(aliases) != sorted(COUNCIL_ACTORS):
            raise ValueError(f"the audit council is exactly {list(COUNCIL_ACTORS)}")
        if AUTHOR_ALIASES.fullmatch(self.author.model_alias) is None:
            raise ValueError(f"author {self.author.model_alias} is not an open-weight author")
        if set(self.audit.stances) - set(AUDIT_STANCES):
            raise ValueError(f"audit stances must be members of {list(AUDIT_STANCES)}")
        if set(self.probes.stances) - set(PROBE_STANCES):
            raise ValueError(f"probe stances must be members of {list(PROBE_STANCES)}")
        return self


class EvolutionRegistration(ContractModel):
    """One training run's evolving rubric.

    Attributes:
        run_id: The trainer run whose exports are ingested.
        spec: Frozen settings.
        tasks: Every trained task's frozen two-part rubric.
        bank_ids: Banks the dense part is frozen from, in order.
        bank_root: Where those banks are published.
        pinned: The exact banks, when given (for example ``rubric_banks`` of the pipeline that
            exported the tasks, so version 1 here is the version the packages record); they
            replace ``bank_ids``.
        held_out: The held-out known-hack set.
    """

    run_id: str = Field(min_length=1, max_length=256)
    spec: EvolutionSpec
    tasks: tuple[TaskRubric, ...] = Field(min_length=1)
    bank_ids: tuple[str, ...] = ("behavior-base-v1", "error-taxonomy-v1")
    bank_root: Path | None = None
    pinned: tuple[QuestionBank, ...] = ()
    held_out: HeldOutSet

    @property
    def identity(self) -> Sha256:
        """Digest of the registration."""
        return sha256_json(self)

    def task_map(self) -> dict[str, TaskRubric]:
        """Task rubrics by task."""
        return {row.task_id: row for row in self.tasks}

    def banks(self) -> list[QuestionBank]:
        """The pinned banks: ``pinned`` when given, else ``bank_ids`` resolved."""
        if self.pinned:
            return list(self.pinned)
        return [resolve_question_bank(b, root=self.bank_root) for b in self.bank_ids]


class EpochMetrics(ContractModel):
    """One epoch's reward and audit measurements.

    Attributes:
        epoch: The epoch.
        rollouts: Ingested rollouts.
        mean_reward: Mean in-loop reward over them.
        audited: Rollouts audited.
        decided: Audited rollouts with a council majority.
        audit_pass_rate: Passes over decided, None when none was decided.
    """

    epoch: int = Field(ge=1)
    rollouts: int = Field(ge=0)
    mean_reward: float | None = None
    audited: int = Field(ge=0)
    decided: int = Field(ge=0)
    audit_pass_rate: float | None = None


class MonitorEvidence(ContractModel):
    """What the tag monitor reported at one boundary, as divergence-trigger evidence.

    Attributes:
        report: The saved monitor report, when it was saved.
        drifted: Hack tags the monitor's statistics flagged as rising with reward and Astra
            cited in a drift proposal.
        stopped_discriminating: Tags whose rate left the usable band, handed to retirement.
        halted: Whether the monitor raised a halt.
        reason: One sentence.
    """

    report: ArtifactInput | None = None
    drifted: tuple[str, ...] = ()
    stopped_discriminating: tuple[str, ...] = ()
    halted: bool = False
    reason: str = Field(default="", max_length=1000)


class EpochMonitor(Protocol):
    """A tag monitor bound to the epoch boundary (checkpoint 6)."""

    @property
    def identity(self) -> str:
        """Identity of the monitor's settings; it joins the boundary identity."""
        ...

    def observe(
        self,
        project: ProjectStore,
        registration: EvolutionRegistration,
        state: EvolutionState,
        epoch: int,
        readings: Sequence[RolloutReading],
        rewards: Mapping[str, float],
    ) -> MonitorEvidence:
        """Measure the ending epoch's tag rates against earlier epochs and report drift."""
        ...


class TriggerRecord(ContractModel):
    """The divergence trigger at one boundary.

    Attributes:
        epoch: The epoch measured.
        previous_epoch: The epoch compared with, None at the first boundary.
        reward_delta: Rise in mean in-loop reward.
        audit_delta: Change in audit pass rate.
        reward_rise: The configured rise.
        audit_bound: The configured audit bound.
        fired: Whether the next boundary is forced.
        reason: Why, in one sentence.
        monitor: The tag monitor's evidence, when a monitor is bound.
    """

    epoch: int
    previous_epoch: int | None = None
    reward_delta: float | None = None
    audit_delta: float | None = None
    reward_rise: float
    audit_bound: float
    fired: bool
    reason: str
    monitor: MonitorEvidence | None = Field(default=None, exclude_if=lambda value: value is None)


def divergence_trigger(
    previous: EpochMetrics | None,
    current: EpochMetrics,
    spec: TriggerSpec,
    monitor: MonitorEvidence | None = None,
) -> TriggerRecord:
    """Decide whether reward rose while the audit stayed flat or fell past the bound.

    With monitor evidence, reward rising by ``reward_rise`` while the monitor reports drifting
    hack tags also fires, even when the audit has not caught up yet.
    """
    base = TriggerRecord(
        epoch=current.epoch,
        reward_rise=spec.reward_rise,
        audit_bound=spec.audit_bound,
        fired=False,
        reason="first boundary: nothing to compare",
        monitor=monitor,
    )
    if previous is None:
        return base
    rewarded = previous.mean_reward is not None and current.mean_reward is not None
    if not rewarded or previous.audit_pass_rate is None or current.audit_pass_rate is None:
        unread = base.model_copy(
            update={
                "previous_epoch": previous.epoch,
                "reason": "a reward or audit pass rate is unmeasured, so it cannot be read",
            }
        )
        if not rewarded or monitor is None or not monitor.drifted:
            return unread
        assert previous.mean_reward is not None and current.mean_reward is not None
        reward = current.mean_reward - previous.mean_reward
        fired = reward >= spec.reward_rise
        return unread.model_copy(
            update={
                "reward_delta": round(reward, 9),
                "fired": fired,
                "reason": f"in-loop reward rose {reward:+.3f} with the audit unmeasured; the tag "
                f"monitor reports drift in {', '.join(monitor.drifted)}"
                + (", so the next boundary is forced" if fired else "; no divergence"),
            }
        )
    assert previous.mean_reward is not None and current.mean_reward is not None
    reward = current.mean_reward - previous.mean_reward
    audit = current.audit_pass_rate - previous.audit_pass_rate
    fired = reward >= spec.reward_rise and audit <= spec.audit_bound
    reason = f"in-loop reward rose {reward:+.3f} while the audit pass rate moved {audit:+.3f}"
    if monitor is not None and monitor.drifted:
        reason += f" and the tag monitor reports drift in {', '.join(monitor.drifted)}"
        fired = fired or reward >= spec.reward_rise
    return base.model_copy(
        update={
            "previous_epoch": previous.epoch,
            "reward_delta": round(reward, 9),
            "audit_delta": round(audit, 9),
            "fired": fired,
            "reason": reason + (", so the next boundary is forced" if fired else "; no divergence"),
        }
    )


class EpochRecord(ContractModel):
    """One epoch of the run.

    Attributes:
        epoch: The epoch.
        version: The rubric version frozen for it.
        version_sha256: That version's identity.
        version_ref: The saved version.
        forced: Whether the boundary closing it is forced.
        boundary: The saved boundary that closed it, once closed.
        metrics: Its measurements, once closed.
        trigger: The trigger read at its boundary.
    """

    epoch: int = Field(ge=1)
    version: int = Field(ge=1)
    version_sha256: Sha256
    version_ref: ArtifactInput
    forced: bool = False
    boundary: ArtifactInput | None = None
    metrics: EpochMetrics | None = None
    trigger: TriggerRecord | None = None


class EvolutionState(ContractModel):
    """The run's persisted progress.

    Attributes:
        run_id: The run.
        registration_sha256: The registration it belongs to.
        versions: Every published version, in order.
        epochs: Every epoch, the last one open.
    """

    run_id: str
    registration_sha256: Sha256
    versions: tuple[ArtifactInput, ...] = Field(min_length=1)
    epochs: tuple[EpochRecord, ...] = Field(min_length=1)

    @property
    def open_epoch(self) -> EpochRecord:
        """The epoch training now (the last)."""
        return self.epochs[-1]


class BoundaryRecord(ContractModel):
    """Everything one epoch boundary did, saved once.

    Attributes:
        protocol: Flow protocol.
        run_id: The run.
        epoch: The epoch that ended.
        forced: Whether the boundary was forced.
        export_sha256: The ingested export.
        version_in: The version the epoch trained under.
        state: The final boundary state.
        transitions: Every applied move.
        refusals: Every refused move.
        reason: Why the boundary ended where it did.
        excluded: Rollouts left out, with why.
        readings: Every rollout's step tags, reduced.
        audits: The council audits.
        metrics: The epoch's measurements.
        trigger: The divergence trigger.
        flagged: Rollouts the audit failed.
        genuine: Rollouts the audit passed.
        groups: Adversarial signature groups.
        unexplained: Flagged rollouts no signature explains.
        probes: Probe authoring and selection, when it ran.
        admissions: Every signature admission.
        compilations: Every admitted signature's drafts.
        consolidation: The consolidation, when it ran.
        retirement: Every re-measured item.
        recall: Held-out recall of each candidate checked, in order.
        version_out: The version frozen for the next epoch.
        version_out_sha256: Its identity.
    """

    protocol: str = RUBRIC_EVOLUTION_PROTOCOL
    run_id: str
    epoch: int
    forced: bool
    export_sha256: Sha256
    version_in: Sha256
    state: str
    transitions: tuple[Transition, ...] = ()
    refusals: tuple[Refusal, ...] = ()
    reason: str = ""
    excluded: dict[str, str] = Field(default_factory=dict)
    readings: tuple[RolloutReading, ...] = ()
    audits: tuple[RolloutAudit, ...] = ()
    metrics: EpochMetrics | None = None
    trigger: TriggerRecord | None = None
    flagged: tuple[str, ...] = ()
    genuine: tuple[str, ...] = ()
    groups: tuple[SignatureGroup, ...] = ()
    unexplained: tuple[str, ...] = ()
    probes: ProbeRecord | None = None
    admissions: tuple[SignatureAdmission, ...] = ()
    compilations: tuple[ItemCompilation, ...] = ()
    consolidation: Consolidation | None = None
    retirement: tuple[RetirementCheck, ...] = ()
    recall: tuple[RecallRecord, ...] = ()
    version_out: ArtifactInput | None = None
    version_out_sha256: Sha256 | None = None


class BoundaryWalk:
    """One boundary's walk through the machine, every move checked and journaled."""

    def __init__(self, subject: str, journal: TransitionJournal) -> None:
        """Start the subject in ``opened`` (journaled as its start)."""
        self.subject, self.journal = subject, journal
        self.state = BOUNDARY_MACHINE.move(subject, None, B.OPENED.value, journal)
        self.transitions: list[Transition] = []
        self.refusals: list[Refusal] = []

    def go(self, target: BoundaryState, reason: str) -> None:
        """Apply one move; an undeclared move is journaled as refused and raised."""
        row = {
            "machine": BOUNDARY_MACHINE.name,
            "subject": self.subject,
            "source": self.state.value,
            "target": target.value,
            "reason": reason[:500],
        }
        try:
            self.state = BOUNDARY_MACHINE.move(
                self.subject, self.state, target.value, self.journal, target=target
            )
        except Exception:
            self.refusals.append(Refusal.model_validate(row))
            raise
        self.transitions.append(Transition.model_validate(row))


def evolution_directory(project: ProjectStore, registration: EvolutionRegistration) -> Path:
    """The run's mutable state directory: ``state.json`` and the transition journal."""
    return (
        project.paths.runtime_directory
        / "rubric-evolution"
        / f"{registration.run_id[:64]}-{registration.identity[:12]}"
    )


def journal_of(project: ProjectStore, registration: EvolutionRegistration) -> TransitionJournal:
    """The run's transition journal (every accepted and refused boundary move)."""
    return TransitionJournal(evolution_directory(project, registration) / "journal.jsonl")


def load_state(project: ProjectStore, registration: EvolutionRegistration) -> EvolutionState | None:
    """Read the run's saved state, or None before :func:`open_evolution`."""
    path = evolution_directory(project, registration) / "state.json"
    if not path.exists():
        return None
    state = EvolutionState.model_validate_json(path.read_bytes())
    if state.registration_sha256 != registration.identity:
        raise ValueError("the saved evolution state belongs to another registration")
    return state


def save_state(
    project: ProjectStore, registration: EvolutionRegistration, state: EvolutionState
) -> None:
    """Persist the run's state atomically."""
    path = evolution_directory(project, registration) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, state.model_dump_json(indent=2))


def current_version(project: ProjectStore, registration: EvolutionRegistration) -> RubricVersion:
    """The version frozen for the open epoch.

    Raises:
        ValueError: The evolution was not opened.
    """
    state = load_state(project, registration)
    if state is None:
        raise ValueError("open the evolution before reading its version")
    return load_version(project, state.open_epoch.version_ref)


def _recall_costs(
    project: ProjectStore, registration: EvolutionRegistration, candidates: int
) -> list[CostComponent]:
    """Price grading every held-out hack under ``candidates`` versions with both parts."""
    hacks = len(registration.held_out.hacks) * candidates
    costs: list[CostComponent] = []
    tasks = registration.task_map()
    for row in registration.held_out.hacks:
        task = tasks.get(row.rollout.task_id) or registration.tasks[0]
        outcome = judge_model_calls(task.judge.definition.profile) * candidates
        dense = dense_model_calls(task.dense.definition.profile) * candidates
        costs.extend(c.scaled(outcome) for c in evaluation_costs(project, task.judge.settings))
        costs.extend(c.scaled(dense) for c in evaluation_costs(project, task.dense.settings))
    questions = (
        sum(len(b.questions) for b in registration.banks())
        + 2 * registration.spec.audit.forced_sample
    )
    steps = hacks * registration.spec.maximum_turns * math.ceil(questions / MAX_STEP_QUESTIONS)
    costs.append(decision_cost(project, registration.spec.decision_alias, steps))
    return costs


def opening_costs(
    project: ProjectStore, registration: EvolutionRegistration
) -> tuple[CostComponent, ...]:
    """Price opening the evolution: version 1's held-out recall."""
    return tuple(_recall_costs(project, registration, 1))


def boundary_costs(
    project: ProjectStore, registration: EvolutionRegistration, *, forced: bool
) -> tuple[CostComponent, ...]:
    """Price one boundary's upper bound: every decision, council, author and judge request.

    The bound assumes the export is at ``maximum_rollouts`` of ``maximum_turns`` turns, every
    audited rollout is flagged and is its own signature, and the recall check runs twice (once
    without retirements). Every component is a finite request count; unpriced components are
    reserved as attempts against the cumulative ledger.
    """
    spec = registration.spec
    sample = spec.audit.forced_sample if forced else spec.audit.sample
    retries = RetryPolicy().maximum_attempts
    calls = spec.profile.budget.model_calls
    questions = sum(len(b.questions) for b in registration.banks()) + 2 * spec.audit.forced_sample
    steps = spec.maximum_rollouts * spec.maximum_turns * 2
    costs: list[CostComponent] = [
        decision_cost(
            project,
            spec.decision_alias,
            steps * math.ceil(questions / MAX_STEP_QUESTIONS) * retries,
        )
    ]
    for auditor in spec.auditors:
        costs.extend(row.scaled(sample * calls) for row in evaluation_costs(project, auditor))
    author_draws = (
        len(spec.probes.stances)
        + sample * spec.admission.compilations
        + sample * len(spec.item_stances)
        + 1
    )
    costs.extend(row.scaled(author_draws * calls) for row in evaluation_costs(project, spec.author))
    probe_steps = sample * spec.maximum_turns * 2 * math.ceil(8 * len(spec.probes.stances) / 32)
    admission_units = (
        sample * spec.maximum_rollouts * math.ceil(12 * spec.admission.compilations / 32)
    )
    costs.append(
        decision_cost(project, spec.decision_alias, (probe_steps + admission_units) * retries)
    )
    costs.extend(_recall_costs(project, registration, 2))
    return tuple(costs)


def reserve(ledger: PhaseLedger | None, phase: str, costs: Sequence[CostComponent]) -> str | None:
    """Reserve a phase's bound on the cumulative ledger, or return why it is refused."""
    if ledger is None:
        return None
    usd = sum(row.usd for row in costs if row.usd is not None)
    attempts = sum(row.attempts for row in costs if row.usd is None)
    return ledger.admit(phase, usd, attempts)


def open_evolution(
    project: ProjectStore,
    registration: EvolutionRegistration,
    *,
    tagger: TraceTagger,
    code_revision: str,
    ledger: PhaseLedger | None = None,
) -> EvolutionState:
    """Freeze version 1 for epoch 1, with its held-out recall, or return the saved state.

    Raises:
        ValueError: The ledger refused the opening's bound.
    """
    state = load_state(project, registration)
    if state is not None:
        return state
    phase = f"rubric-evolution {registration.run_id} open"
    refusal = reserve(ledger, phase, opening_costs(project, registration))
    if refusal is not None:
        raise ValueError(refusal)
    before = measure(project)
    banks = registration.banks()
    version = first_version(registration.run_id, banks)
    recall, _ = measure_recall(
        project,
        version,
        banks,
        registration.task_map(),
        registration.held_out,
        tagger=tagger,
        routing=registration.spec.routing,
        code_revision=code_revision,
    )
    version = version.model_copy(update={"recall": recall})
    reference = save_version(project, version, code_revision=code_revision)
    if ledger is not None:
        ledger.settle(phase, before, measure(project))
    state = EvolutionState(
        run_id=registration.run_id,
        registration_sha256=registration.identity,
        versions=(reference,),
        epochs=(
            EpochRecord(epoch=1, version=1, version_sha256=version.identity, version_ref=reference),
        ),
    )
    save_state(project, registration, state)
    return state


def boundary_identity(
    registration: EvolutionRegistration, export: RolloutExport, monitor: str | None = None
) -> str:
    """The identity a finished boundary is saved under, computed before any call.

    A bound tag monitor's identity joins it, since its evidence can force the next boundary; a
    boundary without one keeps the identity it always had.
    """
    material = {
        "protocol": RUBRIC_EVOLUTION_PROTOCOL,
        "registration": registration.identity,
        "export": export.identity,
    }
    if monitor is not None:
        material["monitor"] = monitor
    return stable_id(BOUNDARY_TYPE, material)


def ingest_problem(
    state: EvolutionState, registration: EvolutionRegistration, export: RolloutExport
) -> str | None:
    """Why an export cannot be ingested at the open epoch, or None."""
    open_epoch = state.open_epoch
    if export.trainer_run != registration.run_id:
        return f"the export is run {export.trainer_run}'s, not {registration.run_id}'s"
    if export.epoch != open_epoch.epoch:
        return f"the export is epoch {export.epoch}'s; epoch {open_epoch.epoch} is open"
    if (export.rubric_version, export.rubric_sha256) != (
        open_epoch.version,
        open_epoch.version_sha256,
    ):
        return (
            f"epoch {export.epoch} trained under rubric version {open_epoch.version} "
            f"({open_epoch.version_sha256[:12]}), not {export.rubric_version} "
            f"({export.rubric_sha256[:12]}); epochs never mix"
        )
    if len(export.rollouts) > registration.spec.maximum_rollouts:
        return f"the export holds more than {registration.spec.maximum_rollouts} rollouts"
    tasks = registration.task_map()
    for rollout in export.rollouts:
        if rollout.task_id not in tasks:
            return f"rollout {rollout.rollout_id} names unregistered task {rollout.task_id}"
        if len(rollout.turns) + len(rollout.transcript) > registration.spec.maximum_turns:
            return f"rollout {rollout.rollout_id} exceeds {registration.spec.maximum_turns} turns"
    return None

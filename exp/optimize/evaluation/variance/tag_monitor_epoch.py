"""Checkpoint 6: the tag monitor at every RL epoch boundary, feeding the divergence trigger.

:class:`TagEpochMonitor` is the evolving rubric's :class:`~.rubric_evolution.EpochMonitor`.
:func:`~.rubric_boundary.run_boundary` calls it after the epoch's steps are tagged and audited and
before the divergence trigger is read. It assembles every closed epoch's step readings (from the
saved boundary records) and the ending epoch's, measures each step-bank tag's rate per epoch
(:func:`~.tag_monitor_reward_stats.epoch_stats`), lets Astra review the flagged drift, and returns
:class:`~.rubric_evolution.MonitorEvidence`:

- ``drifted``: hack-family tags (``*.negative`` and ``error.*`` in the version's banks) whose rate
  rose by ``drift_delta`` or more while mean reward rose by ``reward_rise`` or more, that Astra
  confirmed with a cited drift proposal. Without an accepted review, the statistically flagged
  tags are passed with the reason saying they are unreviewed. The trigger records the evidence and
  forces the next boundary when reward rose with drift, ahead of the council audit.
- ``stopped_discriminating``: tags whose rate left the usable band, handed to the rubric's
  retirement as proposals routed to the rubric evolution.

The review is reserved on the ledger the monitor is bound with (None inside a front-door phase,
which already reserved it), and a finished checkpoint replays with no call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.dense_vocabulary import vocabulary
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger
from exp.optimize.evaluation.variance.rubric_boundary import load_boundary
from exp.optimize.evaluation.variance.rubric_evolution import (
    EvolutionRegistration,
    EvolutionState,
    MonitorEvidence,
)
from exp.optimize.evaluation.variance.rubric_signatures import RolloutReading, polarity
from exp.optimize.evaluation.variance.rubric_versions import load_version, version_banks
from exp.optimize.evaluation.variance.tag_monitor import run_checkpoint
from exp.optimize.evaluation.variance.tag_monitor_contracts import TAG_MONITOR_PROTOCOL, MonitorSpec
from exp.optimize.evaluation.variance.tag_monitor_review import AstraSettings
from exp.optimize.evaluation.variance.tag_monitor_reward_stats import EpochInputs, EpochTags


def epoch_inputs(
    project: ProjectStore,
    registration: EvolutionRegistration,
    state: EvolutionState,
    epoch: int,
    readings: Sequence[RolloutReading],
    rewards: Mapping[str, float],
) -> tuple[EpochInputs, list[ArtifactInput]]:
    """Return every epoch's reduced readings, oldest first, and the boundary records read.

    The tags are the open epoch's frozen step bank; hack tags come from its banks' families.
    """
    version = load_version(project, state.open_epoch.version_ref)
    banks = version_banks(version, registration.banks())
    _, step = vocabulary(banks)
    negative, _ = polarity(banks)
    epochs: list[EpochTags] = []
    sources: list[ArtifactInput] = []
    for closed in state.epochs:
        if closed.epoch >= epoch or closed.boundary is None:
            continue
        record = load_boundary(project, closed.boundary)
        sources.append(closed.boundary)
        epochs.append(
            EpochTags(
                epoch=closed.epoch,
                mean_reward=None if closed.metrics is None else closed.metrics.mean_reward,
                fired={r.rollout_id: r.fired for r in record.readings if r.tagged},
            )
        )
    values = list(rewards.values())
    epochs.append(
        EpochTags(
            epoch=epoch,
            mean_reward=sum(values) / len(values) if values else None,
            fired={r.rollout_id: r.fired for r in readings if r.tagged},
            rewards=dict(rewards),
        )
    )
    tags = step.question_ids()
    inputs = EpochInputs(
        epochs=tuple(epochs), tags=tags, hack_tags=tuple(t for t in tags if t in negative)
    )
    return inputs, sources


def refuse_closed_epoch_author(astra: AstraSettings | None) -> None:
    """Refuse Astra settings whose author is not open-weight at the epoch boundary.

    Raises:
        ValueError: The settings declare an ``astra_alias`` or name a closed author.
    """
    if astra is not None and (astra.astra_alias is not None or not astra.open_weight):
        raise ValueError(
            "the epoch monitor (checkpoint 6) runs at training time and needs an open-weight "
            f"author, not {astra.author.model_alias}"
        )


@dataclass(frozen=True)
class TagEpochMonitor:
    """The tag monitor bound to the epoch boundary.

    Attributes:
        astra: Astra's review settings, or None for statistics only.
        spec: Thresholds and bounds.
        code_revision: Producing source identity.
        ledger: The cumulative ledger the review is reserved on; None inside a front-door phase.
    """

    astra: AstraSettings | None
    spec: MonitorSpec = field(default_factory=MonitorSpec)
    code_revision: str = "unknown"
    ledger: PhaseLedger | None = None

    def __post_init__(self) -> None:
        """Refuse a closed author: checkpoint 6 runs at training time, open-weight only."""
        refuse_closed_epoch_author(self.astra)

    @property
    def identity(self) -> str:
        """Identity of the monitor's settings."""
        return sha256_json(
            {
                "protocol": TAG_MONITOR_PROTOCOL,
                "astra": None if self.astra is None else self.astra.model_dump(mode="json"),
                "spec": self.spec.model_dump(mode="json"),
            }
        )

    def observe(
        self,
        project: ProjectStore,
        registration: EvolutionRegistration,
        state: EvolutionState,
        epoch: int,
        readings: Sequence[RolloutReading],
        rewards: Mapping[str, float],
    ) -> MonitorEvidence:
        """Run checkpoint 6 for the ending epoch and return the trigger's evidence."""
        inputs, sources = epoch_inputs(project, registration, state, epoch, readings, rewards)
        report, reference = run_checkpoint(
            project,
            "epoch",
            f"{registration.run_id}/epoch-{epoch}",
            inputs,
            astra=self.astra,
            spec=self.spec,
            sources=sources,
            code_revision=self.code_revision,
            ledger=self.ledger,
        )
        flagged = [
            row.subject.removeprefix("epoch_tag:")
            for row in report.stats.subjects
            if "hack_drift" in row.flags
        ]
        stopped = tuple(
            row.subject.removeprefix("epoch_tag:")
            for row in report.stats.subjects
            if "stopped_discriminating" in row.flags
        )
        reviewed = report.review is not None and report.review.accepted
        if reviewed:
            confirmed = {
                target.removeprefix("epoch_tag:")
                for proposal in report.proposals
                if proposal.kind == "flag_drift"
                for target in proposal.targets
            }
            drifted = tuple(tag for tag in flagged if tag in confirmed)
            reason = f"Astra confirmed drift in {len(drifted)} of {len(flagged)} flagged hack tags"
        else:
            drifted = tuple(flagged)
            reason = (
                f"{len(flagged)} hack tags flagged by the statistics, unreviewed ({report.reason})"
            )
        return MonitorEvidence(
            report=reference,
            drifted=drifted,
            stopped_discriminating=stopped,
            halted=report.halted,
            reason=reason,
        )

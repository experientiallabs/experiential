"""Uncertainty and baselines for continuation experiments and campaigns, at the right unit.

Saved experiment and campaign reports hold measurements: planned, known and unknown branch
outcomes per arm and outcome-level Brier scores. This module derives what those measurements
support without changing them, so saved reports keep reproducing their own metrics.

Two kinds of uncertainty stay separate. Repeated continuations from one checkpoint estimate
that checkpoint's success under the frozen policy; their Wilson interval describes that
checkpoint only. Independent task lineages are the unit for anything about unseen tasks: repeats
are averaged within each cell, cells within each lineage, and the interval is taken at the
lineage count. Selected checkpoints, matched controls and critics forecasting the same task are
comparison arms of one lineage, not replications. Unknown outcomes stay bounds over every
planned branch. Critic forecasts are compared on the same realized outcomes with a base rate
taken from disjoint development outcomes (a campaign's initial source attempts), never from the
outcomes being forecast.
"""

from collections.abc import Sequence

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.evaluations.uncertainty import (
    missing_outcome_bounds,
    proportion_summary,
    unit_mean_summary,
)
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.campaign import ContinuationCampaign
from exp.optimize.evaluation.continuations.experiment import CheckpointExperiment
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import actor_budget_exhausted, known_outcomes
from exp.simulation.continuations import load_continuation_checkpoint


class BaseRate(ContractModel):
    """A development success rate used as the reference forecast for critics.

    Attributes:
        rate: Known successes over known outcomes.
        successes: Known successes.
        assessable: Known outcomes.
        unknown: Planned attempts whose outcome is unknown, retained for disclosure.
        source: Where the outcomes came from; never the outcomes being forecast.
    """

    rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    successes: int = Field(ge=0)
    assessable: int = Field(ge=1)
    unknown: int = Field(ge=0)
    source: str = Field(min_length=1)


class CellOutcomes(ContractModel):
    """Known outcomes of one experiment cell with its lineage and forecast.

    Attributes:
        arm: ``proposed`` or ``random``.
        pair: Matched pair index; both arms of a pair share it.
        checkpoint: Continued checkpoint artifact.
        lineage: Independent task lineage of the checkpoint.
        planned: Planned continuations, including unknown outcomes.
        outcomes: Known binary outcomes (1 success, 0 failure).
        actor_budget_exhausted: Planned continuations whose actor spent its output allowance
            without a complete tool call, reported beside the unknown count.
        forecast: Prospective critic probability recorded before outcomes, if any.
        failed: Whether the cell produced no study at all.
        critic_failure: Why the registered critic produced no forecast for this cell, if it
            failed; None when it forecast or when no critic was registered.
    """

    arm: str
    pair: int = Field(ge=0)
    checkpoint: str
    lineage: str | None
    planned: int = Field(ge=0)
    outcomes: tuple[int, ...]
    actor_budget_exhausted: int = Field(default=0, ge=0)
    forecast: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    failed: bool = False
    critic_failure: str | None = None


def experiment_cells(
    project: ProjectStore, report: CheckpointExperiment
) -> tuple[CellOutcomes, ...]:
    """Read every cell's known outcomes, forecast and task lineage from saved evidence.

    Args:
        project: Project owning the saved studies and checkpoints.
        report: Verified saved experiment report.

    Returns:
        One record per planned cell in registered order; a cell without a study keeps its
        planned branches as unknown.
    """
    cells = []
    for position, cell in enumerate(report.cells):
        checkpoint = load_continuation_checkpoint(project.artifacts, cell.job.study.checkpoint)
        study = None if cell.result is None else load_study(project.artifacts, cell.result)
        cells.append(
            CellOutcomes(
                arm=cell.arm,
                pair=position // 2,
                checkpoint=cell.job.study.checkpoint.artifact_id,
                lineage=checkpoint.task.lineage_group_id,
                planned=cell.job.study.repeats,
                outcomes=()
                if study is None
                else known_outcomes(row.assessment for row in study.branches),
                actor_budget_exhausted=0
                if study is None
                else actor_budget_exhausted(study.branches),
                forecast=cell.job.study.predicted_success
                if study is None
                else study.spec.predicted_success,
                failed=cell.result is None,
                critic_failure=cell.critic_failure,
            )
        )
    return tuple(cells)


def experiment_inference(
    project: ProjectStore, report: CheckpointExperiment, *, base_rate: BaseRate | None = None
) -> JsonObject:
    """Derive per-checkpoint and per-lineage uncertainty and baselines for one experiment.

    Args:
        project: Project owning the saved studies and checkpoints.
        report: Verified saved experiment report.
        base_rate: Disjoint development base rate for critic comparison, when available.

    Returns:
        See :func:`cell_inference`.
    """
    return cell_inference(experiment_cells(project, report), base_rate=base_rate)


def cell_inference(cells: Sequence[CellOutcomes], *, base_rate: BaseRate | None) -> JsonObject:
    """Separate repeated-attempt and lineage-level uncertainty and compare against baselines.

    Args:
        cells: Every planned cell, in registered order.
        base_rate: Disjoint development base rate for critic comparison, when available.

    Returns:
        Per arm: each checkpoint's repeated-attempt interval and missing-outcome bounds, the
        equal-weight lineage estimate with its Wilson interval at the lineage count, and
        outcome-level Brier scores beside the base-rate forecast on the same outcomes. Also
        the proposed-minus-random effect per lineage over matched pairs with known outcomes.
    """
    lineages = {cell.lineage for cell in cells if cell.lineage is not None}
    arms: JsonObject = {}
    for arm in ("proposed", "random"):
        group = [cell for cell in cells if cell.arm == arm]
        successes = sum(sum(cell.outcomes) for cell in group)
        known = sum(len(cell.outcomes) for cell in group)
        planned = sum(cell.planned for cell in group)
        exhausted = sum(cell.actor_budget_exhausted for cell in group)
        by_lineage: dict[str, list[float]] = {}
        for cell in group:
            if cell.outcomes and cell.lineage is not None:
                by_lineage.setdefault(cell.lineage, []).append(
                    sum(cell.outcomes) / len(cell.outcomes)
                )
        arms[arm] = {
            "checkpoints": [
                {
                    "checkpoint": cell.checkpoint,
                    "lineage": cell.lineage,
                    "planned": cell.planned,
                    "unknown": cell.planned - len(cell.outcomes),
                    "actor_budget_exhausted": cell.actor_budget_exhausted,
                    "repeated_attempts": proportion_summary(sum(cell.outcomes), len(cell.outcomes)),
                    "missing_outcome_bounds": missing_outcome_bounds(
                        sum(cell.outcomes), cell.planned - len(cell.outcomes), cell.planned
                    ),
                }
                for cell in group
            ],
            "missing_outcome_bounds": missing_outcome_bounds(successes, planned - known, planned),
            "actor_budget_exhausted": exhausted,
            "actor_budget_exhausted_share": exhausted / planned if planned else None,
            "across_lineages": unit_mean_summary(by_lineage),
            "critic": _critic(group, base_rate),
        }
    return {
        "independent_lineages": len(lineages),
        "arms": arms,
        "random_control_comparison": _paired(cells),
        "interpretation": _interpretation(len(lineages)),
    }


def campaign_inference(project: ProjectStore, report: ContinuationCampaign) -> JsonObject:
    """Derive uncertainty and baselines for a verified saved campaign.

    The initial source attempts give the development base rate every critic is compared
    against; their outcomes are disjoint from every continuation the critics forecast.

    Args:
        project: Project owning the saved source study, experiments and checkpoints.
        report: Verified saved campaign report.

    Returns:
        The source attempts' repeated-attempt interval and base rate, every completed
        experiment's inference, and the number of independent task lineages behind all of it.
    """
    source = load_study(project.artifacts, report.source)
    known = known_outcomes(row.assessment for row in source.branches)
    planned = len(source.branches)
    exhausted = actor_budget_exhausted(source.branches)
    lineage = load_continuation_checkpoint(
        project.artifacts, source.spec.checkpoint
    ).task.lineage_group_id
    base_rate = (
        BaseRate(
            rate=sum(known) / len(known),
            successes=sum(known),
            assessable=len(known),
            unknown=planned - len(known),
            source="initial source attempts from the campaign's starting checkpoint",
        )
        if known
        else None
    )
    experiments = []
    lineages = {lineage}
    for cell in report.cells:
        if cell.experiment is None:
            continue
        experiment = CheckpointExperiment.model_validate_json(
            project.artifacts.read_bytes(cell.experiment.artifact_id, "report.json")
        )
        cells = experiment_cells(project, experiment)
        lineages.update(row.lineage for row in cells if row.lineage is not None)
        experiments.append(
            {
                "selector": cell.selector,
                "critic": cell.critic,
                **cell_inference(cells, base_rate=base_rate),
            }
        )
    return {
        "independent_lineages": len(lineages),
        "source_attempts": {
            "lineage": lineage,
            "planned": planned,
            "unknown": planned - len(known),
            "actor_budget_exhausted": exhausted,
            "actor_budget_exhausted_share": exhausted / planned if planned else None,
            "repeated_attempts": proportion_summary(sum(known), len(known)),
            "missing_outcome_bounds": missing_outcome_bounds(
                sum(known), planned - len(known), planned
            ),
        },
        "base_rate": None if base_rate is None else base_rate.model_dump(mode="json"),
        "experiments": experiments,
        "interpretation": _interpretation(len(lineages))
        + (
            " Every selector/critic pair forecasts and continues the same source task; they are "
            "comparison arms, not replications."
        ),
    }


def experiment_summary(project: ProjectStore, report: CheckpointExperiment) -> JsonObject:
    """The saved experiment measurements with their derived inference, for presentation.

    Args:
        project: Project owning the saved evidence.
        report: Verified saved experiment report.

    Returns:
        Scope, unmatched count and saved metrics unchanged, plus ``inference``.
    """
    return {
        "scope": report.scope,
        "unmatched": report.unmatched,
        **report.metrics,
        "inference": experiment_inference(project, report),
    }


def campaign_summary(project: ProjectStore, report: ContinuationCampaign) -> JsonObject:
    """The saved campaign measurements with their derived inference, for presentation.

    Args:
        project: Project owning the saved evidence.
        report: Verified saved campaign report.

    Returns:
        Scope and saved metrics unchanged, plus ``inference``.
    """
    return {
        "scope": report.scope,
        **report.metrics,
        "inference": campaign_inference(project, report),
    }


def _critic(group: Sequence[CellOutcomes], base_rate: BaseRate | None) -> JsonObject:
    """Outcome-level Brier scores of forecasts beside the base-rate forecast on equal outcomes.

    Checkpoints where the critic failed are counted as critic failures, not dropped as if the
    critic had abstained; they have no forecast, so they carry no Brier score.
    """
    forecast = [cell for cell in group if cell.forecast is not None and cell.outcomes]
    failures = sum(cell.critic_failure is not None for cell in group)
    if not forecast:
        return {
            "forecast_checkpoints": 0,
            "critic_failures": failures,
            "status": "no forecast with known outcomes",
        }
    critic = [
        sum((cell.forecast - y) ** 2 for y in cell.outcomes) / len(cell.outcomes)
        for cell in forecast
        if cell.forecast is not None
    ]
    result: JsonObject = {
        "forecast_checkpoints": len(forecast),
        "critic_failures": failures,
        "mean_checkpoint_brier": sum(critic) / len(critic),
        "checkpoint_brier": critic,
    }
    if base_rate is None:
        result["base_rate_comparison"] = {
            "status": "no disjoint development base rate registered; no skill is reported"
        }
        return result
    baseline = [
        sum((base_rate.rate - y) ** 2 for y in cell.outcomes) / len(cell.outcomes)
        for cell in forecast
    ]
    mean_baseline = sum(baseline) / len(baseline)
    result["base_rate_comparison"] = {
        "base_rate": base_rate.rate,
        "source": base_rate.source,
        "mean_checkpoint_brier": mean_baseline,
        "brier_skill": None if mean_baseline == 0 else 1 - _mean(critic) / mean_baseline,
        "critic_better_checkpoints": sum(c < b for c, b in zip(critic, baseline, strict=True)),
        "scope": "same realized outcomes; skill above zero means lower Brier than the base rate",
    }
    return result


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean of a nonempty sequence."""
    return sum(values) / len(values)


def _paired(cells: Sequence[CellOutcomes]) -> JsonObject:
    """Proposed-minus-random success differences, averaged within then across lineages."""
    pairs: dict[int, dict[str, CellOutcomes]] = {}
    for cell in cells:
        pairs.setdefault(cell.pair, {})[cell.arm] = cell
    effects: dict[str, list[float]] = {}
    unknown_pairs = 0
    for pair in pairs.values():
        proposed, random = pair.get("proposed"), pair.get("random")
        if (
            proposed is None
            or random is None
            or not proposed.outcomes
            or not random.outcomes
            or proposed.lineage is None
        ):
            unknown_pairs += 1
            continue
        effects.setdefault(proposed.lineage, []).append(
            _mean(proposed.outcomes) - _mean(random.outcomes)
        )
    lineage_effects = {lineage: _mean(values) for lineage, values in sorted(effects.items())}
    return {
        "matched_pairs": len(pairs),
        "pairs_with_unknown_outcomes": unknown_pairs,
        "lineage_effects": lineage_effects,
        "mean_lineage_effect": _mean(list(lineage_effects.values())) if lineage_effects else None,
        "scope": "random matched controls are the selector baseline; effects are descriptive per "
        "lineage, without an interval, until several independent lineages are measured",
    }


def _interpretation(lineages: int) -> str:
    """State what the lineage count supports."""
    if lineages <= 1:
        return (
            "One independent task lineage: repeated-attempt intervals describe those checkpoints "
            "only, and the lineage interval says nothing reliable about unseen tasks."
        )
    return (
        f"{lineages} independent task lineages: lineage intervals average repeats within each "
        "cell first; population claims need lineages drawn from the declared task distribution."
    )

"""Policy replay and ablation-ladder assembly for routing scorecards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from wmo.optimize.routing.fit import route_scenarios
from wmo.optimize.routing.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.routing.scorecard_core import (
    DEFAULT_COMPLETION,
    Arm,
    CompletionRule,
    Ladder,
    LadderRung,
    _anchor_dominates,
    _by_scenario,
    _pairwise_scored,
    _require_comparable,
    build_scorecard,
)

if TYPE_CHECKING:
    from wmo.common.providers.base import Embedder
    from wmo.optimize.routing.policy import RoutingPolicy


def rows_for_policy(
    matrix: OutcomeMatrix,
    policy: RoutingPolicy,
    *,
    ids: Sequence[str] | None = None,
    embedder: Embedder | None = None,
) -> list[ScenarioOutcome]:
    """The rows a routing policy's own choices select: how a routed ladder rung is built.

    Replays the policy through `wmo.optimize.routing.route_scenarios`, the shared offline replay
    that `evaluate_policy` also scores through, and takes each scenario's rows for the model the
    policy chose. A rung is therefore a POLICY CONFIG evaluated offline against an existing
    matrix: no episode is rerun, and the routed arm is measured on exactly the rows the pool
    already produced.

    When the policy routes a scenario to a model the matrix never measured there, this emits an
    UNSCORED placeholder row rather than nothing. Dropping it would shrink the routed arm's
    scenario set invisibly, which is the same silent-narrowing bug `build_ladder`'s common set
    exists to prevent, and it would make the arm disagree with `evaluate_policy`, which counts
    that case as `unscored_scenarios`.

    Args:
        matrix: the precomputed pool x scenario grid.
        policy: the routing policy this rung represents.
        ids: which scenarios to route, defaulting to every scenario in the matrix.
        embedder: one embedder for the whole replay (an azure spec otherwise builds an HTTP
            client per call). Must be the function `policy.embedder` describes.

    Note:
        This is the one function in this module that can touch the network, since a non-static
        policy embeds its queries. Single-model arms via `rows_for_model` stay pure.

    Returns:
        The selected outcome rows, including unscored placeholders for unmeasured choices.
    """
    wanted = list(ids) if ids is not None else matrix.scenario_ids()
    decisions = route_scenarios(policy, matrix, wanted, embedder=embedder)

    rows: list[ScenarioOutcome] = []
    for scenario_id, decision in decisions.items():
        scenario_rows = matrix.for_scenario(scenario_id)
        chosen = [row for row in scenario_rows if row.model == decision.model]
        if chosen:
            rows.extend(chosen)
            continue
        rows.append(
            ScenarioOutcome(
                scenario_id=scenario_id,
                task=scenario_rows[0].task if scenario_rows else "",
                model=decision.model,
                reward=None,
                error=(
                    f"policy routed to '{decision.model}', which the matrix never measured on "
                    f"this scenario; the choice is unmeasured, not failed"
                ),
            )
        )
    return rows


def build_ladder(
    name: str,
    *,
    anchor: Arm,
    arms: Sequence[Arm],
    completion: CompletionRule = DEFAULT_COMPLETION,
) -> Ladder:
    """Score an ordered ablation ladder (distill-only, +routing, +compaction) against one anchor.

    Rung order is `arms` order and is meaningful: a ladder is read as a sequence of additions.

    Every rung is measured over ONE scenario set: those scored by the anchor and by every rung.
    A per-rung pairwise intersection with the anchor is not sufficient here, because `pareto()`
    compares rungs against each other; a rung whose episodes went unscored on the hard scenarios
    would otherwise be graded on an easier subset than the rung it is said to dominate. The
    scenarios that drop out are counted on the ladder, not discarded quietly.

    Raises:
        ValueError: when `arms` is empty, when two arms share a name or a condition label, when
            an arm collides with the anchor on either, or when no scenario was scored by the
            anchor and every rung. A label collision means two rungs are the same experiment
            under different display names, which is how ablation results get silently
            overwritten.

    Args:
        name: Stable name for the reported ablation sequence.
        anchor: Untuned or baseline arm every rung is compared against.
        arms: Ordered experimental arms to score as ladder rungs.
        completion: Rule used to count completed tasks for effective cost.

    Returns:
        An ablation ladder with scorecards measured on one common scenario set.
    """
    if not arms:
        raise ValueError(f"ladder '{name}' needs at least one arm besides the anchor")

    seen_names: dict[str, str] = {}
    seen_keys = {anchor.condition.key(): anchor.name}
    for arm in arms:
        if arm.name == anchor.name:
            raise ValueError(
                f"ladder '{name}': rung '{arm.name}' has the same name as the anchor; every "
                f"scorecard names the anchor it was measured against, so the rung and the "
                f"anchor need distinct names"
            )
        if arm.name in seen_names:
            raise ValueError(
                f"ladder '{name}': two rungs are both named '{arm.name}'; rung names are the "
                f"display handles for the ladder, so give each one a distinct name"
            )
        seen_names[arm.name] = arm.name
        key = arm.condition.key()
        if key in seen_keys:
            raise ValueError(
                f"ladder '{name}': rungs '{seen_keys[key]}' and '{arm.name}' carry the SAME "
                f"condition label ({key}), so they name the same experiment and one would "
                f"overwrite the other; a condition must encode every axis the rungs differ on "
                f"(use `optimizer`, `seed`, or `notes` to record what actually changed)"
            )
        seen_keys[key] = arm.name

    # The one scenario set the whole ladder is measured on. Ordered by the anchor's own rows so
    # the artifact is deterministic, and intersected rung by rung so a single arm that went
    # unscored somewhere narrows every rung equally rather than only its own.
    anchor_rows = _by_scenario(anchor.rows)
    universe = list(anchor_rows)
    common = set(universe)
    for arm in arms:
        _require_comparable(arm, anchor)
        arm_rows = _by_scenario(arm.rows)
        universe.extend(sid for sid in arm_rows if sid not in universe)
        common &= set(_pairwise_scored(arm_rows, anchor_rows, universe))
    if not common:
        raise ValueError(
            f"ladder '{name}': no scenario was scored by anchor '{anchor.name}' AND by every "
            f"rung ({', '.join(a.name for a in arms)}), so the rungs cannot be compared with "
            f"each other; every number on a ladder is measured on one common scenario set, so "
            f"rerun the unscored episodes or drop the rung that scored none of them"
        )
    ordered = [sid for sid in universe if sid in common]

    rungs: list[LadderRung] = []
    for index, arm in enumerate(arms):
        card = build_scorecard(arm=arm, anchor=anchor, completion=completion, restrict_to=ordered)
        rungs.append(
            LadderRung(
                index=index,
                scorecard=card,
                dial_position=arm.dial_position,
                dominated_by_anchor=_anchor_dominates(card, "p50"),
            )
        )
    return Ladder(
        name=name,
        anchor=anchor.name,
        anchor_condition=anchor.condition,
        scenarios_compared=len(ordered),
        scenarios_excluded=len(universe) - len(ordered),
        rungs=rungs,
    )

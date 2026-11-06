"""Checkpoint statistics 4 to 6: the point bar and blast radius, dense grading, RL epochs.

Arithmetic only, as in :mod:`~.tag_monitor_stats`:

- :func:`mining_stats` (checkpoint 4): each bar meta-rule's fire rate over every tagged step
  (``never_fires`` / ``always_fires``; a degenerate required rule halts), qualified points per tag,
  and whether one tag dominates the top of the blast-radius ranking.
- :func:`dense_stats` (checkpoint 5): for every dense tag, how often it is routed to the judge
  (``always_routed`` means Clef never clears it, so Clef is uninformative for it), how often Clef
  and the judge disagree on routed episodes (Clef fired but the judge found it absent, or Clef was
  clear but the judge established it), and the behavior-base tag rates per actor tier.
- :func:`epoch_stats` (checkpoint 6): every tag's rate per epoch from the rubric boundary's step
  readings, ``drift`` when it moved by ``drift_delta`` or more, ``hack_drift`` when a hack-family
  tag rose while mean reward rose by ``reward_rise`` or more (the early hack signal the divergence
  trigger receives), and ``stopped_discriminating`` when a tag whose rate sat inside the usable
  band left it.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonValue
from exp.optimize.evaluation.variance.tag_monitor_contracts import Halt, MonitorSpec, SubjectStats
from exp.optimize.evaluation.variance.tag_monitor_stats import halt, sample


def _round(value: float) -> float:
    """Round a statistic for a stable record."""
    return round(float(value), 6)


class BarRule(ContractModel):
    """One meta-rule of the qualification bar.

    Attributes:
        rule_id: The rule.
        required: Whether a point needs it.
    """

    rule_id: str
    required: bool = True


class MinedPoint(ContractModel):
    """One candidate decision point after the bar and the blast radius.

    Attributes:
        point_id: Its identity (trace and point).
        qualified: Whether the bar qualified it.
        tags: Tags held at the point (bar rules that held, error tags fired).
        rank: Its blast-radius rank, from 1, when ranked.
        score: Its blast radius, when scored.
    """

    point_id: str
    qualified: bool
    tags: tuple[str, ...] = ()
    rank: int | None = Field(default=None, ge=1)
    score: float | None = None


class MiningInputs(ContractModel):
    """What checkpoint 4 reads.

    Attributes:
        bar: The bar's meta-rules.
        steps: Each tagged step's answers to the bar rules.
        points: Every candidate point.
    """

    bar: tuple[BarRule, ...] = Field(min_length=1)
    steps: tuple[dict[str, float], ...] = ()
    points: tuple[MinedPoint, ...] = ()


def mining_stats(
    inputs: MiningInputs, spec: MonitorSpec
) -> tuple[list[SubjectStats], list[Halt], dict[str, JsonValue]]:
    """Checkpoint 4: bar-rule fire rates, qualified points by tag, blast-ranking dominance."""
    threshold = spec.health.threshold
    subjects: list[SubjectStats] = []
    halts: list[Halt] = []
    for rule in inputs.bar:
        answered = [s[rule.rule_id] for s in inputs.steps if rule.rule_id in s]
        rate = sum(v > threshold for v in answered) / len(answered) if answered else 0.0
        flags: tuple[str, ...] = ()
        if answered and rate == 0.0:
            flags = ("never_fires",)
        elif answered and rate >= spec.health.always_rate:
            flags = ("always_fires",)
        name = f"bar_rule:{rule.rule_id}"
        subjects.append(
            SubjectStats(
                subject=name,
                kind="bar_rule",
                values={
                    "fire_rate": _round(rate),
                    "answered": float(len(answered)),
                    "required": 1.0 if rule.required else 0.0,
                },
                flags=flags,
            )
        )
        if flags and rule.required:
            what = "never fires, so no point qualifies" if rate == 0.0 else "fires on every step"
            halts.append(
                halt(
                    "mining",
                    name,
                    "bar_rule_degenerate",
                    f"required meta-rule {rule.rule_id} {what} ({len(answered)} steps)",
                    "fire_rate",
                )
            )
    qualified = [p for p in inputs.points if p.qualified]
    ranked = sorted((p for p in inputs.points if p.rank is not None), key=lambda p: p.rank or 0)
    top = ranked[: spec.top_points]
    counts = Counter(tag for p in qualified for tag in set(p.tags))
    top_counts = Counter(tag for p in top for tag in set(p.tags))
    for tag in sorted(set(counts) | set(top_counts)):
        share = top_counts[tag] / len(top) if top else 0.0
        flags = ()
        if len(top) >= 2 and share >= spec.dominance_share:
            flags = ("dominates_ranking",)
        subjects.append(
            SubjectStats(
                subject=f"point_tag:{tag}",
                kind="point_tag",
                values={
                    "qualified_points": float(counts[tag]),
                    "qualified_share": _round(counts[tag] / len(qualified) if qualified else 0.0),
                    "top_share": _round(share),
                },
                flags=flags,
            )
        )
    evidence: dict[str, JsonValue] = {
        "evidence:blast-top": [
            {"point_id": p.point_id, "rank": p.rank, "score": p.score, "tags": list(p.tags)}
            for p in top
        ]
    }
    return subjects, halts, evidence


StepStatus = Literal["fired", "uncertain", "clear"]
"""Clef's strongest status for a tag over an episode's steps."""

JudgeDecision = Literal["established", "acceptable", "absent", "uncertain"]
"""The dense judge's decision on a routed tag."""


class DenseEpisode(ContractModel):
    """One graded episode as checkpoint 5 reads it.

    Attributes:
        episode_id: The episode.
        tier: The actor tier that produced it.
        clef: Clef's strongest status per dense tag over the episode's steps.
        routed: Tags routed to the judge.
        decisions: The judge's decision per routed tag it answered.
    """

    episode_id: str
    tier: str
    clef: dict[str, StepStatus] = Field(default_factory=dict)
    routed: tuple[str, ...] = ()
    decisions: dict[str, JudgeDecision] = Field(default_factory=dict)


class DenseInputs(ContractModel):
    """What checkpoint 5 reads.

    Attributes:
        episodes: Every graded episode.
        behavior_tags: The behavior-base tags whose per-tier rates are reported.
    """

    episodes: tuple[DenseEpisode, ...] = Field(min_length=1)
    behavior_tags: tuple[str, ...] = ()


def dense_stats(
    inputs: DenseInputs, spec: MonitorSpec
) -> tuple[list[SubjectStats], list[Halt], dict[str, JsonValue]]:
    """Checkpoint 5: routing rate, Clef/judge disagreement, behavior rates per tier."""
    episodes = inputs.episodes
    tags = sorted({t for e in episodes for t in (*e.clef, *e.routed)})
    tiers = sorted({e.tier for e in episodes})
    subjects: list[SubjectStats] = []
    halts: list[Halt] = []
    evidence: dict[str, JsonValue] = {}
    for tag in tags:
        seen = [e for e in episodes if tag in e.clef or tag in e.routed]
        routed = [e for e in seen if tag in e.routed]
        decided = [
            e for e in routed if e.decisions.get(tag) in ("established", "acceptable", "absent")
        ]
        yes_no = [e for e in decided if e.clef.get(tag) == "fired" and e.decisions[tag] == "absent"]
        no_yes = [
            e for e in decided if e.clef.get(tag) == "clear" and e.decisions[tag] == "established"
        ]
        rate = (len(yes_no) + len(no_yes)) / len(decided) if decided else 0.0
        values = {
            "episodes": float(len(seen)),
            "routed_rate": _round(len(routed) / len(seen) if seen else 0.0),
            "decided": float(len(decided)),
            "disagreement": _round(rate),
            "clef_yes_judge_absent": float(len(yes_no)),
            "clef_clear_judge_established": float(len(no_yes)),
        }
        flags: list[str] = []
        if len(seen) >= 2 and len(routed) == len(seen):
            flags.append("always_routed")
        if len(decided) >= spec.min_routed and rate >= spec.dense_disagreement:
            flags.append("clef_judge_disagree")
        if tag in inputs.behavior_tags:
            rates = []
            for tier in tiers:
                rows = [e for e in seen if e.tier == tier]
                fired = sum(e.clef.get(tag) == "fired" for e in rows) / len(rows) if rows else 0.0
                values[f"rate[{tier}]"] = _round(fired)
                rates.append(fired)
            if len(rates) >= 2 and max(rates) - min(rates) >= spec.drift_delta:
                flags.append("tier_gap")
        name = f"dense_tag:{tag}"
        subjects.append(
            SubjectStats(subject=name, kind="dense_tag", values=values, flags=tuple(flags))
        )
        if "clef_judge_disagree" in flags:
            evidence[f"evidence:{name}"] = [
                {
                    "episode_id": e.episode_id,
                    "tier": e.tier,
                    "clef": e.clef.get(tag),
                    "judge": e.decisions[tag],
                }
                for e in (yes_no + no_yes)[: max(1, spec.samples_per_flag)]
            ]
            if spec.dense_halt is not None and rate >= spec.dense_halt:
                halts.append(
                    halt(
                        "dense",
                        name,
                        "dense_disagreement",
                        f"Clef and the judge disagree on {tag} in {rate:.2f} of decided routed "
                        f"episodes (>= {spec.dense_halt})",
                        "disagreement",
                    )
                )
    for tier in tiers:
        rows = [e for e in episodes if e.tier == tier]
        for tag in inputs.behavior_tags:
            fired = sum(e.clef.get(tag) == "fired" for e in rows) / len(rows)
            subjects.append(
                SubjectStats(
                    subject=f"tier_tag:{tier}/{tag}",
                    kind="tier_tag",
                    values={"rate": _round(fired), "episodes": float(len(rows))},
                )
            )
    return subjects, halts, evidence


class EpochTags(ContractModel):
    """One epoch's step readings, reduced per rollout.

    Attributes:
        epoch: The epoch.
        mean_reward: Its mean in-loop reward.
        fired: Each rollout's confidently fired tags, by rollout.
        rewards: Each rollout's in-loop reward, by rollout.
    """

    epoch: int = Field(ge=1)
    mean_reward: float | None = None
    fired: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    rewards: dict[str, float] = Field(default_factory=dict)


class EpochInputs(ContractModel):
    """What checkpoint 6 reads.

    Attributes:
        epochs: Every closed epoch and the one ending, oldest first.
        tags: The tags of the frozen step bank.
        hack_tags: Tags of hack families (``*.negative``, ``error.*``) whose rise signals a hack.
    """

    epochs: tuple[EpochTags, ...] = Field(min_length=1)
    tags: tuple[str, ...] = ()
    hack_tags: tuple[str, ...] = ()


def epoch_stats(
    inputs: EpochInputs, spec: MonitorSpec
) -> tuple[list[SubjectStats], list[Halt], dict[str, JsonValue]]:
    """Checkpoint 6: tag-rate drift across epochs and questions that stop discriminating."""
    epochs = inputs.epochs
    current = epochs[-1]
    previous = epochs[-2] if len(epochs) > 1 else None
    reward_delta = None
    if (
        previous is not None
        and previous.mean_reward is not None
        and current.mean_reward is not None
    ):
        reward_delta = current.mean_reward - previous.mean_reward
    subjects: list[SubjectStats] = []
    halts: list[Halt] = []
    evidence: dict[str, JsonValue] = {}
    low, high = spec.health.min_fire_rate, spec.health.max_fire_rate
    for tag in inputs.tags:
        values: dict[str, float] = {}
        for epoch in epochs:
            count = len(epoch.fired)
            rate = sum(tag in fired for fired in epoch.fired.values()) / count if count else 0.0
            values[f"rate[epoch {epoch.epoch}]"] = _round(rate)
        flags: list[str] = []
        if previous is not None:
            now = values[f"rate[epoch {current.epoch}]"]
            before = values[f"rate[epoch {previous.epoch}]"]
            delta = now - before
            values["delta"] = _round(delta)
            if reward_delta is not None:
                values["reward_delta"] = _round(reward_delta)
            if abs(delta) >= spec.drift_delta:
                flags.append("drift")
            if (
                tag in inputs.hack_tags
                and delta >= spec.drift_delta
                and reward_delta is not None
                and reward_delta >= spec.reward_rise
            ):
                flags.append("hack_drift")
            if low <= before <= high and not low <= now <= high:
                flags.append("stopped_discriminating")
        name = f"epoch_tag:{tag}"
        subjects.append(
            SubjectStats(subject=name, kind="epoch_tag", values=values, flags=tuple(flags))
        )
        if "hack_drift" in flags:
            rows = [r for r, fired in current.fired.items() if tag in fired]
            evidence[f"evidence:{name}"] = [
                {"rollout_id": r, "reward": current.rewards.get(r)}
                for r in sample(rows, max(1, spec.samples_per_flag), name)
            ]
            if spec.halt_on_drift:
                halts.append(
                    halt(
                        "epoch",
                        name,
                        "hack_drift",
                        f"{tag} rose {values['delta']:+.2f} while mean reward rose "
                        f"{reward_delta:+.3f}",
                        "flag:hack_drift",
                    )
                )
    return subjects, halts, evidence

"""Reference combination of the outcome advantage with the dense reward, per actor token.

This is the executable form of the formula documented under "Dense error-mode reward" in
``docs/reference/decision-variance.md`` for schema ``dense-reward-v1``. An external GRPO trainer
may reimplement it; this pure function and its tests are the specification.

For a group of ``G`` episodes of one prompt (one start state), with outcome rewards ``R_i`` from
the frozen episode judge (pass 1, fail 0, None unscored):

1. Group-normalized outcome advantage, over scored episodes only (the existing
   :func:`~exp.optimize.evaluation.variance.rewards.group_advantages`)::

       A_i = (R_i - mean(R)) / std(R)      (population std; A_i = 0 when std = 0)

   An episode with an unscored outcome gets no advantage at all: every token is masked (None),
   and its dense findings are not trained on either.
2. Dense term on actor token ``t`` of episode ``i``, from every scored finding ``f`` whose span
   covers ``t``::

       D_i,t = sum_f r_f              (r_f from the export, already bounded by aggregation)
       d_i,t = clip(beta * D_i,t, -kappa, +kappa)

   Tokens no finding covers get ``d = 0``. The dense term is not group-normalized: it is sparse,
   and dividing it by a group statistic would inflate a single finding.

   Which tokens a finding covers depends on what was bound. A span is token-local only when it
   carries offsets into the actor's actual sampled tokens (``sampled``) under the same binding
   the trainer's rollout declares; then it covers exactly those sampled tokens. Otherwise
   (rendered offsets only, or another binding) it falls back to action-level credit: every
   sampled token of the whole action (message or tool call) it names. Rendered character offsets
   are never treated as token positions. :func:`token_rewards` applies this rule and
   :class:`EpisodeTokens` refuses token-local rewards without a declared sampled binding.
3. Per-token advantage::

       A_i,t = clip(A_i + d_i,t, -C, +C)

Defaults: ``beta = 0.5``, ``kappa = 0.5``, ``C = 5``. Because ``|d| <= kappa``, the dense term can
move a token's advantage by at most ``kappa`` either way, and credits are already capped below
every decisive magnitude, so positive dense credit cannot outweigh a decisive violation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.optimize.evaluation.variance.dense_reward import DENSE_REWARD_SCHEMA, DenseFinding
from exp.optimize.evaluation.variance.rewards import group_advantages


class CombinationSettings(ContractModel):
    """The trainer combination's constants.

    Attributes:
        schema_version: Dense reward schema the formula belongs to.
        dense_scale: ``beta``, the weight of the dense term.
        token_clip: ``kappa``, the bound on the dense term per token.
        advantage_clip: ``C``, the bound on each combined per-token advantage.
    """

    schema_version: Literal["dense-reward-v1"] = DENSE_REWARD_SCHEMA
    dense_scale: float = Field(default=0.5, ge=0.0, le=4.0)
    token_clip: float = Field(default=0.5, gt=0.0, le=4.0)
    advantage_clip: float = Field(default=5.0, gt=0.0, le=100.0)


DEFAULT_COMBINATION = CombinationSettings()
"""The documented default constants: beta 0.5, kappa 0.5, C 5."""


class TokenReward(ContractModel):
    """One scored finding's reward over a range of the episode's sampled actor tokens.

    Attributes:
        token_start: Inclusive token index.
        token_end: Exclusive token index.
        reward: The finding's bounded reward ``r_f``.
        granularity: ``sampled_tokens`` for token-local credit from bound sampled tokens,
            ``action`` for the whole action's tokens.
    """

    token_start: int = Field(ge=0)
    token_end: int = Field(ge=0)
    reward: float = Field(ge=-1.0, le=1.0)
    granularity: Literal["sampled_tokens", "action"] = "action"


class EpisodeTokens(ContractModel):
    """One episode's outcome reward, actor token count and dense token rewards.

    Attributes:
        outcome: The outcome reward, or None when unscored.
        tokens: Number of actor tokens the trainer optimizes in this episode.
        rewards: Dense rewards over token ranges of that stream.
        sampled_binding: Identity of the rollout's sampled-token record; required for any
            token-local reward.
    """

    outcome: float | None
    tokens: int = Field(ge=0)
    rewards: tuple[TokenReward, ...] = ()
    sampled_binding: str | None = None

    @model_validator(mode="after")
    def _inside(self) -> Self:
        """Require ranges inside the stream and a sampled binding for token-local credit."""
        for row in self.rewards:
            if row.granularity == "sampled_tokens" and self.sampled_binding is None:
                raise ValueError(
                    "token-local dense credit needs the rollout's bound sampled tokens; "
                    "use action-level credit when they are not bound"
                )
            if row.token_end < row.token_start or row.token_end > self.tokens:
                raise ValueError("a dense token range lies outside the episode's actor tokens")
        return self


def combine_advantages(
    group: Sequence[EpisodeTokens], settings: CombinationSettings = DEFAULT_COMBINATION
) -> tuple[tuple[float, ...] | None, ...]:
    """Return each episode's per-token advantage, or None for an unscored outcome.

    Args:
        group: Every episode of one GRPO group.
        settings: The formula's constants.

    Returns:
        Per episode, one advantage per actor token, or None when the episode is masked.
    """
    outcome = group_advantages("dense-combination", [row.outcome for row in group]).advantages
    combined: list[tuple[float, ...] | None] = []
    for episode, advantage in zip(group, outcome, strict=True):
        if advantage is None:
            combined.append(None)
            continue
        dense = [0.0] * episode.tokens
        for row in episode.rewards:
            for token in range(row.token_start, row.token_end):
                dense[token] += row.reward
        kappa, ceiling = settings.token_clip, settings.advantage_clip
        combined.append(
            tuple(
                max(
                    -ceiling,
                    min(ceiling, advantage + max(-kappa, min(kappa, settings.dense_scale * d))),
                )
                for d in dense
            )
        )
    return tuple(combined)


PartKey = tuple[int, str, int | None]
"""A trajectory part: (turn index, ``content`` or ``tool_call``, tool-call index or None)."""


def token_rewards(
    findings: Sequence[DenseFinding],
    action_ranges: Mapping[PartKey, tuple[int, int]],
    *,
    sampled_binding: str | None,
) -> tuple[TokenReward, ...]:
    """Place scored findings on the trainer's sampled actor token stream.

    Args:
        findings: An export's findings.
        action_ranges: The [start, end) range of each action part's sampled tokens in the
            trainer's stream, keyed by (turn index, part, tool-call index).
        sampled_binding: The rollout's sampled-token record identity, or None when the trainer
            cannot bind its sampled tokens to the export.

    Returns:
        Per rewarded finding and action: a token-local reward when the span's sampled offsets
        share ``sampled_binding`` and fit the action, otherwise one action-level reward over the
        whole action.

    Raises:
        ValueError: A rewarded span names an action the trainer did not place.
    """
    rewards: list[TokenReward] = []
    for finding in findings:
        if not finding.reward:
            continue
        whole: set[PartKey] = set()
        for span in finding.spans:
            key: PartKey = (span.turn_index, span.part, span.tool_call_index)
            if key not in action_ranges:
                raise ValueError(f"the trainer did not place trajectory action {key}")
            start, end = action_ranges[key]
            bound = span.sampled
            if (
                bound is not None
                and sampled_binding is not None
                and bound.binding == sampled_binding
                and bound.token_end <= end - start
            ):
                rewards.append(
                    TokenReward(
                        token_start=start + bound.token_start,
                        token_end=start + bound.token_end,
                        reward=finding.reward,
                        granularity="sampled_tokens",
                    )
                )
            elif key not in whole:
                whole.add(key)
                rewards.append(TokenReward(token_start=start, token_end=end, reward=finding.reward))
    return tuple(rewards)

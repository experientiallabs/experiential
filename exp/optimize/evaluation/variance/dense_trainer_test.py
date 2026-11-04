"""The reference trainer combination: outcome advantage plus a bounded dense term per token."""

from __future__ import annotations

import pytest

from exp.optimize.evaluation.variance.dense_reward import DenseFinding
from exp.optimize.evaluation.variance.dense_trainer import (
    CombinationSettings,
    EpisodeTokens,
    PartKey,
    TokenReward,
    combine_advantages,
    token_rewards,
)

SHA = "e" * 64


def test_dense_terms_shift_only_cited_tokens_within_the_token_clip() -> None:
    """A_i,t = clip(A_i + clip(beta * D_i,t, -kappa, kappa), -C, C) on cited tokens only."""
    group = (
        EpisodeTokens(
            outcome=1.0,
            tokens=4,
            rewards=(TokenReward(token_start=1, token_end=3, reward=-1.0),),
        ),
        EpisodeTokens(
            outcome=0.0,
            tokens=3,
            rewards=(
                TokenReward(token_start=0, token_end=2, reward=0.05),
                TokenReward(token_start=1, token_end=2, reward=0.05),
            ),
        ),
        EpisodeTokens(outcome=None, tokens=2),
    )
    settings = CombinationSettings(dense_scale=0.5, token_clip=0.3, advantage_clip=5.0)

    first, second, masked = combine_advantages(group, settings)

    assert first == pytest.approx((1.0, 0.7, 0.7, 1.0))
    assert second == pytest.approx((-0.975, -0.95, -1.0))
    assert masked is None


def test_identical_outcomes_leave_only_the_bounded_dense_term() -> None:
    """With no outcome spread the group advantage is zero; clipping still bounds every token."""
    group = tuple(
        EpisodeTokens(
            outcome=1.0, tokens=2, rewards=(TokenReward(token_start=0, token_end=1, reward=-1.0),)
        )
        for _ in range(3)
    )
    settings = CombinationSettings(dense_scale=4.0, token_clip=0.5, advantage_clip=0.25)
    assert combine_advantages(group, settings) == ((-0.25, 0.0),) * 3


def _finding(reward: float, *spans: dict) -> DenseFinding:
    """A scored finding with spans."""
    return DenseFinding.model_validate(
        {
            "finding_id": "f",
            "tag_id": "modified_evaluation",
            "decision": "established",
            "rationale": "r",
            "scored": True,
            "reward": reward,
            "spans": [
                {
                    "event_id": "e1",
                    "cited_event_id": "e1",
                    "turn_index": 0,
                    "part": "tool_call",
                    "tool_call_index": 0,
                    "char_start": 2,
                    "char_end": 9,
                    "text_sha256": SHA,
                    **span,
                }
                for span in spans
            ],
        }
    )


def test_token_local_credit_needs_the_rollouts_bound_sampled_tokens() -> None:
    """Sampled offsets under the rollout's binding are token-local; the rest is action-level."""
    sampled = {
        "sampled": {"binding": "rollout:1", "token_start": 1, "token_end": 3},
        "granularity": "sampled_tokens",
    }
    findings = (_finding(-1.0, sampled), _finding(0.05, {}, {}), _finding(0.0, {}))
    ranges: dict[PartKey, tuple[int, int]] = {(0, "tool_call", 0): (10, 20)}

    bound = token_rewards(findings, ranges, sampled_binding="rollout:1")
    unbound = token_rewards(findings, ranges, sampled_binding=None)

    assert [(r.token_start, r.token_end, r.granularity) for r in bound] == [
        (11, 13, "sampled_tokens"),
        (10, 20, "action"),
    ]
    assert [(r.token_start, r.token_end, r.granularity) for r in unbound] == [
        (10, 20, "action"),
        (10, 20, "action"),
    ]
    with pytest.raises(ValueError, match="bound sampled tokens"):
        EpisodeTokens(outcome=1.0, tokens=30, rewards=bound)
    assert EpisodeTokens(outcome=1.0, tokens=30, rewards=bound, sampled_binding="rollout:1")
    with pytest.raises(ValueError, match="did not place"):
        token_rewards(findings, {}, sampled_binding=None)
    with pytest.raises(ValueError, match="outside"):
        EpisodeTokens(outcome=1.0, tokens=5, rewards=unbound)

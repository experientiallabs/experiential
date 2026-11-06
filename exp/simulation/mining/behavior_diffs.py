"""Group evidence-backed behavioral differences between matched task rollouts."""

from itertools import combinations

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, stable_id


class BehaviorStep(ContractModel):
    """One labeled decision boundary with exact evidence identity.

    Attributes:
        step_id: Immutable locator of the action or completion boundary.
        tags: Confident positive question identities; uncertain labels stay upstream.
        observed_error: Whether native evidence records an error at this boundary.
    """

    step_id: str = Field(min_length=1)
    tags: frozenset[str]
    observed_error: bool = False


class MatchedBehavior(ContractModel):
    """One rollout whose matching scope is independent of the actor model.

    Attributes:
        rollout_id: Immutable rollout identity.
        task_id: Exact public task, never a topic or instruction-text approximation.
        base_id: Frozen environment base identity.
        checkpoint_sha256: Exact shared initial actor/world boundary.
        policy_sha256: Model-independent horizon, prompt and sampling settings.
        reward: Observed reward; None denotes an unavailable outcome.
        goal_verified: Independent observed goal check, when available.
        steps: Tagged decision boundaries with their evidence locators.
    """

    rollout_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    base_id: str = Field(min_length=1)
    checkpoint_sha256: str = Field(min_length=1)
    policy_sha256: str = Field(min_length=1)
    reward: float | None = Field(default=None, strict=True, allow_inf_nan=False)
    goal_verified: bool | None = Field(default=None, strict=True)
    steps: tuple[BehaviorStep, ...]

    @model_validator(mode="after")
    def unique_steps(self) -> "MatchedBehavior":
        """Refuse duplicate evidence boundaries within a rollout."""
        if len({step.step_id for step in self.steps}) != len(self.steps):
            raise ValueError("behavior step identities must be unique within a rollout")
        return self

    def scope(self) -> tuple[str, str, str, str]:
        """Return the exact conditions required before two rollouts can be paired."""
        return self.task_id, self.base_id, self.checkpoint_sha256, self.policy_sha256


class BehaviorDifference(ContractModel):
    """An oriented symbolic difference with evidence for every added or removed tag.

    Attributes:
        left: Reference rollout, independently goal-verified when only one is.
        right: Contrasting rollout.
        task_id: Shared exact task identity.
        same_reward: Whether both known rewards are equal.
        added: Tags present only on the right, mapped to their exact step evidence.
        removed: Tags present only on the left, mapped to their exact step evidence.
        left_errors: All observed error boundaries on the left.
        right_errors: All observed error boundaries on the right.
    """

    left: str
    right: str
    task_id: str
    same_reward: bool
    added: dict[str, tuple[str, ...]]
    removed: dict[str, tuple[str, ...]]
    left_errors: tuple[str, ...]
    right_errors: tuple[str, ...]


class BehaviorDifferenceGroup(ContractModel):
    """One repeated symbolic pattern, potentially shared across different tasks.

    Attributes:
        pattern_id: Content identity of the ordered added/removed tag sets.
        added: Right-only tag identities.
        removed: Left-only tag identities.
        examples: Matched within-task pairs supporting this pattern.
    """

    pattern_id: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    examples: tuple[BehaviorDifference, ...]


def _evidence(run: MatchedBehavior) -> dict[str, tuple[str, ...]]:
    """Collect the exact boundaries supporting every observed positive tag."""
    tags = sorted({tag for step in run.steps for tag in step.tags})
    return {tag: tuple(step.step_id for step in run.steps if tag in step.tags) for tag in tags}


def mine_behavior_differences(
    rollouts: tuple[MatchedBehavior, ...], *, same_reward_only: bool = True
) -> tuple[BehaviorDifferenceGroup, ...]:
    """Pair only identical task conditions, then group repeated added/removed patterns.

    Equal-reward pairs are deliberately retained. Unknown rewards never become equality
    evidence. Groups may span tasks, while each example must match its own task, base,
    starting checkpoint and model-independent policy exactly.
    """
    if len({run.rollout_id for run in rollouts}) != len(rollouts):
        raise ValueError("rollout identities must be unique")
    groups: dict[tuple[tuple[str, ...], tuple[str, ...]], list[BehaviorDifference]] = {}
    for left, right in combinations(sorted(rollouts, key=lambda run: run.rollout_id), 2):
        if left.scope() != right.scope() or left.reward is None or right.reward is None:
            continue
        same = left.reward == right.reward
        if same_reward_only and not same:
            continue
        if right.goal_verified is True and left.goal_verified is not True:
            left, right = right, left
        left_tags, right_tags = _evidence(left), _evidence(right)
        added = tuple(sorted(right_tags.keys() - left_tags.keys()))
        removed = tuple(sorted(left_tags.keys() - right_tags.keys()))
        if not added and not removed:
            continue
        difference = BehaviorDifference(
            left=left.rollout_id,
            right=right.rollout_id,
            task_id=left.task_id,
            same_reward=same,
            added={tag: right_tags[tag] for tag in added},
            removed={tag: left_tags[tag] for tag in removed},
            left_errors=tuple(step.step_id for step in left.steps if step.observed_error),
            right_errors=tuple(step.step_id for step in right.steps if step.observed_error),
        )
        groups.setdefault((added, removed), []).append(difference)
    return tuple(
        BehaviorDifferenceGroup(
            pattern_id=stable_id("behavior-difference", {"added": added, "removed": removed}),
            added=added,
            removed=removed,
            examples=tuple(examples),
        )
        for (added, removed), examples in sorted(groups.items())
    )

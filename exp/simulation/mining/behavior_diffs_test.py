"""Symbolic mining retains equal rewards and enforces exact task/environment scope."""

import pytest

from exp.simulation.mining.behavior_diffs import (
    BehaviorStep,
    MatchedBehavior,
    mine_behavior_differences,
)


def _run(
    identity: str, tags: frozenset[str], **changes: str | float | bool | None
) -> MatchedBehavior:
    """Create one explicitly matched native rollout."""
    data = {
        "rollout_id": identity,
        "task_id": "task",
        "base_id": "base",
        "checkpoint_sha256": "checkpoint",
        "policy_sha256": "policy",
        "reward": 1.0,
        "goal_verified": False,
        "steps": [{"step_id": identity + "/step/1", "tags": tags, "observed_error": True}],
    }
    data.update(changes)
    return MatchedBehavior.model_validate(data)


def test_equal_reward_shortcut_difference_keeps_step_evidence() -> None:
    """A genuine result and a shortcut remain mineable even when rewards are identical."""
    good = _run(
        "good", frozenset({"changed_implementation", "verified_result"}), goal_verified=True
    )
    bad = _run("bad", frozenset({"modified_evaluation", "unsupported_success_claim"}))
    groups = mine_behavior_differences((bad, good))
    assert len(groups) == 1
    example = groups[0].examples[0]
    assert example.left == "good" and example.right == "bad" and example.same_reward
    assert example.added["modified_evaluation"] == ("bad/step/1",)
    assert example.removed["verified_result"] == ("good/step/1",)
    assert example.right_errors == ("bad/step/1",)


@pytest.mark.parametrize("field", ["task_id", "base_id", "checkpoint_sha256", "policy_sha256"])
def test_similar_tasks_or_changed_conditions_never_pair(field: str) -> None:
    """Topic similarity cannot join different tasks, bases or execution policies."""
    left = _run("left", frozenset({"verified_result"}))
    right = _run("right", frozenset({"modified_evaluation"}), **{field: "different"})
    assert not mine_behavior_differences((left, right))


def test_unknown_reward_is_not_equal_reward_evidence() -> None:
    """Two unavailable outcomes cannot create a same-reward shortcut claim."""
    left = _run("left", frozenset({"verified_result"}), reward=None)
    right = _run("right", frozenset({"modified_evaluation"}), reward=None)
    assert not mine_behavior_differences((left, right))


def test_same_patterns_group_across_independently_matched_tasks() -> None:
    """Grouping generalizes patterns without pairing unrelated tasks."""
    runs = (
        _run("a-good", frozenset({"verified_result"}), goal_verified=True),
        _run("a-bad", frozenset({"modified_evaluation"})),
        _run("b-good", frozenset({"verified_result"}), task_id="task-b", goal_verified=True),
        _run("b-bad", frozenset({"modified_evaluation"}), task_id="task-b"),
    )
    groups = mine_behavior_differences(runs)
    assert len(groups) == 1 and len(groups[0].examples) == 2
    assert {x.task_id for x in groups[0].examples} == {"task", "task-b"}


def test_different_rewards_are_an_explicit_additional_mining_mode() -> None:
    """Failure contrasts can be requested without excluding equal-reward pairs by default."""
    left = _run("left", frozenset({"verified_result"}), reward=1.0, goal_verified=True)
    right = _run("right", frozenset({"observed_failure"}), reward=0.0)
    assert not mine_behavior_differences((left, right))
    assert len(mine_behavior_differences((left, right), same_reward_only=False)) == 1


def test_duplicate_evidence_and_duplicate_runs_are_refused() -> None:
    """Repeated records cannot inflate support counts."""
    run = _run("one", frozenset({"verified_result"}))
    with pytest.raises(ValueError, match="rollout identities"):
        mine_behavior_differences((run, run))
    with pytest.raises(ValueError, match="step identities"):
        MatchedBehavior.model_validate(
            {
                **run.model_dump(),
                "steps": [BehaviorStep(step_id="same", tags=frozenset())] * 2,
            }
        )


def test_unresolved_answer_never_becomes_a_symbolic_removal() -> None:
    """A confident positive versus an unknown is not evidence of a behavioral difference."""
    left = _run("left", frozenset({"verified_result"}), goal_verified=True)
    right = _run("right", frozenset())
    right = right.model_copy(
        update={
            "steps": (
                BehaviorStep(
                    step_id="right/step/1",
                    tags=frozenset(),
                    unknown_tags=frozenset({"verified_result"}),
                ),
            )
        }
    )
    assert not mine_behavior_differences((left, right))

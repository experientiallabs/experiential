"""Match archived native observations without calling or correcting an environment."""

from collections.abc import Sequence

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, strict_equal
from exp.common.evaluations.reconstruction import WorkflowStep
from exp.common.tasks import TaskCase


class RecordedStep(ContractModel):
    """One archived native action and its observation.

    Attributes:
        action: Exact public tool name and arguments.
        reply: Native observation, available only to the post-run analyst.
    """

    action: JsonObject
    reply: JsonValue


class RecordedEpisode(ContractModel):
    """An immutable archive slice anchored to its own public initialization.

    Attributes:
        episode_id: Source artifact identity, checked by the loading owner.
        task_lineage: Customer or task-group boundary.
        initial_context: Exact public context before the first recorded action.
        steps: Chronological native actions and replies, never delivered to the actor.
    """

    episode_id: str = Field(min_length=1)
    task_lineage: str = Field(min_length=1)
    initial_context: JsonObject
    steps: tuple[RecordedStep, ...] = Field(min_length=1)


def matched_native_replies(
    task: TaskCase,
    steps: Sequence[WorkflowStep],
    episodes: Sequence[RecordedEpisode],
    *,
    cell_id: str,
) -> list[JsonObject]:
    """Expose a recorded reply only when lineage, initializer and entire history match.

    Args:
        task: Original public task, without subsequently injected actor instructions.
        steps: Actual carried simulator observations, including failed slots.
        episodes: Verified archived native episodes; no environment implementation.
        cell_id: Actor run identity for evidence locators.

    Returns:
        One applicability record per actual action. Unmatched replies stay undisclosed.
    """
    result: list[JsonObject] = []
    history: list[JsonObject] = []
    for step in steps:
        candidates: list[tuple[RecordedEpisode, int, RecordedStep]] = []
        action_seen = False
        for episode in episodes:
            if episode.task_lineage != task.lineage_group_id:
                continue
            for index, native in enumerate(episode.steps):
                if not strict_equal(native.action, step.action):
                    continue
                action_seen = True
                prior = [row.model_dump(mode="json") for row in episode.steps[:index]]
                if strict_equal(episode.initial_context, task.initial_context) and strict_equal(
                    prior, history
                ):
                    candidates.append((episode, index, native))
        row: JsonObject = {
            "ref": f"counterfactual:{cell_id}:step:{step.step}",
            "actor_step": step.step,
            "action": step.action,
            "applicable": False,
            "scope": "local reply substitution only; no alternate actor continuation executed",
        }
        if candidates and all(
            strict_equal(candidates[0][2].reply, candidate[2].reply) for candidate in candidates
        ):
            episode, index, native = candidates[0]
            row.update(
                {
                    "applicable": True,
                    "native_reply": native.reply,
                    "candidate_reply": step.reply,
                    "reply_agrees": step.status == "valid"
                    and strict_equal(step.reply, native.reply),
                    "native_source": episode.episode_id,
                    "native_step": index,
                    "match_basis": (
                        "exact task lineage, public initialization "
                        "and complete prior action/reply prefix"
                    ),
                }
            )
        else:
            row["reason"] = (
                "conflicting archived replies"
                if candidates
                else "action matched but initialization or prior history differs"
                if action_seen
                else "no recorded action for this lineage"
            )
        result.append(row)
        history.append({"action": step.action, "reply": step.reply})
    return result

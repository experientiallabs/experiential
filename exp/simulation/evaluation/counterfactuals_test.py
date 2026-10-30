"""Archived replies are local counterfactuals only under matched observable histories."""

from exp.common.evaluations.reconstruction import WorkflowStep
from exp.common.tasks import TaskCase
from exp.simulation.evaluation.counterfactuals import RecordedEpisode, matched_native_replies


def test_action_match_after_divergent_history_does_not_reveal_native_reply() -> None:
    """Identical arguments cannot recover native truth from a different carried world."""
    task = TaskCase(
        task_id="t",
        lineage_group_id="a",
        partition="held_out",
        instruction="Read",
        initial_context={"public_prefix": []},
        tools=(),
        workload_weight=1,
        source_trace_ids=("t",),
    )
    source = RecordedEpisode.model_validate(
        {
            "episode_id": "native-a",
            "task_lineage": "a",
            "initial_context": {"public_prefix": []},
            "steps": [
                {"action": {"name": "edit", "arguments": {}}, "reply": "correct"},
                {"action": {"name": "read", "arguments": {}}, "reply": "SECRET_NATIVE"},
            ],
        }
    )
    steps = (
        WorkflowStep(
            step=0,
            action={"name": "edit", "arguments": {}},
            status="valid",
            reply="wrong",
            before={},
            after={},
        ),
        WorkflowStep(
            step=1,
            action={"name": "read", "arguments": {}},
            status="valid",
            reply="wrong",
            before={},
            after={},
        ),
    )
    result = matched_native_replies(task, steps, (source,), cell_id="c")
    assert result[0]["applicable"] is True
    assert result[0]["native_reply"] == "correct"
    assert result[1]["applicable"] is False
    assert "native_reply" not in result[1]
    assert "SECRET_NATIVE" not in str(result)

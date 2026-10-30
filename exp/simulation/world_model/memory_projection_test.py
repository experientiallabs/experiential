"""Research ablation preserves facts, expands mixed summaries and refuses ambiguous authority."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ToolCall
from exp.optimize.evaluation.tests.fidelity_memory_test import memory_project
from exp.simulation.retrieval.selection import AllOriginals
from exp.simulation.world_model.experience_memory import memory_head, publish_trajectory
from exp.simulation.world_model.frozen_memory import TaskView
from exp.simulation.world_model.memory_projection import (
    observation_overview,
    project_research_request,
)
from exp.simulation.world_model.memory_tree import build_pending, overview


def test_research_projection_preserves_nonresearch_and_does_not_modify_original() -> None:
    """Constructor target/library and all session hypotheses are removed through one boundary."""
    body = {
        "environment_state": {"ticket": "closed"},
        "visible_conversation": ["close ticket"],
        "behavioral_knowledge": {"rule": "must have ID"},
        "library_originals": ["original"],
        "memory_observations": ["historical"],
        "research_history": {
            "target": ["target secret"],
            "prior_library": ["library secret"],
            "session": ["session secret"],
            "prior_trajectories": ["prior secret"],
            "memory_overview": [{"kind": "observed", "text": "recorded"}],
        },
    }
    request = ModelRequest(
        messages=(
            ModelMessage(role="system", content="instructions"),
            ModelMessage(role="user", content=json.dumps(body)),
        )
    )
    projected = project_research_request(request)
    result = json.loads(projected.messages[1].content or "")
    assert result.pop("research_history") == {
        "memory_overview": [{"kind": "observed", "text": "recorded"}]
    }
    assert result == {key: value for key, value in body.items() if key != "research_history"}
    assert json.loads(request.messages[1].content or "") == body
    assert projected.messages[0] == request.messages[0]
    assert "secret" not in projected.model_dump_json()


def test_mixed_overview_opens_only_eligible_observation_leaves(tmp_path: Path) -> None:
    """Summary removal cannot erase historical evidence or reintroduce retired sources."""
    project, base = memory_project(tmp_path)
    now = datetime.now(UTC)
    version, _ = publish_trajectory(
        project.artifacts,
        lineage="bank",
        parent=None,
        world_model=base.world_model,
        selection_policy_sha256="0" * 64,
        trajectory={"task_lineage": "source", "actions": [], "observed_replies": []},
        research=(),
        lines=(("observed", "real outcome", "source"), ("hypothesis", "speculation", "rule")),
        created_at=now,
        code_revision="fixture",
        publisher="source",
    )

    def complete(body: JsonObject, logical_id: str) -> JsonObject:
        """Produce a deliberately blended model summary, not new original evidence."""
        return {"summary": "real outcome implies speculation"}

    build_pending(
        project.artifacts,
        "bank",
        version.log_length or 0,
        complete,
        limit=1,
        created_at=now,
        code_revision="fixture",
    )
    rows = tuple(overview(project.artifacts, "bank", version.log_length or 0, 1))
    assert rows[0]["kind"] == "summary"
    before = project.artifacts.list_ids()
    projected = observation_overview(project.artifacts, "bank", rows, admits=lambda line: True)
    assert len(projected) == 1 and projected[0]["text"] == "real outcome"
    assert observation_overview(project.artifacts, "bank", rows, admits=lambda line: False) == ()
    assert (
        project.artifacts.list_ids() == before and memory_head(project.artifacts, "bank") == version
    )
    records: dict[str, JsonObject] = {
        "o": {"kind": "original-observation", "reply": "native"},
        "h": {"kind": "investigation", "hypothesis": "speculation"},
    }
    full = TaskView(records=records, overview=rows, selection=AllOriginals())
    off = TaskView(
        records=records, overview=projected, selection=AllOriginals(), research_history=False
    )
    action = AssistantAction(tool_calls=(ToolCall(call_id="c", name="read", arguments={}),))
    assert full.identity != off.identity
    assert off.context((), action, {}).observations == full.context((), action, {}).observations
    assert not off.context((), action, {}).prior_trajectories


def test_unopened_summary_fails_closed() -> None:
    """A caller cannot silently retain a summary that may still contain withheld research."""
    request = ModelRequest(
        messages=(
            ModelMessage(
                role="user",
                content=json.dumps(
                    {
                        "environment_state": {},
                        "research_history": {
                            "memory_overview": [{"kind": "summary", "text": "blend"}]
                        },
                    }
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="mixed overview"):
        project_research_request(request)

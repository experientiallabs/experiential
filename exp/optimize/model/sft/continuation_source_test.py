"""Real workbook continuations enter the existing SFT builder with prefix-only context."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ModelMessage
from exp.common.project import ArtifactStore, ProjectStore
from exp.optimize.evaluation.continuations import admission_test as admission_fixtures
from exp.optimize.evaluation.continuations.admission_test import admitted_workbook_study
from exp.optimize.model.sft.builder import (
    SFTBuildError,
    build_sft_dataset,
    load_sft_dataset,
    load_verified_sft_dataset,
    write_sft_dataset,
)
from exp.optimize.model.sft.continuation_source import _tool_results, resolve_continuation_source
from exp.optimize.model.sft.contracts import AssistantActionEvent, SFTBuildSpec, ToolEvent
from exp.optimize.model.sft.sources import SFTSourceVerificationError
from exp.optimize.model.sft.tinker import tinker_messages_from_example
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.simulation.continuations import ContinuationCheckpoint


def test_verified_export_keeps_failed_denominator_and_only_trains_continuation(
    tmp_path: Path,
) -> None:
    """Rebuilt data preserves tools and prefix mistakes without making them targets."""
    project = ProjectStore(tmp_path, "case")
    acceptance = admitted_workbook_study(project)
    dataset = build_sft_dataset(
        store=project,
        production_sources=(),
        teacher_sources=(),
        continuation_sources=(acceptance,),
        spec=SFTBuildSpec(held_out_fraction=0),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    assert len(dataset.sources) == 2 and sum(row.accepted for row in dataset.sources) == 1
    assert len(dataset.rows) == 1
    assert any(row.reason == "unapproved_action" for row in dataset.inspection.exclusions)
    assert all(row.partition == "train" for row in dataset.rows)
    assert len({row.example.leakage_group_id for row in dataset.rows}) == 1
    for row in dataset.rows:
        example = row.example
        assert example.target.content != "Earlier mistaken hypothesis."
        assert example.tools and example.tools[0].name == "edit_workbook"
        messages = tinker_messages_from_example(example)
        assert not any(message["content"].startswith("Task:\n") for message in messages)
        assert messages[-1]["role"] == "assistant"
    assert all(row.example.target.content is None for row in dataset.rows)
    assert any("assistant prose" in row.detail for row in dataset.inspection.exclusions)
    write_sft_dataset(project, dataset)
    assert dataset.dataset.status == "diagnostic"
    assert load_sft_dataset(project, dataset.dataset.dataset_id, require_accepted=False) == dataset
    with pytest.raises(SFTBuildError, match="diagnostic"):
        load_verified_sft_dataset(project, dataset.dataset.dataset_id)


def test_all_refused_study_still_rebuilds_its_exclusions(tmp_path: Path) -> None:
    """An empty accepted set cannot erase its failed study from the artifact chain."""
    project = ProjectStore(tmp_path, "case")
    acceptance = admitted_workbook_study(project, registered=False)
    dataset = build_sft_dataset(
        store=project,
        production_sources=(),
        teacher_sources=(),
        continuation_sources=(acceptance,),
        spec=SFTBuildSpec(held_out_fraction=0),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    assert not dataset.rows and len(dataset.sources) == len(dataset.inspection.exclusions) == 2
    write_sft_dataset(project, dataset)
    assert load_sft_dataset(project, dataset.dataset.dataset_id, require_accepted=False) == dataset
    with pytest.raises(SFTBuildError, match="insufficient"):
        load_verified_sft_dataset(project, dataset.dataset.dataset_id)


def test_narrated_tool_action_is_not_silently_rewritten_into_a_target(tmp_path: Path) -> None:
    """A correct edit with an unsupported textual claim cannot become an approved target."""
    project = ProjectStore(tmp_path, "case")
    acceptance = admitted_workbook_study(project, narrated_action=True)
    dataset = build_sft_dataset(
        store=project,
        production_sources=(),
        teacher_sources=(),
        continuation_sources=(acceptance,),
        spec=SFTBuildSpec(held_out_fraction=0),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    assert not dataset.rows
    assert len(dataset.sources) == 2 and sum(row.accepted for row in dataset.sources) == 1
    assert sum("assistant prose" in row.detail for row in dataset.inspection.exclusions) == 2
    write_sft_dataset(project, dataset)
    assert load_sft_dataset(project, dataset.dataset.dataset_id, require_accepted=False) == dataset
    with pytest.raises(SFTBuildError, match="insufficient"):
        load_verified_sft_dataset(project, dataset.dataset.dataset_id)


def test_unverified_narration_survives_as_context_for_later_tool_targets(tmp_path: Path) -> None:
    """Skipping a narrated target never erases the actor's actual preceding observations."""
    project = ProjectStore(tmp_path, "case")
    acceptance = admitted_workbook_study(project, narrated_action=True, followup_action=True)
    dataset = build_sft_dataset(
        store=project,
        production_sources=(),
        teacher_sources=(),
        continuation_sources=(acceptance,),
        spec=SFTBuildSpec(held_out_fraction=0),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    assert len(dataset.rows) == 1
    example = dataset.rows[0].example
    assert example.target.tool_calls[0].call_id == "repeat"
    assert any(
        isinstance(event, AssistantActionEvent)
        and event.action.content == "The output has zero rows."
        and event.action.tool_calls
        for event in example.history
    )
    assert any(isinstance(event, ToolEvent) for event in example.history)
    write_sft_dataset(project, dataset)
    assert dataset.dataset.status == "diagnostic"
    assert load_sft_dataset(project, dataset.dataset.dataset_id, require_accepted=False) == dataset
    with pytest.raises(SFTBuildError, match="diagnostic"):
        load_verified_sft_dataset(project, dataset.dataset.dataset_id)


@pytest.mark.parametrize("batched", [False, True])
def test_recovered_tool_error_is_context_never_a_positive_target(
    tmp_path: Path, batched: bool
) -> None:
    """A completed native workflow cannot approve an error or prune an indivisible batch."""
    project = ProjectStore(tmp_path, "case")
    good = workbook_case().first
    bad = good.model_copy(
        update={
            "call_id": "failed",
            "arguments": {
                **good.arguments,
                "deletions": [{"sheet": "Orders", "start": 999, "count": 1}],
            },
        }
    )
    failed = AssistantAction(tool_calls=(bad, good) if batched else (bad,))
    acceptance = admitted_workbook_study(
        project,
        actor_actions=(failed,) if batched else (failed, AssistantAction(tool_calls=(good,))),
    )
    dataset = build_sft_dataset(
        store=project,
        production_sources=(),
        teacher_sources=(),
        continuation_sources=(acceptance,),
        spec=SFTBuildSpec(held_out_fraction=0),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    assert sum(row.accepted for row in dataset.sources) == 1
    assert len(dataset.rows) == (0 if batched else 1)
    prepared, _, _ = resolve_continuation_source(project, acceptance)
    history = prepared[0].transcript_events
    if not batched:
        example = dataset.rows[0].example
        assert example.target == AssistantAction(tool_calls=(good,))
        history = example.history
    assert any(
        isinstance(event, AssistantActionEvent) and event.action == failed for event in history
    )
    assert any(
        isinstance(event, ToolEvent) and "beyond existing rows" in event.content
        for event in history
    )
    assert any("tool result" in row.detail for row in dataset.inspection.exclusions)
    write_sft_dataset(project, dataset)
    if not batched:
        assert dataset.dataset.status == "diagnostic"
        assert (
            load_sft_dataset(project, dataset.dataset.dataset_id, require_accepted=False) == dataset
        )
        with pytest.raises(SFTBuildError, match="diagnostic"):
            load_verified_sft_dataset(project, dataset.dataset.dataset_id)
    else:
        with pytest.raises(SFTBuildError, match="insufficient"):
            load_verified_sft_dataset(project, dataset.dataset.dataset_id)


@pytest.mark.parametrize("flag", [None, 0, "false", True])
def test_missing_or_coerced_success_does_not_admit_a_tool_target(flag: str | int | None) -> None:
    """Observation defaults and coercions cannot invent an affirmative execution result."""
    action = workbook_case().first
    reply: JsonObject = {"content": "looks successful"}
    if flag is not None:
        reply["is_error"] = flag
    messages, reason = _tool_results(
        (action,), [{"action": action.model_dump(mode="json"), "status": "valid", "reply": reply}]
    )
    assert messages[0].content == "looks successful"
    assert reason is not None


@pytest.mark.parametrize("change", ["missing", "arguments", "id", "extra"])
def test_tool_evidence_must_match_the_exact_response_batch(change: str) -> None:
    """A successful reply to another action cannot establish this target's validity."""
    action = workbook_case().first
    saved = action.model_dump(mode="json")
    if change == "arguments":
        saved["arguments"] = {"other": "action"}
    elif change == "id":
        saved["call_id"] = "different-call"
    step: JsonObject = {
        "action": saved,
        "status": "valid",
        "reply": {"content": "done", "is_error": False},
    }
    with pytest.raises(SFTSourceVerificationError):
        _tool_results(
            (action,), [] if change == "missing" else [step, step] if change == "extra" else [step]
        )


def test_export_refuses_fabricated_checkpoint_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A succeeding native suffix cannot validate a tool reply invented in its prefix."""
    original = admission_fixtures.capsule

    def fabricated(store: ArtifactStore) -> tuple[ContinuationCheckpoint, ChatAgentRuntime]:
        """Keep source hashes and world intact while injecting an unexecuted observation."""
        point, actor = original(store)
        call = workbook_case().first
        messages = (
            *point.actor.messages,
            ModelMessage(role="assistant", assistant_action=AssistantAction(tool_calls=(call,))),
            ModelMessage(role="tool", tool_call_id=call.call_id, content="invented workbook state"),
        )
        return point.model_copy(
            update={"actor": point.actor.model_copy(update={"messages": messages})}
        ), actor

    monkeypatch.setattr(admission_fixtures, "capsule", fabricated)
    project = ProjectStore(tmp_path, "case")
    acceptance = admitted_workbook_study(project)
    with pytest.raises(SFTSourceVerificationError, match="unexecuted conversation"):
        resolve_continuation_source(project, acceptance)

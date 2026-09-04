"""Behavior tests for representative-task mining over chat captures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from exp.common.core.artifacts import JsonObject
from exp.simulation.mining.chat_captures import (
    ChatCapture,
    mine_tasks_from_chat_captures,
)
from exp.simulation.mining.descriptors import HashingDescriptorEmbedder
from exp.simulation.mining.service import MiningSpec

_START = datetime(2026, 8, 11, tzinfo=UTC)


def _capture(
    index: int,
    *,
    user: str,
    system: str | None = "You are the support agent for Acme.",
    group_key: str | None = None,
    messages: tuple[JsonObject, ...] | None = None,
) -> ChatCapture:
    """Build one capture with simple system and user text messages."""
    if messages is None:
        system_messages: tuple[JsonObject, ...] = (
            ({"role": "system", "content": system},) if system is not None else ()
        )
        messages = (*system_messages, {"role": "user", "content": user})
    return ChatCapture(
        request_id=f"req-{index}",
        messages=messages,
        captured_at=_START + timedelta(minutes=index),
        group_key=group_key,
    )


def _distinct_captures(count: int) -> tuple[ChatCapture, ...]:
    """Build captures whose user turns are semantically distinct."""
    topics = (
        "reset the billing portal password for an enterprise seat",
        "export every invoice from March as one CSV attachment",
        "merge two duplicate customer accounts without losing notes",
        "escalate a refund that was rejected twice by the auto-check",
        "schedule a quarterly usage report for the finance list",
        "explain why the API key rotation email never arrived",
        "restore a project that was archived by a former employee",
        "change the data residency region for one workspace",
    )
    return tuple(
        _capture(index, user=f"Please {topics[index % len(topics)]} (case {index}).")
        for index in range(count)
    )


def test_mines_tasks_and_reports_capture_honesty() -> None:
    """Text captures are mined; unusable captures are counted, not silently dropped."""
    usable = _distinct_captures(6)
    tool_only = ChatCapture(
        request_id="req-tool-only",
        messages=(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
        ),
        captured_at=_START,
    )
    structured_parts = _capture(
        97,
        user="unused",
        messages=({"role": "user", "content": [{"type": "text", "text": "hi"}]},),
    )
    result = mine_tasks_from_chat_captures(
        (*usable, tool_only, structured_parts),
        MiningSpec(fit_task_budget=4, held_out_task_budget=2),
        embedder=HashingDescriptorEmbedder(),
    )
    assert result.mining is not None
    assert result.tasks
    expected_texts = {
        "You are the support agent for Acme.\n\n" + str(capture.messages[-1]["content"])
        for capture in usable
    }
    assert {task.instruction for task in result.tasks} <= expected_texts
    summary = result.summary
    assert summary.input_capture_count == 8
    assert summary.captures_without_text == 2
    assert summary.eligible_trace_count == 6
    assert summary.selected_task_count == len(result.tasks)
    assert (
        summary.fit_selected_task_count + summary.held_out_selected_task_count
        == summary.selected_task_count
    )
    assert 0.0 <= summary.fit_workload_covered <= 1.0
    assert 0.0 <= summary.held_out_workload_covered <= 1.0
    assert summary.split_separation_verified


def test_workload_covered_discriminates_between_full_and_tight_budgets() -> None:
    """The covered fraction is distance-qualified, not the always-complete assigned mass."""
    captures = _distinct_captures(6)
    full_budget = mine_tasks_from_chat_captures(
        captures,
        MiningSpec(fit_task_budget=4, held_out_task_budget=2),
        embedder=HashingDescriptorEmbedder(),
    )
    assert full_budget.summary.fit_workload_covered == 1.0
    assert full_budget.summary.held_out_workload_covered == 1.0
    tight_budget = mine_tasks_from_chat_captures(
        captures,
        MiningSpec(fit_task_budget=1, held_out_task_budget=1),
        embedder=HashingDescriptorEmbedder(),
    )
    assert tight_budget.summary.fit_workload_covered < 1.0
    assert tight_budget.summary.held_out_workload_covered < 1.0


def test_whitespace_only_content_is_not_usable_text() -> None:
    """A capture whose only text is whitespace counts as unusable, not as a task."""
    capture = ChatCapture(
        request_id="req-blank",
        messages=(
            {"role": "system", "content": " \n\t"},
            {"role": "user", "content": "   "},
        ),
        captured_at=_START,
    )
    result = mine_tasks_from_chat_captures((capture,), embedder=HashingDescriptorEmbedder())
    assert result.mining is None
    assert result.summary.captures_without_text == 1


def test_task_text_joins_system_developer_and_first_user_turn() -> None:
    """The task basis is system plus developer contents plus the first text user turn."""
    capture = ChatCapture(
        request_id="req-shaped",
        messages=(
            {"role": "system", "content": "System rules."},
            {"role": "developer", "content": "Developer rules."},
            {"role": "user", "content": [{"type": "text", "text": "structured, skipped"}]},
            {"role": "user", "content": "First plain user turn."},
            {"role": "user", "content": "Second user turn, never mined."},
        ),
        captured_at=_START,
    )
    result = mine_tasks_from_chat_captures((capture,), embedder=HashingDescriptorEmbedder())
    assert [task.instruction for task in result.tasks] == [
        "System rules.\n\nDeveloper rules.\n\nFirst plain user turn."
    ]


def test_group_keys_resolve_through_the_representative_capture() -> None:
    """Each mined task carries the group key of a capture the task represents."""
    captures = tuple(
        capture.model_copy(update={"group_key": f"sha-{capture.request_id}"})
        for capture in _distinct_captures(6)
    )
    result = mine_tasks_from_chat_captures(
        captures,
        MiningSpec(fit_task_budget=4, held_out_task_budget=2),
        embedder=HashingDescriptorEmbedder(),
    )
    assert set(result.group_keys_by_task_id) == {task.task_id for task in result.tasks}
    for task in result.tasks:
        group_key = result.group_keys_by_task_id[task.task_id]
        assert group_key in {f"sha-{trace_id}" for trace_id in task.source_trace_ids}


def test_group_key_stays_optional() -> None:
    """Captures without a group key mine normally and map to None."""
    result = mine_tasks_from_chat_captures(
        _distinct_captures(3),
        MiningSpec(fit_task_budget=2, held_out_task_budget=1),
        embedder=HashingDescriptorEmbedder(),
    )
    assert result.tasks
    assert set(result.group_keys_by_task_id.values()) == {None}


def test_no_usable_text_returns_an_empty_run_not_an_error() -> None:
    """A window of unusable captures is a normal empty run with honest counts."""
    captures = (
        ChatCapture(
            request_id="req-a",
            messages=({"role": "user", "content": [{"type": "text", "text": "hi"}]},),
            captured_at=_START,
        ),
        ChatCapture(
            request_id="req-b",
            messages=({"role": "assistant", "content": None, "tool_calls": []},),
            captured_at=_START,
        ),
    )
    result = mine_tasks_from_chat_captures(captures, embedder=HashingDescriptorEmbedder())
    assert result.mining is None
    assert result.tasks == ()
    assert result.group_keys_by_task_id == {}
    assert result.summary.input_capture_count == 2
    assert result.summary.captures_without_text == 2
    assert result.summary.selected_task_count == 0


def test_rejects_empty_captures() -> None:
    """Calling the seam with no captures at all is a caller bug."""
    with pytest.raises(ValueError, match="at least one capture"):
        mine_tasks_from_chat_captures((), embedder=HashingDescriptorEmbedder())


def test_rejects_duplicate_request_ids() -> None:
    """Duplicate request ids would corrupt trace identity and fail loudly."""
    capture = _capture(1, user="Please check the audit log for the export.")
    with pytest.raises(ValueError, match="unique"):
        mine_tasks_from_chat_captures((capture, capture), embedder=HashingDescriptorEmbedder())


def test_remining_an_overlapping_window_converges_on_task_ids() -> None:
    """Stable content-derived task ids make overlapping re-mines converge."""
    spec = MiningSpec(fit_task_budget=4, held_out_task_budget=2)
    first = mine_tasks_from_chat_captures(
        _distinct_captures(6),
        spec,
        embedder=HashingDescriptorEmbedder(),
    )
    second = mine_tasks_from_chat_captures(
        _distinct_captures(6),
        spec,
        embedder=HashingDescriptorEmbedder(),
    )
    assert {task.task_id for task in first.tasks} == {task.task_id for task in second.tasks}

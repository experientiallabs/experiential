"""The rollout-ingest contract: shape checks, both fidelities and the evidence adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.dense_spans import actor_spans
from exp.optimize.evaluation.variance.rollout_ingest import (
    ROLLOUT_TRAJECTORY,
    PartTokens,
    Rollout,
    RolloutTurn,
    TranscriptTurn,
    load_rollout_export,
    rollout_evidence,
)
from exp.optimize.evaluation.variance.rubric_evolution_test import (
    ANSWER,
    RUN,
    TASK_SHA,
    WORLD_SHA,
    alternative,
    export,
)


def test_a_full_rollout_becomes_ordered_evidence_with_sampled_tokens() -> None:
    """Instruction as context, every call and reply scored, the answer's tokens bound."""
    rollout = alternative("r-alt", "L1")
    evidence, trajectory, sampled = rollout_evidence(
        rollout, task_sha256=TASK_SHA, world_sha256=WORLD_SHA, trainer_run=RUN
    )
    kinds = [(e.event_id, e.kind, e.owner, e.scope) for e in evidence.events]
    assert kinds[0] == ("e0", "user_message", "user", "context")
    assert [k[1] for k in kinds[1:]] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "final_answer",
    ]
    assert evidence.episode_id == "rollout-r-alt" and evidence.events[1].source.startswith(
        f"rollout:{RUN}:r-alt"
    )
    assert trajectory.kind == ROLLOUT_TRAJECTORY and len(trajectory.turns) == 4
    assert sampled is not None and sampled.parts[0].part == "content"
    (span,) = actor_spans(evidence, trajectory, ["e7"], sampled=sampled)
    assert span.turn_index == 3 and span.sampled is not None


def test_the_transcript_fallback_keeps_turn_order_and_whole_message_spans() -> None:
    """Without tool calls, assistant turns are scored messages and the last is the answer."""
    rollout = Rollout(
        rollout_id="r-text",
        task_id="t",
        lineage_id="L1",
        instruction="Fix it.",
        in_loop_reward=0.7,
        fidelity="transcript",
        transcript=(
            TranscriptTurn(role="assistant", content="Running the tests."),
            TranscriptTurn(role="tool", content="3 passed"),
            TranscriptTurn(role="assistant", content=ANSWER),
        ),
    )
    evidence, trajectory, sampled = rollout_evidence(
        rollout, task_sha256=TASK_SHA, world_sha256=WORLD_SHA, trainer_run=RUN
    )
    assert [e.kind for e in evidence.events] == [
        "user_message",
        "assistant_message",
        "harness_note",
        "final_answer",
    ]
    assert sampled is None and [t.content for t in trajectory.turns][-1] == ANSWER


def test_malformed_rollouts_and_exports_are_refused_whole(tmp_path: Path) -> None:
    """Turn gaps, repeated call ids, mismatched tokens, mixed layouts and other contracts."""
    good = alternative("r-alt", "L1")
    with pytest.raises(ValueError, match="indexed"):
        Rollout.model_validate(
            {**good.model_dump(), "turns": [t.model_dump() for t in good.turns[1:]]}
        )
    with pytest.raises(ValueError, match="repeats a tool call id"):
        Rollout.model_validate(
            {
                **good.model_dump(),
                "turns": [
                    t.model_dump()
                    | {"calls": [c.model_dump() | {"call_id": "same"} for c in t.calls]}
                    for t in good.turns
                ],
            }
        )
    with pytest.raises(ValueError, match="decoded text"):
        PartTokens(token_ids=(1, 2), token_texts=("a",))
    with pytest.raises(ValueError, match="neither content nor tool calls"):
        RolloutTurn(turn_index=0)
    with pytest.raises(ValueError, match="transcript only"):
        Rollout.model_validate(
            {
                **good.model_dump(),
                "fidelity": "transcript",
                "transcript": [{"role": "assistant", "content": "x"}],
            }
        )
    with pytest.raises(ValueError, match="repeats a rollout id"):
        export(1, 1, "a" * 64, (good, good))
    path = tmp_path / "epoch-1.json"
    path.write_text(json.dumps({"contract": "rollout-ingest-v0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="rollout-ingest-v1"):
        load_rollout_export(path)
    saved = export(1, 1, "a" * 64, (good,))
    path.write_text(saved.model_dump_json(), encoding="utf-8")
    assert load_rollout_export(path).identity == saved.identity

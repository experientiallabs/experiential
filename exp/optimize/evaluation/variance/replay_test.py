"""Replays to the decision over a real grounded world, imposing the actor's recorded calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.continuations.job import execute_continuation_job
from exp.optimize.evaluation.variance.contracts import (
    BASELINE_STATE,
    DecisionState,
    VariationAxis,
)
from exp.optimize.evaluation.variance.contrast import run_observation_witness
from exp.optimize.evaluation.variance.replay import replay_to_decision
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    GroundedFixture,
    grounded_fixture,
    tier_decision,
    tier_scenario,
)
from exp.optimize.evaluation.variance.variants import (
    VariantCheckpoint,
    build_variant_checkpoint,
    grounded_state,
    verify_restoration,
)
from exp.simulation.continuations import load_continuation_checkpoint


def _trajectory(fixture: GroundedFixture) -> tuple[ArtifactInput, ...]:
    """Run the fixture job and return the first attempt's boundaries, start first."""
    study = execute_continuation_job(fixture.project, fixture.job, code_revision="test")
    return (study.spec.checkpoint, *study.branches[0].checkpoints)


def _locked(step: int) -> DecisionState:
    """The tier decision with the ticket lock as a second axis."""
    decision = tier_decision(step)
    lock = VariationAxis(path="/tickets/lock", values=("none", "held"), effect="A hold blocks it.")
    return decision.model_copy(update={"axes": (*decision.axes, lock)})


def _variant(
    fixture: GroundedFixture,
    boundaries: tuple[ArtifactInput, ...],
    decision: DecisionState,
    scenario_id: str,
    lock: str | None = None,
) -> VariantCheckpoint:
    """Build a low-tier variant that must fork before the ``check_sla`` turn."""
    variant = build_variant_checkpoint(
        fixture.project,
        base=boundaries[decision.step],
        branch_checkpoints=boundaries,
        decision=decision,
        scenario=tier_scenario(scenario_id, "low", lock=lock),
        facts=None,
        code_revision="test",
        maximum_model_calls=fixture.job.maximum_model_calls,
    )
    assert variant.problems == () and variant.fork_turn == 0
    return variant


def test_a_variant_forked_before_the_reveal_replays_to_the_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded ``check_sla`` call runs again in the edited world and shows the new tier."""
    fixture = grounded_fixture(tmp_path, monkeypatch, first="check_sla")
    boundaries = _trajectory(fixture)
    variant = _variant(fixture, boundaries, tier_decision(1), "routine")
    assert variant.replay_turns == 1
    actor_calls = len(fixture.actor.requests)

    replay = replay_to_decision(
        fixture.project,
        variant=variant,
        branch_checkpoints=boundaries,
        scenario=tier_scenario("routine", "low"),
        job=fixture.job,
        code_revision="test",
    )

    assert replay.status == "replayed" and replay.problems == ()
    assert replay.checkpoint is not None and len(replay.calls) == 1
    assert replay.frozen is not None and replay.frozen.unrestorable(BASELINE_STATE) == ()
    assert len(fixture.actor.requests) == actor_calls
    call = replay.calls[0]
    assert call.call.call_id == "actor-0" and call.intervention is not None
    assert call.original == '{"lock":"none","severity":"high"}'
    assert call.observation is not None
    assert call.observation.content == '{"lock":"none","severity":"low"}'
    original = load_continuation_checkpoint(fixture.project.artifacts, boundaries[1])
    replayed = load_continuation_checkpoint(fixture.project.artifacts, replay.checkpoint)
    assert replayed.actor.calls == original.actor.calls
    assert replayed.actor.messages[:-1] == original.actor.messages[:-1]
    assert replayed.actor.messages[-1].content == call.observation.content
    state = grounded_state(replayed.world)
    assert state is not None and state["tickets"] == {
        "severity": "low",
        "lock": "none",
        "status": "open",
        "outcome": "pending",
    }
    assert verify_restoration(fixture.project, replay.checkpoint)
    witness = run_observation_witness(
        fixture.project,
        checkpoint=replay.checkpoint,
        scenario=tier_scenario("routine", "low"),
        decision=tier_decision(1),
        job=fixture.job,
        code_revision="test",
    )
    assert witness.holds, witness.problems
    predictions = len(fixture.predictor.requests)
    fixture.job.runtime_file.unlink()

    again = replay_to_decision(
        fixture.project,
        variant=variant,
        branch_checkpoints=boundaries,
        scenario=tier_scenario("routine", "low"),
        job=fixture.job,
        code_revision="test",
    )

    assert again == replay and len(fixture.predictor.requests) == predictions


def test_a_recorded_call_the_edited_world_refuses_diverges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the ticket held, the replayed ``check_sla`` errors, so nothing reaches the decision."""
    fixture = grounded_fixture(tmp_path, monkeypatch, first="check_sla")
    boundaries = _trajectory(fixture)
    variant = _variant(fixture, boundaries, _locked(1), "held", lock="held")

    replay = replay_to_decision(
        fixture.project,
        variant=variant,
        branch_checkpoints=boundaries,
        scenario=tier_scenario("held", "low", lock="held"),
        job=fixture.job,
        code_revision="test",
    )

    assert replay.status == "replay_diverged" and replay.checkpoint is None
    assert replay.problems == (
        "turn 1 call actor-0: the edited world returned an error observation",
    )
    assert replay.calls[0].observation is not None and replay.calls[0].observation.is_error
    assert replay.calls[0].original is not None
    assert replay.calls[0].original != replay.calls[0].observation.content
    assert replay.frozen is None
    predictions = len(fixture.predictor.requests)
    fixture.job.runtime_file.unlink()

    again = replay_to_decision(
        fixture.project,
        variant=variant,
        branch_checkpoints=boundaries,
        scenario=tier_scenario("held", "low", lock="held"),
        job=fixture.job,
        code_revision="test",
    )

    assert again == replay and len(fixture.predictor.requests) == predictions


def test_a_recorded_message_stating_a_replaced_value_diverges_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actor wrote the old tier when escalating, so its recorded prefix depends on it."""
    fixture = grounded_fixture(tmp_path, monkeypatch, first="check_sla", narrate=True)
    boundaries = _trajectory(fixture)
    variant = _variant(fixture, boundaries, tier_decision(2), "routine")
    assert variant.replay_turns == 2
    predictions = len(fixture.predictor.requests)

    replay = replay_to_decision(
        fixture.project,
        variant=variant,
        branch_checkpoints=boundaries,
        scenario=tier_scenario("routine", "low"),
        job=fixture.job,
        code_revision="test",
    )

    assert replay.status == "replay_diverged" and replay.calls == ()
    assert replay.problems == (
        'turn 2: the actor\'s recorded messages state replaced value "high", so its recorded '
        "calls depend on the original fact",
    )
    assert len(fixture.predictor.requests) == predictions


def test_a_variant_without_replay_turns_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a persisted variant that forks before its decision is replayed."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    boundaries = _trajectory(fixture)
    variant = build_variant_checkpoint(
        fixture.project,
        base=boundaries[1],
        branch_checkpoints=boundaries,
        decision=tier_decision(1),
        scenario=tier_scenario("routine", "low"),
        facts=None,
        code_revision="test",
    )
    assert variant.replay_turns == 0

    with pytest.raises(ValueError, match="forks at its decision"):
        replay_to_decision(
            fixture.project,
            variant=variant,
            branch_checkpoints=boundaries,
            scenario=tier_scenario("routine", "low"),
            job=fixture.job,
            code_revision="test",
        )

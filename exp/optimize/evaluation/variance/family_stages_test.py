"""Which checkpoint a family variant decides at; its stages run in the end-to-end tests."""

from __future__ import annotations

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.variance.family_stages import FamilyVariant
from exp.optimize.evaluation.variance.replay import DecisionReplay
from exp.optimize.evaluation.variance.variants import VariantCheckpoint


def _ref(name: str) -> ArtifactInput:
    """A reference with a fixed digest."""
    return ArtifactInput(artifact_id=name, sha256="0" * 64)


def test_a_replayed_variant_decides_at_the_replayed_checkpoint_or_nowhere() -> None:
    """Without a replay the variant itself; a diverged replay leaves no decision checkpoint."""
    variant = VariantCheckpoint(
        scenario_id="routine",
        base=_ref("decision"),
        reference=_ref("fork"),
        fork=_ref("start"),
        fork_turn=0,
        revealing_turn=1,
        replay_turns=1,
        edited_state_sha256="1" * 64,
        checks=(),
        problems=(),
    )
    row = FamilyVariant(scenario_id="routine", variant=variant)
    replay = DecisionReplay(
        scenario_id="routine",
        fork=_ref("fork"),
        decision=_ref("decision"),
        decision_step=1,
        status="replayed",
        checkpoint=_ref("replayed"),
    )
    diverged = replay.model_copy(update={"status": "replay_diverged", "checkpoint": None})

    assert row.decision_checkpoint == _ref("fork")
    assert row.model_copy(update={"replay": replay}).decision_checkpoint == _ref("replayed")
    assert row.model_copy(update={"replay": diverged}).decision_checkpoint is None

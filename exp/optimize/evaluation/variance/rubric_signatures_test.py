"""Readings, polarity, the symbolic diff, and the admission corpus guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.behavior_adversarial import HackSignature
from exp.optimize.evaluation.variance.dense_hybrid import RoutingSpec
from exp.optimize.evaluation.variance.rollout_ingest import rollout_evidence
from exp.optimize.evaluation.variance.rubric_evolution_test import world
from exp.optimize.evaluation.variance.rubric_signatures import (
    AdmissionSpec,
    RolloutReading,
    SignatureGroup,
    admit_signature,
    polarity,
    read_rollout,
    symbolic_diff,
)
from exp.simulation.mining.tagging import TagMatrix, TagRow


def reading(rollout_id: str, fired: tuple[str, ...], task: str = "t") -> RolloutReading:
    """One rollout's fired questions."""
    return RolloutReading(rollout_id=rollout_id, task_id=task, lineage_id=rollout_id, fired=fired)


def test_a_rollout_reading_keeps_confident_fires_and_band_answers_apart() -> None:
    """Above the band fires; inside the band is uncertain; a refused matrix is untagged."""
    matrix = TagMatrix(
        bank_id="b",
        bank_version=1,
        question_ids=("a", "b", "c"),
        view="step",
        rows=(
            TagRow(trace_id="s0", unit_hash="h0", values=(0.9, 0.5, 0.1)),
            TagRow(trace_id="s1", unit_hash="h1", values=(0.1, 0.1, 0.1)),
        ),
    )
    row = read_rollout("r", "t", "L", [matrix], RoutingSpec())
    assert row.fired == ("a",) and row.uncertain == ("b",) and row.tagged
    assert not read_rollout("r", "t", "L", [None], RoutingSpec()).tagged


def test_the_diff_groups_flagged_rollouts_by_their_hack_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Polarity comes from bank families; an invisible hack is unexplained."""
    _, registration, _, _ = world(tmp_path, monkeypatch)
    negative, positive = polarity(registration.banks())
    assert "modified_evaluation" in negative and "error.failed_tool_call" in negative
    assert positive == {"changed_implementation", "verified_result"}
    good = [reading("g1", ("changed_implementation", "verified_result"))]
    flagged = [
        reading("f1", ("modified_evaluation", "verified_result")),
        reading("f2", ("modified_evaluation", "verified_result")),
        reading("f3", ("verified_result",)),
    ]
    groups, unexplained = symbolic_diff(flagged, good, negative, positive)
    (group,) = groups
    assert group.signature.name == "+modified_evaluation -changed_implementation"
    assert group.flagged == ("f1", "f2") and group.lineages == ("f1", "f2")
    assert unexplained == ("f3",)


def test_the_held_out_set_never_enters_the_admission_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corpus holding a held-out lineage is refused before any call."""
    project, registration, clef, wire = world(tmp_path, monkeypatch)
    group = SignatureGroup(
        group_id="sig-x",
        signature=HackSignature(added=("modified_evaluation",)),
        added=("modified_evaluation",),
        removed=(),
        flagged=("r",),
        lineages=("H1",),
        genuine=(),
    )
    rollout = registration.held_out.hacks[0].rollout
    evidence, _, _ = rollout_evidence(
        rollout, task_sha256="a" * 64, world_sha256="b" * 64, trainer_run="x"
    )
    with pytest.raises(ValueError, match="held-out lineages"):
        admit_signature(
            project,
            registration.spec.author,
            registration.spec.profile,
            AdmissionSpec(),
            group,
            {"r": ("H1", evidence)},
            held_out_lineages=registration.held_out.lineages(),
            tagger=clef,
            code_revision="test",
        )
    assert wire.requests == [] and clef.calls == 0

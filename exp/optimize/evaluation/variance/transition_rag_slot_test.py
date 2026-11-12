"""The state-rag slot's transition-tag setting: off by default, priced packed when on."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.executable.state_recall import StateRecallBinding
from exp.optimize.evaluation.variance import frontdoor_executors as fx
from exp.optimize.evaluation.variance.frontdoor_family_scope_test import context, corpus
from exp.simulation.retrieval.transition_tags import TransitionRetrievalPolicy


def test_the_default_slot_and_binding_are_todays() -> None:
    """Intent: without the setting nothing changes: state tags, same binding JSON."""
    slot = fx.StateRagSlot(author_alias="glm")
    assert slot.retrieval == "state-tags"
    reference = ArtifactInput(artifact_id="state-tag-index-1", sha256="0" * 64)
    binding = StateRecallBinding(index=reference)
    assert set(binding.model_dump(mode="json")) == {"index", "policy", "decision_alias"}
    transition = StateRecallBinding(
        index=reference,
        transition_bank=ArtifactInput(artifact_id="transition-tag-bank-1", sha256="1" * 64),
        transition_policy=TransitionRetrievalPolicy(),
    )
    assert StateRecallBinding.model_validate(transition.model_dump(mode="json")) == transition


def test_transition_mode_prices_one_packed_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: a transition round is priced by sample, development and authoring, not every step."""
    steps = {"index-fam-a": 1000, "index-fam-b": 7000, "index-fam-c": 5}
    monkeypatch.setattr(
        fx,
        "load_rag_index",
        lambda store, artifact_id: SimpleNamespace(transitions=[None] * steps[artifact_id]),
    )
    slot = fx.StateRagSlot.model_validate(
        {
            "author_alias": "glm",
            "scope": "run_tasks",
            "retrieval": "transition-tags",
            "transition": {"sample_steps": 5000, "development_steps": 120},
        }
    )
    families, calls, _ = fx.state_rag_bounds(context(("t-1",)), slot, corpus())
    assert families == 1
    per = 32 // slot.transition.authoring.maximum_tags
    assert calls.corpus_decisions == -(-5000 // per) + 2 * -(-120 // per)
    assert calls.authoring_completions == slot.transition.authoring.calls
    today = fx.state_rag_bounds(
        context(("t-1",)), fx.StateRagSlot(author_alias="glm", scope="run_tasks"), corpus()
    )[1]
    assert today.corpus_decisions == 7000

"""A per-family slot scoped to ``run_tasks`` works on the run's own families only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance import frontdoor_executors as fx


def family(family_id: str, size: int) -> SimpleNamespace:
    """An admitted family of ``size`` members."""
    return SimpleNamespace(
        family_id=family_id, members=tuple(f"{family_id}-{i}" for i in range(size))
    )


def corpus() -> fx.Corpus:
    """Three admitted families; ``c01`` holds probe ``t-1`` and ``t-2``, ``c00`` holds ``t-0``."""
    families = (family("fam-a", 40), family("fam-b", 12), family("fam-c", 7))
    scoping = SimpleNamespace(
        placements=(
            SimpleNamespace(task_id="t-0", family_id="fam-a"),
            SimpleNamespace(task_id="t-1", family_id="fam-b"),
            SimpleNamespace(task_id="t-2", family_id="fam-b"),
            SimpleNamespace(task_id="t-x", family_id=None),
        ),
        scopes=tuple(
            SimpleNamespace(
                family_id=row.family_id,
                path=f"c0{index}",
                index=SimpleNamespace(artifact_id=f"index-{row.family_id}"),
            )
            for index, row in enumerate(families)
        ),
    )
    report = SimpleNamespace(environment_families=scoping, identity="variance-build-1")
    return fx.Corpus(report=report, families=families, traces={})  # type: ignore[arg-type]


def context(task_ids: tuple[str, ...]) -> fx.SlotContext:
    """A context whose preparation selected ``task_ids``."""
    limits = SimpleNamespace(maximum_families=4, maximum_family_traces=30)
    preparation = SimpleNamespace(task_ids=task_ids, limits=limits)
    project = SimpleNamespace(artifacts=None)
    return fx.SlotContext(project, preparation, "test")  # type: ignore[arg-type]


TAG_BANKS: JsonObject = {
    "author": {
        "model_alias": "glm",
        "model": {
            "provider": "openai-compatible",
            "model_id": "glm-5.3",
            "billing_source": "customer_managed",
            "capabilities_sha256": "0" * 64,
            "connection_sha256": "0" * 64,
        },
        "catalog_sha256": "0" * 64,
        "output_tokens": 16000,
        "maximum_input_bytes": 400000,
        "maximum_physical_attempts": 3,
        "deadline": "2026-11-20T00:00:00Z",
    },
    "loop": {"max_rounds": 0},
}


def test_the_run_families_are_those_holding_the_selected_tasks() -> None:
    """Intent: a spec never names a family by hand; placements decide it."""
    assert fx.run_families(context(("t-1", "t-2")), corpus()) == ("fam-b",)
    assert fx.run_families(context(("t-0", "t-1")), corpus()) == ("fam-a", "fam-b")
    with pytest.raises(ValueError, match="needs preparation.task_ids"):
        fx.run_families(context(()), corpus())
    with pytest.raises(ValueError, match="not placed in any family: t-x"):
        fx.run_families(context(("t-1", "t-x")), corpus())


def test_tag_bank_bounds_count_only_the_scoped_family() -> None:
    """Intent: pricing and plan text follow the narrower scope; the default prices every family."""
    every = fx.tag_bank_bounds(
        context(("t-1",)), fx.TagBankSlot.model_validate(TAG_BANKS), corpus()
    )
    scoped = fx.tag_bank_bounds(
        context(("t-1",)),
        fx.TagBankSlot.model_validate({**TAG_BANKS, "scope": "run_tasks"}),
        corpus(),
    )
    assert (every.families, scoped.families) == (3, 1)
    assert every.decisions * 12 == scoped.decisions * (40 + 12 + 7)
    assert every.author == 3 * scoped.author
    assert every.note == ""
    assert scoped.note == " (scope run_tasks: only c01, of 3 admitted)"


def test_state_rag_bounds_count_only_the_scoped_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: only the run family's fit steps are priced for tagging."""
    steps = {"index-fam-a": 1000, "index-fam-b": 70, "index-fam-c": 5}
    monkeypatch.setattr(
        fx,
        "load_rag_index",
        lambda store, artifact_id: SimpleNamespace(transitions=[None] * steps[artifact_id]),
    )
    every_slot = fx.StateRagSlot(author_alias="glm")
    scoped_slot = fx.StateRagSlot(author_alias="glm", scope="run_tasks")
    families, every, _ = fx.state_rag_bounds(context(("t-1",)), every_slot, corpus())
    scoped_families, scoped, note = fx.state_rag_bounds(context(("t-1",)), scoped_slot, corpus())
    assert (families, scoped_families) == (3, 1)
    assert scoped.corpus_decisions * 1075 == every.corpus_decisions * 70
    assert scoped.authoring_completions * 3 == every.authoring_completions
    assert "only c01" in note
    assert [
        row.family_id for row in fx.state_rag_scopes(context(("t-1",)), scoped_slot, corpus())
    ] == ["fam-b"]


def test_unsaved_families_price_at_most_one_family_per_selected_task() -> None:
    """Intent: before the families are saved, a scoped slot never prices more than its tasks."""
    slot = fx.TagBankSlot.model_validate({**TAG_BANKS, "scope": "run_tasks"})
    assert fx.tag_bank_bounds(context(("t-1",)), slot, None).families == 1
    every = fx.TagBankSlot.model_validate(TAG_BANKS)
    assert fx.tag_bank_bounds(context(("t-1",)), every, None).families == 4


def test_the_scope_joins_the_slot_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: an all-family result and a scoped one, or two scoped families, never collide.

    An explicit ``scope: "all"`` keeps the identity of a spec that never named a scope, so
    every saved all-family result still replays.
    """
    monkeypatch.setattr(fx, "refined_identity", lambda project, roles: {})

    def identity(name: str, settings: dict[str, Any], tasks: tuple[str, ...] = ("t-1",)) -> str:
        return fx.slot_identity(name, settings, context(tasks), corpus())

    base = identity("tag-banks", TAG_BANKS)
    assert identity("tag-banks", {**TAG_BANKS, "scope": "all"}) == base
    scoped = identity("tag-banks", {**TAG_BANKS, "scope": "run_tasks"})
    assert scoped != base
    assert identity("tag-banks", {**TAG_BANKS, "scope": "run_tasks"}, ("t-2",)) == scoped
    assert identity("tag-banks", {**TAG_BANKS, "scope": "run_tasks"}, ("t-0",)) != scoped
    state = {"author_alias": "glm"}
    assert identity("state-rag", {**state, "scope": "run_tasks"}) != identity("state-rag", state)
    # A slot that takes no scope hashes its settings unchanged.
    assert identity("rule-admission", {"scope": "x"}) != identity("rule-admission", {})


def test_task_types_refuse_a_scope_wider_than_the_banks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Intent: families outside the banks' scope have no bank, so task types cannot cover them."""
    monkeypatch.setattr(fx, "saved_corpus", lambda context: corpus())
    monkeypatch.setattr(fx, "slot_identity", lambda *args, **kwargs: "task-types-1")
    monkeypatch.setattr(
        fx,
        "read_slot",
        lambda context, reference, model: fx.FamilyBanks(banks={}, scope=("fam-b",)),
    )
    store = SimpleNamespace(exists=lambda identity: False)
    ctx = fx.SlotContext(
        SimpleNamespace(artifacts=store),  # type: ignore[arg-type]
        context(("t-1",)).preparation,
        "test",
        {"tag-banks": SimpleNamespace(artifact_id="banks")},  # type: ignore[dict-item]
    )
    with pytest.raises(ValueError, match="wider than the tag-banks scope"):
        fx.TaskTypeExecutor().run(ctx, {"namer_alias": "glm-flash"})

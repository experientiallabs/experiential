"""Embedder binding refuses a different embedder; memory-first simulators bind none."""

from pathlib import Path

import pytest

from exp.common.models import load_model_catalog
from exp.optimize.evaluation.held_out_fidelity_test import (
    USERS,
    SimulatedProvider,
    build_project,
    episode,
)
from exp.optimize.evaluation.retrieval_binding import index_embedder_alias
from exp.simulation.world_model.runtime import (
    load_grounded_world_model_artifact,
    persist_memory_first_world_model,
)


def test_semantic_index_needs_the_build_embedder_and_memory_first_needs_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured embedder must be the index's; a memory-first simulator retrieves nothing."""
    _, store, _ = build_project(
        tmp_path, monkeypatch, SimulatedProvider([episode(i) for i in range(USERS)])
    )
    build = store.load_project().build
    assert build is not None
    catalog = load_model_catalog(store.model_catalog_path)
    served = load_grounded_world_model_artifact(store.artifacts, build.world_model)
    assert index_embedder_alias(store, catalog, served) == "embed"
    memory_first = load_grounded_world_model_artifact(
        store.artifacts, persist_memory_first_world_model(store.artifacts, build.world_model)
    )
    assert memory_first.original_evidence == "none"
    assert index_embedder_alias(store, catalog, memory_first) is None
    record = catalog.models["embed"]
    other = catalog.model_copy(
        update={"models": {**catalog.models, "embed": record.model_copy(update={"model": "x"})}}
    )
    with pytest.raises(ValueError, match="was embedded with embed-v1"):
        index_embedder_alias(store, other, served)

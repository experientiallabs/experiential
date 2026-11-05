"""Native runtime catalog bindings remain isolated and preserve explicit retrieval aliases."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.models import ModelCapabilities, load_model_catalog
from exp.optimize.evaluation.audit.panel_test import (
    USERS,
    SimulatedProvider,
    build_project,
    episode,
)
from exp.optimize.evaluation.executable.catalog_binding import (
    load_runtime_catalog,
    register_runtime_catalog,
)
from exp.optimize.evaluation.executable.costs import grounded_allowance_costs
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    preflight_grounded_runtime,
)
from exp.optimize.evaluation.fidelity.retrieval_binding import index_embedder_alias
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


def test_bound_catalog_registration_replays_without_mutating_global_project(tmp_path: Path) -> None:
    """Native registration preserves global catalog bytes and refuses forged alias/ref metadata."""
    project, _ = source_project(tmp_path)
    before = project.model_catalog_path.read_bytes()
    catalog = load_model_catalog(project.model_catalog_path)
    embed = catalog.models["model"].model_copy(
        update={
            "model": "fixture-embedding",
            "capabilities": ModelCapabilities(
                supports_embeddings=True, context_window_tokens=32768
            ),
        }
    )
    target = catalog.model_copy(update={"models": {**catalog.models, "bound-embed": embed}})
    binding = register_runtime_catalog(
        project, target, embedder_alias="bound-embed", code_revision="fixture"
    )
    assert load_runtime_catalog(project, binding) == target
    assert (
        register_runtime_catalog(
            project, target, embedder_alias="bound-embed", code_revision="fixture"
        )
        == binding
    )
    assert project.model_catalog_path.read_bytes() == before
    with pytest.raises(ValueError, match="alias changed"):
        load_runtime_catalog(project, binding.model_copy(update={"embedder_alias": "model"}))
    with pytest.raises(ValueError, match="reference changed"):
        load_runtime_catalog(
            project,
            binding.model_copy(
                update={"catalog": binding.catalog.model_copy(update={"sha256": "0" * 64})}
            ),
        )


def test_explicit_runtime_alias_wins_over_old_project_embedding_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A versioned runtime alias is verified against the real built index snapshot."""
    _, project, _ = build_project(
        tmp_path, monkeypatch, SimulatedProvider([episode(i) for i in range(USERS)])
    )
    build = project.load_project().build
    assert build is not None
    catalog = load_model_catalog(project.model_catalog_path)
    original = catalog.models["embed"]
    target = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "embed": original.model_copy(update={"model": "old-project-other-model"}),
                "bound-embed": original,
            }
        }
    )
    world = load_grounded_world_model_artifact(project.artifacts, build.world_model)
    with pytest.raises(ValueError, match="was embedded"):
        index_embedder_alias(project, target, world)
    assert (
        index_embedder_alias(project, target, world, configured_alias="bound-embed")
        == "bound-embed"
    )


def test_actual_runtime_preflight_and_price_inventory_use_bound_catalog(tmp_path: Path) -> None:
    """The actual runtime boundary loads the explicit catalog; no model/runtime call is needed."""
    project, fidelity = source_project(tmp_path)
    catalog = load_model_catalog(project.model_catalog_path)
    embed = catalog.models["model"].model_copy(
        update={
            "model": "target-embedding",
            "capabilities": ModelCapabilities(
                supports_embeddings=True, context_window_tokens=32768
            ),
        }
    )
    target = catalog.model_copy(update={"models": {**catalog.models, "bound-embed": embed}})
    binding = register_runtime_catalog(
        project, target, embedder_alias="bound-embed", code_revision="fixture"
    )
    definition = GroundedRuntimeDefinition(
        task=fidelity.task.model_copy(
            update={"origin": "authored", "source_trace_ids": (), "partition": "fit"}
        ),
        world_model=fidelity.world_model,
        catalog_sha256=sha256_json(target),
        catalog_binding=binding,
        manifest=WorldManifest(
            domains=(StateDomain(name="records", owner="simulated"),),
            tools=(ToolEffectPolicy(name="inspect", reads=("records",), writes=("records",)),),
        ),
        initial_state={"records": {}},
        probe_paths={"status": "/records"},
        maximum_physical_attempts=3,
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )
    preflight_grounded_runtime(project, definition)
    costs = grounded_allowance_costs(
        project, definition.world_model, attempts=3, output_tokens=4096, catalog_binding=binding
    )
    assert {row.name for row in costs} == {"model", "bound-embed"}
    assert all(row.attempts == 3 for row in costs)
    assert next(row for row in costs if row.name == "bound-embed").usd is None
    legacy = definition.model_copy(update={"catalog_binding": None})
    with pytest.raises(ValueError, match="catalog differs"):
        preflight_grounded_runtime(project, legacy)

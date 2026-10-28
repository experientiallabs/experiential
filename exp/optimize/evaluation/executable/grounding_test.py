"""A simulated world grounded in the build: its simulator, learned memory and trace facts."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_bytes
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase, load_task_set
from exp.optimize.evaluation.audit.panel_test import (
    USERS,
    SimulatedProvider,
    build_project,
    episode,
    learn_fit,
)
from exp.optimize.evaluation.executable import grounded
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.grounding import GroundedRuntimeRequest, grounded_capture
from exp.optimize.evaluation.learning.lineage import plan_world_model
from exp.optimize.evaluation.learning.traces import learning_plan
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.simulation.development_scope import development_tasks
from exp.simulation.world_model.experience_memory import memory_head

DEADLINE = datetime.now(UTC) + timedelta(hours=1)


def status_fact(index: int) -> JsonObject:
    """The instance facts one recorded episode's package supplies: its order's status."""
    return {
        "facts": [
            {
                "key": f"orders/#W{1000 + index}/status",
                "value": "pending" if index % 3 else "delivered",
                "visibility": "conditioning",
                "provenance": {
                    "adapter": "fixture-orders",
                    "source": "order service before the episode",
                    "validation": {"status": "validated", "method": "read from the service"},
                    "state_version": 0,
                },
            }
        ]
    }


def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, learn: bool) -> ProjectStore:
    """Build, and learn every trace when asked, a package whose even episodes supply facts."""
    episodes = [episode(index) for index in range(USERS)]
    for index, row in enumerate(episodes):
        if index % 2 == 0:
            row["facts"] = status_fact(index)
    _, project, _ = build_project(
        tmp_path, monkeypatch, SimulatedProvider(episodes), learn=learn, episodes=episodes
    )
    return project


def request(task: TaskCase, **changes: object) -> GroundedRuntimeRequest:
    """A simulated order domain for one task, with the given request fields changed."""
    fields: JsonObject = {
        "task_id": task.task_id,
        "manifest": WorldManifest(
            domains=(StateDomain(name="orders", owner="simulated"),),
            tools=(ToolEffectPolicy(name="get_order_details", reads=("orders",)),),
        ).model_dump(mode="json"),
        "initial_state": {"orders": {"status": "pending"}},
        "probe_paths": {"status": "/orders/status"},
        "maximum_physical_attempts": 8,
        "deadline": DEADLINE.isoformat(),
    }
    return GroundedRuntimeRequest.model_validate({**fields, **changes})


def source_index(task: TaskCase) -> int:
    """The fixture episode a build task was mined from."""
    return int(str(task.initial_context["task_lineage"]).removeprefix("user_"))


def test_grounding_pins_the_build_simulator_memory_and_trace_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The capture names the built-in runtime and the build's fit-only references, without a call.

    The memory ``exp build`` learned from every trace, held-out lineages included, is never
    grounded: a development world waits for the fit-only memory and reads the fit-only index.
    """
    project = built(tmp_path, monkeypatch, learn=True)
    build = project.load_project().build
    assert build is not None
    tasks = development_tasks(project)
    with_facts = next(task for task in tasks if source_index(task) % 2 == 0)
    assert memory_head(project.artifacts, learning_plan(project, scope="all").lineage)
    with pytest.raises(ValueError, match="the build has no fit-only simulator memory"):
        grounded_capture(project, request(with_facts, trace_facts=True))
    learn_fit(project.paths.root, project)
    fit_plan = learning_plan(project, scope="fit")
    fit_index = plan_world_model(project.artifacts, build, "fit")
    assert fit_index != build.world_model
    capture = grounded_capture(project, request(with_facts, trace_facts=True))
    definition = GroundedRuntimeDefinition.model_validate(capture.configuration)
    assert definition.task == capture.task == with_facts
    assert definition.memory is not None and definition.memory.version >= 1
    assert definition.memory.lineage == fit_plan.lineage
    assert definition.world_model == fit_index
    assert capture.sources == (definition.world_model, build.trace_dataset)
    assert definition.facts is not None
    (fact,) = definition.facts.facts.facts
    assert fact.key == f"orders/#W{1000 + source_index(with_facts)}/status"
    assert capture.runtime_file == Path(grounded.__file__)
    assert capture.runtime_sha256 == sha256_bytes(Path(grounded.__file__).read_bytes())
    assert capture.probe_sha256 == definition.probe_sha256

    bare = grounded_capture(project, request(with_facts, memory="none"))
    bare_definition = GroundedRuntimeDefinition.model_validate(bare.configuration)
    assert bare_definition.memory is None and bare_definition.facts is None
    assert bare_definition.world_model == fit_index == bare.sources[0]

    without = next(task for task in tasks if source_index(task) % 2)
    with pytest.raises(ValueError, match="supply no instance facts; declare facts explicitly"):
        grounded_capture(project, request(without, trace_facts=True))


def test_grounding_refuses_held_out_tasks_and_inconsistent_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Held-out tasks, two fact sources, absent memory and stray state are refused."""
    project = built(tmp_path, monkeypatch, learn=False)
    build = project.load_project().build
    assert build is not None
    tasks = load_task_set(project.artifacts, build.task_set.artifact_id).tasks
    held_out = next(task for task in tasks if task.partition == "held_out")
    with pytest.raises(ValueError, match="is held out; development may use fit tasks only"):
        grounded_capture(project, request(held_out))
    fit = development_tasks(project)[0]
    with pytest.raises(ValueError, match="the build has no fit-only simulator memory"):
        grounded_capture(project, request(fit))
    with pytest.raises(ValueError, match="initial state must name exactly the simulated domains"):
        grounded_capture(project, request(fit, memory="none", initial_state={"other": {}}))
    facts = {"facts": status_fact(0)}
    with pytest.raises(ValueError, match="declare facts or set trace_facts, not both"):
        request(fit, facts=facts, trace_facts=True)


def test_grounding_preserves_default_identity_and_pins_explicit_predictor_reasoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider-free capture carries a requested profile without changing the saved catalog."""
    project = built(tmp_path, monkeypatch, learn=False)
    task = development_tasks(project)[0]
    baseline = request(task, memory="none")
    assert "reasoning_effort" not in baseline.model_dump(mode="json")
    unchanged_catalog = project.model_catalog_path.read_bytes()
    ordinary = grounded_capture(project, baseline)
    assert "reasoning_effort" not in ordinary.configuration
    explicit = grounded_capture(project, request(task, memory="none", reasoning_effort="none"))
    assert explicit.configuration["reasoning_effort"] == "none"
    assert explicit.configuration["world_model"] == ordinary.configuration["world_model"]
    assert project.model_catalog_path.read_bytes() == unchanged_catalog
    with pytest.raises(ValueError, match="reasoning effort"):
        grounded_capture(project, request(task, memory="none", reasoning_effort="high"))

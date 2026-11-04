"""One simulator per environment family: shared learning, shared definition, isolated attempts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue, sha256_json
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase, ToolSchema
from exp.optimize.evaluation.continuations.study import CheckpointSession
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    RuleMemoryBinding,
)
from exp.optimize.evaluation.variance.family_simulators import (
    AttemptSlot,
    FamilyRulesInput,
    FamilySimulator,
    LearnedRules,
    bind_runtime,
    definition_for,
    learn_family_rules,
    load_family_simulator,
    plan_attempts,
    run_isolated_attempts,
    save_family_simulator,
)
from exp.runtime.environments.record_world import RecordTool, RecordWorldFactory
from exp.runtime.environments.world_contracts import (
    StateDomain,
    ToolEffectPolicy,
    WorldAccess,
    WorldManifest,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime, WorldContext
from exp.simulation.mining.environment_families import family_task
from exp.simulation.retrieval.scope import CorpusScope
from exp.simulation.rules.recall import RecallPolicy

ACTORS = ("gemma-4-26b-a4b-it", "glm-5.3-flash", "glm-5.3", "kimi-k3", "mimo-v2.6-pro")


def ref(name: str) -> ArtifactInput:
    """A pinned artifact reference."""
    return ArtifactInput(artifact_id=f"{name}-0123456789abcdef0123", sha256=sha256_json(name))


def task(task_id: str) -> TaskCase:
    """A fit task exposing the add tool."""
    return TaskCase(
        task_id=task_id,
        lineage_group_id=f"lineage-{task_id}",
        partition="fit",
        instruction="Add one item.",
        initial_context={"start": task_id},
        tools=(ToolSchema(name="add", description="add", input_schema={"type": "object"}),),
        workload_weight=1,
        origin="authored",
    )


def simulator(*task_ids: str) -> FamilySimulator:
    """An unlearned family simulator over two member traces."""
    return FamilySimulator(
        family_id="environment-family-0123456789abcdef0123",
        status="compatible",
        cluster_id="c00",
        path="c00/0",
        members=("w1", "w2"),
        corpus=CorpusScope.declare(
            name="family-x", kind="trace", ids=("w1", "w2"), source_plan="a" * 64
        ),
        index=ref("rag-index"),
        world_model=ref("world-model"),
        rules_traces=("w1", "w2"),
        tasks=tuple(family_task(task(t), "mining_probes") for t in task_ids),
        families=ref("environment-families"),
    )


@dataclass
class CountingLearner:
    """A learner that counts its calls and binds a fixed rule book and memory."""

    identity: str = "learner-v1"
    calls: list[FamilyRulesInput] = field(default_factory=list)

    def __call__(self, inputs: FamilyRulesInput) -> LearnedRules:
        """Learn once."""
        self.calls.append(inputs)
        return LearnedRules(
            rule_book=ref("rule-book"),
            rules_report=ref("rules-stage-report"),
            rule_memory=RuleMemoryBinding(
                memory_set=ref("rule-memory-set"), recall=RecallPolicy(jev_alias="jev")
            ),
        )


def template() -> GroundedRuntimeDefinition:
    """A runtime template of the records world."""
    return GroundedRuntimeDefinition(
        task=task("template"),
        world_model=ref("other-world"),
        catalog_sha256="c" * 64,
        manifest=WorldManifest(
            domains=(StateDomain(name="items", owner="simulated"),),
            tools=(ToolEffectPolicy(name="add", reads=("items",), writes=("items",)),),
        ),
        initial_state={"items": []},
        probe_paths={"items": "/items"},
        maximum_physical_attempts=4,
        deadline=datetime(2030, 1, 1, tzinfo=UTC) + timedelta(minutes=5),
    )


def items(arguments: JsonObject) -> WorldAccess:
    """The add tool reads and writes the items table."""
    del arguments
    return WorldAccess(reads=frozenset({"items"}), writes=frozenset({"items"}))


def add(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Insert one item and report how many there are now."""
    del context
    db.execute("INSERT INTO items (name) VALUES (?)", (arguments["name"],))
    return {"count": db.execute("SELECT COUNT(*) FROM items").fetchone()[0]}


def records_world() -> ExecutableWorldRuntime:
    """The family's executable world: one items table holding one seed row."""
    return ExecutableWorldRuntime(
        WorldManifest(
            domains=(StateDomain(name="items", owner="native"),),
            tools=(ToolEffectPolicy(name="add", reads=("items",), writes=("items",)),),
        ),
        RecordWorldFactory(
            service={"name": "items"},
            schema="CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);"
            "INSERT INTO items (name) VALUES ('seed');",
            tools={"add": RecordTool(access=items, run=add)},
        ),
    )


def test_rules_are_learned_once_per_family_and_replayed(tmp_path: Path) -> None:
    """A family's rule memory is learned once; a later run replays it with no learning call."""
    store = ProjectStore(tmp_path, "case").artifacts
    learner = CountingLearner()

    learned = learn_family_rules(store, simulator("t1"), learner, code_revision="test")
    again = learn_family_rules(store, simulator("t1", "t2"), learner, code_revision="test")

    assert len(learner.calls) == 1 and learner.calls[0].traces == ("w1", "w2")
    assert learned.rule_book == again.rule_book == ref("rule-book")
    assert again.rules_report == ref("rules-stage-report")
    assert learned.identity() == again.identity()
    saved = save_family_simulator(store, learned, code_revision="test")
    assert load_family_simulator(store, saved) == learned


def test_every_task_of_a_family_shares_one_simulator_definition() -> None:
    """Two tasks derive definitions that differ only in their task and initial state."""
    learned = simulator("t1", "t2").model_copy(
        update={"rule_book": ref("rule-book"), "tag_retrieval": ref("binding")}
    )
    shared = bind_runtime(learned, template())

    first = definition_for(shared, task("t1"), {"items": []})
    second = definition_for(shared, task("t2"), {"items": [{"name": "x"}]})

    assert first.world_model == second.world_model == learned.world_model
    assert first.rule_book == second.rule_book == ref("rule-book")
    assert first.tag_retrieval == ref("binding")
    keep = {"task", "initial_state"}
    assert first.model_dump(exclude=keep) == second.model_dump(exclude=keep)
    assert first.task.task_id == "t1" and second.task.task_id == "t2"
    with pytest.raises(KeyError, match="not in environment family"):
        definition_for(shared, task("elsewhere"), {"items": []})


def test_five_actors_times_sixteen_attempts_are_eighty_isolated_runs_of_one_simulator(
    tmp_path: Path,
) -> None:
    """80 attempts restore one start into 80 fresh worlds; no learning call is added."""
    store = ProjectStore(tmp_path, "case").artifacts
    learner = CountingLearner()
    shared = bind_runtime(
        learn_family_rules(store, simulator("t1"), learner, code_revision="test"), template()
    )
    world = records_world()
    with world.open(task("t1")) as opened:
        start = opened.checkpoint()
    slots = plan_attempts(shared, "t1", ACTORS, 16, start)
    seen: list[int] = []

    def episode(session: CheckpointSession, slot: AttemptSlot) -> int:
        """Add one item; an isolated world always holds the seed plus this attempt's item."""
        reply = session.execute(
            ToolCall(call_id="a", name="add", arguments={"name": f"{slot.actor}-{slot.attempt}"})
        )
        count = int(str(reply.content).split(":")[1].rstrip("}"))
        seen.append(count)
        return count

    runs = run_isolated_attempts(world, task("t1"), start, slots, episode)

    assert len(runs) == 80 == len({run.slot.identity() for run, _ in runs})
    assert {run.slot.simulator for run, _ in runs} == {shared.identity()}
    assert seen == [2] * 80
    assert len({run.final_sha256 for run, _ in runs}) == 80
    assert len(learner.calls) == 1
    replayed = learn_family_rules(store, simulator("t1"), CountingLearner(), code_revision="test")
    assert replayed.rule_memory == shared.rule_memory


def test_attempts_against_another_simulator_or_start_are_refused() -> None:
    """Slots must share one simulator definition, task and start."""
    world = records_world()
    with world.open(task("t1")) as opened:
        start = opened.checkpoint()
    slots = plan_attempts(simulator("t1"), "t1", ACTORS[:1], 1, start)
    other = slots[0].model_copy(update={"simulator": "another"})
    with pytest.raises(ValueError, match="one simulator"):
        run_isolated_attempts(world, task("t1"), start, (*slots, other), lambda s, slot: 0)
    with pytest.raises(ValueError, match="this task"):
        run_isolated_attempts(world, task("t2"), start, slots, lambda s, slot: 0)

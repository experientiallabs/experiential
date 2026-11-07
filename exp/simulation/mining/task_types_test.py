"""Task types inside environment families: planted structure, placement, naming and replay."""

from __future__ import annotations

from pathlib import Path

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectStore
from exp.simulation.mining.environment_families import (
    CompatibilityEvidence,
    EnvironmentFamily,
    FamilyTask,
    StartingState,
)
from exp.simulation.mining.tag_cluster_names import ClusterNamer
from exp.simulation.mining.tag_cluster_names_test import SNAPSHOT, ScriptedNamer, name_reply
from exp.simulation.mining.tag_clusters_test import planted
from exp.simulation.mining.tagging import TagMatrix
from exp.simulation.mining.task_types import (
    TASK_TYPES_TYPE,
    TaskDifficulty,
    TaskTypes,
    TaskTypeSpec,
    discover_task_types,
    split_family,
    task_type_family,
    task_types,
    with_difficulty,
)

QUESTIONS = {f"q{i}": f"Does proposition {i} hold?" for i in range(12)}


def task(task_id: str, *sources: str) -> FamilyTask:
    """A world-learning family task with the given source traces."""
    return FamilyTask(
        task_id=task_id,
        lineage_id=f"lineage-{task_id}",
        role="world_learning",
        source_trace_ids=sources,
        start=StartingState(task_id=task_id, initial_context_sha256="0" * 64),
    )


def family(members: tuple[str, ...], tasks: tuple[FamilyTask, ...] = ()) -> EnvironmentFamily:
    """A compatible family over the given members."""
    return EnvironmentFamily(
        family_id="environment-family-x",
        cluster_id="c00",
        path="c00",
        status="compatible",
        members=members,
        world_semantics="one spreadsheet world",
        tasks=tasks,
        evidence=CompatibilityEvidence(rationale="one environment"),
    )


def matrices() -> tuple[list[TagMatrix], dict[str, int]]:
    """A task-view and an authored observations-view matrix over the planted groups."""
    task_view, truth = planted()
    authored, _ = planted(view="observations", seed=11)
    return [task_view, authored], truth


def test_planted_task_types_are_found_named_and_given_their_tasks() -> None:
    """Three planted groups become three named task types; tasks follow their source traces."""
    joined, truth = matrices()
    members = tuple(sorted(t for t in truth if not t.endswith("-07")))
    outside = "g2-07"  # tagged but not a family member: placed under the fitted mixture
    tasks = (
        task("t-a", "g0-00", "g0-01", "g1-00"),
        task("t-b", "g1-03"),
        task("t-c", outside),
        task("t-unknown", "nowhere"),
    )
    client = ScriptedNamer(name_reply)
    row, clustering, names = split_family(
        family(members, tasks),
        joined,
        ClusterNamer(client, SNAPSHOT),
        questions=QUESTIONS,
        exemplar_texts={t: f"episode {t}" for t in members},
    )

    assert len(row.task_types) == 3
    for task_type in row.task_types:
        assert len({truth[t] for t in task_type.members}) == 1
        assert task_type.name == f"Final {task_type.cluster_id}"
        assert task_type.family_id == "environment-family-x"
        assert task_type.difficulty.level is None
    by_group = {truth[t.members[0]]: t for t in row.task_types}
    assert by_group[0].task_ids() == ("t-a",)
    assert by_group[1].task_ids() == ("t-b",)
    assert by_group[2].task_ids() == ("t-c",)
    assert row.unplaced == ("t-unknown",)
    assert row.clustering == clustering.identity()
    assert len(names.names) == 3
    assert len(client.requests) == 3 * 4
    stances = {str(r.messages[-1].content).split("\n")[1] for r in client.requests[:3]}
    assert any("discrete kind of task" in s for s in stances)


def test_a_small_family_stays_one_task_type() -> None:
    """Below the split floor a family is never split."""
    joined, truth = matrices()
    members = tuple(sorted(truth))[:5]
    row, _, _ = split_family(
        family(members),
        joined,
        ClusterNamer(ScriptedNamer(name_reply), SNAPSHOT),
        questions=QUESTIONS,
        exemplar_texts={},
        spec=TaskTypeSpec(min_members_to_split=6),
    )
    assert len(row.task_types) == 1 and set(row.task_types[0].members) == set(members)


def test_a_task_type_narrows_its_family_and_difficulty_fills_from_tasks() -> None:
    """A task type feeds its own family; difficulty summarizes its tasks by lower median."""
    joined, truth = matrices()
    members = tuple(sorted(truth))
    parent = family(members, (task("t-a", "g0-00"), task("t-b", "g0-01"), task("t-c", "g0-02")))
    record = discover_task_types(
        [parent],
        joined,
        ClusterNamer(ScriptedNamer(name_reply), SNAPSHOT),
        families_record="environment-families-x",
        questions=QUESTIONS,
        exemplar_texts={},
    )
    first = next(t for t in record.all() if t.tasks)
    narrowed = task_type_family(parent, first)
    assert narrowed.members == first.members and narrowed.tasks == first.tasks
    assert narrowed.world_semantics == parent.world_semantics
    assert narrowed.path == f"c00/{first.cluster_id}"
    assert narrowed.family_id == first.task_type_id

    def tier(level: int) -> TaskDifficulty:
        """A difficulty at one level of a five-actor ladder."""
        return TaskDifficulty(
            level=level,
            label=f"level {level}",
            low=level,
            high=level,
            actors=5,
            score=0.5,
            confident=True,
        )

    filled = with_difficulty(record, {"t-a": tier(1), "t-b": tier(4)})
    rated = next(t for t in filled.all() if t.tasks)
    assert rated.difficulty.level == 1 and rated.difficulty.label == "level 1"
    assert (rated.difficulty.lowest, rated.difficulty.highest) == (1, 4)
    assert rated.difficulty.unrated == ("t-c",)
    assert filled.task_type_of("t-a") is not None
    assert filled.task_type_of("missing") is None


def test_task_types_replay_from_their_artifacts(tmp_path: Path) -> None:
    """The record, each clustering and each set of names are saved; a replay calls nothing."""
    joined, truth = matrices()
    parent = family(tuple(sorted(truth)))
    store = ProjectStore(tmp_path, "case").artifacts
    client = ScriptedNamer(name_reply)

    def run() -> tuple[TaskTypes, ArtifactInput]:
        """Find or replay the task types."""
        return task_types(
            store,
            [parent],
            joined,
            ClusterNamer(client, SNAPSHOT),
            families_record="environment-families-x",
            questions=QUESTIONS,
            exemplar_texts={},
            code_revision="t",
        )

    first, reference = run()
    assert store.read(reference.artifact_id).manifest.artifact_type == TASK_TYPES_TYPE
    cited = {ref.artifact_id for ref in store.read(reference.artifact_id).manifest.inputs}
    assert first.by_family[0].clustering in cited and first.by_family[0].names in cited
    calls = len(client.requests)
    again, same = run()
    assert again == first and same == reference and len(client.requests) == calls


def test_another_familys_authored_bank_is_left_out_of_a_familys_space() -> None:
    """A matrix tagging none of the family's members does not empty its tag space."""
    joined, truth = matrices()
    other, _ = planted(view="task", seed=5)
    elsewhere = other.model_copy(
        update={
            "question_ids": tuple(f"other.{q}" for q in other.question_ids),
            "rows": tuple(
                row.model_copy(update={"trace_id": f"other-{row.trace_id}"}) for row in other.rows
            ),
        }
    )
    members = tuple(sorted(truth))
    row, clustering, _ = split_family(
        family(members),
        [*joined, elsewhere],
        ClusterNamer(ScriptedNamer(name_reply), SNAPSHOT),
        questions=QUESTIONS,
        exemplar_texts={t: f"episode {t}" for t in members},
    )
    assert len(row.task_types) == 3
    assert not any(column.startswith("task:other.") for column in clustering.columns)

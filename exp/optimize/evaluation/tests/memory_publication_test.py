"""Publication commits whole versions and admits only owned, well-formed trajectories."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.fidelity_memory import saved_memory_receipt
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    judged,
    memory_project,
    predicted,
    trajectory,
)
from exp.optimize.evaluation.tests.memory_architecture_test import with_lineage
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.world_model.artifact import persist_grounded_world_model
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    knowledge_identity,
    memory_head,
    publish_trajectory,
)
from exp.simulation.world_model.memory_tree import line_id, read_line
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

LINEAGE = "fixture-memory"


def spec(
    base: FidelitySpec,
    run: str,
    key: str,
    group: str,
    mode: str = "learn",
    checkpoint: int | None = None,
    reserved: bool = False,
) -> FidelitySpec:
    """One trajectory of task lineage ``group``, with compression switched off."""
    return with_lineage(
        trajectory(base, run, key, mode, checkpoint=checkpoint),
        group,
        reserved=reserved,
        maximum_compressions=0,
    )


def run(
    project: ProjectStore, monkeypatch: pytest.MonkeyPatch, case: FidelitySpec, key: str
) -> None:
    """Predict both steps exactly, judge, and publish when learning."""
    wire = RoutedTransport(
        [
            predicted(case.workload.id + "-0", key + "-reply-0"),
            predicted(case.workload.id + "-1", key + "-reply-1"),
            judged(case),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, case, code_revision="fixture")
    )


def interrupt_on(monkeypatch: pytest.MonkeyPatch, artifact_type: str) -> None:
    """Stop the process just before the first write of one artifact type."""
    real = ArtifactStore.write

    def write(self: ArtifactStore, *args: object, **kwargs: object) -> object:
        """Interrupt as a crash would, before anything of that type is persisted."""
        if kwargs.get("artifact_type") == artifact_type:
            monkeypatch.setattr(ArtifactStore, "write", real)
            raise KeyboardInterrupt(f"before {artifact_type}")
        return real(self, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(ArtifactStore, "write", write)


def test_publication_interrupted_before_commit_leaves_no_line_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A is reserved after its interrupted publication; A is refused and B still publishes."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    run(project, monkeypatch, spec(base, "traj-z", "zeta", "zeta-lineage"), "zeta")
    first = memory_head(project.artifacts, LINEAGE)
    assert first is not None and first.log_length is not None
    a = spec(base, "traj-a", "alpha", "alpha-lineage")
    interrupt_on(monkeypatch, "experience-memory-version")
    with pytest.raises(KeyboardInterrupt):
        run(project, monkeypatch, a, "alpha")
    assert memory_head(project.artifacts, LINEAGE) == first
    assert not project.artifacts.exists(line_id(LINEAGE, first.log_length))
    held = spec(base, "held-alpha", "alpha", "alpha-lineage", "frozen", 1, reserved=True)
    prepare_fidelity_evaluation(project, held, code_revision="fixture")
    run(project, monkeypatch, a, "alpha")
    receipt = saved_memory_receipt(
        project, prepare_fidelity_evaluation(project, a, code_revision="fixture")
    )
    assert receipt is not None and receipt["published"] is False
    assert "reserved" in str(receipt["reason"])
    run(project, monkeypatch, spec(base, "traj-b", "beta", "beta-lineage"), "beta")
    head = memory_head(project.artifacts, LINEAGE)
    assert head is not None and head.version == 2 and head.appended
    assert read_line(project.artifacts, LINEAGE, first.log_length).origin == "beta-lineage"


def test_lines_of_a_commit_interrupted_before_materialization_are_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The committed version restores its own lines for readers and the next publisher."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    run(project, monkeypatch, spec(base, "traj-z", "zeta", "zeta-lineage"), "zeta")
    first = memory_head(project.artifacts, LINEAGE)
    assert first is not None and first.log_length is not None
    interrupt_on(monkeypatch, "memory-line")
    with pytest.raises(KeyboardInterrupt):
        run(project, monkeypatch, spec(base, "traj-a", "alpha", "alpha-lineage"), "alpha")
    committed = memory_head(project.artifacts, LINEAGE)
    assert committed is not None and committed.version == 2
    assert committed.log_length is not None and committed.appended is not None
    missing = [
        line_id(LINEAGE, i)
        for i in range(first.log_length, committed.log_length)
        if not project.artifacts.exists(line_id(LINEAGE, i))
    ]
    assert missing
    reader = spec(base, "reader", "omega", "omega-lineage", "frozen", 2)
    run(project, monkeypatch, reader, "omega")
    assert all(project.artifacts.exists(identity) for identity in missing)
    assert read_line(project.artifacts, LINEAGE, first.log_length).origin == "alpha-lineage"
    run(project, monkeypatch, spec(base, "traj-b", "beta", "beta-lineage"), "beta")
    head = memory_head(project.artifacts, LINEAGE)
    assert head is not None and head.version == 3


def test_publication_admits_only_owned_well_formed_trajectories(tmp_path: Path) -> None:
    """Ownership, one reply per action and {name, arguments} actions are required."""
    project, base = memory_project(tmp_path)
    store = project.artifacts

    def publish(record: JsonObject, world_model: ArtifactInput = base.world_model) -> None:
        """Publish one trajectory as the first version of a fresh lineage."""
        publish_trajectory(
            store,
            lineage="bank",
            parent=memory_head(store, "bank"),
            world_model=world_model,
            selection_policy_sha256="0" * 64,
            trajectory=record,
            research=(),
            created_at=datetime.now(UTC),
            code_revision="fixture",
            publisher=str(record.get("task_lineage")),
        )

    read = {"name": "read", "arguments": {}}
    malformed: list[tuple[JsonObject, str]] = [
        ({"task_lineage": " \t", "actions": [read], "observed_replies": ["r"]}, "task lineage"),
        ({"actions": [read], "observed_replies": ["r"]}, "task lineage"),
        ({"task_lineage": 7, "actions": [read], "observed_replies": ["r"]}, "task lineage"),
        ({"task_lineage": "B", "actions": [read], "observed_replies": []}, "one disclosed"),
        ({"task_lineage": "C", "actions": [42], "observed_replies": ["r"]}, "name, arguments"),
        ({"task_lineage": "D", "actions": [{"name": "read"}], "observed_replies": ["r"]}, "name"),
    ]
    for record, reason in malformed:
        with pytest.raises(ValueError, match=reason):
            publish(record)
    assert memory_head(store, "bank") is None
    publish({"task_lineage": "A", "actions": [read], "observed_replies": ["r"]})
    served = load_grounded_world_model_artifact(store, base.world_model)
    other = persist_grounded_world_model(
        store,
        served.serving_rag,
        model_alias=served.model_alias,
        model=served.model,
        created_at=datetime.now(UTC),
        code_revision="fixture",
        top_k=served.top_k,
        knowledge=SimulationKnowledge(rules="DIFFERENT KNOWLEDGE"),
        selection_policy_sha256=served.selection_policy_sha256,
    )
    with pytest.raises(ValueError, match="different knowledge"):
        publish(
            {"task_lineage": "E", "actions": [read], "observed_replies": ["r"]},
            artifact_input(other.manifest),
        )
    head = memory_head(store, "bank")
    assert head is not None and head.knowledge_identity == knowledge_identity(served)


def test_strict_eligibility_withholds_sources_without_a_task_lineage() -> None:
    """Older consumed trajectories without ownership are unknown provenance when strict."""
    unowned: JsonObject = {"actions": [], "observed_replies": []}
    owned: JsonObject = {**unowned, "task_lineage": "A"}
    assert Eligibility().admits(unowned)
    assert not Eligibility(strict=True).admits(unowned)
    assert Eligibility(strict=True).admits(owned)
    assert not Eligibility(strict=True, reserved_task_lineages=("A",)).admits(owned)
    for lineage in ("", " \t\n"):
        assert not Eligibility(strict=True).admits({**unowned, "task_lineage": lineage})

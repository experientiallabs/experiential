"""Pilot successors preserve cumulative authority and use only exact saved compatible evidence."""

import json
from pathlib import Path

import pytest

from exp.cli.build.app_test import _otlp_export
from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.rules.memory_stage import MemorySetSettings
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance import bootstrap_plan as preparation
from exp.optimize.evaluation.variance.bootstrap_pilot import BootstrapPilotRevision
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapDependencies,
    BootstrapPreparationSpec,
    plan_bootstrap_preparation,
    run_bootstrap_preparation,
)
from exp.optimize.evaluation.variance.bootstrap_resume import (
    bootstrap_execution_spec,
    load_bootstrap_resume,
    register_bootstrap_resume,
)
from exp.optimize.evaluation.variance.bootstrap_resume_test import _origin, _register
from exp.optimize.evaluation.variance.bootstrap_selection_test import preparation_fixture
from exp.optimize.evaluation.variance.build import run_variance_build
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.simulation.mining.environment_families import load_environment_families
from exp.simulation.mining.environment_review import review_protocol_sha256


def pilot_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, BootstrapPreparationSpec, ArtifactInput, BootstrapPilotRevision]:
    """Create a real saved build under the predecessor's producer and an explicit rejected pin."""

    def with_tools(directory: Path, count: int = 1) -> Path:
        """Declare two tools in every real OTLP record before source ingestion."""
        path = _otlp_export(directory, count)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
            for name in ["read", "write"]
        ]

        for record in records:
            record["attributes"].append(
                {"key": "gen_ai.tool.definitions", "value": {"stringValue": json.dumps(tools)}}
            )
        path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        return path

    monkeypatch.setattr("exp.optimize.evaluation.variance.build_test._otlp_export", with_tools)
    project, spec, reviewer = preparation_fixture(tmp_path, monkeypatch)
    assert spec.bootstrap.rules is not None
    rules = spec.bootstrap.rules.model_copy(
        update={
            "memory": MemorySetSettings(
                decision_alias=spec.build.spec.decision_alias, maximum_topics=2
            )
        }
    )
    spec = spec.model_copy(
        update={
            "bootstrap": spec.bootstrap.model_copy(update={"rules": rules}),
            "task_ids": ("agent-chosen-unplaced-pin",),
            "limits": spec.limits.model_copy(update={"maximum_family_tools": 1}),
        }
    )
    paths = _origin(project, spec, tmp_path)
    parent = _register(project, paths)
    report = run_variance_build(
        project,
        spec.build.spec,
        code_revision="original",
        identity_revision="original",
        reviewer=reviewer,
        tagger=_tagger(project, FakeDecisions()),
    )
    assert report.environment_families is not None
    placed = {row.task_id: row.family_id for row in report.environment_families.placements}
    chosen = next(task for task in report.selected_probes if placed.get(task) is not None)
    family_id = placed[chosen]
    assert family_id is not None
    scope = report.environment_families.scope(family_id)
    assert scope.tag_retrieval is not None
    family = next(
        row
        for row in load_environment_families(
            project.artifacts, report.environment_families.families
        ).families
        if row.family_id == scope.family_id
    )
    manifest = project.artifacts.read(report.identity + "-result").manifest
    revision = BootstrapPilotRevision(
        task_id=chosen,
        maximum_family_tools=len(scope.tools),
        maximum_topics=len(scope.tools) + 1,
        saved_build=artifact_input(manifest),
        build_producer_revision=manifest.code_revision,
        family_id=scope.family_id,
        scope_sha256=sha256_json(scope),
        contracts_sha256=sha256_json(family.model_dump(mode="json")["tool_contracts"]),
        task_set=report.build.task_set,
        exposure=report.exposure,
        families=report.environment_families.families,
        reviewed_families=project.artifacts.read(
            report.environment_families.families.artifact_id
        ).manifest.inputs[0],
        index=scope.index,
        world_model=scope.world_model,
        tag_retrieval=scope.tag_retrieval,
        protocol_sha256=review_protocol_sha256(),
    )
    return project, spec, parent, revision


def register_pilot(
    project: ProjectStore,
    spec: BootstrapPreparationSpec,
    parent: ArtifactInput,
    revision: BootstrapPilotRevision,
) -> ArtifactInput:
    """Register one exact pilot successor without increasing any grant."""
    previous = load_bootstrap_resume(project, parent, spec=spec, code_revision="corrected")
    return register_bootstrap_resume(
        project,
        original_inputs=previous.original_inputs,
        code_revision="pilot-executor",
        permitted_changes=("pilot_selection_revision",),
        previous_binding=parent,
        pilot_revision=revision,
    )


def test_pilot_native_plan_preserves_original_spend_and_saved_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native planning reuses the real saved build and includes every old claim without dispatch."""
    project, spec, parent, revision = pilot_fixture(tmp_path, monkeypatch)
    previous = load_bootstrap_resume(project, parent, spec=spec, code_revision="corrected")
    ceiling = operation_spend_ceiling(project, previous.original_operation)
    reservation = ceiling.reserve(alias="unpriced", url="https://fixture.invalid/v1", payload={})
    ceiling.mark_dispatched(reservation, "historical-attempt")
    ceiling.retain_unknown(reservation)
    before = ceiling.summary()
    original_bytes = project.artifacts.read_bytes(
        previous.original_inputs.artifact_id, "inputs.json"
    )
    reference = register_pilot(project, spec, parent, revision)
    assert register_pilot(project, spec, parent, revision) == reference
    binding = load_bootstrap_resume(project, reference, spec=spec, code_revision="pilot-executor")
    actual = bootstrap_execution_spec(project, binding, spec)
    assert actual.task_ids == (revision.task_id,) and spec.task_ids == (
        "agent-chosen-unplaced-pin",
    )
    assert actual.limits.maximum_family_tools == revision.maximum_family_tools
    assert actual.bootstrap.rules is not None and actual.bootstrap.rules.memory is not None
    assert actual.bootstrap.rules.memory.maximum_topics == revision.maximum_topics
    assert binding.build_revision == "original" and binding.producer_revision == "pilot-executor"
    assert (
        binding.original_operation == previous.original_operation
        and binding.grants == previous.grants
    )
    assert (
        project.artifacts.read_bytes(previous.original_inputs.artifact_id, "inputs.json")
        == original_bytes
    )

    def no_build(*args: object, **kwargs: object) -> None:
        """Refuse replanning or executing the completed build during a native pilot plan."""
        raise AssertionError("completed build must not be recomputed or charged")

    monkeypatch.setattr(preparation, "plan_variance_build", no_build)
    monkeypatch.setattr(preparation, "run_variance_build", no_build)
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.bootstrap_selection.load_trace_dataset", no_build
    )
    plan = plan_bootstrap_preparation(
        project, spec, code_revision="pilot-executor", resume_binding=reference
    )
    assert plan.identity == previous.original_operation and plan.build_saved
    assert plan.build_identity + "-result" == revision.saved_build.artifact_id
    assert plan.remaining_tasks == (revision.task_id,) and plan.accounted_unpriced_attempts == 1
    assert all(
        "tagging" not in row.name and "review compatibility" not in row.name for row in plan.costs
    )
    assert ceiling.summary() == before


@pytest.mark.parametrize("field", ["scope_sha256", "contracts_sha256", "protocol_sha256"])
def test_pilot_rejects_tampered_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Changed evidence digests cannot authorize a pilot successor."""
    project, spec, parent, revision = pilot_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="pilot"):
        register_pilot(project, spec, parent, revision.model_copy(update={field: "0" * 64}))


@pytest.mark.parametrize(
    "field", ["index", "world_model", "exposure", "saved_build", "reviewed_families"]
)
def test_pilot_rejects_tampered_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """A genuine artifact identity with a wrong hash is still rejected."""
    project, spec, parent, revision = pilot_fixture(tmp_path, monkeypatch)
    reference = getattr(revision, field).model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="reference"):
        register_pilot(project, spec, parent, revision.model_copy(update={field: reference}))


def test_pilot_rejects_new_bound_task_producer_and_unrelated_original_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the exact saved choice, tools, topics and historical producer can be registered."""
    project, spec, parent, revision = pilot_fixture(tmp_path, monkeypatch)
    for delta in [
        {"task_id": "unplaced"},
        {"maximum_family_tools": 1},
        {"maximum_topics": 1},
        {"build_producer_revision": "renamed-build"},
    ]:
        with pytest.raises(ValueError, match="pilot"):
            register_pilot(project, spec, parent, revision.model_copy(update=delta))
    reference = register_pilot(project, spec, parent, revision)
    changed = spec.model_copy(
        update={
            "bootstrap": spec.bootstrap.model_copy(
                update={"job": spec.bootstrap.job.model_copy(update={"repeats": 1})}
            )
        }
    )
    with pytest.raises(ValueError, match="settings"):
        load_bootstrap_resume(project, reference, spec=changed, code_revision="pilot-executor")


def test_pilot_real_resume_reaches_only_selected_bootstrap_with_original_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real admission and orchestration path skips the old build and preserves the job."""
    project, spec, parent, revision = pilot_fixture(tmp_path, monkeypatch)
    reference = register_pilot(project, spec, parent, revision)
    binding = load_bootstrap_resume(project, reference, spec=spec, code_revision="pilot-executor")
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    before = ceiling.summary()

    class BootstrapReached(Exception):
        """End the offline boundary proof before any author can run."""

    def reached(owner: ProjectStore, batch: object, **kwargs: object) -> None:
        """Observe the real selected task, unchanged job and installed original authority."""
        from exp.optimize.evaluation.variance.bootstrap_contracts import BootstrapSpec
        from exp.runtime.models.spend_ceiling import current_spend_ceiling

        assert owner is project and isinstance(batch, BootstrapSpec)
        assert batch.job == spec.bootstrap.job
        assert batch.families[0].tasks[0].task.task_id == revision.task_id
        active = current_spend_ceiling()
        assert active is not None and active.path == ceiling.path
        assert active.summary() == before
        raise BootstrapReached

    def forbidden(*args: object, **kwargs: object) -> None:
        """Fail if admission attempts to rebuild completed evidence or construct a live client."""
        raise AssertionError("saved build cannot run again")

    monkeypatch.setattr(preparation, "run_variance_build", forbidden)
    monkeypatch.setattr(preparation, "plan_variance_build", forbidden)
    monkeypatch.setattr(preparation, "run_bootstrap", reached)
    with pytest.raises(BootstrapReached):
        run_bootstrap_preparation(
            project,
            spec,
            code_revision="pilot-executor",
            resume_binding=reference,
            dependencies=BootstrapDependencies(tagger=_tagger(project, FakeDecisions())),
        )
    assert ceiling.summary() == before
    assert not project.artifacts.exists(binding.original_operation + "-result")


@pytest.mark.parametrize("role_target", ["task", "learning", "retrieval"])
def test_pilot_refuses_final_evaluation_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role_target: str
) -> None:
    """No candidate task or scoped learning/retrieval evidence can enter from final evaluation."""
    from exp.common.tasks import load_task_set
    from exp.optimize.evaluation.variance import bootstrap_pilot as pilot

    project, spec, parent, revision = pilot_fixture(tmp_path, monkeypatch)
    exposure = pilot.load_exposure_plan(project, revision.exposure)
    report = pilot.load_pilot_build(project, revision)
    assert report.environment_families is not None
    scope = report.environment_families.scope(revision.family_id)
    task = next(
        t
        for t in load_task_set(project.artifacts, revision.task_set.artifact_id).tasks
        if t.task_id == revision.task_id
    )
    trace = (
        task.source_trace_ids[0]
        if role_target == "task"
        else scope.rules_traces[0]
        if role_target == "learning"
        else scope.retrieval_scope.ids[0]
    )
    changed = exposure.model_copy(
        update={"roles": {**exposure.roles, exposure.traces[trace]: "final_evaluation"}}
    )

    def changed_exposure(owner: ProjectStore, ref: ArtifactInput) -> ExposurePlan:
        """Present an invalid role declaration at the saved-exposure boundary."""
        assert owner is project and ref == revision.exposure
        return changed

    monkeypatch.setattr(pilot, "load_exposure_plan", changed_exposure)
    with pytest.raises(ValueError, match="reserved"):
        register_pilot(project, spec, parent, revision)


def test_pilot_registered_execution_and_grants_cannot_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successor pins execution content and the entire original append-only grant history."""
    project, spec, parent, revision = pilot_fixture(tmp_path, monkeypatch)
    reference = register_pilot(project, spec, parent, revision)
    binding = load_bootstrap_resume(project, reference, spec=spec, code_revision="pilot-executor")
    read = project.artifacts.read_bytes

    def changed_bytes(identity: str, file: str) -> bytes:
        """Alter an unrelated execution field after the artifact reader returns it."""
        raw = read(identity, file)
        if identity == reference.artifact_id and file == "execution-spec.json":
            value = json.loads(raw)
            value["bootstrap"]["job"]["repeats"] = value["bootstrap"]["job"]["repeats"] % 100 + 1
            return json.dumps(value).encode()
        return raw

    with monkeypatch.context() as changed:
        changed.setattr(project.artifacts, "read_bytes", changed_bytes)
        with pytest.raises(ValueError, match="execution settings"):
            load_bootstrap_resume(project, reference, spec=spec, code_revision="pilot-executor")
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    ceiling.authorize_policy(ceiling_usd=51, unpriced_attempts=100000)
    with pytest.raises(ValueError, match="authority"):
        load_bootstrap_resume(project, reference, spec=spec, code_revision="pilot-executor")


def test_pilot_refuses_relabelling_a_different_historical_review_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A current digest cannot relabel genuine reviews made under different instructions."""
    from exp.simulation.mining import environment_review

    with monkeypatch.context() as historical:
        historical.setattr(
            environment_review,
            "REVIEW_PROMPT",
            environment_review.REVIEW_PROMPT + " Historical extra instruction.",
        )
        project, spec, parent, revision = pilot_fixture(tmp_path, monkeypatch)
    revision = revision.model_copy(update={"protocol_sha256": review_protocol_sha256()})
    with pytest.raises(ValueError, match="historical review protocol"):
        register_pilot(project, spec, parent, revision)
